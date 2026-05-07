# Copyright (c) Alibaba, Inc. and its affiliates.
"""
DAT (Divergent Association Task) 评估指标实现
通过计算10个词两两之间的语义距离来评估创造力
语义距离 = 1 / 余弦相似度
"""

import json
import os
import re
from pathlib import Path
from typing import List

import numpy as np

from evalscope.api.metric import Metric, SingletonMetric
from evalscope.api.registry import register_metric
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

_REPO_ROOT = Path(__file__).resolve().parents[2]


class DATMetricsBase(SingletonMetric):
    """DAT Metrics 基类，提供共享功能"""

    def _init_once(
        self,
        bert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        **kwargs
    ):
        """初始化 DAT Metrics 基类

        Args:
            bert_model: BERT 模型名称，用于词向量编码
        """
        # 检查依赖
        check_import('sentence_transformers', 'sentence-transformers', raise_error=True, feature_name='DAT Metrics')

        from sentence_transformers import SentenceTransformer

        self.bert_model_name = bert_model

        # 加载 BERT 模型
        logger.info(f'Loading BERT model: {bert_model}')
        
        # 设置模型缓存目录
        model_cache_dir = _REPO_ROOT / 'dataprocess/model'
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 优先检查指定的本地模型路径（HuggingFace 缓存格式）
        specified_model_path = model_cache_dir / 'models--sentence-transformers--all-MiniLM-L6-v2'
        
        # 检查指定的本地模型是否存在（可能是 HuggingFace 缓存格式，需要查找 snapshots 目录）
        model_to_load = None
        use_offline = False
        
        if specified_model_path.exists():
            # 检查是否是 HuggingFace 缓存格式（有 snapshots 目录）
            snapshots_dir = specified_model_path / 'snapshots'
            if snapshots_dir.exists():
                # 查找 snapshots 下的实际模型目录
                snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if snapshot_dirs:
                    # 使用第一个 snapshot 目录（通常只有一个）
                    actual_model_path = snapshot_dirs[0]
                    if (actual_model_path / 'config.json').exists():
                        logger.info(f'Found specified local model at: {actual_model_path}, using local model (offline)')
                        model_to_load = str(actual_model_path)
                        use_offline = True
            # 如果不是缓存格式，直接检查路径
            elif (specified_model_path / 'config.json').exists():
                logger.info(f'Found specified local model at: {specified_model_path}, using local model (offline)')
                model_to_load = str(specified_model_path)
                use_offline = True
        
        if model_to_load is None:
            # 检查其他可能的本地模型路径
            model_name_clean = bert_model.replace('sentence-transformers/', '').replace('/', '--')
            local_model_path = model_cache_dir / model_name_clean
            
            # 检查本地模型是否存在
            if local_model_path.exists() and (local_model_path / 'config.json').exists():
                logger.info(f'Found local model at: {local_model_path}, using local model (offline)')
                model_to_load = str(local_model_path)
                use_offline = True
            else:
                logger.info(f'Local model not found at {specified_model_path} or {local_model_path}, will download to this location')
                model_to_load = bert_model
                use_offline = False
                # 设置缓存目录
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(model_cache_dir)
        
        # 检查是否有可用的 CUDA 设备
        import torch
        forced_device = os.getenv('EVALSCOPE_DAT_METRICS_DEVICE') or os.getenv('EVALSCOPE_METRICS_DEVICE')
        device = 'cpu'  # 默认使用 CPU

        if forced_device:
            if forced_device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning(f'Forced device {forced_device} unavailable, using CPU')
            else:
                device = forced_device
                logger.info(f'Using forced device for DAT metrics: {device}')
        elif torch.cuda.is_available():
            # 检查 GPU 内存是否足够
            try:
                current_device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(current_device).total_memory
                allocated_memory = torch.cuda.memory_allocated(current_device)
                free_memory = total_memory - allocated_memory
                
                min_required = 500 * 1024 * 1024  # 500MB
                
                if free_memory > min_required:
                    test_tensor = torch.zeros(1, device='cuda')
                    del test_tensor
                    torch.cuda.empty_cache()
                    device = 'cuda'
                    logger.info(f'Using CUDA device {current_device} (free memory: {free_memory/1024**2:.0f}MB)')
                else:
                    logger.warning(f'CUDA device {current_device} has insufficient memory ({free_memory/1024**2:.0f}MB free, need {min_required/1024**2:.0f}MB), using CPU')
                    device = 'cpu'
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f'CUDA out of memory, falling back to CPU: {e}')
                    device = 'cpu'
                else:
                    logger.warning(f'CUDA error, falling back to CPU: {e}')
                    device = 'cpu'
        else:
            logger.info('CUDA not available, using CPU')
        
        # 加载模型
        if use_offline:
            # 保存原始环境变量
            original_hf_offline = os.environ.get('HF_HUB_OFFLINE', None)
            original_transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE', None)
            try:
                # 设置离线模式环境变量
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                # 使用 local_files_only=True 确保完全离线
                self.bert_model = SentenceTransformer(model_to_load, device=device, local_files_only=True)
                logger.info('Loaded model from local cache (offline mode)')
            finally:
                # 恢复环境变量
                if original_hf_offline is None:
                    os.environ.pop('HF_HUB_OFFLINE', None)
                else:
                    os.environ['HF_HUB_OFFLINE'] = original_hf_offline
                if original_transformers_offline is None:
                    os.environ.pop('TRANSFORMERS_OFFLINE', None)
                else:
                    os.environ['TRANSFORMERS_OFFLINE'] = original_transformers_offline
        else:
            self.bert_model = SentenceTransformer(model_to_load, device=device, cache_folder=str(model_cache_dir))
            logger.info(f'Model downloaded/cached to: {model_cache_dir}')

        logger.info('DAT Metrics initialization completed.')

    def _parse_json_response(self, prediction: str) -> List[str]:
        """解析模型输出的 JSON 格式，提取 words 数组"""
        # 尝试直接解析 JSON
        try:
            data = json.loads(prediction)
            if isinstance(data, dict) and 'words' in data:
                words = data['words']
                if isinstance(words, list):
                    # 过滤空字符串，并限制为10个词
                    words = [str(word).strip() for word in words if word and str(word).strip()]
                    return words[:10]  # 只取前10个词
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(code_block_pattern, prediction, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and 'words' in data:
                    words = data['words']
                    if isinstance(words, list):
                        words = [str(word).strip() for word in words if word and str(word).strip()]
                        return words[:10]
            except json.JSONDecodeError:
                pass

        # 如果都失败，返回空列表
        logger.warning(f'Failed to parse JSON from prediction: {prediction[:100]}')
        return []

    def _calculate_semantic_distance(self, words: List[str]) -> float:
        """计算10个词两两之间的平均语义距离
        
        Args:
            words: 词列表（应该包含10个词）
            
        Returns:
            平均语义距离（余弦相似度的倒数）
        """
        if len(words) < 2:
            return 0.0
        
        # 对词进行 embedding
        embeddings = self.bert_model.encode(words, show_progress_bar=False)
        
        # 计算两两之间的余弦相似度
        n = len(embeddings)
        distances = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # 计算余弦相似度
                vec_i = embeddings[i]
                vec_j = embeddings[j]
                
                # 归一化
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                
                if norm_i == 0 or norm_j == 0:
                    # 如果向量为零向量，相似度为0，距离为无穷大，这里设为一个大值
                    cosine_similarity = 0.0
                else:
                    cosine_similarity = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                
                # 语义距离 = 1 / 余弦相似度
                # 如果相似度为0或负数，距离设为一个大值
                if cosine_similarity <= 0:
                    semantic_distance = 100.0  # 设置一个较大的值
                else:
                    semantic_distance = 1.0 / cosine_similarity
                
                distances.append(semantic_distance)
        
        # 返回平均语义距离
        if len(distances) == 0:
            return 0.0
        
        avg_distance = np.mean(distances)
        return float(avg_distance)


@register_metric(name='dat_semantic_distance')
class DATSemanticDistance(DATMetricsBase):
    """DAT 语义距离指标：计算10个词两两之间的平均语义距离"""

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        """计算语义距离分数

        Args:
            predictions: 模型预测的 JSON 字符串列表
            references: 参考答案（此指标不使用）

        Returns:
            平均语义距离列表（距离越大，创造力越高）
        """
        results = []

        for prediction in predictions:
            # 解析 JSON，提取 words 数组
            words = self._parse_json_response(prediction)
            
            if len(words) < 2:
                # 如果词数少于2个，无法计算语义距离
                results.append(0.0)
                continue
            
            # 计算平均语义距离
            avg_distance = self._calculate_semantic_distance(words)
            results.append(avg_distance)

        return results
