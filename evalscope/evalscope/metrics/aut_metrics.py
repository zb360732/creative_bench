# Copyright (c) Alibaba, Inc. and its affiliates.
"""
AUT (Alternative Uses Test) 评估指标实现
包含四个指标：流畅性、精致性、灵活性、新颖性
"""

import hashlib
import json
import pickle
import re
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from evalscope.api.metric import Metric, SingletonMetric
from evalscope.api.registry import register_metric
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_AUT_COMPLETE_JSON_PATH = str(
    _REPO_ROOT / 'dataprocess/exploration/AUT/Cambridge-AUT-dataset/aut_complete.json'
)


class AUTMetricsBase(SingletonMetric):
    """AUT Metrics 基类，提供共享功能"""

    def _init_once(
        self,
        aut_complete_json_path: str = _DEFAULT_AUT_COMPLETE_JSON_PATH,
        bert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        fluency_similarity_threshold: float = 0.6,
        **kwargs
    ):
        """初始化 AUT Metrics 基类

        Args:
            aut_complete_json_path: aut_complete.json 文件路径
            bert_model: BERT 模型名称
            fluency_similarity_threshold: 流畅性语义去重阈值（余弦相似度）
        """
        # 检查依赖
        check_import('sentence_transformers', 'sentence-transformers', raise_error=True, feature_name='AUT Metrics')
        check_import('sklearn', 'scikit-learn', raise_error=True, feature_name='AUT Metrics')
        check_import('nltk', 'nltk', raise_error=True, feature_name='AUT Metrics')

        from sentence_transformers import SentenceTransformer
        import nltk

        # 下载 NLTK 数据（如果需要）
        nltk_resources = ['punkt', 'punkt_tab', 'stopwords']
        for resource in nltk_resources:
            try:
                if resource == 'punkt_tab':
                    # 检查 punkt_tab 资源
                    try:
                        nltk.data.find(f'tokenizers/{resource}/english/')
                    except LookupError:
                        nltk.download(resource, quiet=True)
                elif resource == 'punkt':
                    # 检查 punkt 资源（旧版本）
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download(resource, quiet=True)
                elif resource == 'stopwords':
                    # 检查 stopwords 资源
                    try:
                        nltk.data.find('corpora/stopwords')
                    except LookupError:
                        nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f'Failed to download NLTK resource {resource}: {e}')

        self.bert_model_name = bert_model
        self.fluency_similarity_threshold = fluency_similarity_threshold

        # 加载 BERT 模型
        logger.info(f'Loading BERT model: {bert_model}')
        # 保护模型加载/encode 的并发访问
        self._bert_lock = threading.RLock()

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

        # 记录加载配置，便于后续恢复
        self._model_kwargs = {
            'low_cpu_mem_usage': False,
            'device_map': None,
            'torch_dtype': 'float32',
        }
        self._model_to_load = model_to_load
        self._model_cache_dir = model_cache_dir
        self._use_offline = use_offline

        # 检查是否有可用的 CUDA 设备
        import torch
        forced_device = os.getenv('EVALSCOPE_AUT_METRICS_DEVICE') or os.getenv('EVALSCOPE_METRICS_DEVICE')
        device = 'cpu'
        preferred_cuda_index = 7

        if forced_device:
            if forced_device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning(f'Forced device {forced_device} unavailable, using CPU')
            else:
                device = forced_device
                logger.info(f'Using forced device for AUT metrics: {device}')
        elif torch.cuda.is_available():
            # 检查指定 GPU 是否可用及内存是否足够
            try:
                if torch.cuda.device_count() <= preferred_cuda_index:
                    logger.warning(
                        f'CUDA device {preferred_cuda_index} not available, using CPU'
                    )
                else:
                    total_memory = torch.cuda.get_device_properties(preferred_cuda_index).total_memory
                    allocated_memory = torch.cuda.memory_allocated(preferred_cuda_index)
                    free_memory = total_memory - allocated_memory

                    # 需要至少 500MB 空闲内存来加载模型
                    min_required = 500 * 1024 * 1024  # 500MB

                    if free_memory > min_required:
                        # 尝试分配少量内存来测试
                        test_tensor = torch.zeros(1, device=f'cuda:{preferred_cuda_index}')
                        del test_tensor
                        torch.cuda.empty_cache()
                        device = f'cuda:{preferred_cuda_index}'
                        logger.info(
                            f'Using CUDA device {preferred_cuda_index} (free memory: {free_memory/1024**2:.0f}MB)'
                        )
                    else:
                        logger.warning(
                            f'CUDA device {preferred_cuda_index} has insufficient memory '
                            f'({free_memory/1024**2:.0f}MB free, need {min_required/1024**2:.0f}MB), using CPU'
                        )
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f'CUDA out of memory, falling back to CPU: {e}')
                else:
                    logger.warning(f'CUDA error, falling back to CPU: {e}')
        else:
            logger.info('CUDA not available, using CPU')
        
        # 加载模型（先加载到 CPU，避免 meta tensor 问题）
        if use_offline:
            # 使用本地模型，设置离线模式避免连接 HuggingFace
            # 保存原始环境变量
            original_hf_offline = os.environ.get('HF_HUB_OFFLINE', None)
            original_transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE', None)
            try:
                # 设置离线模式环境变量
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                # 使用 local_files_only=True 确保完全离线
                self.bert_model = SentenceTransformer(
                    model_to_load,
                    device='cpu',
                    local_files_only=True,
                    model_kwargs=self._model_kwargs,
                )
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
            # 下载模型到指定目录
            self.bert_model = SentenceTransformer(
                model_to_load,
                device='cpu',
                cache_folder=str(model_cache_dir),
                model_kwargs=self._model_kwargs,
            )
            logger.info(f'Model downloaded/cached to: {model_cache_dir}')

        # 始终使用 CPU，避免 meta 迁移
        device = 'cpu'

        # 修复 meta tensor 模型（避免后续 .to() 报错）
        try:
            params_meta = any(getattr(p, 'is_meta', False) for p in self.bert_model.parameters())
            buffers_meta = any(getattr(b, 'is_meta', False) for b in self.bert_model.buffers())
            if params_meta or buffers_meta:
                logger.warning('BERT model loaded on meta device, reloading on CPU then moving to target device')
                if use_offline:
                    original_hf_offline = os.environ.get('HF_HUB_OFFLINE', None)
                    original_transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE', None)
                    try:
                        os.environ['HF_HUB_OFFLINE'] = '1'
                        os.environ['TRANSFORMERS_OFFLINE'] = '1'
                        self.bert_model = SentenceTransformer(
                            model_to_load,
                            device='cpu',
                            local_files_only=True,
                            model_kwargs=self._model_kwargs,
                        )
                    finally:
                        if original_hf_offline is None:
                            os.environ.pop('HF_HUB_OFFLINE', None)
                        else:
                            os.environ['HF_HUB_OFFLINE'] = original_hf_offline
                        if original_transformers_offline is None:
                            os.environ.pop('TRANSFORMERS_OFFLINE', None)
                        else:
                            os.environ['TRANSFORMERS_OFFLINE'] = original_transformers_offline
                else:
                    self.bert_model = SentenceTransformer(
                        model_to_load,
                        device='cpu',
                        cache_folder=str(model_cache_dir),
                        model_kwargs=self._model_kwargs,
                    )
                if device != 'cpu':
                    self.bert_model.to(device)
                    try:
                        import torch
                        self.bert_model._target_device = torch.device(device)
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning(f'Failed to recover meta tensor model, using current model: {exc}')

        # 保存实际设备
        self._device = device
        # 最后再确保不是 meta 模型
        self._ensure_bert_ready(device=device or 'cpu')

        # 加载停用词
        from nltk.corpus import stopwords
        self.stopwords = set(stopwords.words('english'))

        # 加载参考数据
        logger.info(f'Loading AUT complete data from: {aut_complete_json_path}')
        self.aut_complete_path = Path(aut_complete_json_path)
        self.reference_data = self._load_reference_data()

        # 确定缓存目录
        cache_dir = self.aut_complete_path.parent / '.aut_metrics_cache'
        cache_dir.mkdir(exist_ok=True)
        self.cache_dir = cache_dir

        # 为每个物品构建聚类池（使用缓存）
        logger.info('Building clustering pools for each item...')
        self.clustering_pools = self._build_clustering_pools()

        # Double-check to avoid lingering meta tensors
        self._ensure_bert_ready(device=self._device or 'cpu')

        logger.info('AUT Metrics initialization completed.')

    def _safe_encode(self, texts: List[str], device: Optional[str] = None):
        """Thread-safe encode wrapper with meta recovery."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        target = self._ensure_bert_ready(device=device)
        attempts = 0
        last_exc: Optional[Exception] = None
        while attempts < 3:
            attempts += 1
            try:
                with self._bert_lock:
                    return self.bert_model.encode(texts, show_progress_bar=False, device=target)
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if 'meta' in msg or 'to_empty' in msg or 'no data' in msg:
                    # 强制重载并退回 CPU
                    self._reload_bert_model(device='cpu')
                    target = 'cpu'
                    continue
                raise
        if last_exc:
            raise last_exc
        return self.bert_model.encode(texts, show_progress_bar=False, device=target)

    def _reload_bert_model(self, device: Optional[str] = None):
        """Reload BERT model to recover from meta tensor issues."""
        import os  # Ensure local binding exists even if global import is shadowed in some envs
        with self._bert_lock:
            target = device or getattr(self, '_device', 'cpu')
            model_to_load = getattr(self, '_model_to_load', None)
            model_cache_dir = getattr(self, '_model_cache_dir', None)
            use_offline = bool(getattr(self, '_use_offline', False))
            model_kwargs = dict(getattr(self, '_model_kwargs', {}) or {})
            if not model_to_load or not model_cache_dir:
                logger.warning('Missing model configuration for reload, skipping')
                return
            from sentence_transformers import SentenceTransformer
            try:
                if use_offline:
                    original_hf_offline = os.environ.get('HF_HUB_OFFLINE', None)
                    original_transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE', None)
                    try:
                        os.environ['HF_HUB_OFFLINE'] = '1'
                        os.environ['TRANSFORMERS_OFFLINE'] = '1'
                        self.bert_model = SentenceTransformer(
                            model_to_load,
                            device='cpu',
                            local_files_only=True,
                            model_kwargs=model_kwargs,
                        )
                    finally:
                        if original_hf_offline is None:
                            os.environ.pop('HF_HUB_OFFLINE', None)
                        else:
                            os.environ['HF_HUB_OFFLINE'] = original_hf_offline
                        if original_transformers_offline is None:
                            os.environ.pop('TRANSFORMERS_OFFLINE', None)
                        else:
                            os.environ['TRANSFORMERS_OFFLINE'] = original_transformers_offline
                else:
                    self.bert_model = SentenceTransformer(
                        model_to_load,
                        device='cpu',
                        cache_folder=str(model_cache_dir),
                        model_kwargs=model_kwargs,
                    )
                if target != 'cpu':
                    self.bert_model.to(target)
                    try:
                        import torch
                        self.bert_model._target_device = torch.device(target)
                    except Exception:
                        pass
                self._device = target
                logger.info(f'Reloaded BERT model on {target}')
            except Exception as exc:
                logger.warning(f'Failed to reload BERT model: {exc}')

    def _is_meta_bert(self) -> bool:
        """Check whether the current bert_model lives on meta device."""
        model = getattr(self, 'bert_model', None)
        if model is None:
            return True
        try:
            for param in model.parameters():
                if getattr(param, 'is_meta', False):
                    return True
                dev = getattr(param, 'device', None)
                if dev is not None and str(dev) == 'meta':
                    return True
        except Exception:
            pass
        try:
            for buf in model.buffers():
                if getattr(buf, 'is_meta', False):
                    return True
                dev = getattr(buf, 'device', None)
                if dev is not None and str(dev) == 'meta':
                    return True
        except Exception:
            pass
        dev = getattr(model, 'device', None)
        try:
            if getattr(dev, 'type', '') == 'meta' or str(dev) == 'meta':
                return True
        except Exception:
            pass
        return False

    def _ensure_bert_ready(self, device: Optional[str] = None) -> str:
        """Ensure bert_model has real weights on a concrete device."""
        with self._bert_lock:
            target = device or getattr(self, '_device', 'cpu') or 'cpu'
            if target == 'meta':
                target = 'cpu'
            model_to_load = getattr(self, '_model_to_load', None)
            model_cache_dir = getattr(self, '_model_cache_dir', None)
            use_offline = bool(getattr(self, '_use_offline', False))
            model_kwargs = dict(getattr(self, '_model_kwargs', {}) or {})
            # Force concrete load path; avoid device_map/meta initialisation.
            model_kwargs['device_map'] = {'': 'cpu'}
            model_kwargs['low_cpu_mem_usage'] = False
            self._model_kwargs = model_kwargs

            if getattr(self, 'bert_model', None) is None or self._is_meta_bert():
                self._reload_bert_model(device=target)

            if (getattr(self, 'bert_model', None) is None or self._is_meta_bert()) and model_to_load and model_cache_dir:
                try:
                    from sentence_transformers import SentenceTransformer
                    safe_kwargs = dict(model_kwargs)
                    safe_kwargs['device_map'] = {'': 'cpu'}
                    safe_kwargs['low_cpu_mem_usage'] = False
                    if use_offline:
                        original_hf_offline = os.environ.get('HF_HUB_OFFLINE', None)
                        original_transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE', None)
                        try:
                            os.environ['HF_HUB_OFFLINE'] = '1'
                            os.environ['TRANSFORMERS_OFFLINE'] = '1'
                            self.bert_model = SentenceTransformer(
                                model_to_load,
                                device='cpu',
                                local_files_only=True,
                                model_kwargs=safe_kwargs,
                            )
                        finally:
                            if original_hf_offline is None:
                                os.environ.pop('HF_HUB_OFFLINE', None)
                            else:
                                os.environ['HF_HUB_OFFLINE'] = original_hf_offline
                            if original_transformers_offline is None:
                                os.environ.pop('TRANSFORMERS_OFFLINE', None)
                            else:
                                os.environ['TRANSFORMERS_OFFLINE'] = original_transformers_offline
                    else:
                        self.bert_model = SentenceTransformer(
                            model_to_load,
                            device='cpu',
                            cache_folder=str(model_cache_dir),
                            model_kwargs=safe_kwargs,
                        )
                    target = 'cpu'
                except Exception as exc:
                    logger.warning(f'Failed to rebuild BERT model on CPU: {exc}')

            self._device = target or 'cpu'
            try:
                import torch
                if getattr(self, 'bert_model', None) is not None:
                    self.bert_model._target_device = torch.device(self._device)
            except Exception:
                pass
            return self._device

    def _load_reference_data(self) -> Dict[str, List[str]]:
        """加载参考数据，按物品组织"""
        if not self.aut_complete_path.exists():
            raise FileNotFoundError(f'AUT complete data file not found: {self.aut_complete_path}')

        with open(self.aut_complete_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        reference_data = {}
        for item_data in data:
            item = item_data['item']
            responses = [r['response'] for r in item_data['responses']]
            reference_data[item] = responses

        logger.info(f'Loaded reference data for {len(reference_data)} items')
        return reference_data

    def _get_cache_file_path(self, item: str) -> Path:
        """获取缓存文件路径"""
        # 使用物品名称和模型名称生成缓存文件名
        model_hash = hashlib.md5(self.bert_model_name.encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f'{item}_{model_hash}.pkl'
        return cache_file

    def _load_clustering_cache(self, item: str, responses: List[str]) -> Optional[Dict]:
        """从缓存加载聚类结果"""
        cache_file = self._get_cache_file_path(item)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # 验证缓存数据是否匹配
            # 检查响应数量是否一致（简单验证）
            if cache_data.get('num_responses') != len(responses):
                logger.info(f'Cache for {item} is outdated (response count mismatch), recomputing...')
                return None

            # 检查模型是否匹配
            if cache_data.get('bert_model') != self.bert_model_name:
                logger.info(f'Cache for {item} is outdated (model mismatch), recomputing...')
                return None

            # 检查是否有向量数据（必须存在）
            if 'embeddings' not in cache_data or cache_data.get('embeddings') is None:
                logger.info(f'Cache for {item} missing embeddings, recomputing...')
                return None

            logger.info(f'Loaded clustering cache for {item} from {cache_file}')
            return cache_data

        except Exception as e:
            logger.warning(f'Failed to load cache for {item}: {e}, recomputing...')
            return None

    def _save_clustering_cache(self, item: str, pool_data: Dict, responses: List[str]):
        """保存聚类结果到缓存（包括每个回答的向量）"""
        cache_file = self._get_cache_file_path(item)

        try:
            cache_data = {
                'cluster_centers': pool_data['cluster_centers'],
                'embeddings': pool_data['embeddings'],  # 保存所有回答的向量
                'responses': responses,  # 保存回答文本，便于可视化时对应
                'k': pool_data['k'],
                'avg_intra_distance': pool_data['avg_intra_distance'],
                'num_responses': len(responses),
                'bert_model': self.bert_model_name,
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.info(f'Saved clustering cache for {item} to {cache_file} (including {len(responses)} response embeddings)')

        except Exception as e:
            logger.warning(f'Failed to save cache for {item}: {e}')

    def _build_clustering_pools(self) -> Dict[str, Dict]:
        """为每个物品构建 BERT embedding + K-means 聚类池（使用缓存）"""
        clustering_pools = {}

        for item, responses in self.reference_data.items():
            if not responses:
                continue

            # 尝试从缓存加载
            cache_data = self._load_clustering_cache(item, responses)

            if cache_data is not None:
                # 使用缓存数据（包含向量）
                clustering_pools[item] = {
                    'cluster_centers': cache_data['cluster_centers'],
                    'embeddings': cache_data['embeddings'],  # 从缓存加载向量（必须存在）
                    'responses': cache_data.get('responses', responses),  # 使用缓存中的回答（如果存在）
                    'k': cache_data['k'],
                    'avg_intra_distance': cache_data['avg_intra_distance'],
                }
                continue

            # 缓存不存在，需要计算
            # logger.info(f'Computing clustering for {item}: {len(responses)} responses')
            encode_device = self._ensure_bert_ready()
            embeddings = self._safe_encode(responses, device=encode_device)

            # 自动计算 K 值
            n = len(responses)
            # 使用多种方法计算 K，取合理值
            k_sqrt = max(2, int(np.sqrt(n / 2)))
            k_ratio = max(2, int(n / 10))
            k = min(k_sqrt, k_ratio, max(2, n // 20))  # 确保 K 不会太大

            # 如果数据点太少，直接使用所有点作为"簇"
            if n <= k:
                # 每个点作为一个簇中心
                cluster_centers = embeddings
                labels = np.arange(n)
            else:
                # 使用 K-means 聚类
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                cluster_centers = kmeans.cluster_centers_

            # 计算平均簇内距离（用于灵活性阈值）
            avg_intra_cluster_distances = []
            for cluster_id in range(k):
                cluster_points = embeddings[labels == cluster_id]
                if len(cluster_points) > 0:
                    center = cluster_centers[cluster_id]
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    avg_intra_cluster_distances.append(np.mean(distances))

            avg_intra_distance = np.mean(avg_intra_cluster_distances) if avg_intra_cluster_distances else 1.0

            pool_data = {
                'cluster_centers': cluster_centers,
                'embeddings': embeddings,
                'responses': responses,
                'k': k,
                'avg_intra_distance': avg_intra_distance,
            }

            clustering_pools[item] = pool_data

            # 保存到缓存
            self._save_clustering_cache(item, pool_data, responses)

        #logger.info(f'Built clustering pools for {len(clustering_pools)} items')
        return clustering_pools

    def _parse_json_response(self, prediction: str) -> List[str]:
        """解析模型输出的 JSON 格式，提取 uses 数组"""
        # 尝试直接解析 JSON
        try:
            data = json.loads(prediction)
            if isinstance(data, dict) and 'uses' in data:
                uses = data['uses']
                if isinstance(uses, list):
                    return [str(use).strip() for use in uses if use]
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(code_block_pattern, prediction, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and 'uses' in data:
                    uses = data['uses']
                    if isinstance(uses, list):
                        return [str(use).strip() for use in uses if use]
            except json.JSONDecodeError:
                pass

        # 如果都失败，返回空列表
        logger.warning(f'Failed to parse JSON from prediction: {prediction[:100]}')
        return []

    def _remove_stopwords(self, text: str) -> List[str]:
        """去除停用词并返回词列表"""
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token.isalnum() and token not in self.stopwords]


@register_metric(name='aut_fluency')
class AUTFluency(AUTMetricsBase):
    """流畅性指标：基于语义距离去重后的数量"""

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        """计算流畅性分数

        Args:
            predictions: 模型预测的 JSON 字符串列表
            references: 参考答案（此指标不使用）

        Returns:
            去重后的用途数量列表
        """
        results = []

        for prediction in predictions:
            # 解析 JSON，提取 uses 数组
            uses = self._parse_json_response(prediction)
            if not uses:
                results.append(0.0)
                continue

            # 语义去重
            unique_uses = self._semantic_deduplicate(uses)
            results.append(float(len(unique_uses)))

        return results

    def _semantic_deduplicate(self, uses: List[str]) -> List[str]:
        """基于语义相似度去重（贪心算法）"""
        if not uses:
            return []

        # 对用途进行 embedding
        encode_device = self._ensure_bert_ready()
        embeddings = self._safe_encode(uses, device=encode_device)

        # 贪心去重：保留第一个，然后检查后续是否与已保留的相似
        unique_indices = [0]
        unique_embeddings = [embeddings[0]]

        for i in range(1, len(uses)):
            current_embedding = embeddings[i]

            # 计算与所有已保留用途的相似度
            similarities = np.dot(unique_embeddings, current_embedding) / (
                np.linalg.norm(unique_embeddings, axis=1) * np.linalg.norm(current_embedding)
            )

            # 如果最大相似度小于阈值，则保留
            if np.max(similarities) < self.fluency_similarity_threshold:
                unique_indices.append(i)
                unique_embeddings.append(current_embedding)

        return [uses[i] for i in unique_indices]


@register_metric(name='aut_elaboration')
class AUTElaboration(AUTMetricsBase):
    """精致性指标：去除停用词后的平均词数"""

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        """计算精致性分数

        Args:
            predictions: 模型预测的 JSON 字符串列表
            references: 参考答案（此指标不使用）

        Returns:
            去除停用词后的平均词数列表
        """
        results = []

        for prediction in predictions:
            # 解析 JSON，提取 uses 数组
            uses = self._parse_json_response(prediction)
            if not uses:
                results.append(0.0)
                continue

            # 统计去除停用词后的平均词数，避免数量效应
            total_words = 0
            for use in uses:
                words = self._remove_stopwords(use)
                total_words += len(words)

            results.append(float(total_words) / float(len(uses)))

        return results


@register_metric(name='aut_flexibility')
class AUTFlexibility(AUTMetricsBase):
    """灵活性指标：覆盖参考簇并奖励明显偏离已有簇的用途"""

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        """计算灵活性分数

        Args:
            predictions: 模型预测的 JSON 字符串列表
            references: 参考答案，应包含物品名称（如 "box"）

        Returns:
            灵活性得分列表
        """
        results = []

        for prediction, reference in zip(predictions, references):
            # 解析 JSON，提取 uses 数组
            uses = self._parse_json_response(prediction)
            if not uses:
                results.append(0.0)
                continue

            # 从 reference 中提取物品名称（假设 reference 是物品名称）
            item = str(reference).strip().lower()

            # 获取该物品的聚类池
            if item not in self.clustering_pools:
                logger.warning(f'Item "{item}" not found in clustering pools')
                results.append(0.0)
                continue

            pool = self.clustering_pools[item]
            cluster_centers = pool['cluster_centers']
            avg_intra_distance = pool['avg_intra_distance']

            # 对用途进行 embedding
            encode_device = self._ensure_bert_ready()
            use_embeddings = self._safe_encode(uses, device=encode_device)

            # 计算灵活性得分
            score = self._calculate_flexibility_score(use_embeddings, cluster_centers, avg_intra_distance)
            results.append(float(score))

        return results

    def _calculate_flexibility_score(
        self, use_embeddings: np.ndarray, cluster_centers: np.ndarray, avg_intra_distance: float
    ) -> float:
        """计算灵活性得分

        规则：
        - 落在已有簇阈值内的用途按覆盖簇数计分
        - 超出阈值的用途按新颖用途额外奖励
        - 用 sqrt(len(uses)) 归一化，降低单纯堆数量的收益
        """
        if len(use_embeddings) == 0:
            return 0.0

        # 计算每个用途到所有聚类中心的距离
        # 使用欧氏距离
        distances = np.linalg.norm(
            use_embeddings[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2
        )

        # 找到每个用途最近的聚类中心
        min_distances = np.min(distances, axis=1)
        nearest_cluster_indices = np.argmin(distances, axis=1)

        threshold = avg_intra_distance * 1.0
        in_cluster_mask = min_distances <= threshold
        covered_clusters = len(set(nearest_cluster_indices[in_cluster_mask].tolist()))
        outlier_count = int(np.sum(~in_cluster_mask))
        denom = max(1.0, np.sqrt(float(len(use_embeddings))))
        return float(covered_clusters + 3.0 * outlier_count) / float(denom)


@register_metric(name='aut_originality')
class AUTOriginality(AUTMetricsBase):
    """新颖性指标：超出参考簇尺度的离群距离累积"""

    def apply(self, predictions: List[str], references: List[str]) -> List[float]:
        """计算新颖性分数

        Args:
            predictions: 模型预测的 JSON 字符串列表
            references: 参考答案，应包含物品名称（如 "box"）

        Returns:
            新颖性分数列表（越多明显超出参考簇边界的用途，得分越高）
        """
        results = []

        for prediction, reference in zip(predictions, references):
            # 解析 JSON，提取 uses 数组
            uses = self._parse_json_response(prediction)
            if not uses:
                results.append(0.0)
                continue

            # 从 reference 中提取物品名称
            item = str(reference).strip().lower()

            # 获取该物品的聚类池
            if item not in self.clustering_pools:
                logger.warning(f'Item "{item}" not found in clustering pools')
                results.append(0.0)
                continue

            pool = self.clustering_pools[item]
            cluster_centers = pool['cluster_centers']
            avg_intra_distance = float(pool['avg_intra_distance'])

            # 对用途进行 embedding
            encode_device = self._ensure_bert_ready()
            use_embeddings = self._safe_encode(uses, device=encode_device)

            # 计算到最近聚类中心的距离
            distances = np.linalg.norm(
                use_embeddings[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2
            )
            min_distances = np.min(distances, axis=1)

            # 将距离按参考簇自身尺度归一化，只累计明显越界的尾部部分。
            normalized_distances = min_distances / max(avg_intra_distance, 1e-6)
            exceedances = normalized_distances[normalized_distances > 1.5]
            if exceedances.size == 0:
                results.append(0.0)
                continue

            score = float(np.sum(exceedances - 1.5))
            results.append(score)

        return results
