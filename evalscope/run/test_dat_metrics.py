#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 DAT (Divergent Association Task) 评估指标
测试给定10个词计算语义距离分数
"""

import json
import os
import sys
from pathlib import Path

# 设置默认使用 cuda:0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evalscope.metrics.dat_metrics import DATSemanticDistance


def test_dat_metrics():
    """测试 DAT 语义距离指标计算"""
    
    print("=" * 80)
    print("DAT 评估指标测试")
    print("=" * 80)
    
    # 测试数据：不同语义距离的10个词
    test_cases = [
        {
            "name": "高语义距离（语义差异大）",
            "prediction": json.dumps({
                "words": [
                    "quantum",
                    "butterfly",
                    "volcano",
                    "philosophy",
                    "microwave",
                    "ocean",
                    "mathematics",
                    "diamond",
                    "thunder",
                    "chocolate"
                ]
            }),
            "description": "这些词来自完全不同的领域，语义距离应该很大"
        },
        {
            "name": "中等语义距离（部分相关）",
            "prediction": json.dumps({
                "words": [
                    "dog",
                    "cat",
                    "bird",
                    "fish",
                    "tree",
                    "flower",
                    "mountain",
                    "river",
                    "cloud",
                    "sun"
                ]
            }),
            "description": "这些词部分相关（动物、自然），语义距离中等"
        },
        {
            "name": "低语义距离（语义相似）",
            "prediction": json.dumps({
                "words": [
                    "happy",
                    "joyful",
                    "cheerful",
                    "glad",
                    "pleased",
                    "delighted",
                    "content",
                    "satisfied",
                    "merry",
                    "blissful"
                ]
            }),
            "description": "这些词都是表示快乐的情感词，语义距离应该较小"
        },
        {
            "name": "混合语义距离",
            "prediction": json.dumps({
                "words": [
                    "computer",
                    "algorithm",
                    "software",
                    "pizza",
                    "music",
                    "guitar",
                    "piano",
                    "mountain",
                    "hiking",
                    "adventure"
                ]
            }),
            "description": "混合了不同领域的词，语义距离中等偏大"
        },
        {
            "name": "极端语义距离（完全无关）",
            "prediction": json.dumps({
                "words": [
                    "neutron",
                    "kangaroo",
                    "symphony",
                    "tornado",
                    "philosophy",
                    "diamond",
                    "volcano",
                    "quantum",
                    "butterfly",
                    "microwave"
                ]
            }),
            "description": "这些词来自完全不同的领域，语义距离应该非常大"
        },
    ]
    
    # 初始化指标
    print("\n初始化指标...")
    print("注意：首次运行会加载BERT模型，可能需要一些时间...\n")
    
    try:
        dat_metric = DATSemanticDistance(
            bert_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        print("✓ DATSemanticDistance 初始化完成\n")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=" * 80)
    print("开始测试语义距离计算...")
    print("=" * 80 + "\n")
    
    # 对每个测试用例计算分数
    results = []
    for i, test_case in enumerate(test_cases, 1):
        name = test_case["name"]
        prediction = test_case["prediction"]
        description = test_case["description"]
        
        print(f"测试用例 {i}: {name}")
        print("-" * 80)
        print(f"描述: {description}")
        
        # 解析预测结果
        try:
            pred_data = json.loads(prediction)
            words = pred_data.get("words", [])
            print(f"词数量: {len(words)}")
            print(f"词列表: {words}")
        except Exception as e:
            print(f"✗ 解析预测结果失败: {e}")
            print(f"预测结果: {prediction[:100]}...")
            continue
        
        # 计算语义距离
        try:
            print(f"\n[语义距离计算 - dat_semantic_distance]")
            
            # 解析词列表
            words_list = dat_metric._parse_json_response(prediction)
            print(f"  解析后的词数量: {len(words_list)}")
            
            if len(words_list) < 2:
                print(f"  ⚠ 词数量少于2个，无法计算语义距离")
                results.append({
                    "name": name,
                    "words": words_list,
                    "score": 0.0,
                    "error": "词数量不足"
                })
                continue
            
            # 计算语义距离
            semantic_distance = dat_metric._calculate_semantic_distance(words_list)
            
            # 显示详细信息
            print(f"  计算过程:")
            print(f"    - 对 {len(words_list)} 个词进行 embedding")
            print(f"    - 计算 {len(words_list) * (len(words_list) - 1) // 2} 对词之间的语义距离")
            print(f"    - 平均语义距离: {semantic_distance:.4f}")
            
            # 解释分数
            if semantic_distance > 10:
                interpretation = "非常高（词之间语义差异很大，创造力很高）"
            elif semantic_distance > 5:
                interpretation = "高（词之间语义差异较大，创造力较高）"
            elif semantic_distance > 2:
                interpretation = "中等（词之间语义差异中等，创造力一般）"
            else:
                interpretation = "低（词之间语义相似，创造力较低）"
            
            print(f"  分数解释: {interpretation}")
            print(f"  总分数: {semantic_distance:.4f}")
            
            results.append({
                "name": name,
                "words": words_list,
                "score": semantic_distance,
                "interpretation": interpretation
            })
            
        except Exception as e:
            print(f"✗ 计算分数时出错: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": name,
                "words": words_list if 'words_list' in locals() else [],
                "score": 0.0,
                "error": str(e)
            })
        
        print("\n" + "=" * 80 + "\n")
    
    # 汇总结果
    print("=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    print(f"\n{'测试用例':<30} {'分数':<15} {'解释':<50}")
    print("-" * 95)
    for result in results:
        name = result["name"][:28]
        score = f"{result['score']:.4f}" if 'error' not in result else "ERROR"
        interpretation = result.get("interpretation", result.get("error", ""))[:48]
        print(f"{name:<30} {score:<15} {interpretation:<50}")
    
    # 找出最高和最低分数
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        max_result = max(valid_results, key=lambda x: x['score'])
        min_result = min(valid_results, key=lambda x: x['score'])
        
        print(f"\n最高分数: {max_result['name']} - {max_result['score']:.4f}")
        print(f"最低分数: {min_result['name']} - {min_result['score']:.4f}")
    
    print("\n测试完成！")


def test_json_parsing():
    """测试 JSON 解析功能"""
    print("=" * 80)
    print("测试 JSON 解析功能")
    print("=" * 80)
    
    # 测试不同的 JSON 格式
    test_cases = [
        ('标准 JSON', '{"words": ["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8", "word9", "word10"]}'),
        ('Markdown 代码块', '```json\n{"words": ["word1", "word2", "word3"]}\n```'),
        ('带额外文本', 'Some text\n```json\n{"words": ["test1", "test2"]}\n```\nMore text'),
        ('超过10个词', '{"words": ["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12"]}'),
        ('少于10个词', '{"words": ["w1", "w2", "w3"]}'),
        ('空词列表', '{"words": []}'),
        ('无效 JSON', 'This is not JSON'),
    ]
    
    try:
        dat_metric = DATSemanticDistance(
            bert_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        for name, prediction in test_cases:
            words = dat_metric._parse_json_response(prediction)
            print(f"\n{name}:")
            print(f"  输入: {prediction[:60]}...")
            print(f"  解析结果: {words}")
            print(f"  数量: {len(words)}")
            if len(words) > 10:
                print(f"  ⚠ 注意: 只取前10个词")
    
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_semantic_distance_calculation():
    """测试语义距离计算的详细过程"""
    print("=" * 80)
    print("测试语义距离计算详细过程")
    print("=" * 80)
    
    try:
        dat_metric = DATSemanticDistance(
            bert_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # 测试两个极端情况
        test_cases = [
            {
                "name": "语义相似词",
                "words": ["happy", "joyful", "cheerful", "glad", "pleased", "delighted", "content", "satisfied", "merry", "blissful"]
            },
            {
                "name": "语义差异大的词",
                "words": ["quantum", "butterfly", "volcano", "philosophy", "microwave", "ocean", "mathematics", "diamond", "thunder", "chocolate"]
            }
        ]
        
        for test_case in test_cases:
            print(f"\n{test_case['name']}:")
            print(f"  词列表: {test_case['words']}")
            
            # 计算语义距离
            distance = dat_metric._calculate_semantic_distance(test_case['words'])
            print(f"  平均语义距离: {distance:.4f}")
            
            # 显示词对之间的相似度（前5对）
            embeddings = dat_metric.bert_model.encode(test_case['words'], show_progress_bar=False)
            import numpy as np
            
            print(f"  前5对词的相似度和距离:")
            count = 0
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    if count >= 5:
                        break
                    vec_i = embeddings[i]
                    vec_j = embeddings[j]
                    norm_i = np.linalg.norm(vec_i)
                    norm_j = np.linalg.norm(vec_j)
                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                        semantic_dist = 1.0 / cosine_sim if cosine_sim > 0 else 100.0
                        print(f"    {test_case['words'][i]} <-> {test_case['words'][j]}: "
                              f"相似度={cosine_sim:.4f}, 距离={semantic_dist:.4f}")
                        count += 1
                if count >= 5:
                    break
    
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 DAT 评估指标")
    parser.add_argument(
        "--test",
        choices=["metrics", "parsing", "distance", "all"],
        default="all",
        help="选择要运行的测试"
    )
    
    args = parser.parse_args()
    
    if args.test in ["metrics", "all"]:
        test_dat_metrics()
    
    if args.test in ["parsing", "all"]:
        test_json_parsing()
    
    if args.test in ["distance", "all"]:
        test_semantic_distance_calculation()




