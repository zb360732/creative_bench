#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 AUT 评估指标
测试给定回答计算各种分数：流畅性、精致性、灵活性、新颖性
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 设置默认使用 cuda:2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from evalscope.metrics.aut_metrics import (
    AUTFluency,
    AUTElaboration,
    AUTFlexibility,
    AUTOriginality,
)


def save_clustering_results(item, pool, output_dir):
    """保存聚类结果到 JSON 文件"""
    try:
        cluster_centers = pool['cluster_centers']
        reference_embeddings = pool['embeddings']
        reference_responses = pool.get('responses', [])
        k = pool['k']
        
        # 计算每个参考回答到聚类中心的距离，找到最近的簇
        distances = np.linalg.norm(
            reference_embeddings[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2
        )
        cluster_labels = np.argmin(distances, axis=1).tolist()
        min_distances = np.min(distances, axis=1).tolist()
        
        # 构建结果数据
        clustering_data = {
            'item': item,
            'num_clusters': int(k),
            'num_responses': len(reference_responses),
            'cluster_centers': cluster_centers.tolist(),
            'responses': []
        }
        
        # 为每个回答添加聚类信息
        for i, (response, label, distance) in enumerate(zip(reference_responses, cluster_labels, min_distances)):
            clustering_data['responses'].append({
                'index': i,
                'response': response,
                'cluster_id': int(label),
                'distance_to_center': float(distance)
            })
        
        # 按簇分组统计和回答列表
        cluster_stats = {}
        clusters = {}  # 每个簇包含所有回答文本
        for cluster_id in range(k):
            cluster_responses = [r for r in clustering_data['responses'] if r['cluster_id'] == cluster_id]
            cluster_stats[int(cluster_id)] = {
                'num_responses': len(cluster_responses),
                'response_indices': [r['index'] for r in cluster_responses],
                'avg_distance_to_center': float(np.mean([r['distance_to_center'] for r in cluster_responses])) if cluster_responses else 0.0
            }
            # 添加该簇的所有回答文本
            clusters[int(cluster_id)] = {
                'responses': [r['response'] for r in cluster_responses],
                'num_responses': len(cluster_responses),
                'avg_distance_to_center': cluster_stats[int(cluster_id)]['avg_distance_to_center']
            }
        
        clustering_data['cluster_statistics'] = cluster_stats
        clustering_data['clusters'] = clusters  # 按簇分组的回答列表
        
        # 保存到 JSON 文件
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{item}_clustering.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clustering_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ 聚类结果已保存: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"  ✗ 保存聚类结果失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_aut_metrics(enable_visualization=True, save_clustering=True):
    """测试 AUT 评估指标
    
    Args:
        enable_visualization: 是否生成可视化图
        save_clustering: 是否保存聚类结果到 JSON
    """
    """测试 AUT 指标计算"""

    print("=" * 80)
    print("AUT 评估指标测试")
    print("=" * 80)

    # 测试数据：不同物品的示例回答
    test_cases = [
        {
            "item": "box",
            "prediction": json.dumps({
                "uses": [
                    "storage container",
                    "storage",
                    "hat",
                    "cap",

                    "table",
                    "house",
                    "fort",
                    "desk",
                    "mailbox",
                    "cage",
                    "container",
                    "pet carrier",
                    "plant holder",
                    "gift box",
                    "toy box",
                    "tool box"
                ]
            }),
            "reference": "box"
        },
        {
            "item": "rope",
            "prediction": json.dumps({
                "uses": [
                    "tying things together",
                    "climbing",
                    "jumping rope",
                    "clothesline",
                    "pulley system",
                    "fishing line",
                    "hammock",
                    "swing",
                    "lasso",
                    "tug of war"
                ]
            }),
            "reference": "rope"
        },
        {
            "item": "brick",
            "prediction": json.dumps({
                "uses": [
                    "building material",
                    "paperweight",
                    "doorstop",
                    "weapon",
                    "exercise weight",
                    "bookend",
                    "stepping stone",
                    "garden border",
                    "fireplace",
                    "sculpture material"
                ]
            }),
            "reference": "brick"
        },
        {
            "item": "knife",
            "prediction": json.dumps({
                "uses": [
                    "cutting food",
                    "cutting",
                    "slicing",
                    "chopping",
                    "opening packages",
                    "screwdriver",
                    "pry tool",
                    "letter opener",
                    "scraper",
                    "weapon"
                ]
            }),
            "reference": "knife"
        },
    ]

    # 初始化指标（这会加载数据和构建聚类池）
    print("\n初始化指标...")
    print("注意：首次运行会计算聚类，可能需要一些时间...")
    print("后续运行会从缓存加载，速度会更快。\n")

    try:
        fluency_metric = AUTFluency(
            aut_complete_json_path='/root/data/code/evalscope/dataprocess/combination/aut/Cambridge-AUT-dataset/aut_complete.json',
            bert_model='sentence-transformers/all-MiniLM-L6-v2',
            fluency_similarity_threshold=0.8
        )
        print("✓ AUTFluency 初始化完成")

        elaboration_metric = AUTElaboration(
            aut_complete_json_path='/root/data/code/evalscope/dataprocess/combination/aut/Cambridge-AUT-dataset/aut_complete.json',
            bert_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        print("✓ AUTElaboration 初始化完成")

        flexibility_metric = AUTFlexibility(
            aut_complete_json_path='/root/data/code/evalscope/dataprocess/combination/aut/Cambridge-AUT-dataset/aut_complete.json',
            bert_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        print("✓ AUTFlexibility 初始化完成")

        originality_metric = AUTOriginality(
            aut_complete_json_path='/root/data/code/evalscope/dataprocess/combination/aut/Cambridge-AUT-dataset/aut_complete.json',
            bert_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        print("✓ AUTOriginality 初始化完成")

    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("开始测试各个指标...")
    print("=" * 80 + "\n")

    # 对每个测试用例计算分数
    for i, test_case in enumerate(test_cases, 1):
        item = test_case["item"]
        prediction = test_case["prediction"]
        reference = test_case["reference"]

        print(f"测试用例 {i}: {item}")
        print("-" * 80)

        # 解析预测结果
        try:
            pred_data = json.loads(prediction)
            uses = pred_data.get("uses", [])
            print(f"用途数量: {len(uses)}")
            print(f"用途列表: {uses[:5]}{'...' if len(uses) > 5 else ''}")
        except:
            print(f"预测结果: {prediction[:100]}...")

        # 解析用途列表
        pred_data = json.loads(prediction)
        uses_list = pred_data.get("uses", [])

        # 计算各个指标
        try:
            # 流畅性 - 显示每个用途的去重情况
            print(f"\n[流畅性 - aut_fluency]")
            # 手动计算去重过程
            unique_uses = fluency_metric._semantic_deduplicate(uses_list)
            fluency_score = len(unique_uses)
            print(f"  原始用途数: {len(uses_list)}")
            print(f"  去重后用途数: {len(unique_uses)}")
            print(f"  总分数: {fluency_score:.2f}")
            print(f"  保留的用途:")
            for i, use in enumerate(unique_uses, 1):
                print(f"    {i}. {use}")

            # 精致性 - 显示每个用途的词数
            print(f"\n[精致性 - aut_elaboration]")
            elaboration_score = 0
            print(f"  每个用途的词数（去除停用词后）:")
            for i, use in enumerate(uses_list, 1):
                words = elaboration_metric._remove_stopwords(use)
                word_count = len(words)
                elaboration_score += word_count
                print(f"    {i}. {use}")
                print(f"       词数: {word_count} ({', '.join(words) if words else '无有效词'})")
            print(f"  总词数: {elaboration_score:.2f}")

            # 灵活性 - 显示每个用途的簇归属
            print(f"\n[灵活性 - aut_flexibility]")
            # 需要手动计算以显示详细信息
            item = str(reference).strip().lower()
            flexibility_score = 0.0
            if item in flexibility_metric.clustering_pools:
                pool = flexibility_metric.clustering_pools[item]
                cluster_centers = pool['cluster_centers']
                avg_intra_distance = pool['avg_intra_distance']
                
                # 对用途进行 embedding
                use_embeddings = flexibility_metric.bert_model.encode(uses_list, show_progress_bar=False)
                
                # 计算距离
                distances = np.linalg.norm(
                    use_embeddings[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2
                )
                min_distances = np.min(distances, axis=1)
                nearest_cluster_indices = np.argmin(distances, axis=1)
                threshold = avg_intra_distance * 2.0
                
                covered_clusters = set()
                new_clusters = 0
                cluster_details = []
                
                for i, (use, min_dist, cluster_idx) in enumerate(zip(uses_list, min_distances, nearest_cluster_indices)):
                    if min_dist <= threshold:
                        covered_clusters.add(cluster_idx)
                        cluster_details.append((use, cluster_idx, min_dist, 'existing'))
                    else:
                        new_clusters += 1
                        cluster_details.append((use, None, min_dist, 'new'))
                
                flexibility_score = len(covered_clusters) + new_clusters * 2
                
                print(f"  阈值: {threshold:.4f} (平均簇内距离的2倍)")
                print(f"  每个用途的簇归属:")
                for i, (use, cluster_idx, dist, cluster_type) in enumerate(cluster_details, 1):
                    if cluster_type == 'existing':
                        print(f"    {i}. {use}")
                        print(f"       簇: {cluster_idx}, 距离: {dist:.4f}, 得分: 1 (已有簇)")
                    else:
                        print(f"    {i}. {use}")
                        print(f"       簇: 新簇, 距离: {dist:.4f}, 得分: 2 (新簇)")
                print(f"  覆盖的簇数: {len(covered_clusters)}")
                print(f"  新簇数: {new_clusters}")
                flexibility_score = len(covered_clusters) + new_clusters * 2
                print(f"  总分数: {flexibility_score:.2f}")
            else:
                flexibility_scores = flexibility_metric.apply([prediction], [reference])
                flexibility_score = flexibility_scores[0]
                print(f"  分数: {flexibility_score:.2f}")
                print(f"  警告: 物品 '{item}' 不在聚类池中")

            # 新颖性 - 显示每个用途的距离
            print(f"\n[新颖性 - aut_originality]")
            item = str(reference).strip().lower()
            originality_score = 0.0
            if item in originality_metric.clustering_pools:
                pool = originality_metric.clustering_pools[item]
                cluster_centers = pool['cluster_centers']
                
                # 对用途进行 embedding
                use_embeddings = originality_metric.bert_model.encode(uses_list, show_progress_bar=False)
                
                # 计算距离
                distances = np.linalg.norm(
                    use_embeddings[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2
                )
                min_distances = np.min(distances, axis=1)
                originality_score = np.mean(min_distances)
                
                print(f"  每个用途到最近聚类中心的距离:")
                for i, (use, dist) in enumerate(zip(uses_list, min_distances), 1):
                    print(f"    {i}. {use}")
                    print(f"       距离: {dist:.4f}")
                print(f"  平均距离: {originality_score:.4f}")
                print(f"  总分数: {originality_score:.4f}")
            else:
                originality_scores = originality_metric.apply([prediction], [reference])
                originality_score = originality_scores[0]
                print(f"  分数: {originality_score:.4f}")
                print(f"  警告: 物品 '{item}' 不在聚类池中")

            # 汇总
            print(f"\n[汇总]")
            print(f"  流畅性: {fluency_score:.2f}")
            print(f"  精致性: {elaboration_score:.2f}")
            print(f"  灵活性: {flexibility_score:.2f}")
            print(f"  新颖性: {originality_score:.4f}")

            # 保存聚类结果到 JSON
            if save_clustering:
                try:
                    item_lower = str(item).strip().lower()
                    if item_lower in flexibility_metric.clustering_pools:
                        pool = flexibility_metric.clustering_pools[item_lower]
                        output_dir = Path(__file__).parent / 'clustering_results'
                        save_clustering_results(item_lower, pool, output_dir)
                    else:
                        print(f"\n⚠ 物品 '{item}' 不在聚类池中，无法保存聚类结果")
                except Exception as e:
                    print(f"\n⚠ 保存聚类结果失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 可视化聚类结果
            if enable_visualization:
                try:
                    visualize_clustering(item, reference, uses_list, flexibility_metric, originality_metric)
                except Exception as e:
                    print(f"\n⚠ 可视化失败: {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"✗ 计算分数时出错: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 80 + "\n")

    print("测试完成！")


def visualize_clustering(item, reference, test_uses, flexibility_metric, originality_metric):
    """可视化聚类结果和测试回答"""
    print(f"\n[可视化] 生成聚类可视化图...")
    
    item = str(reference).strip().lower()
    
    if item not in flexibility_metric.clustering_pools:
        print(f"  警告: 物品 '{item}' 不在聚类池中，跳过可视化")
        return
    
    pool = flexibility_metric.clustering_pools[item]
    cluster_centers = pool['cluster_centers']
    reference_embeddings = pool['embeddings']
    reference_responses = pool.get('responses', [])
    k = pool['k']
    
    # 对测试回答进行 embedding
    test_embeddings = flexibility_metric.bert_model.encode(test_uses, show_progress_bar=False)
    
    # 使用 t-SNE 进行降维（384维 -> 2维）
    try:
        from sklearn.manifold import TSNE
        
        # 合并所有向量进行降维
        all_embeddings = np.vstack([reference_embeddings, test_embeddings, cluster_centers])
        
        print(f"  使用 t-SNE 降维 (384维 -> 2维)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
        all_2d = tsne.fit_transform(all_embeddings)
        
        # 分离各个部分
        n_ref = len(reference_embeddings)
        n_test = len(test_embeddings)
        n_centers = len(cluster_centers)
        
        ref_2d = all_2d[:n_ref]
        test_2d = all_2d[n_ref:n_ref + n_test]
        centers_2d = all_2d[n_ref + n_test:]
        
        # 为参考数据分配簇标签（计算每个参考回答到聚类中心的距离，找到最近的簇）
        distances = np.linalg.norm(
            reference_embeddings[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :], axis=2
        )
        ref_labels = np.argmin(distances, axis=1)
        
        # 创建图形
        plt.figure(figsize=(14, 10))
        
        # 绘制参考数据的聚类结果
        colors = cm.get_cmap('tab20', k)
        for cluster_id in range(k):
            cluster_mask = ref_labels == cluster_id
            cluster_points = ref_2d[cluster_mask]
            if len(cluster_points) > 0:
                plt.scatter(
                    cluster_points[:, 0], 
                    cluster_points[:, 1],
                    c=[colors(cluster_id)],
                    alpha=0.4,
                    s=20,
                    label=f'Cluster {cluster_id} ({np.sum(cluster_mask)} ref)',
                    edgecolors='none'
                )
        
        # 绘制聚类中心
        plt.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c='red',
            marker='x',
            s=200,
            linewidths=3,
            label='Cluster Centers',
            zorder=5
        )
        
        # 绘制测试回答
        plt.scatter(
            test_2d[:, 0],
            test_2d[:, 1],
            c='green',
            marker='*',
            s=300,
            linewidths=2,
            edgecolors='black',
            label='Test Responses',
            zorder=6
        )
        
        # 添加测试回答的标签
        for i, (use, point) in enumerate(zip(test_uses, test_2d)):
            plt.annotate(
                f'{i+1}',
                (point[0], point[1]),
                fontsize=8,
                ha='center',
                va='center',
                color='white',
                weight='bold',
                zorder=7
            )
        
        plt.title(f'Clustering Visualization for "{item}"\n(Reference: {n_ref} responses, Test: {n_test} responses, Clusters: {k})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        output_dir = Path(__file__).parent / 'visualizations'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'{item}_clustering.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ 可视化图已保存: {output_file}")
        plt.close()
        
    except ImportError:
        print("  ⚠ sklearn 未安装，无法进行可视化")
    except Exception as e:
        print(f"  ✗ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


def test_json_parsing():
    """测试 JSON 解析功能"""
    print("=" * 80)
    print("测试 JSON 解析功能")
    print("=" * 80)

    from evalscope.metrics.aut_metrics import AUTFluency

    # 测试不同的 JSON 格式
    test_cases = [
        ('标准 JSON', '{"uses": ["use 1", "use 2", "use 3"]}'),
        ('Markdown 代码块', '```json\n{"uses": ["use 1", "use 2"]}\n```'),
        ('带额外文本', 'Some text\n```json\n{"uses": ["test"]}\n```\nMore text'),
        ('无效 JSON', 'This is not JSON'),
    ]

    # 使用 AUTFluency 实例来测试解析（它继承自 AUTMetricsBase）
    try:
        # 使用 CPU 避免内存问题
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        metric = AUTFluency(
            aut_complete_json_path='/root/data/code/evalscope/dataprocess/combination/aut/Cambridge-AUT-dataset/aut_complete.json',
            bert_model='sentence-transformers/all-MiniLM-L6-v2'
        )

        for name, prediction in test_cases:
            uses = metric._parse_json_response(prediction)
            print(f"\n{name}:")
            print(f"  输入: {prediction[:50]}...")
            print(f"  解析结果: {uses}")
            print(f"  数量: {len(uses)}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试 AUT 评估指标")
    parser.add_argument(
        "--test",
        choices=["metrics", "parsing", "all"],
        default="all",
        help="选择要运行的测试"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="禁用可视化图生成"
    )
    parser.add_argument(
        "--no-clustering-save",
        action="store_true",
        help="禁用聚类结果保存到 JSON"
    )
    parser.add_argument(
        "--save-clustering-only",
        action="store_true",
        help="仅保存聚类结果，不生成可视化图"
    )

    args = parser.parse_args()

    # 确定功能开关
    enable_visualization = not args.no_visualization and not args.save_clustering_only
    save_clustering = not args.no_clustering_save

    if args.test in ["metrics", "all"]:
        test_aut_metrics(
            enable_visualization=enable_visualization,
            save_clustering=save_clustering
        )

    if args.test in ["parsing", "all"]:
        test_json_parsing()

