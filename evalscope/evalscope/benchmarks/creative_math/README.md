# CreativeMath Benchmark Integration

CreativeMath benchmark for evaluating mathematical creativity through novel solution generation.

## Overview

CreativeMath评估大语言模型生成新颖数学解决方案的能力。对于每个问题：
- 给定k个参考解决方案
- 要求模型生成第(k+1)个新颖解决方案
- 评估正确性和新颖性

## Features

- **数据集**: 400个数学问题，自动扩展为约605个样本（每个问题k=1到n）
- **评估模式**:
  - **simplified** (默认): 快速测试，假设正确性和新颖性
  - **full**: 使用3个LLM评估器（Claude-3-Opus, Gemini-1.5-Pro, GPT-4）进行完整评估
- **评估指标**: 5个创造性指标
  - correctness_ratio: 正确率
  - novelty_ratio: 新颖率（粗粒度）
  - novel_unknown_ratio: 未知新颖率（细粒度）
  - novelty_to_correctness_ratio: 新颖率/正确率
  - novel_unknown_to_novelty_ratio: 未知新颖率/新颖率

## Usage

### 简化模式（推荐用于测试）

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen2.5-7B-Instruct',
    api_url='http://localhost:8007/v1/chat/completions',
    api_key='EMPTY',
    eval_type='openai_api',
    datasets=['creative_math'],
    limit=5,  # 限制样本数量
    dataset_args={
        'creative_math': {
            'extra_params': {
                'evaluation_mode': 'simplified',  # 快速模式
            }
        }
    },
    generation_config={
        'max_tokens': 2048,
        'temperature': 0.7,
    },
    work_dir='outputs/creative_math_test',
)

run_task(task_cfg=task_cfg)
```

### 完整模式（需要API密钥）

```python
task_cfg = TaskConfig(
    model='your-model',
    datasets=['creative_math'],
    dataset_args={
        'creative_math': {
            'extra_params': {
                'evaluation_mode': 'full',  # 完整评估
                'evaluator_models': ['claude-3-opus', 'gemini-1.5-pro', 'gpt-4'],
            }
        }
    },
    # ... 其他配置，需要配置评估器的API密钥
)
```

### 使用测试脚本

```bash
# 简化模式测试（本地8007端口API）
python temp/test_creative_math.py --mode simplified

# 完整模式测试
python temp/test_creative_math.py --mode full

# 仅测试数据加载
python temp/test_creative_math.py --mode load
```

## Implementation Details

### 核心创新

1. **样本扩展策略**: 每个有n个解决方案的问题扩展为n个测试样本
   - 样本1: 给定1个解决方案，生成第2个
   - 样本2: 给定2个解决方案，生成第3个
   - ...
   - 样本n: 给定n个解决方案，生成第(n+1)个

2. **三阶段评估**:
   - 阶段1: 正确性评估（3个评估器全部同意YES）
   - 阶段2: 粗粒度新颖性（与前k个解决方案对比，多数投票）
   - 阶段3: 细粒度新颖性（与第k+1到n个解决方案对比，多数投票）

3. **Prompt工程**: 直接实现CreativeMath原始prompt函数

### 文件结构

```
evalscope/benchmarks/creative_math/
├── __init__.py                    # 空文件
├── creative_math_adapter.py       # 主适配器（700+行）
└── README.md                      # 本文档

temp/
└── test_creative_math.py          # 测试脚本
```

## Test Results

测试配置：
- 模型: Qwen2.5-7B-Instruct (本地8007端口)
- 样本数: 5个
- 评估模式: simplified

结果：
- ✅ 数据加载成功: 605样本限制到5个
- ✅ 模型推理成功: 5/5样本生成完成（平均15秒/样本）
- ✅ 评估成功: 简化模式快速评估（268样本/秒）
- ✅ 指标计算:
  - correctness_ratio: 1.000 (100%)
  - novelty_ratio: 1.000 (100%)
  - novel_unknown_ratio: 0.000 (简化模式默认)
  - novelty_to_correctness_ratio: 1.000
  - novel_unknown_to_novelty_ratio: 0.000

## Performance

- **简化模式**: 约10-20秒/样本（仅生成）
- **完整模式**: 约20-30秒/样本（含9次评估器调用）

对于全部605个样本：
- 简化模式: ~3-5小时
- 完整模式: ~8-10小时

## Notes

1. **API密钥**: 完整模式需要配置Claude、Gemini、GPT-4的API密钥
2. **成本控制**: 完整模式会调用大量API，建议先用limit参数测试
3. **数据集位置**: 自动检测 `/root/data/code/evalscope/dataprocess/exploration/CreativeMath/data/subset.json`
4. **自定义数据集**: 可通过`dataset_path`参数指定自定义路径

## Reference

- Paper: "Assessing the Creativity of LLMs in Proposing Novel Solutions to Mathematical Problems"
- Venue: AAAI-25 Conference (oral presentation)
- Original code: `/root/data/code/evalscope/dataprocess/exploration/CreativeMath/`
