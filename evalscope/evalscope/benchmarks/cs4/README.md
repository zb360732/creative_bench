# CS4 Benchmark Integration

CS4 (Comparing the Skill of Creating Stories by Controlling the Synthesized Constraint Specificity) benchmark for evaluating story creativity under varying constraint levels.

## Overview

CS4评估大语言模型在不同约束级别下的故事创作能力和创造力。核心特点：
- **数据组织**: 50个独特的故事，每个故事有5个约束级别版本
- 给定基础故事 + 不同数量的约束 → 生成满足约束的修订故事
- 5种约束级别：7, 15, 23, 31, 39 个约束
- 总计 50 stories × 5 constraint levels = 250个样本

## Features

- **数据集**: 50个故事，每个故事5个约束级别，共250个样本
- **Story分组**: 同一个故事的不同约束级别会被自动分组，计算story级别的平均分
  - 每个story的总分 = 该story在所有约束级别上的平均分
  - 整体总分 = 所有stories的平均分

- **子集组织**:
  - **default**: 全部250个样本（50 stories × 5 levels），自动按story分组聚合
  - **constraints_7**: 7个约束的50个样本（50 stories × 1 level）
  - **constraints_15**: 15个约束的50个样本
  - **constraints_23**: 23个约束的50个样本
  - **constraints_31**: 31个约束的50个样本
  - **constraints_39**: 39个约束的50个样本

- **评估模式**:
  - **simplified** (默认): 快速测试，假设约束满足度100%，质量满分，多样性正常计算
  - **full**: 使用LLM judges进行完整评估（约束满足度 + 故事质量）

- **评估指标**: 7个创造性指标
  - **constraint_satisfaction_ratio**: 约束满足百分比
  - **grammar_score**: 语法评分（1-5）
  - **coherence_score**: 连贯性评分（1-5）
  - **likability_score**: 喜爱度评分（1-5）
  - **diversity_score**: 多样性综合分数（N-gram统计）
  - **quc_score**: Quality Under Constraints = normalized_coherence × constraint_satisfaction
  - **rcs_score**: Relative Creativity Score = QUC_high - QUC_low（需要多个约束级别）

- **分数报告**: 同时返回整体和约束级别详细分数
  - **整体分数** (6个指标): 所有stories在所有约束级别上的平均分
  - **约束级别分数** (5 levels × 6 metrics = 30个指标): 每个约束级别的单独分数
    - `quc_score_c7`: 7个约束级别的QUC分数
    - `quc_score_c15`: 15个约束级别的QUC分数
    - `quc_score_c23`: 23个约束级别的QUC分数
    - `quc_score_c31`: 31个约束级别的QUC分数
    - `quc_score_c39`: 39个约束级别的QUC分数
    - （同样适用于其他5个指标）
  - **RCS分数** (1个指标): 基于最高和最低约束级别的对比
  - **总计**: 6 + 30 + 1 = 37个聚合指标

## Usage

### 简化模式（推荐用于测试）

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen2.5-7B-Instruct',
    api_url='http://localhost:8007/v1/chat/completions',
    api_key='EMPTY',
    eval_type='openai_api',
    datasets=['cs4'],
    limit=5,  # 限制样本数量
    dataset_args={
        'cs4': {
            'extra_params': {
                'evaluation_mode': 'simplified',  # 快速模式
            }
        }
    },
    generation_config={
        'max_tokens': 512,  # CS4故事需要更多token
        'temperature': 0.8,  # 创意写作需要较高温度
    },
    work_dir='outputs/cs4_test',
)

run_task(task_cfg=task_cfg)
```

### 完整模式（需要LLM judges）

```python
task_cfg = TaskConfig(
    model='your-model',
    datasets=['cs4'],
    dataset_args={
        'cs4': {
            'extra_params': {
                'evaluation_mode': 'full',  # 完整评估
                'judge_api_url': 'http://localhost:8007/v1/chat/completions',
                'judge_api_key': 'EMPTY',
                'judge_model_id': 'Qwen2.5-7B-Instruct'
            }
        }
    },
    # ... 其他配置
)
```

### 单个约束级别子集

```python
task_cfg = TaskConfig(
    model='your-model',
    datasets=['cs4'],
    dataset_args={
        'cs4': {
            'subset_list': ['constraints_23'],  # 仅23个约束级别
            'extra_params': {
                'evaluation_mode': 'simplified',
            }
        }
    },
)
```

### 多约束级别（计算RCS）

```python
task_cfg = TaskConfig(
    model='your-model',
    datasets=['cs4'],
    limit=3,  # 每个子集3个样本
    dataset_args={
        'cs4': {
            'subset_list': ['constraints_7', 'constraints_23', 'constraints_39'],
            'extra_params': {
                'evaluation_mode': 'simplified',
            }
        }
    },
)
```

### 使用测试脚本

```bash
# 简化模式测试（本地8007端口API）
python temp/test_cs4.py --mode simplified

# 完整模式测试（使用本地LLM judges）
python temp/test_cs4.py --mode full

# 单个约束级别子集测试
python temp/test_cs4.py --mode subset

# 多约束级别测试（计算RCS）
python temp/test_cs4.py --mode multi

# 仅测试数据加载
python temp/test_cs4.py --mode load
```

## Implementation Details

### 核心创新

1. **Story分组聚合**（类似CreativeMath的样本扩展策略）:
   - 数据组织：50个story，每个story有5个约束级别版本
   - 评估策略：先计算每个story在5个约束级别上的平均分（story级别总分）
   - 最终分数：所有stories的story级别总分的平均值
   - 这确保了每个story的贡献是平等的，不受约束级别数量的影响

2. **多维度独立评估**: 不同于CreativeMath的三阶段顺序评估，CS4采用并行评估
   - 约束满足度（LLM judge或简化）
   - 故事质量（LLM judge或简化）
   - 多样性（N-gram统计，不需要LLM）
   - QUC和RCS（基于前面的结果计算）

3. **灵活子集组织**: 支持全量评估和分级别评估
   - default子集：全部250个样本，按story分组聚合
   - 5个约束级别子集：每个50个样本
   - 可按约束级别分析创造力变化

4. **两种评估模式**:
   - simplified: 快速测试，适合开发和调试
   - full: LLM judges评估，适合正式评估

### 文件结构

```
evalscope/benchmarks/cs4/
├── __init__.py                    # 包初始化文件
├── cs4_adapter.py                 # 主适配器（750+行）
└── README.md                      # 本文档

temp/
└── test_cs4.py                    # 测试脚本（5种测试模式）
```

### 关键实现

- **load_subset()**: 加载CSV，添加story_id（基于Instruction分组），支持按约束级别过滤
- **match_score()**: 多维度并行评估
- **aggregate_scores()**: **按story分组聚合**
  - Step 1: 按story_id分组所有样本
  - Step 2: 计算每个story在其所有约束级别上的平均分（story级别总分）
  - Step 3: 计算所有stories的总平均分（整体分数）
  - 示例：如果testing 10 samples (2 stories × 5 constraint levels)
    - Story A在5个约束级别上的QUC: [1.0, 0.9, 0.95, 0.92, 0.88] → Story A总分: 0.93
    - Story B在5个约束级别上的QUC: [1.0, 0.85, 0.90, 0.87, 0.83] → Story B总分: 0.89
    - 整体分数: (0.93 + 0.89) / 2 = 0.91
- **_evaluate_constraint_satisfaction()**: 约束满足度评估（简化/完整）
- **_evaluate_story_quality()**: 故事质量评估（简化/完整）
- **_calculate_diversity()**: N-gram多样性计算

### Prompt工程

1. **故事修订Prompt**:
   ```
   Story Instruction: {instruction}
   BaseStory: {base_story}
   Task: Now revise the given BaseStory to satisfy the following constraints within 500 words:
   {constraints}
   ```

2. **约束满足度评估Prompt**: 逐条评估每个约束是否满足
3. **故事质量评估Prompt**: 评估语法、连贯性、喜爱度（1-5分）

## Test Results

### 测试1：多约束级别（Story分组 + 约束级别详细分数验证）
测试配置：
- 模型: Qwen2.5-7B-Instruct (本地8007端口)
- 样本数: 10个 (2 stories × 5 constraint levels)
- 评估模式: simplified
- 总耗时: ~1.8分钟

结果：
- ✅ 数据加载成功: 识别出50个unique stories
- ✅ Story分组正常: "Aggregating scores across 2 stories with 10 total samples"
- ✅ 约束级别识别: "Constraint levels present: [7, 15, 23, 31, 39]"
- ✅ 模型推理成功: 10/10样本生成完成（平均11秒/样本）
- ✅ 分数聚合:
  - **整体分数**: 6个指标（所有stories的平均）
  - **约束级别分数**: 5 levels × 6 metrics = 30个指标
    - Constraint level 7: QUC=1.0000 (n=2 stories)
    - Constraint level 15: QUC=1.0000 (n=2 stories)
    - Constraint level 23: QUC=1.0000 (n=2 stories)
    - Constraint level 31: QUC=1.0000 (n=2 stories)
    - Constraint level 39: QUC=1.0000 (n=2 stories)
  - **RCS分数**: 0.0000 (QUC_high@39=1.0000, QUC_low@7=1.0000)
  - **总计**: 37个聚合指标

### 测试2：全量子集测试
测试配置：
- 模型: Qwen2.5-7B-Instruct (本地8007端口)
- 样本数: 5个 × 6个子集 = 30个
- 评估模式: simplified
- 总耗时: ~5.5分钟

结果：
- ✅ 数据加载成功: 250样本按约束级别正确过滤
- ✅ 模型推理成功: 30/30样本生成完成（平均11秒/样本）
- ✅ 评估成功: 简化模式快速评估（~150样本/秒）
- ✅ 指标计算:
  - constraint_satisfaction_ratio: 1.000 (100%)
  - grammar_score: 5.000 (满分)
  - coherence_score: 5.000 (满分)
  - likability_score: 5.000 (满分)
  - diversity_score: ~0.950 (N-gram计算)
  - quc_score: 1.000
  - rcs_score: 0.000 (简化模式下所有级别QUC相同)

## Performance

- **简化模式**: 约5-10秒/样本（仅生成 + 多样性计算）
- **完整模式**: 约20-30秒/样本（含约束满足度和质量评估）

对于全部250个样本：
- 简化模式: ~30-40分钟
- 完整模式: ~2-3小时

## Notes

1. **依赖安装**: 需要安装 `nltk` 和 `punkt` tokenizer
   ```bash
   pip install nltk
   python -c "import nltk; nltk.download('punkt')"
   ```

2. **Token限制**: CS4故事较长，建议 `max_tokens >= 512`

3. **Temperature**: 创意写作建议使用较高temperature (0.7-0.9)

4. **评估成本**: 完整模式每样本需要2次LLM调用（约束+质量）

5. **子集选择**: 可以按约束级别评估，便于分析创造力随约束增加的变化

6. **RCS计算**: 需要多个约束级别的数据才能计算相对创造力分数

7. **数据集位置**: 自动检测 `/root/data/code/evalscope/dataprocess/exploration/cs4_benchmark/CS4_dataset/Story-based Base Stories.csv`

8. **自定义数据集**: 可通过`dataset_path`参数指定自定义路径

##与 CreativeMath 的对比

| 特性 | CreativeMath | CS4 |
|------|-------------|-----|
| 任务类型 | 数学问题新颖解 | 约束故事创作 |
| 样本组织 | 样本扩展（1→n） | 固定样本250个 |
| 子集划分 | 单一default | default + 5个约束级别 |
| 评估阶段 | 3阶段顺序 | 多维度独立并行 |
| LLM评估 | 正确性+新颖性 | 约束满足度+故事质量 |
| 非LLM指标 | 无 | 多样性（N-gram） |
| 综合指标 | 5个比率 | QUC + RCS |
| 评估耗时 | ~10s/样本（简化） | ~5-10s/样本（简化） |
| 数据量 | 605样本 | 250样本 |
| Tags | Math | Reasoning, Custom |

## Reference

- Paper: "Comparing the Skill of Creating Stories by Controlling the Synthesized Constraint Specificity"
- Dataset: `/root/data/code/evalscope/dataprocess/exploration/cs4_benchmark/CS4_dataset/`
- Original evaluation code: `/root/data/code/evalscope/dataprocess/exploration/cs4_benchmark/evaluation/`

## Example Output

### 简化模式示例输出（包含整体和约束级别分数）
```json
{
  "整体分数": {
    "constraint_satisfaction_ratio": 1.000,
    "grammar_score": 5.000,
    "coherence_score": 5.000,
    "likability_score": 5.000,
    "diversity_score": 0.902,
    "quc_score": 1.000
  },
  "约束级别分数": {
    "constraints_7": {
      "constraint_satisfaction_ratio_c7": 1.000,
      "grammar_score_c7": 5.000,
      "coherence_score_c7": 5.000,
      "likability_score_c7": 5.000,
      "diversity_score_c7": 0.892,
      "quc_score_c7": 1.000
    },
    "constraints_15": {
      "constraint_satisfaction_ratio_c15": 1.000,
      "grammar_score_c15": 5.000,
      "coherence_score_c15": 5.000,
      "likability_score_c15": 5.000,
      "diversity_score_c15": 0.918,
      "quc_score_c15": 1.000
    },
    "constraints_23": {
      "constraint_satisfaction_ratio_c23": 1.000,
      "grammar_score_c23": 5.000,
      "coherence_score_c23": 5.000,
      "likability_score_c23": 5.000,
      "diversity_score_c23": 0.872,
      "quc_score_c23": 1.000
    },
    "constraints_31": {
      "constraint_satisfaction_ratio_c31": 1.000,
      "grammar_score_c31": 5.000,
      "coherence_score_c31": 5.000,
      "likability_score_c31": 5.000,
      "diversity_score_c31": 0.904,
      "quc_score_c31": 1.000
    },
    "constraints_39": {
      "constraint_satisfaction_ratio_c39": 1.000,
      "grammar_score_c39": 5.000,
      "coherence_score_c39": 5.000,
      "likability_score_c39": 5.000,
      "diversity_score_c39": 0.926,
      "quc_score_c39": 1.000
    }
  },
  "rcs_score": 0.000
}
```

### 完整模式示例输出（预期）
```json
{
  "整体分数": {
    "constraint_satisfaction_ratio": 0.75,
    "grammar_score": 4.2,
    "coherence_score": 4.0,
    "likability_score": 3.8,
    "diversity_score": 0.65,
    "quc_score": 0.60
  },
  "约束级别分数": {
    "constraints_7": {
      "quc_score_c7": 0.80
    },
    "constraints_15": {
      "quc_score_c15": 0.72
    },
    "constraints_23": {
      "quc_score_c23": 0.65
    },
    "constraints_31": {
      "quc_score_c31": 0.52
    },
    "constraints_39": {
      "quc_score_c39": 0.45
    }
  },
  "rcs_score": -0.35
}
```

**说明**:
- 整体分数反映了所有stories在所有约束级别上的综合表现
- 每个约束级别的分数显示了该约束数量下的平均表现
- RCS为负值表示随着约束增加，创造力质量下降（符合预期）
