# NeoCoder Benchmark

NeoCoder 是一个用于评估语言模型在代码生成中创造力的基准测试。它包含带有编程约束的问题，测试模型找到创造性解决方案的能力。

## 使用方法

### 基本用法

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen/Qwen2-0.5B-Instruct',
    datasets=['neocoder'],
    dataset_args={
        'neocoder': {
            # 可选：指定自定义数据集路径
            # 'dataset_path': '/path/to/NeoCoder.json'
        }
    },
)
run_task(task_cfg=task_cfg)
```

### 指定自定义数据集路径

```python
task_cfg = TaskConfig(
    model='Qwen/Qwen2-0.5B-Instruct',
    datasets=['neocoder'],
    dataset_args={
        'neocoder': {
            'extra_params': {
                'dataset_path': '/path/to/your/NeoCoder.json'
            }
        }
    },
)
run_task(task_cfg=task_cfg)
```

## 数据格式

NeoCoder 数据集是一个 JSON 文件，包含以下结构：

- `problem_id`: 问题 ID
- `problem_statements`: 不同约束级别的问题描述列表
- `constraints_list`: 对应的约束列表
- `codes`: 对应的代码解决方案（如果有）

## 评测指标

- `correctness`: 代码正确性（使用 NeoCoder 的评测函数）

## 注意事项

1. 默认数据集路径：`dataprocess/exploration/NeoCoder/datasets/CodeForce/NeoCoder/NeoCoder.json`
2. 测试用例文件：`test_cases_annotated.json` 应该与 `NeoCoder.json` 在同一目录下
3. 代码执行超时时间：默认 6 秒
4. 需要 NeoCoder 源代码在 `dataprocess/exploration/NeoCoder/src` 目录下

