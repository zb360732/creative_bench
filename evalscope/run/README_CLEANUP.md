# Creative Benchmark 仓库整理说明

## 当前统一入口

推荐统一使用以下入口：

- `python run/run_suite.py --suite combination`
- `python run/run_suite.py --suite exploration`
- `python run/run_suite.py --suite transformation`
- `python run/run_suite.py --suite fulltask`

或使用对应薄包装脚本：

- `run/run_scripts/run_combination_full.sh`
- `run/run_scripts/run_exploration_full.sh`
- `run/run_scripts/run_transformation_full.sh`
- `run/run_scripts/run_fulltask.sh`

suite 配置统一放在：

- `run/task_suites.json`

## 当前层级映射

- Combination: `dat`, `bats`, `rat`, `metaphor`
- Exploration: `aut`, `creative_math`, `drivel_writing`, `neocoder`
- Transformation: `transformation`

当前 `transformation` 已接到 `evalscope/benchmarks/transformation/`，默认读取：

- `dataprocess/transformation/generated/final_runs/transformation_eval_1235_all.json`

默认评分模式为 `llm_judge`，judge 配置读取：

- `run/llm_judge.json`

## 建议保留

### 核心运行与汇总
- `run/run_parallel_eval.py`
- `run/run_eval.py`
- `run/run_suite.py`
- `run/summarize_reports.py`
- `run/make_score_matrix.py`
- `run/rescore_aut_offline.py`
- `run/models.json`
- `run/task_suites.json`

### 推荐主脚本
- `run/run_scripts/run_combination_full.sh`
- `run/run_scripts/run_exploration_full.sh`
- `run/run_scripts/run_transformation_full.sh`
- `run/run_scripts/run_fulltask.sh`

## 建议标记为 legacy（暂不立即删除）

这些脚本可能仍承载历史实验复现价值，但不应再作为新人默认入口：

- `run/run_scripts/run_aut_dat_bats_rat_metaphor_full.sh`
- `run/run_scripts/run_aut_dat_bats_rat_metaphor_add4_existing.sh`
- `run/run_scripts/run_aut_full.sh`
- `run/run_scripts/run_aut_multiround_qwen2.5-7b_full.sh`
- `run/run_scripts/run_newdeploy_three_datasets.sh`
- `run/run_scripts/run_problem_method_full_parallel.sh`
- `run/run_scripts/resume_olmo_newdeploy_local.sh`
- `run/run_scripts/resume_olmo_newdeploy_proxy.sh`
- `run/run_scripts/rerun_olmo_creative_math_proxy.sh`
- `run/run_scripts/wait_then_rerun_olmo_creative_math_proxy.sh`
- `run/run_scripts/inspect_qwen25_32b_deploy.sh`

## 建议下一步（中风险）

1. 新建 `run/legacy/`，把上述 legacy 脚本整体挪过去。
2. 把 `run/test_*.py` 和 `temp/test_*.py` 合并到统一的 `tests/creative_benchmark/`。
3. 给 `custom_eval/` 和 `dataprocess/` 增加统一的 task registry，避免路径与任务名散落在各处。
4. 给 benchmark 增加一份统一的 `task_taxonomy.json`，用来维护：层级、dataset 名、描述、输入输出、指标。
5. 进一步拆分 `models.json`：生产模型、实验模型、embedding 模型分离。
