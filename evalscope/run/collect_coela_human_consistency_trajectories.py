#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TASK_COLUMNS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("0", "0_read_book"),
        ("10", "10_put_dishwasher"),
        ("20", "20_prepare_food"),
        ("30", "30_put_fridge"),
        ("40", "40_setup_table"),
    ]
)

MODEL_DIRS: List[str] = [
    "cwah-0-agent-1-human_deepseekduida",
    "cwah-0-agent-1-human_gpt",
    "cwah-0-agent-1-human_qwen72b",
    "cwah-0-agent-1-human_qwen7b",
    "cwah-0-agent-1-human_qwen7brl",
    "cwah-0-agent-1-human_gpt_2",
]

RUN_NAME_PATTERN = re.compile(r"^LLMs_act_(?P<mode_tag>.+)_(?P<persona>\d+)_task(?P<task>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect selected CoELA trajectories for human consistency scoring."
    )
    parser.add_argument(
        "--coela-root",
        default="/inspire/hdd/project/ai4education/qianhong-p-qianhong/coela_11/CoELA",
        help="Path to the CoELA root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/outputs/coela_human_consistency_trajectories_20260327",
        help="Directory where the collected trajectories will be written.",
    )
    return parser.parse_args()


def load_target_personas(map_csv: Path) -> "OrderedDict[str, List[Dict[str, Any]]]":
    result: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict((task, []) for task in TASK_COLUMNS)
    with map_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["Personality"].strip()
            for task_id, column in TASK_COLUMNS.items():
                result[task_id].append(
                    {
                        "task_id": task_id,
                        "task_column": column,
                        "persona_id": int(row[column]),
                        "personality_label": label,
                    }
                )
    return result


def index_runs(runs_dir: Path) -> Dict[Tuple[str, int], Path]:
    index: Dict[Tuple[str, int], Path] = {}
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        match = RUN_NAME_PATTERN.match(run_dir.name)
        if not match:
            continue
        key = (match.group("task"), int(match.group("persona")))
        if key in index:
            raise RuntimeError(f"Duplicate run found for {key} in {runs_dir}: {index[key]} and {run_dir}")
        index[key] = run_dir
    return index


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def safe_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_jsonable(v) for v in value]
    return repr(value)


def flatten_goal_dict(goals: Dict[str, Any]) -> str:
    pieces = []
    for key, value in goals.items():
        pieces.append(f"{key}={value}")
    return "; ".join(pieces)


def summarize_progress(progress: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not progress:
        return {"initial_unsatisfied": {}, "final_unsatisfied": {}}
    first = progress[0]
    last = progress[-1]
    return {
        "initial_unsatisfied": safe_jsonable(first.get("unsatisfied", {})),
        "final_unsatisfied": safe_jsonable(last.get("unsatisfied", {})),
        "final_satisfied": safe_jsonable(last.get("satisfied", {})),
    }


def build_trajectory_payload(
    sample_id: str,
    experiment_id: str,
    run_dir: Path,
    task_id: str,
    persona_id: int,
    personality_label: str,
    log_data: Dict[str, Any],
    results_data: Dict[Any, Any],
) -> Dict[str, Any]:
    result_key = next(iter(results_data))
    result_item = results_data[result_key]
    actions = log_data.get("action", {})
    plans = log_data.get("plan", {})
    agent0_actions = actions.get(0, [])
    agent1_actions = actions.get(1, [])
    agent0_plans = plans.get(0, [])
    agent1_plans = plans.get(1, [])

    step_count = max(len(agent0_actions), len(agent1_actions))
    steps: List[Dict[str, Any]] = []
    for step_idx in range(step_count):
        step_record = {
            "step": step_idx,
            "agent_0_action": agent0_actions[step_idx] if step_idx < len(agent0_actions) else None,
            "agent_1_action": agent1_actions[step_idx] if step_idx < len(agent1_actions) else None,
            "agent_0_plan": agent0_plans[step_idx] if step_idx < len(agent0_plans) else None,
            "agent_1_plan": agent1_plans[step_idx] if step_idx < len(agent1_plans) else None,
        }
        steps.append(step_record)

    goals = log_data.get("goals", {})
    goal_summary = goals.get(0) or goals.get(1) or {}
    llm_entries = log_data.get("LLM", {})
    persona_prompt = None
    for agent_id in (0, 1):
        entries = llm_entries.get(agent_id) or []
        if entries:
            persona_prompt = entries[0].get("personality")
            break

    return {
        "sample_id": sample_id,
        "experiment_id": experiment_id,
        "run_dir_name": run_dir.name,
        "source_run_dir": str(run_dir),
        "task_id": int(task_id),
        "task_name": log_data.get("task_name"),
        "persona_id": persona_id,
        "personality_label": personality_label,
        "personality_prompt": persona_prompt,
        "success": bool(result_item.get("S", [False])[0]),
        "steps_taken": int(result_item.get("L", [step_count])[0]),
        "goal_summary": safe_jsonable(goal_summary),
        "progress_summary": summarize_progress(log_data.get("progress", [])),
        "trajectory": steps,
    }


def write_trajectory_text(output_path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append(f"Sample ID: {payload['sample_id']}")
    lines.append(f"Experiment: {payload['experiment_id']}")
    lines.append(f"Run Directory: {payload['run_dir_name']}")
    lines.append(f"Task: {payload['task_id']} ({payload['task_name']})")
    lines.append(f"Persona ID: {payload['persona_id']}")
    lines.append(f"Personality Label: {payload['personality_label']}")
    lines.append(f"Success: {payload['success']}")
    lines.append(f"Steps Taken: {payload['steps_taken']}")
    lines.append("")
    lines.append("Goal Summary:")
    goal_summary = payload.get("goal_summary", {})
    if goal_summary:
        lines.append(flatten_goal_dict(goal_summary))
    else:
        lines.append("(empty)")
    lines.append("")
    lines.append("Final Remaining Goals:")
    final_unsatisfied = payload.get("progress_summary", {}).get("final_unsatisfied", {})
    if final_unsatisfied:
        lines.append(flatten_goal_dict(final_unsatisfied))
    else:
        lines.append("(empty)")
    lines.append("")
    lines.append("Trajectory:")
    for step in payload["trajectory"]:
        header = f"Step {step['step']:03d}"
        lines.append(header)
        lines.append(f"  Agent 0 Action: {step['agent_0_action']}")
        if step.get("agent_0_plan") is not None:
            lines.append(f"  Agent 0 Plan:   {step['agent_0_plan']}")
        lines.append(f"  Agent 1 Action: {step['agent_1_action']}")
        if step.get("agent_1_plan") is not None:
            lines.append(f"  Agent 1 Plan:   {step['agent_1_plan']}")
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_selected_files(src_dir: Path, dst_dir: Path) -> Dict[str, str]:
    copied: Dict[str, str] = {}
    for filename in ("log.pik", "results.pik", "console.txt"):
        src = src_dir / filename
        if src.exists():
            dst = dst_dir / filename
            shutil.copy2(src, dst)
            copied[filename] = str(dst)

    for src in sorted(src_dir.glob("logs_agent_*.pik")):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied[src.name] = str(dst)
    return copied


def write_readme(output_dir: Path) -> None:
    readme = f"""# CoELA Human Consistency Trajectories

This directory contains 150 selected trajectories for manual human scoring.

Selection rule:
- Experiments: {", ".join(MODEL_DIRS)}
- Tasks: {", ".join(TASK_COLUMNS.keys())}
- Personalities per task: taken from `map_data.csv`
- Total samples: 6 experiments x 5 tasks x 5 personalities = 150

Files:
- `manifest.csv`: index of all collected samples
- `manifest.jsonl`: same index in JSONL format
- `samples/<sample_id>__.../trajectory.txt`: readable action trajectory
- `samples/<sample_id>__.../trajectory.json`: structured trajectory export
- `samples/<sample_id>__.../log.pik`, `results.pik`, `console.txt`, `logs_agent_*.pik`: copied raw files
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    coela_root = Path(args.coela_root)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"

    target_personas = load_target_personas(coela_root / "map_data.csv")

    ensure_clean_dir(output_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    sample_counter = 0

    for model_dir_name in MODEL_DIRS:
        experiment_dir = coela_root / model_dir_name
        experiment_id = model_dir_name.replace("cwah-0-agent-1-human_", "", 1)
        run_index = index_runs(experiment_dir / "runs")

        for task_id, personas in target_personas.items():
            for persona_entry in personas:
                persona_id = persona_entry["persona_id"]
                personality_label = persona_entry["personality_label"]
                key = (task_id, persona_id)
                if key not in run_index:
                    raise RuntimeError(f"Missing run for {experiment_id}, task={task_id}, persona={persona_id}")

                run_dir = run_index[key]
                log_path = run_dir / "log.pik"
                results_path = run_dir / "results.pik"
                if not log_path.exists() or not results_path.exists():
                    raise RuntimeError(f"Missing log/results in {run_dir}")

                sample_counter += 1
                sample_id = f"{sample_counter:03d}"
                sample_dir_name = f"{sample_id}__{experiment_id}__task{task_id}__persona{persona_id}"
                sample_dir = samples_dir / sample_dir_name
                sample_dir.mkdir(parents=True, exist_ok=True)

                copied_files = copy_selected_files(run_dir, sample_dir)
                log_data = load_pickle(log_path)
                results_data = load_pickle(results_path)
                payload = build_trajectory_payload(
                    sample_id=sample_id,
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    task_id=task_id,
                    persona_id=persona_id,
                    personality_label=personality_label,
                    log_data=log_data,
                    results_data=results_data,
                )

                trajectory_json_path = sample_dir / "trajectory.json"
                trajectory_txt_path = sample_dir / "trajectory.txt"
                trajectory_json_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                write_trajectory_text(trajectory_txt_path, payload)

                mode_tag_match = RUN_NAME_PATTERN.match(run_dir.name)
                assert mode_tag_match is not None

                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "experiment_id": experiment_id,
                        "mode_tag": mode_tag_match.group("mode_tag"),
                        "task_id": payload["task_id"],
                        "task_name": payload["task_name"],
                        "persona_id": payload["persona_id"],
                        "personality_label": personality_label,
                        "success": payload["success"],
                        "steps_taken": payload["steps_taken"],
                        "source_run_dir": str(run_dir),
                        "sample_dir": str(sample_dir),
                        "trajectory_txt": str(trajectory_txt_path),
                        "trajectory_json": str(trajectory_json_path),
                        "log_pik": copied_files.get("log.pik", ""),
                        "results_pik": copied_files.get("results.pik", ""),
                        "console_txt": copied_files.get("console.txt", ""),
                    }
                )

    manifest_csv = output_dir / "manifest.csv"
    manifest_jsonl = output_dir / "manifest.jsonl"

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_readme(output_dir)
    print(f"Wrote {len(manifest_rows)} samples to {output_dir}")


if __name__ == "__main__":
    main()
