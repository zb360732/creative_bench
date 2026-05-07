#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from collect_coela_human_consistency_trajectories import (
    MODEL_DIRS,
    RUN_NAME_PATTERN,
    TASK_COLUMNS,
    index_runs,
    load_pickle,
    load_target_personas,
)


TASK_NAME_MAP: Dict[str, str] = {
    "0": "read_book",
    "10": "put_dishwasher",
    "20": "prepare_food",
    "30": "put_fridge",
    "40": "setup_table",
}

CANDIDATE_LABELS = ["A", "B", "C", "D", "E", "F"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare blinded human-ranking packets for CoELA trajectories."
    )
    parser.add_argument(
        "--coela-root",
        default="/inspire/hdd/project/ai4education/qianhong-p-qianhong/coela_11/CoELA",
        help="Path to the CoELA root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/outputs/coela_human_ranking_packet_20260327",
        help="Directory where the ranking packet will be written.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def infer_agent_human_ids(variant_name: str) -> Tuple[int, int]:
    if m := re.match(r"^cwah-(\d+)-agent-(\d+)-human(?:_|$)", variant_name):
        return int(m.group(1)), int(m.group(2))
    if m := re.match(r"^cwah-(\d+)-human-(\d+)-agent(?:_|$)", variant_name):
        return int(m.group(2)), int(m.group(1))
    raise ValueError(f"Cannot infer agent/human ids from variant name: {variant_name}")


def obs_to_text(obs: Any) -> str:
    if not isinstance(obs, dict):
        return str(obs)
    keep: Dict[str, Any] = {}
    for key in [
        "current_room",
        "grabbed_objects",
        "opponent_grabbed_objects",
        "reachable_objects",
        "progress",
        "satisfied",
    ]:
        if key in obs:
            keep[key] = obs[key]
    return json.dumps(keep, ensure_ascii=False)


def build_decision_trace(
    log: Dict[str, Any],
    *,
    agent_id: int,
) -> List[Dict[str, Any]]:
    llm_entries = log.get("LLM", {}).get(agent_id, []) or []
    obs_series = log.get("obs", {}).get(agent_id, []) or []
    action_series = log.get("action", {}).get(agent_id, []) or []
    plan_series = log.get("plan", {}).get(agent_id, []) or []

    decision_trace: List[Dict[str, Any]] = []
    for decision_index, entry in enumerate(llm_entries):
        step_now_raw = entry.get("step_now")
        try:
            step_now = int(step_now_raw)
        except Exception:
            step_now = -1

        observation = ""
        env_action = None
        plan_at_step = None
        if 0 <= step_now < len(obs_series):
            observation = obs_to_text(obs_series[step_now])
        if 0 <= step_now < len(action_series):
            env_action = action_series[step_now]
        if 0 <= step_now < len(plan_series):
            plan_at_step = plan_series[step_now]

        decision_trace.append(
            {
                "decision_index": decision_index,
                "env_step": step_now,
                "goal_desc": str(entry.get("goal_desc", "")),
                "progress_desc": str(entry.get("progress_desc", "")),
                "action_history_desc": str(entry.get("action_history_desc", "")),
                "dialogue_history_desc": str(entry.get("dialogue_history_desc", "")),
                "observation": observation,
                "think": str(entry.get("think", "")),
                "selected_action": str(entry.get("plan", "")),
                "message": str(entry.get("message", "")) if entry.get("message") is not None else "",
                "env_action": env_action,
                "plan_at_step": plan_at_step,
            }
        )
    return decision_trace


def build_candidate_payload(
    *,
    experiment_id: str,
    variant_name: str,
    run_dir: Path,
    task_id: str,
    persona_id: int,
    personality_label: str,
    log: Dict[str, Any],
) -> Dict[str, Any]:
    agent_id, human_id = infer_agent_human_ids(variant_name)
    decision_trace = build_decision_trace(log, agent_id=agent_id)
    return {
        "experiment_id": experiment_id,
        "variant_name": variant_name,
        "run_dir": str(run_dir),
        "task_id": int(task_id),
        "task_name": str(log.get("task_name", TASK_NAME_MAP.get(task_id, ""))),
        "persona_id": persona_id,
        "target_human_personality": personality_label,
        "agent_id": agent_id,
        "human_id": human_id,
        "num_decisions": len(decision_trace),
        "decision_trace": decision_trace,
    }


def write_candidate_markdown(path: Path, *, item_id: str, candidate_label: str, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append(f"# Candidate {candidate_label}")
    lines.append("")
    lines.append(f"- Item ID: {item_id}")
    lines.append(f"- Task: {payload['task_id']} ({payload['task_name']})")
    lines.append(f"- Target Human Personality: {payload['target_human_personality']}")
    lines.append(f"- Persona ID: {payload['persona_id']}")
    lines.append(f"- Agent ID: {payload['agent_id']}")
    lines.append(f"- Human ID: {payload['human_id']}")
    lines.append(f"- Number of decision points: {payload['num_decisions']}")
    lines.append("")
    lines.append("## Decision Trace")
    lines.append("")

    for step in payload["decision_trace"]:
        lines.append(f"### Decision {step['decision_index']:02d} (env step {step['env_step']})")
        lines.append("")
        if step["goal_desc"]:
            lines.append("Goal:")
            lines.append(step["goal_desc"])
            lines.append("")
        if step["progress_desc"]:
            lines.append("Progress Summary:")
            lines.append(step["progress_desc"])
            lines.append("")
        lines.append("Observation:")
        lines.append("```json")
        lines.append(step["observation"] or "{}")
        lines.append("```")
        lines.append("")
        lines.append("Think:")
        lines.append(step["think"] or "(empty)")
        lines.append("")
        if step["message"]:
            lines.append("Message:")
            lines.append(step["message"])
            lines.append("")
        lines.append("Selected Action:")
        lines.append(step["selected_action"] or "(empty)")
        lines.append("")
        lines.append("Executed Env Action:")
        lines.append(str(step["env_action"]))
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_item_instructions(
    path: Path,
    *,
    item_id: str,
    task_id: str,
    task_name: str,
    personality_label: str,
    persona_id: int,
    candidate_labels: List[str],
) -> None:
    text = f"""# {item_id}

Task: {task_id} ({task_name})
Target human personality: {personality_label}
Persona ID: {persona_id}

Candidates to rank: {", ".join(candidate_labels)}

Ranking rule:
1. Rank the six candidates from most human-consistent to least human-consistent.
2. Judge based on whether the target human personality is reflected in the agent's observation -> thinking -> action chain.
3. Use a strict ranking with no ties.
4. Only provide ranking, not numeric scores.
"""
    path.write_text(text, encoding="utf-8")


def write_group_readme(path: Path, *, task_id: str, task_name: str) -> None:
    text = f"""# Task Group {task_id} ({task_name})

This task group contains 5 ranking items.
Each item corresponds to one target human personality and contains 6 anonymous candidates.

Files:
- `item_manifest.csv`: overview of the 5 items in this task group
- `ranking_template_rater_1.csv`
- `ranking_template_rater_2.csv`
- `ranking_template_rater_3.csv`
- `items/<item_id>/candidate_<A-F>.md`: candidate materials for ranking

How to annotate:
1. Read the item instructions and the six candidate files.
2. Fill the ranking template using candidate labels A-F.
3. `best_label` means rank 1, `worst_label` means rank 6.
4. Do not assign ties and do not give numeric scores.
"""
    path.write_text(text, encoding="utf-8")


def write_root_readme(path: Path) -> None:
    text = """# CoELA Human Ranking Packet

This package is prepared for human ranking of CoELA trajectories.

Design:
- 5 task groups
- 5 personality items per task group
- 6 anonymous candidates per item
- 3 ranking templates per task group

Main directories:
- `for_raters/`: blinded data shown to annotators
- `internal/`: hidden mapping from candidate labels to model variants
"""
    path.write_text(text, encoding="utf-8")


def make_candidate_order(task_id: str, persona_id: int) -> List[str]:
    labels = CANDIDATE_LABELS[:]
    rng = random.Random(f"coela_rank::{task_id}::{persona_id}")
    rng.shuffle(labels)
    return labels


def main() -> None:
    args = parse_args()
    coela_root = Path(args.coela_root)
    output_dir = Path(args.output_dir)
    for_raters_dir = output_dir / "for_raters"
    internal_dir = output_dir / "internal"

    ensure_clean_dir(output_dir)
    for_raters_dir.mkdir(parents=True, exist_ok=True)
    internal_dir.mkdir(parents=True, exist_ok=True)
    write_root_readme(output_dir / "README.md")

    target_personas = load_target_personas(coela_root / "map_data.csv")

    experiment_indexes: Dict[str, Dict[Tuple[str, int], Path]] = {}
    for model_dir_name in MODEL_DIRS:
        experiment_indexes[model_dir_name] = index_runs(coela_root / model_dir_name / "runs")

    master_rows: List[Dict[str, Any]] = []

    for task_id, persona_entries in target_personas.items():
        task_name = TASK_NAME_MAP.get(task_id, f"task_{task_id}")
        task_group_id = f"task_{task_id}_{task_name}"
        task_dir = for_raters_dir / task_group_id
        items_dir = task_dir / "items"
        task_dir.mkdir(parents=True, exist_ok=True)
        items_dir.mkdir(parents=True, exist_ok=True)
        write_group_readme(task_dir / "README.md", task_id=task_id, task_name=task_name)

        item_rows: List[Dict[str, Any]] = []
        ranking_rows: List[Dict[str, Any]] = []

        for item_idx, persona_entry in enumerate(persona_entries, start=1):
            persona_id = int(persona_entry["persona_id"])
            personality_label = persona_entry["personality_label"]
            item_id = f"T{int(task_id):02d}_I{item_idx:02d}_P{persona_id:02d}"
            item_dir = items_dir / item_id
            item_dir.mkdir(parents=True, exist_ok=True)

            candidate_order = make_candidate_order(task_id, persona_id)
            candidate_payloads: List[Tuple[str, Dict[str, Any]]] = []

            for model_dir_name in MODEL_DIRS:
                run_dir = experiment_indexes[model_dir_name][(task_id, persona_id)]
                variant_name = model_dir_name
                experiment_id = model_dir_name.replace("cwah-0-agent-1-human_", "", 1)
                payload = build_candidate_payload(
                    experiment_id=experiment_id,
                    variant_name=variant_name,
                    run_dir=run_dir,
                    task_id=task_id,
                    persona_id=persona_id,
                    personality_label=personality_label,
                    log=load_pickle(run_dir / "log.pik"),
                )
                candidate_payloads.append((experiment_id, payload))

            if len(candidate_payloads) != len(CANDIDATE_LABELS):
                raise RuntimeError(f"Expected {len(CANDIDATE_LABELS)} candidates for {item_id}")

            candidate_map_rows: List[Dict[str, Any]] = []
            for candidate_label, (experiment_id, payload) in zip(candidate_order, candidate_payloads):
                candidate_md = item_dir / f"candidate_{candidate_label}.md"
                candidate_json = item_dir / f"candidate_{candidate_label}.json"
                write_candidate_markdown(
                    candidate_md,
                    item_id=item_id,
                    candidate_label=candidate_label,
                    payload=payload,
                )
                candidate_json.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                candidate_map_rows.append(
                    {
                        "task_group_id": task_group_id,
                        "item_id": item_id,
                        "task_id": int(task_id),
                        "task_name": task_name,
                        "persona_id": persona_id,
                        "personality_label": personality_label,
                        "candidate_label": candidate_label,
                        "experiment_id": experiment_id,
                        "variant_name": payload["variant_name"],
                        "source_run_dir": payload["run_dir"],
                        "candidate_md": str(candidate_md),
                        "candidate_json": str(candidate_json),
                    }
                )

            write_item_instructions(
                item_dir / "README.md",
                item_id=item_id,
                task_id=task_id,
                task_name=task_name,
                personality_label=personality_label,
                persona_id=persona_id,
                candidate_labels=sorted(candidate_order),
            )

            item_rows.append(
                {
                    "task_group_id": task_group_id,
                    "item_id": item_id,
                    "task_id": int(task_id),
                    "task_name": task_name,
                    "persona_id": persona_id,
                    "personality_label": personality_label,
                    "candidate_labels": " ".join(sorted(candidate_order)),
                    "item_dir": str(item_dir),
                }
            )
            ranking_rows.append(
                {
                    "task_group_id": task_group_id,
                    "item_id": item_id,
                    "task_id": int(task_id),
                    "task_name": task_name,
                    "persona_id": persona_id,
                    "personality_label": personality_label,
                    "candidate_labels": " ".join(sorted(candidate_order)),
                    "best_label": "",
                    "second_label": "",
                    "third_label": "",
                    "fourth_label": "",
                    "fifth_label": "",
                    "worst_label": "",
                    "comments": "",
                }
            )
            master_rows.extend(candidate_map_rows)

        with (task_dir / "item_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(item_rows[0].keys()))
            writer.writeheader()
            writer.writerows(item_rows)

        for rater_idx in range(1, 4):
            rater_path = task_dir / f"ranking_template_rater_{rater_idx}.csv"
            with rater_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(ranking_rows[0].keys()))
                writer.writeheader()
                writer.writerows(ranking_rows)

    master_csv = internal_dir / "master_key.csv"
    with master_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()))
        writer.writeheader()
        writer.writerows(master_rows)

    with (internal_dir / "master_key.jsonl").open("w", encoding="utf-8") as f:
        for row in master_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote ranking packet to {output_dir}")
    print(f"Items: {len(master_rows) // len(CANDIDATE_LABELS)}")
    print(f"Candidates: {len(master_rows)}")


if __name__ == "__main__":
    main()
