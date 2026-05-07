#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_PACKET_DIR = Path(
    "/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/outputs/coela_human_ranking_packet_20260327"
)
DEFAULT_OUTPUT_DIR = Path(
    "/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/outputs/coela_human_ranking_delivery_text_20260327"
)
RATER_IDS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 5 directly shareable task-group markdown/txt files for human ranking."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_PACKET_DIR),
        help="Existing blinded ranking packet directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the shareable markdown/txt files will be written.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> Any:
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def truncate(text: Optional[str], max_chars: int = 2200) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ...[truncated]"


def plan_to_high_level_action(plan: str) -> str:
    plan = (plan or "").strip()
    if not plan:
        return ""
    m = re.match(r"^\s*(\[[^\]]+\])\s*(.*)$", plan)
    if not m:
        return plan
    return f"{m.group(1)} {m.group(2)}".strip()


def _build_step_index(entries: List[Dict[str, Any]]) -> Tuple[List[int], List[Dict[str, Any]]]:
    pairs: List[Tuple[int, Dict[str, Any]]] = []
    for e in entries:
        try:
            s = int(e.get("step_now"))
        except Exception:
            continue
        if s >= 0:
            pairs.append((s, e))
    pairs.sort(key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _latest_entry_at_or_before(
    step_index: Tuple[List[int], List[Dict[str, Any]]], step_now: int
) -> Optional[Dict[str, Any]]:
    steps, vals = step_index
    if not steps:
        return None
    if step_now < 0:
        return vals[-1]
    pos = bisect.bisect_right(steps, step_now) - 1
    if pos < 0:
        return vals[0]
    return vals[pos]


def _obs_to_text(obs_series: Any, step_now: int) -> str:
    if not isinstance(obs_series, list):
        return ""
    if step_now < 0 or step_now >= len(obs_series):
        return ""
    o = obs_series[step_now]
    if not isinstance(o, dict):
        return str(o)
    keep = {}
    for k in ["current_room", "grabbed_objects", "opponent_grabbed_objects", "reachable_objects", "progress", "satisfied"]:
        if k in o:
            keep[k] = o[k]
    try:
        return json.dumps(keep, ensure_ascii=False)
    except Exception:
        return str(keep)


def build_aligned_decisions(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    run_dir = Path(payload["run_dir"])
    log = load_pickle(run_dir / "log.pik")
    agent_id = int(payload["agent_id"])
    human_id = int(payload["human_id"])
    fallback_human_personality = str(payload.get("target_human_personality", ""))

    llm_entries: List[Dict[str, Any]] = log.get("LLM", {}).get(agent_id, []) or []
    obs_series_agent = log.get("obs", {}).get(agent_id) if isinstance(log.get("obs"), dict) else None
    obs_series_human = (
        log.get("obs", {}).get(human_id) if isinstance(log.get("obs"), dict) else None
    )
    human_entries: List[Dict[str, Any]] = log.get("LLM", {}).get(human_id, []) or []
    human_step_index = _build_step_index(human_entries) if human_entries else ([], [])

    decisions: List[Dict[str, Any]] = []
    for idx, e in enumerate(llm_entries):
        try:
            step_now = int(e.get("step_now"))
        except Exception:
            step_now = -1

        human_e = _latest_entry_at_or_before(human_step_index, step_now) if human_entries else None
        human_personality = str(human_e.get("personality", "")) if isinstance(human_e, dict) else ""
        if not human_personality.strip():
            human_personality = fallback_human_personality
        human_progress_desc = str(human_e.get("progress_desc", "")) if isinstance(human_e, dict) else ""

        msg = e.get("message", None)
        msg_text = msg if isinstance(msg, str) else (None if msg is None else str(msg))

        outputs_source = ""
        outputs_text: Optional[str] = None
        if not (msg_text or "").strip():
            if isinstance(e.get("raw_output"), str) and e["raw_output"].strip():
                outputs_source = "raw_output"
                outputs_text = e["raw_output"]
            elif isinstance(e.get("outputs"), str) and e["outputs"].strip():
                outputs_source = "outputs"
                outputs_text = e["outputs"]
            elif isinstance(e.get("raw_outputs"), list) and e["raw_outputs"]:
                outputs_source = "raw_outputs"
                outputs_text = "\n".join(str(x) for x in e["raw_outputs"] if x is not None)

        decisions.append(
            {
                "llm_index": idx,
                "step_now": step_now,
                "agent_name": str(e.get("agent_name", "")),
                "oppo_name": str(e.get("oppo_name", "")),
                "human_personality": human_personality,
                "goal_desc": str(e.get("goal_desc", "")),
                "progress_desc": str(e.get("progress_desc", "")),
                "oppo_progress_desc": human_progress_desc,
                "observation": _obs_to_text(obs_series_agent, step_now),
                "oppo_observation": _obs_to_text(obs_series_human, step_now),
                "action_history_desc": str(e.get("action_history_desc", "")),
                "dialogue_history_desc": str(e.get("dialogue_history_desc", "")),
                "cot": str(e.get("think", "")),
                "message": msg_text,
                "plan": str(e.get("plan", "")),
                "outputs": outputs_text,
                "outputs_source": outputs_source,
            }
        )
    return decisions


def build_item_markdown(item_row: Dict[str, str], item_dir: Path) -> str:
    candidate_jsons = sorted(item_dir.glob("candidate_*.json"))
    if not candidate_jsons:
        raise FileNotFoundError(f"No candidate_*.json found in {item_dir}")

    lines: List[str] = []
    lines.append(f"# {item_row['item_id']}")
    lines.append("")
    lines.append(f"- Task: {item_row['task_id']} ({item_row['task_name']})")
    lines.append(f"- Target Human Personality: {item_row['personality_label']}")
    lines.append(f"- Persona ID: {item_row['persona_id']}")
    lines.append("- Rank candidates A-F from most human-consistent to least human-consistent.")
    lines.append("- Use strict ranking only. No ties. No numeric scores.")
    lines.append("")
    lines.append("## Ranking Sheet")
    lines.append("")
    lines.append("| Rank | Candidate Label |")
    lines.append("| --- | --- |")
    for rank in ["1", "2", "3", "4", "5", "6"]:
        lines.append(f"| {rank} |  |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for idx, candidate_json in enumerate(candidate_jsons):
        payload = load_json(candidate_json)
        label = candidate_json.stem.split("_")[-1]
        decisions = build_aligned_decisions(payload)
        lines.append(f"## Candidate {label}")
        lines.append("")
        lines.append(f"- Number of decision points: {len(decisions)}")
        lines.append("")

        for step_num, step in enumerate(decisions, start=1):
            use_outputs = (not (step["message"] or "").strip()) and bool((step["outputs"] or "").strip())
            lines.append(
                f"### [Step {step_num}] llm_index={step['llm_index']} step_now={step['step_now']} "
                f"agent={step['agent_name']} oppo={step['oppo_name']}"
            )
            lines.append("")
            lines.append("human_personality: " + truncate(step["human_personality"]))
            lines.append("目标(goal): " + truncate(step["goal_desc"]))
            lines.append("观测(progress): " + truncate(step["progress_desc"]))
            lines.append("对方观测(oppo_progress): " + truncate(step["oppo_progress_desc"]))
            lines.append("观测(observation_raw): " + truncate(step["observation"]))
            lines.append("对方观测(oppo_observation_raw): " + truncate(step["oppo_observation"]))
            lines.append("动作历史(action_history): " + truncate(step["action_history_desc"]))
            lines.append("对话历史(dialogue_history): " + truncate(step["dialogue_history_desc"]))
            if use_outputs:
                output_line = "模型原始输出(raw_output): " + truncate(step["outputs"])
                if step["outputs_source"]:
                    output_line += f" (source={step['outputs_source']})"
                lines.append(output_line)
            else:
                lines.append("思维链(cot): " + truncate(step["cot"]))
                lines.append("message: " + truncate(step["message"] or ""))
                lines.append(
                    "高阶动作(action/plan): "
                    + truncate(plan_to_high_level_action(step["plan"]))
                )
            lines.append("")

        if idx != len(candidate_jsons) - 1:
            lines.append("---")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def markdown_to_text(markdown_text: str) -> str:
    lines: List[str] = []
    in_code = False
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code and line.startswith("#"):
            lines.append(line.lstrip("#").strip())
            lines.append("")
            continue
        if line.startswith("|"):
            if line.startswith("| ---"):
                continue
            line = line.strip("|")
            line = " | ".join(cell.strip() for cell in line.split("|"))
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def build_group_markdown(task_group_name: str, item_rows: List[Dict[str, str]], items_dir: Path) -> str:
    lines: List[str] = []
    lines.append(f"# {task_group_name}")
    lines.append("")
    lines.append("This file contains everything one rater needs for this task group.")
    lines.append("")
    lines.append("Instructions:")
    lines.append("1. Read the five items in order.")
    lines.append("2. For each item, rank candidates A-F from most human-consistent to least human-consistent.")
    lines.append("3. Use strict ranking only. No ties. No numeric scores.")
    lines.append("")
    lines.append("Included items:")
    for row in item_rows:
        lines.append(f"- {row['item_id']}: persona {row['persona_id']} / {row['personality_label']}")
    lines.append("")
    lines.append("=" * 72)
    lines.append("")

    for idx, row in enumerate(item_rows):
        item_md = build_item_markdown(row, items_dir / row["item_id"])
        lines.append(item_md.rstrip())
        if idx != len(item_rows) - 1:
            lines.append("")
            lines.append("=" * 72)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def add_rater_header(markdown_text: str, *, task_group_name: str, rater_id: int) -> str:
    header = "\n".join(
        [
            f"# Assignment: {task_group_name} / rater_{rater_id}",
            "",
            "This file is assigned to one independent annotator.",
            "Please complete ranking for all 5 items in this file.",
            "",
            "=" * 72,
            "",
        ]
    )
    return header + markdown_text


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    for_raters_dir = input_dir / "for_raters"

    ensure_clean_dir(output_dir)

    summary_rows: List[Dict[str, str]] = []
    assignment_rows: List[Dict[str, str]] = []

    for task_dir in sorted([p for p in for_raters_dir.iterdir() if p.is_dir()]):
        item_rows = load_csv(task_dir / "item_manifest.csv")
        group_markdown = build_group_markdown(task_dir.name, item_rows, task_dir / "items")
        md_path = output_dir / f"{task_dir.name}.md"
        txt_path = output_dir / f"{task_dir.name}.txt"
        md_path.write_text(group_markdown, encoding="utf-8")
        txt_path.write_text(markdown_to_text(group_markdown), encoding="utf-8")

        summary_rows.append(
            {
                "task_group": task_dir.name,
                "markdown_path": str(md_path),
                "text_path": str(txt_path),
                "num_items": str(len(item_rows)),
            }
        )

        for rater_id in RATER_IDS:
            assigned_markdown = add_rater_header(
                group_markdown,
                task_group_name=task_dir.name,
                rater_id=rater_id,
            )
            assigned_md_path = output_dir / f"{task_dir.name}_rater_{rater_id}.md"
            assigned_txt_path = output_dir / f"{task_dir.name}_rater_{rater_id}.txt"
            assigned_md_path.write_text(assigned_markdown, encoding="utf-8")
            assigned_txt_path.write_text(markdown_to_text(assigned_markdown), encoding="utf-8")
            assignment_rows.append(
                {
                    "task_group": task_dir.name,
                    "rater_id": str(rater_id),
                    "markdown_path": str(assigned_md_path),
                    "text_path": str(assigned_txt_path),
                    "num_items": str(len(item_rows)),
                }
            )

    summary_csv = output_dir / "manifest.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    assignment_csv = output_dir / "assignment_manifest.csv"
    with assignment_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(assignment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(assignment_rows)

    readme = """# CoELA Human Ranking Delivery Text

This directory contains:
- 5 base task-group files (`task_*.md` / `task_*.txt`)
- 15 directly shareable assignment files (`task_*_rater_*.md` / `task_*_rater_*.txt`)

Each assignment file contains all 5 ranking items for one task group and is intended for one independent rater.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(
        f"Wrote {len(summary_rows)} base task-group bundles and {len(assignment_rows)} rater assignments to {output_dir}"
    )


if __name__ == "__main__":
    main()
