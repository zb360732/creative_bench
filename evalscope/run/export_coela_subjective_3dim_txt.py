#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List

from export_coela_ranking_delivery_text import (
    build_aligned_decisions,
    load_csv,
    load_json,
    ensure_clean_dir,
    plan_to_high_level_action,
    truncate,
)


DEFAULT_PACKET_DIR = Path(
    "/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/outputs/coela_human_ranking_packet_20260327"
)
DEFAULT_OUTPUT_DIR = Path(
    "/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/outputs/coela_subjective_3dim_ranking_txt_only_20260328"
)
RATER_IDS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export txt-only subjective 3-dimension ranking files for human annotators."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_PACKET_DIR),
        help="Existing blinded ranking packet directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the txt-only subjective package will be written.",
    )
    return parser.parse_args()


def build_item_text(item_row: Dict[str, str], item_dir: Path) -> str:
    candidate_jsons = sorted(item_dir.glob("candidate_*.json"))
    if not candidate_jsons:
        raise FileNotFoundError(f"No candidate_*.json found in {item_dir}")

    lines: List[str] = []
    lines.append(f"ITEM: {item_row['item_id']}")
    lines.append(f"TASK: {item_row['task_id']} ({item_row['task_name']})")
    lines.append(f"HUMAN PERSONALITY CONTEXT: {item_row['personality_label']}")
    lines.append(f"PERSONA ID: {item_row['persona_id']}")
    lines.append("")
    lines.append("YOUR RESPONSE FOR THIS ITEM:")
    lines.append("helpfulness: A > B > C > D > E > F")
    lines.append("trustfulness: A > B > C > D > E > F")
    lines.append("empathy: A > B > C > D > E > F")
    lines.append("")
    lines.append("Do not use ties. Do not give scores. Rank all 6 candidates on each dimension.")
    lines.append("")
    lines.append("-" * 72)
    lines.append("")

    for idx, candidate_json in enumerate(candidate_jsons):
        payload = load_json(candidate_json)
        label = candidate_json.stem.split("_")[-1]
        decisions = build_aligned_decisions(payload)

        lines.append(f"CANDIDATE {label}")
        lines.append(f"Number of decision points: {len(decisions)}")
        lines.append("")

        for step_num, step in enumerate(decisions, start=1):
            use_outputs = (not (step["message"] or "").strip()) and bool((step["outputs"] or "").strip())
            lines.append(
                f"[Step {step_num}] llm_index={step['llm_index']} step_now={step['step_now']} "
                f"agent={step['agent_name']} oppo={step['oppo_name']}"
            )
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
            lines.append("-" * 72)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_group_text(task_group_name: str, item_rows: List[Dict[str, str]], items_dir: Path, *, rater_id: int) -> str:
    lines: List[str] = []
    lines.append(f"ASSIGNMENT: {task_group_name} / rater_{rater_id}")
    lines.append("")
    lines.append("This is a subjective interaction-quality ranking task.")
    lines.append("You must rank the 6 anonymous candidates (A-F) for EACH item on THREE dimensions separately.")
    lines.append("")
    lines.append("DIMENSIONS:")
    lines.append("1. helpfulness")
    lines.append("   Judge whether the candidate is useful for task progress and collaboration.")
    lines.append("   Focus on task focus, usefulness, proactive help, adaptation, clear communication, avoiding redundancy or ineffective actions.")
    lines.append("")
    lines.append("2. trustfulness")
    lines.append("   Judge whether the candidate is reliable and worthy of trust as a collaborator.")
    lines.append("   Focus on instruction following, timely updates, reliable commitments, synchronization, and whether the candidate reduces uncertainty.")
    lines.append("")
    lines.append("3. empathy")
    lines.append("   Judge whether the candidate shows understanding of the partner and communicates with appropriate warmth and partner fit.")
    lines.append("   Focus on partner personality fit, warmth, reassurance, tone, and whether the candidate anticipates how the partner interprets the message.")
    lines.append("")
    lines.append("GENERAL RULES:")
    lines.append("- For every item, produce THREE rankings: one for helpfulness, one for trustfulness, one for empathy.")
    lines.append("- Each ranking must include all 6 candidates A-F.")
    lines.append("- No ties.")
    lines.append("- No numeric scores.")
    lines.append("- Judge from the evidence in the file only.")
    lines.append("- Do not rank by task success alone; use the interaction process as the main evidence.")
    lines.append("")
    lines.append("OUTPUT FORMAT FOR EACH ITEM:")
    lines.append("helpfulness: A > B > C > D > E > F")
    lines.append("trustfulness: A > B > C > D > E > F")
    lines.append("empathy: A > B > C > D > E > F")
    lines.append("")
    lines.append("This file contains 5 items for the same task group.")
    lines.append("")
    lines.append("Included items:")
    for row in item_rows:
        lines.append(f"- {row['item_id']}: persona {row['persona_id']} / {row['personality_label']}")
    lines.append("")
    lines.append("=" * 72)
    lines.append("")

    for idx, row in enumerate(item_rows):
        item_text = build_item_text(row, items_dir / row["item_id"])
        lines.append(item_text.rstrip())
        if idx != len(item_rows) - 1:
            lines.append("")
            lines.append("=" * 72)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    for_raters_dir = input_dir / "for_raters"

    ensure_clean_dir(output_dir)

    assignment_rows: List[Dict[str, str]] = []

    for task_dir in sorted([p for p in for_raters_dir.iterdir() if p.is_dir()]):
        item_rows = load_csv(task_dir / "item_manifest.csv")
        for rater_id in RATER_IDS:
            group_text = build_group_text(
                task_group_name=task_dir.name,
                item_rows=item_rows,
                items_dir=task_dir / "items",
                rater_id=rater_id,
            )
            txt_path = output_dir / f"{task_dir.name}_rater_{rater_id}.txt"
            txt_path.write_text(group_text, encoding="utf-8")
            assignment_rows.append(
                {
                    "task_group": task_dir.name,
                    "rater_id": str(rater_id),
                    "text_path": str(txt_path),
                    "num_items": str(len(item_rows)),
                }
            )

    manifest_path = output_dir / "assignment_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(assignment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(assignment_rows)

    readme = """# CoELA Subjective 3-Dimension Ranking TXT Package

This directory contains 15 txt files:
- 5 task groups
- 3 independent raters per task group

Each file requires separate rankings for:
- helpfulness
- trustfulness
- empathy
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    print(f"Wrote {len(assignment_rows)} subjective 3-dimension txt assignments to {output_dir}")


if __name__ == "__main__":
    main()
