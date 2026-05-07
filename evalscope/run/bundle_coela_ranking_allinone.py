#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from export_coela_ranking_singlepage import markdown_to_plain_text, render_text_pdf


DEFAULT_INPUT_DIR = Path(
    "/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/outputs/coela_human_ranking_singlepage_20260327"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle all CoELA human-ranking items into one markdown and one PDF."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory of the single-page ranking packet.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Output markdown path. Defaults to <input-dir>/ALL_IN_ONE.md",
    )
    parser.add_argument(
        "--output-pdf",
        default="",
        help="Output PDF path. Defaults to <input-dir>/ALL_IN_ONE.pdf",
    )
    return parser.parse_args()


def build_bundle_markdown(root: Path) -> str:
    lines: List[str] = []
    lines.append("# CoELA Human Ranking Packet")
    lines.append("")
    lines.append("This file bundles all 25 ranking items into one document.")
    lines.append("")
    lines.append("Rules:")
    lines.append("1. For each item, rank the six candidates from most human-consistent to least human-consistent.")
    lines.append("2. Use strict ranking only. No ties.")
    lines.append("3. Only provide ranking, not numeric scores.")
    lines.append("")
    lines.append("Task groups included:")

    task_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for task_dir in task_dirs:
        lines.append(f"- {task_dir.name}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for task_dir in task_dirs:
        lines.append(f"# {task_dir.name}")
        lines.append("")
        index_md = task_dir / "INDEX.md"
        if index_md.exists():
            lines.append(index_md.read_text(encoding="utf-8").strip())
            lines.append("")
            lines.append("---")
            lines.append("")

        item_dir = task_dir / "items"
        for item_md in sorted(item_dir.glob("*.md")):
            lines.append(item_md.read_text(encoding="utf-8").strip())
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_md = Path(args.output_md) if args.output_md else input_dir / "ALL_IN_ONE.md"
    output_pdf = Path(args.output_pdf) if args.output_pdf else input_dir / "ALL_IN_ONE.pdf"

    markdown_text = build_bundle_markdown(input_dir)
    output_md.write_text(markdown_text, encoding="utf-8")
    render_text_pdf(markdown_to_plain_text(markdown_text), output_pdf)

    print(f"Wrote markdown: {output_md}")
    print(f"Wrote pdf: {output_pdf}")


if __name__ == "__main__":
    main()
