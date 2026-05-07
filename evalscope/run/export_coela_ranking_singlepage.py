#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont


DEFAULT_PACKET_DIR = Path(
    "/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/outputs/coela_human_ranking_packet_20260327"
)
DEFAULT_OUTPUT_DIR = Path(
    "/inspire/hdd/project/ai4education/qianhong-p-qianhong/benchmark/evalscope/outputs/coela_human_ranking_singlepage_20260327"
)
FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export single-file markdown and PDF packets for human CoELA ranking."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_PACKET_DIR),
        help="Existing blinded ranking packet directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for single-file markdown / PDF packets.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def english_personality_label(label: str) -> str:
    return label.split("(", 1)[0].strip()


def pretty_json_or_text(value: str) -> str:
    if not value:
        return "{}"
    try:
        obj = json.loads(value)
    except Exception:
        return value
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_candidate_markdown(candidate_label: str, payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"## Candidate {candidate_label}")
    lines.append("")
    lines.append(f"- Task: {payload['task_id']} ({payload['task_name']})")
    lines.append(f"- Target Human Personality: {payload['target_human_personality']}")
    lines.append(f"- Persona ID: {payload['persona_id']}")
    lines.append(f"- Number of decision points: {payload['num_decisions']}")
    lines.append("")
    lines.append("### Decision Trace")
    lines.append("")

    for step in payload["decision_trace"]:
        lines.append(f"#### Decision {step['decision_index']:02d} (env step {step['env_step']})")
        lines.append("")
        if step.get("goal_desc"):
            lines.append("Goal:")
            lines.append(step["goal_desc"])
            lines.append("")
        if step.get("progress_desc"):
            lines.append("Progress Summary:")
            lines.append(step["progress_desc"])
            lines.append("")
        lines.append("Observation:")
        lines.append("```json")
        lines.append(pretty_json_or_text(step.get("observation", "")))
        lines.append("```")
        lines.append("")
        lines.append("Think:")
        lines.append(step.get("think") or "(empty)")
        lines.append("")
        if step.get("message"):
            lines.append("Message:")
            lines.append(step["message"])
            lines.append("")
        lines.append("Selected Action:")
        lines.append(step.get("selected_action") or "(empty)")
        lines.append("")
        lines.append("Executed Env Action:")
        lines.append(str(step.get("env_action")))
        lines.append("")

    return "\n".join(lines).rstrip()


def build_item_markdown(item_meta: Dict[str, str], candidate_payloads: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"# {item_meta['item_id']}")
    lines.append("")
    lines.append(f"- Task: {item_meta['task_id']} ({item_meta['task_name']})")
    lines.append(f"- Target Human Personality: {item_meta['personality_label']}")
    lines.append(f"- Persona ID: {item_meta['persona_id']}")
    lines.append("- Ranking rule: rank the six candidates from most human-consistent to least human-consistent.")
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

    for idx, payload in enumerate(candidate_payloads):
        lines.append(build_candidate_markdown(payload["candidate_label"], payload["candidate_payload"]))
        if idx != len(candidate_payloads) - 1:
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_index_markdown(task_group_name: str, item_rows: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append(f"# {task_group_name}")
    lines.append("")
    lines.append("This folder contains single-file packets for one task group.")
    lines.append("")
    lines.append("Files:")
    lines.append("- `items/*.md`: single-file markdown packets")
    lines.append("- `items/*.pdf`: single-file PDF packets")
    lines.append("- `ranking_template_rater_1.csv`")
    lines.append("- `ranking_template_rater_2.csv`")
    lines.append("- `ranking_template_rater_3.csv`")
    lines.append("")
    lines.append("## Items")
    lines.append("")
    lines.append("| Item ID | Persona ID | Target Human Personality | Markdown | PDF |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in item_rows:
        item_file = f"{row['item_id']}.md"
        pdf_file = f"{row['item_id']}.pdf"
        lines.append(
            f"| {row['item_id']} | {row['persona_id']} | {row['personality_label']} | [{item_file}](items/{item_file}) | [{pdf_file}](items/{pdf_file}) |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_group_packet_markdown(task_group_name: str, item_rows: List[Dict[str, str]], task_dir: Path) -> str:
    lines: List[str] = []
    lines.append(f"# {task_group_name}")
    lines.append("")
    lines.append("This single-file packet contains all items for one rater.")
    lines.append("")
    lines.append("Instructions:")
    lines.append("1. Read the five items in order.")
    lines.append("2. For each item, rank candidates A-F from most human-consistent to least human-consistent.")
    lines.append("3. Use strict ranking only. No ties. No numeric scores.")
    lines.append("")
    lines.append("Items in this packet:")
    for row in item_rows:
        lines.append(
            f"- {row['item_id']}: persona {row['persona_id']} / {row['personality_label']}"
        )
    lines.append("")
    lines.append("=" * 72)
    lines.append("")

    for idx, row in enumerate(item_rows):
        item_md = (task_dir / "items" / f"{row['item_id']}.md").read_text(encoding="utf-8").strip()
        lines.append(item_md)
        if idx != len(item_rows) - 1:
            lines.append("")
            lines.append("=" * 72)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def markdown_to_plain_text(markdown_text: str) -> str:
    lines: List[str] = []
    in_code = False
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code and line.startswith("#"):
            stripped = line.lstrip("#").strip()
            lines.append(stripped)
            lines.append("")
            continue
        if line == "---":
            lines.append("")
            lines.append("=" * 72)
            lines.append("")
            continue
        if line.startswith("|"):
            # Drop markdown tables in PDF, keep ranking rows readable.
            if line.startswith("| ---"):
                continue
            line = line.strip("|")
            cells = [c.strip() for c in line.split("|")]
            line = " | ".join(cells)
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def wrap_line(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    if not text:
        return [""]

    indent = len(text) - len(text.lstrip(" "))
    prefix = " " * indent
    content = text.lstrip(" ")

    words = content.split(" ")
    lines: List[str] = []
    current = prefix

    def measure(s: str) -> float:
        return font.getlength(s)

    for word in words:
        candidate = word if current.strip() == "" else current + (" " if not current.endswith(" ") and current != prefix else "") + word
        if measure(candidate) <= max_width:
            current = candidate
            continue

        if current.strip():
            lines.append(current.rstrip())
            current = prefix

        if measure(prefix + word) <= max_width:
            current = prefix + word
            continue

        chunk = ""
        for ch in word:
            test = chunk + ch
            if measure(prefix + test) <= max_width:
                chunk = test
            else:
                if chunk:
                    lines.append(prefix + chunk)
                chunk = ch
        current = prefix + chunk

    if current.strip() or current == "":
        lines.append(current.rstrip())
    return lines


def render_text_pdf(text: str, out_path: Path) -> None:
    page_width = 900
    page_height = 1273
    margin_x = 50
    margin_y = 50
    font_size = 18
    line_gap = 8

    font = ImageFont.truetype(str(FONT_PATH), font_size)
    max_text_width = page_width - 2 * margin_x
    line_height = font_size + line_gap

    prepared_lines: List[str] = []
    for raw_line in text.splitlines():
        wrapped = wrap_line(raw_line, font, max_text_width)
        prepared_lines.extend(wrapped)
        if raw_line == "":
            prepared_lines.append("")

    pages: List[Image.Image] = []
    image = Image.new("L", (page_width, page_height), 255)
    draw = ImageDraw.Draw(image)
    y = margin_y

    for line in prepared_lines:
        if y + line_height > page_height - margin_y:
            pages.append(image.convert("1"))
            image = Image.new("L", (page_width, page_height), 255)
            draw = ImageDraw.Draw(image)
            y = margin_y
        draw.text((margin_x, y), line, fill="black", font=font)
        y += line_height

    pages.append(image.convert("1"))
    pages[0].save(out_path, save_all=True, append_images=pages[1:], resolution=110.0)


def copy_templates(src_task_dir: Path, dst_task_dir: Path) -> None:
    for src in sorted(src_task_dir.glob("ranking_template_rater_*.csv")):
        shutil.copy2(src, dst_task_dir / src.name)


def load_item_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def export_task_group(src_task_dir: Path, dst_task_dir: Path) -> Dict[str, int]:
    dst_items_dir = dst_task_dir / "items"
    dst_task_dir.mkdir(parents=True, exist_ok=True)
    dst_items_dir.mkdir(parents=True, exist_ok=True)
    copy_templates(src_task_dir, dst_task_dir)

    item_rows = load_item_manifest(src_task_dir / "item_manifest.csv")

    for item_row in item_rows:
        item_id = item_row["item_id"]
        src_item_dir = src_task_dir / "items" / item_id
        candidate_payloads: List[Dict[str, Any]] = []
        for candidate_json in sorted(src_item_dir.glob("candidate_*.json")):
            candidate_label = candidate_json.stem.split("_")[-1]
            candidate_payloads.append(
                {
                    "candidate_label": candidate_label,
                    "candidate_payload": load_json(candidate_json),
                }
            )

        candidate_payloads.sort(key=lambda x: x["candidate_label"])
        markdown_text = build_item_markdown(item_row, candidate_payloads)
        markdown_path = dst_items_dir / f"{item_id}.md"
        pdf_path = dst_items_dir / f"{item_id}.pdf"

        markdown_path.write_text(markdown_text, encoding="utf-8")
        render_text_pdf(markdown_to_plain_text(markdown_text), pdf_path)

    index_md = build_index_markdown(src_task_dir.name, item_rows)
    (dst_task_dir / "INDEX.md").write_text(index_md, encoding="utf-8")
    render_text_pdf(markdown_to_plain_text(index_md), dst_task_dir / "INDEX.pdf")

    packet_md = build_group_packet_markdown(src_task_dir.name, item_rows, dst_task_dir)
    (dst_task_dir / f"{src_task_dir.name}_packet.md").write_text(packet_md, encoding="utf-8")
    render_text_pdf(
        markdown_to_plain_text(packet_md),
        dst_task_dir / f"{src_task_dir.name}_packet.pdf",
    )

    return {
        "items": len(item_rows),
        "markdown_files": len(item_rows),
        "pdf_files": len(item_rows),
    }


def write_root_readme(out_dir: Path) -> None:
    text = """# CoELA Single-Page Ranking Packet

This directory contains easier-to-distribute human-annotation packets.

For each task group:
- `INDEX.md` and `INDEX.pdf` provide the entry page
- `items/<item_id>.md` provides one single-file markdown packet per ranking item
- `items/<item_id>.pdf` provides one single-file PDF packet per ranking item
- `ranking_template_rater_1.csv` / `2` / `3` are copied for direct annotation
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    for_raters_dir = input_dir / "for_raters"
    delivery_dir = output_dir / "delivery"

    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Required font not found: {FONT_PATH}")

    ensure_clean_dir(output_dir)
    delivery_dir.mkdir(parents=True, exist_ok=True)
    write_root_readme(output_dir)

    total_items = 0
    total_markdown = 0
    total_pdf = 0

    for src_task_dir in sorted([p for p in for_raters_dir.iterdir() if p.is_dir()]):
        dst_task_dir = output_dir / src_task_dir.name
        stats = export_task_group(src_task_dir, dst_task_dir)
        shutil.copy2(
            dst_task_dir / f"{src_task_dir.name}_packet.pdf",
            delivery_dir / f"{src_task_dir.name}.pdf",
        )
        shutil.copy2(
            dst_task_dir / f"{src_task_dir.name}_packet.md",
            delivery_dir / f"{src_task_dir.name}.md",
        )
        total_items += stats["items"]
        total_markdown += stats["markdown_files"]
        total_pdf += stats["pdf_files"]

    print(f"Wrote single-page packet to {output_dir}")
    print(f"Task groups: {len([p for p in for_raters_dir.iterdir() if p.is_dir()])}")
    print(f"Items: {total_items}")
    print(f"Markdown files: {total_markdown}")
    print(f"PDF files: {total_pdf}")


if __name__ == "__main__":
    main()
