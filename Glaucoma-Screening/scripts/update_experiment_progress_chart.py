#!/usr/bin/env python3
"""Regenerate the Paper 2 AUROC progress chart from the milestone CSV.

The CSV is intentionally hand-curated: it should contain only the experiments
that changed the modeling direction or produced a meaningful AUROC jump.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "Glaucoma-Screening" / "results" / "experiment_progress_milestones.csv"
SVG_PATH = ROOT / "Glaucoma-Screening" / "results" / "experiment_progress_chart.svg"
MD_PATH = ROOT / "Glaucoma-Screening" / "results" / "experiment_progress_chart.md"


COLORS = {
    "Fixed split diagnostic": "#7a7f8c",
    "Fixed split validation-selected": "#7a7f8c",
    "80/20 validation": "#d97706",
    "5-fold CV": "#2563eb",
}


def read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["sequence"]))
    return rows


def fmt(value: str) -> str:
    if value == "":
        return ""
    return f"{float(value):.3f}"


def generate_svg(rows: list[dict[str, str]]) -> None:
    width, height = 1200, 680
    left, right, top, bottom = 92, 48, 86, 132
    chart_w = width - left - right
    chart_h = height - top - bottom
    y_min, y_max = 0.30, 0.80

    def x_pos(sequence: int) -> float:
        if len(rows) == 1:
            return left + chart_w / 2
        return left + (sequence - 1) * chart_w / (len(rows) - 1)

    def y_pos(auroc: float) -> float:
        return top + (y_max - auroc) * chart_h / (y_max - y_min)

    points = [
        (x_pos(i + 1), y_pos(float(row["auroc"])), row)
        for i, row in enumerate(rows)
    ]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        ".title{font:700 28px Arial,sans-serif;fill:#111827}",
        ".subtitle{font:400 15px Arial,sans-serif;fill:#4b5563}",
        ".axis{stroke:#111827;stroke-width:1.5}",
        ".grid{stroke:#e5e7eb;stroke-width:1}",
        ".tick{font:12px Arial,sans-serif;fill:#4b5563}",
        ".xlabel{font:12px Arial,sans-serif;fill:#374151}",
        ".pointlabel{font:700 12px Arial,sans-serif;fill:#111827}",
        ".note{font:11px Arial,sans-serif;fill:#4b5563}",
        ".legend{font:13px Arial,sans-serif;fill:#374151}",
        "</style>",
        '<text x="92" y="38" class="title">Angle-Closure Model Progress</text>',
        '<text x="92" y="64" class="subtitle">Selected milestone experiments only. Y-axis is AUROC; X-axis is experiment sequence.</text>',
    ]

    for tick in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-14}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>')

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>')
    parts.append(f'<text x="{left-56}" y="{top+chart_h/2:.1f}" transform="rotate(-90 {left-56} {top+chart_h/2:.1f})" text-anchor="middle" class="xlabel">AUROC</text>')
    parts.append(f'<text x="{left+chart_w/2:.1f}" y="{height-28}" text-anchor="middle" class="xlabel">Experiment number</text>')

    # Draw connecting path to show the development sequence.
    path = " ".join(
        ("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}"
        for i, (x, y, _) in enumerate(points)
    )
    parts.append(f'<path d="{path}" fill="none" stroke="#9ca3af" stroke-width="2" stroke-dasharray="5 5"/>')

    label_offsets = [-30, -44, -30, -44, -30, -44, -30, -48]
    for i, (x, y, row) in enumerate(points):
        color = COLORS.get(row["validation_design"], "#111827")
        label_y = y + label_offsets[i % len(label_offsets)]
        parts.append(f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+6}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+23}" text-anchor="middle" class="tick">{row["sequence"]}</text>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle" class="pointlabel">{html.escape(row["display_label"])}</text>')
        parts.append(f'<text x="{x:.1f}" y="{label_y+15:.1f}" text-anchor="middle" class="note">AUROC {fmt(row["auroc"])}</text>')

    legend_x, legend_y = width - 390, 28
    for j, (name, color) in enumerate(COLORS.items()):
        y = legend_y + j * 22
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+16}" y="{y+4}" class="legend">{html.escape(name)}</text>')

    parts.append(f'<text x="{left}" y="{height-72}" class="note">Important: 80/20 and 5-fold CV points are shown together for experiment history, but they are not equivalent validation designs.</text>')
    parts.append(f'<text x="{left}" y="{height-54}" class="note">Current best rigorous candidate: experiment 8, regularized unfrozen ConvNeXt-Tiny angle-6 anatomy stack.</text>')
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")


def generate_markdown(rows: list[dict[str, str]]) -> None:
    lines = [
        "# AUROC Progress Chart",
        "",
        "This chart tracks only the milestone experiments that changed the modeling direction or produced a meaningful AUROC jump.",
        "",
        f"![AUROC progress](experiment_progress_chart.svg)",
        "",
        "Update workflow:",
        "",
        "```bash",
        "python Glaucoma-Screening/scripts/update_experiment_progress_chart.py",
        "```",
        "",
        "Source data: `Glaucoma-Screening/results/experiment_progress_milestones.csv`.",
        "",
        "Important: fixed-split, 80/20, and 5-fold CV points are shown together for history. They should not be interpreted as directly equivalent evidence.",
        "",
        "| # | Method | Validation | AUROC | Sens | Spec | Why it matters |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {seq} | {method} | {validation} | {auroc} | {sens} | {spec} | {notes} |".format(
                seq=row["sequence"],
                method=row["method"],
                validation=row["validation_design"],
                auroc=fmt(row["auroc"]),
                sens=fmt(row["sensitivity"]),
                spec=fmt(row["specificity"]),
                notes=row["notes"],
            )
        )
    lines.extend(
        [
            "",
            "## Milestone Summaries",
            "",
            "These short summaries explain how each experiment changed the next step. The goal is to keep the development logic visible, not just list scores.",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['sequence']}. {row['display_label']}",
                "",
                f"**Question:** {row.get('question', '').strip()}",
                "",
                f"**Approach:** {row.get('approach', '').strip()}",
                "",
                (
                    "**Result:** "
                    f"{row.get('result_summary', '').strip()} "
                    f"Validation design: {row['validation_design']}. "
                    f"Run path: `{row['run_path']}`."
                ),
                "",
                f"**Interpretation:** {row.get('interpretation', '').strip()}",
                "",
                f"**Decision:** {row.get('decision', '').strip()}",
                "",
            ]
        )
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = read_rows()
    generate_svg(rows)
    generate_markdown(rows)
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
