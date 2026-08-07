#!/usr/bin/env python3
"""Validate the published open tabular slit-lamp/AS-OCT release."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED = {
    "rows": 562,
    "participants": 286,
    "left_eyes": 282,
    "right_eyes": 280,
    "angle_counts": {"0": 24, "1": 31, "2": 89, "3": 363, "4": 54, "not seen": 1},
}

REQUIRED_COLUMNS = {
    "patient_id",
    "age",
    "sex",
    "Iop",
    "angle",
    "lens",
    "CCT",
    "Eye",
    "ACD[Endo.]",
    "LV",
    "AOD500(L)",
    "AOD500(R)",
    "TISA500(L)",
    "TISA500(R)",
}


def normalize_eye(value: object) -> str:
    text = str(value).strip().upper()
    if text in {"OD", "R", "RIGHT", "OD(RIGHT)"}:
        return "R"
    if text in {"OS", "L", "LEFT", "OS(LEFT)"}:
        return "L"
    return ""


def normalize_angle(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text.endswith(".0") and text[:-2] in {"0", "1", "2", "3", "4"}:
        return text[:-2]
    return text


def validate(path: Path) -> list[str]:
    frame = pd.read_csv(path, low_memory=False)
    failures: list[str] = []

    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        failures.append(f"missing required columns: {missing_columns}")
        return failures

    if len(frame) != EXPECTED["rows"]:
        failures.append(f"rows: observed {len(frame)}, expected {EXPECTED['rows']}")

    participant_count = frame["patient_id"].nunique(dropna=True)
    if participant_count != EXPECTED["participants"]:
        failures.append(
            f"participants: observed {participant_count}, expected {EXPECTED['participants']}"
        )

    eye = frame["Eye"].map(normalize_eye)
    invalid_eye_rows = int(eye.eq("").sum())
    if invalid_eye_rows:
        failures.append(f"unrecognized eye laterality rows: {invalid_eye_rows}")

    eye_counts = eye.value_counts().to_dict()
    for side, expected in (("L", EXPECTED["left_eyes"]), ("R", EXPECTED["right_eyes"])):
        if eye_counts.get(side, 0) != expected:
            failures.append(
                f"{side} eyes: observed {eye_counts.get(side, 0)}, expected {expected}"
            )

    key = frame["patient_id"].astype("string") + "_" + eye.astype("string")
    duplicate_keys = int(key.duplicated(keep=False).sum())
    if duplicate_keys:
        failures.append(f"rows in duplicated patient-eye keys: {duplicate_keys}")

    angle = frame["angle"].map(normalize_angle)
    angle_counts = angle.value_counts().to_dict()
    if angle_counts != EXPECTED["angle_counts"]:
        failures.append(
            f"angle distribution: observed {angle_counts}, expected {EXPECTED['angle_counts']}"
        )

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to Zenodo data.csv")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Data file not found: {args.data}")

    failures = validate(args.data)
    if failures:
        print("Release validation: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Release validation: PASS")
    print("- 562 unique eye records")
    print("- 286 participants")
    print("- 282 left eyes and 280 right eyes")
    print("- published Shaffer-grade distribution reproduced")


if __name__ == "__main__":
    main()
