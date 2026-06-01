#!/usr/bin/env python3
"""Evaluate threshold calibration and error patterns from CV eye predictions.

This does not retrain the image model. It reuses out-of-fold eye-level stack
scores and asks whether a threshold selected from other folds can improve the
balanced sensitivity/specificity tradeoff.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOF threshold calibration for angle-closure CV predictions.")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--score-col", default="stack_score")
    p.add_argument("--label-col", default="closure_label")
    return p.parse_args()


def finite_roc(y_true, score):
    fpr, tpr, thresholds = roc_curve(y_true, score)
    finite = np.isfinite(thresholds)
    return fpr[finite], tpr[finite], thresholds[finite]


def choose_balanced_threshold(y_true, score) -> float:
    fpr, tpr, thresholds = finite_roc(y_true, score)
    spec = 1.0 - fpr
    return float(thresholds[np.argmax(np.minimum(tpr, spec))])


def choose_youden_threshold(y_true, score) -> float:
    fpr, tpr, thresholds = finite_roc(y_true, score)
    return float(thresholds[np.argmax(tpr - fpr)])


def choose_sens_floor_threshold(y_true, score, target_sens: float) -> float:
    fpr, tpr, thresholds = finite_roc(y_true, score)
    spec = 1.0 - fpr
    ok = np.flatnonzero(tpr >= target_sens)
    if len(ok):
        return float(thresholds[ok[np.argmax(spec[ok])]])
    return float(thresholds[np.argmax(tpr)])


def threshold_metrics(y_true, score, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan,
        "auprc": float(average_precision_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(tp / (tp + fp)) if tp + fp else np.nan,
        "npv": float(tn / (tn + fn)) if tn + fn else np.nan,
        "accuracy": float((pred == y_true).mean()),
        "balanced_min": float(np.nanmin([sens, spec])),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_min"]
    return metrics.groupby("threshold_rule")[cols].agg(["mean", "std", "min", "max"]).round(4)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def error_summary(scored: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for rule, sub in scored.groupby("threshold_rule"):
        tmp = sub.copy()
        tmp["error_type"] = np.select(
            [
                (tmp[label_col] == 1) & (tmp["pred"] == 0),
                (tmp[label_col] == 0) & (tmp["pred"] == 1),
            ],
            ["false_negative", "false_positive"],
            default="correct",
        )
        for (etype, grade), g in tmp.groupby(["error_type", "angle_grade"], dropna=False):
            rows.append(
                {
                    "threshold_rule": rule,
                    "error_type": etype,
                    "angle_grade": grade,
                    "eyes": int(len(g)),
                }
            )
    return pd.DataFrame(rows).sort_values(["threshold_rule", "error_type", "angle_grade"])


def run(args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions, low_memory=False)
    df = df[df["split"].eq("val")].copy()
    df[args.label_col] = df[args.label_col].astype(int)
    folds = sorted(df["fold"].unique())

    metric_rows = []
    scored_rows = []
    for fold in folds:
        train_cal = df[~df["fold"].eq(fold)].copy()
        val = df[df["fold"].eq(fold)].copy()
        thresholds = {
            "balanced_from_other_oof_folds": choose_balanced_threshold(train_cal[args.label_col], train_cal[args.score_col]),
            "youden_from_other_oof_folds": choose_youden_threshold(train_cal[args.label_col], train_cal[args.score_col]),
            "sens70_from_other_oof_folds": choose_sens_floor_threshold(train_cal[args.label_col], train_cal[args.score_col], 0.70),
            "sens80_from_other_oof_folds": choose_sens_floor_threshold(train_cal[args.label_col], train_cal[args.score_col], 0.80),
            "balanced_from_current_fold_internal": choose_balanced_threshold(val[args.label_col], val[args.score_col]),
        }
        for rule, threshold in thresholds.items():
            metrics = threshold_metrics(val[args.label_col], val[args.score_col], threshold)
            metrics.update(
                {
                    "fold": int(fold),
                    "threshold_rule": rule,
                    "val_eyes": int(len(val)),
                    "val_positive_eyes": int(val[args.label_col].sum()),
                    "reached_70_70": bool(metrics["sensitivity"] >= 0.70 and metrics["specificity"] >= 0.70),
                    "reached_80_80": bool(metrics["sensitivity"] >= 0.80 and metrics["specificity"] >= 0.80),
                }
            )
            metric_rows.append(metrics)
            scored = val.copy()
            scored["threshold_rule"] = rule
            scored["threshold"] = threshold
            scored["pred"] = (scored[args.score_col] >= threshold).astype(int)
            scored_rows.append(scored)

    metrics_df = pd.DataFrame(metric_rows)
    scored_df = pd.concat(scored_rows, ignore_index=True)
    err_df = error_summary(scored_df, args.label_col)
    summary = summarize_metrics(metrics_df)
    metrics_df.to_csv(args.outdir / "threshold_calibration_fold_metrics.csv", index=False)
    scored_df.to_csv(args.outdir / "threshold_calibration_predictions.csv", index=False)
    err_df.to_csv(args.outdir / "threshold_calibration_error_by_grade.csv", index=False)
    summary.to_csv(args.outdir / "threshold_calibration_summary.csv")

    summary_flat = summary.reset_index()
    summary_flat.columns = [
        "_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else str(col)
        for col in summary_flat.columns
    ]
    with open(args.outdir / "RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# OOF Threshold Calibration Results\n\n")
        f.write(f"Input predictions: `{args.predictions}`\n\n")
        f.write("This analysis reuses out-of-fold eye-level scores and selects thresholds from all other folds before evaluating each held-out fold.\n\n")
        f.write("## Summary\n\n")
        f.write(markdown_table(summary_flat))
        f.write("\n\n## Fold Metrics\n\n")
        f.write(markdown_table(metrics_df))
        f.write("\n\n## Error Counts by Grade\n\n")
        f.write(markdown_table(err_df))
        f.write("\n")
    print(f"[DONE] Wrote {args.outdir}")


if __name__ == "__main__":
    run(parse_args())
