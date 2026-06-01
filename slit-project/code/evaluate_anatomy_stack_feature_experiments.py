#!/usr/bin/env python3
"""Evaluate shallow risk models on anatomy-stack prediction CSVs.

This script tests two no-retraining experiments on top of an existing image-level
anatomy prediction run:

1. View-aware binary aggregation: summarize predicted anatomy separately for
   center, van_nasal, and van_temporal views.
2. Three-class risk: train multinomial logistic regression for closed (0/1),
   borderline (2), and open (3/4), then evaluate closed probability as the
   binary angle-closure score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


META_COLS = ["fold", "split", "participant_id", "eye_code", "combo_key", "angle_grade", "closure_label"]
VIEWS = ["center", "van_nasal", "van_temporal"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate view-aware and 3-class anatomy-stack risk models.")
    p.add_argument("--image-predictions", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def finite_roc(y_true, score):
    fpr, tpr, thresholds = roc_curve(y_true, score)
    finite = np.isfinite(thresholds)
    return fpr[finite], tpr[finite], thresholds[finite]


def choose_balanced_threshold(y_true, score) -> float:
    fpr, tpr, thresholds = finite_roc(y_true, score)
    spec = 1.0 - fpr
    return float(thresholds[np.argmax(np.minimum(tpr, spec))])


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


def summarize_global_mean(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(META_COLS, as_index=False)
        .agg({**{c: "mean" for c in pred_cols}, "image_id": "count"})
        .rename(columns={"image_id": "n_images"})
    )


def summarize_view_aware(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    base = df.groupby(META_COLS, as_index=False).agg({"image_id": "count"}).rename(columns={"image_id": "n_images"})
    parts = [base.set_index(META_COLS)]
    for view in VIEWS:
        sub = df[df["view_label"].eq(view)].copy()
        if sub.empty:
            continue
        stats = sub.groupby(META_COLS)[pred_cols].agg(["mean", "std", "min"])
        stats.columns = [f"{view}_{stat}_{col.replace('pred_', '')}" for col, stat in stats.columns]
        q10 = sub.groupby(META_COLS)[pred_cols].quantile(0.10)
        q10.columns = [f"{view}_q10_{c.replace('pred_', '')}" for c in pred_cols]
        count = sub.groupby(META_COLS).size().to_frame(f"{view}_n_images")
        parts.extend([stats, q10, count])
    out = pd.concat(parts, axis=1).reset_index()
    return out


def fit_binary_logistic(train_df: pd.DataFrame, feature_cols: list[str], seed: int):
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5, random_state=seed),
    )
    clf.fit(train_df[feature_cols], train_df["closure_label"].astype(int))
    return clf


def add_binary_scores(df: pd.DataFrame, clf, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["risk_score"] = clf.predict_proba(out[feature_cols])[:, 1]
    return out


def grade_class(angle_grade: float) -> int:
    if angle_grade <= 1:
        return 0
    if angle_grade == 2:
        return 1
    return 2


def fit_three_class_logistic(train_df: pd.DataFrame, feature_cols: list[str], seed: int):
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5, random_state=seed, multi_class="multinomial"),
    )
    clf.fit(train_df[feature_cols], train_df["grade_class"].astype(int))
    return clf


def add_three_class_scores(df: pd.DataFrame, clf, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    prob = clf.predict_proba(out[feature_cols])
    classes = list(clf.named_steps["logisticregression"].classes_)
    closed_idx = classes.index(0)
    borderline_idx = classes.index(1) if 1 in classes else None
    out["risk_score"] = prob[:, closed_idx]
    out["prob_closed_01"] = prob[:, closed_idx]
    out["prob_borderline_2"] = prob[:, borderline_idx] if borderline_idx is not None else np.nan
    return out


def evaluate_per_fold(features: pd.DataFrame, feature_cols: list[str], model_kind: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []
    for fold in sorted(features["fold"].unique()):
        fold_df = features[features["fold"].eq(fold)].copy()
        train_df = fold_df[fold_df["split"].eq("train")].copy()
        val_df = fold_df[fold_df["split"].eq("val")].copy()
        if model_kind == "binary":
            clf = fit_binary_logistic(train_df, feature_cols, seed + int(fold))
            train_scored = add_binary_scores(train_df, clf, feature_cols)
            val_scored = add_binary_scores(val_df, clf, feature_cols)
        elif model_kind == "three_class":
            train_df["grade_class"] = train_df["angle_grade"].apply(grade_class)
            val_df["grade_class"] = val_df["angle_grade"].apply(grade_class)
            clf = fit_three_class_logistic(train_df, feature_cols, seed + int(fold))
            train_scored = add_three_class_scores(train_df, clf, feature_cols)
            val_scored = add_three_class_scores(val_df, clf, feature_cols)
        else:
            raise ValueError(model_kind)
        train_threshold = choose_balanced_threshold(train_scored["closure_label"], train_scored["risk_score"])
        val_threshold = choose_balanced_threshold(val_scored["closure_label"], val_scored["risk_score"])
        for rule, threshold in [
            ("balanced_min_from_train", train_threshold),
            ("balanced_min_from_val_internal", val_threshold),
        ]:
            row = threshold_metrics(val_scored["closure_label"], val_scored["risk_score"], threshold)
            row.update(
                {
                    "fold": int(fold),
                    "model_kind": model_kind,
                    "threshold_rule": rule,
                    "train_eyes": int(train_scored["combo_key"].nunique()),
                    "val_eyes": int(val_scored["combo_key"].nunique()),
                    "val_positive_eyes": int(val_scored["closure_label"].sum()),
                    "reached_70_70": bool(row["sensitivity"] >= 0.70 and row["specificity"] >= 0.70),
                }
            )
            metric_rows.append(row)
        val_scored["fold"] = fold
        pred_rows.append(val_scored)
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_min"]
    return metrics.groupby(["model_kind", "threshold_rule"])[cols].agg(["mean", "std", "min", "max"]).round(4)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(args.image_predictions, low_memory=False)
    images["view_label"] = images["view_label"].astype(str).str.strip().str.lower()
    pred_cols = [c for c in images.columns if c.startswith("pred_")]
    global_features = summarize_global_mean(images, pred_cols)
    view_features = summarize_view_aware(images, pred_cols)

    view_feature_cols = [c for c in view_features.columns if c not in META_COLS]
    mean_feature_cols = [c for c in global_features.columns if c.startswith("pred_")]
    view_metrics, view_preds = evaluate_per_fold(view_features, view_feature_cols, "binary", args.seed)
    three_metrics, three_preds = evaluate_per_fold(global_features, mean_feature_cols, "three_class", args.seed)

    view_features.to_csv(args.outdir / "view_aware_eye_features.csv", index=False)
    global_features.to_csv(args.outdir / "global_mean_eye_features.csv", index=False)
    view_metrics.to_csv(args.outdir / "view_aware_binary_metrics.csv", index=False)
    three_metrics.to_csv(args.outdir / "three_class_mean_metrics.csv", index=False)
    view_preds.to_csv(args.outdir / "view_aware_binary_val_predictions.csv", index=False)
    three_preds.to_csv(args.outdir / "three_class_mean_val_predictions.csv", index=False)
    all_metrics = pd.concat([view_metrics, three_metrics], ignore_index=True)
    all_metrics.to_csv(args.outdir / "all_metrics.csv", index=False)
    summary = summarize(all_metrics)
    summary.to_csv(args.outdir / "metric_summary.csv")
    with open(args.outdir / "RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# Anatomy Stack Feature Experiments\n\n")
        f.write("Input predictions:\n\n")
        f.write(f"`{args.image_predictions}`\n\n")
        f.write("Fold metrics:\n\n")
        f.write(markdown_table(all_metrics))
        f.write("\n\nSummary:\n\n")
        summary_flat = summary.reset_index()
        summary_flat.columns = [
            "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
            for col in summary_flat.columns
        ]
        f.write(markdown_table(summary_flat))
        f.write("\n")
    print(f"[DONE] Outputs written to {args.outdir}")


if __name__ == "__main__":
    run(parse_args())
