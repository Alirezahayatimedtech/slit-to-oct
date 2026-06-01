#!/usr/bin/env python3
"""
Fast Week-1 experiment:

1. Clean angle labels at eye level.
2. Train a regression-only multi-task ResNet-50 for 10 AS-OCT anatomical targets.
3. Predict anatomical values for images in each held-out patient fold.
4. Aggregate predicted anatomy to eye level.
5. Fit logistic regression on predicted anatomy from training folds and evaluate
   on the held-out fold.

This is intentionally a bounded experiment script. It does not add a neural
classification head; angle-closure classification is done only by the shallow
logistic model on predicted anatomy.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

ANATOMY_TARGETS = [
    ("CCT", "cct_um"),
    ("ACD[Endo.]", "acd_endo_mm"),
    ("LV", "lens_vault_mm"),
    ("ACW", "acw_mm"),
    ("AOD500_temporal", "aod500_temporal_mm"),
    ("AOD500_nasal", "aod500_nasal_mm"),
    ("TISA500_temporal", "tisa500_temporal_mm2"),
    ("TISA500_nasal", "tisa500_nasal_mm2"),
    ("TIA500_temporal", "tia500_temporal_deg"),
    ("TIA500_nasal", "tia500_nasal_deg"),
]
USABLE_VIEWS = {"center", "van_nasal", "van_temporal"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anatomy-regression stacker for angle closure.")
    p.add_argument("--image-csv", type=Path, default=PROJECT_ROOT / "code" / "ready_for_training_clustered_anatomical_with_means_with_views_anonymized.csv")
    p.add_argument("--clinical-csv", type=Path, default=PROJECT_ROOT / "code" / "ready_for_upload_publish.csv")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "paper2_runs" / "resnet50_anatomy_stack_cv")
    p.add_argument("--split-manifest", type=Path, default=None, help="Optional fixed patient split manifest with participant_id and train/val/test split columns.")
    p.add_argument("--folds", type=int, default=1, help="Use 1 for a fast 80/20 train/validation run; use 5 for CV.")
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--backbone", choices=["resnet50", "convnext_tiny"], default="resnet50")
    p.add_argument(
        "--target-preset",
        choices=["all10", "angle6"],
        default="all10",
        help="all10 uses all AS-OCT targets; angle6 focuses on ACD, lens vault, AOD500, and TISA500.",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--view-mode", choices=["usable", "center", "van_nasal", "van_temporal", "all"], default="usable")
    p.add_argument("--closure-grade-max", type=float, default=1.0)
    p.add_argument(
        "--exclude-angle-grades",
        type=str,
        default="",
        help="Comma-separated grades to exclude as ambiguous, e.g. '2' for a clean-label exploratory run.",
    )
    p.add_argument("--max-train-images-per-eye", type=int, default=12)
    p.add_argument("--max-val-images-per-eye", type=int, default=0, help="0 means predict every validation image.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true", help="Train only the regression head for speed.")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--bootstrap", type=int, default=500, help="Patient-level bootstrap resamples for fixed-split selected test metrics.")
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"---": np.nan, "": np.nan, "not seen": np.nan, "Not seen": np.nan}), errors="coerce")


def eye_code(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if s in {"OD", "R", "RIGHT"} or s.startswith("R"):
        return "R"
    if s in {"OS", "L", "LEFT"} or s.startswith("L"):
        return "L"
    return None


def combo_key(participant_id, eye: str) -> str:
    return f"{int(participant_id)}_{eye}"


def resolve_image_path(raw_path: str) -> str | None:
    raw = str(raw_path).replace("\\", "/")
    fname = os.path.basename(raw)
    candidates = [
        Path(raw),
        PROJECT_ROOT / "data" / "center_roi_images" / "all_slit_images_448" / fname,
        REPO_ROOT / "slit-oct" / "colab_ready_images" / fname,
        REPO_ROOT / "colab_ready_images" / fname,
        PROJECT_ROOT / "code" / fname,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def parse_excluded_grades(raw: str) -> set[float]:
    if not raw.strip():
        return set()
    return {float(x.strip()) for x in raw.split(",") if x.strip()}


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    clinical = pd.read_csv(args.clinical_csv, low_memory=False)
    image = pd.read_csv(args.image_csv, low_memory=False)

    clin = pd.DataFrame()
    clin["participant_id"] = clean_numeric(clinical["subject_id"])
    clin["eye_code"] = clinical["eye"].apply(eye_code)
    clin["angle_grade"] = clean_numeric(clinical["angle_grade"])
    excluded = parse_excluded_grades(args.exclude_angle_grades)
    clin = clin.dropna(subset=["participant_id", "eye_code", "angle_grade"]).copy()
    if excluded:
        clin = clin[~clin["angle_grade"].isin(excluded)].copy()
    clin["closure_label"] = (clin["angle_grade"] <= args.closure_grade_max).astype(int)
    clin["participant_id"] = clin["participant_id"].astype(int)
    clin["combo_key"] = clin.apply(lambda r: combo_key(r["participant_id"], r["eye_code"]), axis=1)
    clin = clin[["participant_id", "eye_code", "combo_key", "angle_grade", "closure_label"]].drop_duplicates("combo_key")

    required_image_cols = {"Patient_Num", "eye_clean", "Image_Path", "View_Label"} | {src for src, _ in ANATOMY_TARGETS}
    missing = required_image_cols - set(image.columns)
    if missing:
        raise SystemExit(f"Image CSV missing required column(s): {sorted(missing)}")

    img = pd.DataFrame()
    img["participant_id"] = clean_numeric(image["Patient_Num"])
    img["eye_code"] = image["eye_clean"].apply(eye_code)
    img["view_label"] = image["View_Label"].astype(str).str.strip().str.lower()
    img["image_path"] = image["Image_Path"].apply(resolve_image_path)
    for src, dst in ANATOMY_TARGETS:
        img[dst] = clean_numeric(image[src])
    img = img.dropna(subset=["participant_id", "eye_code", "image_path"] + [dst for _, dst in ANATOMY_TARGETS]).copy()
    img["participant_id"] = img["participant_id"].astype(int)
    img["combo_key"] = img.apply(lambda r: combo_key(r["participant_id"], r["eye_code"]), axis=1)
    if args.view_mode == "usable":
        img = img[img["view_label"].isin(USABLE_VIEWS)].copy()
    elif args.view_mode != "all":
        img = img[img["view_label"] == args.view_mode].copy()
    img = img[img["image_path"].apply(lambda p: p is not None and os.path.exists(str(p)))].copy()

    df = img.merge(clin, on=["participant_id", "eye_code", "combo_key"], how="inner")
    if df.empty:
        raise SystemExit("No rows after joining image and clinical labels.")
    df["image_id"] = np.arange(len(df))
    return df.reset_index(drop=True)


def write_label_audit(df: pd.DataFrame, args: argparse.Namespace) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    eye_df = df.drop_duplicates("combo_key").copy()
    grade_counts = eye_df["angle_grade"].value_counts().sort_index().rename_axis("angle_grade").reset_index(name="eyes")
    grade_counts.to_csv(args.outdir / "label_grade_counts.csv", index=False)
    summary = {
        "rows_images": int(len(df)),
        "eyes": int(eye_df["combo_key"].nunique()),
        "participants": int(eye_df["participant_id"].nunique()),
        "positive_eyes": int(eye_df["closure_label"].sum()),
        "negative_eyes": int((eye_df["closure_label"] == 0).sum()),
        "closure_grade_max": float(args.closure_grade_max),
        "excluded_angle_grades": sorted(parse_excluded_grades(args.exclude_angle_grades)),
        "view_mode": args.view_mode,
        "targets": [dst for _, dst in ANATOMY_TARGETS],
    }
    with open(args.outdir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), **summary}, f, indent=2, default=str)
    with open(args.outdir / "README.md", "w", encoding="utf-8") as f:
        f.write("# Anatomy Stack CV\n\n")
        f.write(f"Regression-only `{args.backbone}` predicts 10 AS-OCT anatomical parameters. ")
        f.write("A logistic regression then classifies strict angle closure from predicted anatomy.\n\n")
        f.write(f"- Images: {summary['rows_images']}\n")
        f.write(f"- Eyes: {summary['eyes']}\n")
        f.write(f"- Participants: {summary['participants']}\n")
        f.write(f"- Positive eyes: {summary['positive_eyes']}\n")
        f.write(f"- Negative eyes: {summary['negative_eyes']}\n")
        f.write(f"- Excluded angle grades: {summary['excluded_angle_grades']}\n")
        f.write(f"- View mode: `{args.view_mode}`\n")


def target_columns_for_preset(preset: str) -> list[str]:
    all_cols = [dst for _, dst in ANATOMY_TARGETS]
    if preset == "all10":
        return all_cols
    if preset == "angle6":
        return [
            "acd_endo_mm",
            "lens_vault_mm",
            "aod500_temporal_mm",
            "aod500_nasal_mm",
            "tisa500_temporal_mm2",
            "tisa500_nasal_mm2",
        ]
    raise ValueError(f"Unsupported target preset: {preset}")


class ImageAnatomyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_cols: list[str], scaler: StandardScaler | None, tfm):
        self.df = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.scaler = scaler
        self.tfm = tfm
        y = self.df[target_cols].to_numpy(dtype=np.float32)
        self.y = scaler.transform(y).astype(np.float32) if scaler is not None else y

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.tfm(img), torch.tensor(self.y[idx], dtype=torch.float32), int(row["image_id"])


class ImageRegressor(nn.Module):
    def __init__(self, out_dim: int, backbone: str = "resnet50", pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base = models.resnet50(weights=weights)
            hidden = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.convnext_tiny(weights=weights)
            hidden = base.classifier[2].in_features
            base.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone = base
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, out_dim))

    def forward(self, x):
        feat = self.backbone(x)
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return self.head(feat)


def make_transforms(img_size: int):
    train_tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfm, eval_tfm


def cap_images_per_eye(df: pd.DataFrame, max_images: int, seed: int) -> pd.DataFrame:
    if not max_images or max_images <= 0:
        return df
    return (
        df.groupby("combo_key", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), max_images), random_state=seed))
        .reset_index(drop=True)
    )


def run_epoch(model, loader, optimizer, scaler, device, train: bool, amp: bool):
    model.train(train)
    loss_fn = nn.SmoothL1Loss()
    losses = []
    preds = []
    ids = []
    for x, y, image_ids in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                pred = model(x)
                loss = loss_fn(pred, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if not train:
            preds.append(pred.detach().cpu().numpy())
            ids.extend(image_ids.numpy().tolist())
    pred_arr = np.concatenate(preds, axis=0) if preds else np.empty((0, 0))
    return float(np.mean(losses)) if losses else np.nan, ids, pred_arr


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


def choose_balanced_threshold(y_true, score) -> float:
    fpr, tpr, thresholds = finite_roc(y_true, score)
    spec = 1.0 - fpr
    return float(thresholds[np.argmax(np.minimum(tpr, spec))])


def finite_roc(y_true, score):
    fpr, tpr, thresholds = roc_curve(y_true, score)
    finite = np.isfinite(thresholds)
    return fpr[finite], tpr[finite], thresholds[finite]


def choose_youden_threshold(y_true, score) -> float:
    fpr, tpr, thresholds = finite_roc(y_true, score)
    return float(thresholds[np.argmax(tpr - fpr)])


def choose_threshold_at_sensitivity(y_true, score, target_sens: float = 0.80) -> float:
    fpr, tpr, thresholds = finite_roc(y_true, score)
    spec = 1.0 - fpr
    ok = np.flatnonzero(tpr >= target_sens)
    if len(ok):
        return float(thresholds[ok[np.argmax(spec[ok])]])
    return float(thresholds[np.argmax(tpr)])


def choose_threshold_at_specificity(y_true, score, target_spec: float = 0.80) -> float:
    fpr, tpr, thresholds = finite_roc(y_true, score)
    spec = 1.0 - fpr
    ok = np.flatnonzero(spec >= target_spec)
    if len(ok):
        return float(thresholds[ok[np.argmax(tpr[ok])]])
    return float(thresholds[np.argmax(spec)])


def aggregate_eye_predictions(pred_df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    pred_cols = [f"pred_{c}" for c in target_cols]
    agg = (
        pred_df.groupby(["fold", "split", "participant_id", "eye_code", "combo_key", "angle_grade", "closure_label"], as_index=False)
        .agg({**{c: "mean" for c in pred_cols}, "image_id": "count"})
        .rename(columns={"image_id": "n_images"})
    )
    return agg


def split_ids_from_manifest(df: pd.DataFrame, manifest_path: Path) -> dict[str, set[int]]:
    manifest = pd.read_csv(manifest_path, low_memory=False)
    required = {"participant_id", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"Split manifest missing required column(s): {sorted(missing)}")
    manifest = manifest[["participant_id", "split"]].copy()
    manifest["participant_id"] = clean_numeric(manifest["participant_id"])
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    manifest = manifest.dropna(subset=["participant_id"])
    manifest["participant_id"] = manifest["participant_id"].astype(int)
    manifest = manifest[manifest["split"].isin({"train", "val", "test"})].drop_duplicates()
    conflicts = manifest.groupby("participant_id")["split"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise SystemExit(f"Split manifest has participants assigned to multiple splits: {conflicts.index.tolist()[:10]}")
    available = set(df["participant_id"].astype(int))
    split_ids = {
        split: set(manifest.loc[manifest["split"] == split, "participant_id"].astype(int)) & available
        for split in ["train", "val", "test"]
    }
    for split, ids in split_ids.items():
        if not ids:
            raise SystemExit(f"No available image rows for split `{split}` after applying {manifest_path}")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_ids[a] & split_ids[b]
        if overlap:
            raise SystemExit(f"Patient overlap between {a} and {b}: {sorted(overlap)[:10]}")
    return split_ids


def fit_stack_model(train_eye: pd.DataFrame, target_cols: list[str], seed: int):
    pred_cols = [f"pred_{c}" for c in target_cols]
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=seed),
    )
    clf.fit(train_eye[pred_cols], train_eye["closure_label"].astype(int))
    return clf


def add_stack_scores(eye_df: pd.DataFrame, clf, target_cols: list[str]) -> pd.DataFrame:
    pred_cols = [f"pred_{c}" for c in target_cols]
    out = eye_df.copy()
    out["stack_score"] = clf.predict_proba(out[pred_cols])[:, 1]
    return out


def bootstrap_ci(df: pd.DataFrame, threshold: float, n_boot: int, seed: int) -> pd.DataFrame:
    if n_boot <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    participants = np.array(sorted(df["participant_id"].astype(int).unique()))
    rows = []
    for _ in range(n_boot):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        sub = pd.concat([df[df["participant_id"].astype(int) == pid] for pid in sampled], ignore_index=True)
        if sub["closure_label"].nunique() < 2:
            continue
        rows.append(threshold_metrics(sub["closure_label"], sub["stack_score"], threshold))
    if not rows:
        return pd.DataFrame()
    boot = pd.DataFrame(rows)
    ci_rows = []
    for metric in ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_min"]:
        vals = boot[metric].dropna()
        if vals.empty:
            continue
        ci_rows.append({"metric": metric, "ci_low": vals.quantile(0.025), "ci_high": vals.quantile(0.975)})
    return pd.DataFrame(ci_rows)


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


def logistic_stack(train_eye: pd.DataFrame, val_eye: pd.DataFrame, target_cols: list[str], seed: int) -> tuple[pd.DataFrame, dict]:
    pred_cols = [f"pred_{c}" for c in target_cols]
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=seed),
    )
    clf.fit(train_eye[pred_cols], train_eye["closure_label"].astype(int))
    train_score = clf.predict_proba(train_eye[pred_cols])[:, 1]
    threshold = choose_balanced_threshold(train_eye["closure_label"], train_score)
    out = val_eye.copy()
    out["stack_score"] = clf.predict_proba(val_eye[pred_cols])[:, 1]
    out["stack_pred"] = (out["stack_score"] >= threshold).astype(int)
    metrics = threshold_metrics(out["closure_label"], out["stack_score"], threshold)
    metrics["threshold_rule"] = "balanced_min_from_train"
    metrics["reached_70_70"] = bool(metrics["sensitivity"] >= 0.70 and metrics["specificity"] >= 0.70)
    return out, metrics


def iter_splits(eye_labels: pd.DataFrame, args: argparse.Namespace):
    if args.folds <= 1:
        train_ids, val_ids = train_test_split(
            eye_labels["participant_id"],
            test_size=args.val_fraction,
            random_state=args.seed,
            stratify=eye_labels["closure_label"],
        )
        yield 1, set(train_ids.astype(int)), set(val_ids.astype(int))
        return
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(eye_labels["participant_id"], eye_labels["closure_label"]), start=1):
        train_ids = set(eye_labels.iloc[train_idx]["participant_id"].astype(int))
        val_ids = set(eye_labels.iloc[val_idx]["participant_id"].astype(int))
        yield fold, train_ids, val_ids


def run_fixed_split(
    args: argparse.Namespace,
    df: pd.DataFrame,
    target_cols: list[str],
    train_tfm,
    eval_tfm,
    device: torch.device,
) -> None:
    split_ids = split_ids_from_manifest(df, args.split_manifest)
    split_dir = args.outdir / "fixed_split"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["participant_id"].isin(split_ids["train"])].copy()
    val_df = df[df["participant_id"].isin(split_ids["val"])].copy()
    test_df = df[df["participant_id"].isin(split_ids["test"])].copy()
    train_df = cap_images_per_eye(train_df, args.max_train_images_per_eye, args.seed)
    val_df = cap_images_per_eye(val_df, args.max_val_images_per_eye, args.seed)
    test_df = cap_images_per_eye(test_df, args.max_val_images_per_eye, args.seed)

    split_summary = []
    for split, part_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        eye_df = part_df.drop_duplicates("combo_key")
        split_summary.append(
            {
                "split": split,
                "participants": int(eye_df["participant_id"].nunique()),
                "eyes": int(eye_df["combo_key"].nunique()),
                "positive_eyes": int(eye_df["closure_label"].sum()),
                "negative_eyes": int((eye_df["closure_label"] == 0).sum()),
                "images": int(len(part_df)),
            }
        )
    pd.DataFrame(split_summary).to_csv(split_dir / "fixedsplit_summary.csv", index=False)

    scaler_y = StandardScaler().fit(train_df[target_cols].to_numpy(dtype=np.float32))
    train_ds = ImageAnatomyDataset(train_df, target_cols, scaler_y, train_tfm)
    val_ds = ImageAnatomyDataset(val_df, target_cols, scaler_y, eval_tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ImageRegressor(
        len(target_cols),
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_state = None
    best_loss = float("inf")
    wait = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, amp_scaler, device, True, args.amp)
        val_loss, _, _ = run_epoch(model, val_loader, None, None, device, False, args.amp)
        print(f"[fixed] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "targets": target_cols, "args": vars(args)}, split_dir / f"{args.backbone}_anatomy.pt")
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    image_frames = []
    for split, source_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        loader = DataLoader(
            ImageAnatomyDataset(source_df, target_cols, scaler_y, eval_tfm),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        _, image_ids, pred_scaled = run_epoch(model, loader, None, None, device, False, args.amp)
        pred_raw = scaler_y.inverse_transform(pred_scaled)
        pred_df = source_df.set_index("image_id").loc[image_ids].reset_index()
        for i, col in enumerate(target_cols):
            pred_df[f"pred_{col}"] = pred_raw[:, i]
        pred_df["fold"] = 1
        pred_df["split"] = split
        keep_cols = [
            "fold",
            "split",
            "image_id",
            "participant_id",
            "eye_code",
            "combo_key",
            "angle_grade",
            "closure_label",
            "view_label",
            "image_path",
        ] + [f"pred_{c}" for c in target_cols]
        pred_df = pred_df[keep_cols]
        pred_df.to_csv(split_dir / f"{split}_image_predictions.csv", index=False)
        image_frames.append(pred_df)

    image_preds = pd.concat(image_frames, ignore_index=True)
    eye_preds = aggregate_eye_predictions(image_preds, target_cols)
    train_eye = eye_preds[eye_preds["split"] == "train"].copy()
    val_eye = eye_preds[eye_preds["split"] == "val"].copy()
    test_eye = eye_preds[eye_preds["split"] == "test"].copy()
    clf = fit_stack_model(train_eye, target_cols, args.seed)
    scored = {
        "train": add_stack_scores(train_eye, clf, target_cols),
        "val": add_stack_scores(val_eye, clf, target_cols),
        "test": add_stack_scores(test_eye, clf, target_cols),
    }

    val_y = scored["val"]["closure_label"].astype(int)
    val_score = scored["val"]["stack_score"]
    train_y = scored["train"]["closure_label"].astype(int)
    train_score = scored["train"]["stack_score"]
    thresholds = {
        "balanced_from_train": choose_balanced_threshold(train_y, train_score),
        "balanced_from_val": choose_balanced_threshold(val_y, val_score),
        "youden_from_val": choose_youden_threshold(val_y, val_score),
        "sens80_from_val": choose_threshold_at_sensitivity(val_y, val_score, 0.80),
        "spec80_from_val": choose_threshold_at_specificity(val_y, val_score, 0.80),
    }

    metric_rows = []
    all_eye_frames = []
    for split, split_eye in scored.items():
        out = split_eye.copy()
        for threshold_rule, threshold in thresholds.items():
            out[f"stack_pred_{threshold_rule}"] = (out["stack_score"] >= threshold).astype(int)
            row = threshold_metrics(out["closure_label"], out["stack_score"], threshold)
            row.update(
                {
                    "split": split,
                    "threshold_rule": threshold_rule,
                    "train_eyes": int(train_eye["combo_key"].nunique()),
                    "val_eyes": int(val_eye["combo_key"].nunique()),
                    "test_eyes": int(test_eye["combo_key"].nunique()),
                    "val_positive_eyes": int(val_eye["closure_label"].sum()),
                    "test_positive_eyes": int(test_eye["closure_label"].sum()),
                }
            )
            metric_rows.append(row)
        all_eye_frames.append(out)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.outdir / "fixedsplit_metrics.csv", index=False)
    metrics.to_csv(split_dir / "fixedsplit_metrics.csv", index=False)
    image_preds.to_csv(args.outdir / "fixedsplit_image_predictions.csv", index=False)
    all_eye = pd.concat(all_eye_frames, ignore_index=True)
    all_eye.to_csv(args.outdir / "fixedsplit_eye_stack_predictions.csv", index=False)
    all_eye.to_csv(split_dir / "fixedsplit_eye_stack_predictions.csv", index=False)

    selectable = metrics[(metrics["split"] == "val") & (metrics["threshold_rule"] != "balanced_from_train")].copy()
    selectable = selectable.sort_values(["balanced_min", "sensitivity", "specificity", "auroc"], ascending=False)
    best_rule = str(selectable.iloc[0]["threshold_rule"])
    best_threshold = float(selectable.iloc[0]["threshold"])
    best = metrics[metrics["threshold_rule"].eq(best_rule)].copy()
    best.to_csv(args.outdir / "fixedsplit_best_by_validation.csv", index=False)
    best.to_csv(split_dir / "fixedsplit_best_by_validation.csv", index=False)
    best_test = scored["test"].copy()
    best_test["stack_pred"] = (best_test["stack_score"] >= best_threshold).astype(int)
    best_test["threshold_rule"] = best_rule
    best_test["threshold"] = best_threshold
    best_test.to_csv(args.outdir / "fixedsplit_test_predictions.csv", index=False)
    best_test.to_csv(split_dir / "fixedsplit_test_predictions.csv", index=False)
    ci = bootstrap_ci(best_test, best_threshold, args.bootstrap, args.seed)
    if not ci.empty:
        ci.to_csv(args.outdir / "fixedsplit_best_test_bootstrap_ci.csv", index=False)
        ci.to_csv(split_dir / "fixedsplit_best_test_bootstrap_ci.csv", index=False)

    with open(args.outdir / "RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# Anatomy Stack Fixed-Split Results\n\n")
        f.write(f"Regression-only `{args.backbone}` predicts 10 AS-OCT anatomical parameters. ")
        f.write("A balanced logistic regression then classifies strict angle closure from predicted anatomy. ")
        f.write("The stacker is fit on train eyes; threshold selection is based on validation eyes only.\n\n")
        f.write("Split summary:\n\n")
        f.write(markdown_table(pd.DataFrame(split_summary)))
        f.write("\n\nValidation-selected threshold rule:\n\n")
        f.write(markdown_table(best[["split", "threshold_rule", "auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "tp", "fp", "tn", "fn"]]))
        if not ci.empty:
            f.write("\n\nPatient-level bootstrap CI for selected locked-test row:\n\n")
            f.write(markdown_table(ci))
        f.write("\n")
    print(f"[DONE] Fixed-split outputs written to {args.outdir}")


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    df = load_dataset(args)
    write_label_audit(df, args)
    if args.prepare_only:
        print(f"[PREPARE] Wrote label audit to {args.outdir}")
        return

    target_cols = target_columns_for_preset(args.target_preset)
    eye_labels = (
        df.drop_duplicates("combo_key")
        .groupby("participant_id")["closure_label"]
        .max()
        .reset_index()
        .sort_values("participant_id")
    )
    train_tfm, eval_tfm = make_transforms(args.img_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.split_manifest is not None:
        run_fixed_split(args, df, target_cols, train_tfm, eval_tfm, device)
        return

    all_image_preds = []
    all_eye_preds = []
    metric_rows = []

    for fold, train_ids, val_ids in iter_splits(eye_labels, args):
        fold_dir = args.outdir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df = df[df["participant_id"].isin(train_ids)].copy()
        val_df = df[df["participant_id"].isin(val_ids)].copy()
        train_df = cap_images_per_eye(train_df, args.max_train_images_per_eye, args.seed + fold)
        val_df = cap_images_per_eye(val_df, args.max_val_images_per_eye, args.seed + fold)

        scaler_y = StandardScaler().fit(train_df[target_cols].to_numpy(dtype=np.float32))
        train_ds = ImageAnatomyDataset(train_df, target_cols, scaler_y, train_tfm)
        val_ds = ImageAnatomyDataset(val_df, target_cols, scaler_y, eval_tfm)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        train_pred_loader = DataLoader(ImageAnatomyDataset(train_df, target_cols, scaler_y, eval_tfm), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = ImageRegressor(
            len(target_cols),
            backbone=args.backbone,
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbone,
        ).to(device)
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        amp_scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
        best_state = None
        best_loss = float("inf")
        wait = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = run_epoch(model, train_loader, optimizer, amp_scaler, device, True, args.amp)
            val_loss, _, _ = run_epoch(model, val_loader, None, None, device, False, args.amp)
            print(f"[fold {fold}] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save({"model": best_state, "targets": target_cols, "args": vars(args)}, fold_dir / f"{args.backbone}_anatomy.pt")
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        fold_pred_frames = []
        for split, source_df, loader in [("train", train_df, train_pred_loader), ("val", val_df, val_loader)]:
            _, image_ids, pred_scaled = run_epoch(model, loader, None, None, device, False, args.amp)
            pred_raw = scaler_y.inverse_transform(pred_scaled)
            pred_df = source_df.set_index("image_id").loc[image_ids].reset_index()
            for i, col in enumerate(target_cols):
                pred_df[f"pred_{col}"] = pred_raw[:, i]
            pred_df["fold"] = fold
            pred_df["split"] = split
            keep_cols = [
                "fold",
                "split",
                "image_id",
                "participant_id",
                "eye_code",
                "combo_key",
                "angle_grade",
                "closure_label",
                "view_label",
                "image_path",
            ] + [f"pred_{c}" for c in target_cols]
            pred_df = pred_df[keep_cols]
            pred_df.to_csv(fold_dir / f"{split}_image_predictions.csv", index=False)
            fold_pred_frames.append(pred_df)
            all_image_preds.append(pred_df)

        fold_images = pd.concat(fold_pred_frames, ignore_index=True)
        fold_eye = aggregate_eye_predictions(fold_images, target_cols)
        train_eye = fold_eye[fold_eye["split"] == "train"].copy()
        val_eye = fold_eye[fold_eye["split"] == "val"].copy()
        val_scored, metrics = logistic_stack(train_eye, val_eye, target_cols, args.seed + fold)
        val_balanced_threshold = choose_balanced_threshold(val_scored["closure_label"], val_scored["stack_score"])
        val_scored["stack_pred_val_balanced"] = (val_scored["stack_score"] >= val_balanced_threshold).astype(int)
        val_metrics = threshold_metrics(val_scored["closure_label"], val_scored["stack_score"], val_balanced_threshold)
        val_metrics["threshold_rule"] = "balanced_min_from_val_internal"
        val_metrics["reached_70_70"] = bool(val_metrics["sensitivity"] >= 0.70 and val_metrics["specificity"] >= 0.70)
        false_rows = val_scored[val_scored["closure_label"] != val_scored["stack_pred"]].copy()
        false_rows.to_csv(fold_dir / "shortfall_participants.csv", index=False)
        common = {
                "fold": fold,
                "train_eyes": int(train_eye["combo_key"].nunique()),
                "val_eyes": int(val_eye["combo_key"].nunique()),
                "val_positive_eyes": int(val_eye["closure_label"].sum()),
                "excluded_angle_grades": args.exclude_angle_grades,
                "false_participants": ",".join(map(str, sorted(false_rows["participant_id"].unique()))),
                "false_angle_grades": ",".join(map(str, sorted(false_rows["angle_grade"].dropna().unique()))),
        }
        metrics.update(common)
        val_metrics.update(common)
        metric_rows.append(metrics)
        metric_rows.append(val_metrics)
        val_scored.to_csv(fold_dir / "val_eye_stack_predictions.csv", index=False)
        all_eye_preds.append(val_scored)

    image_preds = pd.concat(all_image_preds, ignore_index=True)
    eye_preds = pd.concat(all_eye_preds, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    image_preds.to_csv(args.outdir / "all_fold_image_predictions.csv", index=False)
    eye_preds.to_csv(args.outdir / "all_fold_val_eye_stack_predictions.csv", index=False)
    metrics.to_csv(args.outdir / "fold_metrics.csv", index=False)
    summary = metrics[["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_min"]].agg(["mean", "std", "min", "max"]).T
    summary.to_csv(args.outdir / "metric_summary.csv")
    with open(args.outdir / "RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# Anatomy Stack Results\n\n")
        f.write("Fold-level logistic regression results using predicted anatomy:\n\n")
        f.write(markdown_table(metrics))
        f.write("\n\nSummary:\n\n")
        f.write(markdown_table(summary.reset_index(names="metric")))
        f.write("\n")
    print(f"[DONE] Outputs written to {args.outdir}")


if __name__ == "__main__":
    run(parse_args())
