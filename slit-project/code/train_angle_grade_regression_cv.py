#!/usr/bin/env python3
"""
Image-to-Shaffer-grade regression baseline with 5-fold Cross-Validation.

The model predicts eye-level gonioscopic Shaffer grade (0..4) from slit-lamp
images using a regression head. Predictions are averaged per eye and thresholded
as closed if predicted grade <= --grade-threshold, e.g. 1.5.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import average_precision_score, confusion_matrix, mean_absolute_error, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
USABLE_VIEWS = {"center", "van_nasal", "van_temporal"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ANATOMY_REQUIRED_COLS = [
    "CCT",
    "ACD[Endo.]",
    "LV",
    "ACW",
    "AOD500_temporal",
    "AOD500_nasal",
    "TISA500_temporal",
    "TISA500_nasal",
    "TIA500_temporal",
    "TIA500_nasal",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train image-to-Shaffer-grade regression and threshold predicted grade.")
    p.add_argument("--image-csv", type=Path, default=PROJECT_ROOT / "code" / "ready_for_training_clustered_anatomical_with_means_with_views_anonymized.csv")
    p.add_argument("--clinical-csv", type=Path, default=PROJECT_ROOT / "code" / "ready_for_upload_publish.csv")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "paper2_runs" / "angle_grade_regression_cv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--exclude-grade-2", action="store_true", help="Exclude all eyes with Shaffer grade 2.")
    p.add_argument("--require-anatomy-complete", action="store_true", help="Restrict to rows with all 10 anatomy targets, matching the 476-eye anatomy cohort.")
    p.add_argument("--grade-threshold", type=float, default=1.5)
    p.add_argument("--view-mode", choices=["usable", "center", "van_nasal", "van_temporal", "all"], default="usable")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-train-images-per-eye", type=int, default=12)
    p.add_argument("--max-val-images-per-eye", type=int, default=0)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--weighted-grade-loss", action="store_true", help="Use inverse-frequency sample weights by Shaffer grade.")
    p.add_argument("--normalize-grade-target", action="store_true", help="Train on grade/4 and convert predictions back to 0..4.")
    p.add_argument("--loss", choices=["huber", "mse"], default="huber")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--amp", action="store_true")
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


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    clinical = pd.read_csv(args.clinical_csv, low_memory=False)
    image = pd.read_csv(args.image_csv, low_memory=False)

    clin = pd.DataFrame()
    clin["participant_id"] = clean_numeric(clinical["subject_id"])
    clin["eye_code"] = clinical["eye"].apply(eye_code)
    clin["angle_grade"] = clean_numeric(clinical["angle_grade"])
    clin = clin.dropna(subset=["participant_id", "eye_code", "angle_grade"]).copy()
    clin = clin[clin["angle_grade"].isin([0, 1, 2, 3, 4])].copy()
    clin["participant_id"] = clin["participant_id"].astype(int)
    clin["combo_key"] = clin.apply(lambda r: combo_key(r["participant_id"], r["eye_code"]), axis=1)
    clin["closure_label"] = (clin["angle_grade"] <= 1).astype(int)
    
    if args.exclude_grade_2:
        clin = clin[clin["angle_grade"] != 2].copy()
        
    clin = clin[["participant_id", "eye_code", "combo_key", "angle_grade", "closure_label"]].drop_duplicates("combo_key")

    required = {"Patient_Num", "eye_clean", "Image_Path", "View_Label"}
    if args.require_anatomy_complete:
        required |= set(ANATOMY_REQUIRED_COLS)
    missing = required - set(image.columns)
    if missing:
        raise SystemExit(f"Image CSV missing required column(s): {sorted(missing)}")
    img = pd.DataFrame()
    img["participant_id"] = clean_numeric(image["Patient_Num"])
    img["eye_code"] = image["eye_clean"].apply(eye_code)
    img["view_label"] = image["View_Label"].astype(str).str.strip().str.lower()
    img["image_path"] = image["Image_Path"].apply(resolve_image_path)
    if args.require_anatomy_complete:
        for col in ANATOMY_REQUIRED_COLS:
            img[col] = clean_numeric(image[col])
    img = img.dropna(subset=["participant_id", "eye_code", "image_path"]).copy()
    if args.require_anatomy_complete:
        img = img.dropna(subset=ANATOMY_REQUIRED_COLS).copy()
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


def cap_images_per_eye(df: pd.DataFrame, max_images: int, seed: int) -> pd.DataFrame:
    if not max_images or max_images <= 0:
        return df
    return (
        df.groupby("combo_key", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), max_images), random_state=seed))
        .reset_index(drop=True)
    )


class GradeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tfm, normalize_grade_target: bool):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
        self.normalize_grade_target = normalize_grade_target

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        y = np.float32(row["angle_grade"] / 4.0 if self.normalize_grade_target else row["angle_grade"])
        w = np.float32(row.get("sample_weight", 1.0))
        return self.tfm(img), torch.tensor(y), torch.tensor(w), int(row["image_id"])


class ResNet50GradeRegressor(nn.Module):
    def __init__(self, pretrained: bool, freeze_backbone: bool):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=weights)
        hidden = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def make_transforms(img_size: int):
    train = transforms.Compose(
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
    return train, eval_tfm


def run_epoch(model, loader, optimizer, scaler, device, train: bool, amp: bool, loss_name: str):
    model.train(train)
    loss_fn = nn.MSELoss(reduction="none") if loss_name == "mse" else nn.SmoothL1Loss(reduction="none")
    losses, preds, ids = [], [], []
    for x, y, weights, image_ids in loader:
        x = x.to(device)
        y = y.to(device)
        weights = weights.to(device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                pred = model(x)
                loss_items = loss_fn(pred, y)
                loss = (loss_items * weights).sum() / weights.sum().clamp(min=1e-6)
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
            preds.extend(pred.detach().cpu().numpy().tolist())
            ids.extend(image_ids.numpy().tolist())
    return float(np.mean(losses)) if losses else np.nan, ids, np.array(preds, dtype=float)


def classification_metrics(y_true, pred_grade, grade_threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred_grade = np.asarray(pred_grade).astype(float)
    score = -pred_grade
    pred = (pred_grade <= grade_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {
        "threshold_pred_grade_le": float(grade_threshold),
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


def choose_balanced_grade_threshold(y_true, pred_grade) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = -np.asarray(pred_grade).astype(float)
    fpr, tpr, risk_thresholds = roc_curve(y_true, score)
    finite = np.isfinite(risk_thresholds)
    fpr, tpr, risk_thresholds = fpr[finite], tpr[finite], risk_thresholds[finite]
    spec = 1.0 - fpr
    risk_thr = risk_thresholds[np.argmax(np.minimum(tpr, spec))]
    return float(-risk_thr)


def add_fold_metrics(rows: list[dict], df: pd.DataFrame, fold: int, threshold: float, threshold_rule: str) -> None:
    m = classification_metrics(df["closure_label"], df["pred_angle_grade"], threshold)
    m.update(
        {
            "fold": fold,
            "split": "val",
            "threshold_rule": threshold_rule,
            "val_eyes": int(df["combo_key"].nunique()),
            "val_positive_eyes": int(df["closure_label"].sum()),
        }
    )
    rows.append(m)


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
    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args)
    eye = df.drop_duplicates("combo_key").copy().reset_index(drop=True)
    
    sgkf = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_splits = list(sgkf.split(eye, eye["closure_label"], eye["participant_id"]))
    
    grade_counts = eye["angle_grade"].value_counts().sort_index().rename_axis("angle_grade").reset_index(name="eyes")
    grade_counts.to_csv(args.outdir / "label_grade_counts.csv", index=False)
    with open(args.outdir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "analytic_eyes": int(eye["combo_key"].nunique()),
                "participants": int(eye["participant_id"].nunique()),
                "positive_eyes": int(eye["closure_label"].sum()),
                "negative_eyes": int((eye["closure_label"] == 0).sum()),
            },
            f,
            indent=2,
            default=str,
        )
    if args.prepare_only:
        print(f"[PREPARE] Wrote label audit to {args.outdir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tfm, eval_tfm = make_transforms(args.img_size)
    
    all_val_preds = []
    metric_rows = []

    for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
        print(f"\\n=== Starting Fold {fold}/{args.cv_folds} ===")
        train_eye_combos = set(eye.iloc[train_idx]["combo_key"])
        val_eye_combos = set(eye.iloc[val_idx]["combo_key"])
        
        train_df = df[df["combo_key"].isin(train_eye_combos)].copy()
        val_df = df[df["combo_key"].isin(val_eye_combos)].copy()
        
        train_df = cap_images_per_eye(train_df, args.max_train_images_per_eye, args.seed)
        val_df = cap_images_per_eye(val_df, args.max_val_images_per_eye, args.seed)
        
        train_df["sample_weight"] = 1.0
        val_df["sample_weight"] = 1.0
        if args.weighted_grade_loss:
            grade_freq = train_df.drop_duplicates("combo_key")["angle_grade"].value_counts().to_dict()
            grade_weights = {grade: 1.0 / max(count, 1) for grade, count in grade_freq.items()}
            mean_weight = np.mean(list(grade_weights.values()))
            grade_weights = {grade: weight / mean_weight for grade, weight in grade_weights.items()}
            train_df["sample_weight"] = train_df["angle_grade"].map(grade_weights).fillna(1.0).astype(float)
            
        train_loader = DataLoader(GradeDataset(train_df, train_tfm, args.normalize_grade_target), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        train_eval_loader = DataLoader(GradeDataset(train_df, eval_tfm, args.normalize_grade_target), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        val_loader = DataLoader(GradeDataset(val_df, eval_tfm, args.normalize_grade_target), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = ResNet50GradeRegressor(pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone).to(device)
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        amp_scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
        
        best_loss, best_state, wait = float("inf"), None, 0
        for epoch in range(1, args.epochs + 1):
            tr_loss, _, _ = run_epoch(model, train_loader, optimizer, amp_scaler, device, True, args.amp, args.loss)
            va_loss, _, _ = run_epoch(model, val_loader, None, None, device, False, args.amp, args.loss)
            print(f"Fold {fold} - Epoch {epoch}/{args.epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")
            if va_loss < best_loss:
                best_loss = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break
                    
        if best_state is not None:
            model.load_state_dict(best_state)
            
        fold_eye_preds = {}
        for split, source_df, loader in [("train", train_df, train_eval_loader), ("val", val_df, val_loader)]:
            _, image_ids, pred_grade = run_epoch(model, loader, None, None, device, False, args.amp, args.loss)
            if args.normalize_grade_target:
                pred_grade = pred_grade * 4.0
            pred_df = source_df.set_index("image_id").loc[image_ids].reset_index()
            pred_df["pred_angle_grade"] = np.clip(pred_grade, 0, 4)
            pred_df["fold"] = fold
            pred_df["split"] = split
            keep = ["fold", "split", "image_id", "participant_id", "eye_code", "combo_key", "angle_grade", "closure_label", "view_label", "image_path", "pred_angle_grade"]
            eye_pred = (
                pred_df[keep]
                .groupby(["fold", "split", "participant_id", "eye_code", "combo_key", "angle_grade", "closure_label"], as_index=False)
                .agg(pred_angle_grade=("pred_angle_grade", "mean"), n_images=("image_id", "count"))
            )
            fold_eye_preds[split] = eye_pred
            if split == "val":
                all_val_preds.append(pred_df[keep])

        train_eye = fold_eye_preds["train"]
        val_eye = fold_eye_preds["val"]
        train_balanced_thr = choose_balanced_grade_threshold(train_eye["closure_label"], train_eye["pred_angle_grade"])
        val_balanced_thr = choose_balanced_grade_threshold(val_eye["closure_label"], val_eye["pred_angle_grade"])
        add_fold_metrics(metric_rows, val_eye, fold, args.grade_threshold, f"fixed_pred_grade_le_{args.grade_threshold}")
        add_fold_metrics(metric_rows, val_eye, fold, train_balanced_thr, "balanced_from_train")
        add_fold_metrics(metric_rows, val_eye, fold, val_balanced_thr, "balanced_on_val_internal")

    image_preds = pd.concat(all_val_preds, ignore_index=True)
    eye_preds = (
        image_preds.groupby(["fold", "split", "participant_id", "eye_code", "combo_key", "angle_grade", "closure_label"], as_index=False)
        .agg(pred_angle_grade=("pred_angle_grade", "mean"), n_images=("image_id", "count"))
    )
    eye_preds["pred_closed_at_1p5"] = (eye_preds["pred_angle_grade"] <= args.grade_threshold).astype(int)
    eye_preds.to_csv(args.outdir / "cv_eye_predictions.csv", index=False)

    rows = []
    # Overall metrics across all folds
    m = classification_metrics(eye_preds["closure_label"], eye_preds["pred_angle_grade"], args.grade_threshold)
    m.update({"split": "cv_all", "threshold_rule": f"fixed_pred_grade_le_{args.grade_threshold}"})
    rows.append(m)
    
    balanced_thr = choose_balanced_grade_threshold(eye_preds["closure_label"], eye_preds["pred_angle_grade"])
    mb = classification_metrics(eye_preds["closure_label"], eye_preds["pred_angle_grade"], balanced_thr)
    mb.update({"split": "cv_all", "threshold_rule": "balanced_on_cv_overall"})
    rows.append(mb)
        
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.outdir / "classification_metrics.csv", index=False)
    fold_metrics = pd.DataFrame(metric_rows)
    fold_metrics.to_csv(args.outdir / "fold_metrics.csv", index=False)
    summary = (
        fold_metrics.groupby("threshold_rule")[["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_min"]]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    summary.reset_index().to_csv(args.outdir / "metric_summary_by_threshold.csv", index=False)

    eye_preds["abs_grade_error"] = (eye_preds["pred_angle_grade"] - eye_preds["angle_grade"]).abs()
    regression = pd.DataFrame(
        [
            {
                "split": "cv_all",
                "mae_grade": mean_absolute_error(eye_preds["angle_grade"], eye_preds["pred_angle_grade"]),
                "pearson_r": pearsonr(eye_preds["angle_grade"], eye_preds["pred_angle_grade"])[0],
            }
        ]
    )
    regression.to_csv(args.outdir / "grade_regression_metrics.csv", index=False)
    
    errors = eye_preds[eye_preds["closure_label"] != eye_preds["pred_closed_at_1p5"]].copy()
    errors.to_csv(args.outdir / "cv_threshold_1p5_errors.csv", index=False)
    
    with open(args.outdir / "RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# Angle Grade Regression CV Results\n\n")
        f.write("Model predicts Shaffer grade `0..4`; closed if predicted grade <= threshold.\n\n")
        f.write("## Classification (All Folds Combined)\n\n")
        f.write(markdown_table(metrics))
        f.write("\n\n## Fold Summary by Threshold Rule\n\n")
        f.write(markdown_table(summary.reset_index()))
        f.write("\n\n## Grade Regression (All Folds Combined)\n\n")
        f.write(markdown_table(regression))
        f.write("\n")
    print(f"[DONE] Outputs written to {args.outdir}")


if __name__ == "__main__":
    run(parse_args())
