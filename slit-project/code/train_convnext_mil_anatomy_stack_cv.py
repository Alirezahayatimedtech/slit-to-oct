#!/usr/bin/env python3
"""
ConvNeXt-Tiny attention-MIL anatomy stack for angle-closure screening.

Each eye is treated as one bag of slit-lamp images. A ConvNeXt-Tiny encoder
extracts per-image embeddings, an attention module learns image weights within
the eye, and a regression head predicts 10 AS-OCT anatomy targets. A shallow
logistic regression then classifies angle closure from the predicted anatomy.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models

import train_resnet50_anatomy_stack_cv as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLS = [dst for _, dst in base.ANATOMY_TARGETS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ConvNeXt-Tiny attention-MIL anatomy stack.")
    p.add_argument("--image-csv", type=Path, default=PROJECT_ROOT / "code" / "ready_for_training_clustered_anatomical_with_means_with_views_anonymized.csv")
    p.add_argument("--clinical-csv", type=Path, default=PROJECT_ROOT / "code" / "ready_for_upload_publish.csv")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "paper2_runs" / "convnext_tiny_mil_anatomy_stack_cv")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.set_defaults(backbone="convnext_tiny_attention_mil")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--view-mode", choices=["usable", "center", "van_nasal", "van_temporal", "all"], default="usable")
    p.add_argument("--closure-grade-max", type=float, default=1.0)
    p.add_argument("--exclude-angle-grades", type=str, default="")
    p.add_argument("--max-train-images-per-eye", type=int, default=12)
    p.add_argument("--max-val-images-per-eye", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eye_level_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby("combo_key", sort=False):
        first = g.iloc[0]
        row = {
            "participant_id": int(first["participant_id"]),
            "eye_code": first["eye_code"],
            "combo_key": key,
            "angle_grade": float(first["angle_grade"]),
            "closure_label": int(first["closure_label"]),
            "image_paths": g["image_path"].tolist(),
            "view_labels": g["view_label"].tolist(),
            "n_images_available": int(len(g)),
        }
        for col in TARGET_COLS:
            row[col] = float(g[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


class EyeBagDataset(Dataset):
    def __init__(
        self,
        eye_df: pd.DataFrame,
        target_cols: list[str],
        scaler_y: StandardScaler,
        tfm,
        max_images: int,
        train: bool,
        seed: int,
    ):
        self.eye_df = eye_df.reset_index(drop=True)
        self.target_cols = target_cols
        self.scaler_y = scaler_y
        self.tfm = tfm
        self.max_images = max_images
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.eye_df)

    def _select_indices(self, n: int) -> np.ndarray:
        if self.max_images and self.max_images > 0 and n > self.max_images:
            if self.train:
                return self.rng.choice(n, size=self.max_images, replace=False)
            return np.arange(self.max_images)
        return np.arange(n)

    def __getitem__(self, idx: int) -> dict:
        row = self.eye_df.iloc[idx]
        paths = list(row["image_paths"])
        views = list(row["view_labels"])
        chosen = self._select_indices(len(paths))
        imgs = []
        chosen_views = []
        for i in chosen:
            try:
                img = Image.open(paths[int(i)]).convert("RGB")
            except Exception:
                img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            imgs.append(self.tfm(img))
            chosen_views.append(str(views[int(i)]))
        y = row[self.target_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        y = self.scaler_y.transform(y).astype(np.float32).squeeze(0)
        return {
            "x": torch.stack(imgs, dim=0),
            "y": torch.tensor(y, dtype=torch.float32),
            "participant_id": int(row["participant_id"]),
            "eye_code": row["eye_code"],
            "combo_key": row["combo_key"],
            "angle_grade": float(row["angle_grade"]),
            "closure_label": int(row["closure_label"]),
            "view_labels": chosen_views,
            "n_images": len(chosen_views),
        }


def collate_bags(batch: list[dict]) -> dict:
    max_v = max(item["x"].shape[0] for item in batch)
    xs = []
    mask = []
    for item in batch:
        x = item["x"]
        valid = torch.ones(x.shape[0], dtype=torch.bool)
        if x.shape[0] < max_v:
            pad = torch.zeros((max_v - x.shape[0], *x.shape[1:]), dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
            valid = torch.cat([valid, torch.zeros(max_v - valid.shape[0], dtype=torch.bool)], dim=0)
        xs.append(x)
        mask.append(valid)
    return {
        "x": torch.stack(xs, dim=0),
        "view_mask": torch.stack(mask, dim=0),
        "y": torch.stack([item["y"] for item in batch]),
        "participant_id": [item["participant_id"] for item in batch],
        "eye_code": [item["eye_code"] for item in batch],
        "combo_key": [item["combo_key"] for item in batch],
        "angle_grade": [item["angle_grade"] for item in batch],
        "closure_label": [item["closure_label"] for item in batch],
        "view_labels": [item["view_labels"] for item in batch],
        "n_images": [item["n_images"] for item in batch],
    }


class ConvNeXtAttentionMIL(nn.Module):
    def __init__(self, out_dim: int, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        feature_dim = backbone.classifier[2].in_features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
        self.attn = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
        self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(0.25), nn.Linear(feature_dim, out_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = x.shape
        flat = x.reshape(b * v, c, h, w)
        feat = self.features(flat)
        feat = self.avgpool(feat).flatten(1)
        return feat.reshape(b, v, -1)

    def forward(self, x: torch.Tensor, view_mask: torch.Tensor):
        feats = self.encode(x)
        scores = self.attn(feats).squeeze(-1)
        scores = scores.masked_fill(~view_mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = (feats * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled), weights


def run_epoch(model, loader, optimizer, scaler, device, train: bool, amp: bool):
    model.train(train)
    loss_fn = nn.SmoothL1Loss()
    losses = []
    rows = []
    preds = []
    attn_strings = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        view_mask = batch["view_mask"].to(device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                pred, weights = model(x, view_mask)
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
            weights_np = weights.detach().cpu().numpy()
            for i, combo_key in enumerate(batch["combo_key"]):
                n_images = int(batch["n_images"][i])
                rows.append(
                    {
                        "participant_id": batch["participant_id"][i],
                        "eye_code": batch["eye_code"][i],
                        "combo_key": combo_key,
                        "angle_grade": batch["angle_grade"][i],
                        "closure_label": batch["closure_label"][i],
                        "n_images": n_images,
                        "view_labels": "|".join(batch["view_labels"][i]),
                    }
                )
                attn_strings.append("|".join(f"{w:.4f}" for w in weights_np[i, :n_images]))
    pred_arr = np.concatenate(preds, axis=0) if preds else np.empty((0, 0))
    rows_df = pd.DataFrame(rows)
    if not rows_df.empty:
        rows_df["attention_weights"] = attn_strings
    return float(np.mean(losses)) if losses else np.nan, rows_df, pred_arr


def make_prediction_frame(rows: pd.DataFrame, pred_scaled: np.ndarray, scaler_y: StandardScaler, split: str, fold: int) -> pd.DataFrame:
    pred_raw = scaler_y.inverse_transform(pred_scaled)
    out = rows.copy()
    for i, col in enumerate(TARGET_COLS):
        out[f"pred_{col}"] = pred_raw[:, i]
    out["split"] = split
    out["fold"] = fold
    return out


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = base.load_dataset(args)
    base.write_label_audit(df, args)
    eye_df = eye_level_table(df)
    eye_df.to_csv(args.outdir / "eye_bag_manifest.csv", index=False)
    if args.prepare_only:
        print(f"[PREPARE] Wrote eye bag manifest to {args.outdir}")
        return

    participant_labels = (
        eye_df.groupby("participant_id")["closure_label"]
        .max()
        .reset_index()
        .sort_values("participant_id")
    )
    train_tfm, eval_tfm = base.make_transforms(args.img_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_rows = []
    pred_frames = []

    for fold, train_ids, val_ids in base.iter_splits(participant_labels, args):
        fold_dir = args.outdir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_eye = eye_df[eye_df["participant_id"].isin(train_ids)].copy()
        val_eye = eye_df[eye_df["participant_id"].isin(val_ids)].copy()
        scaler_y = StandardScaler().fit(train_eye[TARGET_COLS].to_numpy(dtype=np.float32))
        train_ds = EyeBagDataset(train_eye, TARGET_COLS, scaler_y, train_tfm, args.max_train_images_per_eye, True, args.seed + fold)
        val_ds = EyeBagDataset(val_eye, TARGET_COLS, scaler_y, eval_tfm, args.max_val_images_per_eye, False, args.seed + fold)
        train_eval_ds = EyeBagDataset(train_eye, TARGET_COLS, scaler_y, eval_tfm, args.max_val_images_per_eye, False, args.seed + fold)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_bags)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_bags)
        train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_bags)

        model = ConvNeXtAttentionMIL(
            len(TARGET_COLS),
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
                torch.save({"model": best_state, "targets": TARGET_COLS, "args": vars(args)}, fold_dir / "convnext_tiny_attention_mil.pt")
                wait = 0
            else:
                wait += 1
                if wait >= args.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        fold_preds = []
        for split, loader in [("train", train_eval_loader), ("val", val_loader)]:
            _, rows, pred_scaled = run_epoch(model, loader, None, None, device, False, args.amp)
            pred_df = make_prediction_frame(rows, pred_scaled, scaler_y, split, fold)
            pred_df.to_csv(fold_dir / f"{split}_eye_predictions.csv", index=False)
            fold_preds.append(pred_df)
            pred_frames.append(pred_df)

        fold_eye = pd.concat(fold_preds, ignore_index=True)
        train_pred = fold_eye[fold_eye["split"] == "train"].copy()
        val_pred = fold_eye[fold_eye["split"] == "val"].copy()
        val_scored, metrics = base.logistic_stack(train_pred, val_pred, TARGET_COLS, args.seed + fold)
        val_threshold = base.choose_balanced_threshold(val_scored["closure_label"], val_scored["stack_score"])
        val_scored["stack_pred_val_balanced"] = (val_scored["stack_score"] >= val_threshold).astype(int)
        val_metrics = base.threshold_metrics(val_scored["closure_label"], val_scored["stack_score"], val_threshold)
        val_metrics["threshold_rule"] = "balanced_min_from_val_internal"
        val_metrics["reached_70_70"] = bool(val_metrics["sensitivity"] >= 0.70 and val_metrics["specificity"] >= 0.70)
        false_rows = val_scored[val_scored["closure_label"] != val_scored["stack_pred"]].copy()
        false_rows.to_csv(fold_dir / "shortfall_participants.csv", index=False)
        common = {
            "fold": fold,
            "train_eyes": int(train_pred["combo_key"].nunique()),
            "val_eyes": int(val_pred["combo_key"].nunique()),
            "val_positive_eyes": int(val_pred["closure_label"].sum()),
            "excluded_angle_grades": args.exclude_angle_grades,
            "false_participants": ",".join(map(str, sorted(false_rows["participant_id"].unique()))),
            "false_angle_grades": ",".join(map(str, sorted(false_rows["angle_grade"].dropna().unique()))),
        }
        metrics.update(common)
        val_metrics.update(common)
        metric_rows.append(metrics)
        metric_rows.append(val_metrics)
        val_scored.to_csv(fold_dir / "val_eye_stack_predictions.csv", index=False)

    all_preds = pd.concat(pred_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    all_preds.to_csv(args.outdir / "all_fold_eye_predictions.csv", index=False)
    metrics_df.to_csv(args.outdir / "fold_metrics.csv", index=False)
    summary = metrics_df[["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "accuracy", "balanced_min"]].agg(["mean", "std", "min", "max"]).T
    summary.to_csv(args.outdir / "metric_summary.csv")
    with open(args.outdir / "RESULTS.md", "w", encoding="utf-8") as f:
        f.write("# ConvNeXt-Tiny Attention-MIL Anatomy Stack Results\n\n")
        f.write("Eye-level bags of slit-lamp images are attention-pooled before predicting 10 AS-OCT anatomy targets. ")
        f.write("A logistic regression then classifies angle closure from predicted anatomy.\n\n")
        f.write(base.markdown_table(metrics_df))
        f.write("\n\nSummary:\n\n")
        f.write(base.markdown_table(summary.reset_index(names="metric")))
        f.write("\n")
    with open(args.outdir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"[DONE] Outputs written to {args.outdir}")


if __name__ == "__main__":
    run(parse_args())
