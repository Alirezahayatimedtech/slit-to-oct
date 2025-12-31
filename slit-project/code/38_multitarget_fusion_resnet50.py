"""
ACD regression with fusion of all views using ResNet-50.

- Groups images by patient/eye (filename prefix 'patient_eye_*').
- Uses ALL available views per combo (requires at least MIN_VIEWS; caps at MAX_VIEWS to save memory).
- Fusion: average transformed views into a single image tensor before the backbone (toggle logic externally if you add other fusion modes).
- Regression head outputs the target(s) in TARGET_COLS (currently ACD only).
- Patient-grouped split, target standardization, MSE loss, cosine LR, EMA, early stopping.
"""
# Best run: Val MSE 0.6486, Test MSE 0.5554, Test MAE(raw) 0.2587, Test r 0.6893 (ResNet-50 MIL attention, mixup 0.2).
# Best ACD config: MIL attention, LR 5e-4, wd 5e-3, patience 6, mixup 0.2, val/test 0.15/0.15, no erasing.

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torchvision import transforms as tv_transforms  # alias for functional TTA ops


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12)
    return float((x * y).sum() / denom)


def mixup_batch(imgs: torch.Tensor, targets: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return imgs, targets, None, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    mixed_imgs = lam * imgs + (1.0 - lam) * imgs[idx]
    targets_a, targets_b = targets, targets[idx]
    return mixed_imgs, targets_a, targets_b, lam


# --- CONFIGURATION --- (edit TARGET_COLS to switch targets; can override via CLI if desired)
SOURCE_CSV = "ready_for_training_clustered_anatomical_with_means.csv"
CROP_ROOT = Path("data/center_roi_images/processed_images_448")
# Single-target ACD regression; adjust list to add more targets later.
TARGET_COLS = ["ACD[Endo.]"]
MIN_VIEWS = 1
MAX_VIEWS = 15  # cap views to save memory
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 40
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 10
MIN_DELTA = 0.005
EMA_DECAY = 0.999
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

WINDOWS_PREFIX = "G:\\thesis-slit-oct-project\\data\\processed_images\\"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_FALLBACKS = [
    REPO_ROOT / "slit-oct" / "colab_ready_images",
    REPO_ROOT / "colab_ready_images",
]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def parse_args():
    p = argparse.ArgumentParser(description="ACD fusion training/eval (ResNet50).")
    p.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation.")
    p.add_argument("--method", type=str, choices=["fusion", "mil"], default="fusion", help="Training method: fusion (early/late) or MIL attention pooling.")
    p.add_argument("--fusion", type=str, choices=["early", "late"], default="early", help="Fusion mode (only used when --method fusion).")
    p.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search instead of full training.")
    p.add_argument("--tune-trials", type=int, default=10, help="Number of Optuna trials.")
    p.add_argument("--tune-epochs", type=int, default=5, help="Epochs per trial during tuning.")
    p.add_argument("--tune-patience", type=int, default=5, help="Early-stop patience (epochs) inside each Optuna trial.")
    p.add_argument("--tune-save", type=Path, default=Path("tune_best_params.json"), help="Where to save best tuning params (JSON).")
    default_ckpt = Path("resnet50_fusion_acd.pth")
    default_scaler = Path("scaler_fusion_acd.npz")
    p.add_argument("--checkpoint", type=Path, default=default_ckpt, help="Path to load/save model weights.")
    p.add_argument("--scaler-path", type=Path, default=default_scaler, help="Path to load/save target scaler.")
    p.add_argument(
        "--attention-dir",
        type=Path,
        default=None,
        help="If set, save Grad-CAM attention overlays (eval only, early fusion).",
    )
    p.add_argument(
        "--attention-limit",
        type=int,
        default=24,
        help="Maximum number of attention maps to save across splits.",
    )
    p.add_argument(
        "--test-scatter",
        type=Path,
        default=Path("fusion_test_scatter.png"),
        help="Save scatter plot of test preds vs true (set to empty string to disable).",
    )
    p.add_argument("--mixup-alpha", type=float, default=0.2, help="Mixup alpha; 0 to disable.")
    p.add_argument("--eta-min", type=float, default=1e-6, help="Min LR for cosine annealing.")
    p.add_argument("--freeze-epochs", type=int, default=0, help="Freeze backbone for N epochs before unfreezing.")
    p.add_argument("--unfreeze-lr-factor", type=float, default=1.0, help="Multiply LR by this when unfreezing.")
    p.add_argument("--tta", type=int, default=0, help="Number of TTA passes for MIL (0 disables).")
    p.add_argument("--max-views", type=int, default=MAX_VIEWS, help="Cap on number of views per combo.")
    p.add_argument("--min-views", type=int, default=MIN_VIEWS, help="Require at least this many views per combo; drop combos below.")
    p.add_argument("--views-per-bag", type=int, default=None, help="If set, use exactly this many views per combo (drop combos with fewer).")
    p.add_argument("--top-k", type=int, default=5, help="Top-K views for MIL pooling (ignored for fusion).")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training/eval.")
    p.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate for main training.")
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay for main training.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return p.parse_args()


def resolve_crop_path(p: str) -> str | None:
    fname = os.path.basename(str(p).replace("\\", "/"))
    crop_path = CROP_ROOT / fname
    if crop_path.exists():
        return str(crop_path)
    for root in LOCAL_FALLBACKS:
        candidate = root / fname
        if candidate.exists():
            return str(candidate)
    normalized = str(p).replace("\\", "/")
    if os.path.exists(normalized):
        return normalized
    return None


class MultiViewDataset(Dataset):
    def __init__(self, samples, target_cols, base_transform=None, max_views: int = MAX_VIEWS, fixed_views: int | None = None):
        self.samples = samples  # list of dicts with Paths, targets, combo_key
        self.target_cols = target_cols
        self.base_transform = base_transform
        self.max_views = max_views
        self.fixed_views = fixed_views

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        paths = item["Paths"]
        target_views = self.fixed_views if self.fixed_views is not None else self.max_views
        # randomly subsample views per eye to target_views cap
        if len(paths) > target_views:
            paths = random.sample(paths, target_views)
        y = np.array(item["targets"], dtype=np.float32)
        views = []
        for p in paths[:target_views]:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            if self.base_transform:
                views.append(self.base_transform(img))
            else:
                views.append(transforms.ToTensor()(img))
        # return stacked views [V,C,H,W]; fusion handled later
        x = torch.stack(views, dim=0)
        return x, torch.tensor(y, dtype=torch.float32), item["combo_key"]


def collate_views(batch):
    """Pad/truncate view dimension so a batch stacks cleanly."""
    views_list, targets_list, keys = zip(*batch)
    max_v = max(v.shape[0] for v in views_list)
    padded = []
    for v in views_list:
        v_trim = v[:max_v] if v.shape[0] > max_v else v
        if v_trim.shape[0] < max_v:
            pad = torch.zeros((max_v - v_trim.shape[0], *v_trim.shape[1:]), dtype=v_trim.dtype)
            v_trim = torch.cat([v_trim, pad], dim=0)
        padded.append(v_trim)
    inputs = torch.stack(padded, dim=0)
    targets = torch.stack(targets_list, dim=0)
    return inputs, targets, keys


class EarlyFusionResNet(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # up to avgpool
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(base.fc.in_features, out_dim)

    def forward(self, x):
        f = self.backbone(x)  # [B, C, 1, 1]
        f = f.view(f.size(0), -1)
        f = self.dropout(f)
        return self.head(f)


class MILResNet(nn.Module):
    """MIL with top-K selection and gated attention pooling."""

    def __init__(self, out_dim: int, top_k: int = 5):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # [B, 2048, 1, 1]
        hidden = base.fc.in_features
        self.top_k = top_k
        self.relevance = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 1))
        # Gated attention (Ilse et al.)
        self.attn_V = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 256), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 256), nn.Sigmoid())
        self.attn_W = nn.Linear(256, 1)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, mask):
        # x: [B, V, C, H, W], mask: [B, V] bool
        b, v, c, h, w = x.shape
        flat = x.view(b * v, c, h, w)
        feats = self.feature_extractor(flat).view(b, v, -1)  # [B, V, D]
        mask_f = mask.unsqueeze(-1)  # [B, V, 1]
        # top-k selection based on relevance
        rel_scores = self.relevance(feats).squeeze(-1)  # [B, V]
        rel_scores = rel_scores.masked_fill(~mask, -1e9)
        k = min(self.top_k, v)
        topk_scores, topk_idx = torch.topk(rel_scores, k=k, dim=1)
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, feats.size(-1))
        topk_feats = torch.gather(feats, 1, idx_exp)  # [B, K, D]
        # gated attention on top-k
        Vh = self.attn_V(topk_feats)
        Uh = self.attn_U(topk_feats)
        attn_scores = self.attn_W(Vh * Uh).squeeze(-1)  # [B, K]
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        bag = (topk_feats * attn_weights).sum(dim=1)  # [B, D]
        bag = bag  # already aggregated; mask not needed post top-k
        return self.head(bag)


def freeze_backbone(model):
    if hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False
    if hasattr(model, "feature_extractor"):
        for p in model.feature_extractor.parameters():
            p.requires_grad = False


def unfreeze_backbone(model):
    if hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = True
    if hasattr(model, "feature_extractor"):
        for p in model.feature_extractor.parameters():
            p.requires_grad = True


def load_and_prepare(min_views: int):
    df = pd.read_csv(SOURCE_CSV)
    for col in ["Image_Path"] + TARGET_COLS:
        if col not in df.columns:
            raise SystemExit(f"Missing column {col} in {SOURCE_CSV}")
    for col in TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Image_Path"] = df["Image_Path"].apply(resolve_crop_path)
    df = df.dropna(subset=["Image_Path"] + TARGET_COLS)
    df = df[df["Image_Path"].apply(os.path.exists)]
    if df.empty:
        raise SystemExit("No valid rows after path resolution.")
    df["combo_key"] = df["Image_Path"].apply(lambda p: "_".join(os.path.basename(str(p)).split("_")[:2]).upper())
    counts = df["combo_key"].value_counts()
    keep_keys = set(counts[counts >= min_views].index)
    df = df[df["combo_key"].isin(keep_keys)]
    if df.empty:
        raise SystemExit(f"No combos with at least {min_views} images.")
    samples = []
    for key, group in df.groupby("combo_key"):
        paths = group["Image_Path"].tolist()
        targets = group[TARGET_COLS].mean().values  # mean targets per combo
        samples.append({"combo_key": key, "Paths": paths, "targets": targets, "num_views": len(paths)})
    return pd.DataFrame(samples)


def _gradcam_overlays(model, fused_inputs, keys, out_dir: Path, prefix: str, already_saved: int, max_to_save: int) -> int:
    """
    Compute Grad-CAM for the last conv block (layer4) and save simple red overlays.
    Returns how many images were saved from this batch.
    """
    if already_saved >= max_to_save:
        return 0

    target_layer = model.backbone[-2]  # layer4 output before avgpool
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["value"] = output

    def bwd_hook(_, __, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    fused_inputs = fused_inputs.requires_grad_(True)
    model.zero_grad()
    preds = model(fused_inputs)
    preds.sum().backward()

    h1.remove()
    h2.remove()

    if "value" not in activations or "value" not in gradients:
        return 0

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=fused_inputs.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

    imgs_cpu = fused_inputs.detach().cpu()
    cam_cpu = cam.detach().cpu()

    out_dir.mkdir(parents=True, exist_ok=True)
    mean = IMAGENET_MEAN.view(1, 3, 1, 1)
    std = IMAGENET_STD.view(1, 3, 1, 1)

    saved_here = 0
    for i in range(cam_cpu.shape[0]):
        if already_saved + saved_here >= max_to_save:
            break
        cam_i = cam_cpu[i]
        cam_i -= cam_i.min()
        cam_i = cam_i / (cam_i.max() + 1e-8)
        h_cam, w_cam = cam_i.shape[-2:]
        heat_rgb = (cm.get_cmap("magma")(cam_i.numpy())[:, :, :3] * 255).astype(np.uint8)

        img = imgs_cpu[i] * std + mean
        # tolerate accidental extra dims (e.g., if fused_inputs had an extra leading dim)
        while img.dim() > 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        img = img.clamp(0, 1)
        base = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        overlay = (0.65 * base + 0.35 * heat_rgb).clip(0, 255).astype(np.uint8)

        fname = out_dir / f"{prefix}_{already_saved + saved_here:04d}_{keys[i]}.png"
        Image.fromarray(overlay).save(fname)
        saved_here += 1

    return saved_here


def _save_scatter_plot(y_true, y_pred, out_path: Path, title: str, pearson_r: float | None = None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=12, alpha=0.6, edgecolor="none")
    lims = [
        float(np.min([y_true.min(), y_pred.min()])),
        float(np.max([y_true.max(), y_pred.max()])),
    ]
    plt.plot(lims, lims, "k--", linewidth=1)
    title_suffix = f" (r={pearson_r:.3f})" if pearson_r is not None else ""
    plt.title(title + title_suffix)
    plt.xlabel("True (raw)")
    plt.ylabel("Predicted (raw)")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_attention_for_loader(loader, model, args, split_name: str, already_saved: int) -> int:
    """Iterate a loader and dump Grad-CAM overlays (early fusion only)."""
    if args.fusion != "early":
        return 0
    saved_total = 0
    for inputs, _, keys in loader:
        if already_saved + saved_total >= args.attention_limit:
            break
        inputs = inputs.to(DEVICE)
        # pick the first non-padded view for a sharper overlay; fallback to mean if all padded
        view_mask = inputs.abs().sum(dim=(2, 3, 4)) > 0  # [B, V]
        reps = []
        for b in range(inputs.size(0)):
            valid_idx = view_mask[b].nonzero(as_tuple=False)
            if len(valid_idx):
                reps.append(inputs[b, valid_idx[0].item()])
            else:
                reps.append(inputs[b].mean(dim=0))
        fused = torch.stack(reps, dim=0)
        saved = _gradcam_overlays(
            model,
            fused,
            keys,
            args.attention_dir,
            split_name,
            already_saved + saved_total,
            args.attention_limit,
        )
        saved_total += saved
    return saved_total


def predict_tta(model, inputs, mask, n_aug: int = 16):
    """Simple MIL TTA: rotate/flip valid views, average predictions."""
    preds = []
    for _ in range(n_aug):
        aug_inputs = inputs.clone()
        for b in range(inputs.size(0)):
            for v in range(inputs.size(1)):
                if mask[b, v]:
                    img = aug_inputs[b, v]
                    angle = torch.randint(-5, 6, (1,), device=inputs.device).item()
                    img = tv_transforms.functional.rotate(img, angle)
                    if torch.rand(1, device=inputs.device) > 0.5:
                        img = tv_transforms.functional.hflip(img)
                    aug_inputs[b, v] = img
        preds.append(model(aug_inputs, mask))
    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Allow disabling scatter by passing an empty string
    if args.test_scatter and str(args.test_scatter).strip() == "":
        args.test_scatter = None
    mode_desc = f"{'EVAL' if args.eval_only else 'TRAINING'} {'MIL' if args.method == 'mil' else f'Fusion ({args.fusion})'}"
    print(f"--- {mode_desc} (all views per combo): {', '.join(TARGET_COLS)} ---")
    df = load_and_prepare(min_views=args.min_views)
    groups = df["combo_key"]
    # report raw target stats before scaling
    raw_targets = np.vstack(df["targets"].to_list())
    print(
        f"Target stats (raw): min={raw_targets.min():.4f}, max={raw_targets.max():.4f}, "
        f"mean={raw_targets.mean():.4f}, std={raw_targets.std():.4f}"
    )

    # Two-stage group split: train/val/test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=42)
    train_tmp_idx, test_idx = next(gss_test.split(df, groups=groups))
    train_tmp_df = df.iloc[train_tmp_idx].copy()
    test_df = df.iloc[test_idx].copy()
    # Adjust val fraction relative to remaining data
    val_rel = VAL_FRACTION / max(1e-9, (1.0 - TEST_FRACTION))
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=24)
    train_idx, val_idx = next(gss_val.split(train_tmp_df, groups=train_tmp_df["combo_key"]))
    train_df = train_tmp_df.iloc[train_idx].copy()
    val_df = train_tmp_df.iloc[val_idx].copy()
    print(f"Combos -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} (grouped by patient/eye)")

    scaler = StandardScaler()
    if args.eval_only:
        if not args.scaler_path.exists():
            raise SystemExit(f"Scaler file not found: {args.scaler_path}")
        data = np.load(args.scaler_path)
        scaler.mean_ = data["mean"]
        scaler.scale_ = data["scale"]
        scaler.var_ = scaler.scale_**2
        scaler.n_samples_seen_ = len(train_df)
    train_df[TARGET_COLS] = scaler.fit_transform(pd.DataFrame(train_df["targets"].tolist(), columns=TARGET_COLS)) if not args.eval_only else scaler.transform(pd.DataFrame(train_df["targets"].tolist(), columns=TARGET_COLS))
    val_df[TARGET_COLS] = scaler.transform(pd.DataFrame(val_df["targets"].tolist(), columns=TARGET_COLS))
    test_df[TARGET_COLS] = scaler.transform(pd.DataFrame(test_df["targets"].tolist(), columns=TARGET_COLS))
    # replace targets with scaled
    train_df["targets"] = train_df[TARGET_COLS].values.tolist()
    val_df["targets"] = val_df[TARGET_COLS].values.tolist()
    test_df["targets"] = test_df[TARGET_COLS].values.tolist()
    if not args.eval_only:
        np.savez(args.scaler_path, mean=scaler.mean_, scale=scaler.scale_, cols=TARGET_COLS)

    base_train_tf = transforms.Compose(
        [
            transforms.Resize(int(IMG_SIZE * 1.05)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    base_val_tf = transforms.Compose(
        [
            transforms.Resize(int(IMG_SIZE * 1.05)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_samples = train_df.to_dict(orient="records")
    val_samples = val_df.to_dict(orient="records")
    test_samples = test_df.to_dict(orient="records")

    effective_max_views = args.views_per_bag if args.views_per_bag is not None else args.max_views
    if args.views_per_bag is not None:
        # drop combos with fewer views than required
        train_df = train_df[train_df["num_views"] >= args.views_per_bag].copy()
        val_df = val_df[val_df["num_views"] >= args.views_per_bag].copy()
        test_df = test_df[test_df["num_views"] >= args.views_per_bag].copy()
    train_samples = train_df.to_dict(orient="records")
    val_samples = val_df.to_dict(orient="records")
    test_samples = test_df.to_dict(orient="records")

    train_ds = MultiViewDataset(train_samples, TARGET_COLS, base_transform=base_train_tf, max_views=effective_max_views, fixed_views=args.views_per_bag)
    val_ds = MultiViewDataset(val_samples, TARGET_COLS, base_transform=base_val_tf, max_views=effective_max_views, fixed_views=args.views_per_bag)
    test_ds = MultiViewDataset(test_samples, TARGET_COLS, base_transform=base_val_tf, max_views=effective_max_views, fixed_views=args.views_per_bag)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_views)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_views)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_views)

    if args.method == "fusion":
        model = EarlyFusionResNet(out_dim=len(TARGET_COLS)).to(DEVICE)
    else:
        model = MILResNet(out_dim=len(TARGET_COLS), top_k=args.top_k).to(DEVICE)
    criterion = nn.SmoothL1Loss(beta=1.0)
    backbone_frozen = False
    if args.freeze_epochs > 0:
        freeze_backbone(model)
        backbone_frozen = True
        print(f"[TRAIN] Freezing backbone for first {args.freeze_epochs} epoch(s).")

    if args.tune:
        try:
            import optuna
        except ImportError:
            raise SystemExit("Optuna is not installed. Please install with `pip install optuna` to use --tune.")

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
            wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [4, 8])
            mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.4)
            trial_views = trial.suggest_categorical("views_per_bag", [4, 6, 8, 10, 12, 15])
            trial_topk = trial.suggest_categorical("top_k", [3, 5, 7, 9])
            # filter combos to those with enough views
            trial_train_df = train_df[train_df["num_views"] >= trial_views]
            trial_val_df = val_df[val_df["num_views"] >= trial_views]
            if len(trial_train_df) == 0 or len(trial_val_df) == 0:
                raise optuna.TrialPruned()
            trial_train_ds = MultiViewDataset(trial_train_df.to_dict(orient="records"), TARGET_COLS, base_transform=base_train_tf, max_views=trial_views, fixed_views=trial_views)
            trial_val_ds = MultiViewDataset(trial_val_df.to_dict(orient="records"), TARGET_COLS, base_transform=base_val_tf, max_views=trial_views, fixed_views=trial_views)
            trial_train_loader = DataLoader(trial_train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_views)
            trial_val_loader = DataLoader(trial_val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_views)
            if args.method == "fusion":
                trial_model = EarlyFusionResNet(out_dim=len(TARGET_COLS)).to(DEVICE)
            else:
                trial_model = MILResNet(out_dim=len(TARGET_COLS), top_k=trial_topk).to(DEVICE)
            trial_ema = copy.deepcopy(trial_model)
            for p in trial_ema.parameters():
                p.requires_grad_(False)
            optimizer = optim.AdamW(trial_model.parameters(), lr=lr, weight_decay=wd)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tune_epochs)
            best_val = float("inf")
            best_epoch = 0
            wait = 0

            for epoch in range(1, args.tune_epochs + 1):
                trial_model.train()
                running = 0.0
                for inputs, targets, _ in trial_train_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    if mixup_alpha > 0:
                        inputs, targets_a, targets_b, lam = mixup_batch(inputs, targets, mixup_alpha)
                    if args.method == "fusion":
                        if args.fusion == "early":
                            fused = inputs.mean(dim=1)
                            outputs = trial_model(fused)
                        else:
                            b, v, c, h, w = inputs.shape
                            flat = inputs.view(b * v, c, h, w)
                            out_flat = trial_model(flat)
                            outputs = out_flat.view(b, v, -1).mean(dim=1)
                    else:
                        mask = inputs.abs().sum(dim=(2, 3, 4)) > 0
                        outputs = trial_model(inputs, mask)
                    if mixup_alpha > 0:
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    else:
                        loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(trial_model.parameters(), 1.0)
                    optimizer.step()
                    with torch.no_grad():
                        for p_ema, p in zip(trial_ema.parameters(), trial_model.parameters()):
                            p_ema.mul_(EMA_DECAY).add_(p, alpha=1.0 - EMA_DECAY)
                    running += loss.item() * inputs.size(0)
                scheduler.step()

                # validation
                trial_model.eval()
                val_running = 0.0
                with torch.no_grad():
                    for inputs, targets, _ in trial_val_loader:
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        if args.method == "fusion":
                            if args.fusion == "early":
                                fused = inputs.mean(dim=1)
                                outputs = trial_ema(fused)
                            else:
                                b, v, c, h, w = inputs.shape
                                flat = inputs.view(b * v, c, h, w)
                                out_flat = trial_ema(flat)
                                outputs = out_flat.view(b, v, -1).mean(dim=1)
                        else:
                            mask = inputs.abs().sum(dim=(2, 3, 4)) > 0
                            outputs = trial_ema(inputs, mask)
                        diff = outputs - targets
                        val_running += torch.mean(diff ** 2).item() * inputs.size(0)
                val_loss = val_running / len(val_ds)
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    wait = 0
                else:
                    wait += 1
                best_val = min(best_val, val_loss)
                trial.report(val_loss, epoch)
                print(f"[TUNE][Trial {trial.number}] Epoch {epoch}/{args.tune_epochs} | lr={lr:.2e}, wd={wd:.2e}, bs={batch_size}, mixup={mixup_alpha:.3f} | val_mse={val_loss:.4f} | best={best_val:.4f} @ {best_epoch}")
                if trial.should_prune():
                    raise optuna.TrialPruned()
                if wait >= args.tune_patience:
                    break

            return best_val

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.tune_trials)
        print(f"[TUNE] Best trial value: {study.best_trial.value:.4f}")
        print(f"[TUNE] Best params: {study.best_trial.params}")
        if args.tune_save:
            save_obj = {
                "best_value": study.best_trial.value,
                "best_params": study.best_trial.params,
                "best_trial": study.best_trial.number,
                "trials": len(study.trials),
            }
            args.tune_save.parent.mkdir(parents=True, exist_ok=True)
            with open(args.tune_save, "w") as f:
                json.dump(save_obj, f, indent=2)
            print(f"[TUNE] Saved best params to {args.tune_save}")
        return

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    if args.eval_only:
        if not args.checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        ema_model.load_state_dict(state, strict=False)
        print(f"[EVAL] Loaded weights from {args.checkpoint}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=args.eta_min)

    best_loss = float("inf")
    best_state = None
    wait = 0

    if not args.eval_only:
        for epoch in range(1, NUM_EPOCHS + 1):
            if backbone_frozen and epoch > args.freeze_epochs:
                unfreeze_backbone(model)
                backbone_frozen = False
                for g in optimizer.param_groups:
                    g["lr"] *= args.unfreeze_lr_factor
                print(f"[TRAIN] Unfroze backbone at epoch {epoch}, LR scaled by {args.unfreeze_lr_factor}.")
            model.train()
            running = 0.0
            for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
                # inputs: [B, V, C, H, W]
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                if args.mixup_alpha > 0:
                    inputs, targets_a, targets_b, lam = mixup_batch(inputs, targets, args.mixup_alpha)
                if args.method == "fusion":
                    if args.fusion == "early":
                        fused = inputs.mean(dim=1)
                        outputs = model(fused)
                    else:  # late fusion: run each view then average predictions
                        b, v, c, h, w = inputs.shape
                        flat = inputs.view(b * v, c, h, w)
                        out_flat = model(flat)
                        outputs = out_flat.view(b, v, -1).mean(dim=1)
                else:
                    mask = inputs.abs().sum(dim=(2, 3, 4)) > 0
                    outputs = model(inputs, mask)
                optimizer.zero_grad()
                if args.mixup_alpha > 0:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.mul_(EMA_DECAY).add_(p, alpha=1.0 - EMA_DECAY)
                running += loss.item() * inputs.size(0)

            scheduler.step()
            train_loss = running / len(train_ds)

            model.eval()
            ema_model.eval()
            val_running = 0.0
            val_mae_running = 0.0
            val_true_raw = []
            val_pred_raw = []
            with torch.no_grad():
                for inputs, targets, _ in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    if args.method == "fusion":
                        if args.fusion == "early":
                            fused = inputs.mean(dim=1)
                            outputs = ema_model(fused)
                        else:
                            b, v, c, h, w = inputs.shape
                            flat = inputs.view(b * v, c, h, w)
                            out_flat = ema_model(flat)
                            outputs = out_flat.view(b, v, -1).mean(dim=1)
                    else:
                        mask = inputs.abs().sum(dim=(2, 3, 4)) > 0
                        outputs = predict_tta(ema_model, inputs, mask, n_aug=args.tta) if args.tta > 0 else ema_model(inputs, mask)
                    diff = outputs - targets
                    val_running += torch.mean(diff ** 2).item() * inputs.size(0)
                    val_mae_running += torch.mean(torch.abs(diff)).item() * inputs.size(0)
                    mu = torch.tensor(scaler.mean_, device=DEVICE)
                    sig = torch.tensor(scaler.scale_, device=DEVICE)
                    pred_raw = outputs * sig + mu
                    targ_raw = targets * sig + mu
                    val_pred_raw.append(pred_raw.detach().cpu().numpy())
                    val_true_raw.append(targ_raw.detach().cpu().numpy())

            val_loss = val_running / len(val_ds)
            val_mae = val_mae_running / len(val_ds)
            if val_true_raw:
                y_true = np.vstack(val_true_raw).ravel()
                y_pred = np.vstack(val_pred_raw).ravel()
                mae_raw = float(np.mean(np.abs(y_pred - y_true)))
                rmse_raw = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
                r_raw = pearsonr_np(y_true, y_pred)
            else:
                mae_raw = rmse_raw = r_raw = float("nan")
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Train (MSE): {train_loss:.4f} | Val (MSE): {val_loss:.4f} | "
                f"Val MAE(z): {val_mae:.4f} | Val MAE(raw): {mae_raw:.4f} | "
                f"Val RMSE(raw): {rmse_raw:.4f} | Val Pearson r: {r_raw:.4f} | LR: {current_lr:.6f}"
            )

            if val_loss < best_loss - MIN_DELTA:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, args.checkpoint)
                print(f"  >>> New Best Saved! ({best_loss:.4f})")
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"Early stopping at epoch {epoch} (patience {PATIENCE})")
                    break

        print(f"DONE! Best Val Loss: {best_loss:.4f}")
    else:
        best_state = copy.deepcopy(model.state_dict())

    # Final evaluation with EMA weights (or loaded weights in eval-only)
    ema_model.eval()
    model.eval()
    val_running = 0.0
    val_mae_running = 0.0
    val_true_raw = []
    val_pred_raw = []
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            if args.method == "fusion":
                if args.fusion == "early":
                    fused = inputs.mean(dim=1)
                    outputs = ema_model(fused)
                else:
                    b, v, c, h, w = inputs.shape
                    flat = inputs.view(b * v, c, h, w)
                    out_flat = ema_model(flat)
                    outputs = out_flat.view(b, v, -1).mean(dim=1)
            else:
                mask = inputs.abs().sum(dim=(2, 3, 4)) > 0
                outputs = predict_tta(ema_model, inputs, mask, n_aug=args.tta) if args.tta > 0 else ema_model(inputs, mask)
            diff = outputs - targets
            val_running += torch.mean(diff ** 2).item() * inputs.size(0)
            val_mae_running += torch.mean(torch.abs(diff)).item() * inputs.size(0)
            mu = torch.tensor(scaler.mean_, device=DEVICE)
            sig = torch.tensor(scaler.scale_, device=DEVICE)
            pred_raw = outputs * sig + mu
            targ_raw = targets * sig + mu
            val_pred_raw.append(pred_raw.detach().cpu().numpy())
            val_true_raw.append(targ_raw.detach().cpu().numpy())

    val_loss = val_running / len(val_ds)
    val_mae = val_mae_running / len(val_ds)
    if val_true_raw:
        y_true = np.vstack(val_true_raw).ravel()
        y_pred = np.vstack(val_pred_raw).ravel()
        mae_raw = float(np.mean(np.abs(y_pred - y_true)))
        rmse_raw = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        r_raw = pearsonr_np(y_true, y_pred)
    else:
        mae_raw = rmse_raw = r_raw = float("nan")
    print(
        f"[FINAL] Val (MSE): {val_loss:.4f} | Val MAE(z): {val_mae:.4f} | "
        f"Val MAE(raw): {mae_raw:.4f} | Val RMSE(raw): {rmse_raw:.4f} | Pearson r: {r_raw:.4f}"
    )

    # Test evaluation (if any test combos exist)
    if len(test_ds):
        test_running = 0.0
        test_mae_running = 0.0
        test_true_raw = []
        test_pred_raw = []
        with torch.no_grad():
            for inputs, targets, _ in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                if args.method == "fusion":
                    if args.fusion == "early":
                        fused = inputs.mean(dim=1)
                        outputs = ema_model(fused)
                    else:
                        b, v, c, h, w = inputs.shape
                        flat = inputs.view(b * v, c, h, w)
                        out_flat = ema_model(flat)
                        outputs = out_flat.view(b, v, -1).mean(dim=1)
                else:
                    mask = inputs.abs().sum(dim=(2, 3, 4)) > 0
                    outputs = predict_tta(ema_model, inputs, mask, n_aug=args.tta) if args.tta > 0 else ema_model(inputs, mask)
                diff = outputs - targets
                test_running += torch.mean(diff ** 2).item() * inputs.size(0)
                test_mae_running += torch.mean(torch.abs(diff)).item() * inputs.size(0)
                mu = torch.tensor(scaler.mean_, device=DEVICE)
                sig = torch.tensor(scaler.scale_, device=DEVICE)
                pred_raw = outputs * sig + mu
                targ_raw = targets * sig + mu
                test_pred_raw.append(pred_raw.detach().cpu().numpy())
                test_true_raw.append(targ_raw.detach().cpu().numpy())

        test_loss = test_running / len(test_ds)
        test_mae = test_mae_running / len(test_ds)
        if test_true_raw:
            y_true_t = np.vstack(test_true_raw).ravel()
            y_pred_t = np.vstack(test_pred_raw).ravel()
            mae_raw_t = float(np.mean(np.abs(y_pred_t - y_true_t)))
            rmse_raw_t = float(np.sqrt(np.mean((y_pred_t - y_true_t) ** 2)))
            r_raw_t = pearsonr_np(y_true_t, y_pred_t)
        else:
            mae_raw_t = rmse_raw_t = r_raw_t = float("nan")
        print(
            f"[TEST] Test (MSE): {test_loss:.4f} | Test MAE(z): {test_mae:.4f} | "
            f"Test MAE(raw): {mae_raw_t:.4f} | Test RMSE(raw): {rmse_raw_t:.4f} | Pearson r: {r_raw_t:.4f}"
        )
        if test_true_raw and args.test_scatter:
            title = f"Test Pred vs True ({', '.join(TARGET_COLS)})"
            _save_scatter_plot(y_true_t, y_pred_t, args.test_scatter, title, pearson_r=r_raw_t)
            print(f"[TEST] Scatter saved to {args.test_scatter}")
    else:
        print("[TEST] No test combos available; skipped test evaluation.")

    # Optional attention overlays (Grad-CAM on early-fusion path)
    if args.attention_dir:
        if args.method != "fusion" or args.fusion != "early":
            print("[ATTN] Attention saving only implemented for early fusion; skipping.")
        else:
            total_saved = 0
            total_saved += _save_attention_for_loader(val_loader, ema_model, args, "val", total_saved)
            if len(test_ds) and total_saved < args.attention_limit:
                total_saved += _save_attention_for_loader(test_loader, ema_model, args, "test", total_saved)
            if total_saved:
                print(f"[ATTN] Saved {total_saved} attention overlays to {args.attention_dir}")
            else:
                print("[ATTN] No attention overlays were saved (check limit or data).")


if __name__ == "__main__":
    main()
