"""
Simple keyboard-driven labeling tool for slit-lamp views.

Labels:
  c = center
  n = Van Herick nasal
  t = Van Herick temporal
  u = unsure (leaves empty)

Controls (GUI mode):
  - Press c/n/t/u to label current image and advance
  - Backspace: go back one image
  - s: save progress to CSV (default: view_labels_manual.csv)
  - q: save and quit

CLI mode (--cli) for headless terminals:
  - Type c/n/t/u to label and advance
  - b to go back, s to save, q to quit
  - Prints the image path so you can open it externally if needed

Paths:
  - Input CSV must contain Image_Path and eye_clean (used to resolve nasal/temporal later)
  - Images resolved relative to --img-root if not directly found
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

LABEL_KEYS = {"c": "center", "n": "vh_nasal", "t": "vh_temporal", "u": None}


def resolve_path(p: str, img_root: Path) -> str | None:
    if not isinstance(p, str):
        return None
    if os.path.exists(p):
        return p
    fname = os.path.basename(p.replace("\\", "/"))
    candidate = img_root / fname
    if candidate.exists():
        return str(candidate)
    return None


def main():
    ap = argparse.ArgumentParser(description="Keyboard-driven labeling tool for center/vh_nasal/vh_temporal.")
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV with Image_Path (and optional existing label_mapped).")
    ap.add_argument(
        "--img-root",
        type=Path,
        default=Path("/home/alireza/Code/RETFound_GeneLab/slit-project/data/center_roi_images/processed_images_448"),
        help="Root dir for images.",
    )
    ap.add_argument("--out-csv", type=Path, default=Path("view_labels_manual.csv"), help="Output CSV to save labels.")
    ap.add_argument("--start-idx", type=int, default=0, help="Index to start labeling from (after shuffling).")
    ap.add_argument("--cli", action="store_true", help="Use CLI (non-GUI) labeling for headless environments.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["Image_Path"] = df["Image_Path"].apply(lambda p: resolve_path(p, args.img_root))
    df = df.dropna(subset=["Image_Path"]).reset_index(drop=True)

    # attach existing labels if present (either from input or previous run)
    if "label_mapped" not in df.columns:
        df["label_mapped"] = None
    if args.out_csv.exists():
        prev = pd.read_csv(args.out_csv)
        prev_map = dict(zip(prev["Image_Path"], prev.get("label_mapped", [None] * len(prev))))
        df["label_mapped"] = df["Image_Path"].map(prev_map).fillna(df["label_mapped"])

    # shuffle for labeling session
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    idx = max(0, min(args.start_idx, len(df) - 1))

    def save_progress():
        df[["Image_Path", "label_mapped"]].to_csv(args.out_csv, index=False)
        print(f"Saved progress to {args.out_csv}")

    if args.cli:
        print("CLI mode. Controls: c=center, n=vh_nasal, t=vh_temporal, u=unsure, b=back, s=save, q=save+quit")
        while True:
            row = df.iloc[idx]
            print(f"[{idx+1}/{len(df)}] {row['Image_Path']} | current: {row.get('label_mapped')}")
            cmd = input("Label (c/n/t/u), b=back, s=save, q=quit: ").strip().lower()
            if cmd in LABEL_KEYS:
                df.at[idx, "label_mapped"] = LABEL_KEYS[cmd]
                idx = min(idx + 1, len(df) - 1)
            elif cmd == "b":
                idx = max(0, idx - 1)
            elif cmd == "s":
                save_progress()
            elif cmd == "q":
                save_progress()
                break
            else:
                print("Unrecognized command.")
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.axis("off")

        def render():
            ax.clear()
            row = df.iloc[idx]
            img = Image.open(row["Image_Path"]).convert("RGB")
            ax.imshow(img)
            title = f"{idx+1}/{len(df)} | Path: {os.path.basename(row['Image_Path'])}"
            if pd.notna(row.get("label_mapped")) and row["label_mapped"]:
                title += f" | Current: {row['label_mapped']}"
            ax.set_title(title)
            ax.axis("off")
            fig.canvas.draw_idle()

        def on_key(event):
            nonlocal idx
            if event.key in LABEL_KEYS:
                df.at[idx, "label_mapped"] = LABEL_KEYS[event.key]
                idx = min(idx + 1, len(df) - 1)
                render()
            elif event.key == "backspace":
                idx = max(0, idx - 1)
                render()
            elif event.key == "s":
                save_progress()
            elif event.key == "q":
                save_progress()
                plt.close(fig)
                sys.exit(0)

        fig.canvas.mpl_connect("key_press_event", on_key)
        render()
        print("Controls: c=center, n=vh_nasal, t=vh_temporal, u=unsure, backspace=prev, s=save, q=save+quit")
        plt.show()


if __name__ == "__main__":
    main()
