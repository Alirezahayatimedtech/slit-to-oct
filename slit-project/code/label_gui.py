"""
Simple Tkinter labeling tool:
- Loads images from a CSV (expects column 'Image_Path').
- Shows one image at a time with buttons for image-based labels.
- Saves labels incrementally with:
  * View_Image: left / right / center (image-based)
  * View_Label: center / van_nasal / van_temporal (derived using eye metadata)

Keys: [A]=left, [S]=center, [D]=right, [O]=other, [U]=no slit, [K]=skip, [Z]=undo, [Q]=quit
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd

INPUT_CSV = "ready_for_training.csv"
OUTPUT_CSV = "labels_output.csv"
# If you have 448px images in a different folder, set this to that folder.
IMAGE_OVERRIDE_DIR = r"G:\thesis-slit-oct-project\data\processed_images_448"
FAIL_LOG = "label_gui_failures.csv"
EYE_COLUMN = "eye_clean"
RIGHT_EYES = {"OD", "R", "RIGHT", "OD(RIGHT)"}
LEFT_EYES = {"OS", "L", "LEFT", "OS(LEFT)"}
LABEL_NASAL = "van_nasal"
LABEL_TEMPORAL = "van_temporal"
SHOW_UNCERTAIN_ONLY = False  # show all rows (set True to focus on 'no_slit')
START_FROM_BEGINNING = False  # show all rows from the first image
RELABEL_RANGE = True   # relabel a specific index range
RANGE_START = 0        # inclusive
RANGE_END = 3050       # inclusive
RELABEL_EVERY_OTHER = False  # only relabel odd/even indices
RELABEL_PARITY = 0           # 0=even, 1=odd (based on row index)
RELABEL_LABEL_FILTER = "van_temporal"  # set to a label to relabel only that class
RELABEL_ONLY_UNLABELED = False  # when True, only include rows with empty View_Image
SKIP_PRESERVE_LABEL = True  # when skipping, keep existing label unchanged
BACKUP_ON_START = True
MAX_W, MAX_H = 1024, 768  # resize display


def normalize_eye(val):
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().upper()
    return s


def map_to_anatomy(view_image, eye_val):
    if view_image is None or pd.isna(view_image) or view_image == "" or view_image == "no_slit":
        return "no_slit"
    if view_image == "other":
        return "other"
    if view_image == "center":
        return "center"
    eye_norm = normalize_eye(eye_val)
    # Mapping adjusted: nasal/temporal were reversed in practice
    if eye_norm in RIGHT_EYES:
        return LABEL_TEMPORAL if view_image == "left" else LABEL_NASAL
    if eye_norm in LEFT_EYES:
        return LABEL_NASAL if view_image == "left" else LABEL_TEMPORAL
    # fallback if eye not known
    return f"van_{view_image}"


def resolve_image_path(path):
    if path and os.path.exists(path):
        return path
    if IMAGE_OVERRIDE_DIR:
        candidate = os.path.join(IMAGE_OVERRIDE_DIR, os.path.basename(str(path)))
        if os.path.exists(candidate):
            return candidate
    return path


def load_data():
    if not os.path.exists(INPUT_CSV):
        messagebox.showerror("Error", f"Input CSV not found: {INPUT_CSV}")
        sys.exit(1)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    if "Image_Path" not in df.columns:
        messagebox.showerror("Error", "CSV must contain an 'Image_Path' column.")
        sys.exit(1)
    # load existing labels if present
    if os.path.exists(OUTPUT_CSV):
        if BACKUP_ON_START:
            try:
                import datetime
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = f"labels_output_backup_{ts}.csv"
                pd.read_csv(OUTPUT_CSV).to_csv(backup, index=False)
            except Exception:
                pass
        out_df = pd.read_csv(OUTPUT_CSV)
        keep_cols = ["Image_Path"]
        if "View_Image" in out_df.columns:
            keep_cols.append("View_Image")
        if "View_Label" in out_df.columns:
            keep_cols.append("View_Label")
        out_df = out_df[keep_cols].drop_duplicates("Image_Path")
        df = df.merge(out_df, on="Image_Path", how="left", suffixes=("", "_saved"))

        if "View_Image_saved" in df.columns:
            df["View_Image"] = df["View_Image_saved"].combine_first(df.get("View_Image"))
        if "View_Label_saved" in df.columns:
            df["View_Label"] = df["View_Label_saved"].combine_first(df.get("View_Label"))
        df = df.drop(columns=[c for c in df.columns if c.endswith("_saved")], errors="ignore")

    # ensure columns exist
    if "View_Image" not in df.columns:
        df["View_Image"] = pd.NA
    if "View_Label" not in df.columns:
        df["View_Label"] = pd.NA
    # ensure object dtype to avoid pandas dtype warnings
    df["View_Image"] = df["View_Image"].astype("object")
    df["View_Label"] = df["View_Label"].astype("object")

    # Build index list to label
    if RELABEL_RANGE:
        start = max(0, int(RANGE_START))
        end = min(len(df) - 1, int(RANGE_END))
        if end < start:
            messagebox.showerror("Error", "RANGE_END must be >= RANGE_START.")
            sys.exit(1)
        idx_list = df.index[start:end + 1].tolist()
    elif START_FROM_BEGINNING:
        idx_list = df.index.tolist()
    elif SHOW_UNCERTAIN_ONLY:
        idx_list = df.index[df["View_Image"] == "no_slit"].tolist()
        if not idx_list:
            messagebox.showinfo("Info", "No no-slit rows found; showing all rows instead.")
            idx_list = df.index[df["View_Image"].isna()].tolist()
    else:
        idx_list = df.index[df["View_Image"].isna()].tolist()

    if RELABEL_EVERY_OTHER:
        idx_list = [i for i in idx_list if (int(i) % 2) == int(RELABEL_PARITY)]

    if RELABEL_LABEL_FILTER:
        idx_list = [i for i in idx_list if str(df.at[i, "View_Label"]) == RELABEL_LABEL_FILTER]

    if RELABEL_ONLY_UNLABELED:
        idx_list = [i for i in idx_list if pd.isna(df.at[i, "View_Image"]) or str(df.at[i, "View_Image"]).strip() == ""]

    if not idx_list:
        messagebox.showinfo("Info", "No rows to label.")
        sys.exit(0)

    return df, idx_list


class LabelApp:
    def __init__(self, master, df, idx_list):
        self.master = master
        self.df = df  # full dataframe
        self.idx_list = idx_list  # list of row indices to label
        self.pos = 0  # position within idx_list
        self.history = []  # stack of positions for undo
        master.title("Slit-Lamp Labeler")

        self.canvas = tk.Canvas(master, width=MAX_W, height=MAX_H, bg="black")
        self.canvas.pack()
        self.canvas.focus_set()
        master.focus_force()

        btn_frame = tk.Frame(master)
        btn_frame.pack(fill=tk.X)
        self.info = tk.Label(btn_frame, text="", anchor="w")
        self.info.pack(side=tk.LEFT, padx=5)

        for txt, label in [
            ("Left (A)", "left"),
            ("Center (S)", "center"),
            ("Right (D)", "right"),
            ("Other (O)", "other"),
            ("No Slit (U)", "no_slit"),
            ("Skip (K)", None),
            ("Undo (Z)", "undo"),
        ]:
            tk.Button(btn_frame, text=txt, command=lambda lab=label: self.apply_label(lab)).pack(side=tk.LEFT, padx=3)

        # Bind globally so clicks on the canvas don't steal focus
        key_map = {
            "a": "left",
            "s": "center",
            "d": "right",
            "o": "other",
            "u": "no_slit",
            "k": None,      # skip
            "z": "undo",
            "q": "quit",
        }
        for k, lab in key_map.items():
            # bind on both root + canvas (avoid double-fire from bind_all)
            master.bind(f"<{k}>", lambda e, lab=lab: self.handle_key(lab))
            master.bind(f"<{k.upper()}>", lambda e, lab=lab: self.handle_key(lab))
            self.canvas.bind(f"<{k}>", lambda e, lab=lab: self.handle_key(lab))
            self.canvas.bind(f"<{k.upper()}>", lambda e, lab=lab: self.handle_key(lab))

        self.photo = None
        self.show_current()

    def resize_image(self, img):
        w, h = img.size
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS)

    def show_current(self):
        if self.pos >= len(self.idx_list):
            messagebox.showinfo("Done", "All selected images labeled.")
            return
        idx = self.idx_list[self.pos]
        path = self.df.at[idx, "Image_Path"]
        eye_val = self.df.at[idx, EYE_COLUMN] if EYE_COLUMN in self.df.columns else ""
        current_label = self.df.at[idx, "View_Image"] if "View_Image" in self.df.columns else ""
        self.info.config(text=f"{self.pos+1}/{len(self.idx_list)} - {path} | eye={eye_val} | prev={current_label}")
        try:
            img = Image.open(resolve_image_path(path)).convert("RGB")
            img = self.resize_image(img)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(MAX_W//2, MAX_H//2, image=self.photo)
        except Exception as e:
            self.info.config(text=f"Failed to load: {path} ({e})")
            try:
                with open(FAIL_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{path},\"{str(e).replace('\"','\"\"')}\"\n")
            except Exception:
                pass
            self.apply_label(None)  # skip

    def save_progress(self):
        keep_cols = ["Image_Path", "View_Image", "View_Label"]
        if EYE_COLUMN in self.df.columns:
            keep_cols.append(EYE_COLUMN)
        out = self.df[keep_cols].copy()
        out.to_csv(OUTPUT_CSV, index=False)

    def handle_key(self, label):
        if label == "quit":
            self.quit_app()
        else:
            self.apply_label(label)

    def apply_label(self, label):
        if label == "undo":
            if not self.history:
                return
            self.pos = self.history.pop()
            idx = self.idx_list[self.pos]
            self.df.at[idx, "View_Label"] = pd.NA
            self.save_progress()
            self.show_current()
            return

        if self.pos >= len(self.idx_list):
            return

        # record history
        self.history.append(self.pos)

        idx = self.idx_list[self.pos]
        # Skip behavior: preserve existing label if configured
        if label is None and SKIP_PRESERVE_LABEL:
            self.pos += 1
            self.show_current()
            return

        # set label (None for skip)
        self.df.at[idx, "View_Image"] = label if label is not None else pd.NA
        # derive anatomy label from eye metadata
        eye_val = self.df.at[idx, EYE_COLUMN] if EYE_COLUMN in self.df.columns else None
        self.df.at[idx, "View_Label"] = map_to_anatomy(label, eye_val) if label is not None else pd.NA
        self.save_progress()

        # advance to next position
        self.pos += 1
        self.show_current()

    def quit_app(self):
        self.save_progress()
        self.master.destroy()


def main():
    df, idx_list = load_data()
    root = tk.Tk()
    app = LabelApp(root, df, idx_list)
    root.mainloop()


if __name__ == "__main__":
    main()
