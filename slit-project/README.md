# Slit-lamp Labeling + Active Learning Guide

Practical notes on how we labeled slit-lamp images, ran active learning loops, and reviewed predictions for reproducibility.

## Core files
- `code/label_gui.py`: Tkinter labeling app with keyboard shortcuts.
- `code/labels_output.csv`: running ground-truth labels (Image_Path, View_Image, View_Label, eye_clean).
- `code/active_learning_view_pipeline.py`: semi-supervised/active-learning trainer + predictor.
- `code/outputs/active_learning_iter*/`: per-iteration predictions/checkpoints.
- `code/ready_for_training.csv`: source CSV used by the GUI (merged with any existing labels_output on launch).

## Manual labeling (label_gui.py)
1) Activate the env and launch:
   ```
   conda activate awg
   cd slit-project/code
   python label_gui.py
   ```
2) Shortcuts: `A`=left, `S`=center, `D`=right, `O`=other, `U`=no_slit, `K`=skip, `Z`=undo, `Q`=quit.
3) Anatomy mapping: left/right + eye metadata (OD/OS) are auto-mapped to `van_temporal`/`van_nasal`; center stays `center`; `other` and `no_slit` pass through.
4) It backs up `labels_output.csv` on start (timestamped), merges any prior labels, and writes progress after every action.
5) Relabel controls live at the top of the file (e.g., `RELABEL_RANGE`, `RANGE_START/END`, `RELABEL_ONLY_UNLABELED`). The tool resolves Windows-style paths to local 448px images when available.

## Active learning loop
Repeat each round:
1) Ensure `labels_output.csv` has your latest manual fixes (including reviewed low/high conf from prior round).
2) Run the pipeline from `slit-project/code` (9 epochs, score all remaining unlabeled):
   ```
   conda activate awg
   cd slit-project/code
   python3 active_learning_view_pipeline.py \
     --epochs 9 --batch-size 32 \
     --unlabeled-limit 20000 --uncertain-topk 20000 \
     --high-conf-thresh 0.9 --margin-min 0.15 \
     --outdir outputs/active_learning_iter6   # bump iterN each round
   ```
   - Saves checkpoint `view_classifier_active.pth` and CSVs in the chosen outdir.
   - Key CSVs: `pseudo_labels_high_conf.csv` (auto-add), `high_conf_predictions.csv` (same rows + paths), `low_conf_predictions.csv` (to manually review), `uncertain_for_review.csv`, `uncertain_by_pred_class.csv`, `unlabeled_predictions.csv`.
3) Integrate results:
   - Append `pseudo_labels_high_conf.csv` to `labels_output.csv` (Image_Path + View_Label).
   - Manually relabel `low_conf_predictions.csv` (or `uncertain_by_pred_class.csv` for class balance) using the GUI or spreadsheet, then append to `labels_output.csv`.
4) Rerun the pipeline with a new `--outdir` for the next iteration.

## Iteration log (latest)
- Iter1–4: 666-image batches; high-confidence auto labels were appended and low-confidence rows manually fixed each round.
- Iter6 (latest run, 9 epochs): scored all 5,529 remaining unlabeled images; produced 5,307 high-confidence pseudo labels and 222 low-confidence rows (`outputs/active_learning_iter6/`).
- Current label counts in `labels_output.csv` (approx.): center 2,612; van_nasal 3,852; van_temporal 3,461; other 284; no_slit 174; unlabeled ~5,529.

## VS Code tips
- Open the workspace folder (`RETFound_GeneLab`) and pick the `awg` Python interpreter.
- Use the integrated terminal for the commands above.
- Preview large CSVs via VS Code's built-in “Open Preview” or a CSV viewer extension; filter/sort to review `low_conf_predictions.csv`.
- Keep `labels_output.csv` under version control for traceability; commit after each review round.
