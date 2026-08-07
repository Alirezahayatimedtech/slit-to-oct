# Slit-Lamp View Annotation Utility

`code/label_gui.py` is a local keyboard-driven tool for assigning image-space
view labels and converting them to anatomical nasal/temporal labels.

The annotation utility is provided for method reuse. The controlled dataset
images are not stored in this repository.

## Input

The GUI expects a CSV containing an `Image_Path` column. If present, `eye_clean`
must encode eye laterality as `OD`/`R` or `OS`/`L`.

The working label file contains:

| Column | Meaning |
| --- | --- |
| `Image_Path` | Local path to the controlled image |
| `View_Image` | Image-space label: `left`, `center`, `right`, `no_slit`, or `other` |
| `View_Label` | Anatomical label derived from image-space view and eye laterality |
| `eye_clean` | Normalized eye laterality |

## Run

From the repository root:

```bash
python slit-project/code/label_gui.py
```

The script's input and output paths are configured at the top of the file. Keep
these paths local and do not commit label files containing controlled image
paths.

## Keyboard Controls

| Key | Action |
| --- | --- |
| `A` | image-space left |
| `S` | centre |
| `D` | image-space right |
| `O` | other |
| `U` | no slit |
| `K` | skip |
| `Z` | undo |
| `Q` | save and quit |

## Anatomical Mapping

Image-space left and right are not anatomical labels until eye laterality is
known.

| Eye | Image-space left | Image-space right |
| --- | --- | --- |
| Right eye (`OD`) | temporal | nasal |
| Left eye (`OS`) | nasal | temporal |

`center`, `no_slit`, and `other` are unchanged. Laterality must be validated
before applying this mapping.

## Review Rules

- Use `skip` when the view cannot be assigned confidently.
- Do not infer laterality from image appearance alone.
- Review low-quality, eyelid-obscured, defocused, and off-axis images before
  using labels for training.
- Keep the original annotation, reviewer correction, and final label distinct
  when conducting an inter-reviewer study.
