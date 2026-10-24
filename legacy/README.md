# Archived pre-refactor code

These three scripts are the original implementation, kept unchanged for
provenance. They are **not** used by anything in `src/` and are not runnable as-is
(hard-coded relative paths, OpenCV image loading, augmentation that ignores
bounding boxes).

| File | Original role | Replaced by |
|---|---|---|
| `data_processing.py` | `Dataset` + splitter + transforms | `src/dataset.py`, `scripts/prepare_dataset.py` |
| `train_model.py` | dataset class, splitter, model wrapper, training loop | `src/dataset.py`, `src/model.py`, `src/train.py` |
| `visual.py` | draw predictions, unused confusion-matrix/PR helpers | `src/visualize.py`, `src/predict.py`, `src/metrics.py` |

See the "Reproducibility notes" section of the main README for the specific bugs
these versions contained and how they were fixed.
