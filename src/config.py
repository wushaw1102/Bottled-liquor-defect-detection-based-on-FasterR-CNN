"""Central place for class definitions, default paths and hyper-parameters.

Keeping every constant here means the training, evaluation, inference and demo
code paths can never drift apart (a common source of silent label bugs in
object-detection projects).
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Label space
# --------------------------------------------------------------------------- #
# The dataset keeps the COCO convention of a real annotation category with
# ``category_id == 0`` (named 背景 / background).  torchvision's detectors also
# reserve label 0 for "background", so that category can never be predicted.
# We therefore keep the original 0..10 indexing (needed to load the released
# checkpoint) and simply exclude class 0 from the reported metrics.
CLASS_NAMES = [
    "background",       # 0 - reserved slot, not a learnable class
    "cap_cracked",      # 1 - 瓶盖破损
    "cap_deformed",     # 2 - 瓶盖变形
    "cap_edge_broken",  # 3 - 瓶盖坏边
    "cap_swirl",        # 4 - 瓶盖打旋
    "cap_dot_missing",  # 5 - 瓶盖断点
    "label_skew",       # 6 - 标贴歪斜
    "label_wrinkled",   # 7 - 标贴起皱
    "label_bubble",     # 8 - 标贴气泡
    "code_normal",      # 9 - 喷码正常
    "code_defect",      # 10 - 喷码异常
]

CLASS_NAMES_ZH = [
    "背景",
    "瓶盖破损",
    "瓶盖变形",
    "瓶盖坏边",
    "瓶盖打旋",
    "瓶盖断点",
    "标贴歪斜",
    "标贴起皱",
    "标贴气泡",
    "喷码正常",
    "喷码异常",
]

#: Classes that actually get evaluated / displayed (everything but the reserved
#: background slot).
EVAL_CLASS_IDS = list(range(1, len(CLASS_NAMES)))
NUM_CLASSES = len(CLASS_NAMES)  # 11, matches the released checkpoint

ID_TO_NAME = {i: n for i, n in enumerate(CLASS_NAMES)}
ID_TO_NAME_ZH = {i: n for i, n in enumerate(CLASS_NAMES_ZH)}

# --------------------------------------------------------------------------- #
# Paths (all relative to the repository root, resolved at import time)
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(REPO_ROOT, "data")
ANNOTATION_DIR = os.path.join(DATA_DIR, "annotations")
SAMPLE_IMAGE_DIR = os.path.join(DATA_DIR, "sample", "images")
SAMPLE_ANNOTATION_FILE = os.path.join(DATA_DIR, "sample", "annotations.json")
FULL_IMAGE_DIR = os.path.join(DATA_DIR, "images")

DEFAULT_TRAIN_JSON = os.path.join(ANNOTATION_DIR, "instances_train.json")
DEFAULT_VAL_JSON = os.path.join(ANNOTATION_DIR, "instances_val.json")
DEFAULT_TEST_JSON = os.path.join(ANNOTATION_DIR, "instances_test.json")

#: Where the demo app looks for a checkpoint, in order.
DEFAULT_CHECKPOINTS = [
    os.path.join(REPO_ROOT, "outputs", "fasterrcnn_resnet50_fpn", "best_model.pth"),
    os.path.join(REPO_ROOT, "saved_models", "model.pth"),
]

#: Checkpoints produced by the original (pre-refactor) code.  Those were trained
#: on twice-normalised inputs, so the loader compensates automatically.
LEGACY_CHECKPOINTS = [os.path.join(REPO_ROOT, "saved_models", "model.pth")]

# --------------------------------------------------------------------------- #
# Model / training defaults
# --------------------------------------------------------------------------- #
#: Short side fed to the detector.  800 is the torchvision default; 600 is a
#: good compromise for 4-6 GB laptop GPUs.
DEFAULT_MIN_SIZE = 800
DEFAULT_MAX_SIZE = 1333

DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 10
DEFAULT_LR = 5e-4
DEFAULT_SEED = 42

SUPPORTED_ARCHS = [
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
]


def class_name(label_id, chinese=False):
    """Human readable name for a label id, robust to out-of-range values."""
    table = ID_TO_NAME_ZH if chinese else ID_TO_NAME
    return table.get(int(label_id), f"class_{label_id}")
