"""COCO-style dataset for bottled-liquor surface defects.

Two deliberate differences from the original ``data_processing.py``:

1. **Images are decoded with Pillow, not OpenCV.**  ``opencv-python`` is a heavy
   dependency that frequently breaks on headless machines and, in the
   environment this repository was rebuilt in, resolved to a broken namespace
   package.  Pillow ships with matplotlib anyway.
2. **Photometric / geometric augmentation is applied consistently.**  The
   original pipeline flipped and rotated the *image* but kept the *boxes*
   untouched, so roughly half of the training samples had a box drawn on the
   wrong half of the picture.  Here a horizontal flip mirrors the boxes too, and
   rotation is dropped (it would require recomputing boxes and would cut off
   defects at the border).

The tensors handed to the model stay in ``[0, 1]`` RGB and are **not**
normalised here: torchvision's ``GeneralizedRCNNTransform`` already applies the
ImageNet mean/std internally.  The original code normalised twice (once in the
dataset, once inside the model), which pushed the backbone far off the
distribution its COCO weights were trained on.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .config import EVAL_CLASS_IDS, ID_TO_NAME


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def load_coco(annotation_file: str) -> Dict:
    """Read a COCO annotation file (utf-8, with a gb18030 fallback)."""
    try:
        with open(annotation_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except UnicodeDecodeError:
        with open(annotation_file, "r", encoding="gb18030") as handle:
            return json.load(handle)


def index_annotations(coco: Dict) -> Dict[int, List[Dict]]:
    """Group annotations by ``image_id``."""
    grouped: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped


def class_distribution(coco: Dict, include_background: bool = True) -> Dict[str, int]:
    """Number of annotated instances per class, keyed by class name."""
    counts = {name: 0 for name in ID_TO_NAME.values()}
    for ann in coco.get("annotations", []):
        name = ID_TO_NAME.get(int(ann["category_id"]))
        if name is None:
            continue
        if not include_background and int(ann["category_id"]) not in EVAL_CLASS_IDS:
            continue
        counts[name] += 1
    return counts


def dataset_summary(annotation_file: str, image_dir: Optional[str] = None) -> Dict:
    """Quick sanity report: image count, instance count, per-class counts."""
    coco = load_coco(annotation_file)
    summary = {
        "annotation_file": annotation_file,
        "num_images": len(coco.get("images", [])),
        "num_annotations": len(coco.get("annotations", [])),
        "per_class": class_distribution(coco),
    }
    if image_dir and os.path.isdir(image_dir):
        present = sum(
            1
            for img in coco["images"]
            if os.path.exists(os.path.join(image_dir, img["file_name"]))
        )
        summary["num_images_on_disk"] = present
    return summary


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class DefectDataset(Dataset):
    """Object-detection dataset returning ``(image_tensor, target)`` pairs.

    Parameters
    ----------
    image_dir:
        Directory holding the ``.jpg`` files referenced by the annotation file.
    annotation_file:
        COCO JSON with ``images`` / ``annotations`` / ``categories`` sections.
    train:
        Enables augmentation and filters out images without any annotation
        (Faster R-CNN cannot compute a loss on an empty target set).
    hflip_prob:
        Probability of a horizontal flip (boxes are mirrored accordingly).
    color_jitter:
        Strength of the brightness / contrast / saturation jitter, ``0`` to
        disable.
    skip_background:
        Drop annotations whose ``category_id`` is 0.  Label 0 is the reserved
        background slot of torchvision detectors, so those annotations can never
        be learned; dropping them avoids confusing gradients.  Defaults to
        ``False`` so that the released checkpoint stays reproducible.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        train: bool = False,
        hflip_prob: float = 0.5,
        color_jitter: float = 0.2,
        skip_background: bool = False,
        verbose: bool = True,
    ) -> None:
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.train = train
        self.hflip_prob = hflip_prob if train else 0.0
        self.color_jitter = color_jitter if train else 0.0
        self.skip_background = skip_background

        coco = load_coco(annotation_file)
        self.coco = coco
        grouped = index_annotations(coco)

        self.records: List[Dict] = []
        dropped_missing = 0
        dropped_empty = 0
        for image in coco.get("images", []):
            path = os.path.join(image_dir, image["file_name"])
            if not os.path.exists(path):
                dropped_missing += 1
                continue
            anns = grouped.get(image["id"], [])
            if skip_background:
                anns = [a for a in anns if int(a["category_id"]) != 0]
            if train and not anns:
                dropped_empty += 1
                continue
            boxes = np.asarray(
                [
                    [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                    for a in anns
                ],
                dtype=np.float32,
            ).reshape(-1, 4)
            labels = np.asarray([int(a["category_id"]) for a in anns], dtype=np.int64)
            self.records.append(
                {
                    "image_id": image["id"],
                    "file_name": image["file_name"],
                    "path": path,
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "boxes": boxes,
                    "labels": labels,
                }
            )

        if verbose:
            mode = "train" if train else "eval"
            print(
                f"[dataset] {mode}: {len(self.records)} images"
                f" | {int(sum(len(r['labels']) for r in self.records))} instances"
                f" | skipped {dropped_missing} missing, {dropped_empty} unlabelled"
            )

    # -- torch API ---------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        boxes = record["boxes"].copy()
        labels = record["labels"].copy()

        # Geometric augmentation: flip image *and* boxes together.
        if self.hflip_prob and random.random() < self.hflip_prob:
            width = image.width
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(boxes):
                x1 = boxes[:, 0].copy()
                boxes[:, 0] = width - boxes[:, 2]
                boxes[:, 2] = width - x1

        # Photometric augmentation (no geometry involved).
        if self.color_jitter:
            jitter = self.color_jitter
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(1 - jitter, 1 + jitter)
                )
            if random.random() < 0.5:
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(1 - jitter, 1 + jitter)
                )
            if random.random() < 0.3:
                image = ImageEnhance.Color(image).enhance(
                    random.uniform(1 - jitter, 1 + jitter)
                )

        array = np.array(image, dtype=np.uint8)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)

        # Clamp to the image extent so mirrored / rounded coordinates stay valid.
        if len(boxes):
            height, width = array.shape[0], array.shape[1]
            boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
            boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes, labels = boxes[keep], labels[keep]

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64).reshape(-1),
            "image_id": torch.tensor([record["image_id"]]),
        }
        return tensor, target


def collate_fn(batch):
    """Detection models expect a list of images and a list of targets."""
    return tuple(zip(*batch))
