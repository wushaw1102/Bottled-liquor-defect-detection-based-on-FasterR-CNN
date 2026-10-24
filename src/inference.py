"""Shared inference helper used by the CLI, the evaluation script and the demo."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .config import (
    DEFAULT_MAX_SIZE,
    DEFAULT_MIN_SIZE,
    ID_TO_NAME,
    ID_TO_NAME_ZH,
    class_name,
)
from .model import load_checkpoint, resolve_checkpoint


def to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB image -> float tensor in ``[0, 1]`` with shape ``(3, H, W)``.

    No normalisation here: the detector's own ``GeneralizedRCNNTransform``
    applies the ImageNet statistics.
    """
    # np.array (not asarray) copies, so the tensor is writable.
    array = np.array(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


class Detector:
    """Thin wrapper around a Faster R-CNN checkpoint."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        arch: str = "fasterrcnn_resnet50_fpn",
        device: Optional[str] = None,
        min_size: int = DEFAULT_MIN_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        score_threshold: float = 0.5,
    ) -> None:
        path = resolve_checkpoint(checkpoint)
        if path is None:
            raise FileNotFoundError(
                "No checkpoint found. Train one with `python -m src.train` or pass "
                "--checkpoint /path/to/model.pth"
            )
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint = path
        self.score_threshold = score_threshold
        self.model = load_checkpoint(
            path,
            arch=arch,
            device=self.device,
            min_size=min_size,
            max_size=max_size,
        )

    def predict(self, image: Image.Image) -> List[Dict]:
        """Run detection on one PIL image and return a list of detections."""
        tensor = to_tensor(image).to(self.device)
        with torch.inference_mode():
            output = self.model([tensor])[0]

        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()

        detections: List[Dict] = []
        for box, label, score in zip(boxes, labels, scores):
            score = float(score)
            if score < self.score_threshold:
                continue
            label = int(label)
            detections.append(
                {
                    "box": [round(float(v), 2) for v in box],
                    "label": label,
                    "label_name": class_name(label),
                    "label_zh": class_name(label, chinese=True),
                    "score": round(score, 4),
                }
            )
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

    def predict_file(self, path: str) -> List[Dict]:
        return self.predict(Image.open(path))

    def raw_predict(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """Untouched model output (used for AP computation, no thresholding)."""
        tensor = to_tensor(image).to(self.device)
        with torch.inference_mode():
            output = self.model([tensor])[0]
        return {
            "boxes": output["boxes"].detach().cpu().numpy(),
            "labels": output["labels"].detach().cpu().numpy(),
            "scores": output["scores"].detach().cpu().numpy(),
        }

    def describe(self) -> str:
        return (
            f"checkpoint={self.checkpoint} device={self.device} "
            f"classes={len(ID_TO_NAME)}"
        )
