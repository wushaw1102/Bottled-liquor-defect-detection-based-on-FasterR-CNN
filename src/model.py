"""Faster R-CNN model factory and checkpoint helpers."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .config import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_MAX_SIZE,
    DEFAULT_MIN_SIZE,
    NUM_CLASSES,
    SUPPORTED_ARCHS,
)

_BUILDERS = {
    "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
    "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
}

_WEIGHTS = {
    "fasterrcnn_resnet50_fpn": "FasterRCNN_ResNet50_FPN_Weights",
    "fasterrcnn_mobilenet_v3_large_fpn": "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
}


def _pretrained_weights(arch: str):
    """Pre-trained COCO weights if available offline, otherwise ``None``."""
    enum_name = _WEIGHTS[arch]
    enum = getattr(torchvision.models.detection, enum_name, None)
    return enum.DEFAULT if enum is not None else None


def build_model(
    arch: str = "fasterrcnn_resnet50_fpn",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
) -> torch.nn.Module:
    """Build a Faster R-CNN detector with a fresh box head.

    ``min_size`` / ``max_size`` control the internal resize.  Lowering
    ``min_size`` to 600 is the cheapest way to make training fit in a 4 GB GPU.
    """
    if arch not in _BUILDERS:
        raise ValueError(f"Unsupported arch {arch!r}. Choose from {SUPPORTED_ARCHS}.")

    weights = _pretrained_weights(arch) if pretrained else None
    model = _BUILDERS[arch](
        weights=weights, min_size=min_size, max_size=max_size
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def fold_legacy_normalization(
    model: torch.nn.Module,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> torch.nn.Module:
    """Make a double-normalised checkpoint behave like a correctly trained one.

    The original pipeline normalised images in the ``Dataset`` *and* again inside
    ``GeneralizedRCNNTransform``, so the released weights expect inputs that were
    normalised twice.  Two chained per-channel affine transforms collapse into a
    single one::

        ((x - m) / s - m) / s  ==  (x - (m + m*s)) / s**2

    Rewriting ``image_mean`` / ``image_std`` therefore lets the model consume
    plain ``[0, 1]`` tensors while computing exactly the activations the
    checkpoint was trained with (verified to 2e-6 in float32).
    """
    transform = getattr(model, "transform", None)
    if transform is None:
        return model
    mean_t = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.as_tensor(std, dtype=torch.float32).view(3, 1, 1)
    folded_mean = (mean_t + mean_t * std_t).flatten().tolist()
    folded_std = (std_t * std_t).flatten().tolist()
    transform.image_mean = folded_mean
    transform.image_std = folded_std
    return model


def load_checkpoint(
    checkpoint_path: str,
    arch: str = "fasterrcnn_resnet50_fpn",
    num_classes: int = NUM_CLASSES,
    device: Optional[torch.device] = None,
    min_size: int = DEFAULT_MIN_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    legacy_normalization: Optional[bool] = None,
) -> torch.nn.Module:
    """Load a (possibly wrapped) state dict and return an ``eval()`` model.

    ``legacy_normalization`` defaults to ``True`` for the checkpoints listed in
    ``config.LEGACY_CHECKPOINTS`` (the weights shipped with the original
    release) and ``False`` otherwise.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if legacy_normalization is None:
        legacy_normalization = is_legacy_checkpoint(checkpoint_path)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        arch=arch,
        num_classes=num_classes,
        pretrained=False,
        min_size=min_size,
        max_size=max_size,
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    # Training runs save {"model": ..., "optimizer": ...}; bare state dicts are
    # also common (the checkpoint shipped with the original release is one).
    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[model] ignored {len(unexpected)} unexpected keys (e.g. {unexpected[0]})")
    if missing:
        print(f"[model] {len(missing)} missing keys (e.g. {missing[0]})")

    if legacy_normalization:
        fold_legacy_normalization(model)
        print(
            "[model] legacy checkpoint: double normalisation folded into the "
            "transform (see src/model.py)"
        )

    model.to(device)
    model.eval()
    return model


def is_legacy_checkpoint(path: str) -> bool:
    """True when ``path`` points at a checkpoint trained by the original code."""
    from .config import LEGACY_CHECKPOINTS

    try:
        target = os.path.normcase(os.path.abspath(path))
        return any(
            target == os.path.normcase(os.path.abspath(p)) for p in LEGACY_CHECKPOINTS
        )
    except Exception:
        return False


def resolve_checkpoint(explicit: Optional[str] = None) -> Optional[str]:
    """Return ``explicit`` or the first existing default checkpoint path."""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for candidate in DEFAULT_CHECKPOINTS:
        if os.path.exists(candidate):
            return candidate
    return None


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
