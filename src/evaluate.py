"""Evaluate a checkpoint on a split and write a machine-readable report.

    python -m src.evaluate --checkpoint saved_models/model.pth

Outputs ``metrics.json`` (mAP, per-class AP, PR curves, operating point) and,
optionally, ``predictions.json`` with every raw detection.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
from PIL import Image

from .config import (
    DEFAULT_MAX_SIZE,
    DEFAULT_MIN_SIZE,
    DEFAULT_TEST_JSON,
    FULL_IMAGE_DIR,
    ID_TO_NAME,
    REPO_ROOT,
    SAMPLE_IMAGE_DIR,
)
from .dataset import index_annotations, load_coco
from .inference import Detector
from .metrics import evaluate_detections, operating_point


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a Faster R-CNN checkpoint (COCO-style metrics)."
    )
    parser.add_argument("--checkpoint", default=None, help="path to a .pth checkpoint")
    parser.add_argument("--arch", default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--annotation-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-dir", default=None, help="defaults to data/images")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "eval"))
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="confidence cut-off for the precision/recall/F1 report")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def resolve_image_dir(explicit, annotation_file):
    if explicit:
        return explicit
    if os.path.isdir(SAMPLE_IMAGE_DIR) and os.path.basename(annotation_file) == "annotations.json":
        return SAMPLE_IMAGE_DIR
    return FULL_IMAGE_DIR


def main(argv=None) -> int:
    args = parse_args(argv)
    image_dir = resolve_image_dir(args.image_dir, args.annotation_file)
    if not os.path.isdir(image_dir):
        raise SystemExit(f"Image directory not found: {image_dir}")

    coco = load_coco(args.annotation_file)
    grouped = index_annotations(coco)
    images = coco["images"]
    if args.limit:
        images = images[: args.limit]

    detector = Detector(
        checkpoint=args.checkpoint,
        arch=args.arch,
        device=args.device,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    if not args.quiet:
        print(f"[eval] {detector.describe()}")
        print(f"[eval] {len(images)} images from {args.annotation_file}")

    ground_truth: Dict[int, Dict] = {}
    predictions: Dict[int, Dict] = {}
    started = time.time()

    for index, image_info in enumerate(images, start=1):
        path = os.path.join(image_dir, image_info["file_name"])
        if not os.path.exists(path):
            continue
        anns = grouped.get(image_info["id"], [])
        boxes = np.asarray(
            [
                [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in anns
            ],
            dtype=np.float32,
        ).reshape(-1, 4)
        labels = np.asarray([int(a["category_id"]) for a in anns], dtype=np.int64)
        ground_truth[image_info["id"]] = {"boxes": boxes, "labels": labels}

        predictions[image_info["id"]] = detector.raw_predict(Image.open(path))

        if not args.quiet and (index % 25 == 0 or index == len(images)):
            elapsed = time.time() - started
            per_image = elapsed / index
            remaining = per_image * (len(images) - index)
            print(
                f"[eval] {index}/{len(images)} images | "
                f"{per_image * 1000:.0f} ms/img | ETA {remaining / 60:.1f} min",
                flush=True,
            )

    results = evaluate_detections(ground_truth, predictions)
    results["operating_point"] = operating_point(
        ground_truth, predictions, score_threshold=args.score_threshold
    )
    results["checkpoint"] = detector.checkpoint
    results["annotation_file"] = os.path.abspath(args.annotation_file)
    results["image_dir"] = os.path.abspath(image_dir)
    results["inference_seconds"] = round(time.time() - started, 2)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    if args.save_predictions:
        serialisable = {
            str(image_id): {
                "boxes": pred["boxes"].round(2).tolist(),
                "labels": pred["labels"].tolist(),
                "scores": pred["scores"].round(4).tolist(),
            }
            for image_id, pred in predictions.items()
        }
        with open(os.path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as handle:
            json.dump(serialisable, handle)

    op = results["operating_point"]
    print("\n=== Detection metrics ===")
    print(f"mAP@[.50:.95] : {results['mAP']:.4f}")
    print(f"mAP@0.50      : {results['mAP50']:.4f}")
    print(f"mAP@0.75      : {results['mAP75']:.4f}")
    print(f"P / R / F1 @ {args.score_threshold}: "
          f"{op['precision']:.4f} / {op['recall']:.4f} / {op['f1']:.4f}")
    print(f"image-level exact-match accuracy: {op['image_exact_match_accuracy']:.4f}")
    print("\nper-class AP@0.50")
    for name, stats in results["per_class"].items():
        print(f"  {name:<16} AP50={stats['ap50']:.4f}  AP={stats['ap']:.4f}  n_gt={stats['num_gt']}")
    print(f"\nmetrics written to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
