"""Run detection on one image or a directory and save annotated results.

    python -m src.predict --image data/sample/images/img_0000234.jpg
    python -m src.predict --input-dir data/sample/images --output-dir outputs/pred --threshold 0.4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List

from PIL import Image

from .config import DEFAULT_CHECKPOINTS, REPO_ROOT
from .inference import Detector
from .visualize import draw_detections

SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Detect bottle defects in images.")
    parser.add_argument("--checkpoint", default=None,
                        help=f"default: first of {DEFAULT_CHECKPOINTS}")
    parser.add_argument("--arch", default="fasterrcnn_resnet50_fpn")
    parser.add_argument("--image", default=None, help="path to a single image")
    parser.add_argument("--input-dir", default=None, help="path to a folder of images")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "predictions"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chinese", action="store_true", help="draw Chinese class names")
    parser.add_argument("--no-save-image", action="store_true")
    parser.add_argument("--json", default=None, help="write detections to this JSON file")
    return parser.parse_args(argv)


def collect_images(args) -> List[str]:
    if args.image:
        return [args.image]
    if args.input_dir:
        return sorted(
            os.path.join(args.input_dir, name)
            for name in os.listdir(args.input_dir)
            if name.lower().endswith(SUPPORTED_SUFFIXES)
        )
    raise SystemExit("Provide --image or --input-dir")


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = collect_images(args)
    if not paths:
        raise SystemExit("No images found.")

    detector = Detector(
        checkpoint=args.checkpoint,
        arch=args.arch,
        device=args.device,
        min_size=args.min_size,
        max_size=args.max_size,
        score_threshold=args.threshold,
    )
    print(detector.describe())

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    for path in paths:
        image = Image.open(path).convert("RGB")
        detections = detector.predict(image)
        results[os.path.basename(path)] = detections

        summary = ", ".join(
            f"{d['label_name']}({d['score']:.2f})" for d in detections[:4]
        ) or "nothing above threshold"
        print(f"{os.path.basename(path):<28} {len(detections):>3} detections | {summary}")

        if not args.no_save_image:
            draw_detections(image, detections, chinese=args.chinese).save(
                os.path.join(args.output_dir, f"pred_{os.path.basename(path)}")
            )

    json_path = args.json or os.path.join(args.output_dir, "detections.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.output_dir, "detections.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "label_id", "label", "label_zh", "score", "x1", "y1", "x2", "y2"])
        for file_name, detections in results.items():
            for detection in detections:
                writer.writerow(
                    [file_name, detection["label"], detection["label_name"],
                     detection["label_zh"], detection["score"], *detection["box"]]
                )

    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")
    if not args.no_save_image:
        print(f"annotated images in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
