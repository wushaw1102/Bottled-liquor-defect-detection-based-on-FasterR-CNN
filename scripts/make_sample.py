"""Build a small, class-balanced sample of the dataset for demos and smoke tests.

    python scripts/make_sample.py --image-dir data/images \
        --annotation-file data/annotations/instances_test.json --count 60

Writes ``data/sample/images/*.jpg`` plus ``data/sample/annotations.json``.  The
full image set (4,516 files, ~260 MB) is intentionally not committed - see
``data/README.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict

CLASS_NAMES = {
    0: "background", 1: "cap_cracked", 2: "cap_deformed", 3: "cap_edge_broken",
    4: "cap_swirl", 5: "cap_dot_missing", 6: "label_skew", 7: "label_wrinkled",
    8: "label_bubble", 9: "code_normal", 10: "code_defect",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create the committed sample split.")
    parser.add_argument("--image-dir", default=os.path.join("data", "images"))
    parser.add_argument("--annotation-file", default=os.path.join("data", "annotations", "instances_test.json"))
    parser.add_argument("--output-dir", default=os.path.join("data", "sample"))
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--per-class", type=int, default=4, help="minimum images per class")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    with open(args.annotation_file, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    rng = random.Random(args.seed)
    images = list(coco["images"])
    rng.shuffle(images)

    coverage = defaultdict(int)
    chosen = []
    # Pass 1: make sure every class is represented at least --per-class times.
    for image in images:
        if len(chosen) >= args.count:
            break
        classes = {int(a["category_id"]) for a in annotations_by_image.get(image["id"], [])}
        if not classes:
            continue
        needs_class = any(coverage[c] < args.per_class for c in classes)
        if needs_class:
            chosen.append(image)
            for c in classes:
                coverage[c] += 1

    # Pass 2: top up to --count with random images.
    for image in images:
        if len(chosen) >= args.count:
            break
        if image not in chosen and annotations_by_image.get(image["id"]):
            chosen.append(image)

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    copied = []
    for image in chosen:
        source = os.path.join(args.image_dir, image["file_name"])
        if not os.path.exists(source):
            continue
        shutil.copy2(source, os.path.join(args.output_dir, "images", image["file_name"]))
        copied.append(image)

    keep_ids = {image["id"] for image in copied}
    sample = {
        "info": {"description": "Bottled-liquor defect detection - demo sample"},
        "images": copied,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep_ids],
        "categories": coco["categories"],
    }
    with open(os.path.join(args.output_dir, "annotations.json"), "w", encoding="utf-8") as handle:
        json.dump(sample, handle, ensure_ascii=False, indent=1)

    total_bytes = sum(
        os.path.getsize(os.path.join(args.output_dir, "images", image["file_name"]))
        for image in copied
    )
    print(f"[sample] {len(copied)} images, {len(sample['annotations'])} annotations, "
          f"{total_bytes / 1e6:.1f} MB")
    print("[sample] per class:", ", ".join(
        f"{CLASS_NAMES[c]}={coverage.get(c, 0)}" for c in sorted(CLASS_NAMES) if c
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
