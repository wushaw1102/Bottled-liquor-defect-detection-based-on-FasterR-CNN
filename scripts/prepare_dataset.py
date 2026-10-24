"""Split a COCO annotation file into train / val / test and sanity-check it.

    python scripts/prepare_dataset.py --annotation-file raw/annotations.json \
        --image-dir raw/images --output-dir data/annotations --train 0.7 --val 0.2

The original ``train_model.py`` shipped a splitter that wrote
``"annotations": data['annotations']`` into **every** split, i.e. the validation
and test files referenced all 6,945 annotations while holding only a third of the
images.  Anything evaluated on those files was silently scored against the
training labels.  This script keeps annotations and images together and refuses
to write a split that contains an annotation for an image it does not own.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List

CLASS_NAMES = {
    0: "background",
    1: "cap_cracked",
    2: "cap_deformed",
    3: "cap_edge_broken",
    4: "cap_swirl",
    5: "cap_dot_missing",
    6: "label_skew",
    7: "label_wrinkled",
    8: "label_bubble",
    9: "code_normal",
    10: "code_defect",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create reproducible dataset splits.")
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default=os.path.join("data", "annotations"))
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-background", action="store_true",
                        help="remove annotations whose category_id is 0 (the reserved "
                             "background slot) - recommended when training from scratch")
    parser.add_argument("--verify-only", action="store_true",
                        help="only report statistics, do not write files")
    return parser.parse_args(argv)


def verify(coco: Dict, image_dir: str, name: str) -> Dict[str, int]:
    """Report missing files, orphan annotations and the per-class histogram."""
    image_ids = {img["id"] for img in coco["images"]}
    missing = [
        img["file_name"]
        for img in coco["images"]
        if not os.path.exists(os.path.join(image_dir, img["file_name"]))
    ]
    orphans = [ann for ann in coco["annotations"] if ann["image_id"] not in image_ids]
    histogram = Counter(int(ann["category_id"]) for ann in coco["annotations"])

    print(f"\n[{name}] {len(image_ids)} images | {len(coco['annotations'])} annotations")
    if missing:
        print(f"  !! {len(missing)} images missing from {image_dir} (e.g. {missing[0]})")
    if orphans:
        print(f"  !! {len(orphans)} annotations point at images that are not in this split")
    for class_id in sorted(CLASS_NAMES):
        print(f"    {class_id:>2} {CLASS_NAMES[class_id]:<16} {histogram.get(class_id, 0):>5}")
    return {"images": len(image_ids), "annotations": len(coco["annotations"]),
            "missing": len(missing), "orphans": len(orphans)}


def split(coco: Dict, train_ratio: float, val_ratio: float, seed: int):
    images = list(coco["images"])
    random.Random(seed).shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    buckets = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    splits = {}
    for name, split_images in buckets.items():
        keep_ids = {img["id"] for img in split_images}
        split_annotations = [
            ann for image_id in keep_ids for ann in annotations_by_image.get(image_id, [])
        ]
        splits[name] = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": split_images,
            "annotations": split_annotations,
            "categories": coco["categories"],
        }
    return splits


def main(argv=None) -> int:
    args = parse_args(argv)
    with open(args.annotation_file, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    if args.drop_background:
        before = len(coco["annotations"])
        coco["annotations"] = [a for a in coco["annotations"] if int(a["category_id"]) != 0]
        print(f"[prepare] dropped {before - len(coco['annotations'])} background annotations")

    if args.verify_only:
        verify(coco, args.image_dir, "full")
        return 0

    splits = split(coco, args.train, args.val, args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    total_images = len(coco["images"])
    for name, data in splits.items():
        stats = verify(data, args.image_dir, name)
        if stats["missing"]:
            print(f"  -> refusing to write {name}: images are missing on disk")
            continue
        if stats["orphans"]:
            print(f"  -> refusing to write {name}: split contains orphan annotations")
            continue
        path = os.path.join(args.output_dir, f"instances_{name}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False)
        share = 100 * len(data["images"]) / total_images
        print(f"  -> {path}  ({len(data['images'])} images, {share:.1f}%)")

    print("\n[prepare] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
