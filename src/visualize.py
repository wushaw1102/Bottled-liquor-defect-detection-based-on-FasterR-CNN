"""Drawing helpers: annotated detections and result charts.

    python -m src.visualize --metrics outputs/eval_checkpoint/metrics.json \
        --history outputs/fasterrcnn_resnet50_fpn/history.json --output-dir docs/images
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from .config import (  # noqa: E402
    ID_TO_NAME,
    ID_TO_NAME_ZH,
    REPO_ROOT,
    class_name,
)

# A palette with enough contrast on both bright and dark bottle surfaces.
PALETTE = [
    (230, 57, 70),    # 1 red
    (29, 53, 87),     # 2 navy
    (42, 157, 143),   # 3 teal
    (233, 196, 106),  # 4 amber
    (144, 12, 63),    # 5 wine
    (69, 123, 157),   # 6 steel blue
    (168, 218, 220),  # 7 pale cyan
    (255, 107, 107),  # 8 coral
    (38, 70, 83),     # 9 dark slate
    (244, 162, 97),   # 10 orange
]


def class_color(label_id: int) -> Tuple[int, int, int]:
    return PALETTE[(int(label_id) - 1) % len(PALETTE)]


_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size: int = 18):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: Sequence[Dict],
    ground_truth: Optional[Sequence[Dict]] = None,
    chinese: bool = False,
    show_labels: bool = True,
    font_size: int = 18,
) -> Image.Image:
    """Return a copy of ``image`` with predicted (and optional GT) boxes.

    ``detections`` items need ``box`` (x1, y1, x2, y2), ``label`` and ``score``.
    ``ground_truth`` items need ``box`` and ``label`` and are drawn dashed-looking
    in grey so a prediction can be compared with the annotation.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_size)
    placed: List[Tuple[int, int, int, int]] = []  # label rectangles already drawn

    def place_label(x1: int, y1: int, x2: int, y2: int, width: int, height: int):
        """Pick a spot for the caption that does not cover an earlier caption."""
        candidates = [
            (x1, y1 - height - 6),           # above the box
            (x1, y2 - height - 4),           # inside, bottom edge
            (x1, y2 + 4),                    # just below the box
            (x1, y1 + 4),                    # inside, top edge
        ]
        for left, top in candidates:
            rect = (left, top, left + width + 8, top + height + 6)
            clash = any(
                not (rect[2] < other[0] or rect[0] > other[2]
                     or rect[3] < other[1] or rect[1] > other[3])
                for other in placed
            )
            if not clash:
                placed.append(rect)
                return left, top
        placed.append((x1, y1 - height - 6, x1 + width + 8, y1))
        return x1, y1 - height - 6

    if ground_truth:
        for truth in ground_truth:
            x1, y1, x2, y2 = truth["box"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)
            if show_labels:
                name = (ID_TO_NAME_ZH if chinese else ID_TO_NAME).get(int(truth["label"]), "gt")
                text = f"GT {name}"
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                text_y = min(y2 - (bottom - top) - 4, y2)
                label_rect = (
                    x1, max(text_y - 2, y1),
                    x1 + (right - left) + 8, min(text_y + (bottom - top) + 4, y2),
                )
                placed.append(label_rect)
                draw.rectangle(list(label_rect), fill=(255, 255, 255))
                draw.text((x1 + 4, text_y), text, fill=(20, 20, 20), font=font)

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        color = class_color(detection["label"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if not show_labels:
            continue
        name = class_name(detection["label"], chinese=chinese)
        text = f"{name} {detection['score']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = right - left, bottom - top
        label_x, text_y = place_label(
            int(x1), int(y1), int(x2), int(y2), text_width, text_height
        )
        text_y = max(text_y, 0)
        draw.rectangle(
            [label_x, text_y, label_x + text_width + 8, text_y + text_height + 6], fill=color
        )
        draw.text((label_x + 4, text_y + 2), text, fill=(255, 255, 255), font=font)

    return canvas


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_training_curves(history_path: str, output_path: str) -> Optional[str]:
    with open(history_path, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    if not history:
        return None

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row.get("val_loss") for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, train_loss, marker="o", label="train loss", color="#2a9d8f")
    if all(v is not None for v in val_loss):
        axes[0].plot(epochs, val_loss, marker="s", label="val loss", color="#e76f51")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training / validation loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["lr"] for row in history], marker="o", color="#264653")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("Learning-rate schedule")
    axes[1].grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_per_class_ap(metrics_path: str, output_path: str) -> Optional[str]:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    names = [name for name in metrics["per_class"] if metrics["per_class"][name]["num_gt"] > 0]
    if not names:
        return None
    values = [metrics["per_class"][name]["ap50"] for name in names]
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    figure, axis = plt.subplots(figsize=(8, 4.6))
    bars = axis.barh(names, values, color="#2a9d8f")
    for bar, value in zip(bars, values):
        axis.text(
            min(value + 0.01, 1.0),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("AP@0.50")
    axis.set_title(f"Per-class AP@0.50  (mAP@0.50 = {metrics['mAP50']:.3f})")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_pr_curves(metrics_path: str, output_path: str, max_classes: int = 8) -> Optional[str]:
    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    curves = metrics.get("pr_curves") or {}
    if not curves:
        return None

    # Show the classes with the most ground-truth instances; low-support classes
    # produce jagged, uninformative curves.
    ranked = sorted(
        curves.items(),
        key=lambda item: metrics["per_class"].get(item[0], {}).get("num_gt", 0),
        reverse=True,
    )[:max_classes]

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for name, curve in ranked:
        recall = curve["recall"]
        precision = curve["precision"]
        if not recall:
            continue
        ap = metrics["per_class"][name]["ap50"]
        axis.plot(recall, precision, linewidth=1.6, label=f"{name} (AP={ap:.2f})")
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_xlim(0, 1.0)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="lower left")
    axis.set_title("Precision-recall curves at IoU 0.50")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_gallery(
    detector,
    image_dir: str,
    annotation_file: str,
    output_dir: str,
    count: int = 6,
    seed: int = 7,
    chinese: bool = False,
) -> List[str]:
    """Render a handful of random test images with GT + predictions side by side."""
    from .dataset import index_annotations, load_coco

    coco = load_coco(annotation_file)
    grouped = index_annotations(coco)
    candidates = [img for img in coco["images"] if os.path.exists(os.path.join(image_dir, img["file_name"]))]
    random.Random(seed).shuffle(candidates)

    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []
    for image_info in candidates:
        if len(written) >= count:
            break
        image = Image.open(os.path.join(image_dir, image_info["file_name"])).convert("RGB")
        detections = detector.predict(image)
        if not detections:
            continue  # a gallery of empty frames is not useful
        truth = [
            {
                "box": [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]],
                "label": a["category_id"],
            }
            for a in grouped.get(image_info["id"], [])
        ]
        annotated = draw_detections(image, detections, ground_truth=truth, chinese=chinese)
        path = os.path.join(output_dir, f"prediction_{image_info['file_name']}")
        annotated.save(path)
        written.append(path)
    print(f"[visualize] wrote {len(written)} gallery images to {output_dir}")
    return written


def make_montage(paths: Sequence[str], output_path: str, cols: int = 3,
                 cell: Tuple[int, int] = (430, 322), title_height: int = 26) -> Optional[str]:
    """Combine rendered predictions into one grid image for the README."""
    if not paths:
        return None

    rows = (len(paths) + cols - 1) // cols
    cell_width, cell_height = cell
    font = load_font(15)
    sheet = Image.new(
        "RGB", (cols * cell_width, rows * (cell_height + title_height)), (255, 255, 255)
    )
    draw = ImageDraw.Draw(sheet)

    for index, path in enumerate(paths):
        row, column = divmod(index, cols)
        box = Image.open(path).convert("RGB")
        box.thumbnail((cell_width, cell_height))
        offset_x = column * cell_width + (cell_width - box.width) // 2
        offset_y = row * (cell_height + title_height) + title_height + (cell_height - box.height) // 2
        sheet.paste(box, (offset_x, offset_y))
        draw.text(
            (column * cell_width + 8, row * (cell_height + title_height) + 5),
            os.path.basename(path).replace("prediction_", ""),
            fill=(70, 70, 70),
            font=font,
        )

    sheet.save(output_path)
    print(f"[visualize] montage -> {output_path}")
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Render result figures.")
    parser.add_argument("--metrics", default=os.path.join(REPO_ROOT, "outputs", "eval_checkpoint", "metrics.json"))
    parser.add_argument("--history", default=None)
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "docs", "images"))
    parser.add_argument("--checkpoint", default=None, help="also render a prediction gallery")
    parser.add_argument("--annotation-file", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--gallery-count", type=int, default=6)
    parser.add_argument("--montage", action="store_true", help="also build a grid image")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.metrics):
        print(plot_per_class_ap(args.metrics, os.path.join(args.output_dir, "per_class_ap.png")))
        print(plot_pr_curves(args.metrics, os.path.join(args.output_dir, "pr_curves.png")))
    if args.history and os.path.exists(args.history):
        print(plot_training_curves(args.history, os.path.join(args.output_dir, "training_curves.png")))

    if args.checkpoint:
        from .config import DEFAULT_TEST_JSON, FULL_IMAGE_DIR
        from .inference import Detector

        detector = Detector(checkpoint=args.checkpoint)
        gallery_dir = os.path.join(args.output_dir, "gallery")
        paths = save_gallery(
            detector,
            args.image_dir or FULL_IMAGE_DIR,
            args.annotation_file or DEFAULT_TEST_JSON,
            gallery_dir,
            count=args.gallery_count,
        )
        if args.montage and paths:
            make_montage(paths, os.path.join(args.output_dir, "prediction_grid.png"),
                         cols=2 if len(paths) <= 4 else 3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
