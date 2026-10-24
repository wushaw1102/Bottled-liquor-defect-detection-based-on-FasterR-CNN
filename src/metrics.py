"""Dependency-free COCO-style detection metrics.

``pycocotools`` is the reference implementation but needs a C compiler on
Windows, which is exactly the kind of thing that stops a newcomer from
reproducing a project.  This module re-implements the parts we report:

* per-class AP with 101-point interpolation (the COCO definition),
* mAP@0.5, mAP@0.75 and mAP@[.5:.95],
* precision / recall / F1 at a fixed confidence threshold,
* the recall-precision pairs needed to draw PR curves.

Greedy matching is used (highest-scoring unmatched detection takes the
highest-IoU unmatched ground truth of the same class), which mirrors what
``COCOeval`` does for a single IoU threshold.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .config import EVAL_CLASS_IDS, ID_TO_NAME

#: COCO keeps at most 100 detections per image when scoring.
MAX_DETECTIONS = 100
#: The 101 recall levels used by the COCO AP definition.
RECALL_LEVELS = np.linspace(0.0, 1.0, 101)


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``boxes_a`` and every box in ``boxes_b``.

    Boxes are ``[x1, y1, x2, y2]`` in pixels.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clip(0) * (boxes_a[:, 3] - boxes_a[:, 1]).clip(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clip(0) * (boxes_b[:, 3] - boxes_b[:, 1]).clip(0)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _match_image(
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float,
) -> List[Dict]:
    """Greedy one-to-one matching for a single image."""
    matches: List[Dict] = []
    if len(pred_boxes) == 0:
        return matches

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]
    pred_scores = pred_scores[order][:MAX_DETECTIONS]
    pred_boxes = pred_boxes[:MAX_DETECTIONS]
    pred_labels = pred_labels[:MAX_DETECTIONS]

    iou = box_iou_matrix(pred_boxes, gt_boxes)
    used = np.zeros(len(gt_boxes), dtype=bool)

    for det_index in range(len(pred_boxes)):
        best_iou, best_gt = 0.0, -1
        for gt_index in range(len(gt_boxes)):
            if used[gt_index] or gt_labels[gt_index] != pred_labels[det_index]:
                continue
            if iou[det_index, gt_index] > best_iou:
                best_iou = float(iou[det_index, gt_index])
                best_gt = gt_index
        is_true_positive = best_iou >= iou_threshold and best_gt >= 0
        if is_true_positive:
            used[best_gt] = True
        matches.append(
            {
                "class_id": int(pred_labels[det_index]),
                "score": float(pred_scores[det_index]),
                "iou": best_iou,
                "tp": is_true_positive,
            }
        )
    return matches


def _average_precision(tp: np.ndarray, fp: np.ndarray, num_gt: int):
    """101-point interpolated AP plus the raw PR points (for plotting)."""
    if num_gt == 0:
        return float("nan"), np.array([]), np.array([])

    tp_cum = np.cumsum(tp).astype(np.float64)
    fp_cum = np.cumsum(fp).astype(np.float64)
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Make precision monotonically decreasing, as COCO does.
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]

    indices = np.searchsorted(recall, RECALL_LEVELS, side="left")
    sampled = np.zeros(len(RECALL_LEVELS), dtype=np.float64)
    for i, idx in enumerate(indices):
        if idx < len(precision_envelope):
            sampled[i] = precision_envelope[idx]
    return float(sampled.mean()), recall, precision


def evaluate_detections(
    ground_truth: Dict[int, Dict],
    predictions: Dict[int, Dict],
    iou_thresholds: Sequence[float] = tuple(np.arange(0.5, 1.0, 0.05)),
    class_ids: Sequence[int] = tuple(EVAL_CLASS_IDS),
    collect_curves: bool = True,
) -> Dict:
    """Score predictions against ground truth.

    Parameters
    ----------
    ground_truth:
        ``{image_id: {"boxes": ndarray (N,4), "labels": ndarray (N,)}}``.
    predictions:
        ``{image_id: {"boxes": ndarray (M,4), "labels": ndarray (M,),
        "scores": ndarray (M,)}}``.  Missing image ids are treated as "no
        detection".
    """
    class_ids = [int(c) for c in class_ids]
    # np.arange drifts (0.75 -> 0.7500000000000001); normalise so lookups work.
    iou_thresholds = [round(float(t), 2) for t in iou_thresholds]
    gt_per_class = {c: 0 for c in class_ids}
    for data in ground_truth.values():
        for label in data["labels"]:
            label = int(label)
            if label in gt_per_class:
                gt_per_class[label] += 1

    per_threshold: Dict[float, Dict[int, Dict[str, np.ndarray]]] = {}
    pr_curves: Dict[int, Dict[str, List[float]]] = {}
    for threshold in iou_thresholds:
        per_class = {c: {"tp": [], "fp": [], "score": []} for c in class_ids}
        for image_id, gt in ground_truth.items():
            pred = predictions.get(image_id)
            if pred is None:
                continue
            matches = _match_image(
                gt["boxes"],
                gt["labels"],
                pred["boxes"],
                pred["labels"],
                pred["scores"],
                float(threshold),
            )
            for match in matches:
                entry = per_class.get(match["class_id"])
                if entry is None:
                    continue  # background slot or unknown class
                entry["tp"].append(1.0 if match["tp"] else 0.0)
                entry["fp"].append(0.0 if match["tp"] else 1.0)
                entry["score"].append(match["score"])

        scored: Dict[int, Dict[str, float]] = {}
        curves: Dict[int, Dict[str, List[float]]] = {}
        for class_id, entry in per_class.items():
            if not entry["score"]:
                # COCO counts a class that has ground truth but never gets a
                # detection as AP = 0; a class absent from the split is skipped.
                ap = 0.0 if gt_per_class[class_id] > 0 else float("nan")
                scored[class_id] = {"ap": ap, "num_gt": gt_per_class[class_id],
                                    "no_predictions": True}
                continue
            order = np.argsort(-np.asarray(entry["score"]))
            tp = np.asarray(entry["tp"])[order]
            fp = np.asarray(entry["fp"])[order]
            ap, recall, precision = _average_precision(tp, fp, gt_per_class[class_id])
            scored[class_id] = {"ap": ap, "num_gt": gt_per_class[class_id]}
            if collect_curves and threshold == 0.5 and len(recall):
                curves[class_id] = {
                    "recall": recall.round(6).tolist(),
                    "precision": precision.round(6).tolist(),
                }
        per_threshold[round(float(threshold), 2)] = scored
        if collect_curves and threshold == 0.5:
            pr_curves = curves

    def mean_ap(threshold: float, class_id: int) -> float:
        return per_threshold[threshold][class_id]["ap"]

    per_class = {}
    for class_id in class_ids:
        per_threshold_ap = [mean_ap(t, class_id) for t in iou_thresholds]
        finite = [v for v in per_threshold_ap if not np.isnan(v)]
        per_class[ID_TO_NAME.get(class_id, str(class_id))] = {
            "class_id": class_id,
            "num_gt": gt_per_class[class_id],
            "ap50": mean_ap(0.5, class_id),
            "ap75": mean_ap(0.75, class_id),
            "ap": float(np.mean(finite)) if finite else float("nan"),
            "no_predictions": bool(
                per_threshold.get(0.5, {}).get(class_id, {}).get("no_predictions", False)
            ),
        }

    def nanmean(values):
        values = [v for v in values if not np.isnan(v)]
        return float(np.mean(values)) if values else float("nan")

    return {
        "mAP50": nanmean([mean_ap(0.5, c) for c in class_ids]),
        "mAP75": nanmean([mean_ap(0.75, c) for c in class_ids]),
        "mAP": nanmean(
            [mean_ap(t, c) for t in iou_thresholds for c in class_ids]
        ),
        "per_class": per_class,
        "pr_curves": {
            ID_TO_NAME.get(c, str(c)): pr_curves[c] for c in pr_curves
        }
        if collect_curves
        else {},
        "num_gt": int(sum(gt_per_class.values())),
        "num_images": len(ground_truth),
        "iou_thresholds": [float(t) for t in iou_thresholds],
    }


def operating_point(
    ground_truth: Dict[int, Dict],
    predictions: Dict[int, Dict],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    class_ids: Sequence[int] = tuple(EVAL_CLASS_IDS),
) -> Dict:
    """Precision / recall / F1 at a fixed confidence cut-off.

    Also reports a *per-image* view: an image counts as correct when the set of
    predicted classes matches the set of ground-truth classes (a practical
    "did we catch every defect on this bottle" metric).
    """
    class_ids = set(int(c) for c in class_ids)
    tp = fp = fn = 0
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in class_ids}
    images_total = 0
    images_exact = 0

    for image_id, gt in ground_truth.items():
        pred = predictions.get(image_id, {"boxes": np.zeros((0, 4)), "labels": np.zeros(0), "scores": np.zeros(0)})
        keep = pred["scores"] >= score_threshold
        pred_boxes = pred["boxes"][keep]
        pred_labels = pred["labels"][keep]
        pred_scores = pred["scores"][keep]

        gt_boxes, gt_labels = gt["boxes"], gt["labels"]
        gt_kept = np.array([int(l) in class_ids for l in gt_labels], dtype=bool)
        gt_boxes, gt_labels = gt_boxes[gt_kept], gt_labels[gt_kept]
        pred_kept = np.array([int(l) in class_ids for l in pred_labels], dtype=bool)
        pred_boxes, pred_labels = pred_boxes[pred_kept], pred_labels[pred_kept]

        images_total += 1
        gt_set = set(int(l) for l in gt_labels)
        pred_set = set(int(l) for l in pred_labels)
        if gt_set == pred_set:
            images_exact += 1

        matches = _match_image(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold)
        matched_gt = 0
        for match in matches:
            entry = per_class.get(match["class_id"])
            if entry is None:
                continue
            if match["tp"]:
                entry["tp"] += 1
                matched_gt += 1
            else:
                entry["fp"] += 1
        for label in gt_labels:
            entry = per_class.get(int(label))
            if entry is not None:
                entry["fn"] += 1
        # Remove the true positives from the false-negative count.
        for match in matches:
            if match["tp"]:
                per_class[match["class_id"]]["fn"] -= 1

        tp += sum(1 for m in matches if m["tp"])
        fp += sum(1 for m in matches if not m["tp"])
        fn += max(len(gt_boxes) - matched_gt, 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "image_exact_match_accuracy": images_exact / images_total if images_total else 0.0,
        "images_evaluated": images_total,
        "per_class": {
            ID_TO_NAME.get(c, str(c)): per_class[c] for c in sorted(per_class)
        },
    }
