"""Web demo: upload a bottle photo and inspect the detected defects.

    python app/app.py --port 5000

The page is self-contained (no CDN, no build step): a single HTML template with
inline CSS/JS.  Detection results come back as JSON with a base64-encoded
annotated JPEG, so the UI updates without a round-trip to disk.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from typing import Dict, List

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

# Make `python app/app.py` work without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (  # noqa: E402
    CLASS_NAMES,
    CLASS_NAMES_ZH,
    ID_TO_NAME,
    ID_TO_NAME_ZH,
    SAMPLE_ANNOTATION_FILE,
    SAMPLE_IMAGE_DIR,
)
from src.dataset import index_annotations, load_coco  # noqa: E402
from src.inference import Detector  # noqa: E402
from src.visualize import draw_detections  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

STATE: Dict = {
    "detector": None,
    "threshold": 0.5,
    "sample_images": [],
    "sample_annotations": {},
    "metrics": None,
}


def bootstrap(checkpoint=None, device=None, threshold=0.5) -> None:
    """Load the model once and cache the sample gallery metadata."""
    STATE["detector"] = Detector(checkpoint=checkpoint, device=device, score_threshold=threshold)
    STATE["threshold"] = threshold

    # First inference pays for CUDA kernel autotuning (~500 ms); do it now so the
    # latency shown in the UI reflects a steady-state call.
    STATE["detector"].predict(Image.new("RGB", (658, 492), (24, 24, 24)))

    if os.path.exists(SAMPLE_ANNOTATION_FILE):
        coco = load_coco(SAMPLE_ANNOTATION_FILE)
        grouped = index_annotations(coco)
        images = [
            image
            for image in coco["images"]
            if os.path.exists(os.path.join(SAMPLE_IMAGE_DIR, image["file_name"]))
        ]
        random.Random(2024).shuffle(images)
        STATE["sample_images"] = images
        STATE["sample_annotations"] = grouped

    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "eval_checkpoint", "metrics.json",
    )
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as handle:
            STATE["metrics"] = json.load(handle)


def encode_jpeg(image: Image.Image, quality: int = 88) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def summarise(detections: List[Dict]) -> Dict:
    """Group detections by class - the same defect often appears twice."""
    counts: Dict[str, int] = {}
    for detection in detections:
        counts[detection["label_zh"]] = counts.get(detection["label_zh"], 0) + 1
    return {"by_class": counts, "total": len(detections)}


@app.route("/")
def index():
    metrics = STATE["metrics"] or {}
    return render_template(
        "index.html",
        checkpoint=os.path.basename(STATE["detector"].checkpoint) if STATE["detector"] else "-",
        device=str(STATE["detector"].device) if STATE["detector"] else "-",
        threshold=STATE["threshold"],
        class_names=CLASS_NAMES[1:],
        class_names_zh=CLASS_NAMES_ZH[1:],
        mAP50=round((metrics.get("mAP50") or 0) * 100, 1) if metrics else None,
        mAP=round((metrics.get("mAP") or 0) * 100, 1) if metrics else None,
        samples=[image["file_name"] for image in STATE["sample_images"][:12]],
    )


@app.route("/api/classes")
def api_classes():
    return jsonify([{ "id": i, "name": CLASS_NAMES[i], "name_zh": CLASS_NAMES_ZH[i]}
                    for i in range(1, len(CLASS_NAMES))])


@app.route("/sample-image/<path:name>")
def sample_image(name):
    """Serve one of the bundled sample photos to the footer gallery."""
    from flask import send_from_directory

    return send_from_directory(SAMPLE_IMAGE_DIR, os.path.basename(name))


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Detect on an uploaded file, or on a bundled sample (``sample=<file name>``)."""
    detector: Detector = STATE["detector"]
    if detector is None:
        return jsonify({"error": "model not loaded"}), 503

    threshold = float(request.form.get("threshold", STATE["threshold"]))
    detector.score_threshold = threshold

    sample_name = request.form.get("sample")
    ground_truth = None
    if sample_name:
        path = os.path.join(SAMPLE_IMAGE_DIR, os.path.basename(sample_name))
        if not os.path.exists(path):
            return jsonify({"error": f"unknown sample {sample_name}"}), 404
    else:
        upload = request.files.get("image")
        if upload is None:
            return jsonify({"error": "no image uploaded"}), 400
        path = upload.filename or "upload"
        try:
            image = Image.open(upload.stream).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({"error": "that file is not a readable image"}), 400

    started = time.time()
    if sample_name:
        image = Image.open(path).convert("RGB")
        image_id = None
        for candidate in STATE["sample_images"]:
            if candidate["file_name"] == os.path.basename(sample_name):
                image_id = candidate["id"]
                break
        if image_id is not None:
            ground_truth = [
                {
                    "box": [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                            a["bbox"][1] + a["bbox"][3]],
                    "label": a["category_id"],
                    "label_zh": ID_TO_NAME_ZH.get(a["category_id"], ""),
                }
                for a in STATE["sample_annotations"].get(image_id, [])
            ]

    detections = detector.predict(image)
    elapsed_ms = (time.time() - started) * 1000

    annotated = draw_detections(image, detections, ground_truth=ground_truth, chinese=True)
    plain = draw_detections(image, detections, chinese=True) if ground_truth else None

    return jsonify(
        {
            "image": encode_jpeg(annotated),
            "plain": encode_jpeg(plain) if plain is not None else None,
            "detections": detections,
            "summary": summarise(detections),
            "ground_truth": ground_truth or [],
            "inference_ms": round(elapsed_ms, 1),
            "threshold": threshold,
            "size": list(image.size),
        }
    )


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "checkpoint": STATE["detector"].checkpoint if STATE["detector"] else None,
        "device": str(STATE["detector"].device) if STATE["detector"] else None,
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the defect-detection web demo.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bootstrap(checkpoint=args.checkpoint, device=args.device, threshold=args.threshold)
    print(f"[app] model={STATE['detector'].checkpoint} device={STATE['detector'].device}")
    print(f"[app] open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
