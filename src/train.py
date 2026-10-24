"""Train (fine-tune) Faster R-CNN on the bottled-liquor defect dataset.

    python -m src.train --epochs 10 --batch-size 4

Everything the run needs is written to ``--output-dir`` so a result can be
reproduced later:

    run_config.json   exact arguments + dataset sizes + library versions
    history.json      per-epoch train/val loss and learning rate
    best_model.pth    weights at the lowest validation loss
    last_model.pth    weights after the final epoch
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MAX_SIZE,
    DEFAULT_MIN_SIZE,
    DEFAULT_SEED,
    DEFAULT_TRAIN_JSON,
    DEFAULT_VAL_JSON,
    FULL_IMAGE_DIR,
    REPO_ROOT,
    SUPPORTED_ARCHS,
)
from .dataset import DefectDataset, collate_fn, load_coco
from .model import build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN for bottle-defect detection.")
    parser.add_argument("--train-json", default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--val-json", default=DEFAULT_VAL_JSON)
    parser.add_argument("--image-dir", default=FULL_IMAGE_DIR)
    parser.add_argument("--arch", default="fasterrcnn_resnet50_fpn", choices=SUPPORTED_ARCHS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-step", type=int, default=3, help="StepLR step size in epochs")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="mixed precision (default on for CUDA)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--patience", type=int, default=0,
                        help="stop after N epochs without val improvement (0 = never)")
    parser.add_argument("--limit-train", type=int, default=0, help="debug: cap the training set")
    parser.add_argument("--print-every", type=int, default=50, help="log every N steps")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "outputs", "fasterrcnn_resnet50_fpn"))
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    return parser.parse_args(argv)


def train_one_epoch(model, optimizer, loader, device, scaler, epoch, print_every=50) -> float:
    model.train()
    running = 0.0
    count = 0
    started = time.time()

    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items() if k != "image_id"} for target in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        running += float(losses.detach())
        count += 1

        if print_every and step % print_every == 0:
            elapsed = time.time() - started
            print(
                f"  epoch {epoch} step {step}/{len(loader)} "
                f"loss={running / count:.4f} ({elapsed / step:.2f}s/step)",
                flush=True,
            )

    return running / max(count, 1)


@torch.no_grad()
def validate_loss(model, loader, device) -> float:
    """Validation loss. Detectors only return losses in train mode, so BN stats
    keep updating while we sweep the validation set - the usual compromise."""
    model.train()
    running = 0.0
    count = 0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items() if k != "image_id"} for target in targets]
        loss_dict = model(images, targets)
        running += float(sum(loss for loss in loss_dict.values()))
        count += 1
    model.eval()
    return running / max(count, 1)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = args.amp and device.type == "cuda"

    train_dataset = DefectDataset(
        args.image_dir, args.train_json, train=True, skip_background=False
    )
    val_dataset = DefectDataset(
        args.image_dir, args.val_json, train=False, skip_background=False
    )
    if args.limit_train:
        train_dataset.records = train_dataset.records[: args.limit_train]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        arch=args.arch,
        pretrained=True,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    print(f"[train] {args.arch} | {count_parameters(model) / 1e6:.1f}M params | device={device} | amp={use_amp}")

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload.get("model", payload))
        if isinstance(payload, dict) and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0
        print(f"[train] resumed from {args.resume} (epoch {start_epoch})")

    os.makedirs(args.output_dir, exist_ok=True)
    run_config = {
        "args": vars(args),
        "device": str(device),
        "amp": use_amp,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "train_instances": int(sum(len(r["labels"]) for r in train_dataset.records)),
        "val_instances": int(sum(len(r["labels"]) for r in val_dataset.records)),
        "torch": torch.__version__,
        "train_json": os.path.abspath(args.train_json),
        "val_json": os.path.abspath(args.val_json),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    history: List[Dict] = []
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        left = time.strftime("%H:%M:%S")
        print(f"\n=== epoch {epoch}/{args.epochs} (started {left}) ===")
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, scaler, epoch, print_every=args.print_every
        )
        val_loss = validate_loss(model, val_loader, device)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": learning_rate,
            }
        )
        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "last_model.pth"))
        if improved:
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pth"))

        print(
            f"[train] epoch {epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} lr={learning_rate:.2e}"
            f"{' *best*' if improved else ''}"
        )

        if args.patience and epochs_without_improvement >= args.patience:
            print(f"[train] early stopping after {args.patience} epochs without improvement")
            break

    print(f"\n[done] best val loss {best_loss:.4f} -> {os.path.join(args.output_dir, 'best_model.pth')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
