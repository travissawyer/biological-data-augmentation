"""
© 2026 Arizona Board of Regents on behalf of the University of Arizona

Train and evaluate the downstream binary classifier.

    python -m classifier.train --manifest manifest.csv --out_dir runs/baseline

The manifest decides everything about the data: which images are used, how they
are split, and whether any translated images are included in the training split.
To compare training sets, write one manifest per condition and keep the
validation and test splits identical across them, so the comparison is paired.

Writes per-image test predictions with their source_id, so performance can be
pooled per specimen as well as per image.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data import TileDataset, build_transforms, load_manifest
from .model import VGG16BinaryClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, criterion, reg_strength, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().unsqueeze(1)
            p = model(x)
            loss = criterion(p, y)
            if reg_strength > 0:
                loss = loss + model.l2_penalty(reg_strength).squeeze()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            correct += ((p >= 0.5).float() == y).sum().item()
            n += x.size(0)
    return total / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, ds) -> pd.DataFrame:
    model.eval()
    rows = []
    for x, y, idx in loader:
        p = model(x.to(device, non_blocking=True)).squeeze(1).cpu().numpy()
        for j, i in enumerate(idx.tolist()):
            rows.append({"path": ds.paths[i], "source_id": ds.source_ids[i],
                         "y_true": int(ds.labels[i]), "prob": float(p[j])})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="CSV with path, label, split")
    p.add_argument("--data_root", default=".")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--preprocess", default="unit", choices=["unit", "imagenet"])
    p.add_argument("--init", default="pretrained", choices=["pretrained", "scratch"])
    p.add_argument("--dropout_p", type=float, default=0.1)
    p.add_argument("--num_train_keras_layers", type=int, default=10)
    p.add_argument("--reg_strength", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr_factor", type=float, default=0.1)
    p.add_argument("--lr_patience", type=int, default=3)
    p.add_argument("--lr_min_delta", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-7)
    p.add_argument("--early_stop_patience", type=int, default=7)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_manifest(args.manifest)
    tf_train, tf_eval = build_transforms(args.image_size, args.preprocess)
    sets = {
        "train": TileDataset(df[df.split == "train"], tf_train, args.data_root),
        "val":   TileDataset(df[df.split == "val"],   tf_eval,  args.data_root),
        "test":  TileDataset(df[df.split == "test"],  tf_eval,  args.data_root),
    }
    loaders = {
        k: DataLoader(v, batch_size=args.batch_size, shuffle=(k == "train"),
                      num_workers=args.num_workers, pin_memory=True)
        for k, v in sets.items()
    }

    model = VGG16BinaryClassifier(dropout_p=args.dropout_p,
                                  num_train_keras_layers=args.num_train_keras_layers,
                                  pretrained=(args.init == "pretrained")).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam([p_ for p_ in model.parameters() if p_.requires_grad],
                                 lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor,
                                  patience=args.lr_patience, threshold=args.lr_min_delta,
                                  min_lr=args.lr_min)

    best_loss, best_state, since_improved = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], device, criterion,
                                    args.reg_strength, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], device, criterion,
                                    args.reg_strength)
        scheduler.step(va_loss)
        print(f"epoch {ep:03d}  train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"val {va_loss:.4f}/{va_acc:.4f}")

        if va_loss < best_loss - args.lr_min_delta:
            best_loss, since_improved = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_improved += 1
            if since_improved >= args.early_stop_patience:
                print(f"early stopping at epoch {ep}")
                break

    if best_state is not None:          # restore the best checkpoint before testing
        model.load_state_dict(best_state)

    preds = predict(model, loaders["test"], device, sets["test"])
    preds.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    acc = float(((preds.prob >= 0.5).astype(int) == preds.y_true).mean())
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "optimizer": "Adam", "loss": "BCE + explicit L2",
                   "lr_schedule": "ReduceLROnPlateau on val loss",
                   "test_tile_accuracy": acc}, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    print(f"test accuracy {acc:.4f}   predictions -> {args.out_dir}/test_predictions.csv")


if __name__ == "__main__":
    main()
