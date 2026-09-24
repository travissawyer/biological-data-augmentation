"""
© 2026 Arizona Board of Regents on behalf of the University of Arizona

Dataset for the downstream classifier.

Reads a CSV that the user supplies. Required columns:

    path    path to the image, absolute or relative to --data_root
    label   class id, 0 or 1
    split   one of train / val / test

Optional columns:

    source_id   specimen identifier. Carried through to the predictions file so
                results can be pooled per specimen rather than per image.
    origin      free-text tag, e.g. to mark which images are real and which are
                translated. Recorded, never used for training.

Splitting is the caller's responsibility. Assign every image of a given
specimen to the same split; splitting at the image level leaks a specimen
across partitions and inflates the result.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REQUIRED = ("path", "label", "split")


def load_manifest(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in REQUIRED:
        if col not in df.columns:
            raise ValueError(f"{csv_path} is missing the '{col}' column")
    bad = set(df["split"].unique()) - {"train", "val", "test"}
    if bad:
        raise ValueError(f"unexpected split values: {sorted(bad)}")
    if "source_id" not in df.columns:
        df["source_id"] = df["path"]
    if "origin" not in df.columns:
        df["origin"] = ""
    return df


def build_transforms(image_size: int, preprocess: str):
    """`unit` scales to [0, 1]. `imagenet` additionally applies the channel
    mean/std the pretrained weights were trained with."""
    train_ops = [transforms.Resize((image_size, image_size)),
                 transforms.RandomVerticalFlip(p=0.5),
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.ToTensor()]
    eval_ops = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if preprocess == "imagenet":
        norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        train_ops.append(norm)
        eval_ops.append(norm)
    elif preprocess != "unit":
        raise ValueError(f"unknown preprocess '{preprocess}'")
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


class TileDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform, data_root: str = "."):
        self.paths = rows["path"].tolist()
        self.labels = rows["label"].astype(int).tolist()
        self.source_ids = rows["source_id"].astype(str).tolist()
        self.transform = transform
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, int]:
        p = self.paths[i]
        if not os.path.isabs(p):
            p = os.path.join(self.data_root, p)
        img = Image.open(p).convert("RGB")
        return self.transform(img), self.labels[i], i
