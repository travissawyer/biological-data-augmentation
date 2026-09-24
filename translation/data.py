"""Dataset for the translation model.

Reads a CSV that the user supplies. Required columns:

    path    path to the image, absolute or relative to --data_root
    domain  0 for the source domain (animal), 1 for the target domain (human)
    label   class id, 0 or 1

Any other columns are ignored. How many images to use, and which ones, is
entirely the caller's choice.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from .config import CFG

_RESAMPLE = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "nearest": Image.NEAREST}


class TranslationDataset(Dataset):
    """Images of one domain, optionally filtered and split into train/val."""

    def __init__(self, cfg: CFG, split: str, domain: int):
        df = pd.read_csv(cfg.csv_path)
        for col in ("path", "domain", "label"):
            if col not in df.columns:
                raise ValueError(f"{cfg.csv_path} is missing the '{col}' column")

        df = df[df["domain"].astype(int) == int(domain)].reset_index(drop=True)
        if cfg.filter_bg and domain == cfg.filter_bg_domain:
            df = self._drop_background_heavy(df, cfg)

        # Deterministic train/val split at the image level.
        rng = np.random.default_rng(cfg.seed)
        order = rng.permutation(len(df))
        n_val = int(round(cfg.val_split * len(df)))
        val_idx = set(order[:n_val].tolist())
        if split == "train":
            df = df.iloc[[i for i in range(len(df)) if i not in val_idx]]
        elif split == "val":
            df = df.iloc[[i for i in range(len(df)) if i in val_idx]]
        elif split != "all":
            raise ValueError(f"unknown split '{split}'")

        self.cfg = cfg
        self.paths: List[str] = df["path"].tolist()
        self.labels: List[int] = df["label"].astype(int).tolist()
        self.domain = int(domain)

    @staticmethod
    def _drop_background_heavy(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
        keep = {int(c) for c in cfg.filter_bg_keep_classes.split(",") if c.strip() != ""}
        rows = []
        for _, r in df.iterrows():
            if int(r["label"]) in keep:
                rows.append(r)
                continue
            im = Image.open(_resolve(cfg.data_root, r["path"])).convert("L")
            im = im.resize((cfg.filter_bg_downsample, cfg.filter_bg_downsample), Image.BILINEAR)
            frac = float((np.asarray(im, dtype=np.float32) / 255.0 < 0.02).mean())
            if frac <= cfg.filter_bg_max_frac:
                rows.append(r)
        return pd.DataFrame(rows).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        cfg = self.cfg
        im = Image.open(_resolve(cfg.data_root, self.paths[i])).convert("RGB")
        if im.size != (cfg.img_size, cfg.img_size):
            im = im.resize((cfg.img_size, cfg.img_size), _RESAMPLE[cfg.resize_mode])
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, self.labels[i]


def _resolve(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def class_balanced_sampler(ds: TranslationDataset, num_classes: int,
                           power: float) -> Optional[WeightedRandomSampler]:
    """Sample classes with weight 1 / count**power. power=1 gives uniform classes."""
    counts = np.bincount(np.asarray(ds.labels, dtype=int), minlength=num_classes).astype(np.float64)
    if (counts == 0).any() or power <= 0:
        return None
    per_class = 1.0 / np.power(counts, power)
    weights = per_class[np.asarray(ds.labels, dtype=int)]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(ds), replacement=True)


def domain_mean_std(ds: Dataset, max_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and std, estimated from a capped number of images."""
    n = min(len(ds), max_samples)
    acc = torch.zeros(3, dtype=torch.float64)
    acc_sq = torch.zeros(3, dtype=torch.float64)
    for i in range(n):
        x = ds[i][0].double()
        acc += x.mean(dim=(1, 2))
        acc_sq += (x ** 2).mean(dim=(1, 2))
    mean = acc / max(n, 1)
    var = (acc_sq / max(n, 1)) - mean ** 2
    return mean.float(), var.clamp_min(1e-8).sqrt().float()


class DomainNormalizer:
    """Applies and inverts the per-domain intensity normalization."""

    def __init__(self, means: Dict[int, torch.Tensor], stds: Dict[int, torch.Tensor]):
        self.means = {k: v.view(1, -1, 1, 1) for k, v in means.items()}
        self.stds = {k: v.view(1, -1, 1, 1) for k, v in stds.items()}

    def to(self, device: str) -> "DomainNormalizer":
        self.means = {k: v.to(device) for k, v in self.means.items()}
        self.stds = {k: v.to(device) for k, v in self.stds.items()}
        return self

    def norm(self, x: torch.Tensor, domain: int) -> torch.Tensor:
        return (x - self.means[domain]) / self.stds[domain]

    def denorm(self, x: torch.Tensor, domain: int) -> torch.Tensor:
        return x * self.stds[domain] + self.means[domain]
