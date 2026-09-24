"""Generate translated images from a trained checkpoint.

Maps images from the source domain (animal) into the target domain (human),
writing one translated image per input image and a manifest recording the
settings each image was produced with.

    python -m translation.translate --ckpt ckpts/best.pt \
        --csv_path labels_translation.csv --out_dir translated
"""

from __future__ import annotations

import argparse
import csv
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .config import CFG
from .data import DomainNormalizer, TranslationDataset, domain_mean_std
from .networks import TwoDomainTranslator

ANIMAL, HUMAN = 0, 1
CLASS_DIR = {0: "normal", 1: "tumor"}


def to_pil(x: torch.Tensor) -> Image.Image:
    x = x.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def translate(cfg: CFG) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(cfg.ckpt, map_location=device)
    saved = ckpt.get("cfg", {})

    # architecture must match the checkpoint, whatever the caller passed
    for key in ("base_ch", "content_dim", "style_dim", "n_res", "mlp_dim",
                "cls_emb_dim", "d_base_ch", "d_n_layers", "num_classes",
                "num_domains", "in_channels", "class_cond", "reflect_pad"):
        if key in saved:
            setattr(cfg, key, saved[key])

    model = TwoDomainTranslator(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = TranslationDataset(cfg, cfg.sample_split, ANIMAL)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)

    normalizer = None
    if cfg.use_domain_norm:
        means, stds = {}, {}
        for d in (ANIMAL, HUMAN):
            full = TranslationDataset(cfg, "all", d)
            means[d], stds[d] = domain_mean_std(full, cfg.norm_max_samples)
        normalizer = DomainNormalizer(means, stds).to(device)

    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.subdir_by_class:
        for name in CLASS_DIR.values():
            os.makedirs(os.path.join(cfg.out_dir, name), exist_ok=True)

    gen = torch.Generator(device="cpu").manual_seed(cfg.style_seed)
    manifest = open(os.path.join(cfg.out_dir, cfg.manifest_name), "w", newline="")
    writer = csv.writer(manifest)
    writer.writerow(["source_path", "output_path", "label", "style_mode", "class_cond"])

    n = 0
    with torch.no_grad():
        for xb, cb in loader:
            xb, cb = xb.to(device), cb.to(device)
            xin = normalizer.norm(xb, ANIMAL) if normalizer is not None else xb
            content = model.Ec[ANIMAL](xin)

            if cfg.style_mode == "zero":
                style = torch.zeros(xb.size(0), cfg.style_dim, device=device)
            else:
                style = torch.randn(xb.size(0), cfg.style_dim, generator=gen).to(device)

            out = model.G[HUMAN](content, style, cb)
            if normalizer is not None:
                out = normalizer.denorm(out, HUMAN)

            for i in range(out.size(0)):
                label = int(cb[i].item())
                if cfg.only_class >= 0 and label != cfg.only_class:
                    continue
                src = ds.paths[n + i]
                stem = os.path.splitext(os.path.basename(src))[0]
                sub = CLASS_DIR.get(label, str(label)) if cfg.subdir_by_class else ""
                dst = os.path.join(cfg.out_dir, sub, f"{stem}_translated.png")
                to_pil(out[i]).save(dst)
                writer.writerow([src, dst, label, cfg.style_mode, int(cfg.class_cond)])
            n += out.size(0)

    manifest.close()
    print(f"wrote translated images and {cfg.manifest_name} to {cfg.out_dir}")


def parse_args() -> CFG:
    d = CFG()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv_path", default=d.csv_path)
    p.add_argument("--data_root", default=d.data_root)
    p.add_argument("--out_dir", default=d.out_dir)
    p.add_argument("--batch_size", type=int, default=d.batch_size)
    p.add_argument("--num_workers", type=int, default=d.num_workers)
    p.add_argument("--sample_split", default=d.sample_split, choices=["train", "val", "all"])
    p.add_argument("--style_mode", default=d.style_mode, choices=["random", "zero"],
                   help="random draws one style per image; zero fixes s = 0")
    p.add_argument("--style_seed", type=int, default=d.style_seed)
    p.add_argument("--only_class", type=int, default=d.only_class)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--val_split", type=float, default=d.val_split)
    p.add_argument("--filter_bg", action="store_true")
    p.add_argument("--filter_bg_keep_classes", default=d.filter_bg_keep_classes)
    return CFG(**vars(p.parse_args()))


if __name__ == "__main__":
    translate(parse_args())
