"""Train the class-conditioned unpaired translation model.

The two domains are unpaired: batches are drawn independently from each, and no
correspondence between individual images is assumed or used.

    python -m translation.train --csv_path labels_translation.csv --save_dir ckpts

Add --no_class_cond for the unconditioned ablation, in which the label reaches
neither the decoder nor the discriminator.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CFG
from .data import (DomainNormalizer, TranslationDataset, class_balanced_sampler,
                   domain_mean_std)
from .losses import class_loss, hinge_d_loss, hinge_g_loss, l1
from .networks import TwoDomainTranslator

ANIMAL, HUMAN = 0, 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def augment(x: torch.Tensor, cfg: CFG) -> torch.Tensor:
    if cfg.aug_reflect_pad > 0:
        p = cfg.aug_reflect_pad
        x = torch.nn.functional.pad(x, (p, p, p, p), mode="reflect")
        i = random.randint(0, 2 * p)
        j = random.randint(0, 2 * p)
        x = x[:, :, i:i + cfg.img_size, j:j + cfg.img_size]
    if cfg.aug_hflip and random.random() < 0.5:
        x = torch.flip(x, dims=[3])
    if cfg.aug_vflip and random.random() < 0.5:
        x = torch.flip(x, dims=[2])
    if cfg.aug_rot90:
        x = torch.rot90(x, random.randint(0, 3), dims=[2, 3])
    return x


def centre_crop(x: torch.Tensor, margin: int) -> torch.Tensor:
    if margin <= 0:
        return x
    return x[:, :, margin:-margin, margin:-margin]


def train(cfg: CFG) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.save_dir, exist_ok=True)

    tr = {d: TranslationDataset(cfg, "train", d) for d in (ANIMAL, HUMAN)}
    va_animal = TranslationDataset(cfg, "val", ANIMAL)

    samplers = {
        ANIMAL: class_balanced_sampler(tr[ANIMAL], cfg.num_classes, cfg.balance_power)
        if cfg.balance_animal else None,
        HUMAN: class_balanced_sampler(tr[HUMAN], cfg.num_classes, cfg.balance_power)
        if cfg.balance_human else None,
    }
    loaders = {
        d: DataLoader(tr[d], batch_size=cfg.batch_size, sampler=samplers[d],
                      shuffle=samplers[d] is None, drop_last=True,
                      num_workers=cfg.num_workers, pin_memory=True)
        for d in (ANIMAL, HUMAN)
    }
    val_loader = DataLoader(va_animal, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    normalizer = None
    if cfg.use_domain_norm:
        means, stds = {}, {}
        for d in (ANIMAL, HUMAN):
            means[d], stds[d] = domain_mean_std(tr[d], cfg.norm_max_samples)
        normalizer = DomainNormalizer(means, stds).to(device)

    model = TwoDomainTranslator(cfg).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    g_params = itertools.chain(model.Ec.parameters(), model.Es.parameters(), model.G.parameters())
    opt_g = torch.optim.AdamW(g_params, lr=cfg.lr_g, betas=cfg.betas, weight_decay=cfg.weight_decay)
    opt_d = torch.optim.AdamW(model.D.parameters(), lr=cfg.lr_d, betas=cfg.betas,
                              weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device == "cuda")

    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump({**asdict(cfg), "optimizer": "AdamW"}, f, indent=2, default=str)

    iters = min(len(loaders[ANIMAL]), len(loaders[HUMAN]))
    it_a, it_h = cycle(loaders[ANIMAL]), cycle(loaders[HUMAN])
    best = float("inf")

    for ep in range(1, cfg.epochs + 1):
        model.train()
        for it in range(iters):
            xa, ca = next(it_a)
            xh, ch = next(it_h)
            xa, ca = augment(xa.to(device), cfg), ca.to(device)
            xh, ch = augment(xh.to(device), cfg), ch.to(device)
            if normalizer is not None:
                xa, xh = normalizer.norm(xa, ANIMAL), normalizer.norm(xh, HUMAN)

            with torch.cuda.amp.autocast(enabled=cfg.amp and device == "cuda"):
                # --- encode both domains
                c_a, (mu_a, lv_a) = model.Ec[ANIMAL](xa), model.Es[ANIMAL](xa)
                c_h, (mu_h, lv_h) = model.Ec[HUMAN](xh), model.Es[HUMAN](xh)
                s_a = model.reparam(mu_a, lv_a)
                s_h = model.reparam(mu_h, lv_h)

                # --- cross-domain translation, each under a style drawn from the prior
                s_h_rand = torch.randn_like(s_h)
                s_a_rand = torch.randn_like(s_a)
                x_ah = model.G[HUMAN](c_a, s_h_rand, ca)     # animal content, human style
                x_ha = model.G[ANIMAL](c_h, s_a_rand, ch)

                # ---------------------------------------------------- discriminator
                d_real_h, cls_real_h = model.D[HUMAN](centre_crop(xh, cfg.disc_crop))
                d_fake_h, _ = model.D[HUMAN](centre_crop(x_ah.detach(), cfg.disc_crop))
                d_real_a, cls_real_a = model.D[ANIMAL](centre_crop(xa, cfg.disc_crop))
                d_fake_a, _ = model.D[ANIMAL](centre_crop(x_ha.detach(), cfg.disc_crop))

                loss_d = cfg.lam_adv * (hinge_d_loss(d_real_h, d_fake_h)
                                        + hinge_d_loss(d_real_a, d_fake_a))
                if cfg.class_cond:
                    loss_d = loss_d + cfg.lam_cls * (class_loss(cls_real_h, ch)
                                                     + class_loss(cls_real_a, ca))

            opt_d.zero_grad(set_to_none=True)
            scaler.scale(loss_d).backward()
            scaler.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(model.D.parameters(), cfg.grad_clip)
            scaler.step(opt_d)

            with torch.cuda.amp.autocast(enabled=cfg.amp and device == "cuda"):
                # ------------------------------------------------------- generator
                g_fake_h, cls_fake_h = model.D[HUMAN](centre_crop(x_ah, cfg.disc_crop))
                g_fake_a, cls_fake_a = model.D[ANIMAL](centre_crop(x_ha, cfg.disc_crop))
                loss_adv = hinge_g_loss(g_fake_h) + hinge_g_loss(g_fake_a)

                # within-domain image reconstruction
                rec_a = model.G[ANIMAL](c_a, s_a, ca)
                rec_h = model.G[HUMAN](c_h, s_h, ch)
                loss_rec_x = l1(rec_a, xa) + l1(rec_h, xh)

                # content and style recovered from the translated images
                c_ah, (mu_ah, _) = model.Ec[HUMAN](x_ah), model.Es[HUMAN](x_ah)
                c_ha, (mu_ha, _) = model.Ec[ANIMAL](x_ha), model.Es[ANIMAL](x_ha)
                loss_rec_c = l1(c_ah, c_a) + l1(c_ha, c_h)
                loss_rec_s = l1(mu_ah, s_h_rand) + l1(mu_ha, s_a_rand)

                # translate back using the recovered content and the original style
                cyc_a = model.G[ANIMAL](c_ah, s_a, ca)
                cyc_h = model.G[HUMAN](c_ha, s_h, ch)
                loss_cyc = l1(cyc_a, xa) + l1(cyc_h, xh)

                loss_kl = model.kl_normal(mu_a, lv_a) + model.kl_normal(mu_h, lv_h)

                loss_g = (cfg.lam_adv * loss_adv
                          + cfg.lam_rec_x * loss_rec_x
                          + cfg.lam_rec_c * loss_rec_c
                          + cfg.lam_rec_s * loss_rec_s
                          + cfg.lam_cyc_x * loss_cyc
                          + cfg.lam_kl_s * loss_kl)
                if cfg.class_cond:
                    # translated images are scored against the SOURCE label, which is
                    # what makes the class survive the crossing between domains
                    loss_g = loss_g + cfg.lam_cls * (class_loss(cls_fake_h, ca)
                                                     + class_loss(cls_fake_a, ch))

            opt_g.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(
                [p for p in itertools.chain(model.Ec.parameters(), model.Es.parameters(),
                                            model.G.parameters())], cfg.grad_clip)
            scaler.step(opt_g)
            scaler.update()

            if cfg.log_every and it % cfg.log_every == 0:
                print(f"[Ep {ep:03d} It {it:04d}/{iters}] G={loss_g.item():.3f} "
                      f"D={loss_d.item():.3f} rec_x={loss_rec_x.item():.3f} "
                      f"cyc={loss_cyc.item():.3f} kl={loss_kl.item():.3f}")

        # ---- checkpoint on source-domain reconstruction error on the held-out split
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xv, cv in val_loader:
                xv, cv = xv.to(device), cv.to(device)
                if normalizer is not None:
                    xv = normalizer.norm(xv, ANIMAL)
                c_v, (mu_v, lv_v) = model.Ec[ANIMAL](xv), model.Es[ANIMAL](xv)
                rec = model.G[ANIMAL](c_v, model.reparam(mu_v, lv_v), cv)
                tot += (rec - xv).abs().mean().item() * xv.size(0)
                n += xv.size(0)
        val_l1 = tot / max(n, 1)
        print(f"Epoch {ep:03d} done. val_recon_l1={val_l1:.6f}")

        torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": ep},
                   os.path.join(cfg.save_dir, "last.pt"))
        if val_l1 < best:
            best = val_l1
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": ep},
                       os.path.join(cfg.save_dir, "best.pt"))
            print(f"  [best] val_recon_l1={val_l1:.6f}")


def parse_args() -> CFG:
    d = CFG()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv_path", default=d.csv_path)
    p.add_argument("--data_root", default=d.data_root)
    p.add_argument("--img_size", type=int, default=d.img_size)
    p.add_argument("--val_split", type=float, default=d.val_split)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch_size", type=int, default=d.batch_size)
    p.add_argument("--lr_g", type=float, default=d.lr_g)
    p.add_argument("--lr_d", type=float, default=d.lr_d)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num_workers", type=int, default=d.num_workers)
    p.add_argument("--save_dir", default=d.save_dir)
    p.add_argument("--no_class_cond", action="store_true",
                   help="unconditioned ablation: the label reaches neither the decoder "
                        "nor the discriminator")
    p.add_argument("--disc_crop", type=int, default=d.disc_crop)
    p.add_argument("--balance_animal", action="store_true")
    p.add_argument("--balance_human", action="store_true")
    p.add_argument("--balance_power", type=float, default=d.balance_power)
    p.add_argument("--filter_bg", action="store_true")
    p.add_argument("--filter_bg_max_frac", type=float, default=d.filter_bg_max_frac)
    p.add_argument("--filter_bg_keep_classes", default=d.filter_bg_keep_classes)
    p.add_argument("--aug_reflect_pad", type=int, default=d.aug_reflect_pad)
    p.add_argument("--aug_hflip", action="store_true")
    p.add_argument("--aug_vflip", action="store_true")
    p.add_argument("--aug_rot90", action="store_true")
    a = p.parse_args()

    cfg = CFG(**{k: v for k, v in vars(a).items() if k != "no_class_cond"})
    cfg.class_cond = not a.no_class_cond
    return cfg


if __name__ == "__main__":
    train(parse_args())
