"""
© 2026 Arizona Board of Regents on behalf of the University of Arizona

Configuration for the unpaired two-domain translation model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CFG:
    # ------------------------------------------------------------------ data
    # CSV listing the images to train on. See README for the column contract.
    csv_path: str = "labels_translation.csv"
    data_root: str = "."
    in_channels: int = 3
    img_size: int = 256
    num_domains: int = 2          # 0 = animal, 1 = human
    num_classes: int = 2          # 0 = normal, 1 = tumor
    val_split: float = 0.1
    resize_mode: str = "bicubic"  # bicubic | bilinear | nearest

    # Per-domain channel mean/std, estimated from the images themselves so the
    # two domains enter the network on a common intensity scale.
    use_domain_norm: bool = True
    norm_max_samples: int = 2000

    # ----------------------------------------------------------------- model
    base_ch: int = 64
    content_dim: int = 256
    style_dim: int = 32
    n_res: int = 4
    mlp_dim: int = 256
    cls_emb_dim: int = 32
    d_base_ch: int = 64
    d_n_layers: int = 4

    # Class conditioning.
    #   True  -> the class embedding is concatenated to the style code before the
    #            AdaIN parameter MLP, and the discriminator carries an auxiliary
    #            class head trained with lam_cls.
    #   False -> the decoder never sees the label and both class losses are
    #            dropped, so the label cannot reach the generator by any route.
    class_cond: bool = True

    # ---------------------------------------------------- padding, augmentation
    reflect_pad: bool = True      # reflection padding inside encoders/decoder
    aug_reflect_pad: int = 0      # pad then random-crop back to img_size (train only)
    aug_hflip: bool = False
    aug_vflip: bool = False
    aug_rot90: bool = False

    # Drop tiles that are mostly empty background. Useful when one domain is
    # tiled over slides with large blank regions.
    filter_bg: bool = False
    filter_bg_domain: int = 0
    filter_bg_max_frac: float = 0.60
    filter_bg_keep_classes: str = ""   # comma-separated class ids exempt from the filter
    filter_bg_downsample: int = 64

    # Class-balanced sampling. weights = 1 / count**power; 1.0 gives uniform
    # class sampling, 0.0 leaves the empirical distribution untouched.
    balance_animal: bool = False
    balance_human: bool = False
    balance_power: float = 1.0

    # The discriminator sees a center crop, which keeps it from scoring border
    # artifacts introduced by padding.
    disc_crop: int = 0

    # -------------------------------------------------------------- training
    epochs: int = 160
    batch_size: int = 8
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    amp: bool = True
    seed: int = 42
    num_workers: int = 4

    # ---------------------------------------------------------------- losses
    lam_adv: float = 1.0
    lam_rec_x: float = 10.0       # image reconstruction
    lam_rec_c: float = 1.0        # content reconstruction
    lam_rec_s: float = 1.0        # style reconstruction
    lam_cyc_x: float = 10.0       # cycle consistency
    lam_kl_s: float = 0.01        # KL on the style posterior
    lam_cls: float = 1.0          # auxiliary classification

    # -------------------------------------------------------------------- io
    save_dir: str = "./checkpoints"
    log_every: int = 50

    # ---------------------------------------------------- sampling/inference
    ckpt: Optional[str] = None
    out_dir: str = "./translated"
    sample_split: str = "train"   # train | val | all
    only_class: int = -1          # -1 = all classes
    subdir_by_class: bool = True
    manifest_name: str = "manifest.csv"
    style_mode: str = "random"    # random = one style draw per image; zero = s fixed to 0
    style_seed: int = 123

    # ------------------------------------------------------------ perf flags
    channels_last: bool = True
    tf32: bool = True
