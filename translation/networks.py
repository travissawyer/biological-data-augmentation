"""Network definitions for the class-conditioned unpaired translation model.

The content/style split with AdaIN decoding follows MUNIT and DRIT; adaptive
instance normalisation follows Huang and Belongie; the KL-regularised Gaussian
style latent follows the VAE formulation; the auxiliary class head on the
discriminator follows AC-GAN.

The one departure from standard MUNIT is where the class label enters. In MUNIT
the AdaIN affine parameters are a function of the style code alone. Here the
label is embedded and concatenated to the style code before the parameter MLP,
so the same content and the same style code yield different normalisation
parameters for each class.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import CFG


def sn(module: nn.Module) -> nn.Module:
    return nn.utils.spectral_norm(module)


def _pm(cfg: CFG) -> str:
    return "reflect" if cfg.reflect_pad else "zeros"


class ResBlockIN(nn.Module):
    def __init__(self, ch: int, pad_mode: str):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False, padding_mode=pad_mode)
        self.in1 = nn.InstanceNorm2d(ch, affine=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False, padding_mode=pad_mode)
        self.in2 = nn.InstanceNorm2d(ch, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.in1(self.conv1(x)))
        h = self.in2(self.conv2(h))
        return self.act(h + x)


class AdaIN(nn.Module):
    """Normalise each channel over its spatial extent, then rescale by (gamma, beta)."""

    def __init__(self, ch: int):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(ch, affine=False)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h = self.inorm(x)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + gamma) + beta


class AdaINResBlock(nn.Module):
    def __init__(self, ch: int, pad_mode: str):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False, padding_mode=pad_mode)
        self.adain1 = AdaIN(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False, padding_mode=pad_mode)
        self.adain2 = AdaIN(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        h = self.conv1(x)
        h = self.act(self.adain1(h, gamma1, beta1))
        h = self.conv2(h)
        h = self.adain2(h, gamma2, beta2)
        return self.act(h + x)


class ContentEncoder(nn.Module):
    """Two strided downsamples followed by residual blocks; output is the content code."""

    def __init__(self, cfg: CFG):
        super().__init__()
        in_ch, base_ch, content_dim, n_res = cfg.in_channels, cfg.base_ch, cfg.content_dim, cfg.n_res
        pm = _pm(cfg)
        self.c1 = nn.Conv2d(in_ch, base_ch, 7, 1, 3, bias=False, padding_mode=pm)
        self.in1 = nn.InstanceNorm2d(base_ch, affine=True)
        self.c2 = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False, padding_mode=pm)
        self.in2 = nn.InstanceNorm2d(base_ch * 2, affine=True)
        self.c3 = nn.Conv2d(base_ch * 2, content_dim, 4, 2, 1, bias=False, padding_mode=pm)
        self.in3 = nn.InstanceNorm2d(content_dim, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.res = nn.Sequential(*[ResBlockIN(content_dim, pm) for _ in range(n_res)])

    def forward(self, x):
        x = self.act(self.in1(self.c1(x)))
        x = self.act(self.in2(self.c2(x)))
        x = self.act(self.in3(self.c3(x)))
        return self.res(x)


class StyleEncoder(nn.Module):
    """Convolutional trunk, global average pool, then the mean and log-variance
    of a Gaussian style posterior."""

    def __init__(self, cfg: CFG):
        super().__init__()
        in_ch, base_ch, style_dim = cfg.in_channels, cfg.base_ch, cfg.style_dim
        pm = _pm(cfg)
        ch = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 7, 1, 3, padding_mode=pm), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1, padding_mode=pm), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1, padding_mode=pm), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 4, ch * 4, 4, 2, 1, padding_mode=pm), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 4, ch * 4, 4, 2, 1, padding_mode=pm), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(ch * 4, style_dim)
        self.fc_lv = nn.Linear(ch * 4, style_dim)

    def forward(self, x):
        h = self.net(x)
        h = self.pool(h).view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_lv(h).clamp(-20.0, 10.0)
        return mu, logvar


class AdaINDecoder(nn.Module):
    """Decodes a content code under a style code and a class label.

    The MLP that produces the AdaIN affine parameters takes the style code
    concatenated with a class embedding. Its last layer is zero-initialised, so
    the decoder starts from an identity-like mapping.
    """

    def __init__(self, cfg: CFG):
        super().__init__()
        out_ch, base_ch, content_dim = cfg.in_channels, cfg.base_ch, cfg.content_dim
        style_dim, n_res, mlp_dim = cfg.style_dim, cfg.n_res, cfg.mlp_dim
        pm = _pm(cfg)
        self.content_dim = content_dim
        self.class_cond = bool(cfg.class_cond)

        # No embedding table at all in the unconditional variant.
        self.cls_emb = nn.Embedding(cfg.num_classes, cfg.cls_emb_dim) if self.class_cond else None

        self.resblocks = nn.ModuleList([AdaINResBlock(content_dim, pm) for _ in range(n_res)])

        self.u1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(content_dim, base_ch * 2, 5, 1, 2, bias=False, padding_mode=pm),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.ReLU(inplace=True),
        )
        self.u2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 2, base_ch, 5, 1, 2, bias=False, padding_mode=pm),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base_ch, out_ch, 7, 1, 3, padding_mode=pm)

        self.num_adain = n_res * 2
        adain_param_dim = self.num_adain * 2 * content_dim
        mlp_in = style_dim + (cfg.cls_emb_dim if self.class_cond else 0)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim), nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, adain_param_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _adain_params(self, s: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        if not self.class_cond:
            return self.mlp(s)
        return self.mlp(torch.cat([s, self.cls_emb(cls)], dim=1))

    def forward(self, c: torch.Tensor, s: torch.Tensor, cls: torch.Tensor):
        params = self._adain_params(s, cls).view(c.size(0), self.num_adain, 2, self.content_dim)
        x, i = c, 0
        for blk in self.resblocks:
            g1, b1 = params[:, i, 0, :], params[:, i, 1, :]; i += 1
            g2, b2 = params[:, i, 0, :], params[:, i, 1, :]; i += 1
            x = blk(x, g1, b1, g2, b2)
        return self.out(self.u2(self.u1(x)))


class PatchDiscriminator(nn.Module):
    """Spectrally-normalised patch discriminator with an optional AC-GAN class head."""

    def __init__(self, cfg: CFG):
        super().__init__()
        ch = cfg.d_base_ch
        layers = [sn(nn.Conv2d(cfg.in_channels, ch, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(1, cfg.d_n_layers):
            ch_next = min(ch * 2, 512)
            layers += [sn(nn.Conv2d(ch, ch_next, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
            ch = ch_next
        self.features = nn.Sequential(*layers)
        self.patch = sn(nn.Conv2d(ch, 1, 3, 1, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.class_cond = bool(cfg.class_cond)
        self.cls_head = sn(nn.Linear(ch, cfg.num_classes)) if self.class_cond else None

    def forward(self, x):
        h = self.features(x)
        patch = self.patch(h)
        if not self.class_cond:
            return patch, None
        return patch, self.cls_head(self.pool(h).view(h.size(0), -1))


class TwoDomainTranslator(nn.Module):
    """One content encoder, style encoder, decoder and discriminator per domain."""

    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.Ec = nn.ModuleList([ContentEncoder(cfg) for _ in range(cfg.num_domains)])
        self.Es = nn.ModuleList([StyleEncoder(cfg) for _ in range(cfg.num_domains)])
        self.G = nn.ModuleList([AdaINDecoder(cfg) for _ in range(cfg.num_domains)])
        self.D = nn.ModuleList([PatchDiscriminator(cfg) for _ in range(cfg.num_domains)])

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar.float())
        z = mu.float() + std * torch.randn_like(std)
        return z.to(dtype=mu.dtype)

    @staticmethod
    def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        mu32, lv32 = mu.float(), logvar.float()
        return 0.5 * torch.sum(mu32.pow(2) + lv32.exp() - lv32 - 1.0, dim=1).mean()
