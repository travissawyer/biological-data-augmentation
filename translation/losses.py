"""
© 2026 Arizona Board of Regents on behalf of the University of Arizona

Loss terms for the translation model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hinge_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def hinge_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return -d_fake.mean()


def l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().mean()


def class_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy from the discriminator's auxiliary head.

    Applied to real images against their true label when updating D, and to
    translated images against the *source* label when updating G, which is what
    carries the label across the domain boundary.
    """
    return F.cross_entropy(logits, target)
