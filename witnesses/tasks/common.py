"""Sampling helpers shared across tasks (device-agnostic; MPS lacks a Gamma sampler)."""

from __future__ import annotations

import torch


def dirichlet(alpha: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """n draws from Dirichlet(alpha) (alpha: [K]) -> [n, K] on `device`."""
    dev = torch.device("cpu") if device.type == "mps" else device
    g = torch._standard_gamma(alpha.to(dev, torch.float32).expand(n, -1).contiguous())
    g = g.clamp_min(1e-30)
    return (g / g.sum(-1, keepdim=True)).to(device)


def beta(a: torch.Tensor, b: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Elementwise Beta(a, b) draws (a, b broadcastable tensors)."""
    dev = torch.device("cpu") if device.type == "mps" else device
    a, b = torch.broadcast_tensors(a.to(dev, torch.float32), b.to(dev, torch.float32))
    x = torch._standard_gamma(a.contiguous()).clamp_min(1e-30)
    y = torch._standard_gamma(b.contiguous()).clamp_min(1e-30)
    return (x / (x + y)).to(device)


def categorical(probs: torch.Tensor, n: int) -> torch.Tensor:
    """n iid categorical draws per row of probs [B, K] -> [B, n] (long)."""
    return torch.multinomial(probs, n, replacement=True)
