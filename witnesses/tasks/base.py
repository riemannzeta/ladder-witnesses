"""Task interface and shared loss / divergence helpers.

A task generates episodes on the fly (on any device), defines its readout heads and their
losses, builds frozen evaluation sets (with exact Bayes-oracle values attached), and evaluates a
model against those oracles. Targets are PFN-style: the episode's TRUE quantity (a probability
vector or a real number), so the minimizer of the expected loss is the Bayes posterior
predictive under the training prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

LOG2 = np.log(2.0)


@dataclass(frozen=True)
class HeadSpec:
    dim: int
    loss: str  # "soft_ce" | "soft_bce" | "mse"
    weight: float = 1.0


class Task:
    """Subclasses set: name, vocab_size, max_len, cont_dim, heads, headline (metric key)."""

    name: str = "task"
    vocab_size: int = 0
    max_len: int = 0
    cont_dim: int = 0
    heads: dict[str, HeadSpec] = {}
    headline: str = ""

    def sample_batch(self, batch_size: int, device: torch.device) -> dict:
        raise NotImplementedError

    def make_eval_sets(self, seed: int, n_episodes: int) -> dict:
        raise NotImplementedError

    def evaluate(self, model, eval_sets: dict, device: torch.device) -> dict:
        raise NotImplementedError


def head_loss(spec: HeadSpec, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Masked mean loss. pred [B,T,D]; target [B,T,D] or [B,1,D] (broadcast); mask [B,T] bool."""
    pred = pred.float()
    target = target.expand_as(pred)
    if spec.loss == "soft_ce":
        per = -(target * F.log_softmax(pred, -1)).sum(-1)
    elif spec.loss == "soft_bce":
        per = F.binary_cross_entropy_with_logits(pred, target, reduction="none").sum(-1)
    elif spec.loss == "mse":
        per = ((pred - target) ** 2).sum(-1)
    else:
        raise ValueError(spec.loss)
    m = mask.float()
    return (per * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------- divergences (numpy, float64)


def kl_bernoulli_bits(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """KL(Bern(p) || Bern(q)) in bits, elementwise."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return (p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))) / LOG2


def kl_categorical_bits(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """KL(p || q) in bits over the last axis."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(-1) / LOG2


@torch.no_grad()
def run_model(model, tokens: torch.Tensor, cont: torch.Tensor | None, device, chunk: int = 1024):
    """Run the model over a (CPU) eval tensor in chunks; returns dict of CPU float64 numpy outputs."""
    model.eval()
    outs: dict[str, list[np.ndarray]] = {}
    for i in range(0, tokens.shape[0], chunk):
        t = tokens[i : i + chunk].to(device)
        c = cont[i : i + chunk].to(device) if cont is not None else None
        o = model(t, c)
        for k, v in o.items():
            outs.setdefault(k, []).append(v.float().cpu().numpy().astype(np.float64))
    model.train()
    return {k: np.concatenate(v) for k, v in outs.items()}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope of y on x (with intercept) and R^2."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) <= 1e-9 * max(1.0, np.abs(x).max()):
        return float("nan"), float("nan")
    xc = x - x.mean()
    b = float((xc * (y - y.mean())).sum() / (xc**2).sum())
    a = y.mean() - b * x.mean()
    ss_res = ((y - a - b * x) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return b, float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
