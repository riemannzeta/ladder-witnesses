"""TinyGPT: a small pre-LN causal transformer with several readout heads.

The block structure follows introspection/introspection/models/transformer.py (itself the
architecture of Misra's RecurrenceTransformer). Differences: attention uses PyTorch's
scaled_dot_product_attention (fused kernels where available), there is no dropout, inputs may
carry a continuous feature vector per position (projected and added to the token embedding),
and the model has a dict of named linear readout heads instead of a single output projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, C // self.n_heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        heads: dict[str, int],
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int | None = None,
        cont_dim: int = 0,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.cont_proj = nn.Linear(cont_dim, d_model) if cont_dim > 0 else None
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_ff or 4 * d_model) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict({name: nn.Linear(d_model, dim) for name, dim in heads.items()})
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2 scaled init on residual projections
            if name.endswith("attn.out_proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(
        self, tokens: torch.Tensor, cont: torch.Tensor | None = None, return_hidden: bool = False
    ) -> dict[str, torch.Tensor]:
        T = tokens.shape[1]
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
        if self.cont_proj is not None and cont is not None:
            x = x + self.cont_proj(cont)
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x)
        out = {name: head(h) for name, head in self.heads.items()}
        if return_hidden:
            out["_hidden"] = h
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
