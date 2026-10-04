"""E3 -- L3 demonstration: the residual twin with a continuous cross-world correlation rho.

Mechanism (per episode): w ~ N(0,1), sigma = 0.5, and for a unit with exogenous (U1, U2) ~ N(0, I):

    Y = w*X + sigma * [ U1                              if X > 0
                        rho*U1 + sqrt(1 - rho^2)*U2     if X <= 0 ]

For every rho the noise is N(0, sigma^2) and independent of X, so the observational (L1) and
interventional (L2) distributions are identical across rho: rho is *pure* Layer-3 information.
For a unit observed at (x*, y*), the counterfactual "Y had X been x'" has expectation

    E[Y_{x'} | w, x*, y*] = w*x' + r*(y* - w*x*),   r = 1 (same side of 0), r = rho (crossing).

Rho = +1 is an additive-noise model (residual transfers), rho = -1 flips the residual, rho = 0
makes the counterfactual equal to the interventional answer w*x'. Training draws rho uniformly
from [-1,-0.3] U [0.3,1]; |rho| < 0.3 is held out for evaluation.

Sequence: [BOS][RHO: (rho,0,0)][UNIT: (x*, y*, x')][PAIR: (x_i, y_i, 0)] x K  (continuous
features through a linear projection). Heads `cf` and `l2` (scalar, MSE) are read at the UNIT
position (k = 0 pairs) and after each pair (k = 1..K).

Targets are Rao-Blackwellized given the latents (w and the unit's identified noise), which leaves
the MSE minimizer -- the Bayes posterior mean -- unchanged:
    cf: w*x' + r*(y* - w*x*)      l2: w*x'
Conditions: rho_mode = "aware" | "shuffled" (format-matched blind control: the RHO token carries
an independent draw); target_mode = "cf" | "l2" (L2-trained control: the cf head is trained on
fresh-noise interventional targets w*x').

Oracle (exact): with m = E[w | data] from conjugate Bayesian linear regression (prior N(0,1),
noise var sigma^2, using the K pairs seen so far AND the unit's own factual pair), the Bayes
answer is m*(x' - r*x*) + r*y*  (linear in w, so the posterior mean suffices).
"""

from __future__ import annotations

import numpy as np
import torch

from witnesses.tasks.base import HeadSpec, Task, fit_slope, run_model

BOS, RHO, UNIT, PAIR = 0, 1, 2, 3
RHO_BIN_EDGES = (-1.0, -0.65, -0.3, -0.1, 0.1, 0.3, 0.65, 1.0)
X_MIN = 0.1  # exclude near-zero x*, x' from slope readouts (sign discontinuity)


class E3Twin(Task):
    name = "e3_twin"
    vocab_size = 4
    cont_dim = 3
    headline = "excess_cf"

    def __init__(self, n_pairs: int = 32, sigma: float = 0.5, rho_holdout: float = 0.3,
                 rho_mode: str = "aware", target_mode: str = "cf"):
        assert rho_mode in ("aware", "shuffled") and target_mode in ("cf", "l2")
        self.K = n_pairs
        self.sigma = sigma
        self.rho_holdout = rho_holdout
        self.rho_mode = rho_mode
        self.target_mode = target_mode
        self.max_len = 3 + n_pairs
        self.heads = {"cf": HeadSpec(1, "mse"), "l2": HeadSpec(1, "mse")}

    def eval_signature(self) -> dict:
        return {"K": self.K, "sigma": self.sigma}

    # ------------------------------------------------------------------ generation
    def _rho_train(self, B: int, device) -> torch.Tensor:
        mag = self.rho_holdout + (1 - self.rho_holdout) * torch.rand(B, device=device)
        return torch.where(torch.rand(B, device=device) < 0.5, -mag, mag)

    def _noise(self, x, rho, u1, u2):
        rr = rho.view(-1, *([1] * (x.dim() - 1)))
        return self.sigma * torch.where(x > 0, u1, rr * u1 + torch.sqrt(1 - rr**2) * u2)

    def _episodes(self, B: int, device, rho: torch.Tensor | None = None) -> dict:
        K = self.K
        w = torch.randn(B, device=device)
        rho = self._rho_train(B, device) if rho is None else rho.to(device)
        x = torch.randn(B, K, device=device)
        y = w[:, None] * x + self._noise(x, rho, torch.randn(B, K, device=device),
                                         torch.randn(B, K, device=device))
        xs, xp = torch.randn(B, device=device), torch.randn(B, device=device)
        ys = w * xs + self._noise(xs, rho, torch.randn(B, device=device),
                                  torch.randn(B, device=device))
        cross = (xs > 0) != (xp > 0)
        r = torch.where(cross, rho, torch.ones_like(rho))
        return dict(w=w, rho=rho, x=x, y=y, xs=xs, ys=ys, xp=xp, cross=cross,
                    cf=w * xp + r * (ys - w * xs), l2=w * xp)

    def _inputs(self, ep: dict, rho_shown: torch.Tensor):
        B, K = ep["x"].shape
        dev = ep["x"].device
        tokens = torch.cat([torch.tensor([BOS, RHO, UNIT], device=dev).expand(B, -1),
                            torch.full((B, K), PAIR, device=dev)], 1)
        cont = torch.zeros(B, 3 + K, 3, device=dev)
        cont[:, 1, 0] = rho_shown
        cont[:, 2] = torch.stack([ep["xs"], ep["ys"], ep["xp"]], -1)
        cont[:, 3:, 0] = ep["x"]
        cont[:, 3:, 1] = ep["y"]
        return tokens, cont

    def sample_batch(self, batch_size: int, device) -> dict:
        ep = self._episodes(batch_size, device)
        shown = self._rho_train(batch_size, device) if self.rho_mode == "shuffled" else ep["rho"]
        tokens, cont = self._inputs(ep, shown)
        pos = torch.arange(self.max_len, device=device)
        mask = (pos >= 2).expand(batch_size, -1)
        cf_t = ep["cf"] if self.target_mode == "cf" else ep["l2"]
        return dict(tokens=tokens, cont=cont,
                    targets={"cf": cf_t[:, None, None], "l2": ep["l2"][:, None, None]},
                    masks={"cf": mask, "l2": mask})

    # ------------------------------------------------------------------ eval sets
    def make_eval_sets(self, seed: int, n_episodes: int) -> dict:
        torch.manual_seed(seed)
        cpu = torch.device("cpu")
        rho = 2 * torch.rand(n_episodes) - 1  # all rho, including the held-out band
        return {"main": {k: v.cpu() for k, v in self._episodes(n_episodes, cpu, rho).items()}}

    # ------------------------------------------------------------------ oracle
    def oracle(self, ep: dict) -> dict:
        x, y = ep["x"].numpy().astype(np.float64), ep["y"].numpy().astype(np.float64)
        xs, ys, xp = (ep[k].numpy().astype(np.float64) for k in ("xs", "ys", "xp"))
        rho, cross = ep["rho"].numpy().astype(np.float64), ep["cross"].numpy()
        B, K = x.shape
        sxy = np.concatenate([np.zeros((B, 1)), np.cumsum(x * y, 1)], 1) + (xs * ys)[:, None]
        sxx = np.concatenate([np.zeros((B, 1)), np.cumsum(x * x, 1)], 1) + (xs * xs)[:, None]
        m = sxy / (self.sigma**2 + sxx)  # [B, K+1] posterior mean of w after k pairs
        r_aware = np.where(cross, rho, 1.0)[:, None]
        r_blind = np.where(cross, 0.0, 1.0)[:, None]  # E[rho] = 0 under the symmetric prior
        cf_aware = m * (xp[:, None] - r_aware * xs[:, None]) + r_aware * ys[:, None]
        cf_blind = m * (xp[:, None] - r_blind * xs[:, None]) + r_blind * ys[:, None]
        l2 = m * xp[:, None]
        # irreducible Bayes MSE of the (Rao-Blackwellized) cf target, per k:
        # target - oracle = (w - m) * (x' - r x*), and Var(w | data) = sigma^2 / (sigma^2 + sxx)
        var_w = self.sigma**2 / (self.sigma**2 + sxx)
        bayes_mse_cf = var_w * (xp[:, None] - r_aware * xs[:, None]) ** 2
        bayes_mse_l2 = var_w * xp[:, None] ** 2
        return dict(m=m, cf=cf_aware, cf_blind=cf_blind, l2=l2,
                    bayes_mse_cf=bayes_mse_cf, bayes_mse_l2=bayes_mse_l2)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, model, eval_sets: dict, device, return_preds: bool = False):
        ep = eval_sets["main"]
        orc = self.oracle(ep)
        tokens, cont = self._inputs(ep, ep["rho"])
        out = run_model(model, tokens, cont, device)
        cf = out["cf"][:, 2:, 0]  # [B, K+1]
        l2 = out["l2"][:, 2:, 0]
        res: dict = {}
        ks = slice(2, None)
        ex_cf = (cf - orc["cf"]) ** 2
        res["excess_cf"] = float(ex_cf[:, ks].mean())
        res["excess_cf_blind"] = float(((cf - orc["cf_blind"]) ** 2)[:, ks].mean())
        res["excess_cf_vs_l2"] = float(((cf - orc["l2"]) ** 2)[:, ks].mean())
        res["excess_l2"] = float(((l2 - orc["l2"]) ** 2)[:, ks].mean())
        res["nexcess_cf"] = float(ex_cf[:, ks].mean() / orc["bayes_mse_cf"][:, ks].mean())
        res["nexcess_l2"] = float(((l2 - orc["l2"]) ** 2)[:, ks].mean()
                                  / orc["bayes_mse_l2"][:, ks].mean())
        res["nexcess_cf_curve"] = (ex_cf.mean(0) / orc["bayes_mse_cf"].mean(0)).tolist()
        # residual-transfer slopes at the final position ------------------------
        xs, xp, ys = ep["xs"].numpy(), ep["xp"].numpy(), ep["ys"].numpy()
        rho, cross = ep["rho"].numpy(), ep["cross"].numpy()
        m = orc["m"][:, -1]
        dev_ = cf[:, -1] - m * xp
        resid = ys - m * xs
        ok = (np.abs(xs) >= X_MIN) & (np.abs(xp) >= X_MIN)
        bins = []
        for lo, hi in zip(RHO_BIN_EDGES[:-1], RHO_BIN_EDGES[1:]):
            sel = ok & cross & (rho >= lo) & (rho < hi)
            s, _ = fit_slope(resid[sel], dev_[sel])
            bins.append(dict(lo=lo, hi=hi, rho_mean=float(rho[sel].mean()), slope=s,
                             heldout=bool(abs(lo) < self.rho_holdout - 1e-9
                                          and abs(hi) < self.rho_holdout + 1e-9),
                             n=int(sel.sum())))
        res["rho_bins"] = bins
        tr = [b for b in bins if not b["heldout"]]
        ho = [b for b in bins if b["heldout"]]
        res["slope_err_train"] = float(np.mean([abs(b["slope"] - b["rho_mean"]) for b in tr]))
        res["slope_err_heldout"] = float(np.mean([abs(b["slope"] - b["rho_mean"]) for b in ho]))
        res["slope_err_vs_zero"] = float(np.mean([abs(b["slope"]) for b in bins]))
        sel = ok & ~cross
        res["slope_same_side"], _ = fit_slope(resid[sel], dev_[sel])
        # global decomposition on crossing items: dev = beta*rho*resid + gamma*resid
        sel = ok & cross
        A = np.stack([rho[sel] * resid[sel], resid[sel]], 1)
        coef, *_ = np.linalg.lstsq(A, dev_[sel], rcond=None)
        res["cross_beta"], res["cross_gamma"] = float(coef[0]), float(coef[1])
        if return_preds:
            preds = dict(cf=cf.astype(np.float32), l2=l2.astype(np.float32),
                         oracle_cf=orc["cf"].astype(np.float32),
                         oracle_cf_blind=orc["cf_blind"].astype(np.float32),
                         oracle_l2=orc["l2"].astype(np.float32), rho=rho, cross=cross,
                         xs=xs, xp=xp, ys=ys, m_final=m.astype(np.float32))
            return res, preds
        return res
