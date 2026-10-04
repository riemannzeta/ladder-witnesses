"""E4 -- the Causal Hierarchy Theorem as an observable: identifiability tracking (2 x 2).

Binary (Z, X, Y), theta ~ Dirichlet(alpha * 1_8) over cells (cell = 4z + 2x + y). The query is
p1 = P(Y=1 | do(X=1)). Two binary condition tokens:
  graph:      CONF (Z->X, Z->Y, X->Y)  -> p1 = sum_z P(y=1|x=1,z) P(z)
              MED  (X->Z->Y, X->Y)     -> p1 = P(y=1|x=1)
  visibility: OBS (samples are (z,x,y) cells)  |  HID (samples are (x,y) cells only)

HID-CONF is the one non-identified cell: the (x,y) data never pin p1 down, so the Bayes posterior
over p1 stops contracting. HID-MED uses exactly the same (x,y) tokens and data distribution but is
identified, which controls for token format. The head is a categorical over N_BINS bins of p1;
training targets are the one-hot bin of the episode's TRUE p1, so the cross-entropy minimizer is the
(binned) Bayes posterior over p1 given the context.

Exact posteriors (verified against importance sampling):
  OBS-*:   theta | data ~ Dirichlet(alpha + n)  (8 cells)
  HID-MED: P(y=1|x=1) | data ~ Beta(2a + n_11, 2a + n_10)  (aggregated Dirichlet; exact CDF)
  HID-CONF: margin over (x,y) ~ Dirichlet(2a + n_xy) and the within-cell z-splits stay at their
            Beta(a, a) prior (Dirichlet aggregation + neutrality); p1 by Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import betainc

from witnesses.tasks.base import HeadSpec, Task, kl_categorical_bits, run_model, softmax
from witnesses.tasks.common import categorical, dirichlet

BOS, OBS, HID, CONF, MED = 0, 1, 2, 3, 4
CELL8, CELL4 = 5, 13
VOCAB = 17
N_BINS = 64
EVAL_K = (0, 4, 8, 16, 32, 64, 128, 256)
CELLS = (("obs", "conf"), ("obs", "med"), ("hid", "conf"), ("hid", "med"))


def p1_true(th: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    """th [B, 8] (cell = 4z+2x+y) -> P(Y=1 | do(X=1)) under CONF or MED."""
    t = th.view(-1, 2, 2, 2)
    adj = (t[:, :, 1, 1] / t[:, :, 1, :].sum(-1) * t.sum((2, 3))).sum(1)
    crude = t[:, :, 1, 1].sum(1) / t[:, :, 1, :].sum((1, 2))
    return torch.where(conf, adj, crude)


def p1_conf_np(t: np.ndarray) -> np.ndarray:
    """t [..., 2, 2, 2] (z,x,y) -> adjusted P(Y=1|do(X=1))."""
    return (t[..., :, 1, 1] / t[..., :, 1, :].sum(-1) * t.sum((-1, -2))).sum(-1)


class E4Identify(Task):
    name = "e4_identify"
    vocab_size = VOCAB
    headline = "kl_post"

    def __init__(self, alpha: float = 0.5, n_samples: int = 256, mc: int = 4096):
        self.alpha = alpha
        self.N = n_samples
        self.mc = mc
        self.max_len = 3 + n_samples
        self.heads = {"post": HeadSpec(N_BINS, "soft_ce")}
        self.edges = np.linspace(0.0, 1.0, N_BINS + 1)

    def eval_signature(self) -> dict:
        return {"alpha": self.alpha, "N": self.N, "mc": self.mc}

    # ------------------------------------------------------------------ generation
    def _episodes(self, B: int, device) -> dict:
        th = dirichlet(torch.full((8,), self.alpha), B, device)
        cells = categorical(th, self.N)
        hid = torch.rand(B, device=device) < 0.5
        conf = torch.rand(B, device=device) < 0.5
        return dict(theta=th, cells=cells, hid=hid, conf=conf, p1=p1_true(th, conf))

    def _tokens(self, ep: dict) -> torch.Tensor:
        B = ep["cells"].shape[0]
        vis = torch.where(ep["hid"], HID, OBS)
        g = torch.where(ep["conf"], CONF, MED)
        samp = torch.where(ep["hid"][:, None], CELL4 + ep["cells"] % 4, CELL8 + ep["cells"])
        head = torch.stack([torch.full_like(vis, BOS), vis, g], 1)
        return torch.cat([head, samp], 1)

    def sample_batch(self, batch_size: int, device) -> dict:
        ep = self._episodes(batch_size, device)
        tgt = torch.zeros(batch_size, 1, N_BINS, device=device)
        b = (ep["p1"] * N_BINS).long().clamp(0, N_BINS - 1)
        tgt[torch.arange(batch_size, device=device), 0, b] = 1.0
        pos = torch.arange(self.max_len, device=device)
        return dict(tokens=self._tokens(ep), targets={"post": tgt},
                    masks={"post": (pos >= 2).expand(batch_size, -1)})

    # ------------------------------------------------------------------ exact posteriors
    def _posterior_hist(self, cells: np.ndarray, hid: bool, conf: bool, k: int,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Binned posterior over p1 for each episode given its first k samples, plus samples' sd."""
        a = self.alpha
        n8 = np.zeros((cells.shape[0], 8))
        np.add.at(n8, (np.repeat(np.arange(cells.shape[0]), k), cells[:, :k].ravel()), 1)
        n = n8.reshape(-1, 2, 2, 2)
        if not conf:  # identified as P(y=1|x=1) ~ Beta(2a + n_.11, 2a + n_.10) (OBS and HID)
            a1 = 2 * a + n[:, :, 1, 1].sum(1)
            b1 = 2 * a + n[:, :, 1, 0].sum(1)
            cdf = betainc(a1[:, None], b1[:, None], self.edges[None, :])
            hist = np.diff(cdf, axis=1)
            mean = a1 / (a1 + b1)
            sd = np.sqrt(a1 * b1 / ((a1 + b1) ** 2 * (a1 + b1 + 1)))
            return hist, sd
        S = self.mc
        if not hid:
            g = rng.gamma((n8 + a)[:, None, :], size=(n8.shape[0], S, 8))
            t = (g / g.sum(-1, keepdims=True)).reshape(-1, S, 2, 2, 2)
        else:
            nxy = n.sum(1).reshape(-1, 4)  # [B, 4] over (x,y)
            g = rng.gamma((nxy + 2 * a)[:, None, :], size=(nxy.shape[0], S, 4))
            marg = (g / g.sum(-1, keepdims=True)).reshape(-1, S, 2, 2)
            pi = rng.beta(a, a, size=marg.shape)  # P(z=1 | x, y) split, untouched by the data
            t = np.stack([marg * (1 - pi), marg * pi], 2)  # [B, S, z, x, y]
        p = p1_conf_np(t)
        hist = np.stack([np.histogram(row, bins=self.edges)[0] for row in p]) / S
        return hist, p.std(1)

    def make_eval_sets(self, seed: int, n_episodes: int) -> dict:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        per = max(64, n_episodes // 4)
        sets = {}
        for vis, g in CELLS:
            ep = self._episodes(per, torch.device("cpu"))
            ep["hid"][:] = vis == "hid"
            ep["conf"][:] = g == "conf"
            ep["p1"] = p1_true(ep["theta"], ep["conf"])
            ep = {k: v.cpu() for k, v in ep.items()}
            cells = ep["cells"].numpy()
            hists, sds = [], []
            for k in EVAL_K:
                if k > self.N:
                    continue
                h, s = self._posterior_hist(cells, vis == "hid", g == "conf", k, rng)
                hists.append(h)
                sds.append(s)
            ep["oracle_hist"] = torch.as_tensor(np.stack(hists, 1), dtype=torch.float32)
            ep["oracle_sd"] = torch.as_tensor(np.stack(sds, 1), dtype=torch.float32)
            # natural (Manski) bounds width for p1 from the empirical (x,y) margin at k = N
            x1 = (cells % 4) // 2 == 1
            ep["bounds_width"] = torch.as_tensor(1.0 - x1.mean(1), dtype=torch.float32)
            sets[f"{vis}_{g}"] = ep
        return sets

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, model, eval_sets: dict, device, return_preds: bool = False):
        res: dict = {}
        preds: dict = {}
        ks = [k for k in EVAL_K if k <= self.N]
        centers = 0.5 * (self.edges[1:] + self.edges[:-1])
        all_kl = []
        for vis, g in CELLS:
            name = f"{vis}_{g}"
            ep = eval_sets[name]
            out = run_model(model, self._tokens(ep), None, device)
            pm = softmax(out["post"][:, [2 + k for k in ks], :])  # [B, len(ks), N_BINS]
            oh = ep["oracle_hist"].numpy().astype(np.float64)
            kl = kl_categorical_bits(oh, pm)  # [B, len(ks)]
            all_kl.append(kl[:, 1:])  # k >= 4
            mu = (pm * centers).sum(-1)
            sd_m = np.sqrt(np.maximum((pm * centers**2).sum(-1) - mu**2, 0))
            sd_o = ep["oracle_sd"].numpy().astype(np.float64)
            # central 80% interval coverage of the TRUE p1 under the model's binned posterior
            cdf = np.cumsum(pm, -1)
            lo = self.edges[np.argmax(cdf >= 0.1, -1)]
            hi = self.edges[np.argmax(cdf >= 0.9, -1) + 1]
            p1 = ep["p1"].numpy()[:, None]
            cov = ((p1 >= lo) & (p1 <= hi)).mean(0)
            big = [i for i, k in enumerate(ks) if k >= 32]
            kk = np.log(np.array([ks[i] for i in big], float))
            slope_m = float(np.polyfit(kk, np.log(np.median(sd_m[:, big], 0)), 1)[0])
            slope_o = float(np.polyfit(kk, np.log(np.median(sd_o[:, big], 0)), 1)[0])
            res[f"{name}_kl"] = float(kl[:, 1:].mean())
            res[f"{name}_sd_ratio"] = (np.median(sd_m, 0) / np.median(sd_o, 0)).tolist()
            res[f"{name}_sd_model"] = np.median(sd_m, 0).tolist()
            res[f"{name}_sd_oracle"] = np.median(sd_o, 0).tolist()
            res[f"{name}_loglog_slope_model"] = slope_m
            res[f"{name}_loglog_slope_oracle"] = slope_o
            res[f"{name}_cov80"] = cov.tolist()
            if vis == "hid" and g == "conf":
                bw = ep["bounds_width"].numpy()
                res["hid_conf_sd_over_bounds"] = float(np.median(sd_m[:, -1] / np.maximum(bw, 1e-6)))
            if return_preds:
                preds[f"{name}_model_hist"] = pm.astype(np.float32)
                preds[f"{name}_oracle_hist"] = oh.astype(np.float32)
                preds[f"{name}_p1"] = ep["p1"].numpy()
        res["eval_k"] = ks
        res["kl_post"] = float(np.mean([k.mean() for k in all_kl]))
        if return_preds:
            return res, preds
        return res
