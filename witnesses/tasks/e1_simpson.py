"""E1 -- L2 demonstration: Pearl's Simpson pair, generalized to all six complete orderings.

Three binary variables with labels L0, L1, L2. Each episode draws a joint theta ~ Dirichlet(alpha
* 1_8) over the 8 cells and N iid samples. A query names a treatment label X and an outcome label
Y (C is the remaining label) and asks for P(Y=1 | do(X=x)), x in {0, 1}, under a causal ordering
given as a token. Every complete ordering is Markov-equivalent to every other, and under the
Dirichlet-joint prior the data carry no information about the ordering (Geiger & Heckerman 1997),
so the ordering token is the *only* source of the L2 information. Depending on the ordering, the
interventional quantity is one of three associational functionals of theta:

    ADJ   = sum_c P(y|x,c) P(c)   (C before X before Y)
    CRUDE = P(y|x)                (X first)
    MARG  = P(y)                  (Y before X)

Matched L1 control: half of the episodes instead name the functional directly (FUNC tokens),
with the same per-functional frequencies (1/6, 2/6, 3/6) as the L2 queries, so the causal step
(ordering + roles -> functional) can be compared against identical arithmetic requested by name.

Sequence: [BOS][Q][TREAT_X][OUT_Y] s_1 ... s_N.  The `ans` head (two Bernoulli logits, for x = 0
and x = 1) is read at positions 3 .. 3+N (k = 0..N samples seen); the `nxt` head predicts the next
sample (L1 sanity check).
"""

from __future__ import annotations

import numpy as np
import torch

from witnesses.tasks.base import (
    HeadSpec,
    Task,
    fit_slope,
    kl_bernoulli_bits,
    kl_categorical_bits,
    run_model,
    sigmoid,
    softmax,
)
from witnesses.tasks.common import categorical, dirichlet

PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
W = (4, 2, 1)  # cell index = 4*v(L0) + 2*v(L1) + v(L2)
BOS, CELL0, ORD0, FUNC0, TREAT0, OUT0 = 0, 1, 9, 15, 18, 21
VOCAB = 24
ADJ, CRUDE, MARG = 0, 1, 2
EST_NAMES = ("adj", "crude", "marg")
FUNC_P = (1 / 6, 2 / 6, 3 / 6)  # P(estimand) under a uniform random ordering, for any (X, Y)
PREFIX = 4
K_MIN = 8  # positions with k >= K_MIN samples enter averaged metrics


def estimand(perm: tuple[int, int, int], x: int, y: int) -> int:
    pos = {lab: i for i, lab in enumerate(perm)}
    if pos[y] < pos[x]:
        return MARG
    if pos[x] == 0:
        return CRUDE
    return ADJ


EST_TABLE = np.full((6, 3, 3), -1, dtype=np.int64)
for _p, _perm in enumerate(PERMS):
    for _x in range(3):
        for _y in range(3):
            if _x != _y:
                EST_TABLE[_p, _x, _y] = estimand(_perm, _x, _y)


def perm_for(est: int, x: int, y: int) -> int:
    """Index of the first ordering that yields estimand `est` for roles (x, y)."""
    for p in range(6):
        if EST_TABLE[p, x, y] == est:
            return p
    raise ValueError


def cxy_index(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[B] roles -> [B, 8] cell indices in (c, x, y) order (c = remaining label)."""
    c = 3 - x - y
    w = torch.tensor(W, device=x.device)
    wc, wx, wy = w[c], w[x], w[y]
    bits = torch.tensor([[a, b, d] for a in (0, 1) for b in (0, 1) for d in (0, 1)], device=x.device)
    return bits[:, 0] * wc[:, None] + bits[:, 1] * wx[:, None] + bits[:, 2] * wy[:, None]


def functionals_true(th_cxy: torch.Tensor) -> torch.Tensor:
    """th_cxy [B,2,2,2] -> [B,3,2]: (ADJ, CRUDE, MARG) x (x=0, x=1) for P(Y=1 | .)."""
    pc = th_cxy.sum((2, 3))
    pcx = th_cxy.sum(3)
    adj = (th_cxy[..., 1] / pcx * pc[:, :, None]).sum(1)
    crude = th_cxy[..., 1].sum(1) / th_cxy.sum((1, 3))
    marg = th_cxy[..., 1].sum((1, 2))[:, None].expand(-1, 2)
    return torch.stack([adj, crude, marg], 1)


def functionals_posterior(n_cxy: np.ndarray, alpha: float) -> np.ndarray:
    """Posterior means under Dirichlet(alpha*1_8): n_cxy [..., 2,2,2] counts -> [..., 3, 2].

    ADJ relies on Dirichlet neutrality along C -> X -> Y (P(c) independent of P(y|x,c)).
    """
    n = n_cxy.astype(np.float64)
    tot = n.sum((-1, -2, -3))[..., None]
    pc = (n.sum((-1, -2)) + 4 * alpha) / (tot + 8 * alpha)  # [..., 2(c)]
    py_cx = (n[..., 1] + alpha) / (n.sum(-1) + 2 * alpha)  # [..., 2(c), 2(x)]
    adj = (py_cx * pc[..., :, None]).sum(-2)
    crude = (n[..., 1].sum(-2) + 2 * alpha) / (n.sum((-1, -3)) + 4 * alpha)
    marg = (n[..., 1].sum((-1, -2)) + 4 * alpha) / (tot[..., 0] + 8 * alpha)
    marg = np.repeat(marg[..., None], 2, -1)
    return np.stack([adj, crude, marg], -2)


class E1Simpson(Task):
    name = "e1_simpson"
    vocab_size = VOCAB
    headline = "kl_all"

    def __init__(self, alpha: float = 0.3, n_samples: int = 128, shuffle_order: bool = False,
                 p_func: float = 0.5):
        self.alpha = alpha
        self.N = n_samples
        self.shuffle_order = shuffle_order
        self.p_func = p_func
        self.max_len = PREFIX + n_samples
        self.heads = {"ans": HeadSpec(2, "soft_bce"), "nxt": HeadSpec(8, "soft_ce", 0.5)}

    def eval_signature(self) -> dict:
        return {"alpha": self.alpha, "N": self.N, "p_func": self.p_func}

    # ------------------------------------------------------------------ generation
    def _episodes(self, B: int, device: torch.device) -> dict:
        th = dirichlet(torch.full((8,), self.alpha), B, device)
        cells = categorical(th, self.N)
        x = torch.randint(0, 3, (B,), device=device)
        y = (x + torch.randint(1, 3, (B,), device=device)) % 3
        perm = torch.randint(0, 6, (B,), device=device)
        is_func = torch.rand(B, device=device) < self.p_func
        func = torch.multinomial(torch.tensor(FUNC_P, device=device).expand(B, -1), 1)[:, 0]
        est_table = torch.as_tensor(EST_TABLE, device=device)
        est = torch.where(is_func, func, est_table[perm, x, y])
        idx = cxy_index(x, y)
        th_cxy = th.gather(1, idx).view(B, 2, 2, 2)
        true = functionals_true(th_cxy)[torch.arange(B, device=device), est]  # [B, 2]
        return dict(theta=th, cells=cells, x=x, y=y, perm=perm, is_func=is_func, est=est,
                    true=true)

    def _tokens(self, ep: dict, perm_shown: torch.Tensor) -> torch.Tensor:
        B = ep["cells"].shape[0]
        q = torch.where(ep["is_func"], FUNC0 + ep["est"], ORD0 + perm_shown)
        head = torch.stack([torch.full_like(q, BOS), q, TREAT0 + ep["x"], OUT0 + ep["y"]], 1)
        return torch.cat([head, CELL0 + ep["cells"]], 1).view(B, -1)

    def sample_batch(self, batch_size: int, device: torch.device) -> dict:
        ep = self._episodes(batch_size, device)
        shown = torch.randint(0, 6, (batch_size,), device=device) if self.shuffle_order else ep["perm"]
        tokens = self._tokens(ep, shown)
        T = self.max_len
        pos = torch.arange(T, device=device)
        m_ans = (pos >= PREFIX - 1).expand(batch_size, -1)
        m_nxt = ((pos >= PREFIX - 1) & (pos < T - 1)).expand(batch_size, -1)
        return dict(
            tokens=tokens,
            targets={"ans": ep["true"][:, None, :], "nxt": ep["theta"][:, None, :]},
            masks={"ans": m_ans, "nxt": m_nxt},
        )

    # ------------------------------------------------------------------ eval sets
    def make_eval_sets(self, seed: int, n_episodes: int) -> dict:
        torch.manual_seed(seed)
        cpu = torch.device("cpu")
        main = {k: v.cpu() for k, v in self._episodes(n_episodes, cpu).items()}
        wit = {k: v.cpu() for k, v in self._episodes(max(256, n_episodes // 4), cpu).items()}
        wit["is_func"][:] = False
        # reversal set: rejection-sample contexts whose posterior ADJ and CRUDE ATEs (at k=N)
        # have opposite signs, both with |ATE| > 0.02
        keep, need = [], max(128, n_episodes // 5)
        while sum(len(k["x"]) for k in keep) < need:
            ep = {k: v.cpu() for k, v in self._episodes(4 * need, cpu).items()}
            post = self._posterior_final(ep)  # [B,3,2]
            ate_adj = post[:, ADJ, 1] - post[:, ADJ, 0]
            ate_cru = post[:, CRUDE, 1] - post[:, CRUDE, 0]
            ok = (np.sign(ate_adj) != np.sign(ate_cru)) & (np.abs(ate_adj) > 0.02) & (
                np.abs(ate_cru) > 0.02)
            sel = torch.as_tensor(np.nonzero(ok)[0])
            keep.append({k: v[sel] for k, v in ep.items()})
        rev = {k: torch.cat([d[k] for d in keep])[:need] for k in keep[0]}
        rev["is_func"][:] = False
        return {"main": main, "witness": wit, "reversal": rev}

    # ------------------------------------------------------------------ oracles
    def _counts_cxy(self, ep: dict) -> np.ndarray:
        """Cumulative counts at k = 0..N in (c, x, y) order -> [B, N+1, 2, 2, 2]."""
        cells = ep["cells"].numpy()
        B, N = cells.shape
        oh = np.zeros((B, N + 1, 8), np.int32)
        oh[np.arange(B)[:, None], np.arange(1, N + 1)[None, :], cells] = 1
        cum = oh.cumsum(1)
        idx = cxy_index(ep["x"], ep["y"]).numpy()  # [B, 8]
        return np.take_along_axis(cum, idx[:, None, :], 2).reshape(B, N + 1, 2, 2, 2)

    def _posterior_final(self, ep: dict) -> np.ndarray:
        n = self._counts_cxy(ep)[:, -1]
        return functionals_posterior(n, self.alpha)

    def oracle(self, ep: dict) -> dict:
        n = self._counts_cxy(ep)
        F = functionals_posterior(n, self.alpha)  # [B, N+1, 3, 2]
        B = F.shape[0]
        est = ep["est"].numpy()
        exact = F[np.arange(B), :, est]  # [B, N+1, 2]
        blind = np.tensordot(F, np.array(FUNC_P), axes=([2], [0]))  # mixture over orderings
        blind = np.where(ep["is_func"].numpy()[:, None, None], exact, blind)
        cells = ep["cells"].numpy()
        oh = np.zeros((B, cells.shape[1] + 1, 8))
        oh[np.arange(B)[:, None], np.arange(1, cells.shape[1] + 1)[None, :], cells] = 1
        cum = oh.cumsum(1)
        nxt = (cum + self.alpha) / (cum.sum(-1, keepdims=True) + 8 * self.alpha)
        return {"ans": exact, "blind": blind, "nxt": nxt, "all": F}

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, model, eval_sets: dict, device: torch.device, return_preds: bool = False):
        m: dict[str, float] = {}
        preds: dict[str, np.ndarray] = {}
        P = PREFIX - 1
        # main ---------------------------------------------------------------
        ep = eval_sets["main"]
        orc = self.oracle(ep)
        out = run_model(model, self._tokens(ep, ep["perm"]), None, device)
        p_ans = sigmoid(out["ans"][:, P:, :])  # [B, N+1, 2]
        p_nxt = softmax(out["nxt"][:, P:-1, :])  # [B, N, 8]
        kl_m = kl_bernoulli_bits(orc["ans"], p_ans).sum(-1)  # [B, N+1]
        kl_b = kl_bernoulli_bits(orc["ans"], orc["blind"]).sum(-1)
        ks = slice(K_MIN, None)
        is_func = ep["is_func"].numpy()
        est = ep["est"].numpy()
        m["kl_all"] = float(kl_m[:, ks].mean())
        for qname, sel in (("l2", ~is_func), ("func", is_func)):
            m[f"kl_{qname}"] = float(kl_m[sel][:, ks].mean())
            for e, en in enumerate(EST_NAMES):
                s2 = sel & (est == e)
                m[f"kl_{qname}_{en}"] = float(kl_m[s2][:, ks].mean())
        l2 = ~is_func
        m["kl_blind_l2"] = float(kl_b[l2][:, ks].mean())
        m["gc_l2"] = 1.0 - m["kl_l2"] / m["kl_blind_l2"]
        for e, en in enumerate(EST_NAMES):
            s2 = l2 & (est == e)
            m[f"gc_l2_{en}"] = 1.0 - kl_m[s2][:, ks].mean() / kl_b[s2][:, ks].mean()
        m["kl_nxt"] = float(kl_categorical_bits(orc["nxt"][:, :-1], p_nxt).mean())
        m["kl_curve_l2"] = kl_m[l2].mean(0).tolist()
        m["kl_curve_func"] = kl_m[is_func].mean(0).tolist()
        m["kl_curve_blind"] = kl_b[l2].mean(0).tolist()
        if return_preds:
            preds.update(main_ans=p_ans.astype(np.float32), main_oracle=orc["ans"].astype(np.float32),
                         main_blind=orc["blind"].astype(np.float32), main_est=est,
                         main_is_func=is_func)
        # witness: same data under all six orderings ---------------------------
        ep = eval_sets["witness"]
        orc_all = self.oracle(ep)["all"][:, -1]  # [B, 3, 2] at k = N
        x, y = ep["x"].numpy(), ep["y"].numpy()
        Bw = len(x)
        mod = np.zeros((Bw, 6, 2))
        orc_w = np.zeros((Bw, 6, 2))
        for p in range(6):
            tok = self._tokens(ep, torch.full((Bw,), p, dtype=torch.long))
            o = run_model(model, tok, None, device)
            mod[:, p] = sigmoid(o["ans"][:, -1, :])
            orc_w[:, p] = orc_all[np.arange(Bw), EST_TABLE[p, x, y]]
        dm = mod - mod.mean(1, keepdims=True)
        do = orc_w - orc_w.mean(1, keepdims=True)
        spread = orc_w.max(1) - orc_w.min(1)  # [B, 2]
        sel = spread.max(1) >= 0.02
        m["split_slope"], m["split_r2"] = fit_slope(do[sel], dm[sel])
        m["split_n"] = int(sel.sum())
        if return_preds:
            preds.update(witness_model=mod.astype(np.float32), witness_oracle=orc_w.astype(np.float32))
        # reversal: Simpson sign flips --------------------------------------------
        ep = eval_sets["reversal"]
        post = self._posterior_final(ep)
        x, y = ep["x"].numpy(), ep["y"].numpy()
        Br = len(x)
        agree = {}
        for e, en in ((ADJ, "adj"), (CRUDE, "crude")):
            perms = torch.as_tensor([perm_for(e, int(a), int(b)) for a, b in zip(x, y)])
            o = run_model(model, self._tokens(ep, perms), None, device)
            pm = sigmoid(o["ans"][:, -1, :])
            agree[en] = np.sign(pm[:, 1] - pm[:, 0]) == np.sign(post[:, e, 1] - post[:, e, 0])
            m[f"reversal_agree_{en}"] = float(agree[en].mean())
        m["reversal_agree_both"] = float((agree["adj"] & agree["crude"]).mean())
        m["reversal_n"] = Br
        if return_preds:
            return m, preds
        return m
