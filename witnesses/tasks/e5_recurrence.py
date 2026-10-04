"""E5 -- the witness pairs ported into Misra's modular-recurrence wind tunnel.

Base task (bayesian-wind-tunnel experiments/construction_boundary/recurrence_bwt.py): p = 17,
T = 16 values x_0 .. x_15, prior pi = 0.5 on
    H_P (program): (a, b) ~ U(Z_17^2) (a = 0 included), x_0 ~ U(Z_17), x_{t+1} = a x_t + b mod 17
    H_R (noise):   x_t iid U(Z_17).
Each position is ONE token, id = flag * 17 + value (no inserted tokens, as upstream; flags are
Dawid regime indicators). Targets are PFN-style soft targets: the episode's true next-token
distribution given its latents, so the CE minimizer is the Bayes posterior predictive.

E5a (L2 witness pair, restart vs clock). One intervention (prob p_interv) at s ~ U{1..T-2} with
value v ~ U(Z_17) drawn independently of everything (the natural value is NOT excluded); flag 1.
  restart: x_s = v and the recurrence continues from v      (x_{s+1} = f(v))
  clock:   x_s = v and every other x_t = f^t(x_0)           (x_{s+1} = f(f(x_{s-1})))
  noise:   x_s = v, others iid.
Unintervened episodes are the same draws under both semantics (identical L1, different L2).
Head `nxt` predicts x_{t+1} at t = 0..T-2, except the prediction OF x_s (exogenous).

E5b (L3 witness pair, abducted vs fresh per-step noise). Factual H_P is noisy:
x_{t+1} = a x_t + b + eps_t, eps_t ~ Bern(eps). A query position q ~ U{0..T-4} is marked IN PLACE
(flag 2 on x_q); then [CF] [v] [y_1] [y_2] [y_3] with v ~ U(Z_17 \\ {x_q}) carried as a flag-1
("set") token and y_k as plain value tokens. Counterfactual labels:
  abducted: H_P y_1 = a v + b + eps_q, y_k = a y_{k-1} + b + eps_{q+k-1} (the episode's ACTUAL
            noise -> one-hot targets); H_R y_k = x_{q+k} (noise held fixed -> copy)
  fresh:    H_P y_k = a y_{k-1} + b + eps'_k with fresh eps' (two-point targets); H_R uniform.
The factual stream is generated identically under both labelings (identical L1/L2, different
L3). A fixed 20% of the 289 programs never appears in training episodes; eval reports seen vs
held-out programs. Heads: `nxt` (factual next token, positions 0..T-2) and `cf` (y_k predicted
at the positions of [v], [y_1], [y_2], teacher-forced).

Oracles are exact: numpy float64, vectorized over episodes x {289 programs, H_R}.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from witnesses.tasks.base import HeadSpec, Task, kl_categorical_bits, run_model, softmax

P = 17
K = P * P  # programs (a, b), index k = a * P + b
PROG_A = np.arange(K) // P
PROG_B = np.arange(K) % P
PROG_A2 = (PROG_A * PROG_A) % P  # f(f(x)) = a^2 x + (a b + b)
PROG_AB = (PROG_A * PROG_B + PROG_B) % P
CF_TOKEN = 3 * P  # E5b only; ids 0..50 are flag * 17 + value
N_CF = 3  # counterfactual steps y_1..y_3
TV_DISC = 0.1  # "discriminative" positions: TV(own oracle, other oracle) > TV_DISC
N_SAVE = 1000  # episodes whose full predictive distributions go into eval_preds.npz
NAN = float("nan")


# =============================================================================== oracle helpers


def _prior_weights(pi: float, prior_prog) -> np.ndarray:
    """Prior mass of each program: pi spread uniformly over the allowed programs."""
    if prior_prog is None:
        return np.full(K, pi / K)
    m = np.asarray(prior_prog, np.float64)
    return pi * m / m.sum()


def _scatter(w: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """w, vals [n, H, L] (weight of hypothesis h on value vals) -> [n, L, P] summed by value."""
    n, _, L = w.shape
    base = (np.arange(n)[:, None, None] * L + np.arange(L)[None, None, :]) * P
    out = np.bincount((base + vals).ravel(), weights=w.ravel(), minlength=n * L * P)
    return out.reshape(n, L, P)


def _normalize(wp: np.ndarray, wr: np.ndarray):
    """Normalize program weights wp [n, K, ...] and noise weight wr [n, ...] jointly over
    hypotheses. If every hypothesis has zero weight (only possible with pi = 1), fall back to
    H_R (uniform predictive)."""
    Z = wp.sum(1) + wr
    ok = Z > 0
    Zs = np.where(ok, Z, 1.0)
    return wp / Zs[:, None], np.where(ok, wr / Zs, 1.0)


# --------------------------------------------------------------------------- E5a: noiseless


def recurrence_oracle(x, s, mode: str, pi: float = 0.5, prior_prog=None, chunk: int = 1024) -> dict:
    """Exact posterior predictive for E5a.

    x [B, T] values; s [B] intervention position (-1 = none); mode:
      "restart": the transition INTO s is not evidence; all others (including out of s) are.
      "clock":   transitions into and out of s are not evidence; x_{s+1} = f(f(x_{s-1})) is.
      "l1":      flag-blind observational posterior: every transition is evidence.
    Under H_P(a, b) the predictive for x_{t+1} is a point mass (f(x_t); f(f(x_{s-1})) for the
    clock prediction of x_{s+1}); under H_R it is uniform. The prediction OF x_s (masked in
    training) uses the natural value f(x_{s-1}).
    Returns pred [B, T-1, P] (position t predicts x_{t+1}), post_prog [B, T] (P(H_P | x_0..x_t)),
    n_cons [B, T] (number of programs consistent with x_0..x_t).
    """
    assert mode in ("restart", "clock", "l1"), mode
    x = np.asarray(x, np.int64)
    s = np.asarray(s, np.int64)
    B, T = x.shape
    wprior = _prior_weights(pi, prior_prog)
    tau = np.arange(T)
    pred = np.empty((B, T - 1, P))
    post = np.empty((B, T))
    ncons = np.empty((B, T), np.int64)
    for i in range(0, B, chunk):
        xb, sb = x[i : i + chunk], s[i : i + chunk]
        n = len(xb)
        prev = np.concatenate([xb[:, :1], xb[:, :-1]], 1)  # x_{tau-1} (column 0 unused)
        pn = (PROG_A[None, :, None] * prev[:, None, :] + PROG_B[None, :, None]) % P  # [n,K,T]
        valid = np.broadcast_to(tau >= 1, (n, T)).copy()  # x_tau is evidence (x_0 cancels)
        if mode != "l1":
            valid &= tau[None, :] != sb[:, None]  # x_s exogenous (s = -1 never matches)
            if mode == "clock":
                r = np.nonzero((sb >= 1) & (sb + 1 < T))[0]
                pn[r, :, sb[r] + 1] = (PROG_A2[None, :] * xb[r, sb[r] - 1][:, None]
                                       + PROG_AB[None, :]) % P
        sat = (pn == xb[:, None, :]) | ~valid[:, None, :]
        cons = np.logical_and.accumulate(sat, axis=2)  # consistent with x_0..x_t
        m = np.cumsum(valid, 1)  # natural tokens after x_0 in x_0..x_t (H_R pays 1/P each)
        wp = wprior[None, :, None] * cons
        wr = (1.0 - pi) * np.power(float(P), -m)
        wp, wr = _normalize(wp, wr)
        pred[i : i + n] = _scatter(wp[:, :, :-1], pn[:, :, 1:]) + (wr[:, :-1] / P)[:, :, None]
        post[i : i + n] = wp.sum(1)
        ncons[i : i + n] = cons.sum(1)
    return {"pred": pred, "post_prog": post, "n_cons": ncons}


# --------------------------------------------------------------------------- E5b: noisy


def _noisy_lik(x: np.ndarray, eps: float):
    """f [n,K,T-1] = a x_t + b; r = (x_{t+1} - f) mod P (the abducted noise of each program);
    lik = P(x_{t+1} | x_t, program) = 1-eps (r = 0), eps (r = 1), else 0."""
    f = (PROG_A[None, :, None] * x[:, None, :-1] + PROG_B[None, :, None]) % P
    r = (x[:, None, 1:] - f) % P
    lik = np.where(r == 0, 1.0 - eps, np.where(r == 1, eps, 0.0))
    return f, r, lik


def noisy_factual_oracle(x, eps: float = 0.2, pi: float = 0.5, prior_prog=None,
                         chunk: int = 1024) -> dict:
    """E5b factual (Bayes-L1) posterior predictive. x [B, T] -> pred [B, T-1, P], post_prog [B, T].
    Per-transition likelihood 1-eps / eps / 0 under H_P(a, b); 1/P per token under H_R."""
    x = np.asarray(x, np.int64)
    B, T = x.shape
    wprior = _prior_weights(pi, prior_prog)
    pred = np.empty((B, T - 1, P))
    post = np.empty((B, T))
    wr0 = (1.0 - pi) * np.power(float(P), -np.arange(T))[None, :]
    for i in range(0, B, chunk):
        xb = x[i : i + chunk]
        n = len(xb)
        f, _, lik = _noisy_lik(xb, eps)
        cum = np.concatenate([np.ones((n, K, 1)), np.cumprod(lik, 2)], 2)  # [n, K, T]
        wp, wr = _normalize(wprior[None, :, None] * cum, np.broadcast_to(wr0, (n, T)))
        w = wp[:, :, :-1]
        pred[i : i + n] = _scatter(np.concatenate([w * (1 - eps), w * eps], 1),
                                   np.concatenate([f, (f + 1) % P], 1)) + (wr[:, :-1] / P)[..., None]
        post[i : i + n] = wp.sum(1)
    return {"pred": pred, "post_prog": post}


def noisy_posterior(x, t_post, eps: float = 0.2, pi: float = 0.5, prior_prog=None):
    """Posterior over hypotheses after x_0..x_{t_post} (t_post [B] or int).
    Returns (w_prog [B, K], w_noise [B]), normalized."""
    x = np.asarray(x, np.int64)
    B, T = x.shape
    tp = np.broadcast_to(np.asarray(t_post, np.int64), (B,))
    _, _, lik = _noisy_lik(x, eps)
    cum = np.concatenate([np.ones((B, K, 1)), np.cumprod(lik, 2)], 2)
    wp = _prior_weights(pi, prior_prog)[None, :] * cum[np.arange(B), :, tp]
    wr = (1.0 - pi) * np.power(float(P), -tp.astype(np.float64))
    return _normalize(wp, wr)


def cf_oracle(x, q, v, y, mode: str, eps: float = 0.2, pi: float = 0.5, prior_prog=None,
              chunk: int = 1024) -> np.ndarray:
    """Exact counterfactual predictive for y_1..y_n (n = y.shape[1]), teacher-forced on y.

    mode "abducted" (Bayes-L3): posterior from the full factual sequence; per program the factual
      noise is determined (eps_u = x_{u+1} - f(x_u)) and y_k = f(y_{k-1}) + eps_{q+k-1}
      (y_0 = v); under H_R y_k = x_{q+k}.
    mode "fresh" (Bayes-L3, fresh noise): same posterior; y_k ~ two-point at f(y_{k-1}); H_R
      uniform.
    mode "l2cf" (Bayes-L2-for-CF): posterior from x_0..x_q only, then as "fresh".
    The posterior is updated on the observed y's. An observation with zero probability under every
    surviving hypothesis (reachable only when an oracle is scored on the OTHER condition's
    episodes) is handled as the delta -> 0 limit of a delta-contaminated observation model:
    weights are prior x likelihood over the hypotheses with the fewest zero-probability
    observations, i.e. such an observation is ignored.
    Returns [B, n, P].
    """
    assert mode in ("abducted", "fresh", "l2cf"), mode
    x = np.asarray(x, np.int64)
    q = np.asarray(q, np.int64)
    v = np.asarray(v, np.int64)
    y = np.asarray(y, np.int64)
    B, T = x.shape
    nY = y.shape[1]
    assert (q + nY <= T - 1).all()
    wprior = _prior_weights(pi, prior_prog)
    out = np.empty((B, nY, P))
    big = np.iinfo(np.int64).max
    for i in range(0, B, chunk):
        xb, qb, vb, yb = x[i : i + chunk], q[i : i + chunk], v[i : i + chunk], y[i : i + chunk]
        n = len(xb)
        ar = np.arange(n)
        _, r, lik = _noisy_lik(xb, eps)
        cum = np.concatenate([np.ones((n, K, 1)), np.cumprod(lik, 2)], 2)
        tpost = np.full(n, T - 1) if mode != "l2cf" else qb
        wpF = wprior[None, :] * cum[ar, :, tpost]  # [n, K]
        wrF = (1.0 - pi) * np.power(float(P), -tpost.astype(np.float64))  # [n]
        alive_p, alive_r = wpF > 0, wrF > 0
        with np.errstate(divide="ignore"):
            lp, lr = np.log(wpF), np.log(wrF)
        inc_p = np.zeros((n, K), np.int64)
        inc_r = np.zeros(n, np.int64)
        par = vb
        for k in range(nY):
            u = qb + k  # CF step k+1 reuses the factual transition u -> u+1
            fpar = (PROG_A[None, :] * par[:, None] + PROG_B[None, :]) % P  # [n, K]
            minc = np.minimum(np.where(alive_p, inc_p, big).min(1), np.where(alive_r, inc_r, big))
            Lp = np.where(alive_p & (inc_p == minc[:, None]), lp, -np.inf)
            Lr = np.where(alive_r & (inc_r == minc), lr, -np.inf)
            mx = np.maximum(Lp.max(1), Lr)
            mx = np.where(np.isfinite(mx), mx, 0.0)
            wp, wr = _normalize(np.exp(Lp - mx[:, None]), np.exp(Lr - mx))
            yk = yb[:, k]
            if mode == "abducted":
                vp = (fpar + r[ar, :, u]) % P  # program point masses
                xr = xb[ar, u + 1]  # H_R: copy of the factual value
                pr = _scatter(wp[:, :, None], vp[:, :, None])[:, 0]
                pr[ar, xr] += wr
                inc_p += vp != yk[:, None]
                inc_r += xr != yk
            else:
                pr = _scatter(np.concatenate([wp * (1 - eps), wp * eps], 1)[:, :, None],
                              np.concatenate([fpar, (fpar + 1) % P], 1)[:, :, None])[:, 0]
                pr += (wr / P)[:, None]
                d = (yk[:, None] - fpar) % P
                lk = np.where(d == 0, 1.0 - eps, np.where(d == 1, eps, 0.0))
                hit = lk > 0
                lp = lp + np.log(np.where(hit, lk, 1.0))
                inc_p += ~hit
                lr = lr - np.log(P)
            out[i + ar, k] = pr
            par = yk
    return out


# =============================================================================== metric helpers


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return kl_categorical_bits(p, q)


def _tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p - q).sum(-1)


def _mean(a: np.ndarray, sel) -> float:
    sel = np.broadcast_to(sel, a.shape)
    return float(a[sel].mean()) if sel.any() else NAN


def _gc(num: np.ndarray, den: np.ndarray, sel) -> float:
    """Gap closed 1 - sum(num) / sum(den) over the selected entries (ratio of means)."""
    sel = np.broadcast_to(sel, num.shape)
    d = float(den[sel].sum()) if sel.any() else 0.0
    return 1.0 - float(num[sel].sum()) / d if d > 0 else NAN


def _curve(a: np.ndarray, sel) -> list[float]:
    sel = np.broadcast_to(sel, a.shape)
    n = sel.sum(0)
    s = np.where(sel, a, 0.0).sum(0)
    return [float(si / ni) if ni > 0 else NAN for si, ni in zip(s, n)]


def _rel_curve(a: np.ndarray, s: np.ndarray) -> list[float]:
    """a [B, L] indexed by prediction position t; mean over episodes at offsets d = t - s >= 0."""
    L = a.shape[1]
    out = []
    for d in range(L):
        t = s + d
        ok = (t >= 0) & (t < L)
        if not ok.any():
            break
        out.append(float(a[np.nonzero(ok)[0], t[ok]].mean()))
    return out


def _fingerprint(ep: dict) -> str:
    h = hashlib.sha1()
    for k in sorted(ep):
        h.update(k.encode())
        h.update(np.ascontiguousarray(ep[k].numpy()).tobytes())
    return h.hexdigest()


def _two_point(f: torch.Tensor, eps: float) -> torch.Tensor:
    return (1 - eps) * F.one_hot(f, P).float() + eps * F.one_hot((f + 1) % P, P).float()


# =============================================================================== E5a


class E5aRestartClock(Task):
    name = "e5a_recurrence"
    vocab_size = 2 * P  # flag in {0, 1}
    headline = "kl_own"

    def __init__(self, semantics: str = "restart", p_interv: float = 0.5, pi: float = 0.5,
                 seq_len: int = 16):
        assert semantics in ("restart", "clock"), semantics
        self.semantics = semantics
        self.other = "clock" if semantics == "restart" else "restart"
        self.p_interv = p_interv
        self.pi = pi
        self.T = seq_len
        self.max_len = seq_len
        self.heads = {"nxt": HeadSpec(P, "soft_ce")}
        self._orc_cache: dict = {}

    def eval_signature(self) -> dict:
        return {"p": P, "T": self.T, "pi": self.pi, "p_interv": self.p_interv,
                "semantics": self.semantics, "gen": 1}

    # ------------------------------------------------------------------ generation
    def _latents(self, B: int, device: torch.device) -> dict:
        """Every latent is drawn for every episode, so both semantics consume the RNG identically
        (same seed -> same draws -> unintervened episodes identical across conditions)."""
        T = self.T
        is_prog = torch.rand(B, device=device) < self.pi
        k = torch.randint(0, K, (B,), device=device)
        x0 = torch.randint(0, P, (B,), device=device)
        noise = torch.randint(0, P, (B, T), device=device)
        has_int = torch.rand(B, device=device) < self.p_interv
        s = torch.randint(1, T - 1, (B,), device=device)  # uniform over 1..T-2
        v = torch.randint(0, P, (B,), device=device)  # independent of everything
        s = torch.where(has_int, s, torch.full_like(s, -1))
        return dict(is_prog=is_prog, a=k // P, b=k % P, x0=x0, noise=noise, s=s, v=v)

    def _values(self, lat: dict, semantics: str) -> torch.Tensor:
        a, b, s, v = lat["a"], lat["b"], lat["s"], lat["v"]
        cur = nat = lat["x0"]
        cols = [cur]
        for t in range(1, self.T):
            nat = (a * nat + b) % P  # f^t(x_0)
            nxt = nat if semantics == "clock" else (a * cur + b) % P
            cur = torch.where(s == t, v, nxt)
            cols.append(cur)
        xp = torch.stack(cols, 1)
        pos = torch.arange(self.T, device=a.device)
        xr = torch.where(pos[None, :] == s[:, None], v[:, None], lat["noise"])
        return torch.where(lat["is_prog"][:, None], xp, xr)

    @staticmethod
    def _tokens(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        return x + P * flags

    def _batch(self, x: torch.Tensor, lat: dict) -> dict:
        B, T = x.shape
        dev = x.device
        pos = torch.arange(T, device=dev)
        s = lat["s"]
        flags = (pos[None, :] == s[:, None]).long()
        oh = torch.cat([F.one_hot(x[:, 1:], P).float(), torch.full((B, 1, P), 1.0 / P, device=dev)], 1)
        tgt = torch.where(lat["is_prog"][:, None, None], oh, torch.full_like(oh, 1.0 / P))
        mask = (pos[None, :] < T - 1) & (pos[None, :] + 1 != s[:, None])
        return dict(tokens=self._tokens(x, flags), targets={"nxt": tgt}, masks={"nxt": mask})

    def sample_batch(self, batch_size: int, device: torch.device, return_latents: bool = False):
        lat = self._latents(batch_size, device)
        x = self._values(lat, self.semantics)
        batch = self._batch(x, lat)
        return (batch, dict(lat, x=x)) if return_latents else batch

    # ------------------------------------------------------------------ eval sets
    def make_eval_sets(self, seed: int, n_episodes: int) -> dict:
        torch.manual_seed(seed)
        cpu = torch.device("cpu")
        lat = self._latents(n_episodes, cpu)
        x = self._values(lat, self.semantics)
        pos = torch.arange(self.T)
        main = dict(x=x, flags=(pos[None, :] == lat["s"][:, None]).long(), s=lat["s"], v=lat["v"],
                    is_prog=lat["is_prog"], a=lat["a"], b=lat["b"], x0=lat["x0"])
        sel = lat["s"] >= 0
        abl = {k: t[sel].clone() for k, t in main.items()}
        abl["flags"].zero_()  # flag ablation: the same intervened episodes, flags zeroed
        return {"main": main, "ablated": abl}

    # ------------------------------------------------------------------ oracles
    def oracles(self, ep: dict, modes=("restart", "clock", "l1")) -> dict:
        """{mode: posterior predictive [B, T-1, P]} for the episodes' values and intervention
        positions (independent of the flags actually shown)."""
        x, s = ep["x"].numpy(), ep["s"].numpy()
        return {mode: recurrence_oracle(x, s, mode, self.pi)["pred"] for mode in modes}

    def _cached_oracles(self, name: str, ep: dict) -> dict:
        key = (name, _fingerprint(ep))
        if key not in self._orc_cache:
            modes = (self.semantics, "l1") if name == "ablated" else (self.semantics, self.other, "l1")
            o = self.oracles(ep, modes)
            self._orc_cache[key] = {"own": o[self.semantics], "other": o.get(self.other), "l1": o["l1"]}
        return self._orc_cache[key]

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, model, eval_sets: dict, device: torch.device, return_preds: bool = False):
        m: dict = {}
        preds: dict = {}
        T = self.T
        t = np.arange(T - 1)[None, :]
        # main ---------------------------------------------------------------
        ep = eval_sets["main"]
        orc = self._cached_oracles("main", ep)
        out = run_model(model, self._tokens(ep["x"], ep["flags"]), None, device)
        pm = softmax(out["nxt"][:, : T - 1])  # [B, T-1, P]
        s = ep["s"].numpy()
        has = s >= 0
        is_prog, a = ep["is_prog"].numpy(), ep["a"].numpy()
        sup = (t + 1) != s[:, None]  # every prediction except OF the intervened token
        post = has[:, None] & (t >= s[:, None])  # predictions of x_{s+1} .. x_{T-1}
        k_own, k_oth, k_l1 = _kl(orc["own"], pm), _kl(orc["other"], pm), _kl(orc["l1"], pm)
        ref_oth, ref_l1 = _kl(orc["own"], orc["other"]), _kl(orc["own"], orc["l1"])
        tv_m = _tv(orc["own"], pm)
        tv_oth, tv_l1 = _tv(orc["own"], orc["other"]), _tv(orc["own"], orc["l1"])
        m["kl_own"] = _mean(k_own, sup)
        m["kl_own_obs"] = _mean(k_own, sup & ~has[:, None])
        m["kl_own_int"] = _mean(k_own, sup & has[:, None])
        m["kl_own_post"] = _mean(k_own, post)
        m["kl_other"] = _mean(k_oth, post)
        m["kl_l1"] = _mean(k_l1, post)
        m["ref_own_other"] = _mean(ref_oth, post)  # oracle-to-oracle gaps, same positions
        m["ref_own_l1"] = _mean(ref_l1, post)
        disc = post & (tv_oth > TV_DISC)
        groups = {"prog": is_prog & (a != 0), "prog_a0": is_prog & (a == 0), "noise": ~is_prog}
        for g, gs in groups.items():
            sel = disc & gs[:, None]
            m[f"gc_{g}"] = _gc(k_own, ref_oth, sel)
            m[f"gc_tv_{g}"] = _gc(tv_m, tv_oth, sel)
            m[f"n_disc_{g}"] = int(sel.sum())
            m[f"kl_own_post_{g}"] = _mean(k_own, post & gs[:, None])
            m[f"kl_other_{g}"] = _mean(k_oth, post & gs[:, None])
            m[f"kl_l1_{g}"] = _mean(k_l1, post & gs[:, None])
        disc1 = post & (tv_l1 > TV_DISC)
        m["gc_l1"] = _gc(k_own, ref_l1, disc1)  # vs the flag-blind answer
        m["gc_tv_l1"] = _gc(tv_m, tv_l1, disc1)
        m["n_disc_l1"] = int(disc1.sum())
        hi = np.nonzero(has)[0]
        m["kl_own_curve"] = _curve(k_own, sup)  # by prediction position t
        m["kl_own_rel_curve"] = _rel_curve(k_own[hi], s[hi])  # by offset t - s
        m["kl_other_rel_curve"] = _rel_curve(k_oth[hi], s[hi])
        m["kl_l1_rel_curve"] = _rel_curve(k_l1[hi], s[hi])
        if return_preds:
            n = min(N_SAVE, len(s))
            preds.update(
                main_x=ep["x"].numpy().astype(np.int8), main_s=s.astype(np.int8),
                main_is_prog=is_prog, main_a=a.astype(np.int8), main_b=ep["b"].numpy().astype(np.int8),
                main_sup=sup, main_post=post,
                main_kl_own=k_own.astype(np.float32), main_kl_other=k_oth.astype(np.float32),
                main_kl_l1=k_l1.astype(np.float32), main_tv_own_other=tv_oth.astype(np.float32),
                main_model=pm[:n].astype(np.float32), main_own=orc["own"][:n].astype(np.float32),
                main_other=orc["other"][:n].astype(np.float32), main_l1=orc["l1"][:n].astype(np.float32))
        # flag ablation: same intervened episodes, flags zeroed -> should collapse to Bayes-L1 ----
        ep = eval_sets["ablated"]
        orc = self._cached_oracles("ablated", ep)
        out = run_model(model, self._tokens(ep["x"], ep["flags"]), None, device)
        pa = softmax(out["nxt"][:, : T - 1])
        sa = ep["s"].numpy()
        post_a = t >= sa[:, None]
        ka_l1, ka_own = _kl(orc["l1"], pa), _kl(orc["own"], pa)
        m["abl_kl_l1"] = _mean(ka_l1, post_a)
        m["abl_kl_own"] = _mean(ka_own, post_a)
        m["abl_kl_l1_rel_curve"] = _rel_curve(ka_l1, sa)
        if return_preds:
            preds.update(abl_kl_l1=ka_l1.astype(np.float32), abl_kl_own=ka_own.astype(np.float32),
                         abl_model=pa[: min(N_SAVE, len(sa))].astype(np.float32))
            return m, preds
        return m


# =============================================================================== E5b


class E5bNoiseCF(Task):
    name = "e5b_recurrence"
    vocab_size = 3 * P + 1  # flag in {0, 1, 2} x value, plus [CF]
    headline = "kl_own_trainprior"  # vs the exact PFN minimizer for the training prior

    def __init__(self, cf_labels: str = "abducted", eps: float = 0.2, held_out_frac: float = 0.2,
                 held_out_seed: int = 12345, pi: float = 0.5, seq_len: int = 16):
        assert cf_labels in ("abducted", "fresh"), cf_labels
        self.cf_labels = cf_labels
        self.other = "fresh" if cf_labels == "abducted" else "abducted"
        self.eps = eps
        self.pi = pi
        self.T = seq_len
        self.q_max = seq_len - 1 - N_CF  # q + N_CF <= T - 1
        self.held_out_frac = held_out_frac
        self.held_out_seed = held_out_seed
        n_held = int(round(held_out_frac * K))
        rng = np.random.default_rng(held_out_seed)
        self.held = np.sort(rng.choice(K, n_held, replace=False)).astype(np.int64)
        self.seen = np.setdiff1d(np.arange(K), self.held).astype(np.int64)
        self.seen_mask = np.isin(np.arange(K), self.seen).astype(np.float64)
        self.max_len = seq_len + 2 + N_CF  # x_0..x_{T-1} [CF] [v] y_1..y_3
        self.cf_pos = seq_len + 1 + np.arange(N_CF)  # positions of [v], [y_1], [y_2]
        self.heads = {"nxt": HeadSpec(P, "soft_ce"), "cf": HeadSpec(P, "soft_ce")}
        self._dev_progs: dict = {}
        self._orc_cache: dict = {}

    def eval_signature(self) -> dict:
        return {"p": P, "T": self.T, "pi": self.pi, "eps": self.eps, "cf_labels": self.cf_labels,
                "held_out_frac": self.held_out_frac, "held_out_seed": self.held_out_seed, "gen": 1}

    # ------------------------------------------------------------------ generation
    def _progs(self, which: str, device: torch.device) -> torch.Tensor:
        key = (which, str(device))
        if key not in self._dev_progs:
            arr = self.seen if which == "seen" else self.held
            self._dev_progs[key] = torch.as_tensor(arr, device=device)
        return self._dev_progs[key]

    def _latents(self, B: int, device: torch.device, which: str = "seen", force_prog: bool = False):
        """All latents drawn for every episode (both labelings consume the RNG identically)."""
        T = self.T
        progs = self._progs(which, device)
        is_prog = torch.rand(B, device=device) < self.pi
        if force_prog:
            is_prog = torch.ones_like(is_prog)
        k = progs[torch.randint(0, len(progs), (B,), device=device)]
        lat = dict(
            is_prog=is_prog, k=k, a=k // P, b=k % P,
            x0=torch.randint(0, P, (B,), device=device),
            eps=(torch.rand(B, T - 1, device=device) < self.eps).long(),  # factual eps_0..eps_{T-2}
            noise=torch.randint(0, P, (B, T), device=device),
            q=torch.randint(0, self.q_max + 1, (B,), device=device),
            dv=torch.randint(1, P, (B,), device=device),  # v = x_q + dv: uniform on Z_17 \ {x_q}
            eps_cf=(torch.rand(B, N_CF, device=device) < self.eps).long(),  # fresh eps'
            noise_cf=torch.randint(0, P, (B, N_CF), device=device),  # fresh H_R draws
        )
        return lat

    def _factual(self, lat: dict) -> torch.Tensor:
        a, b = lat["a"], lat["b"]
        cols = [lat["x0"]]
        for t in range(self.T - 1):
            cols.append((a * cols[-1] + b + lat["eps"][:, t]) % P)
        return torch.where(lat["is_prog"][:, None], torch.stack(cols, 1), lat["noise"])

    def _counterfactual(self, lat: dict, x: torch.Tensor, labels: str):
        a, b, q = lat["a"], lat["b"], lat["q"]
        v = (x.gather(1, q[:, None])[:, 0] + lat["dv"]) % P
        par, ys = v, []
        for k in range(N_CF):
            u = (q + k)[:, None]
            if labels == "abducted":
                yp = (a * par + b + lat["eps"].gather(1, u)[:, 0]) % P
                yr = x.gather(1, u + 1)[:, 0]  # noise held fixed: copy x_{q+k}
            else:
                yp = (a * par + b + lat["eps_cf"][:, k]) % P
                yr = lat["noise_cf"][:, k]
            yk = torch.where(lat["is_prog"], yp, yr)
            ys.append(yk)
            par = yk
        return v, torch.stack(ys, 1)

    def _tokens(self, x, q, v, y) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        xt = x + 2 * P * (pos[None, :] == q[:, None]).long()  # query marked in place (flag 2)
        cf = torch.full((B, 1), CF_TOKEN, dtype=x.dtype, device=x.device)
        return torch.cat([xt, cf, (P + v)[:, None], y], 1)

    def _targets(self, lat: dict, x, v, y, labels: str):
        B, T = x.shape
        L, dev, eps = self.max_len, x.device, self.eps
        a, b = lat["a"][:, None], lat["b"][:, None]
        isp = lat["is_prog"][:, None, None]
        uni = torch.full((B, L, P), 1.0 / P, device=dev)
        tf = torch.where(isp, _two_point((a * x[:, :-1] + b) % P, eps), uni[:, : T - 1])
        t_nxt = torch.cat([tf, uni[:, T - 1 :]], 1)
        if labels == "abducted":
            tc = F.one_hot(y, P).float()  # deterministic under both hypotheses
        else:
            par = torch.cat([v[:, None], y[:, :-1]], 1)
            tc = torch.where(isp, _two_point((a * par + b) % P, eps), uni[:, :N_CF])
        t_cf = torch.cat([uni[:, : T + 1], tc, uni[:, T + 1 + N_CF :]], 1)
        pos = torch.arange(L, device=dev)
        m_nxt = (pos < T - 1).expand(B, -1)
        m_cf = ((pos >= T + 1) & (pos < T + 1 + N_CF)).expand(B, -1)
        return {"nxt": t_nxt, "cf": t_cf}, {"nxt": m_nxt, "cf": m_cf}

    def sample_batch(self, batch_size: int, device: torch.device, return_latents: bool = False):
        lat = self._latents(batch_size, device, "seen")
        x = self._factual(lat)
        v, y = self._counterfactual(lat, x, self.cf_labels)
        targets, masks = self._targets(lat, x, v, y, self.cf_labels)
        batch = dict(tokens=self._tokens(x, lat["q"], v, y), targets=targets, masks=masks)
        return (batch, dict(lat, x=x, v=v, y=y)) if return_latents else batch

    # ------------------------------------------------------------------ eval sets
    def _episode_set(self, lat: dict) -> dict:
        x = self._factual(lat)
        v, y = self._counterfactual(lat, x, self.cf_labels)
        return dict(x=x, q=lat["q"], v=v, y=y, is_prog=lat["is_prog"], a=lat["a"], b=lat["b"],
                    k=lat["k"], eps=lat["eps"])

    def make_eval_sets(self, seed: int, n_episodes: int) -> dict:
        torch.manual_seed(seed)
        cpu = torch.device("cpu")
        sets = {"main": self._episode_set(self._latents(n_episodes, cpu, "seen"))}
        if len(self.held):  # program episodes with held-out programs only
            sets["heldout"] = self._episode_set(
                self._latents(max(256, n_episodes // 4), cpu, "held", force_prog=True))
        return sets

    # ------------------------------------------------------------------ oracles
    def oracles(self, ep: dict, train_prior: bool = False) -> dict:
        """fact [B, T-1, P]; cf_abducted / cf_fresh / cf_l2cf [B, N_CF, P]. With train_prior, the
        program prior is uniform over the training (seen) programs instead of all 289."""
        x, q, v, y = (ep[k].numpy() for k in ("x", "q", "v", "y"))
        kw = dict(eps=self.eps, pi=self.pi, prior_prog=self.seen_mask if train_prior else None)
        o = {"fact": noisy_factual_oracle(x, **kw)["pred"]}
        for mode in ("abducted", "fresh", "l2cf"):
            o[f"cf_{mode}"] = cf_oracle(x, q, v, y, mode, **kw)
        return o

    def _cached_oracles(self, name: str, ep: dict) -> dict:
        """Oracles under the full 289-program prior (the reference throughout), plus the
        training-prior (seen programs only) own oracles: the exact PFN minimizer on `main`, and on
        `heldout` the answer of a learner that is Bayes-optimal for the training distribution."""
        key = (name, _fingerprint(ep))
        if key not in self._orc_cache:
            o = self.oracles(ep)
            c = {"fact": o["fact"], "own": o[f"cf_{self.cf_labels}"],
                 "other": o[f"cf_{self.other}"], "l2cf": o["cf_l2cf"]}
            kw = dict(eps=self.eps, pi=self.pi, prior_prog=self.seen_mask)
            x, q, v, y = (ep[k].numpy() for k in ("x", "q", "v", "y"))
            c["fact_tp"] = noisy_factual_oracle(x, **kw)["pred"]
            c["own_tp"] = cf_oracle(x, q, v, y, self.cf_labels, **kw)
            self._orc_cache[key] = c
        return self._orc_cache[key]

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, model, eval_sets: dict, device: torch.device, return_preds: bool = False):
        m: dict = {}
        preds: dict = {}
        T = self.T
        cf = slice(T + 1, T + 1 + N_CF)
        for name in ("main", "heldout"):
            if name not in eval_sets:
                continue
            ep = eval_sets[name]
            orc = self._cached_oracles(name, ep)
            out = run_model(model, self._tokens(ep["x"], ep["q"], ep["v"], ep["y"]), None, device)
            pf = softmax(out["nxt"][:, : T - 1])  # factual predictions, positions 0..T-2
            pc = softmax(out["cf"][:, cf])  # y_1..y_3 at the positions of [v], [y_1], [y_2]
            kf = _kl(orc["fact"], pf)
            ky_own, ky_oth, ky_l2 = _kl(orc["own"], pc), _kl(orc["other"], pc), _kl(orc["l2cf"], pc)
            ref = _kl(orc["own"], orc["other"])
            tv_ref, tv_m = _tv(orc["own"], orc["other"]), _tv(orc["own"], pc)
            disc = tv_ref > TV_DISC
            is_prog, a = ep["is_prog"].numpy(), ep["a"].numpy()
            pooled = lambda f, y: float((f.sum() + y.sum()) / (f.size + y.size))  # noqa: E731
            # oracle-only reference: full-prior oracle vs training-prior oracle
            rf_tp, ry_tp = _kl(orc["fact"], orc["fact_tp"]), _kl(orc["own"], orc["own_tp"])
            if name == "main":
                m["kl_own"] = pooled(kf, ky_own)
                m["kl_fact"] = float(kf.mean())
                m["kl_y_own"] = float(ky_own.mean())
                m["kl_y_other"] = float(ky_oth.mean())
                m["kl_y_l2cf"] = float(ky_l2.mean())
                m["ref_y_own_other"] = float(ref.mean())
                m["gc_all"] = _gc(ky_own, ref, disc)
                m["gc_tv_all"] = _gc(tv_m, tv_ref, disc)
                # vs the exact PFN minimizer (training prior = seen programs only)
                kf_tp, ky_tp = _kl(orc["fact_tp"], pf), _kl(orc["own_tp"], pc)
                m["kl_own_trainprior"] = pooled(kf_tp, ky_tp)
                m["kl_fact_trainprior"] = float(kf_tp.mean())
                m["kl_y_own_trainprior"] = float(ky_tp.mean())
                m["ref_trainprior_floor"] = pooled(rf_tp, ry_tp)  # kl_own of a perfect PFN learner
                m["kl_fact_curve"] = kf.mean(0).tolist()
                m["kl_y_own_k"] = ky_own.mean(0).tolist()
                m["kl_y_other_k"] = ky_oth.mean(0).tolist()
                groups = {"noise": ~is_prog, "prog": is_prog & (a != 0), "prog_a0": is_prog & (a == 0)}
            else:
                m["kl_fact_curve_heldout"] = kf.mean(0).tolist()
                m["kl_y_own_k_heldout"] = ky_own.mean(0).tolist()
                # what a learner that is Bayes-optimal for the TRAINING prior would score here
                m["ref_trainprior_fact_heldout"] = _mean(rf_tp, (a != 0)[:, None])
                m["ref_trainprior_y_heldout"] = _mean(ry_tp, (a != 0)[:, None])
                groups = {"heldout": a != 0, "heldout_a0": a == 0}
            for g, gs in groups.items():
                m[f"n_{g}"] = int(gs.sum())
                m[f"kl_fact_{g}"] = _mean(kf, gs[:, None])
                m[f"kl_y_own_{g}"] = _mean(ky_own, gs[:, None])
                m[f"kl_y_other_{g}"] = _mean(ky_oth, gs[:, None])
                m[f"kl_y_l2cf_{g}"] = _mean(ky_l2, gs[:, None])
                sel = disc & gs[:, None]
                m[f"gc_{g}"] = _gc(ky_own, ref, sel)
                m[f"gc_tv_{g}"] = _gc(tv_m, tv_ref, sel)
                m[f"n_disc_{g}"] = int(sel.sum())
            if return_preds:
                n = min(N_SAVE, len(a))
                preds.update({
                    f"{name}_x": ep["x"].numpy().astype(np.int8), f"{name}_q": ep["q"].numpy().astype(np.int8),
                    f"{name}_v": ep["v"].numpy().astype(np.int8), f"{name}_y": ep["y"].numpy().astype(np.int8),
                    f"{name}_is_prog": is_prog, f"{name}_k": ep["k"].numpy().astype(np.int16),
                    f"{name}_kl_fact": kf.astype(np.float32), f"{name}_kl_y_own": ky_own.astype(np.float32),
                    f"{name}_kl_y_other": ky_oth.astype(np.float32), f"{name}_kl_y_l2cf": ky_l2.astype(np.float32),
                    f"{name}_tv_own_other": tv_ref.astype(np.float32),
                    f"{name}_cf_model": pc[:n].astype(np.float32), f"{name}_cf_own": orc["own"][:n].astype(np.float32),
                    f"{name}_cf_other": orc["other"][:n].astype(np.float32),
                    f"{name}_cf_l2cf": orc["l2cf"][:n].astype(np.float32),
                    f"{name}_fact_model": pf[:n].astype(np.float32), f"{name}_fact_oracle": orc["fact"][:n].astype(np.float32),
                })
        if "heldout" in eval_sets and "kl_fact_prog" in m:
            m["kl_fact_heldout_minus_seen"] = m["kl_fact_heldout"] - m["kl_fact_prog"]  # matched L1
            m["kl_y_heldout_minus_seen"] = m["kl_y_own_heldout"] - m["kl_y_own_prog"]
        if return_preds:
            return m, preds
        return m
