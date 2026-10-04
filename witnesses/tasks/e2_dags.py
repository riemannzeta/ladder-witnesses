"""E2 -- Pearl's "level one and three-fourths": untried and combined interventions on unseen graphs.

Four binary nodes, DAGs with in-degree <= 2 (25 topologies / 443 labeled DAGs), node labels
permuted per episode, BDeu(ESS=2) CPT prior. The model is GIVEN the graph (edge tokens) and must
answer P(T=1 | do(S=s)) -- or, as the matched L1-with-graph control, P(T=1 | V_S=s) -- for a
target T. Training queries have exactly one *effective* element plus a second slot that is NULL or
*inert* (a do() on a node that cannot affect T once the first do() is applied; a conditioning node
d-separated from T given the first). Tests probe: held-out topologies (5-fold CV over topologies),
COMBINED interventions where both do()'s change the answer (never seen in training), and a
same-data graph swap between Markov-equivalent graphs (an exact L1 witness pair).

mode = "cpt"     (E2-0): the CPT rows are given as tokens (node, parent-config) + value; targets are
                 exact. Isolates graph -> truncated-factorization from statistical estimation.
mode = "samples" (E2-full): N iid observational samples instead of CPTs; the readout after k
                 samples is compared with the exact Bayes posterior mean (truncated factorization at
                 posterior-mean CPTs -- exact for do; Monte Carlo for conditionals) and with the
                 graph-blind Bayesian model average over all 443 DAGs (the GC denominator).
"""

from __future__ import annotations

import numpy as np
import torch

from witnesses.tasks import dag_utils as du
from witnesses.tasks.base import HeadSpec, Task, fit_slope, kl_bernoulli_bits, run_model, sigmoid
from witnesses.tasks.common import beta as beta_draw

# ---------------------------------------------------------------- vocabulary
BOS, EDGE0, NOEDGE, QDO, QCOND = 0, 1, 17, 18, 19
DO0, COND0, NULL, TGT0, CELL0, ROW0, ROWPAD = 20, 28, 36, 37, 41, 57, 73
VOCAB = 74
N_EDGE_SLOTS, N_ROW_SLOTS = 5, 11
PREFIX = 1 + N_EDGE_SLOTS + 1 + 2 + 1  # 10
EVAL_K = (0, 8, 32, 128)
DISC = 0.05  # |L2 - L1| threshold for "discriminative" episodes
SPLITS = ("in_dist", "heldout", "composite_train", "composite_heldout", "swap")

# ---------------------------------------------------------------- static tables (internal reps)
_TOPOS = du.topologies()
REP = np.array([t["rep"] for t in _TOPOS], dtype=np.int64)  # [25, 4] parent masks
N_PA = np.array([[bin(p).count("1") for p in t["rep"]] for t in _TOPOS])  # [25, 4]
# rank of parent p among the parents of a node with mask m (sorted ascending), -1 if not a parent
RANK = np.full((16, 4), -1, dtype=np.int64)
for _m in range(16):
    for _r, _p in enumerate(du.parents_of(_m)):
        RANK[_m, _p] = _r
# ANC_CUT[t, T, A] = ancestor mask of T after cutting edges into A ; ANC[t, T] = ancestors of T
ANC_CUT = np.zeros((25, 4, 4), dtype=np.int64)
ANC = np.zeros((25, 4), dtype=np.int64)
DSEP = np.zeros((25, 4, 4, 4), dtype=bool)  # DSEP[t, T, A, B]: T _||_ B | A
for _t, _topo in enumerate(_TOPOS):
    _pa = _topo["rep"]
    for _T in range(4):
        ANC[_t, _T] = du.ancestors(_pa, _T)
        for _A in range(4):
            ANC_CUT[_t, _T, _A] = du.ancestors(_pa, _T, cut=1 << _A)
            for _B in range(4):
                if len({_T, _A, _B}) == 3:
                    DSEP[_t, _T, _A, _B] = du.d_separated(_pa, _T, _B, 1 << _A)


def _swap_bits(j: np.ndarray) -> np.ndarray:
    return ((j & 1) << 1) | ((j >> 1) & 1)


# inert-candidate bitmasks for single queries: INERT_DO[t, T, A], INERT_COND[t, T, A]
INERT_DO = np.zeros((25, 4, 4), dtype=np.int64)
INERT_COND = np.zeros((25, 4, 4), dtype=np.int64)
for _t in range(25):
    for _T in range(4):
        for _A in range(4):
            if _A == _T:
                continue
            for _B in range(4):
                if _B in (_T, _A):
                    continue
                if not (ANC_CUT[_t, _T, _A] >> _B & 1):
                    INERT_DO[_t, _T, _A] |= 1 << _B
                if DSEP[_t, _T, _A, _B]:
                    INERT_COND[_t, _T, _A] |= 1 << _B

_TORCH_TABLES: dict = {}


def _tt(device) -> dict:
    key = str(device)
    if key not in _TORCH_TABLES:
        _TORCH_TABLES[key] = {
            "REP": torch.as_tensor(REP, device=device), "N_PA": torch.as_tensor(N_PA, device=device),
            "RANK": torch.as_tensor(RANK, device=device),
            "ASSIGN": torch.as_tensor(du.ASSIGN.copy(), device=device),
            "INERT_DO": torch.as_tensor(INERT_DO, device=device),
            "INERT_COND": torch.as_tensor(INERT_COND, device=device),
        }
    return _TORCH_TABLES[key]


def joint_torch(pa: torch.Tensor, theta: torch.Tensor, do_mask: torch.Tensor | None = None,
                do_vals: torch.Tensor | None = None) -> torch.Tensor:
    """Vectorized (truncated) joint over the 16 internal assignments: [B, 16]."""
    T = _tt(pa.device)
    rank = T["RANK"][pa]  # [B, 4 node, 4 parent]
    w = torch.where(rank >= 0, 2 ** rank.clamp_min(0), torch.zeros_like(rank))
    A = T["ASSIGN"]  # [16, 4]
    j = (A[None, None, :, :] * w[:, :, None, :]).sum(-1)  # [B, 4, 16]
    p1 = theta.gather(2, j)
    v = A.T[None]  # [1, 4, 16]
    f = torch.where(v == 1, p1, 1 - p1)
    if do_mask is not None:
        isdo = ((do_mask[:, None] >> torch.arange(4, device=pa.device)) & 1).bool()
        forced = (v == do_vals[:, :, None]).to(f.dtype)
        f = torch.where(isdo[:, :, None], forced, f)
    return f.prod(1)


def answers_torch(pa, theta, q) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact P(T=1|do(S=s)) and P(T=1|V_S=s) per episode (vectorized)."""
    dev = pa.device
    A_ = _tt(dev)["ASSIGN"]
    Bsz = pa.shape[0]
    hasB = q["B"] >= 0
    Bc = q["B"].clamp_min(0)
    do_mask = (1 << q["A"]) | torch.where(hasB, 1 << Bc, torch.zeros_like(Bc))
    dv = torch.zeros(Bsz, 4, dtype=torch.long, device=dev)
    dv.scatter_(1, q["A"][:, None], q["va"][:, None])
    dv = torch.where(hasB[:, None] & (torch.arange(4, device=dev)[None] == Bc[:, None]),
                     q["vb"][:, None].expand(-1, 4), dv)
    tsel = A_.T[q["T"]] == 1  # [B, 16]
    jd = joint_torch(pa, theta, do_mask, dv)
    do_ans = (jd * tsel).sum(1)
    jf = joint_torch(pa, theta)
    sel = A_.T[q["A"]] == q["va"][:, None]
    sel = sel & (~hasB[:, None] | (A_.T[Bc] == q["vb"][:, None]))
    cond_ans = (jf * sel * tsel).sum(1) / (jf * sel).sum(1).clamp_min(1e-30)
    return do_ans, cond_ans


class E2Dags(Task):
    name = "e2_dags"
    vocab_size = VOCAB
    headline = "kl_in_dist"

    def __init__(self, mode: str = "samples", fold: int = 0, n_samples: int = 128,
                 p_cond: float = 0.5, p_inert: float = 0.5, mc: int = 2048,
                 n_eval_per_split: int | None = None):
        assert mode in ("cpt", "samples")
        self.mode = mode
        self.fold = fold
        self.N = n_samples
        self.p_cond = p_cond
        self.p_inert = p_inert
        self.mc = mc
        self.n_eval_per_split = n_eval_per_split
        folds = du.load_folds()
        self.heldout_topos = sorted(folds[fold])
        self.train_topos = sorted(set(range(25)) - set(self.heldout_topos))
        if mode == "samples":
            self.max_len = PREFIX + n_samples
            self.cont_dim = 0
            self.heads = {"ans": HeadSpec(1, "soft_bce"), "nxt": HeadSpec(16, "soft_ce", 0.25)}
        else:
            self.max_len = PREFIX + N_ROW_SLOTS
            self.cont_dim = 1
            self.heads = {"ans": HeadSpec(1, "soft_bce")}

    def eval_signature(self) -> dict:
        return {"mode": self.mode, "fold": self.fold, "N": self.N, "p_cond": self.p_cond,
                "p_inert": self.p_inert, "mc": self.mc, "n_eval": self.n_eval_per_split}

    # ================================================================ episode generation (torch)
    def _structure(self, topo_ids: torch.Tensor, device) -> dict:
        B = topo_ids.shape[0]
        pa = torch.as_tensor(REP, device=device)[topo_ids]  # [B, 4]
        npa = torch.as_tensor(N_PA, device=device)[topo_ids]
        perm = torch.argsort(torch.rand(B, 4, device=device), 1)  # presented label of node i
        a = 2.0 ** (-npa.float())  # BDeu row parameter per node
        theta = beta_draw(a[:, :, None].expand(B, 4, 4), a[:, :, None].expand(B, 4, 4), device)
        rows_used = (torch.arange(4, device=device)[None, None, :] < (2 ** npa)[:, :, None])
        theta = torch.where(rows_used, theta, torch.full_like(theta, 0.5))
        return dict(topo=topo_ids, pa=pa, perm=perm, theta=theta)

    @staticmethod
    def _row_of(pa_i: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        """pa_i [B] parent masks of one node; vals [B, n, 4] values -> row index [B, n]."""
        rank = torch.as_tensor(RANK, device=vals.device)[pa_i]  # [B, 4]
        w = torch.where(rank >= 0, 2 ** rank.clamp_min(0), torch.zeros_like(rank))
        return (vals * w[:, None, :]).sum(-1)

    def _sample_data(self, st: dict, n: int) -> torch.Tensor:
        B = st["pa"].shape[0]
        dev = st["pa"].device
        vals = torch.zeros(B, n, 4, dtype=torch.long, device=dev)
        for i in range(4):
            j = self._row_of(st["pa"][:, i], vals)
            p = st["theta"][:, i, :].gather(1, j)
            vals[:, :, i] = (torch.rand(B, n, device=dev) < p).long()
        return vals

    def _query(self, st: dict, kind: str) -> dict:
        """kind: 'single' (training-type: one effective + NULL/inert) | 'composite'."""
        B = st["pa"].shape[0]
        dev = st["pa"].device
        t = st["topo"].cpu().numpy()
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        is_cond = rng.random(B) < self.p_cond
        T = rng.integers(0, 4, B)
        A = np.zeros(B, np.int64)
        Bn = np.full(B, -1, np.int64)
        for b in range(B):
            others = [x for x in range(4) if x != T[b]]
            if kind == "single":
                A[b] = rng.choice(others)
                if rng.random() < self.p_inert:
                    cand = []
                    for x in others:
                        if x == A[b]:
                            continue
                        if is_cond[b]:
                            if DSEP[t[b], T[b], A[b], x]:
                                cand.append(x)
                        elif not (ANC_CUT[t[b], T[b], A[b]] >> x & 1):
                            cand.append(x)
                    if cand:
                        Bn[b] = rng.choice(cand)
            else:  # composite: both elements must be relevant to T given the other
                pairs = []
                for x in others:
                    for y in others:
                        if x >= y:
                            continue
                        if is_cond[b]:
                            ok = not DSEP[t[b], T[b], x, y] and not DSEP[t[b], T[b], y, x]
                        else:
                            ok = (ANC_CUT[t[b], T[b], x] >> y & 1) and (ANC_CUT[t[b], T[b], y] >> x & 1)
                        if ok:
                            pairs.append((x, y))
                if pairs:
                    A[b], Bn[b] = pairs[rng.integers(len(pairs))]
                else:
                    Bn[b] = -2  # no valid composite for this graph/target -> discard later
        va = rng.integers(0, 2, B)
        vb = rng.integers(0, 2, B)
        return {k: torch.as_tensor(v, device=dev) for k, v in
                dict(is_cond=is_cond, T=T, A=A, B=Bn, va=va, vb=vb).items()}

    # ---------------------------------------------------------------- exact answers from theta
    @staticmethod
    def _answers_np(pa_rows: np.ndarray, theta: np.ndarray, q: dict) -> dict:
        """Exact P(T=1 | do(S=s)) and P(T=1 | V_S=s) for each episode (numpy, internal labels)."""
        B = theta.shape[0]
        do_ans = np.zeros(B)
        cond_ans = np.zeros(B)
        T, A, Bn, va, vb = (q[k] for k in ("T", "A", "B", "va", "vb"))
        for b in range(B):
            pa = tuple(int(x) for x in pa_rows[b])
            mask = 1 << int(A[b])
            vals = np.zeros((1, 4), np.int64)
            vals[0, A[b]] = va[b]
            if Bn[b] >= 0:
                mask |= 1 << int(Bn[b])
                vals[0, Bn[b]] = vb[b]
            jd = du.joint_table(pa, theta[b : b + 1], mask, vals)[0]
            do_ans[b] = jd[du.ASSIGN[:, T[b]] == 1].sum()
            jf = du.joint_table(pa, theta[b : b + 1])[0]
            sel = du.ASSIGN[:, A[b]] == va[b]
            if Bn[b] >= 0:
                sel &= du.ASSIGN[:, Bn[b]] == vb[b]
            cond_ans[b] = jf[sel & (du.ASSIGN[:, T[b]] == 1)].sum() / jf[sel].sum()
        return dict(do=do_ans, cond=cond_ans)

    # ---------------------------------------------------------------- tokens
    def _prefix(self, st: dict, q: dict, pa_override: torch.Tensor | None = None) -> torch.Tensor:
        B = st["pa"].shape[0]
        dev = st["pa"].device
        pa = st["pa"] if pa_override is None else pa_override
        perm = st["perm"]
        # all 16 (parent, child) candidates, then push non-edges back with a random sort key
        par = torch.arange(4, device=dev).repeat_interleave(4)  # p for each of 16 slots
        chi = torch.arange(4, device=dev).repeat(4)  # i
        has = ((pa[:, chi] >> par[None, :]) & 1) == 1  # [B, 16]
        tok = EDGE0 + 4 * perm[:, par] + perm[:, chi]
        tok = torch.where(has, tok, torch.full_like(tok, NOEDGE))
        key = torch.rand(B, 16, device=dev) + (~has).float() * 2.0
        edges = tok.gather(1, torch.argsort(key, 1))[:, :N_EDGE_SLOTS]  # <= 5 edges, then pads
        edges = edges.gather(1, torch.argsort(torch.rand(B, N_EDGE_SLOTS, device=dev), 1))
        isc = q["is_cond"]
        qtok = torch.where(isc, QCOND, QDO)
        base = torch.where(isc, COND0, DO0)
        pA = perm.gather(1, q["A"][:, None])[:, 0]
        s1 = base + 2 * pA + q["va"]
        Bc = q["B"].clamp_min(0)
        pB = perm.gather(1, Bc[:, None])[:, 0]
        s2 = torch.where(q["B"] >= 0, base + 2 * pB + q["vb"], torch.full_like(s1, NULL))
        swap = torch.rand(B, device=dev) < 0.5  # slot order carries no information
        s1, s2 = torch.where(swap, s2, s1), torch.where(swap, s1, s2)
        tgt = TGT0 + perm.gather(1, q["T"][:, None])[:, 0]
        return torch.cat([torch.full((B, 1), BOS, device=dev), edges, qtok[:, None],
                          s1[:, None], s2[:, None], tgt[:, None]], 1)

    def _cells(self, st: dict, vals: torch.Tensor) -> torch.Tensor:
        w = 2 ** st["perm"]  # bit position of node i is its presented label
        return (vals * w[:, None, :]).sum(-1)

    def _row_tokens(self, st: dict, theta: torch.Tensor | None = None,
                    pa_override: torch.Tensor | None = None):
        """CPT rows as tokens (presented node label, presented parent-config index) + values.

        All 16 (node, row) candidates are built at once; unused rows are pushed behind the used
        ones by a random sort key (so used rows appear in random order), and the first
        N_ROW_SLOTS (= max used rows) are kept.
        """
        B = st["pa"].shape[0]
        dev = st["pa"].device
        pa = st["pa"] if pa_override is None else pa_override
        th = (st["theta"] if theta is None else theta).float()
        perm = st["perm"]
        T_ = _tt(dev)
        npa = torch.as_tensor([bin(m).count("1") for m in range(16)], device=dev)[pa]  # [B, 4]
        r = T_["RANK"][pa]  # [B, 4 node, 4 parent]
        # presented label of the internal-first parent vs the smallest presented parent label
        first_lab = torch.where(r == 0, perm[:, None, :], torch.full_like(r, -1)).max(-1).values
        min_lab = torch.where(r >= 0, perm[:, None, :], torch.full_like(r, 99)).min(-1).values
        swapped = (npa == 2) & (first_lab != min_lab)  # [B, 4]
        j = torch.arange(4, device=dev)[None, None, :].expand(B, 4, 4)  # internal row index
        jp = torch.where(swapped[:, :, None], ((j & 1) << 1) | ((j >> 1) & 1), j)
        use = j < (2 ** npa)[:, :, None]  # [B, 4, 4]
        tok = ROW0 + 4 * perm[:, :, None] + jp
        tok = torch.where(use, tok, torch.full_like(tok, ROWPAD)).reshape(B, 16)
        val = torch.where(use, th, torch.zeros_like(th)).reshape(B, 16)
        key = torch.rand(B, 16, device=dev) + (~use.reshape(B, 16)).float() * 2.0
        order = torch.argsort(key, 1)[:, :N_ROW_SLOTS]
        return tok.gather(1, order), val.gather(1, order)

    # ================================================================ batches
    def _draw(self, B: int, device, topos: list[int], kind: str = "single"):
        tl = torch.as_tensor(topos, device=device)
        tid = tl[torch.randint(0, len(topos), (B,), device=device)]
        st = self._structure(tid, device)
        q = self._query_single(st) if kind == "single" else self._query(st, kind)
        return st, q

    def _query_single(self, st: dict) -> dict:
        """Vectorized training-type query: one effective element + NULL or inert second slot."""
        B = st["pa"].shape[0]
        dev = st["pa"].device
        T_ = _tt(dev)
        is_cond = torch.rand(B, device=dev) < self.p_cond
        T = torch.randint(0, 4, (B,), device=dev)
        A = (T + torch.randint(1, 4, (B,), device=dev)) % 4
        cand = torch.where(is_cond, T_["INERT_COND"][st["topo"], T, A], T_["INERT_DO"][st["topo"], T, A])
        bits = ((cand[:, None] >> torch.arange(4, device=dev)) & 1).bool()
        score = torch.where(bits, torch.rand(B, 4, device=dev), torch.full((B, 4), -1.0, device=dev))
        pick = score.argmax(1)
        want = (torch.rand(B, device=dev) < self.p_inert) & (cand > 0)
        Bn = torch.where(want, pick, torch.full_like(pick, -1))
        return dict(is_cond=is_cond, T=T, A=A, B=Bn,
                    va=torch.randint(0, 2, (B,), device=dev), vb=torch.randint(0, 2, (B,), device=dev))

    def sample_batch(self, batch_size: int, device) -> dict:
        st, q = self._draw(batch_size, device, self.train_topos, "single")
        pre = self._prefix(st, q)
        do_a, cond_a = answers_torch(st["pa"], st["theta"], q)
        target = torch.where(q["is_cond"], cond_a, do_a).float()
        if self.mode == "cpt":
            toks, vals = self._row_tokens(st)
            tokens = torch.cat([pre, toks], 1)
            cont = torch.cat([torch.zeros(batch_size, PREFIX, 1, device=device), vals[:, :, None]], 1)
            m = torch.zeros(batch_size, self.max_len, dtype=torch.bool, device=device)
            m[:, -1] = True
            return dict(tokens=tokens, cont=cont, targets={"ans": target[:, None, None]},
                        masks={"ans": m})
        vals = self._sample_data(st, self.N)
        tokens = torch.cat([pre, CELL0 + self._cells(st, vals)], 1)
        jt = self._presented_joint(st)
        pos = torch.arange(self.max_len, device=device)
        return dict(tokens=tokens,
                    targets={"ans": target[:, None, None], "nxt": jt[:, None, :]},
                    masks={"ans": (pos >= PREFIX - 1).expand(batch_size, -1),
                           "nxt": ((pos >= PREFIX - 1) & (pos < self.max_len - 1)).expand(batch_size, -1)})

    def _presented_joint(self, st: dict) -> torch.Tensor:
        """True joint over presented-label cells [B, 16] (vectorized)."""
        A_ = _tt(st["pa"].device)["ASSIGN"]
        jt = joint_torch(st["pa"], st["theta"])
        cell_of = (A_[None, :, :] << st["perm"][:, None, :]).sum(-1)  # [B, 16]
        return torch.zeros_like(jt).scatter_(1, cell_of, jt)

    # ================================================================ eval sets
    def make_eval_sets(self, seed: int, n_episodes: int) -> dict:
        torch.manual_seed(seed)
        cpu = torch.device("cpu")
        n = self.n_eval_per_split or max(64, n_episodes // len(SPLITS))
        sets = {}
        spec = dict(in_dist=(self.train_topos, "single"), heldout=(self.heldout_topos, "single"),
                    composite_train=(self.train_topos, "composite"),
                    composite_heldout=(self.heldout_topos, "composite"))
        for name, (topos, kind) in spec.items():
            sets[name] = self._make_split(n, cpu, topos, kind)
        sets["swap"] = self._make_swap(n, cpu)
        return sets

    def _make_split(self, n: int, cpu, topos, kind) -> dict:
        keep: list[dict] = []
        got = 0
        while got < n:
            st, q = self._draw(4 * n, cpu, topos, kind)
            ok = (q["B"] != -2).numpy()
            qn = {k: v.numpy() for k, v in q.items()}
            th = st["theta"].numpy().astype(np.float64)
            pa = st["pa"].numpy()
            ans = self._answers_np(pa, th, qn)
            if kind == "composite":  # both elements must move the answer by > DISC
                for drop in ("A", "B"):
                    q1 = dict(qn)
                    if drop == "B":
                        q1["B"] = np.full_like(qn["B"], -1)
                    else:
                        q1["A"], q1["va"] = qn["B"].clip(0), qn["vb"]
                        q1["B"] = np.full_like(qn["B"], -1)
                    a1 = self._answers_np(pa, th, q1)
                    key = np.where(qn["is_cond"], 0, 1)
                    full = np.where(key == 1, ans["do"], ans["cond"])
                    part = np.where(key == 1, a1["do"], a1["cond"])
                    ok &= np.abs(full - part) > DISC
                # and conditioning must differ from intervening (an L1 strategy must fail)
                ok &= np.abs(ans["do"] - ans["cond"]) > DISC
            sel = np.nonzero(ok)[0]
            ep = {**{k: v[sel] for k, v in st.items()}, **{f"q_{k}": v[sel] for k, v in q.items()}}
            ep["ans_do"] = torch.as_tensor(ans["do"][sel])
            ep["ans_cond"] = torch.as_tensor(ans["cond"][sel])
            keep.append(ep)
            got += len(sel)
        ep = {k: torch.cat([d[k] for d in keep])[:n] for k in keep[0]}
        return self._finalize(ep, cpu)

    def _finalize(self, ep: dict, cpu, pa_override=None) -> dict:
        """Freeze tokens (prefix + rows or samples) and attach oracles."""
        st = {k: ep[k] for k in ("topo", "pa", "perm", "theta")}
        q = {k[2:]: ep[k] for k in ep if k.startswith("q_")}
        ep["prefix"] = self._prefix(st, q, pa_override)
        if self.mode == "cpt":
            if pa_override is not None:
                th_alt = ep["theta_alt"]
                ep["rows"], ep["row_vals"] = self._row_tokens(st, th_alt, pa_override)
            else:
                ep["rows"], ep["row_vals"] = self._row_tokens(st)
        else:
            if "vals" not in ep:
                ep["vals"] = self._sample_data(st, self.N)
            ep["cells"] = self._cells(st, ep["vals"])
            ep.update(self._oracles(ep, pa_override))
        return ep

    def _make_swap(self, n: int, cpu) -> dict:
        """Held-out-topology episodes paired with a Markov-equivalent alternative graph G'."""
        # candidate (topology, alternative internal parent masks) pairs
        alts = {}
        for t in self.heldout_topos:
            rep = tuple(int(x) for x in REP[t])
            mk = du.markov_key(rep)
            cands = [pa for pa in du.labeled_dags() if pa != rep and du.markov_key(pa) == mk]
            if cands:
                alts[t] = cands
        if not alts:
            return {}
        keep, got = [], 0
        rng = np.random.default_rng(7)
        while got < n:
            st, q = self._draw(4 * n, cpu, sorted(alts), "single")
            q["is_cond"][:] = False
            q["B"][:] = -1
            alt = np.array([alts[int(t)][rng.integers(len(alts[int(t)]))] for t in st["topo"].numpy()])
            qn = {k: v.numpy() for k, v in q.items()}
            th = st["theta"].numpy().astype(np.float64)
            a_true = self._answers_np(st["pa"].numpy(), th, qn)["do"]
            th_alt = self._cpts_from_joint(st["pa"].numpy(), th, alt)
            a_alt = self._answers_np(alt, th_alt, qn)["do"]
            sel = np.nonzero(np.abs(a_true - a_alt) > DISC)[0]
            ep = {**{k: v[sel] for k, v in st.items()}, **{f"q_{k}": v[sel] for k, v in q.items()}}
            ep.update(pa_alt=torch.as_tensor(alt[sel]), theta_alt=torch.as_tensor(th_alt[sel]),
                      ans_do=torch.as_tensor(a_true[sel]), ans_do_alt=torch.as_tensor(a_alt[sel]))
            keep.append(ep)
            got += len(sel)
        ep = {k: torch.cat([d[k] for d in keep])[:n] for k in keep[0]}
        if self.mode == "samples":
            st = {k: ep[k] for k in ("topo", "pa", "perm", "theta")}
            ep["vals"] = self._sample_data(st, self.N)
        orig = self._finalize(dict(ep), cpu)
        alt_ep = dict(ep)
        alt_ep = self._finalize(alt_ep, cpu, pa_override=ep["pa_alt"])
        return {"orig": orig, "alt": alt_ep}

    @staticmethod
    def _cpts_from_joint(pa: np.ndarray, theta: np.ndarray, pa_alt: np.ndarray) -> np.ndarray:
        """CPTs of graph pa_alt that reproduce the joint of (pa, theta) (internal labels)."""
        out = np.full_like(theta, 0.5)
        for b in range(theta.shape[0]):
            jt = du.joint_table(tuple(int(x) for x in pa[b]), theta[b : b + 1])[0]
            for i in range(4):
                j = du.row_index(int(pa_alt[b, i]), du.ASSIGN)
                v = du.ASSIGN[:, i]
                for r in range(2 ** bin(int(pa_alt[b, i])).count("1")):
                    den = jt[j == r].sum()
                    out[b, i, r] = jt[(j == r) & (v == 1)].sum() / den if den > 0 else 0.5
        return out

    # ================================================================ oracles (samples mode)
    def _oracles(self, ep: dict, pa_override=None) -> dict:
        pa_all = (ep["pa"] if pa_override is None else pa_override).numpy()
        vals = ep["vals"].numpy()
        q = {k[2:]: ep[k].numpy() for k in ep if k.startswith("q_")}
        B = vals.shape[0]
        rng = np.random.default_rng(11)
        o_l2 = np.zeros((B, len(EVAL_K)))
        o_l1 = np.zeros((B, len(EVAL_K)))
        o_blind = np.zeros((B, len(EVAL_K)))
        for ki, k in enumerate(EVAL_K):
            for b in range(B):
                pa = tuple(int(x) for x in pa_all[b])
                n1, n0 = self._row_counts(pa, vals[b, :k])
                a = np.array([du.bdeu_ab(bin(p).count("1")) for p in pa])[:, None]
                tbar = (a + n1) / (2 * a + n1 + n0)
                qb = {kk: v[b : b + 1] for kk, v in q.items()}
                o_l2[b, ki] = self._answers_np(np.array([pa]), tbar[None], qb)["do"][0]
                draws = rng.beta(a + n1, a + n0, size=(self.mc, 4, 4))
                o_l1[b, ki] = self._cond_mc(pa, draws, qb)
                o_blind[b, ki] = self._bma(vals[b, :k], qb, rng)
        return dict(oracle_l2=torch.as_tensor(o_l2), oracle_l1=torch.as_tensor(o_l1),
                    oracle_blind=torch.as_tensor(o_blind))

    @staticmethod
    def _row_counts(pa, vals: np.ndarray):
        n1 = np.zeros((4, 4))
        n0 = np.zeros((4, 4))
        for i in range(4):
            j = du.row_index(pa[i], vals) if len(vals) else np.zeros(0, np.int64)
            np.add.at(n1[i], j[vals[:, i] == 1] if len(vals) else j, 1)
            np.add.at(n0[i], j[vals[:, i] == 0] if len(vals) else j, 1)
        return n1, n0

    @staticmethod
    def _cond_mc(pa, draws: np.ndarray, q: dict) -> float:
        jt = du.joint_table(pa, draws)  # [M, 16]
        A, Bn, va, vb, T = (int(q[k][0]) for k in ("A", "B", "va", "vb", "T"))
        sel = du.ASSIGN[:, A] == va
        if Bn >= 0:
            sel &= du.ASSIGN[:, Bn] == vb
        num = jt[:, sel & (du.ASSIGN[:, T] == 1)].sum(1)
        return float((num / jt[:, sel].sum(1)).mean())

    # ---------------------------------------------------------------- graph-blind model averaging
    _BMA_CACHE: dict = {}

    @classmethod
    def _bma_tables(cls):
        if "t" not in cls._BMA_CACHE:
            dags = du.labeled_dags()
            prior = np.array([1.0 / (25 * _TOPOS[du.topology_id(pa)]["n_labelings"]) for pa in dags])
            rows = np.stack([np.stack([du.row_index(pa[i], du.ASSIGN) for i in range(4)]) for pa in dags])
            a = np.array([[du.bdeu_ab(bin(p).count("1")) for p in pa] for pa in dags])
            cls._BMA_CACHE["t"] = (dags, np.log(prior), rows, a)
        return cls._BMA_CACHE["t"]

    def _bma(self, vals: np.ndarray, q: dict, rng) -> float:
        """Graph-blind Bayes answer: average over all 443 DAGs (BDeu marginal likelihoods)."""
        from scipy.special import betaln
        dags, logprior, rows, a = self._bma_tables()
        cnt = np.bincount((vals * (1 << np.arange(4))).sum(1), minlength=16) if len(vals) else np.zeros(16)
        D = len(dags)
        n1 = np.zeros((D, 4, 4))
        n0 = np.zeros((D, 4, 4))
        for i in range(4):
            v = du.ASSIGN[:, i]
            for r in range(4):
                m = rows[:, i, :] == r  # [D, 16]
                n1[:, i, r] = (m * (v == 1) * cnt).sum(1)
                n0[:, i, r] = (m * (v == 0) * cnt).sum(1)
        aa = a[:, :, None]
        used = np.arange(4)[None, None, :] < (2 ** np.round(-np.log2(aa))).astype(int)
        ll = np.where(used, betaln(aa + n1, aa + n0) - betaln(aa, aa), 0.0).sum((1, 2))
        lw = logprior + ll
        w = np.exp(lw - lw.max())
        w /= w.sum()
        tbar = (aa + n1) / (2 * aa + n1 + n0)
        A, Bn, va, vb, T = (int(q[k][0]) for k in ("A", "B", "va", "vb", "T"))
        if not bool(q["is_cond"][0]):
            mask = (1 << A) | ((1 << Bn) if Bn >= 0 else 0)
            dv = np.zeros((1, 4), np.int64)
            dv[0, A] = va
            if Bn >= 0:
                dv[0, Bn] = vb
            ans = np.array([du.joint_table(dags[d], tbar[d : d + 1], mask, dv)[0][du.ASSIGN[:, T] == 1].sum()
                            if w[d] > 1e-12 else 0.0 for d in range(D)])
            return float((w * ans).sum())
        # conditional: Monte Carlo over the joint posterior of (DAG, theta)
        idx = rng.choice(D, size=self.mc, p=w)
        draws = rng.beta(aa[idx] + n1[idx], aa[idx] + n0[idx])
        out = np.empty(self.mc)
        sel = du.ASSIGN[:, A] == va
        if Bn >= 0:
            sel &= du.ASSIGN[:, Bn] == vb
        for d in np.unique(idx):
            m = idx == d
            jt = du.joint_table(dags[d], draws[m])
            out[m] = jt[:, sel & (du.ASSIGN[:, T] == 1)].sum(1) / jt[:, sel].sum(1)
        return float(out.mean())

    # ================================================================ evaluation
    def _model_answers(self, model, ep: dict, device) -> np.ndarray:
        if self.mode == "cpt":
            tokens = torch.cat([ep["prefix"], ep["rows"]], 1)
            cont = torch.cat([torch.zeros(tokens.shape[0], PREFIX, 1), ep["row_vals"][:, :, None].float()], 1)
            out = run_model(model, tokens, cont, device)
            return sigmoid(out["ans"][:, -1, 0])[:, None]  # [B, 1]
        tokens = torch.cat([ep["prefix"], CELL0 + ep["cells"]], 1)
        out = run_model(model, tokens, None, device)
        return sigmoid(out["ans"][:, [PREFIX - 1 + k for k in EVAL_K], 0])  # [B, len(EVAL_K)]

    def _split_metrics(self, model, ep: dict, device, name: str, res: dict, preds: dict | None):
        pm = self._model_answers(model, ep, device)
        isc = ep["q_is_cond"].numpy()
        if self.mode == "cpt":
            right = np.where(isc, ep["ans_cond"].numpy(), ep["ans_do"].numpy())[:, None]
            wrong = np.where(isc, ep["ans_do"].numpy(), ep["ans_cond"].numpy())[:, None]
            l2 = ep["ans_do"].numpy()[:, None]
            l1 = ep["ans_cond"].numpy()[:, None]
            kcols = [0]
        else:
            l2 = ep["oracle_l2"].numpy()
            l1 = ep["oracle_l1"].numpy()
            right = np.where(isc[:, None], l1, l2)
            wrong = ep["oracle_blind"].numpy()  # graph-blind Bayes (GC denominator)
            kcols = [i for i, k in enumerate(EVAL_K) if k >= 8]
        kl_m = kl_bernoulli_bits(right, pm)[:, kcols]
        kl_w = kl_bernoulli_bits(right, wrong)[:, kcols]
        for qn, sel in (("do", ~isc), ("cond", isc)):
            if sel.sum() == 0:
                continue
            res[f"kl_{name}_{qn}"] = float(kl_m[sel].mean())
            res[f"gc_{name}_{qn}"] = float(1 - kl_m[sel].mean() / max(kl_w[sel].mean(), 1e-12))
            disc = sel & (np.abs(l2[:, -1] - l1[:, -1]) > DISC)
            if disc.sum() >= 3:
                # slope of (model - L1) on (L2 - L1) for do; of (model - L2) on (L1 - L2) for cond
                if qn == "do":
                    s, r2 = fit_slope((l2 - l1)[disc, -1], (pm - l1)[disc, -1])
                else:
                    s, r2 = fit_slope((l1 - l2)[disc, -1], (pm - l2)[disc, -1])
                res[f"slope_{name}_{qn}"], res[f"slope_r2_{name}_{qn}"] = s, r2
                res[f"n_disc_{name}_{qn}"] = int(disc.sum())
        res[f"kl_{name}"] = float(kl_m.mean())
        if preds is not None:
            preds[f"{name}_model"] = pm.astype(np.float32)
            preds[f"{name}_l2"] = l2.astype(np.float32)
            preds[f"{name}_l1"] = l1.astype(np.float32)
            preds[f"{name}_is_cond"] = isc
            preds[f"{name}_topo"] = ep["topo"].numpy()

    def evaluate(self, model, eval_sets: dict, device, return_preds: bool = False):
        res: dict = {}
        preds: dict | None = {} if return_preds else None
        for name in ("in_dist", "heldout", "composite_train", "composite_heldout"):
            self._split_metrics(model, eval_sets[name], device, name, res, preds)
        sw = eval_sets.get("swap")
        if sw:
            po = self._model_answers(model, sw["orig"], device)[:, -1]
            pa_ = self._model_answers(model, sw["alt"], device)[:, -1]
            if self.mode == "cpt":
                oo, oa = sw["orig"]["ans_do"].numpy(), sw["alt"]["ans_do_alt"].numpy()
            else:
                oo, oa = sw["orig"]["oracle_l2"].numpy()[:, -1], sw["alt"]["oracle_l2"].numpy()[:, -1]
            res["swap_slope"], res["swap_r2"] = fit_slope(oo - oa, po - pa_)
            res["swap_n"] = int(len(oo))
            if preds is not None:
                preds.update(swap_model_orig=po, swap_model_alt=pa_, swap_oracle_orig=oo, swap_oracle_alt=oa)
        if return_preds:
            return res, preds
        return res
