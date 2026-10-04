"""E2c -- the closest test of Pearl's "level one and three-fourths" claim.

Pearl (2023): reinforcement learners "train on interventions ... However, they cannot infer from
this anything about a third move they haven't tried. They also cannot combine interventions to
infer what will happen if they do both A and B." Here the model gets NO graph. Its context holds
observational samples and samples from two *tried* randomized experiments, do(A) and do(B), each
tagged with a Dawid-style regime token. Queries are the untried combination do(A=a, B=b) -> T and
the untried single do(C=c) -> T (plus tried do(A) / do(B) as a check).

References:
  * Bayes: model averaging over all 443 DAGs (4 nodes, in-degree <= 2) with BDeu(ESS 2) and
    Cooper-Yoo interventional likelihoods (an intervened node's own CPT gets no evidence from the
    samples in which it was set). This is the ideal learner with exactly the same information.
  * "1 3/4" (model-free): tried regimes answered from their own frequencies; the untried
    combination from the two nearest tried regimes, P(T | do(A=a), B=b) and P(T | do(B=b), A=a),
    averaged; an untried single do(C) from the observational conditional P(T | C=c).
Readout: gap closed between the 1 3/4 baseline and the Bayes answer, overall and split by how sharp
the graph posterior makes the answer.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import betaln

from witnesses.tasks import dag_utils as du
from witnesses.tasks.base import HeadSpec, Task, kl_bernoulli_bits, run_model, sigmoid
from witnesses.tasks.common import beta as beta_draw
from witnesses.tasks.e2_dags import N_PA, REP, _tt, answers_torch

BOS, DO0, NULL, TGT0, SAMP0 = 0, 1, 9, 10, 14  # SAMP0 + 16*regime + cell; regime 0 = obs, 1+L = do(L)
VOCAB = SAMP0 + 5 * 16
PREFIX = 4
EVAL_K = (0, 24, 48, 96, 192)
QKINDS = ("combo", "untried", "tried")
JEFF = 0.5  # Beta(1/2, 1/2) smoothing for the model-free baseline


class E2cRegimes(Task):
    name = "e2c_regimes"
    vocab_size = VOCAB
    headline = "kl_all"

    def __init__(self, n_per_regime: int = 64, p_kinds=(0.4, 0.4, 0.2), n_eval_per_kind: int | None = None):
        self.n_per = n_per_regime
        self.N = 3 * n_per_regime
        self.p_kinds = p_kinds
        self.n_eval_per_kind = n_eval_per_kind
        self.max_len = PREFIX + self.N
        self.heads = {"ans": HeadSpec(1, "soft_bce")}

    def eval_signature(self) -> dict:
        return {"n_per": self.n_per, "p_kinds": list(self.p_kinds), "n_eval": self.n_eval_per_kind}

    # ---------------------------------------------------------------- generation
    def _episodes(self, B: int, device, kind: torch.Tensor | None = None) -> dict:
        tl = torch.arange(25, device=device)
        topo = tl[torch.randint(0, 25, (B,), device=device)]
        pa = torch.as_tensor(REP, device=device)[topo]
        npa = torch.as_tensor(N_PA, device=device)[topo]
        perm = torch.argsort(torch.rand(B, 4, device=device), 1)
        a = 2.0 ** (-npa.float())
        theta = beta_draw(a[:, :, None].expand(B, 4, 4), a[:, :, None].expand(B, 4, 4), device)
        used = torch.arange(4, device=device)[None, None, :] < (2 ** npa)[:, :, None]
        theta = torch.where(used, theta, torch.full_like(theta, 0.5))
        roles = torch.argsort(torch.rand(B, 4, device=device), 1)  # internal nodes (A, B, C, T)
        A, Bn, C, T = roles.unbind(1)
        # context: n_per samples in each regime (obs, do(A), do(B)), shuffled
        reg = torch.cat([torch.full((B, self.n_per), r, device=device) for r in range(3)], 1)
        reg = reg.gather(1, torch.argsort(torch.rand(B, self.N, device=device), 1))
        tgt_node = torch.where(reg == 1, A[:, None], torch.where(reg == 2, Bn[:, None], torch.full_like(reg, -1)))
        vals = torch.zeros(B, self.N, 4, dtype=torch.long, device=device)
        rank = _tt(device)["RANK"]
        for i in range(4):
            r = rank[pa[:, i]]
            w = torch.where(r >= 0, 2 ** r.clamp_min(0), torch.zeros_like(r))
            j = (vals * w[:, None, :]).sum(-1)
            p = theta[:, i, :].gather(1, j)
            nat = (torch.rand(B, self.N, device=device) < p).long()
            forced = torch.randint(0, 2, (B, self.N), device=device)
            vals[:, :, i] = torch.where(tgt_node == i, forced, nat)
        if kind is None:
            kind = torch.multinomial(torch.tensor(self.p_kinds, device=device).expand(B, -1), 1)[:, 0]
        # queries: combo do(A,B); untried do(C); tried do(A) or do(B)
        tried_pick = torch.where(torch.rand(B, device=device) < 0.5, A, Bn)
        qA = torch.where(kind == 0, A, torch.where(kind == 1, C, tried_pick))
        qB = torch.where(kind == 0, Bn, torch.full_like(A, -1))
        q = dict(is_cond=torch.zeros(B, dtype=torch.bool, device=device), T=T, A=qA, B=qB,
                 va=torch.randint(0, 2, (B,), device=device), vb=torch.randint(0, 2, (B,), device=device))
        do_a, _ = answers_torch(pa, theta, q)
        return dict(topo=topo, pa=pa, perm=perm, theta=theta, roles=roles, reg=reg, vals=vals,
                    kind=kind, ans=do_a, **{f"q_{k}": v for k, v in q.items()})

    def _tokens(self, ep: dict) -> torch.Tensor:
        B = ep["vals"].shape[0]
        dev = ep["vals"].device
        perm = ep["perm"]
        pA = perm.gather(1, ep["q_A"][:, None])[:, 0]
        s1 = DO0 + 2 * pA + ep["q_va"]
        pB = perm.gather(1, ep["q_B"].clamp_min(0)[:, None])[:, 0]
        s2 = torch.where(ep["q_B"] >= 0, DO0 + 2 * pB + ep["q_vb"], torch.full_like(s1, NULL))
        swap = torch.rand(B, device=dev) < 0.5
        s1, s2 = torch.where(swap, s2, s1), torch.where(swap, s1, s2)
        tgt = TGT0 + perm.gather(1, ep["q_T"][:, None])[:, 0]
        cells = (ep["vals"] * (2 ** perm)[:, None, :]).sum(-1)
        A, Bn = ep["roles"][:, 0], ep["roles"][:, 1]
        reg_label = torch.where(ep["reg"] == 1, 1 + perm.gather(1, A[:, None]),
                                torch.where(ep["reg"] == 2, 1 + perm.gather(1, Bn[:, None]),
                                            torch.zeros_like(ep["reg"])))
        samp = SAMP0 + 16 * reg_label + cells
        return torch.cat([torch.full((B, 1), BOS, device=dev), s1[:, None], s2[:, None], tgt[:, None], samp], 1)

    def sample_batch(self, batch_size: int, device) -> dict:
        ep = self._episodes(batch_size, device)
        pos = torch.arange(self.max_len, device=device)
        return dict(tokens=self._tokens(ep), targets={"ans": ep["ans"].float()[:, None, None]},
                    masks={"ans": (pos >= PREFIX - 1).expand(batch_size, -1)})

    # ---------------------------------------------------------------- eval sets + oracles
    def make_eval_sets(self, seed: int, n_episodes: int) -> dict:
        torch.manual_seed(seed)
        cpu = torch.device("cpu")
        n = self.n_eval_per_kind or max(64, n_episodes // 3)
        sets = {}
        for k, name in enumerate(QKINDS):
            ep = self._episodes(n, cpu, torch.full((n,), k))
            ep["tokens"] = self._tokens(ep)
            ep.update(self._oracles(ep))
            sets[name] = ep
        return sets

    _CACHE: dict = {}

    @classmethod
    def _tables(cls):
        if "t" not in cls._CACHE:
            dags = du.labeled_dags()
            topos = du.topologies()
            logprior = np.log([1.0 / (25 * topos[du.topology_id(pa)]["n_labelings"]) for pa in dags])
            rows = np.stack([np.stack([du.row_index(pa[i], du.ASSIGN) for i in range(4)]) for pa in dags])
            a = np.array([[du.bdeu_ab(bin(p).count("1")) for p in pa] for pa in dags])
            cls._CACHE["t"] = (dags, logprior, rows, a)
        return cls._CACHE["t"]

    def _oracles(self, ep: dict) -> dict:
        dags, logprior, rows, a = self._tables()
        D = len(dags)
        vals = ep["vals"].numpy()
        reg = ep["reg"].numpy()
        roles = ep["roles"].numpy()
        qA, qB, va, vb, T = (ep[f"q_{k}"].numpy() for k in ("A", "B", "va", "vb", "T"))
        pa_true = ep["pa"].numpy()
        B = vals.shape[0]
        bayes = np.zeros((B, len(EVAL_K)))
        sd_graph = np.zeros((B, len(EVAL_K)))
        true_graph = np.zeros((B, len(EVAL_K)))
        base = np.zeros((B, len(EVAL_K)))
        aa = a[:, :, None]
        used = np.arange(4)[None, None, :] < (2 ** np.round(-np.log2(aa))).astype(int)
        V = du.ASSIGN
        for b in range(B):
            A_, B_ = roles[b, 0], roles[b, 1]
            node_of_reg = np.array([-1, A_, B_])
            cell = (vals[b] * (1 << np.arange(4))).sum(1)
            mask = 1 << qA[b]
            dv = np.zeros((1, 4), np.int64)
            dv[0, qA[b]] = va[b]
            if qB[b] >= 0:
                mask |= 1 << qB[b]
                dv[0, qB[b]] = vb[b]
            tsel = V[:, T[b]] == 1
            for ki, k in enumerate(EVAL_K):
                cnt = np.zeros((3, 16))
                np.add.at(cnt, (reg[b, :k], cell[:k]), 1)
                n1 = np.zeros((D, 4, 4))
                n0 = np.zeros((D, 4, 4))
                for i in range(4):
                    ci = cnt[[g for g in range(3) if node_of_reg[g] != i]].sum(0)  # Cooper-Yoo
                    for r in range(4):
                        m = rows[:, i, :] == r
                        n1[:, i, r] = (m * (V[:, i] == 1) * ci).sum(1)
                        n0[:, i, r] = (m * (V[:, i] == 0) * ci).sum(1)
                ll = np.where(used, betaln(aa + n1, aa + n0) - betaln(aa, aa), 0.0).sum((1, 2))
                lw = logprior + ll
                w = np.exp(lw - lw.max())
                w /= w.sum()
                tbar = (aa + n1) / (2 * aa + n1 + n0)
                live = np.nonzero(w > 1e-10)[0]
                ans_d = np.array([du.joint_table(dags[d], tbar[d:d + 1], mask, dv)[0][tsel].sum() for d in live])
                ww = w[live] / w[live].sum()
                bayes[b, ki] = (ww * ans_d).sum()
                sd_graph[b, ki] = np.sqrt((ww * (ans_d - bayes[b, ki]) ** 2).sum())
                dtrue = dags.index(tuple(int(x) for x in pa_true[b]))
                true_graph[b, ki] = du.joint_table(dags[dtrue], tbar[dtrue:dtrue + 1], mask, dv)[0][tsel].sum()
                base[b, ki] = self._baseline(vals[b, :k], reg[b, :k], node_of_reg, qA[b], qB[b], va[b], vb[b], T[b])
        return dict(oracle_bayes=torch.as_tensor(bayes), oracle_sd_graph=torch.as_tensor(sd_graph),
                    oracle_true_graph=torch.as_tensor(true_graph), baseline=torch.as_tensor(base))

    @staticmethod
    def _freq(sel: np.ndarray, t: np.ndarray) -> float:
        return (t[sel].sum() + JEFF) / (sel.sum() + 2 * JEFF)

    def _baseline(self, vals, reg, node_of_reg, A, Bq, va, vb, T) -> float:
        """Model-free '1 3/4' answer: tried regimes by frequency, untried by nearest tried / obs."""
        t = vals[:, T]
        if Bq >= 0:  # combination of the two tried regimes
            gA = int(np.nonzero(node_of_reg == A)[0][0])
            gB = int(np.nonzero(node_of_reg == Bq)[0][0])
            e1 = self._freq((reg == gA) & (vals[:, A] == va) & (vals[:, Bq] == vb), t)
            e2 = self._freq((reg == gB) & (vals[:, Bq] == vb) & (vals[:, A] == va), t)
            return 0.5 * (e1 + e2)
        g = np.nonzero(node_of_reg == A)[0]
        if len(g):  # tried single regime
            return self._freq((reg == int(g[0])) & (vals[:, A] == va), t)
        return self._freq((reg == 0) & (vals[:, A] == va), t)  # untried single: obs conditional

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, model, eval_sets: dict, device, return_preds: bool = False):
        res: dict = {}
        preds: dict = {}
        cols = [PREFIX - 1 + k for k in EVAL_K]
        all_kl = []
        for name in QKINDS:
            ep = eval_sets[name]
            out = run_model(model, ep["tokens"], None, device)
            pm = sigmoid(out["ans"][:, cols, 0])
            bay = ep["oracle_bayes"].numpy()
            bas = ep["baseline"].numpy()
            sdg = ep["oracle_sd_graph"].numpy()
            kc = slice(1, None)  # k >= 24
            klm = kl_bernoulli_bits(bay, pm)[:, kc]
            klb = kl_bernoulli_bits(bay, bas)[:, kc]
            all_kl.append(klm)
            res[f"kl_{name}"] = float(klm.mean())
            res[f"kl_baseline_{name}"] = float(klb.mean())
            res[f"gc_{name}"] = float(1 - klm.mean() / max(klb.mean(), 1e-12))
            sharp = sdg[:, -1] < np.median(sdg[:, -1])
            for tag, sel in (("sharp", sharp), ("diffuse", ~sharp)):
                res[f"gc_{name}_{tag}"] = float(1 - klm[sel].mean() / max(klb[sel].mean(), 1e-12))
            res[f"kl_truegraph_vs_bayes_{name}"] = float(
                kl_bernoulli_bits(ep["oracle_true_graph"].numpy(), bay)[:, kc].mean())
            if return_preds:
                preds.update({f"{name}_model": pm.astype(np.float32), f"{name}_bayes": bay.astype(np.float32),
                              f"{name}_baseline": bas.astype(np.float32), f"{name}_sd_graph": sdg.astype(np.float32),
                              f"{name}_true_graph": ep["oracle_true_graph"].numpy().astype(np.float32)})
        res["kl_all"] = float(np.mean([k.mean() for k in all_kl]))
        if return_preds:
            return res, preds
        return res
