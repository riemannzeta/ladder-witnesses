import numpy as np
import torch
from scipy.special import betaln

from witnesses.tasks import dag_utils as du
from witnesses.tasks.e2_dags import (CELL0, PREFIX, REP, ROW0, ROWPAD, E2Dags, answers_torch,
                                     joint_torch)

CPU = torch.device("cpu")


def _batch(task, B=512, topos=None, kind="single", seed=0):
    torch.manual_seed(seed)
    return task._draw(B, CPU, topos or task.train_topos, kind)


def test_counts():
    assert len(du.labeled_dags()) == 443 and len(du.topologies()) == 25


def test_vectorized_answers_match_reference():
    t = E2Dags(mode="cpt", fold=0)
    for kind in ("single", "composite"):
        st, q = _batch(t, 256, kind=kind, seed=1)
        ok = q["B"] != -2
        do_a, cond_a = answers_torch(st["pa"], st["theta"].double(), q)
        ref = t._answers_np(st["pa"].numpy(), st["theta"].double().numpy(), {k: v.numpy() for k, v in q.items()})
        assert np.allclose(do_a.numpy()[ok], ref["do"][ok], atol=1e-9)
        assert np.allclose(cond_a.numpy()[ok], ref["cond"][ok], atol=1e-9)


def test_inert_second_slot_is_inert():
    t = E2Dags(mode="cpt", fold=0, p_inert=1.0)
    st, q = _batch(t, 2000, seed=2)
    has = q["B"] >= 0
    assert has.float().mean() > 0.3
    q1 = dict(q); q1["B"] = torch.full_like(q["B"], -1)
    d2, c2 = answers_torch(st["pa"], st["theta"].double(), q)
    d1, c1 = answers_torch(st["pa"], st["theta"].double(), q1)
    isc = q["is_cond"]
    assert torch.allclose(d2[has & ~isc], d1[has & ~isc], atol=1e-9)
    assert torch.allclose(c2[has & isc], c1[has & isc], atol=1e-9)


def test_heldout_topologies_never_in_training_batches():
    for fold in range(5):
        t = E2Dags(mode="cpt", fold=fold)
        st, _ = _batch(t, 4000, seed=3)
        assert not set(st["topo"].tolist()) & set(t.heldout_topos)


def _log_ml(pa, vals):
    a = np.array([du.bdeu_ab(bin(p).count("1")) for p in pa])[:, None]
    n1, n0 = E2Dags._row_counts(pa, vals)
    used = np.arange(4)[None, :] < (2 ** np.array([bin(p).count("1") for p in pa]))[:, None]
    return float(np.where(used, betaln(a + n1, a + n0) - betaln(a, a), 0).sum())


def test_bdeu_score_equivalence_no_orientation_leak():
    rng = np.random.default_rng(4)
    dags = du.labeled_dags()
    by_key = {}
    for pa in dags:
        by_key.setdefault(du.markov_key(pa), []).append(pa)
    classes = [v for v in by_key.values() if len(v) > 1]
    for cls in classes[:12]:
        vals = rng.integers(0, 2, size=(40, 4))
        lls = [_log_ml(pa, vals) for pa in cls]
        assert max(lls) - min(lls) < 1e-8, (cls, lls)


def test_presented_row_tokens_reconstruct_answers():
    t = E2Dags(mode="cpt", fold=1)
    st, q = _batch(t, 300, seed=5)
    toks, vals = t._row_tokens(st)
    perm = st["perm"].numpy()
    pa = st["pa"].numpy()
    th = st["theta"].numpy()
    for b in range(40):
        inv = np.argsort(perm[b])  # presented label -> internal node
        # presented parent masks
        pa_p = [0] * 4
        for i in range(4):
            for p in du.parents_of(int(pa[b, i])):
                pa_p[perm[b, i]] |= 1 << perm[b, p]
        th_p = np.full((1, 4, 4), 0.5)
        for tok, v in zip(toks[b].tolist(), vals[b].tolist()):
            if tok == ROWPAD:
                continue
            L, j = divmod(tok - ROW0, 4)
            th_p[0, L, j] = v
        jt_int = du.joint_table(tuple(int(x) for x in pa[b]), th[b:b + 1].astype(np.float64))[0]
        jt_pres = du.joint_table(tuple(pa_p), th_p)[0]
        cell_of = (du.ASSIGN << perm[b][None, :]).sum(-1)
        assert np.allclose(jt_pres[cell_of], jt_int, atol=1e-6)


def test_sample_cells_follow_presented_joint():
    t = E2Dags(mode="samples", fold=0, n_samples=64)
    torch.manual_seed(6)
    st, q = t._draw(8, CPU, t.train_topos)
    vals = t._sample_data(st, 20000)
    cells = t._cells(st, vals)
    jt = t._presented_joint(st)
    freq = torch.nn.functional.one_hot(cells, 16).float().mean(1)
    assert (freq - jt).abs().max() < 0.02


def test_composite_both_elements_relevant():
    t = E2Dags(mode="cpt", fold=0)
    torch.manual_seed(7)
    ep = t._make_split(64, CPU, t.train_topos, "composite")
    q = {k[2:]: ep[k] for k in ep if k.startswith("q_")}
    assert (q["B"] >= 0).all()
    for drop in ("A", "B"):
        q1 = dict(q)
        if drop == "A":
            q1["A"], q1["va"] = q["B"], q["vb"]
        q1["B"] = torch.full_like(q["B"], -1)
        full = answers_torch(ep["pa"], ep["theta"].double(), q)
        part = answers_torch(ep["pa"], ep["theta"].double(), q1)
        isc = q["is_cond"]
        diff = torch.where(isc, (full[1] - part[1]).abs(), (full[0] - part[0]).abs())
        assert (diff > 0.05).all()
