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


def test_pair_table_both_elements_effective():
    """Every PAIRS entry (post-hoc two-element training arm) moves the answer when either element
    is dropped, for generic CPTs; and every structurally valid pair is listed."""
    from witnesses.tasks.e2_dags import ANC_CUT, DSEP, NPAIRS, PAIRS
    t = E2Dags(mode="cpt", fold=0)
    torch.manual_seed(11)
    for topo in range(25):
        st = t._structure(torch.full((64,), topo), CPU)
        for T in range(4):
            for kind in (0, 1):
                listed = {tuple(PAIRS[topo, T, kind, j]) for j in range(NPAIRS[topo, T, kind])}
                others = [x for x in range(4) if x != T]
                for x in others:
                    for y in others:
                        if x >= y:
                            continue
                        if kind == 0:
                            valid = bool(ANC_CUT[topo, T, x] >> y & 1) and bool(ANC_CUT[topo, T, y] >> x & 1)
                        else:
                            valid = not DSEP[topo, T, x, y] and not DSEP[topo, T, y, x]
                        assert valid == ((x, y) in listed)
                for x, y in listed:
                    q = dict(is_cond=torch.full((64,), bool(kind)), T=torch.full((64,), T),
                             A=torch.full((64,), x), B=torch.full((64,), y),
                             va=torch.randint(0, 2, (64,)), vb=torch.randint(0, 2, (64,)))
                    full = answers_torch(st["pa"], st["theta"].double(), q)[kind]
                    for keep in ("A", "B"):
                        q1 = dict(q, B=torch.full((64,), -1))
                        if keep == "B":
                            q1["A"], q1["va"] = q["B"], q["vb"]
                        part = answers_torch(st["pa"], st["theta"].double(), q1)[kind]
                        assert (full - part).abs().max() > 0.05, (topo, T, kind, x, y, keep)


def test_composite_training_mix_is_training_only():
    """p_composite_train changes training batches (about that share of queries gets two effective
    elements) but not the frozen eval splits."""
    base = E2Dags(mode="cpt", fold=1, n_eval_per_split=64)
    arm = E2Dags(mode="cpt", fold=1, n_eval_per_split=64, p_composite_train=0.25)
    torch.manual_seed(5)
    st, q = arm._draw(8000, CPU, arm.train_topos, "single")
    q2 = arm._mix_composite(st, q)
    changed = (q2["A"] != q["A"]) | (q2["B"] != q["B"])
    assert 0.12 < changed.float().mean() < 0.25
    two = q2["B"] >= 0
    d2, c2 = answers_torch(st["pa"], st["theta"].double(), q2)
    q1 = dict(q2, B=torch.full_like(q2["B"], -1))
    d1, c1 = answers_torch(st["pa"], st["theta"].double(), q1)
    moved = torch.where(q2["is_cond"], (c2 - c1).abs(), (d2 - d1).abs()) > 1e-6
    assert (moved & changed).float().sum() > 0.9 * changed.float().sum()
    assert two[changed].all()
    for kind in ("single", "composite"):
        torch.manual_seed(9)
        a = base._make_split(64, CPU, base.heldout_topos, kind)
        torch.manual_seed(9)
        b = arm._make_split(64, CPU, arm.heldout_topos, kind)
        for k in a:
            if isinstance(a[k], torch.Tensor):
                assert torch.equal(a[k], b[k]), k
    assert base.eval_signature() == arm.eval_signature()


def test_conditional_oracle_survives_beta_underflow():
    """Draws in which a Beta(1/4, 1/4) row underflows to exactly 0 or 1 make the conditioning event
    impossible (0/0); the MC oracle must drop them rather than return NaN."""
    pa = tuple(int(x) for x in REP[17])
    rng = np.random.default_rng(0)
    draws = rng.beta(0.25, 0.25, size=(256, 4, 4))
    draws[:8] = np.where(draws[:8] > 0.5, 1.0, 0.0)  # force exact 0/1 rows in some draws
    for T in range(4):
        for A in range(4):
            for Bn in range(4):
                if len({T, A, Bn}) < 3:
                    continue
                for va in (0, 1):
                    q = dict(A=np.array([A]), B=np.array([Bn]), va=np.array([va]), vb=np.array([1 - va]),
                             T=np.array([T]), is_cond=np.array([True]))
                    v = E2Dags._cond_mc(pa, draws, q)
                    assert np.isfinite(v) and 0 <= v <= 1
