"""E5 tests: upstream Bayes-math checks (ported), brute-force oracle checks, witness-pair
invariants, oracle sanity and calibration, masks, and an oracle-as-model evaluate check."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from witnesses.tasks.e5_recurrence import (CF_TOKEN, K, N_CF, P, E5aRestartClock, E5bNoiseCF,
                                           cf_oracle, noisy_factual_oracle, noisy_posterior,
                                           recurrence_oracle)

CPU = torch.device("cpu")
LOG2P = math.log2(P)
PI = 0.5
EPS = 0.2


# ============================================================ upstream reference (ported)
# Logic copied from bayesian-wind-tunnel experiments/construction_boundary/recurrence_bwt.py
# (count_consistent_recurrences, recover_recurrence, bayes_factor_recurrence,
# class_posterior_recurrence, bayesian_predictive_recurrence, _predictive_entropy); comments trimmed.


def count_consistent_recurrences(seq, p):
    k = len(seq)
    if k <= 1:
        return p * p
    if k == 2:
        return p
    dx0 = (seq[1] - seq[0]) % p
    dx1 = (seq[2] - seq[1]) % p
    if dx0 == 0:
        if dx1 != 0:
            return 0
        for t in range(3, k):
            if seq[t] != seq[0]:
                return 0
        return p
    dx0_inv = pow(dx0, p - 2, p)
    a = (dx1 * dx0_inv) % p
    b = (seq[1] - a * seq[0]) % p
    for t in range(2, k - 1):
        if (a * seq[t] + b) % p != seq[t + 1]:
            return 0
    return 1


def recover_recurrence(seq, p):
    if len(seq) < 3:
        return None
    dx0 = (seq[1] - seq[0]) % p
    dx1 = (seq[2] - seq[1]) % p
    if dx0 == 0:
        if dx1 != 0:
            return None
        return (0, seq[0] % p)
    dx0_inv = pow(dx0, p - 2, p)
    a = (dx1 * dx0_inv) % p
    b = (seq[1] - a * seq[0]) % p
    return (a, b)


def bayes_factor_recurrence(seq, p):
    k = len(seq)
    if k == 0:
        return 1.0
    count = count_consistent_recurrences(seq, p)
    if count == 0:
        return 0.0
    return math.exp(math.log(count) + (k - 3) * math.log(p))


def class_posterior_recurrence(seq, p, pi):
    bf = bayes_factor_recurrence(seq, p)
    if bf == 0.0:
        return 0.0
    return pi * bf / (pi * bf + (1 - pi))


def bayesian_predictive_recurrence(seq, p, pi):
    k = len(seq)
    w = class_posterior_recurrence(seq, p, pi)
    if k <= 1:
        return {v: 1.0 / p for v in range(p)}
    if k == 2:
        if (seq[1] - seq[0]) % p == 0:
            pred_hp = {v: (1.0 if v == seq[0] else 0.0) for v in range(p)}
        else:
            pred_hp = {v: 1.0 / p for v in range(p)}
        return {v: w * pred_hp[v] + (1 - w) / p for v in range(p)}
    count = count_consistent_recurrences(seq, p)
    if count == 0:
        return {v: 1.0 / p for v in range(p)}
    ab = recover_recurrence(seq, p)
    if ab is None:
        return {v: 1.0 / p for v in range(p)}
    a, b = ab
    x_next = (a * seq[-1] + b) % p
    if count > 1:
        pred_hp = {v: (1.0 if v == seq[0] else 0.0) for v in range(p)}
    else:
        pred_hp = {v: (1.0 if v == x_next else 0.0) for v in range(p)}
    return {v: w * pred_hp[v] + (1 - w) / p for v in range(p)}


def _predictive_entropy(pred):
    return -sum(q * math.log2(q) for q in np.asarray(pred).ravel() if q > 1e-15)


def _program_seq(a, b, x0, n):
    a, b, seq = int(a), int(b), [int(x0)]
    for _ in range(n - 1):
        seq.append((a * seq[-1] + b) % P)
    return seq


def l1(seqs, pi=PI):
    x = np.asarray(seqs, np.int64)
    return recurrence_oracle(x, np.full(len(x), -1), "l1", pi)


def bf_from_post(w, pi=PI):
    return w / (1 - w) * (1 - pi) / pi


# ============================================================ 1. upstream verify_bayesian_calculations


def test_upstream_bf_is_one_for_first_three_observations():
    rng = np.random.default_rng(0)
    for _ in range(200):
        seq = _program_seq(rng.integers(P), rng.integers(P), rng.integers(P), 4)
        if seq[0] == seq[1]:
            continue
        post = l1([seq])["post_prog"][0]
        for k in (1, 2, 3):  # k observations x_0..x_{k-1} -> oracle position t = k - 1
            assert abs(bf_from_post(post[k - 1]) - 1.0) < 1e-10
            assert abs(bayes_factor_recurrence(seq[:k], P) - 1.0) < 1e-10


def test_upstream_bf_is_p_after_four_observations():
    rng = np.random.default_rng(1)
    found = 0
    for _ in range(200):
        seq = _program_seq(rng.integers(1, P), rng.integers(P), rng.integers(P), 5)
        if seq[0] == seq[1]:
            continue
        post = l1([seq])["post_prog"][0]
        assert abs(bf_from_post(post[3]) - P) < 1e-6
        found += 1
    assert found > 0


def test_upstream_count_trajectory_and_exact_counts():
    rng = np.random.default_rng(2)
    for _ in range(200):
        seq = _program_seq(rng.integers(1, P), rng.integers(P), rng.integers(P), 7)
        if seq[0] != seq[1]:
            break
    n = l1([seq])["n_cons"][0]  # n[t] = programs consistent with x_0..x_t
    assert count_consistent_recurrences([], P) == P * P
    assert list(n[:4]) == [P * P, P, 1, 1]
    # exact agreement with the upstream counter on every prefix of mixed sequences
    seqs = [_program_seq(rng.integers(P), rng.integers(P), rng.integers(P), 16) for _ in range(60)]
    seqs += [[int(c) for c in rng.integers(0, P, 16)] for _ in range(60)]
    seqs += [[c] * 16 for c in range(3)] + [[4, 4, 4, 9] + [4] * 12]
    n = l1(seqs)["n_cons"]
    for i, sq in enumerate(seqs):
        for t in range(16):
            assert n[i, t] == count_consistent_recurrences(sq[: t + 1], P)


def test_upstream_early_entropies():
    rng = np.random.default_rng(3)
    # upstream H(0) is the empty prefix (x_0 is never predicted here, as upstream); H(1) given x_0:
    pred = l1([[x0] + [0] * 15 for x0 in range(P)])["pred"]
    for x0 in range(P):
        assert abs(_predictive_entropy(pred[x0, 0]) - LOG2P) < 1e-10
    # H(2 | fixed point) < log2 p ; H(2 | generic) = log2 p
    pred = l1([[5, 5] + [0] * 14])["pred"]
    assert _predictive_entropy(pred[0, 1]) < LOG2P - 0.01
    for _ in range(200):
        x0, x1 = rng.integers(P), rng.integers(P)
        if x0 == x1:
            continue
        pred = l1([[x0, x1] + [0] * 14])["pred"]
        assert abs(_predictive_entropy(pred[0, 1]) - LOG2P) < 1e-10
        break


def test_upstream_noise_prior_gives_uniform_entropy():
    rng = np.random.default_rng(4)
    pred = l1([list(rng.integers(0, P, 16)) for _ in range(20)], pi=0.0)["pred"]
    assert np.abs(pred - 1.0 / P).max() < 1e-12


def test_upstream_posterior_trajectory():
    seq = _program_seq(3, 7, 2, 9)
    post = l1([seq + [0] * 7])["post_prog"][0]
    for k in range(1, 10):
        w = class_posterior_recurrence(seq[:k], P, PI)
        bf = P ** max(0, k - 3)
        assert abs(w - PI * bf / (PI * bf + 1 - PI)) < 1e-12
        assert abs(post[k - 1] - w) < 1e-12


def test_l1_matches_upstream_predictive_on_all_prefixes():
    rng = np.random.default_rng(5)
    seqs = [_program_seq(rng.integers(P), rng.integers(P), rng.integers(P), 16) for _ in range(40)]
    seqs += [_program_seq(0, rng.integers(P), rng.integers(P), 16) for _ in range(10)]  # a = 0
    seqs += [[int(c) for c in rng.integers(0, P, 16)] for _ in range(40)]
    seqs += [[6] * 16, [6, 6, 6, 1] + [6] * 12, [6, 6, 2] + [0] * 13]
    pred = l1(seqs)["pred"]
    for i, sq in enumerate(seqs):
        for t in range(15):
            up = bayesian_predictive_recurrence(sq[: t + 1], P, PI)
            ref = np.array([up[v] for v in range(P)])
            assert np.abs(pred[i, t] - ref).max() < 1e-9, (sq, t)


# ============================================================ 2. brute-force oracles


def _simulate(a, b, x0, s, v, mode, T):
    """Program trajectory under the semantics (explicit forward simulation)."""
    nat = [x0]
    for _ in range(1, T):
        nat.append((a * nat[-1] + b) % P)
    if mode == "l1" or s < 0:
        return nat, nat
    z = [x0]
    for u in range(1, T):
        if u == s:
            z.append(v)
        elif mode == "restart":
            z.append((a * z[-1] + b) % P)
        else:  # clock
            z.append(nat[u])
    return z, nat


def brute_e5a(x, s, mode, pi=PI):
    """Slow exact posterior predictive: loop over H_R and all 289 programs."""
    T = len(x)
    interv = mode != "l1" and s >= 0
    out = np.zeros((T - 1, P))
    sims = [_simulate(k // P, k % P, x[0], s, x[s] if s >= 0 else -1, mode, T) for k in range(K)]
    for t in range(T - 1):
        w_r = (1 - pi) * (1.0 / P) ** (t + 1)
        pred = np.full(P, w_r / P)
        Z = w_r
        for k in range(K):
            z, nat = sims[k]
            a, b = k // P, k % P
            lik = 1.0 / P  # x_0
            for u in range(1, t + 1):
                lik *= (1.0 / P) if (interv and u == s) else float(x[u] == z[u])
            w = pi / K * lik
            nxt = (a * z[t] + b) % P if (interv and t + 1 == s) else z[t + 1]
            pred[nxt] += w
            Z += w
        out[t] = pred / Z
    return out


def _e5a_cases(n_each=6):
    eps = []
    for sem in ("restart", "clock"):
        torch.manual_seed(11)
        _, lat = E5aRestartClock(sem, p_interv=0.7).sample_batch(4 * n_each, CPU, return_latents=True)
        eps += [(lat["x"][i].tolist(), int(lat["s"][i])) for i in range(4 * n_each)]
    # crafted: identity program, a = 0, fixed points, boundary intervention positions
    for sem in ("restart", "clock"):
        for (a, b, x0, s, v) in [(1, 0, 4, 1, 9), (1, 0, 4, 14, 9), (0, 5, 2, 3, 11), (3, 7, 2, 5, 0),
                                 (3, 7, 2, 1, 13), (16, 1, 8, 2, 8), (2, 0, 0, 6, 1)]:
            z, _ = _simulate(a, b, x0, s, v, sem, 16)
            eps.append((z, s))
    return eps


@pytest.mark.parametrize("mode", ["restart", "clock", "l1"])
def test_e5a_oracles_match_brute_force(mode):
    cases = _e5a_cases()
    x = np.array([c[0] for c in cases])
    s = np.array([c[1] for c in cases])
    vec = recurrence_oracle(x, s, mode, PI, chunk=7)["pred"]  # small chunk exercises chunking
    for i, (xi, si) in enumerate(cases):
        assert np.abs(vec[i] - brute_e5a(xi, si, mode)).max() < 1e-9, (mode, xi, si)


def brute_e5b_factual(x, eps=EPS, pi=PI, prior=None):
    T = len(x)
    pk = np.full(K, pi / K) if prior is None else pi * prior / prior.sum()
    out = np.zeros((T - 1, P))
    for t in range(T - 1):
        w_r = (1 - pi) * (1.0 / P) ** (t + 1)
        pred = np.full(P, w_r / P)
        Z = w_r
        for k in range(K):
            a, b = k // P, k % P
            lik = 1.0 / P
            for u in range(t):
                r = (x[u + 1] - a * x[u] - b) % P
                lik *= (1 - eps) if r == 0 else (eps if r == 1 else 0.0)
            w = pk[k] * lik
            f = (a * x[t] + b) % P
            pred[f] += w * (1 - eps)
            pred[(f + 1) % P] += w * eps
            Z += w
        out[t] = pred / Z
    return out


def brute_e5b_cf(x, q, v, y, mode, eps=EPS, pi=PI, prior=None):
    """Explicit hypothesis list; delta -> 0 contamination limit: keep the hypotheses with the
    fewest zero-probability y observations."""
    T = len(x)
    pk = np.full(K, pi / K) if prior is None else pi * prior / prior.sum()
    tpost = T - 1 if mode != "l2cf" else q
    hyps = [dict(kind="R", w=(1 - pi) * (1.0 / P) ** (tpost + 1), inc=0)]
    for k in range(K):
        a, b = k // P, k % P
        lik = 1.0 / P
        for u in range(tpost):
            r = (x[u + 1] - a * x[u] - b) % P
            lik *= (1 - eps) if r == 0 else (eps if r == 1 else 0.0)
        if pk[k] * lik > 0:
            hyps.append(dict(kind="P", a=a, b=b, w=pk[k] * lik, inc=0))
    out = np.zeros((len(y), P))
    par = v
    for j in range(len(y)):
        u = q + j
        dists = []
        for h in hyps:
            d = np.zeros(P)
            if mode == "abducted":
                if h["kind"] == "R":
                    d[x[u + 1]] = 1.0
                else:
                    e = (x[u + 1] - h["a"] * x[u] - h["b"]) % P  # abducted factual noise
                    d[(h["a"] * par + h["b"] + e) % P] = 1.0
            else:
                if h["kind"] == "R":
                    d[:] = 1.0 / P
                else:
                    f = (h["a"] * par + h["b"]) % P
                    d[f] += 1 - eps
                    d[(f + 1) % P] += eps
            dists.append(d)
        minc = min(h["inc"] for h in hyps)
        Z = sum(h["w"] for h in hyps if h["inc"] == minc)
        out[j] = sum(h["w"] / Z * d for h, d in zip(hyps, dists) if h["inc"] == minc)
        for h, d in zip(hyps, dists):
            lk = d[y[j]]
            if lk > 0:
                h["w"] *= lk
            else:
                h["inc"] += 1
        par = y[j]
    return out


def _e5b_cases(n=24):
    eps = []
    for labels in ("abducted", "fresh"):
        torch.manual_seed(12)
        t = E5bNoiseCF(labels)
        _, lat = t.sample_batch(n, CPU, return_latents=True)
        for i in range(n):
            eps.append((lat["x"][i].numpy(), int(lat["q"][i]), int(lat["v"][i]), lat["y"][i].numpy()))
        # held-out programs and a = 0 programs
        lat = t._latents(6, CPU, "held", force_prog=True)
        x = t._factual(lat)
        v, y = t._counterfactual(lat, x, labels)
        eps += [(x[i].numpy(), int(lat["q"][i]), int(v[i]), y[i].numpy()) for i in range(6)]
    x = np.array(_program_seq(0, 5, 3, 16))
    eps.append((x, 0, 9, np.array([5, 5, 5])))
    eps.append((x, 12, 9, np.array([6, 2, 5])))  # inconsistent with every abducted hypothesis
    return eps


@pytest.mark.parametrize("train_prior", [False, True])
def test_e5b_factual_oracle_matches_brute_force(train_prior):
    cases = _e5b_cases()
    prior = E5bNoiseCF().seen_mask if train_prior else None
    x = np.stack([c[0] for c in cases])
    vec = noisy_factual_oracle(x, EPS, PI, prior, chunk=5)["pred"]
    for i in range(0, len(cases), 3):
        assert np.abs(vec[i] - brute_e5b_factual(cases[i][0], prior=prior)).max() < 1e-9


@pytest.mark.parametrize("mode", ["abducted", "fresh", "l2cf"])
@pytest.mark.parametrize("train_prior", [False, True])
def test_e5b_cf_oracles_match_brute_force(mode, train_prior):
    cases = _e5b_cases()
    prior = E5bNoiseCF().seen_mask if train_prior else None
    x, q, v, y = (np.stack([np.asarray(c[j]) for c in cases]) for j in range(4))
    vec = cf_oracle(x, q, v, y, mode, EPS, PI, prior, chunk=5)
    for i, (xi, qi, vi, yi) in enumerate(cases):
        ref = brute_e5b_cf(xi, qi, vi, yi, mode, prior=prior)
        assert np.abs(vec[i] - ref).max() < 1e-9, (mode, i)


# ============================================================ 3. witness-pair invariants


def test_restart_and_clock_share_the_generator_path():
    torch.manual_seed(0)
    br, lr = E5aRestartClock("restart").sample_batch(4096, CPU, return_latents=True)
    torch.manual_seed(0)
    bc, lc = E5aRestartClock("clock").sample_batch(4096, CPU, return_latents=True)
    for k in ("is_prog", "a", "b", "x0", "noise", "s", "v"):
        assert torch.equal(lr[k], lc[k])
    no = lr["s"] < 0
    assert 0.4 < no.float().mean() < 0.6
    assert torch.equal(br["tokens"][no], bc["tokens"][no])
    assert torch.equal(br["targets"]["nxt"][no], bc["targets"]["nxt"][no])
    assert torch.equal(br["masks"]["nxt"], bc["masks"]["nxt"])
    # intervened: identical through x_s; noise episodes identical everywhere
    pos = torch.arange(16)
    upto = pos[None, :] <= lr["s"][:, None]
    assert torch.equal(br["tokens"][upto], bc["tokens"][upto])
    assert torch.equal(br["tokens"][~lr["is_prog"]], bc["tokens"][~lr["is_prog"]])
    # and they do differ downstream for a != 0 programs whenever v != natural value
    d = (~no) & lr["is_prog"] & (lr["a"] != 0)
    assert (br["tokens"][d] != bc["tokens"][d]).any(1).float().mean() > 0.9


def test_oracles_coincide_on_unintervened_episodes():
    torch.manual_seed(1)
    _, lat = E5aRestartClock("restart").sample_batch(2048, CPU, return_latents=True)
    no = (lat["s"] < 0).numpy()
    x, s = lat["x"].numpy()[no], lat["s"].numpy()[no]
    r = recurrence_oracle(x, s, "restart")["pred"]
    c = recurrence_oracle(x, s, "clock")["pred"]
    o = recurrence_oracle(x, s, "l1")["pred"]
    assert np.array_equal(r, c) and np.array_equal(r, o)


def test_intervention_value_independent_of_natural_value():
    torch.manual_seed(2)
    t = E5aRestartClock("restart", p_interv=1.0)
    _, lat = t.sample_batch(200_000, CPU, return_latents=True)
    a, b, s, v = lat["a"], lat["b"], lat["s"], lat["v"]
    nat = lat["x0"].clone()
    for u in range(1, 16):  # natural value at s: f^s(x_0) under H_P, the iid draw under H_R
        nat = torch.where(s >= u, (a * nat + b) % P, nat)
    nat = torch.where(lat["is_prog"], nat, lat["noise"].gather(1, s[:, None])[:, 0])
    assert torch.equal(lat["x"].gather(1, s[:, None])[:, 0], v)
    assert abs((v == nat).float().mean().item() - 1 / P) < 0.003
    tab = np.zeros((P, P))
    np.add.at(tab, (nat.numpy(), v.numpy()), 1)
    assert np.abs(tab / tab.sum(1, keepdims=True) - 1 / P).max() < 0.015
    assert np.abs(np.bincount(s.numpy(), minlength=16)[1:15] / len(s) - 1 / 14).max() < 0.003
    assert (s >= 1).all() and (s <= 14).all()


def test_e5b_factual_stream_identical_across_labelings():
    T = 16
    torch.manual_seed(3)
    ba, la = E5bNoiseCF("abducted").sample_batch(4096, CPU, return_latents=True)
    torch.manual_seed(3)
    bf, lf = E5bNoiseCF("fresh").sample_batch(4096, CPU, return_latents=True)
    assert torch.equal(ba["tokens"][:, : T + 2], bf["tokens"][:, : T + 2])  # x (+query), [CF], [v]
    assert torch.equal(ba["targets"]["nxt"], bf["targets"]["nxt"])
    for h in ("nxt", "cf"):
        assert torch.equal(ba["masks"][h], bf["masks"][h])
    assert not torch.equal(ba["tokens"][:, T + 2 :], bf["tokens"][:, T + 2 :])
    # abducted labels: H_R copies x_{q+k}; a = 0 programs also equal the factual continuation
    x, q, y = la["x"], la["q"], la["y"]
    fut = x.gather(1, q[:, None] + 1 + torch.arange(N_CF)[None, :])
    copy = ~la["is_prog"] | (la["a"] == 0)
    assert torch.equal(y[copy], fut[copy])
    assert (la["v"] != x.gather(1, q[:, None])[:, 0]).all()


# ============================================================ 4. oracle sanity


def test_flag_regression_l1_uniform_l2_sharp():
    """The earlier confound: a substituted token at s in {3, 4} makes the flag-blind posterior
    conclude noise (exactly uniform); with the regime flag, the restart oracle's program component
    is a point mass on f(v), weighted by P(H_P | x_0..x_{s-1}) (the transition into s carries no
    evidence), and it is sharp (> 0.99) two transitions later."""
    rng = np.random.default_rng(6)
    xs, ss, want, nxt2 = [], [], [], []
    while len(xs) < 1000:
        a, b, x0 = rng.integers(1, P), rng.integers(P), rng.integers(P)
        s = int(rng.integers(3, 5))
        z = _program_seq(a, b, x0, 16)
        if z[0] == z[1]:
            continue
        v = int((z[s] + rng.integers(1, P)) % P)
        z, _ = _simulate(int(a), int(b), z[0], s, v, "restart", 16)
        xs.append(z)
        ss.append(s)
        want.append((a * v + b) % P)
        nxt2.append(z[s + 3])
    x, s = np.array(xs), np.array(ss)
    o1 = recurrence_oracle(x, s, "l1")
    o2 = recurrence_oracle(x, s, "restart")
    for i in range(len(x)):
        si = s[i]
        assert np.abs(o1["pred"][i, si:] - 1 / P).max() < 1e-6
        assert o1["post_prog"][i, si] < 1e-6
        w = class_posterior_recurrence(xs[i][:si], P, PI)  # upstream, prefix before the jump
        assert abs(o2["post_prog"][i, si] - w) < 1e-12
        assert abs(o2["pred"][i, si, want[i]] - (w + (1 - w) / P)) < 1e-12
        assert o2["pred"][i, si + 2, nxt2[i]] > 0.99


def test_e5b_abducted_concentrates_fresh_is_diffuse():
    torch.manual_seed(4)
    t = E5bNoiseCF("abducted")
    _, lat = t.sample_batch(3000, CPU, return_latents=True)
    x, q, v, y = (lat[k].numpy() for k in ("x", "q", "v", "y"))
    B = len(x)
    wp, wr = noisy_posterior(x, 15, EPS, PI)
    ab = cf_oracle(x, q, v, y, "abducted")
    fr = cf_oracle(x, q, v, y, "fresh")
    isp, k = lat["is_prog"].numpy(), lat["k"].numpy()
    ident = isp & (wp[np.arange(B), k] > 0.999)
    assert ident.sum() > 800
    rows = np.nonzero(ident)[0]
    for j in range(N_CF):
        assert (ab[rows, j, y[rows, j]] > 0.99).all()
    sure = ~isp & (wr > 0.999)
    assert sure.sum() > 800
    rows = np.nonzero(sure)[0]
    for j in range(N_CF):
        assert (ab[rows, j, x[rows, q[rows] + j + 1]] > 0.99).all()
    # fresh: a mixture of two-point (max 1-eps) and uniform distributions is never sharper than 1-eps
    assert fr.max() <= 1 - EPS + 1e-9
    assert (fr[ident, 0].max(-1) > 1 - EPS - 0.01).all()
    assert np.abs(fr[sure] - 1 / P).max() < 1e-3


def _calibration_gap(orc, tgt, sel, nbins=10, min_n=3000, min_bins=2):
    """Largest |E[target | oracle bin] - E[oracle | bin]| over (episode, position, value) entries.
    A Bayes predictive of the generator is calibrated against the generator's soft targets."""
    o = orc[sel].ravel()
    g = tgt[sel].ravel()
    bins = np.minimum((o * nbins).astype(int), nbins - 1)
    gaps = [abs(g[bins == i].mean() - o[bins == i].mean()) for i in range(nbins)
            if (bins == i).sum() >= min_n]
    assert len(gaps) >= min_bins
    return max(gaps)


@pytest.mark.parametrize("sem", ["restart", "clock"])
def test_e5a_oracle_calibrated_against_generator(sem):
    torch.manual_seed(7)
    t = E5aRestartClock(sem, p_interv=1.0)
    b, lat = t.sample_batch(20_000, CPU, return_latents=True)
    x, s = lat["x"].numpy(), lat["s"].numpy()
    tgt = b["targets"]["nxt"][:, :-1].numpy()
    sup = b["masks"]["nxt"][:, :-1].numpy()
    post = np.arange(15)[None, :] >= s[:, None]
    own = recurrence_oracle(x, s, sem)["pred"]
    other = recurrence_oracle(x, s, t.other)["pred"]
    assert _calibration_gap(own, tgt, sup) < 0.02
    assert _calibration_gap(own, tgt, post) < 0.02
    assert _calibration_gap(other, tgt, post, min_bins=1) > 0.2  # negative control


@pytest.mark.parametrize("labels", ["abducted", "fresh"])
def test_e5b_oracle_calibrated_against_generator(labels):
    torch.manual_seed(8)
    t = E5bNoiseCF(labels)
    b, lat = t.sample_batch(12_000, CPU, return_latents=True)
    x, q, v, y = (lat[k].numpy() for k in ("x", "q", "v", "y"))
    prior = t.seen_mask  # training episodes use seen programs only
    tf = b["targets"]["nxt"][:, :15].numpy()
    tc = b["targets"]["cf"][:, 17:20].numpy()
    fact = noisy_factual_oracle(x, EPS, PI, prior)["pred"]
    own = cf_oracle(x, q, v, y, labels, EPS, PI, prior)
    other = cf_oracle(x, q, v, y, t.other, EPS, PI, prior)
    l2 = cf_oracle(x, q, v, y, "l2cf", EPS, PI, prior)
    allf, allc = np.ones(fact.shape[:2], bool), np.ones(own.shape[:2], bool)
    assert _calibration_gap(fact, tf, allf) < 0.02
    assert _calibration_gap(own, tc, allc) < 0.02
    if labels == "fresh":  # negative control: abducted point masses vs fresh targets
        assert _calibration_gap(other, tc, allc, min_bins=1) > 0.1
    # (fresh and L2-for-CF are only *marginally* calibrated against abducted targets: they are
    # the abducted label's expectation given less information. Compare cross-entropies instead.)
    ce = lambda o: float(-(tc * np.log2(np.maximum(o, 1e-12))).sum(-1).mean())  # noqa: E731
    assert ce(own) < ce(other) - 0.3 and ce(own) < ce(l2)


def test_oracles_are_normalized_and_own_never_needs_the_fallback():
    torch.manual_seed(9)
    _, lat = E5aRestartClock("clock").sample_batch(512, CPU, return_latents=True)
    for mode in ("restart", "clock", "l1"):
        pr = recurrence_oracle(lat["x"].numpy(), lat["s"].numpy(), mode)["pred"]
        assert np.abs(pr.sum(-1) - 1).max() < 1e-12 and pr.min() >= 0
    for labels in ("abducted", "fresh"):
        t = E5bNoiseCF(labels)
        b, lat = t.sample_batch(512, CPU, return_latents=True)
        x, q, v, y = (lat[k].numpy() for k in ("x", "q", "v", "y"))
        pr = noisy_factual_oracle(x)["pred"]
        assert np.abs(pr.sum(-1) - 1).max() < 1e-12
        for mode in ("abducted", "fresh", "l2cf"):
            pr = cf_oracle(x, q, v, y, mode)
            assert np.abs(pr.sum(-1) - 1).max() < 1e-12 and pr.min() >= 0
        # own oracle gives the realized label positive probability (no contamination fallback)
        own = cf_oracle(x, q, v, y, labels)
        got = np.take_along_axis(own, y[:, :, None], 2)[..., 0]
        assert got.min() > 0


# ============================================================ 5. masks, shapes, held-out programs


@pytest.mark.parametrize("sem", ["restart", "clock"])
def test_e5a_masks_never_supervise_an_intervened_token(sem):
    t = E5aRestartClock(sem)
    torch.manual_seed(5)
    b, lat = t.sample_batch(4096, CPU, return_latents=True)
    tok, mask, tgt = b["tokens"], b["masks"]["nxt"], b["targets"]["nxt"]
    assert tok.shape == (4096, 16) and tok.max() < t.vocab_size and tgt.shape == (4096, 16, P)
    flagged = tok >= P
    assert torch.equal(flagged.sum(1), (lat["s"] >= 0).long())
    assert not flagged[:, 0].any() and not flagged[:, -1].any()
    expected = torch.ones_like(mask)
    expected[:, -1] = False
    expected[:, :-1] &= ~flagged[:, 1:]
    assert torch.equal(mask, expected)
    # H_P targets: one-hot at the realized next token; H_R: uniform
    isp = lat["is_prog"]
    nxt = (tok[:, 1:] % P)
    assert torch.equal(tgt[isp, :-1].argmax(-1), nxt[isp])
    assert torch.allclose(tgt[~isp], torch.full_like(tgt[~isp], 1 / P))


def test_e5b_shapes_tokens_and_masks():
    t = E5bNoiseCF("fresh")
    torch.manual_seed(6)
    b, lat = t.sample_batch(2048, CPU, return_latents=True)
    tok = b["tokens"]
    assert tok.shape == (2048, 21) and tok.max() < t.vocab_size == 52 and t.max_len == 21
    assert (tok[:, 16] == CF_TOKEN).all()
    assert torch.equal(tok[:, 17], P + lat["v"])
    qpos = (tok[:, :16] >= 2 * P).nonzero()
    assert torch.equal(qpos[:, 0], torch.arange(2048)) and torch.equal(qpos[:, 1], lat["q"])
    assert lat["q"].min() == 0 and lat["q"].max() == 12
    pos = torch.arange(21)
    assert torch.equal(b["masks"]["nxt"][0], pos < 15)
    assert torch.equal(b["masks"]["cf"][0], (pos >= 17) & (pos <= 19))
    for h in ("nxt", "cf"):
        assert torch.allclose(b["targets"][h].sum(-1), torch.ones(2048, 21))


def test_e5b_heldout_programs_never_in_training_batches():
    t = E5bNoiseCF("abducted")
    assert len(t.held) == round(0.2 * K) and len(np.intersect1d(t.held, t.seen)) == 0
    assert len(t.held) + len(t.seen) == K
    assert np.array_equal(t.held, E5bNoiseCF("fresh").held)  # fixed by held_out_seed
    torch.manual_seed(7)
    seen_any = np.zeros(K, bool)
    for _ in range(10):
        _, lat = t.sample_batch(8192, CPU, return_latents=True)
        k = lat["k"][lat["is_prog"]].numpy()
        assert not np.isin(k, t.held).any()
        seen_any[k] = True
    assert seen_any[t.seen].all()
    sets = t.make_eval_sets(0, 512)
    assert sets["heldout"]["is_prog"].all() and np.isin(sets["heldout"]["k"].numpy(), t.held).all()
    main = sets["main"]
    assert not np.isin(main["k"][main["is_prog"]].numpy(), t.held).any()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_sample_batch_on_mps_matches_cpu_semantics():
    dev = torch.device("mps")
    for t in (E5aRestartClock("restart"), E5aRestartClock("clock"), E5bNoiseCF("abducted"),
              E5bNoiseCF("fresh")):
        b = t.sample_batch(256, dev)
        assert b["tokens"].device.type == "mps" and b["tokens"].max() < t.vocab_size
        for h in t.heads:
            assert torch.allclose(b["targets"][h].sum(-1).cpu(), torch.ones(b["tokens"].shape))


# ============================================================ evaluate plumbing: oracle as model


class _OracleModel(nn.Module):
    """Emits log Bayes-oracle probabilities computed from the tokens alone, so `evaluate` must
    report ~0 KL to the own oracle (and the flag-blind answer on flag-ablated inputs)."""

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, cont=None):
        tok = tokens.cpu().numpy()
        B, L = tok.shape
        lg = lambda p: np.log(np.maximum(p, 1e-300))  # noqa: E731
        t = self.task
        if isinstance(t, E5aRestartClock):
            x = tok % P
            fl = tok >= P
            s = np.where(fl.any(1), fl.argmax(1), -1)
            mode = t.semantics if (s >= 0).any() else "l1"
            pr = recurrence_oracle(x, s, mode, t.pi)["pred"]
            out = np.zeros((B, L, P))
            out[:, :-1] = lg(pr)
            return {"nxt": torch.as_tensor(out)}
        T = t.T
        x = tok[:, :T] % P
        q = (tok[:, :T] >= 2 * P).argmax(1)
        v, y = tok[:, T + 1] - P, tok[:, T + 2 :]
        nxt = np.zeros((B, L, P))
        cf = np.zeros((B, L, P))
        nxt[:, : T - 1] = lg(noisy_factual_oracle(x, t.eps, t.pi)["pred"])
        cf[:, T + 1 : T + 1 + N_CF] = lg(cf_oracle(x, q, v, y, t.cf_labels, t.eps, t.pi))
        return {"nxt": torch.as_tensor(nxt), "cf": torch.as_tensor(cf)}


@pytest.mark.parametrize("task", [E5aRestartClock("restart"), E5aRestartClock("clock"),
                                  E5bNoiseCF("abducted"), E5bNoiseCF("fresh")],
                         ids=["restart", "clock", "abducted", "fresh"])
def test_evaluate_scores_the_oracle_as_perfect(task):
    sets = task.make_eval_sets(seed=123, n_episodes=300)
    m, preds = task.evaluate(_OracleModel(task), sets, CPU, return_preds=True)
    assert m["kl_own"] < 1e-6
    if isinstance(task, E5aRestartClock):
        assert m["abl_kl_l1"] < 1e-6 and m["abl_kl_own"] > 0.5
        assert m["kl_other"] > 0.5 and m["kl_l1"] > 0.5
        assert m["gc_prog"] == pytest.approx(1.0) and m["gc_tv_prog"] == pytest.approx(1.0)
        assert m["n_disc_prog"] > 0
    else:
        assert m["kl_y_own_heldout"] < 1e-6 and m["kl_fact_heldout"] < 1e-6
        assert m["gc_all"] == pytest.approx(1.0) and m["gc_tv_noise"] == pytest.approx(1.0)
    assert all(np.isfinite(v).all() for v in preds.values() if v.dtype.kind == "f")


def test_evaluate_runs_with_untrained_model():
    from witnesses.model import TinyGPT
    for task in (E5aRestartClock("clock"), E5bNoiseCF("fresh")):
        model = TinyGPT(task.vocab_size, task.max_len, {k: h.dim for k, h in task.heads.items()},
                        d_model=32, n_layers=1, n_heads=2)
        m = task.evaluate(model, task.make_eval_sets(0, 128), CPU)
        assert np.isfinite(m[task.headline]) and m[task.headline] > 0.01
