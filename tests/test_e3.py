import numpy as np
import torch

from witnesses.tasks.e3_twin import E3Twin


def _ep(n, rho=None, seed=0, K=32):
    torch.manual_seed(seed)
    t = E3Twin(n_pairs=K)
    r = None if rho is None else torch.full((n,), float(rho))
    return t, {k: v.cpu() for k, v in t._episodes(n, torch.device("cpu"), r).items()}


def test_observational_distribution_independent_of_rho():
    # pooled residuals y - w x on the x <= 0 branch are N(0, sigma^2) for every rho
    for rho in (-1.0, 0.0, 0.7):
        _, ep = _ep(4000, rho, seed=1)
        res = (ep["y"] - ep["w"][:, None] * ep["x"])[ep["x"] <= 0].numpy()
        assert abs(res.std() - 0.5) < 0.01 and abs(res.mean()) < 0.01


def test_oracle_matches_monte_carlo_over_w_posterior():
    t, ep = _ep(64, seed=2)
    orc = t.oracle(ep)
    rng = np.random.default_rng(0)
    x, y = ep["x"].numpy(), ep["y"].numpy()
    xs, ys, xp = ep["xs"].numpy(), ep["ys"].numpy(), ep["xp"].numpy()
    r = np.where(ep["cross"].numpy(), ep["rho"].numpy(), 1.0)
    k = 5
    for b in range(5):
        X = np.concatenate([x[b, :k], [xs[b]]]); Y = np.concatenate([y[b, :k], [ys[b]]])
        prec = 1 + (X**2).sum() / 0.25; mean = (X * Y).sum() / 0.25 / prec
        w = rng.normal(mean, prec**-0.5, 400_000)
        mc = (w * xp[b] + r[b] * (ys[b] - w * xs[b])).mean()
        assert abs(mc - orc["cf"][b, k]) < 5e-3


def test_blind_crossing_answer_equals_l2_answer():
    t, ep = _ep(256, seed=3)
    orc = t.oracle(ep)
    cross = ep["cross"].numpy()
    assert np.allclose(orc["cf_blind"][cross], orc["l2"][cross])


def test_rao_blackwell_target_is_unbiased():
    # E[cf_target | w, x*, y*] equals the realized-counterfactual expectation over U2
    torch.manual_seed(4)
    t = E3Twin()
    n = 200_000
    w, xs, xp = torch.full((n,), 0.8), torch.full((n,), 1.2), torch.full((n,), -0.9)
    rho = torch.full((n,), -0.6)
    u1 = torch.full((n,), 0.37); u2 = torch.randn(n)
    ys = w * xs + t._noise(xs, rho, u1, u2)
    y_cf = w * xp + t._noise(xp, rho, u1, u2)
    rb = w * xp + rho * (ys - w * xs)
    assert abs(float(y_cf.mean() - rb.mean())) < 5e-3


def test_training_rho_excludes_heldout_band():
    t = E3Twin()
    r = t._rho_train(50_000, torch.device("cpu"))
    assert (r.abs() >= 0.3 - 1e-6).all() and abs(float(r.mean())) < 0.02
