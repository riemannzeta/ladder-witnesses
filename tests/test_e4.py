import numpy as np
import torch

from witnesses.tasks.e4_identify import E4Identify, p1_conf_np, p1_true


def _cells(seed, B, N, conf, hid, alpha=0.5):
    torch.manual_seed(seed)
    t = E4Identify(alpha=alpha, n_samples=N, mc=4096)
    ep = t._episodes(B, torch.device("cpu"))
    ep["conf"][:] = conf; ep["hid"][:] = hid
    return t, ep


def _importance(cells_row, hid, conf, a, rng, M=1_500_000):
    P = rng.dirichlet(np.full(8, a), M).reshape(-1, 2, 2, 2)
    n = np.bincount(cells_row, minlength=8).reshape(2, 2, 2)
    if hid:
        logw = (np.log(P.sum(1)) * n.sum(0)).sum((1, 2))
    else:
        logw = (np.log(P) * n).sum((1, 2, 3))
    w = np.exp(logw - logw.max()); w /= w.sum()
    p1 = p1_conf_np(P) if conf else P[:, :, 1, 1].sum(1) / P[:, :, 1, :].sum((1, 2))
    m = (w * p1).sum(); return m, np.sqrt((w * (p1 - m) ** 2).sum())


def test_exact_posteriors_match_importance_sampling():
    rng = np.random.default_rng(0)
    for conf in (True, False):
        for hid in (True, False):
            t, ep = _cells(1, 3, 24, conf, hid)
            cells = ep["cells"].numpy()
            hist, sd = t._posterior_hist(cells, hid, conf, 24, rng)
            centers = 0.5 * (t.edges[1:] + t.edges[:-1])
            for b in range(3):
                m_is, s_is = _importance(cells[b, :24], hid, conf, 0.5, rng)
                m_h = (hist[b] * centers).sum()
                assert abs(m_h - m_is) < 0.01, (conf, hid, m_h, m_is)
                assert abs(sd[b] - s_is) < 0.01, (conf, hid, sd[b], s_is)


def test_hidden_confounded_posterior_does_not_contract():
    rng = np.random.default_rng(1)
    t, ep = _cells(2, 64, 256, True, True)
    cells = ep["cells"].numpy()
    _, sd32 = t._posterior_hist(cells, True, True, 32, rng)
    _, sd256 = t._posterior_hist(cells, True, True, 256, rng)
    t2, ep2 = _cells(2, 64, 256, False, True)
    _, s32 = t2._posterior_hist(ep2["cells"].numpy(), True, False, 32, rng)
    _, s256 = t2._posterior_hist(ep2["cells"].numpy(), True, False, 256, rng)
    assert np.median(sd256) / np.median(sd32) > 0.65      # plateau (non-identified)
    assert np.median(s256) / np.median(s32) < 0.45        # ~ N^-1/2 (identified, same tokens)


def test_true_p1_within_natural_bounds():
    torch.manual_seed(3)
    t = E4Identify()
    th = torch.distributions.Dirichlet(torch.full((8,), 0.5)).sample((5000,))
    p1 = p1_true(th, torch.ones(5000, dtype=torch.bool)).numpy()
    tt = th.view(-1, 2, 2, 2).numpy()
    p11 = tt[:, :, 1, 1].sum(1); p0 = tt[:, :, 0, :].sum((1, 2))
    assert ((p1 >= p11 - 1e-6) & (p1 <= p11 + p0 + 1e-6)).all()


def test_hid_tokens_identical_for_conf_and_med():
    t, ep = _cells(4, 16, 32, True, True)
    tok_c = t._tokens(ep)
    ep["conf"][:] = False
    tok_m = t._tokens(ep)
    assert (tok_c[:, 3:] == tok_m[:, 3:]).all() and (tok_c[:, 2] != tok_m[:, 2]).all()
