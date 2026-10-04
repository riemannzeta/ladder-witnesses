import numpy as np
import torch

from witnesses.tasks.e1_simpson import (ADJ, CRUDE, EST_TABLE, MARG, E1Simpson, cxy_index,
                                        functionals_posterior, functionals_true)


def test_estimand_counts_per_role_pair():
    for x in range(3):
        for y in range(3):
            if x == y:
                continue
            col = EST_TABLE[:, x, y]
            assert (col == ADJ).sum() == 1 and (col == CRUDE).sum() == 2 and (col == MARG).sum() == 3


def test_posterior_oracle_matches_monte_carlo():
    rng = np.random.default_rng(0)
    a = 0.3
    for _ in range(4):
        n = rng.integers(0, 12, size=(2, 2, 2))
        F = functionals_posterior(n, a)  # [3, 2]
        S = rng.dirichlet((n + a).ravel(), 300_000).reshape(-1, 2, 2, 2)
        F_mc = functionals_true(torch.as_tensor(S)).numpy().mean(0)
        assert np.abs(F - F_mc).max() < 3e-3, (F, F_mc)


def test_oracle_converges_to_truth_at_large_n():
    torch.manual_seed(1)
    t = E1Simpson(alpha=0.3, n_samples=20_000)
    ep = t._episodes(64, torch.device("cpu"))
    ep = {k: v.cpu() for k, v in ep.items()}
    orc = t.oracle(ep)["ans"][:, -1]
    err = np.abs(orc - ep["true"].numpy())
    assert np.median(err) < 0.01


def test_data_independent_of_ordering():
    torch.manual_seed(2)
    t = E1Simpson(alpha=0.3, n_samples=64)
    ep = t._episodes(60_000, torch.device("cpu"))
    freq = torch.nn.functional.one_hot(ep["cells"], 8).float().mean(1)  # [B, 8]
    means = torch.stack([freq[ep["perm"] == p].mean(0) for p in range(6)])
    assert (means - means.mean(0)).abs().max() < 0.01


def test_cxy_index_roundtrip():
    x = torch.tensor([0, 1, 2, 0]); y = torch.tensor([1, 2, 0, 2])
    idx = cxy_index(x, y)
    assert idx.shape == (4, 8) and all(sorted(r.tolist()) == list(range(8)) for r in idx)


def test_batch_shapes_and_masks():
    t = E1Simpson(alpha=0.3, n_samples=16)
    b = t.sample_batch(8, torch.device("cpu"))
    assert b["tokens"].shape == (8, 20) and b["tokens"].max() < t.vocab_size
    assert b["masks"]["ans"].sum(1).eq(17).all() and b["masks"]["nxt"].sum(1).eq(16).all()
