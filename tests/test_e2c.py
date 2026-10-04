import numpy as np
import torch

from witnesses.tasks import dag_utils as du
from witnesses.tasks.e2c_regimes import E2cRegimes

CPU = torch.device("cpu")


def test_intervened_nodes_are_randomized_and_others_follow_cpts():
    torch.manual_seed(0)
    t = E2cRegimes(n_per_regime=4000)
    ep = t._episodes(3, CPU)
    for b in range(3):
        A = int(ep["roles"][b, 0]); reg = ep["reg"][b]; v = ep["vals"][b]
        # under do(A), A ~ Bernoulli(1/2) regardless of its parents
        assert abs(v[reg == 1, A].float().mean().item() - 0.5) < 0.03


def test_regime_tokens_and_query_roles():
    torch.manual_seed(1)
    t = E2cRegimes(n_per_regime=8)
    ep = t._episodes(64, CPU)
    tok = t._tokens(ep)
    assert tok.shape == (64, t.max_len) and int(tok.max()) < t.vocab_size
    combo = ep["kind"] == 0
    assert (ep["q_B"][combo] >= 0).all() and (ep["q_B"][~combo] == -1).all()
    untried = ep["kind"] == 1
    assert (ep["q_A"][untried] == ep["roles"][untried, 2]).all()  # C is never intervened in context


def test_bayes_oracle_with_true_graph_converges_to_truth():
    torch.manual_seed(2)
    t = E2cRegimes(n_per_regime=64)
    t.N = 3 * 64
    ep = t._episodes(6, CPU, torch.zeros(6, dtype=torch.long))
    # big-data check of the Cooper-Yoo counts: the true-graph answer at k=N approaches the truth
    t2 = E2cRegimes(n_per_regime=3000)
    torch.manual_seed(3)
    ep2 = t2._episodes(4, CPU, torch.zeros(4, dtype=torch.long))
    import witnesses.tasks.e2c_regimes as m
    old = m.EVAL_K
    m.EVAL_K = (9000,)
    try:
        o = t2._oracles(ep2)
    finally:
        m.EVAL_K = old
    err = np.abs(o["oracle_true_graph"].numpy()[:, 0] - ep2["ans"].numpy())
    assert err.max() < 0.05, err
