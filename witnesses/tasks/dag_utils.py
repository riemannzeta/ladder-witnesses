"""Small-DAG machinery for E2/E2c: enumeration, topology classes, Markov equivalence,
d-separation, cross-validation folds, BDeu prior, and exact (vectorized) causal queries.

Conventions
-----------
* 4 binary nodes. A DAG is a tuple of parent bitmasks `pa[i]` (bit j set iff j -> i).
* Each topology has a fixed *internal* representative whose edges only go from lower to higher
  index (so index order is a topological order). Episodes draw a topology, sample CPTs and data in
  internal order, then present nodes under a uniformly random label permutation `perm`
  (presented label of internal node i is perm[i]). The induced distribution over presented labeled
  DAGs is uniform over the topology's distinct labelings.
* CPT rows: node i with parents sorted ascending by INTERNAL index (p0 < p1) has row index
  j = v[p0] + 2 v[p1].  theta[i, j] = P(V_i = 1 | pa = j).
* BDeu with equivalent sample size 2: rows of a node with k parents ~ Beta(2^-k, 2^-k).
  (Score-equivalent: Markov-equivalent DAGs get identical marginal likelihoods.)
"""

from __future__ import annotations

import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

N_NODES = 4
MAX_INDEG = 2
PERMS4 = list(itertools.permutations(range(N_NODES)))
ASSIGN = np.array(list(itertools.product((0, 1), repeat=N_NODES)))[:, ::-1]  # [16, 4]; a[c, i] = bit i of c
# ASSIGN[c, i] == (c >> i) & 1


def parents_of(pa_mask: int) -> list[int]:
    return [j for j in range(N_NODES) if pa_mask >> j & 1]


def is_acyclic(pa: tuple[int, ...]) -> bool:
    indeg = [bin(p).count("1") for p in pa]
    children = [[i for i in range(N_NODES) if pa[i] >> j & 1] for j in range(N_NODES)]
    stack = [i for i in range(N_NODES) if indeg[i] == 0]
    seen = 0
    indeg = indeg[:]
    while stack:
        j = stack.pop()
        seen += 1
        for i in children[j]:
            indeg[i] -= 1
            if indeg[i] == 0:
                stack.append(i)
    return seen == N_NODES


@lru_cache(maxsize=None)
def labeled_dags() -> list[tuple[int, ...]]:
    out = []
    for pa in itertools.product(range(1 << N_NODES), repeat=N_NODES):
        if any(pa[i] >> i & 1 for i in range(N_NODES)):
            continue
        if any(bin(p).count("1") > MAX_INDEG for p in pa):
            continue
        if is_acyclic(pa):
            out.append(tuple(pa))
    return out


def relabel(pa: tuple[int, ...], perm: tuple[int, ...]) -> tuple[int, ...]:
    """Apply node relabeling i -> perm[i]."""
    new = [0] * N_NODES
    for i in range(N_NODES):
        m = 0
        for j in parents_of(pa[i]):
            m |= 1 << perm[j]
        new[perm[i]] = m
    return tuple(new)


def canonical(pa: tuple[int, ...]) -> tuple[int, ...]:
    return min(relabel(pa, p) for p in PERMS4)


def is_upper(pa: tuple[int, ...]) -> bool:
    return all(pa[i] < (1 << i) for i in range(N_NODES))  # parents have lower index


@lru_cache(maxsize=None)
def topologies() -> list[dict]:
    """The unlabeled topologies with a lower-to-higher internal representative each."""
    groups: dict[tuple, list] = {}
    for pa in labeled_dags():
        groups.setdefault(canonical(pa), []).append(pa)
    topos = []
    for key in sorted(groups, key=lambda k: (sum(bin(p).count("1") for p in k), k)):
        reps = sorted(p for p in groups[key] if is_upper(p))
        topos.append(dict(key=key, rep=reps[0], n_edges=sum(bin(p).count("1") for p in key),
                          n_labelings=len(groups[key])))
    for t_id, t in enumerate(topos):
        t["id"] = t_id
    return topos


def topology_id(pa: tuple[int, ...]) -> int:
    key = canonical(pa)
    for t in topologies():
        if t["key"] == key:
            return t["id"]
    raise KeyError(pa)


def skeleton(pa) -> frozenset:
    return frozenset(frozenset((i, j)) for i in range(N_NODES) for j in parents_of(pa[i]))


def v_structures(pa) -> frozenset:
    sk = skeleton(pa)
    out = set()
    for c in range(N_NODES):
        ps = parents_of(pa[c])
        for a, b in itertools.combinations(ps, 2):
            if frozenset((a, b)) not in sk:
                out.add((min(a, b), max(a, b), c))
    return frozenset(out)


def markov_key(pa) -> tuple:
    return (skeleton(pa), v_structures(pa))


def ancestors(pa, node: int, cut: int = 0) -> int:
    """Bitmask of proper ancestors of `node` after removing all edges INTO the nodes in `cut`."""
    res, frontier = 0, [node]
    while frontier:
        x = frontier.pop()
        if cut >> x & 1:
            continue
        for p in parents_of(pa[x]):
            if not res >> p & 1:
                res |= 1 << p
                frontier.append(p)
    return res


def d_separated(pa, a: int, b: int, given: int) -> bool:
    """Is node a d-separated from node b given the node set `given` (bitmask)? (moralization)"""
    keep = (1 << a) | (1 << b) | given
    anc = keep
    for x in range(N_NODES):
        if keep >> x & 1:
            anc |= ancestors(pa, x)
    adj = {i: set() for i in range(N_NODES) if anc >> i & 1}
    for c in adj:
        ps = [p for p in parents_of(pa[c]) if anc >> p & 1]
        for p in ps:
            adj[c].add(p)
            adj[p].add(c)
        for p, q in itertools.combinations(ps, 2):
            adj[p].add(q)
            adj[q].add(p)
    seen, stack = {a}, [a]
    while stack:
        x = stack.pop()
        if x == b:
            return False
        for y in adj[x]:
            if y not in seen and not given >> y & 1:
                seen.add(y)
                stack.append(y)
    return True


# ------------------------------------------------------------------ folds
FOLDS_PATH = Path(__file__).resolve().parents[2] / "configs" / "e2" / "folds.json"


def markov_partners(t_id: int) -> set[int]:
    """Other topologies that have a labeling Markov-equivalent to some labeling of t_id."""
    mine = {markov_key(pa) for pa in labeled_dags() if topology_id(pa) == t_id}
    out = set()
    for pa in labeled_dags():
        tid = topology_id(pa)
        if tid != t_id and markov_key(pa) in mine:
            out.add(tid)
    return out


def make_folds(n_folds: int = 5, seed: int = 20261003) -> list[list[int]]:
    """Stratified (by edge count) assignment of topologies to folds, then repaired so that every
    held-out topology with Markov partners keeps at least one partner in training."""
    rng = np.random.default_rng(seed)
    topos = topologies()
    by_edges: dict[int, list[int]] = {}
    for t in topos:
        by_edges.setdefault(t["n_edges"], []).append(t["id"])
    order = []
    for e in sorted(by_edges):
        ids = by_edges[e][:]
        rng.shuffle(ids)
        order += ids
    folds = [[] for _ in range(n_folds)]
    for k, tid in enumerate(order):
        folds[k % n_folds].append(tid)
    partners = {t["id"]: markov_partners(t["id"]) for t in topos}

    def bad(f):
        return [t for t in f if partners[t] and partners[t] <= set(f)]

    for _ in range(200):
        changed = False
        for fi, f in enumerate(folds):
            for t in bad(f):
                # swap t with a same-edge-count topology from another fold
                for fj, g in enumerate(folds):
                    if fj == fi:
                        continue
                    for u in g:
                        if topos[u]["n_edges"] != topos[t]["n_edges"]:
                            continue
                        f2 = [x for x in f if x != t] + [u]
                        g2 = [x for x in g if x != u] + [t]
                        if not bad(f2) and not bad(g2):
                            folds[fi], folds[fj] = sorted(f2), sorted(g2)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        if not changed:
            break
    return [sorted(f) for f in folds]


def load_folds() -> list[list[int]]:
    return json.loads(FOLDS_PATH.read_text())["folds"]


# ------------------------------------------------------------------ exact queries (numpy)
def row_index(pa_mask: int, values: np.ndarray) -> np.ndarray:
    """values [..., 4] -> CPT row index for the node with parent mask pa_mask."""
    ps = parents_of(pa_mask)
    j = np.zeros(values.shape[:-1], dtype=np.int64)
    for k, p in enumerate(ps):
        j += values[..., p] << k
    return j


def bdeu_ab(n_parents: int) -> float:
    return 2.0 ** (-n_parents)


def joint_table(pa, theta: np.ndarray, do_mask: int = 0, do_vals: np.ndarray | None = None):
    """theta [B, 4, 4] (internal indexing) -> [B, 16] joint (truncated if do_mask).

    do_vals [B, 4] gives the forced values for nodes in do_mask.
    """
    B = theta.shape[0]
    out = np.ones((B, 16))
    for i in range(N_NODES):
        vi = ASSIGN[:, i]  # [16]
        if do_mask >> i & 1:
            out *= (vi[None, :] == do_vals[:, i : i + 1])
            continue
        j = row_index(pa[i], ASSIGN)  # [16]
        p1 = theta[:, i, j]  # [B, 16]
        out *= np.where(vi[None, :] == 1, p1, 1 - p1)
    return out
