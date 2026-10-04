"""E6 items: verbal witness pairs posed to Claude models.

* V1 (L2, Simpson pair). theta ~ Dirichlet(alpha 1_8) over binary (L0, L1, L2), N samples, shown as
  a count table. Roles (treatment X, outcome Y, covariate C) are random labels. `stated` items add
  one sentence giving a complete causal ordering; `blind` items say nothing about causal
  structure. Each context is shown under one adj, one crude and one marg ordering (or all six)
  plus blind, so the split of the answers across orderings can be compared with the oracle's.
  Oracle: E1's posterior-mean functionals (functionals_posterior) at the ordering's estimand;
  blind reference: the ordering mixture 1/6 adj + 2/6 crude + 3/6 marg (honest answer: not
  determinable). Every `reversal_every`-th context is rejection-sampled to be a Simpson reversal
  (adjusted and crude ATEs of opposite sign, both |ATE| > min_abs_ate).
* V2 (L3, residual twin). Y = w X + noise with noise sigma U1 for X > 0 and
  sigma (rho U1 + sqrt(1 - rho^2) U2) for X <= 0. One unit observed at (x*, y*); query Y had X been
  x'. `stated` gives rho, `unstated` says rho is unknown in [-1, 1]. Oracle (known w): same side
  y* + w (x' - x*); crossing w x' + rho (y* - w x*); unstated crossing w x' (E[rho] = 0).
* V3 (identifiability). Z -> X, Z -> Y, X -> Y; theta ~ Dirichlet(alpha 1_8), N samples.
  `measured`: (z, x, y) table, oracle = adjusted posterior mean of P(Y=1 | do(X=1)).
  `unmeasured`: (x, y) table only; not identifiable; natural bounds
  [P(X=1, Y=1), P(X=1, Y=1) + P(X=0)] from the empirical proportions.

Witness pairs share their data, names and wording: only the sentence carrying the higher-rung
information (and, for V3, the Z column) differs. Context i of probe P draws from its own generator
seeded with (seed, P, i), so a pilot of K contexts is exactly the first K contexts of a full run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from witnesses.llm_probe.prompts import WORDING_SLOTS, render_prompt
from witnesses.tasks.e1_simpson import (ADJ, CRUDE, EST_NAMES, EST_TABLE, FUNC_P, MARG, PERMS,
                                        cxy_index, functionals_posterior, functionals_true)

PROBES = ("V1", "V2", "V3")
CONDITIONS = {"V1": ("stated", "blind"), "V2": ("stated", "unstated"),
              "V3": ("measured", "unmeasured")}
_PROBE_CODE = {"V1": 1, "V2": 2, "V3": 3}
MAX_TRIES = 100_000

# ---------------------------------------------------------------- nonsense names
# Grammar (CV){2,3}C, at most 8 letters, with a final consonant in {k, x, z, v}: pronounceable,
# and it avoids the English-like endings (-er, -en, -al, -e, -um, -gram, ...) that make syllable
# names collide with real words. _REAL_WORDS is the complete intersection of this grammar with the
# web2 word list (/usr/share/dict/words, 235k words), so no generated name is a dictionary word.

_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "fr",
           "gr", "kr", "pr", "tr", "bl", "fl", "gl", "kl", "pl", "sk", "st")
_VOWELS = ("a", "e", "i", "o", "u")
_CODAS = ("k", "x", "z", "v")
_MAX_NAME_LEN = 8
_REAL_WORDS = frozenset("""
balak banak batak batik befriz bemix benelux biflex borak borax bulak deflex demodex derek desex
detax diplex donax duplex durax felix filix firebox frederik galax galik genetrix genuflex
gerenuk gudok gulix kafiz kamik kapok katik kazak kelek kinepox kodak komatik konak kulak ladik
lanaz larix latax latex limax makluk makuk malax malik marek marok mastax matax matrix megrez merak
meretrix mirak monitrix mukluk murex nabak namaz natrix ninox novatrix nunatak panak panax paradox
pararek patrix pinax podex pomak prefix premix prolix promotrix proplex protax pulex radek radix
rafik ramex rastik rebox redox redux refix reflex reflux relatrix relax remex remix resex retax
retroflex retroflux rotanev rumex sabik salix sapek saruk semiflex senatrix silex sorex sotik
storax stupex sumak tabriz talak taluk tamarix tanak tarok tatinek televox terek testatrix tetrix
topaz triplex tubifex tubik tugrik tupek tupik tutrix varix veniplex verek vibex vibix visitrix
vivax vivek volapuk zelatrix zirak
""".split())


def nonsense_name(rng: np.random.Generator) -> str:
    while True:
        k = 2 if rng.random() < 0.6 else 3
        s = "".join(_ONSETS[rng.integers(len(_ONSETS))] + _VOWELS[rng.integers(len(_VOWELS))]
                    for _ in range(k))
        s += _CODAS[rng.integers(len(_CODAS))]
        if len(s) <= _MAX_NAME_LEN and s not in _REAL_WORDS:
            return s.capitalize()


def nonsense_names(rng: np.random.Generator, k: int) -> list[str]:
    """k distinct names, none a prefix of another."""
    names: list[str] = []
    while len(names) < k:
        s = nonsense_name(rng)
        if all(not (s.startswith(o) or o.startswith(s)) for o in names):
            names.append(s)
    return names


# ---------------------------------------------------------------- helpers


def context_rng(seed: int, probe: str, i: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), _PROBE_CODE[probe], int(i)])


def _wording(rng: np.random.Generator, probe: str) -> dict[str, int]:
    return {slot: int(rng.integers(n)) for slot, n in WORDING_SLOTS[probe].items()}


_CXY = {(x, y): cxy_index(torch.tensor([x]), torch.tensor([y]))[0].numpy()
        for x in range(3) for y in range(3) if x != y}


def _functionals_dict(F: np.ndarray) -> dict[str, list[float]]:
    """[3, 2] (adj, crude, marg) x (x=0, x=1) -> {"adj": [p0, p1], ...}."""
    return {name: [float(F[e, 0]), float(F[e, 1])] for e, name in enumerate(EST_NAMES)}


# ---------------------------------------------------------------- V1


def v1_context(seed: int, i: int, cfg: dict) -> list[dict]:
    """All items (stated orderings + blind) of V1 context i."""
    rng = context_rng(seed, "V1", i)
    alpha, N = float(cfg["alpha"]), int(cfg["n_samples"])
    want_rev = i % int(cfg["reversal_every"]) == 0
    for _ in range(MAX_TRIES):
        theta = rng.dirichlet(np.full(8, alpha))
        counts = rng.multinomial(N, theta)
        x = int(rng.integers(3))
        y = int((x + rng.integers(1, 3)) % 3)
        n_cxy = counts[_CXY[x, y]].reshape(2, 2, 2)
        if n_cxy.sum(-1).min() < int(cfg["min_stratum_count"]):
            continue  # every (covariate, treatment) stratum must be populated
        F = functionals_posterior(n_cxy, alpha)  # [3, 2]
        ate_adj = float(F[ADJ, 1] - F[ADJ, 0])
        ate_cru = float(F[CRUDE, 1] - F[CRUDE, 0])
        rev = (np.sign(ate_adj) != np.sign(ate_cru) and abs(ate_adj) > cfg["min_abs_ate"]
               and abs(ate_cru) > cfg["min_abs_ate"])
        if want_rev and not rev:
            continue
        break
    else:
        raise RuntimeError(f"V1 context {i}: rejection sampling failed")
    c = 3 - x - y
    names = nonsense_names(rng, 3)
    wording = _wording(rng, "V1")
    if cfg.get("all_orderings", False):
        perms = list(range(6))
    else:
        perms = [int(rng.choice([p for p in range(6) if EST_TABLE[p, x, y] == e]))
                 for e in (ADJ, CRUDE, MARG)]
    with np.errstate(invalid="ignore", divide="ignore"):  # only if min_stratum_count is 0
        F_plug = functionals_posterior(n_cxy, 0.0)
    th_cxy = torch.as_tensor(theta[_CXY[x, y]].reshape(1, 2, 2, 2))
    F_true = functionals_true(th_cxy)[0].numpy()
    mix = np.tensordot(np.array(FUNC_P), F, axes=(0, 0))  # [2]
    common = {
        "probe": "V1", "seed": int(seed), "context_id": i,
        "kind": "reversal" if want_rev else "random",
        "names": names,
        "roles": {"treatment": x, "outcome": y, "covariate": c},
        "treatment": names[x], "outcome": names[y], "covariate": names[c],
        "n_samples": N, "alpha": alpha,
        "theta": theta.tolist(), "counts": counts.tolist(), "counts_cxy": n_cxy.tolist(),
        "functionals": _functionals_dict(F),
        "functionals_plugin": _functionals_dict(F_plug),
        "functionals_true": _functionals_dict(F_true),
        "blind_mixture": [float(mix[0]), float(mix[1])],
        "ate": {"adj": ate_adj, "crude": ate_cru},
        "reversal": bool(rev),
        "wording": wording,
    }
    items = []
    for p in perms:
        e = int(EST_TABLE[p, x, y])
        items.append({
            **common, "condition": "stated", "item_id": f"c{i:03d}-o{p}",
            "ordering": list(PERMS[p]), "ordering_index": p,
            "ordering_names": [names[k] for k in PERMS[p]],
            "estimand": EST_NAMES[e],
            "oracle": {"p_do_x0": float(F[e, 0]), "p_do_x1": float(F[e, 1]), "determinable": True},
        })
    items.append({
        **common, "condition": "blind", "item_id": f"c{i:03d}",
        "ordering": None, "ordering_index": None, "ordering_names": None, "estimand": None,
        # Bayes reference under a uniform prior over the six orderings; honest answer: not
        # determinable without causal assumptions
        "oracle": {"p_do_x0": float(mix[0]), "p_do_x1": float(mix[1]), "determinable": False},
    })
    for it in items:
        it["prompt"] = render_prompt(it)
    return items


# ---------------------------------------------------------------- V2


def v2_oracle(w: float, x_star: float, y_star: float, x_prime: float,
              rho: float | None) -> float:
    """E[Y_{x'} | X = x*, Y = y*] for known w; rho=None means unknown rho ~ U(-1, 1)."""
    if (x_star > 0) == (x_prime > 0):  # same regime: the same noise term is reused
        return y_star + w * (x_prime - x_star)
    resid = y_star - w * x_star
    return w * x_prime + (0.0 if rho is None else rho * resid)


def v2_context(seed: int, i: int, cfg: dict) -> list[dict]:
    rng = context_rng(seed, "V2", i)
    every = int(cfg["same_side_every"])
    same_side = i % every == every - 1
    j = i // every if same_side else i - i // every  # index among items of the same type
    rhos = [float(r) for r in cfg["rho_values"]]
    rho = rhos[j % len(rhos)]
    sigma = float(cfg["sigma"])
    w = 0.0
    while abs(w) < float(cfg["w_min_abs"]):
        w = round(float(rng.uniform(-cfg["w_max"], cfg["w_max"])), 2)
    lo, hi = float(cfg["x_min_abs"]), float(cfg["x_max_abs"])
    s = 1.0 if rng.random() < 0.5 else -1.0
    x_star = s * round(float(rng.uniform(lo, hi)), 2)
    while True:
        x_prime = (s if same_side else -s) * round(float(rng.uniform(lo, hi)), 2)
        if x_prime != x_star:
            break
    u1, u2 = rng.standard_normal(2)

    def noise(xv: float) -> float:
        if xv > 0:
            return sigma * u1
        return sigma * (rho * u1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * u2)

    y_star = round(w * x_star + noise(x_star), 2) + 0.0
    names = nonsense_names(rng, 2)
    wording = _wording(rng, "V2")
    common = {
        "probe": "V2", "seed": int(seed), "context_id": i, "item_id": f"c{i:03d}",
        "names": names, "x_name": names[0], "y_name": names[1],
        "sigma": sigma, "w": w, "x_star": x_star, "y_star": y_star, "x_prime": x_prime,
        "crossing": not same_side,
        "rho_generating": rho,
        "residual": y_star - w * x_star,
        "l2_answer": w * x_prime,  # E[Y | do(X = x')]: the rung-2-only answer
        "additive_answer": y_star + w * (x_prime - x_star),  # reuse the residual (rho = 1)
        "y_cf_realized": w * x_prime + noise(x_prime),  # this unit's actual counterfactual
        "u": [float(u1), float(u2)],
        "wording": wording,
    }
    stated = {**common, "condition": "stated", "rho": rho,
              "oracle": {"counterfactual_y": v2_oracle(w, x_star, y_star, x_prime, rho),
                         "determinable": True}}
    unstated = {**common, "condition": "unstated", "rho": None,
                "oracle": {"counterfactual_y": v2_oracle(w, x_star, y_star, x_prime, None),
                           "determinable": bool(same_side)}}
    items = [stated, unstated]
    for it in items:
        it["prompt"] = render_prompt(it)
    return items


# ---------------------------------------------------------------- V3


def v3_context(seed: int, i: int, cfg: dict) -> list[dict]:
    rng = context_rng(seed, "V3", i)
    alpha, N = float(cfg["alpha"]), int(cfg["n_samples"])
    for _ in range(MAX_TRIES):
        theta = rng.dirichlet(np.full(8, alpha))
        counts = rng.multinomial(N, theta)
        n_zxy = counts.reshape(2, 2, 2)  # cell index 4 z + 2 x + y
        if n_zxy[:, 1, :].sum(-1).min() >= int(cfg["min_stratum_count"]):
            break  # both z strata contain treated units, so the adjusted estimate exists
    else:
        raise RuntimeError(f"V3 context {i}: rejection sampling failed")
    F = functionals_posterior(n_zxy, alpha)  # z plays the covariate role c
    with np.errstate(invalid="ignore", divide="ignore"):  # an empty (z, x=0) stratum is unused
        F_plug = functionals_posterior(n_zxy, 0.0)
    F_true = functionals_true(torch.as_tensor(theta.reshape(1, 2, 2, 2)))[0].numpy()
    n_xy = n_zxy.sum(0)
    lo = n_xy[1, 1] / N
    hi = (n_xy[1, 1] + n_xy[0].sum()) / N
    names = nonsense_names(rng, 3)
    wording = _wording(rng, "V3")
    common = {
        "probe": "V3", "seed": int(seed), "context_id": i, "item_id": f"c{i:03d}",
        "names": names, "z_name": names[0], "x_name": names[1], "y_name": names[2],
        "n_samples": N, "alpha": alpha, "theta": theta.tolist(),
        "counts_zxy": n_zxy.tolist(), "counts_xy": n_xy.tolist(),
        "adjusted_posterior": float(F[ADJ, 1]), "adjusted_plugin": float(F_plug[ADJ, 1]),
        "adjusted_true": float(F_true[ADJ, 1]),
        "crude": float(n_xy[1, 1] / n_xy[1].sum()),
        "natural_bounds": [float(lo), float(hi)],
        "wording": wording,
    }
    measured = {**common, "condition": "measured",
                "oracle": {"identifiable": True, "point_estimate": float(F[ADJ, 1]),
                           "lower_bound": None, "upper_bound": None}}
    unmeasured = {**common, "condition": "unmeasured",
                  "oracle": {"identifiable": False, "point_estimate": None,
                             "lower_bound": float(lo), "upper_bound": float(hi)}}
    items = [measured, unmeasured]
    for it in items:
        it["prompt"] = render_prompt(it)
    return items


# ---------------------------------------------------------------- all items / IO

_GEN = {"V1": v1_context, "V2": v2_context, "V3": v3_context}


def generate_items(cfg: dict, n_per_condition: int, probes=PROBES) -> dict[str, list[dict]]:
    """{probe: items}, ordered by condition then context."""
    seed = int(cfg["seed"])
    out = {}
    for probe in probes:
        pcfg = cfg[probe.lower()]
        items = [it for i in range(n_per_condition) for it in _GEN[probe](seed, i, pcfg)]
        order = {c: k for k, c in enumerate(CONDITIONS[probe])}
        out[probe] = sorted(items, key=lambda it: (order[it["condition"]], it["context_id"],
                                                   it["item_id"]))
    return out


def item_key(it: dict) -> tuple[str, str, str]:
    return it["probe"], it["condition"], it["item_id"]


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def write_items(items: dict[str, list[dict]], items_dir: Path, meta: dict) -> dict:
    items_dir = Path(items_dir)
    items_dir.mkdir(parents=True, exist_ok=True)
    manifest = {**meta, "files": {}}
    for probe, its in items.items():
        text = "".join(_dumps(it) + "\n" for it in its)
        (items_dir / f"{probe}.jsonl").write_text(text, encoding="utf-8")
        counts = {c: sum(it["condition"] == c for it in its) for c in CONDITIONS[probe]}
        manifest["files"][f"{probe}.jsonl"] = {
            "sha256": hashlib.sha256(text.encode()).hexdigest(), "n_items": len(its),
            "n_by_condition": counts}
    (items_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_items(items_dir: Path) -> dict[tuple[str, str, str], dict]:
    out = {}
    for probe in PROBES:
        f = Path(items_dir) / f"{probe}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                it = json.loads(line)
                out[item_key(it)] = it
    return out
