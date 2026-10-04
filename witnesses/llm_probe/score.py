"""E6 scoring: per model and probe/condition, with bootstrap 95% CIs over items (V1: contexts).

    python -m witnesses.llm_probe.score --config configs/e6/probe.yaml [--out-dir DIR]
        [--models M ...]

Reads <out_dir>/items/*.jsonl and <out_dir>/responses/<model>.jsonl and writes
<out_dir>/scores.json and <out_dir>/scores.md. Only answers in category "ok" enter the metrics;
refusals, parse failures, errored/expired/canceled requests and missing responses are counted as
their own categories.

* V1 stated: MAE against the oracle (overall and per estimand); the split readout as in E1's
  witness block (per context, model and oracle answers are centred across the orderings shown,
  for both x; contexts with oracle spread < split_min_spread are dropped; slope and R^2 of model on
  oracle, ideal 1 and 1, ordering-blind 0); Simpson-reversal sign agreement of the adj- and
  crude-ordering ATEs on reversal contexts. V1 blind: share answering determinable = false, MAE
  against the ordering mixture, and which functional each answer is nearest to.
* V2: for crossing items, slope of (answer - w x') on (y* - w x*), per stated rho (ideal: rho) and
  for unstated (ideal 0; also what an L2-only answer gives); MAE against the oracle and the
  determinable rates, separately for crossing and same-side items.
* V3: identifiability accuracy per condition; measured: MAE of the point estimate; unmeasured:
  bounds accuracy (both ends within bound_tol of the natural bounds) and the rate of giving a
  point estimate anyway.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from witnesses.llm_probe.items import CONDITIONS, PROBES, item_key, read_items
from witnesses.tasks.base import fit_slope
from witnesses.tasks.e1_simpson import EST_NAMES

CATEGORIES = ("ok", "refusal", "parse_failure", "errored", "expired", "canceled", "missing")
NAN = float("nan")


class Boot:
    """Point estimate + percentile bootstrap CI, resampling units (items or contexts)."""

    def __init__(self, n_boot: int = 2000, seed: int = 0):
        self.n_boot, self.seed = int(n_boot), int(seed)

    def __call__(self, stat, n: int, target: float | None = None) -> dict:
        value = stat(np.arange(n)) if n > 0 else NAN
        lo = hi = NAN
        if n >= 2:
            rng = np.random.default_rng(self.seed)
            vals = np.array([stat(rng.integers(0, n, n)) for _ in range(self.n_boot)], float)
            vals = vals[np.isfinite(vals)]
            if vals.size >= self.n_boot // 2:
                lo, hi = (float(v) for v in np.percentile(vals, [2.5, 97.5]))
        out = {"value": float(value), "ci": [lo, hi], "n": int(n)}
        if target is not None:
            out["target"] = float(target)
        return out


def mean_stat(values):
    arr = np.asarray(values, float)
    return lambda idx: float(arr[idx].mean()) if len(idx) else NAN


def _fit(x: np.ndarray, y: np.ndarray, which: int) -> float:
    """fit_slope, but NaN when the regressor is constant up to rounding: a bootstrap resample that
    repeats one item can have var(x) ~ 1e-33 instead of exactly 0, and polyfit then fits noise."""
    if x.size < 3 or np.ptp(x) <= 1e-9 * max(1.0, float(np.abs(x).max())):
        return NAN
    return fit_slope(x, y)[which]


def pair_slope(x, y, which: int = 0):
    """Slope (which=0) or R^2 (which=1) of y on x over the resampled items."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return lambda idx: _fit(x[idx], y[idx], which)


def context_slope(xs: list[np.ndarray], ys: list[np.ndarray], which: int = 0):
    """As pair_slope, but each unit (context) contributes a block of points."""
    def f(idx):
        if len(idx) == 0:
            return NAN
        return _fit(np.concatenate([xs[i] for i in idx]), np.concatenate([ys[i] for i in idx]),
                    which)
    return f


def _p(a: dict) -> np.ndarray:
    return np.array([a["p_do_x0"], a["p_do_x1"]], float)


# ---------------------------------------------------------------- per probe


def score_v1(items: list[dict], ans: dict, boot: Boot, min_spread: float) -> dict:
    stated = [it for it in items if it["condition"] == "stated"]
    blind = [it for it in items if it["condition"] == "blind"]
    pairs = [(it, ans[item_key(it)]) for it in stated if item_key(it) in ans]
    s: dict = {}
    err = [float(np.abs(_p(a) - _p(it["oracle"])).mean()) for it, a in pairs]
    s["mae"] = boot(mean_stat(err), len(err), 0.0)
    for e in EST_NAMES:
        sub = [float(np.abs(_p(a) - _p(it["oracle"])).mean()) for it, a in pairs
               if it["estimand"] == e]
        s[f"mae_{e}"] = boot(mean_stat(sub), len(sub), 0.0)
    det = [float(a["determinable"] is True) for _, a in pairs]
    s["determinable_true"] = boot(mean_stat(det), len(det), 1.0)
    s["n_out_of_range"] = int(sum(bool(((_p(a) < 0) | (_p(a) > 1)).any()) for _, a in pairs))

    by_ctx: dict[int, list[dict]] = defaultdict(list)
    for it in stated:
        by_ctx[it["context_id"]].append(it)
    do_l, dm_l, incomplete, flat = [], [], 0, 0
    agree_adj, agree_cru = [], []
    for _, its in sorted(by_ctx.items()):
        if all(item_key(it) in ans for it in its):
            M = np.stack([_p(ans[item_key(it)]) for it in its])  # [orderings, 2]
            R = np.stack([_p(it["oracle"]) for it in its])
            if (R.max(0) - R.min(0)).max() >= min_spread:
                do_l.append((R - R.mean(0)).ravel())
                dm_l.append((M - M.mean(0)).ravel())
            else:
                flat += 1
        else:
            incomplete += 1
        if its[0]["reversal"]:
            adj = [it for it in its if it["estimand"] == "adj" and item_key(it) in ans]
            cru = [it for it in its if it["estimand"] == "crude" and item_key(it) in ans]
            if adj and cru:
                ma, mc = _p(ans[item_key(adj[0])]), _p(ans[item_key(cru[0])])
                agree_adj.append(float(np.sign(ma[1] - ma[0]) == np.sign(its[0]["ate"]["adj"])))
                agree_cru.append(float(np.sign(mc[1] - mc[0]) == np.sign(its[0]["ate"]["crude"])))
    s["split_slope"] = boot(context_slope(do_l, dm_l, 0), len(do_l), 1.0)
    s["split_r2"] = boot(context_slope(do_l, dm_l, 1), len(do_l), 1.0)
    s["split_contexts_incomplete"] = incomplete
    s["split_contexts_flat"] = flat
    both = [a * c for a, c in zip(agree_adj, agree_cru)]
    s["reversal_agree_adj"] = boot(mean_stat(agree_adj), len(agree_adj), 1.0)
    s["reversal_agree_crude"] = boot(mean_stat(agree_cru), len(agree_cru), 1.0)
    s["reversal_agree_both"] = boot(mean_stat(both), len(both), 1.0)

    b: dict = {}
    bpairs = [(it, ans[item_key(it)]) for it in blind if item_key(it) in ans]
    n = len(bpairs)
    b["determinable_false"] = boot(mean_stat([float(a["determinable"] is False)
                                              for _, a in bpairs]), n, 1.0)
    b["mae_vs_mixture"] = boot(mean_stat([float(np.abs(_p(a) - np.array(it["blind_mixture"])).mean())
                                          for it, a in bpairs]), n, 0.0)
    nearest = [EST_NAMES[int(np.argmin([np.abs(_p(a) - np.array(it["functionals"][e])).mean()
                                        for e in EST_NAMES]))] for it, a in bpairs]
    for e in EST_NAMES:
        b[f"nearest_{e}"] = boot(mean_stat([float(x == e) for x in nearest]), n)
    return {"stated": s, "blind": b}


def score_v2(items: list[dict], ans: dict, boot: Boot, rho_values) -> dict:
    out = {}
    for cond in CONDITIONS["V2"]:
        pairs = [(it, ans[item_key(it)]) for it in items
                 if it["condition"] == cond and item_key(it) in ans]
        s: dict = {}
        for kind, crossing in (("crossing", True), ("same_side", False)):
            sub = [(it, a) for it, a in pairs if it["crossing"] == crossing]
            err = [abs(a["counterfactual_y"] - it["oracle"]["counterfactual_y"]) for it, a in sub]
            s[f"mae_{kind}"] = boot(mean_stat(err), len(err), 0.0)
            det = [float(a["determinable"] is True) for _, a in sub]
            ideal = 1.0 if (cond == "stated" or not crossing) else 0.0
            s[f"determinable_true_{kind}"] = boot(mean_stat(det), len(det), ideal)
        cross = [(it, a) for it, a in pairs if it["crossing"]]
        x = np.array([it["residual"] for it, _ in cross], float)  # y* - w x*
        y = np.array([a["counterfactual_y"] - it["l2_answer"] for it, a in cross], float)
        if cond == "stated":
            rho = np.array([it["rho"] for it, _ in cross], float)
            for r in rho_values:
                m = np.isclose(rho, float(r))
                s[f"slope_rho={float(r):g}"] = boot(pair_slope(x[m], y[m]), int(m.sum()), float(r))
            s["rho_tracking_slope"] = boot(pair_slope(rho * x, y), len(x), 1.0)
        else:
            s["slope_crossing"] = boot(pair_slope(x, y), len(x), 0.0)
        out[cond] = s
    return out


def score_v3(items: list[dict], ans: dict, boot: Boot, tol: float) -> dict:
    meas = [(it, ans[item_key(it)]) for it in items
            if it["condition"] == "measured" and item_key(it) in ans]
    unm = [(it, ans[item_key(it)]) for it in items
           if it["condition"] == "unmeasured" and item_key(it) in ans]
    m: dict = {}
    m["identifiable_acc"] = boot(mean_stat([float(a["identifiable"] is True) for _, a in meas]),
                                 len(meas), 1.0)
    pts = [abs(a["point_estimate"] - it["oracle"]["point_estimate"]) for it, a in meas
           if a["point_estimate"] is not None]
    m["point_mae"] = boot(mean_stat(pts), len(pts), 0.0)
    m["point_missing"] = boot(mean_stat([float(a["point_estimate"] is None) for _, a in meas]),
                              len(meas), 0.0)
    u: dict = {}
    u["identifiable_acc"] = boot(mean_stat([float(a["identifiable"] is False) for _, a in unm]),
                                 len(unm), 1.0)

    def bounds_ok(it, a):
        lo, hi = a["lower_bound"], a["upper_bound"]
        lo_t, hi_t = it["oracle"]["lower_bound"], it["oracle"]["upper_bound"]
        return float(lo is not None and hi is not None and abs(lo - lo_t) < tol
                     and abs(hi - hi_t) < tol)

    u["bounds_acc"] = boot(mean_stat([bounds_ok(it, a) for it, a in unm]), len(unm), 1.0)
    u["point_given"] = boot(mean_stat([float(a["point_estimate"] is not None) for _, a in unm]),
                            len(unm), 0.0)
    return {"measured": m, "unmeasured": u}


# ---------------------------------------------------------------- per model


def score_model(items: dict[tuple, dict], records: list[dict], scfg: dict) -> dict:
    """items: key -> item (read_items); records: response records of one model."""
    boot = Boot(scfg.get("n_boot", 2000), scfg.get("seed", 0))
    ans: dict[tuple, dict] = {}
    cats: dict[str, Counter] = defaultdict(Counter)
    stops: Counter = Counter()
    seen, unmatched = set(), 0
    for r in records:
        k = (r["probe"], r["condition"], r["item_id"])
        if k not in items:
            unmatched += 1
            continue
        seen.add(k)
        cats[f"{k[0]}/{k[1]}"][r["category"]] += 1
        if r.get("stop_reason"):
            stops[r["stop_reason"]] += 1
        if r["category"] == "ok":
            ans[k] = r["parsed"]
    for k in items:
        if k not in seen:
            cats[f"{k[0]}/{k[1]}"]["missing"] += 1
    by_probe = defaultdict(list)
    for it in items.values():
        by_probe[it["probe"]].append(it)
    out: dict = {
        "categories": {pc: {c: int(cats[pc][c]) for c in CATEGORIES} for pc in sorted(cats)},
        "stop_reasons": dict(stops),
        "unmatched_records": unmatched,
    }
    if by_probe["V1"]:
        out["V1"] = score_v1(by_probe["V1"], ans, boot, float(scfg.get("split_min_spread", 0.02)))
    if by_probe["V2"]:
        out["V2"] = score_v2(by_probe["V2"], ans, boot, scfg.get("rho_values", []))
    if by_probe["V3"]:
        out["V3"] = score_v3(by_probe["V3"], ans, boot, float(scfg.get("bound_tol", 0.03)))
    return out


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------- output


def _clean(o):
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _fmt(e: dict | None) -> str:
    if not e or e.get("value") is None or not np.isfinite(e["value"]):
        return "–"
    lo, hi = e["ci"]
    ci = f" [{lo:.2f}, {hi:.2f}]" if lo is not None and np.isfinite(lo) else ""
    return f"{e['value']:.3f}{ci} n={e['n']}"


def md_rows(rho_values) -> list[tuple[str, tuple[str, ...]]]:
    rows = [
        ("V1 stated: MAE vs oracle", ("V1", "stated", "mae")),
        *[(f"V1 stated: MAE, {e} orderings", ("V1", "stated", f"mae_{e}")) for e in EST_NAMES],
        ("V1 stated: split slope", ("V1", "stated", "split_slope")),
        ("V1 stated: split R²", ("V1", "stated", "split_r2")),
        ("V1 reversal sign agreement, adj", ("V1", "stated", "reversal_agree_adj")),
        ("V1 reversal sign agreement, crude", ("V1", "stated", "reversal_agree_crude")),
        ("V1 reversal sign agreement, both", ("V1", "stated", "reversal_agree_both")),
        ("V1 stated: determinable=true", ("V1", "stated", "determinable_true")),
        ("V1 blind: determinable=false", ("V1", "blind", "determinable_false")),
        ("V1 blind: MAE vs ordering mixture", ("V1", "blind", "mae_vs_mixture")),
        *[(f"V1 blind: nearest functional = {e}", ("V1", "blind", f"nearest_{e}"))
          for e in EST_NAMES],
        *[(f"V2 stated: crossing slope, ρ={float(r):g}", ("V2", "stated", f"slope_rho={float(r):g}"))
          for r in rho_values],
        ("V2 stated: ρ-tracking slope", ("V2", "stated", "rho_tracking_slope")),
        ("V2 stated: MAE crossing", ("V2", "stated", "mae_crossing")),
        ("V2 stated: MAE same-side", ("V2", "stated", "mae_same_side")),
        ("V2 stated: determinable=true, crossing", ("V2", "stated", "determinable_true_crossing")),
        ("V2 unstated: crossing slope", ("V2", "unstated", "slope_crossing")),
        ("V2 unstated: MAE crossing", ("V2", "unstated", "mae_crossing")),
        ("V2 unstated: MAE same-side", ("V2", "unstated", "mae_same_side")),
        ("V2 unstated: determinable=true, crossing", ("V2", "unstated",
                                                      "determinable_true_crossing")),
        ("V2 unstated: determinable=true, same-side", ("V2", "unstated",
                                                       "determinable_true_same_side")),
        ("V3 measured: identifiable=true", ("V3", "measured", "identifiable_acc")),
        ("V3 measured: point-estimate MAE", ("V3", "measured", "point_mae")),
        ("V3 unmeasured: identifiable=false", ("V3", "unmeasured", "identifiable_acc")),
        ("V3 unmeasured: bounds within tol", ("V3", "unmeasured", "bounds_acc")),
        ("V3 unmeasured: point estimate given anyway", ("V3", "unmeasured", "point_given")),
    ]
    return rows


def render_md(scores: dict, rho_values) -> str:
    models = list(scores["models"])
    lines = ["# E6 scores", "",
             f"Items: {scores['meta']['n_items']} ({scores['meta']['items_dir']}); bootstrap "
             f"{scores['meta']['n_boot']} resamples, 95% percentile CIs; only `ok` answers are "
             "scored. Cells: value [CI] n.", ""]
    lines.append("| metric | ideal | " + " | ".join(models) + " |")
    lines.append("|---|---|" + "---|" * len(models))
    for label, (probe, cond, key) in md_rows(rho_values):
        cells, ideal = [], None
        for m in models:
            e = scores["models"][m].get(probe, {}).get(cond, {}).get(key)
            if e and e.get("target") is not None:
                ideal = e["target"]
            cells.append(_fmt(e))
        lines.append(f"| {label} | {'–' if ideal is None else f'{ideal:g}'} | "
                     + " | ".join(cells) + " |")
    lines += ["", "| model | " + " | ".join(CATEGORIES) + " |",
              "|---|" + "---|" * len(CATEGORIES)]
    for m in models:
        tot = Counter()
        for c in scores["models"][m]["categories"].values():
            tot.update(c)
        lines.append(f"| {m} | " + " | ".join(str(tot[c]) for c in CATEGORIES) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score E6 responses against the item oracles.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", help="default: config out_dir")
    ap.add_argument("--models", nargs="+", metavar="MODEL")
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(args.out_dir or cfg["out_dir"])
    items = read_items(out_dir / "items")
    if not items:
        raise SystemExit(f"no items under {out_dir / 'items'}")
    resp_dir = out_dir / "responses"
    files = sorted(resp_dir.glob("*.jsonl")) if resp_dir.exists() else []
    if args.models:
        files = [f for f in files if f.stem in args.models]
    if not files:
        raise SystemExit(f"no response files under {resp_dir}")
    scfg = {**cfg.get("score", {}), "rho_values": cfg["v2"]["rho_values"],
            "bound_tol": cfg["v3"].get("bound_tol", 0.03)}
    scores = {"meta": {"items_dir": str(out_dir / "items"), "n_items": len(items),
                       "n_boot": scfg.get("n_boot", 2000), "probes": list(PROBES)},
              "models": {f.stem: score_model(items, read_records(f), scfg) for f in files}}
    scores = _clean(scores)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    (out_dir / "scores.md").write_text(render_md(scores, scfg["rho_values"]), encoding="utf-8")
    print(f"wrote {out_dir / 'scores.json'} and {out_dir / 'scores.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
