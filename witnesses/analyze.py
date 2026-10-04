"""Aggregate finished runs, compute pre-registered criteria with bootstrap CIs, write tables.

    python -m witnesses.analyze            # all experiments with results
    python -m witnesses.analyze --exp e1 e3

Outputs: results/summary.json and results/report_tables.md. CIs are 95% percentile bootstrap
intervals. Where per-episode predictions were saved, replicates resample seeds AND episodes
(eval sets are identical across seeds and conditions, so episode indices are shared); otherwise
they resample seeds only. A verdict is PASS/FAIL on the point estimate; "(CI crosses)" flags
estimates whose interval straddles the threshold. Runs that hit the step cap are listed as
unconverged and never counted as negative evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from witnesses.tasks.base import fit_slope, kl_bernoulli_bits

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
B_BOOT = 1000
RNG = np.random.default_rng(20261003)


# ================================================================ loading
def load_runs(exp: str, prefix: str | None = None, exact: str | None = None) -> list[dict]:
    out = []
    d = RES / exp
    if not d.exists():
        return out
    for rd in sorted(d.iterdir()):
        if not rd.is_dir() or rd.name.endswith("_smoke") or "localcheck" in rd.name:
            continue
        cond, _, seed = rd.name.rpartition("_seed")
        if exact is not None and cond != exact:
            continue
        if prefix is not None and not cond.startswith(prefix):
            continue
        m = rd / "metrics.json"
        if not m.exists():
            continue
        js = json.loads(m.read_text())
        if js.get("status") != "done":
            continue
        out.append(dict(cond=cond, seed=int(seed), dir=rd, final=js["final"], log=js["log"],
                        converged=js.get("converged"), steps=js.get("step")))
    return out


def preds(run: dict) -> dict:
    if "_preds" not in run:
        p = run["dir"] / "eval_preds.npz"
        run["_preds"] = dict(np.load(p, allow_pickle=True)) if p.exists() else {}
    return run["_preds"]


# ================================================================ bootstrap helpers
def seed_boot(vals) -> tuple[float, float, float]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float("nan"), float("nan")
    reps = RNG.choice(v, size=(B_BOOT, v.size)).mean(1)
    return float(v.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def ep_seed_boot(n_seeds: int, n_ep: int, fn) -> tuple[float, float, float]:
    """fn(seed_idx_array, episode_idx_array) -> scalar. Two-level percentile bootstrap."""
    point = fn(np.arange(n_seeds), np.arange(n_ep))
    reps = []
    for _ in range(B_BOOT):
        s = RNG.integers(0, n_seeds, n_seeds)
        e = RNG.integers(0, n_ep, n_ep)
        reps.append(fn(s, e))
    reps = np.asarray(reps, float)
    reps = reps[np.isfinite(reps)]
    if reps.size < 10:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def verdict(est: tuple, lo_ok=None, hi_ok=None) -> str:
    v, lo, hi = est
    if not np.isfinite(v):
        return "n/a"
    ok = (lo_ok is None or v >= lo_ok) and (hi_ok is None or v <= hi_ok)
    crosses = False
    if np.isfinite(lo) and np.isfinite(hi):
        if lo_ok is not None and (lo < lo_ok <= hi):
            crosses = True
        if hi_ok is not None and (lo <= hi_ok < hi):
            crosses = True
    return ("PASS" if ok else "FAIL") + (" (CI crosses)" if crosses else "")


def fmt(est: tuple, nd: int = 3) -> str:
    v, lo, hi = est
    if not np.isfinite(v):
        return "n/a"
    if np.isfinite(lo):
        return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
    return f"{v:.{nd}f}"


def conv_note(runs: list[dict]) -> str:
    n = len(runs)
    nc = sum(1 for r in runs if r["converged"])
    return f"{n} runs ({nc} converged, {n - nc} hit the cap)"


# ================================================================ E1
def analyze_e1() -> dict:
    aware, shuf = load_runs("e1", exact="aware"), load_runs("e1", exact="shuffled")
    if not aware:
        return {}
    rows, out = [], {"runs": {"aware": conv_note(aware), "shuffled": conv_note(shuf)}}
    P = [preds(r) for r in aware]
    K = 8

    def gc_fn(si, ei):
        num = den = 0.0
        for s in si:
            p = P[s]
            l2 = ~p["main_is_func"][ei]
            o, m, b = p["main_oracle"][ei][l2][:, K:], p["main_ans"][ei][l2][:, K:], p["main_blind"][ei][l2][:, K:]
            num += kl_bernoulli_bits(o, m).sum(-1).mean()
            den += kl_bernoulli_bits(o, b).sum(-1).mean()
        return 1 - num / den

    n_ep = len(P[0]["main_is_func"])
    gc = ep_seed_boot(len(P), n_ep, gc_fn)

    def split_fn_factory(PP):
        def f(si, ei):
            xs, ys = [], []
            for s in si:
                wm, wo = PP[s]["witness_model"][ei], PP[s]["witness_oracle"][ei]
                dm = wm - wm.mean(1, keepdims=True)
                do = wo - wo.mean(1, keepdims=True)
                sel = (wo.max(1) - wo.min(1)).max(1) >= 0.02
                xs.append(do[sel].ravel())
                ys.append(dm[sel].ravel())
            return fit_slope(np.concatenate(xs), np.concatenate(ys))[0]
        return f

    nw = len(P[0]["witness_model"])
    split = ep_seed_boot(len(P), nw, split_fn_factory(P))
    r2 = seed_boot([r["final"]["split_r2"] for r in aware])
    rows += [("split slope (aware)", fmt(split), "0.9–1.1", verdict(split, 0.9, 1.1)),
             ("split R² (aware)", fmt(r2), "≥ 0.9", verdict(r2, 0.9)),
             ("GC_L2 vs ordering-blind Bayes (aware)", fmt(gc), "≥ 0.85", verdict(gc, 0.85))]
    if shuf:
        PS = [preds(r) for r in shuf]
        ss = ep_seed_boot(len(PS), nw, split_fn_factory(PS))
        rows.append(("split slope (shuffled-ordering control)", fmt(ss), "|·| ≤ 0.05",
                     verdict((abs(ss[0]), np.nan, np.nan), None, 0.05)))
        out["split_slope_shuffled"] = ss
    for e in ("adj", "crude"):
        ra = seed_boot([r["final"][f"reversal_agree_{e}"] for r in aware])
        rows.append((f"Simpson-reversal sign agreement, {e.upper()} ordering", fmt(ra), "≥ 0.95", verdict(ra, 0.95)))
        out[f"reversal_{e}"] = ra
    for e in ("adj", "crude", "marg"):
        d = seed_boot([r["final"][f"kl_l2_{e}"] - r["final"][f"kl_func_{e}"] for r in aware])
        parity = "PASS" if (np.isfinite(d[1]) and d[1] <= 0 <= d[2]) or abs(d[0]) < 1e-3 else "FAIL"
        rows.append((f"L2 − FUNC KL, {e.upper()} (bits)", fmt(d, 4), "CI contains 0", parity))
        out[f"l2_minus_func_{e}"] = d
    out.update(gc=gc, split=split, split_r2=r2, rows=rows)
    return out


# ================================================================ E3
def analyze_e3() -> dict:
    runs = {c: load_runs("e3", exact=c) for c in ("aware", "shuffled", "l2trained")}
    if not runs["aware"]:
        return {}
    out = {"runs": {c: conv_note(r) for c, r in runs.items()}}
    rows = []
    A = runs["aware"]
    st = seed_boot([r["final"]["slope_err_train"] for r in A])
    sh = seed_boot([r["final"]["slope_err_heldout"] for r in A])
    ss = seed_boot([r["final"]["slope_same_side"] for r in A])
    ne = seed_boot([r["final"]["nexcess_cf"] for r in A])
    rows += [("mean |slope − ρ|, training ρ bins", fmt(st), "≤ 0.05", verdict(st, None, 0.05)),
             ("mean |slope − ρ|, held-out ρ bins (|ρ|<0.3)", fmt(sh), "≤ 0.10", verdict(sh, None, 0.10)),
             ("same-side slope", fmt(ss), "1 ± 0.05", verdict(ss, 0.95, 1.05)),
             ("normalized excess MSE (cf head)", fmt(ne), "≤ 0.10", verdict(ne, None, 0.10))]
    for c in ("shuffled", "l2trained"):
        if runs[c]:
            worst = seed_boot([max(abs(b["slope"]) for b in r["final"]["rho_bins"]
                                   if np.isfinite(b["slope"])) for r in runs[c]])
            rows.append((f"control '{c}': max |slope| over ρ bins", fmt(worst), "≤ 0.05", verdict(worst, None, 0.05)))
            out[f"{c}_max_slope"] = worst
    # per-bin slope table for the figure
    bins = {}
    for c, rr in runs.items():
        if rr:
            bb = np.array([[b["slope"] for b in r["final"]["rho_bins"]] for r in rr])
            bins[c] = dict(rho=[b["rho_mean"] for b in rr[0]["final"]["rho_bins"]],
                           heldout=[b["heldout"] for b in rr[0]["final"]["rho_bins"]],
                           slope_mean=np.nanmean(bb, 0).tolist(), slope_sd=np.nanstd(bb, 0).tolist())
    out.update(rows=rows, bins=bins)
    return out


# ================================================================ E4
def analyze_e4() -> dict:
    runs = load_runs("e4", exact="main")
    if not runs:
        return {}
    ks = runs[0]["final"]["eval_k"]
    rows = []
    out = {"runs": conv_note(runs), "eval_k": ks, "cells": {}}
    ranges = {"obs_conf": (-0.6, -0.35), "obs_med": (-0.6, -0.35), "hid_med": (-0.6, -0.35),
              "hid_conf": (-0.25, 0.0)}
    for cell, (lo, hi) in ranges.items():
        sl = seed_boot([r["final"][f"{cell}_loglog_slope_model"] for r in runs])
        so = runs[0]["final"][f"{cell}_loglog_slope_oracle"]
        rows.append((f"log-log slope of posterior sd vs k, {cell.upper()} (oracle {so:+.2f})", fmt(sl),
                     f"[{lo}, {hi}]", verdict(sl, lo, hi)))
        ratio = np.array([r["final"][f"{cell}_sd_ratio"] for r in runs])  # [seeds, len(ks)]
        sel = [i for i, k in enumerate(ks) if k >= 16]
        worst = seed_boot([max(abs(np.log(rr[i])) for i in sel) for rr in ratio])
        rows.append((f"max |log sd ratio| over k ≥ 16, {cell.upper()}", fmt(worst),
                     f"≤ {np.log(1.15):.3f} (ratio in 0.85–1.15)", verdict(worst, None, np.log(1.15))))
        cov = np.array([r["final"][f"{cell}_cov80"] for r in runs])
        cmin, cmax = seed_boot(cov[:, sel].min(1)), seed_boot(cov[:, sel].max(1))
        rows.append((f"80% interval coverage range over k ≥ 16, {cell.upper()}",
                     f"{cmin[0]:.3f} – {cmax[0]:.3f}", "within 0.77–0.83",
                     "PASS" if cmin[0] >= 0.77 and cmax[0] <= 0.83 else "FAIL"))
        out["cells"][cell] = dict(sd_model=np.mean([r["final"][f"{cell}_sd_model"] for r in runs], 0).tolist(),
                                  sd_oracle=runs[0]["final"][f"{cell}_sd_oracle"], slope=sl)
    out["rows"] = rows
    return out


# ================================================================ E2 / E2c
def classify_e2(gc_heldout: float, gc_cond: float, gc_in: float) -> str:
    if gc_in < 0.85:
        return "H_arch (in-distribution fails)" if gc_in <= 0.5 else "in-distribution below criterion"
    if gc_heldout >= 0.8:
        return "H_info"
    if gc_heldout <= 0.5:
        return "H_1¾" if gc_cond >= 0.8 else "H_arch-like (matched L1 also fails)"
    return "inconclusive"


def analyze_e2(mode: str) -> dict:
    out: dict = {"folds": {}}
    rows = []
    allf = []
    for f in range(5):
        runs = load_runs("e2", exact=f"{mode}_f{f}")
        if not runs:
            continue
        res = {}
        for key in ("gc_in_dist_do", "gc_in_dist_cond", "gc_heldout_do", "gc_heldout_cond",
                    "gc_composite_train_do", "gc_composite_heldout_do", "gc_composite_train_cond",
                    "gc_composite_heldout_cond", "slope_heldout_do", "slope_composite_heldout_do",
                    "swap_slope"):
            res[key] = seed_boot([r["final"].get(key) for r in runs])
        res["outcome_heldout"] = classify_e2(res["gc_heldout_do"][0], res["gc_heldout_cond"][0], res["gc_in_dist_do"][0])
        res["outcome_combined"] = classify_e2(res["gc_composite_heldout_do"][0], res["gc_composite_heldout_cond"][0],
                                              res["gc_in_dist_do"][0])
        res["runs"] = conv_note(runs)
        out["folds"][f] = res
        allf.append(runs)
    if not allf:
        return {}
    flat = [r for rr in allf for r in rr]
    for key, thr, label in (("gc_in_dist_do", 0.85, "GC, in-distribution do()"),
                            ("gc_heldout_do", 0.8, "GC, held-out topologies do()"),
                            ("gc_heldout_cond", 0.8, "GC, held-out topologies, matched L1 (COND)"),
                            ("gc_composite_train_do", 0.8, "GC, combined interventions (training topologies)"),
                            ("gc_composite_heldout_do", 0.8, "GC, combined interventions (held-out topologies)"),
                            ("slope_heldout_do", 0.8, "slope (model−L1) on (L2−L1), held-out"),
                            ("swap_slope", 0.8, "same-data Markov-equivalent graph swap: split slope")):
        est = seed_boot([r["final"].get(key) for r in flat])
        rows.append((label + " (pooled over folds)", fmt(est), f"≥ {thr}", verdict(est, thr)))
        out[key] = est
    out["rows"] = rows
    return out


def analyze_e2c() -> dict:
    runs = load_runs("e2c", exact="main")
    if not runs:
        return {}
    rows, out = [], {"runs": conv_note(runs)}
    for kind in ("combo", "untried"):
        g = seed_boot([r["final"][f"gc_{kind}"] for r in runs])
        rows.append((f"GC vs model-free '1¾' baseline, {kind}", fmt(g), "≥ 0.8", verdict(g, 0.8)))
        for tag in ("sharp", "diffuse"):
            gg = seed_boot([r["final"][f"gc_{kind}_{tag}"] for r in runs])
            rows.append((f"  … {tag} half (graph-posterior sd)", fmt(gg), "(reported)", ""))
            out[f"gc_{kind}_{tag}"] = gg
        out[f"gc_{kind}"] = g
    tr = seed_boot([r["final"]["kl_tried"] - r["final"]["kl_baseline_tried"] for r in runs])
    rows.append(("tried regimes: KL(model) − KL(baseline) (bits)", fmt(tr, 4), "≤ 0", verdict(tr, None, 0.0)))
    out["rows"] = rows
    return out


# ================================================================ E5
def analyze_e5() -> dict:
    out, rows = {}, []
    for c in ("restart", "clock"):
        rr = load_runs("e5a", exact=c)
        if rr:
            g = seed_boot([r["final"]["gc_tv_prog"] for r in rr])
            a = seed_boot([r["final"]["abl_kl_l1"] for r in rr])
            rows += [(f"E5a {c}: GC_TV (programs, a≠0)", fmt(g), "≥ 0.85", verdict(g, 0.85)),
                     (f"E5a {c}: flags zeroed → KL to Bayes-L1 (bits)", fmt(a), "≤ 0.05", verdict(a, None, 0.05))]
            out[f"e5a_{c}"] = dict(gc_tv_prog=g, abl_kl_l1=a, runs=conv_note(rr))
    for c in ("cf_abducted", "cf_fresh"):
        rr = load_runs("e5b", exact=c)
        if rr:
            gn = seed_boot([r["final"]["gc_tv_noise"] for r in rr])
            gp = seed_boot([r["final"]["gc_tv_prog"] for r in rr])
            gh = seed_boot([r["final"].get("gc_tv_heldout") for r in rr])
            kh = seed_boot([r["final"]["kl_fact_heldout"] / max(r["final"]["kl_fact_prog"], 1e-9) for r in rr])
            rows += [(f"E5b {c}: GC_TV noise episodes", fmt(gn), "≥ 0.85", verdict(gn, 0.85)),
                     (f"E5b {c}: GC_TV program episodes", fmt(gp), "≥ 0.85", verdict(gp, 0.85)),
                     (f"E5b {c}: GC_TV held-out programs", fmt(gh), "(reported)", ""),
                     (f"E5b {c}: factual KL held-out / seen", fmt(kh), "≤ 2 (L1 generalizes)", verdict(kh, None, 2.0))]
            out[f"e5b_{c}"] = dict(gc_tv_noise=gn, gc_tv_prog=gp, gc_tv_heldout=gh, fact_ratio=kh, runs=conv_note(rr))
    out["rows"] = rows
    return out if rows else {}


# ================================================================ report
SECTIONS = [("e1", "E1 — Simpson pair over six orderings (demonstration)", analyze_e1),
            ("e2_cpt", "E2-0 — graph + CPTs given (the test)", lambda: analyze_e2("cpt")),
            ("e2_samples", "E2-full — graph + samples given (the test)", lambda: analyze_e2("samples")),
            ("e2c", "E2c — no graph; tried do(A), do(B) (the test)", analyze_e2c),
            ("e3", "E3 — residual twin with continuous ρ (demonstration)", analyze_e3),
            ("e4", "E4 — identifiability tracking (demonstration)", analyze_e4),
            ("e5", "E5 — witness pairs in Misra's task", analyze_e5)]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*")
    a = ap.parse_args()
    summary, md = {}, ["# Results against the pre-registered criteria", "",
                       "Generated by `python -m witnesses.analyze`. Estimates are means over seeds; "
                       "brackets are 95% bootstrap CIs.", ""]
    for key, title, fn in SECTIONS:
        if a.exp and not any(key.startswith(e) for e in a.exp):
            continue
        res = fn()
        if not res:
            continue
        summary[key] = res
        md += [f"## {title}", ""]
        if "runs" in res:
            md += [f"Runs: {res['runs']}", ""]
        if key.startswith("e2_") and res.get("folds"):
            md += ["| fold | GC in-dist | GC held-out (do) | GC held-out (COND) | GC combined held-out | swap slope | outcome (held-out) | outcome (combined) |",
                   "|---|---|---|---|---|---|---|---|"]
            for f, r in res["folds"].items():
                md.append(f"| {f} | {fmt(r['gc_in_dist_do'])} | {fmt(r['gc_heldout_do'])} | {fmt(r['gc_heldout_cond'])} | "
                          f"{fmt(r['gc_composite_heldout_do'])} | {fmt(r['swap_slope'])} | {r['outcome_heldout']} | {r['outcome_combined']} |")
            md.append("")
        md += ["| criterion | estimate | threshold | verdict |", "|---|---|---|---|"]
        md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in res["rows"]]
        md.append("")
    (RES / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1, default=str))
    (RES / "report_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
