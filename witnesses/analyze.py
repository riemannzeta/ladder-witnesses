"""Aggregate finished runs, compute pre-registered criteria with bootstrap CIs, write tables.

    python -m witnesses.analyze            # all experiments with results
    python -m witnesses.analyze --exp e1 e3

Outputs: results/summary.json, results/report_tables.md, and the generated-tables block of
REPORT.md (between its BEGIN/END markers); with --exp the selected sections are only printed. CIs are 95% percentile bootstrap
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

from witnesses.tasks import dag_utils as du
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
        final, rescored = js["final"], False
        rs = rd / "metrics_rescored.json"
        if rs.exists():  # E2-full: same frozen items with the NaN oracle entries repaired (Addendum 2)
            final, rescored = json.loads(rs.read_text())["final"], True
        out.append(dict(cond=cond, seed=int(seed), dir=rd, final=final, log=js["log"], rescored=rescored,
                        converged=js.get("converged"), steps=js.get("step"), stop=js.get("stop_reason")))
    return out


def balanced(run: dict) -> dict:
    """Balanced combined-query re-evaluation (scripts/reeval_e2.py, Addendum 1 A), if present."""
    p = run["dir"] / "metrics_balanced.json"
    return json.loads(p.read_text())["final"] if p.exists() else {}


def n_params(run: dict) -> int | None:
    p = run["dir"] / "meta.json"
    return json.loads(p.read_text()).get("n_params") if p.exists() else None


def preds(run: dict) -> dict:
    if "_preds" not in run:
        p = run["dir"] / ("eval_preds_rescored.npz" if run.get("rescored") else "eval_preds.npz")
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
        rows.append((f"80% coverage over k ≥ 16, {cell.upper()} — as implemented (bin-edge rounding)",
                     f"{cmin[0]:.3f} – {cmax[0]:.3f}", "within 0.77–0.83",
                     "PASS" if cmin[0] >= 0.77 and cmax[0] <= 0.83 else "FAIL"))
        ci = np.array([_interp_coverage(preds(r), cell) for r in runs])  # [seeds, len(ks)]
        oc = _interp_coverage(preds(runs[0]), cell, oracle=True)
        lo_i, hi_i = ci[:, sel].min(1).mean(), ci[:, sel].max(1).mean()
        rows.append((f"80% coverage over k ≥ 16, {cell.upper()} — interpolated (oracle: "
                     f"{oc[sel].min():.3f}–{oc[sel].max():.3f})", f"{lo_i:.3f} – {hi_i:.3f}", "within 0.77–0.83",
                     "PASS" if lo_i >= 0.77 and hi_i <= 0.83 else "FAIL"))
        out["cells"][cell] = dict(sd_model=np.mean([r["final"][f"{cell}_sd_model"] for r in runs], 0).tolist(),
                                  sd_oracle=runs[0]["final"][f"{cell}_sd_oracle"], slope=sl)
    out["rows"] = rows
    return out


def _quant(h: np.ndarray, x: float) -> np.ndarray:
    e = np.linspace(0, 1, h.shape[-1] + 1)
    c = np.concatenate([np.zeros((h.shape[0], 1)), np.cumsum(h, -1)], -1)
    return np.array([np.interp(x, ci, e) for ci in c])


def e4_tail_diag(p: dict, cell: str, ki: int) -> dict:
    """Post-hoc E4 diagnostic (Addendum 2): is the sd excess in the posterior core or the far tail?"""
    mh, oh = p[f"{cell}_model_hist"].astype(float)[:, ki], p[f"{cell}_oracle_hist"].astype(float)[:, ki]
    nb = mh.shape[-1]
    cen = (np.arange(nb) + 0.5) / nb
    wm, wo = _quant(mh, 0.9) - _quant(mh, 0.1), _quant(oh, 0.9) - _quant(oh, 0.1)
    iqm, iqo = _quant(mh, 0.75) - _quant(mh, 0.25), _quant(oh, 0.75) - _quant(oh, 0.25)
    om, mm = (oh * cen).sum(-1), (mh * cen).sum(-1)
    osd = np.sqrt(((oh * cen ** 2).sum(-1) - om ** 2).clip(0))
    far = np.abs(cen[None] - om[:, None]) > 5 * osd[:, None]
    p1 = p[f"{cell}_p1"]
    return dict(width80_ratio=float(np.median(wm / wo)), iqr_ratio=float(np.median(iqm / iqo)),
                far_tail_mass=float((mh * far).sum(-1).mean()), far_tail_mass_oracle=float((oh * far).sum(-1).mean()),
                rmse_ratio=float(np.sqrt(((mm - p1) ** 2).mean()) / np.sqrt(((om - p1) ** 2).mean())))


def _e4_preds_k256(run: dict) -> dict:
    """Per-episode histograms at the last eval k (= 256). The 256-bin runs keep only this slice in
    the repository (eval_preds_k256.npz; the full files are ~76 MB each and are gitignored)."""
    assert run["final"]["eval_k"][-1] == 256
    sl = run["dir"] / "eval_preds_k256.npz"
    return dict(np.load(sl)) if sl.exists() else preds(run)


def analyze_e4_bins() -> dict:
    """Post-hoc arm 2 (Addendum 1 D): the 64-bin main runs vs the 256-bin head."""
    arms = {"64 bins (main)": load_runs("e4", exact="main"), "256 bins": load_runs("e4", exact="bins256")}
    if not arms["256 bins"]:
        return {}
    rows, out = [], {"by_steps": {}}
    ks = arms["256 bins"][0]["final"]["eval_k"]
    ki = ks.index(256)
    sel = [i for i, k in enumerate(ks) if k >= 16]
    r256 = seed_boot([r["final"]["obs_conf_sd_ratio"][ki] for r in arms["256 bins"]])
    r64 = seed_boot([r["final"]["obs_conf_sd_ratio"][ki] for r in arms["64 bins (main)"]])
    rows.append((f"OBS-CONF sd ratio at k = 256, 256 bins (64 bins: {fmt(r64)})", fmt(r256),
                 "> 1.10 (excess persists)", _held(r256, 1.10)))
    for cell in ("obs_conf", "obs_med", "hid_med", "hid_conf"):
        cov = np.array([r["final"][f"{cell}_cov80"] for r in arms["256 bins"]])
        lo, hi = cov[:, sel].min(1).mean(), cov[:, sel].max(1).mean()
        rows.append((f"as-implemented 80% coverage over k ≥ 16, {cell.upper()}, 256 bins", f"{lo:.4f} – {hi:.4f}",
                     "within 0.77–0.83", "prediction held" if lo >= 0.77 and hi <= 0.83 else "prediction failed"))
    hdr = ["head", "cell", "steps (stop)", "sd ratio k=256", "80%-width ratio", "IQR ratio",
           "mass beyond 5 oracle sd (model / oracle)", "posterior-mean RMSE ratio"]
    trows = []
    for arm, rr in arms.items():
        for cell in ("obs_conf", "obs_med", "hid_med", "hid_conf"):
            d = [e4_tail_diag(_e4_preds_k256(r), cell, -1) for r in rr]
            m = {k: float(np.mean([x[k] for x in d])) for k in d[0]}
            sr = [r["final"][f"{cell}_sd_ratio"][ki] for r in rr]
            steps = sorted({r["steps"] for r in rr})
            stops = sorted({r["stop"] for r in rr})
            trows.append([arm, cell.upper(), f"{steps[0] // 1000}k–{steps[-1] // 1000}k ({', '.join(stops)})",
                          f"{min(sr):.2f}–{max(sr):.2f}", f"{m['width80_ratio']:.3f}", f"{m['iqr_ratio']:.3f}",
                          f"{100 * m['far_tail_mass']:.2f}% / {100 * m['far_tail_mass_oracle']:.2f}%", f"{m['rmse_ratio']:.3f}"])
            out.setdefault("diag", {}).setdefault(arm, {})[cell] = m
        out["by_steps"][arm] = [(r["steps"], r["final"]["obs_conf_sd_ratio"][ki]) for r in rr]
    out.update(rows=rows, tables=[dict(header=hdr, rows=trows)],
               note="Diagnostics at k = 256, means over 5 seeds; widths from interpolated quantiles. Coverage uses "
                    "the main E4 analysis's aggregation (each seed's min/max over k, then the mean over seeds); "
                    "Addendum 2 averaged over seeds first and so reported 2 of 4 cells outside the band, not 3 "
                    "(HID-MED: 0.8292 vs 0.8301).")
    return out


def _interp_coverage(p: dict, cell: str, oracle: bool = False) -> np.ndarray:
    """Coverage of the TRUE p1 by central 80% intervals with quantiles linearly interpolated within
    the 64 bins (the bin-edge version in E4's evaluate() rounds outward and over-covers even for the
    exact oracle)."""
    h = p[f"{cell}_oracle_hist" if oracle else f"{cell}_model_hist"].astype(float)
    p1 = p[f"{cell}_p1"]
    edges = np.linspace(0, 1, h.shape[-1] + 1)
    cdf = np.concatenate([np.zeros(h.shape[:-1] + (1,)), np.cumsum(h, -1)], -1)
    lo = np.apply_along_axis(lambda c: np.interp(0.1, c, edges), -1, cdf)
    hi = np.apply_along_axis(lambda c: np.interp(0.9, c, edges), -1, cdf)
    return ((p1[:, None] >= lo) & (p1[:, None] <= hi)).mean(0)


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
                            ("slope_composite_heldout_do", 0.8, "slope (model−L1) on (L2−L1), combined, held-out"),
                            ("gc_composite_train_cond", 0.8,
                             "GC, combined, matched L1 (two-variable conditioning), training topologies"),
                            ("gc_composite_heldout_cond", 0.8,
                             "GC, combined, matched L1 (two-variable conditioning), held-out topologies"),
                            ("swap_slope", 0.8, "same-data Markov-equivalent graph swap: split slope")):
        est = seed_boot([r["final"].get(key) for r in flat])
        rows.append((label + " (pooled over folds)", fmt(est), f"≥ {thr}", verdict(est, thr)))
        out[key] = est
    for key, label in (("kl_composite_train_cond", "training"), ("kl_composite_heldout_cond", "held-out")):
        est = seed_boot([r["final"].get(key) for r in flat])
        rows.append((f"KL to Bayes, combined two-variable conditioning, {label} topologies (bits)", fmt(est, 4),
                     "(reported)", ""))
        out[key] = est
    if any(r["rescored"] for r in flat):
        out["note"] = ("Scored on the frozen sets with the NaN oracle entries repaired (PREDICTIONS.md, Addendum 2; "
                       f"{sum(r['rescored'] for r in flat)}/{len(flat)} runs re-scored).")
    # balanced combined-query re-evaluation (scripts/reeval_e2.py), when available
    bal = []
    for rr in allf:
        for r in rr:
            bp = r["dir"] / "metrics_balanced.json"
            if bp.exists():
                bal.append(json.loads(bp.read_text())["final"])
    if bal:
        def pooled(key):
            return seed_boot([b.get(key) for b in bal if b.get(key) is not None and np.isfinite(b.get(key))])
        def count(key):
            return int(sum(b.get(key, 0) for b in bal) / max(1, len(bal)))
        for split, lab in (("train", "training topologies"), ("heldout", "held-out topologies")):
            e = pooled(f"gc_composite_{split}_strict_do")
            rows.append((f"[balanced] combined do(A,B), strict rules, {lab} (~{count(f'n_composite_{split}_strict_do')} items/run)",
                         fmt(e), "≥ 0.8", verdict(e, 0.8)))
            out[f"bal_gc_{split}_strict_do"] = e
            e = pooled(f"gc_composite_{split}_strict_cond")
            rows.append((f"[balanced] matched L1: conditioning on two variables, {lab}", fmt(e), "≥ 0.8", verdict(e, 0.8)))
            out[f"bal_gc_{split}_strict_cond"] = e
        for split in ("train", "heldout"):
            for tag, lab in (("both_parents", "both pinned variables are parents of T"), ("other", "other structures")):
                e = pooled(f"kl_composite_{split}_relaxed_do_{tag}")
                rows.append((f"[exploratory] relaxed combined do(), {split}: {lab} — KL (bits)", fmt(e, 4), "(reported)", ""))
                out[f"bal_kl_{split}_relaxed_do_{tag}"] = e
        out["balanced_runs"] = len(bal)
    out["rows"] = rows
    return out


# ================================================================ E2 post hoc (PREDICTIONS.md, Addenda 1, 3, 4)
def _est1(v) -> tuple[float, float, float]:
    return (float(v) if v is not None else float("nan"), float("nan"), float("nan"))


def _pool(runs: list[dict], key: str, bal: bool = False) -> tuple[float, float, float]:
    return seed_boot([(balanced(r) if bal else r["final"]).get(key) for r in runs])


def _held(est: tuple, lo=None, hi=None) -> str:
    v = verdict(est, lo, hi)
    return v.replace("PASS", "prediction held").replace("FAIL", "prediction failed")


def analyze_e2_cpt_comp() -> dict:
    """Post-hoc arm 1 (Addendum 1 C): E2-0 with 25% two-effective-element training queries."""
    arm = [r for f in range(5) for r in load_runs("e2", exact=f"cpt_f{f}_comp25")]
    if not arm:
        return {}
    base = [r for f in range(5) for r in load_runs("e2", exact=f"cpt_f{f}")]
    rows, out = [], {"runs": conv_note(arm) + "; one seed per fold, CIs resample runs"}
    for split, lab in (("train", "training"), ("heldout", "held-out")):
        e, b = _pool(arm, f"gc_composite_{split}_strict_cond", True), _pool(base, f"gc_composite_{split}_strict_cond", True)
        rows.append((f"[balanced] two-variable conditioning GC, {lab} topologies (base E2-0: {fmt(b)})",
                     fmt(e), "≥ 0.9", _held(e, 0.9)))
        out[f"bal_gc_{split}_strict_cond"], out[f"base_bal_gc_{split}_strict_cond"] = e, b
    e = _pool(arm, "kl_composite_heldout_relaxed_do_both_parents", True)
    b = _pool(base, "kl_composite_heldout_relaxed_do_both_parents", True)
    rows.append((f"[balanced] relaxed combined do(), both pinned variables parents of T, held-out — KL bits "
                 f"(base E2-0: {fmt(b, 4)})", fmt(e, 4), "≤ 0.01", _held(e, None, 0.01)))
    out["bal_kl_heldout_relaxed_do_both_parents"], out["base_bal_kl_heldout_relaxed_do_both_parents"] = e, b
    for key, lab in (("gc_in_dist_do", "in-distribution do()"), ("gc_heldout_do", "held-out do()"),
                     ("gc_heldout_cond", "held-out COND")):
        e = _pool(arm, key)
        rows.append((f"no single-query regression: GC, {lab}", fmt(e), "≥ 0.99", _held(e, 0.99)))
        out[key] = e
    out["rows"] = rows
    return out


ARM3_CRIT = (("slope_composite_train_cond", 0.9, None), ("slope_composite_heldout_cond", 0.9, None),
             ("kl_composite_train_cond", None, 0.02), ("kl_composite_heldout_cond", None, 0.02))


def _arm3_ok(fin: dict) -> str:
    ok = all(np.isfinite(fin.get(k, np.nan)) and (lo is None or fin[k] >= lo) and (hi is None or fin[k] <= hi)
             for k, lo, hi in ARM3_CRIT)
    return "PASS" if ok else "miss"


def analyze_e2_samples_comp() -> dict:
    """Post-hoc arm 3 (Addenda 3-4): E2-full with two-effective-element training. Folds 0-2 were
    re-run to the 240k cap (Addendum 4); those runs are primary when present, the plateau-stopped
    runs are kept as a training-length comparison."""
    short = {f: load_runs("e2", exact=f"samples_f{f}_comp25") for f in range(5)}
    full = {f: load_runs("e2", exact=f"samples_f{f}_comp25full") for f in range(5)}
    if not any(short.values()):
        return {}
    hdr = ["fold", "run", "steps (stop)", "cond slope train / held-out", "cond KL train / held-out (bits)",
           "combined do() GC train / held-out", "[balanced] strict combined do() GC train / held-out",
           "single-query GC in-dist / held-out do()", "Addendum 3 cond criteria"]
    trows, primary, pending = [], [], []
    for f in range(5):
        for tag, rr in (("240k re-run", full[f]), ("arm 3", short[f])):
            for r in rr:
                fin, bal = r["final"], balanced(r)
                g = lambda k: fin.get(k, float("nan"))  # noqa: E731
                gb = lambda k: bal.get(k, float("nan"))  # noqa: E731
                trows.append([str(f), f"{tag} ({r['cond']})", f"{r['steps'] // 1000}k ({r['stop']})",
                              f"{g('slope_composite_train_cond'):.3f} / {g('slope_composite_heldout_cond'):.3f}",
                              f"{g('kl_composite_train_cond'):.4f} / {g('kl_composite_heldout_cond'):.4f}",
                              f"{g('gc_composite_train_do'):.2f} / {g('gc_composite_heldout_do'):.2f}",
                              f"{gb('gc_composite_train_strict_do'):.2f} / {gb('gc_composite_heldout_strict_do'):.2f}",
                              f"{g('gc_in_dist_do'):.3f} / {g('gc_heldout_do'):.3f}", _arm3_ok(fin)])
        if full[f]:
            primary += full[f]
        elif f <= 2:
            pending.append(f)
            primary += short[f]
        else:
            primary += short[f]
    rows = []
    base = [r for f in range(5) for r in load_runs("e2", exact=f"samples_f{f}")]
    topo = {"train": "training topologies", "heldout": "held-out topologies"}
    for key, lo, hi in ARM3_CRIT:
        split = key.split("_")[2]
        e, b = _pool(primary, key), _pool(base, key)
        what = "slope (model−L2) on (L1−L2)" if lo else "KL to Bayes (bits)"
        rows.append((f"combined conditioning {what}, {topo[split]} (base E2-full: {fmt(b, 4 if hi else 3)})",
                     fmt(e, 4 if hi else 3), f"≥ {lo}" if lo else f"≤ {hi}", _held(e, lo, hi)))
    for key in ("gc_composite_train_do", "gc_composite_heldout_do"):
        e = _pool(primary, key)
        rows.append((f"combined do() GC, {topo[key.split('_')[2]]} (folds whose frozen split has do() items)",
                     fmt(e), "≥ 0.8", _held(e, 0.8)))
    for key, thr, lab in (("gc_in_dist_do", 0.85, "in-distribution do()"), ("gc_heldout_do", 0.8, "held-out do()")):
        e = _pool(primary, key)
        rows.append((f"no single-query regression: GC, {lab}", fmt(e), f"≥ {thr}", _held(e, thr)))
    for split in ("train", "heldout"):
        e, b = _pool(primary, f"kl_composite_{split}_strict_cond", True), _pool(base, f"kl_composite_{split}_strict_cond", True)
        rows.append((f"[balanced] two-variable conditioning, {topo[split]} — KL bits (base E2-full: {fmt(b, 4)})",
                     fmt(e, 4), "(reported)", ""))
        e, b = _pool(primary, f"gc_composite_{split}_strict_do", True), _pool(base, f"gc_composite_{split}_strict_do", True)
        rows.append((f"[balanced] strict combined do(A, B) GC, {topo[split]} (base E2-full: {fmt(b)})",
                     fmt(e), "(reported)", ""))
    capped = [r for r in base if r["stop"] == "cap"]
    if capped:
        for split in ("train", "heldout"):
            e = _pool(capped, f"kl_composite_{split}_cond")
            rows.append((f"(control) pre-registered E2-full runs that also ran to the 240k cap ({len(capped)}): combined "
                         f"conditioning KL, {topo[split]} (bits)", fmt(e, 4), "(reported)", ""))
    n_pass = sum(_arm3_ok(r["final"]) == "PASS" for r in primary)
    note = (f"Primary runs: {n_pass}/{len(primary)} folds meet the Addendum 3 combined-conditioning criteria. "
            "One seed per fold; CIs resample runs.")
    if pending:
        note += f" Folds {pending}: 240k re-run pending (Addendum 4); the plateau-stopped runs stand in."
    tables = [dict(header=hdr, rows=trows)]
    by_topo = _topo_breakdown(short, full)
    if by_topo:
        tables.append(by_topo)
        note += (" Second table (post hoc, exploratory): balanced held-out two-element KL by topology. "
                 "'collider with a child' = an unshielded collider whose middle node has a child; "
                 "'in training' = some training topology of that fold has one too.")
    return {"runs": conv_note(primary), "note": note, "rows": rows, "n_pass": n_pass, "pending": pending,
            "tables": tables}


def _collider_with_child(t: int) -> bool:
    rep = du.topologies()[t]["rep"]
    return any(any((rep[c] >> v[2]) & 1 for c in range(4)) for v in du.v_structures(tuple(rep)))


def _topo_kl(run: dict, split: str) -> dict:
    """Balanced strict two-element KL (bits, mean over k >= 8 as in the pooled rows) per held-out topology."""
    p = run["dir"] / "balanced_preds.npz"
    if not p.exists():
        return {}
    z, n = np.load(p), f"composite_{split}_strict"
    isc, topo, pm = z[f"{n}_is_cond"], z[f"{n}_topo"], z[f"{n}_model"].astype(np.float64)
    right = np.where(isc[:, None], z[f"{n}_l1"], z[f"{n}_l2"]).astype(np.float64)
    kc = slice(1, None) if pm.shape[1] > 1 else slice(None)  # samples: k in (8, 32, 128); cpt: exact
    kl = kl_bernoulli_bits(right[:, kc], pm[:, kc]).mean(1)
    out = {}
    for t in np.unique(topo):
        for qn, qs in (("cond", isc), ("do", ~isc)):
            s = qs & (topo == t)
            if s.any():
                out[(int(t), qn)] = (int(s.sum()), float(kl[s].mean()))
    return out


def _topo_breakdown(short: dict, full: dict) -> dict | None:
    folds = du.load_folds()
    trows = []
    for f in range(5):
        if not short[f]:
            continue
        if full[f]:
            a, b = _topo_kl(short[f][0], "heldout"), _topo_kl(full[f][0], "heldout")
        elif short[f][0]["stop"] == "cap":  # folds 3-4: arm 3 itself ran to the cap
            a, b = {}, _topo_kl(short[f][0], "heldout")
        else:
            continue
        c0 = load_runs("e2", exact=f"cpt_f{f}_comp25")
        c = _topo_kl(c0[0], "heldout") if c0 else {}
        train_cwc = any(_collider_with_child(t) for g, fl in enumerate(folds) if g != f for t in fl)
        for (t, qn) in sorted(b):
            cwc = _collider_with_child(t)
            trows.append([str(f), f"T{t} ({int(du.topologies()[t]['n_edges'])} edges)",
                          ("yes" + (" (in training)" if train_cwc else " (not in training)")) if cwc else "no",
                          "two-variable conditioning" if qn == "cond" else "strict combined do()", str(b[(t, qn)][0]),
                          f"{a[(t, qn)][1]:.4f}" if (t, qn) in a else "—", f"{b[(t, qn)][1]:.4f}",
                          f"{c[(t, qn)][1]:.4f}" if (t, qn) in c else "—"])
    if not trows:
        return None
    return dict(header=["fold", "held-out topology", "collider with a child", "query", "n",
                        "KL arm 3 stopped at 36–38k (bits)", "KL at 240k (bits)", "E2-0 arm 1, CPT tokens (bits)"],
                rows=trows)


def analyze_e2_sweep(mode: str) -> dict:
    """Capacity sweep on folds 0-1 (2 seeds per size); the base size uses the main runs' seeds 0-1."""
    sizes = {}
    for f in (0, 1):
        for r in load_runs("e2", prefix=f"{mode}_f{f}_d"):
            sizes.setdefault(r["cond"].split("_")[-1], []).append(r)
        sizes.setdefault("base", []).extend(r for r in load_runs("e2", exact=f"{mode}_f{f}") if r["seed"] in (0, 1))
    if len(sizes) <= 1:
        return {}
    hdr = ["size", "params", "runs (converged)", "GC in-dist do()", "GC held-out do()", "GC held-out COND",
           "combined cond KL train / held-out (bits)", "[balanced] two-variable cond held-out: GC / KL (bits)"]
    trows, out = [], {}
    order = sorted(sizes, key=lambda k: n_params(sizes[k][0]) or 0)
    for k in order:
        rr = sizes[k]
        g = {key: _pool(rr, key) for key in ("gc_in_dist_do", "gc_heldout_do", "gc_heldout_cond",
                                             "kl_composite_train_cond", "kl_composite_heldout_cond")}
        bc = _pool(rr, "gc_composite_heldout_strict_cond", True)
        bk = _pool(rr, "kl_composite_heldout_strict_cond", True)
        label = k if k != "base" else ("d256L6 (base)")
        trows.append([label, f"{(n_params(rr[0]) or 0) / 1e6:.1f}M", f"{len(rr)} ({sum(bool(r['converged']) for r in rr)})",
                      f"{g['gc_in_dist_do'][0]:.3f}", f"{g['gc_heldout_do'][0]:.3f}", f"{g['gc_heldout_cond'][0]:.3f}",
                      f"{g['kl_composite_train_cond'][0]:.4f} / {g['kl_composite_heldout_cond'][0]:.4f}",
                      f"{bc[0]:.3f} / {bk[0]:.4f}"])
        out[label] = dict(g, bal_gc_heldout_strict_cond=bc, bal_kl_heldout_strict_cond=bk, n_params=n_params(rr[0]))
    return {"sizes": out, "rows": [], "tables": [dict(header=hdr, rows=trows)],
            "note": "Folds 0-1, 2 seeds per size (Addendum 1 B); means over runs."}


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
            ("e5", "E5 — witness pairs in Misra's task", analyze_e5),
            ("posthoc_e2_cpt_comp", "Post hoc (Addendum 1 C) — E2-0 with two-element training queries",
             analyze_e2_cpt_comp),
            ("posthoc_e2_samples_comp", "Post hoc (Addenda 3–4) — E2-full with two-element training queries",
             analyze_e2_samples_comp),
            ("posthoc_e2_cpt_sweep", "Capacity sweep — E2-0", lambda: analyze_e2_sweep("cpt")),
            ("posthoc_e2_samples_sweep", "Capacity sweep — E2-full", lambda: analyze_e2_sweep("samples")),
            ("posthoc_e4_bins", "Post hoc (Addendum 1 D) — E4 with a 256-bin head", analyze_e4_bins)]


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


BEGIN, END = "<!-- BEGIN GENERATED TABLES -->", "<!-- END GENERATED TABLES -->"


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
        if "note" in res:
            md += [res["note"], ""]
        for t in res.get("tables", []):
            md += ["| " + " | ".join(t["header"]) + " |", "|" + "---|" * len(t["header"])]
            md += ["| " + " | ".join(row) + " |" for row in t["rows"]]
            md.append("")
        if key.startswith("e2_") and res.get("folds"):
            md += ["| fold | GC in-dist | GC held-out (do) | GC held-out (COND) | GC combined held-out | swap slope | outcome (held-out) | outcome (combined) |",
                   "|---|---|---|---|---|---|---|---|"]
            for f, r in res["folds"].items():
                md.append(f"| {f} | {fmt(r['gc_in_dist_do'])} | {fmt(r['gc_heldout_do'])} | {fmt(r['gc_heldout_cond'])} | "
                          f"{fmt(r['gc_composite_heldout_do'])} | {fmt(r['swap_slope'])} | {r['outcome_heldout']} | {r['outcome_combined']} |")
            md.append("")
        if res["rows"]:
            md += ["| criterion | estimate | threshold | verdict |", "|---|---|---|---|"]
            md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in res["rows"]]
            md.append("")
    if a.exp:  # partial runs print only; the files always hold the full analysis
        print("\n".join(md))
        return
    (RES / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1, default=str))
    (RES / "report_tables.md").write_text("\n".join(md) + "\n")
    rep = ROOT / "REPORT.md"
    if rep.exists():
        txt = rep.read_text()
        i, j = txt.find(BEGIN), txt.find(END)
        if 0 <= i < j:
            body = "\n".join(line.replace("## ", "### ", 1) if line.startswith("## ") else line for line in md[2:])
            rep.write_text(txt[: i + len(BEGIN)] + "\n" + body + "\n" + txt[j:])
    print("\n".join(md))


if __name__ == "__main__":
    main()
