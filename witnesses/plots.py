"""Tufte-style figures for the report and the blog post.

Style: range frames (axes span only the data), no top/right spines, no grid, muted gray reference
lines, direct labels instead of legends, small multiples where the comparison is across cells.
Colors are the first three slots of the validated categorical palette (blue, orange, aqua;
validated all-pairs on white with scripts/validate_palette.js -- aqua is below 3:1 contrast, so
every series carries a visible direct label). Outputs PNG (1600 px wide) + PDF to results/figures
and copies the PNGs to blog/figures.

    python -m witnesses.plots
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from witnesses.analyze import RES, load_runs, preds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = RES / "figures"
BLOG = ROOT / "blog" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, REF = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.titleweight": "normal",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "xtick.major.size": 3, "ytick.major.size": 3,
        "font.family": "serif", "font.size": 9.5, "lines.linewidth": 1.6,
        "savefig.dpi": 200, "figure.dpi": 100, "savefig.bbox": "tight",
    })


def range_frame(ax, x, y) -> None:
    """Spines span only the data range (Tufte's range frame)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size:
        ax.spines["bottom"].set_bounds(x.min(), x.max())
    if y.size:
        ax.spines["left"].set_bounds(y.min(), y.max())


def label(ax, x, y, text, color, dx=4, dy=0, ha="left", va="center", size=8.5) -> None:
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color,
                fontsize=size, ha=ha, va=va)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    BLOG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    shutil.copy(FIG / f"{name}.png", BLOG / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}")


# ================================================================ E1
def fig_e1() -> None:
    aware, shuf = load_runs("e1", exact="aware"), load_runs("e1", exact="shuffled")
    if not aware:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    lim = 0.0
    for runs, color, text in ((shuf, ORANGE, "ordering token scrambled"), (aware, BLUE, "ordering token given")):
        if not runs:
            continue
        p = preds(runs[0])
        wm, wo = p["witness_model"], p["witness_oracle"]
        dm, do = (wm - wm.mean(1, keepdims=True)).ravel(), (wo - wo.mean(1, keepdims=True)).ravel()
        a1.scatter(do, dm, s=3, color=color, alpha=0.25, linewidths=0, rasterized=True)
        lim = max(lim, np.abs(do).max())
    lim = min(lim, 0.6)
    a1.plot([-lim, lim], [-lim, lim], color=REF, lw=1, zorder=0)
    a1.set_xlim(-lim, lim)
    a1.set_ylim(-lim, lim)
    label(a1, lim * 0.55, lim * 0.62, "ordering given", BLUE, ha="right")
    label(a1, lim * 0.95, -0.07, "ordering scrambled", ORANGE, ha="right")
    a1.set_xlabel("exact Bayes answer, deviation across orderings")
    a1.set_ylabel("transformer answer, deviation")
    a1.set_title("Same data, six causal orderings", loc="left")
    range_frame(a1, [-lim, lim], [-lim, lim])
    # learning curve vs samples
    f = aware[0]["final"]
    k = np.arange(len(f["kl_curve_l2"]))
    curves = [(np.mean([r["final"]["kl_curve_l2"] for r in aware], 0), BLUE, "causal query, ordering given", 7),
              (np.mean([r["final"]["kl_curve_func"] for r in aware], 0), AQUA, "same formula, asked by name", -7),
              (np.mean([r["final"]["kl_curve_blind"] for r in aware], 0), REF, "ordering-blind Bayes", -7)]
    if shuf:
        curves.append((np.mean([r["final"]["kl_curve_l2"] for r in shuf], 0), ORANGE, "model, ordering scrambled", 7))
    ys = []
    for y, c, t, dy in curves:
        y = np.maximum(y, 1e-5)
        a2.plot(k[1:], y[1:], color=c, lw=1.6 if c != REF else 1.2, ls="-" if c != REF else "--")
        label(a2, k[-1], y[-1], t, c if c != REF else MUTED, dy=dy)
        ys.append(y[1:])
    a2.set_yscale("log")
    a2.set_xlabel("samples seen, k")
    a2.set_ylabel("KL to exact answer (bits)")
    a2.set_title("Error vs. evidence", loc="left")
    range_frame(a2, k[1:], np.concatenate(ys))
    a2.set_xlim(0, k[-1] * 1.75)
    a2.set_xticks([0, 32, 64, 96, 128])
    fig.tight_layout()
    save(fig, "e1_witness")


# ================================================================ E3
def fig_e3() -> None:
    s = json.loads((RES / "summary.json").read_text()).get("e3") if (RES / "summary.json").exists() else None
    if not s or "bins" not in s:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axvspan(-0.3, 0.3, color="#f2f1ed", zorder=0)
    ax.text(0, 1.08, "held out\nin training", ha="center", va="top", color=MUTED, fontsize=7.5)
    ax.plot([-1, 1], [-1, 1], color=REF, lw=1, zorder=1)
    names = {"aware": (BLUE, "ρ given", 0), "shuffled": (ORANGE, "ρ scrambled", 8),
             "l2trained": (AQUA, "trained on interventions only", -8)}
    allx, ally = [-1, 1], []
    for c, (color, text, dy) in names.items():
        b = s["bins"].get(c)
        if not b:
            continue
        x, y, e = np.array(b["rho"]), np.array(b["slope_mean"]), np.array(b["slope_sd"])
        ax.errorbar(x, y, yerr=e, fmt="o", ms=4.5, color=color, ecolor=color, elinewidth=1, capsize=0, zorder=3)
        label(ax, x[-1], y[-1], text, color, dx=7, dy=dy)
        ally += list(y)
    ax.set_xlabel("cross-world correlation ρ (Layer-3 information only)")
    ax.set_ylabel("residual-transfer slope")
    ax.set_title("Counterfactuals track ρ; without it they collapse to the interventional answer", loc="left", fontsize=9.5)
    ax.set_xlim(-1.08, 1.85)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    range_frame(ax, allx, [min(-1, min(ally)), max(1, max(ally))])
    save(fig, "e3_rho")


# ================================================================ E4
def fig_e4() -> None:
    sfile = RES / "summary.json"
    s = json.loads(sfile.read_text()).get("e4") if sfile.exists() else None
    if not s:
        return
    ks = np.array(s["eval_k"])
    sel = ks > 0
    titles = {"obs_conf": "Z observed, confounder", "obs_med": "Z observed, mediator",
              "hid_med": "Z hidden, mediator", "hid_conf": "Z hidden, confounder"}
    fig, axs = plt.subplots(1, 4, figsize=(9, 2.6), sharey=True)
    for ax, (cell, t) in zip(axs, titles.items()):
        c = s["cells"][cell]
        ax.plot(ks[sel], np.array(c["sd_oracle"])[sel], color=REF, lw=1.2, ls="--")
        ax.plot(ks[sel], np.array(c["sd_model"])[sel], "o", color=BLUE, ms=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(t, loc="left", fontsize=9, color=INK if cell != "hid_conf" else ORANGE)
        ax.set_xlabel("samples k")
        range_frame(ax, ks[sel], np.array(c["sd_model"])[sel])
    axs[0].set_ylabel("posterior sd of P(Y=1|do(X=1))")
    so = np.array(s["cells"]["obs_conf"]["sd_oracle"])[sel]
    label(axs[0], ks[sel][3], so[3], "exact Bayes (dashed)", MUTED, dx=-6, dy=-10, ha="right")
    label(axs[0], ks[sel][-1], np.array(s["cells"]["obs_conf"]["sd_model"])[sel][-1], "transformer", BLUE, dy=-9, ha="right")
    fig.tight_layout()
    save(fig, "e4_identifiability")


# ================================================================ E2 / E2c / E5 dot plots
def _dot_rows(ax, rows, x_ref=(0.5, 0.8)) -> None:
    ys = np.arange(len(rows))[::-1]
    for y, (lab, est, color) in zip(ys, rows):
        v, lo, hi = est
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=color, lw=1.2)
        ax.plot([v], [y], "o", color=color, ms=5)
        ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", color=INK2, fontsize=8.5)
    for xr in x_ref:
        ax.axvline(xr, color=REF, lw=0.9, ls="--", zorder=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)


def _e2_balanced_runs(pattern: str) -> list[dict]:
    import re
    out = []
    for rd in sorted((RES / "e2").iterdir()):
        m = re.match(pattern, rd.name)
        if not m or not (rd / "metrics_balanced.json").exists() or not (rd / "metrics.json").exists():
            continue
        js = json.loads((rd / "metrics.json").read_text())
        if js.get("status") != "done":
            continue
        b = json.loads((rd / "metrics_balanced.json").read_text())["final"]
        out.append(dict(name=rd.name, m=m, final=js["final"], bal=b))
    return out


def fig_e2() -> None:
    base = _e2_balanced_runs(r"cpt_f(\d)_seed(\d)$")
    if not base:
        return
    tests = [("single do(), unseen graphs", lambda r: r["final"].get("gc_heldout_do"), BLUE),
             ("single conditioning, unseen graphs", lambda r: r["final"].get("gc_heldout_cond"), AQUA),
             ("two interventions combined", lambda r: np.nanmean([r["bal"].get("gc_composite_train_strict_do", np.nan),
                                                                  r["bal"].get("gc_composite_heldout_strict_do", np.nan)]), BLUE),
             ("two conditions combined", lambda r: np.nanmean([r["bal"].get("gc_composite_train_strict_cond", np.nan),
                                                               r["bal"].get("gc_composite_heldout_strict_cond", np.nan)]), ORANGE)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ys = np.arange(len(tests))[::-1]
    for y, (lab, fn, color) in zip(ys, tests):
        v = np.array([fn(r) for r in base], float)
        v = v[np.isfinite(v)]
        a1.plot(v, y + np.linspace(-0.12, 0.12, len(v)), "o", ms=3, color=color, alpha=0.45, mec="none")
        a1.plot([v.mean()], [y], "|", ms=16, mew=2, color=color)
        a1.text(-0.03, y, lab, transform=a1.get_yaxis_transform(), ha="right", va="center", color=INK2, fontsize=8.5)
    a1.axvline(0.8, color=REF, lw=0.9, ls="--", zorder=0)
    a1.text(0.81, ys[-1] - 0.42, "pass threshold", color=MUTED, fontsize=7.5, ha="left")
    a1.set_yticks([])
    a1.spines["left"].set_visible(False)
    a1.set_xlim(0, 1.03)
    a1.set_xlabel("gap closed (1 = exact answer)")
    a1.set_title("Given the graph and CPTs (15 runs, 5 folds)", loc="left")
    range_frame(a1, [0, 1], [])
    # capacity sweep
    sweep = _e2_balanced_runs(r"cpt_f([01])(?:_d(\d+)L(\d+))?_seed([01])$")
    xs, ys2 = [], []
    for r in sweep:
        d = int(r["m"].group(2) or 256)
        L = int(r["m"].group(3) or 6)
        g = np.nanmean([r["bal"].get("gc_composite_train_strict_cond", np.nan),
                        r["bal"].get("gc_composite_heldout_strict_cond", np.nan)])
        xs.append(12 * L * d * d / 1e6)
        ys2.append(g)
    xs, ys2 = np.array(xs), np.array(ys2)
    a2.plot(xs, ys2, "o", ms=4, color=ORANGE, alpha=0.7, mec="none")
    ux = np.unique(np.round(xs, 1))
    means = [ys2[np.isclose(np.round(xs, 1), u)].mean() for u in ux]
    a2.plot(ux, means, "-", color=ORANGE, lw=1.2)
    a2.axhline(0.8, color=REF, lw=0.9, ls="--", zorder=0)
    a2.set_xscale("log")
    a2.set_ylim(0, 1.0)
    a2.set_xlabel("model parameters (millions)")
    a2.set_ylabel("gap closed, two conditions")
    a2.set_title("More capacity does not fix it", loc="left")
    label(a2, ux[-1], means[-1], "two conditions combined", ORANGE, dy=-10, ha="right")
    range_frame(a2, xs, ys2)
    fig.tight_layout()
    save(fig, "e2_gap_closed")


def main() -> None:
    style()
    for f in (fig_e1, fig_e3, fig_e4, fig_e2):
        try:
            f()
        except Exception as e:  # keep going; report which figure failed
            print(f"[skip] {f.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
