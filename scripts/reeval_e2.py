"""Re-score finished E2 checkpoints on BALANCED combined-query test sets.

Why: the pre-registered combined-query split draws do() and conditioning queries 50/50 and then
applies the pre-registered acceptance rules (both elements move the answer by > 0.05, and
conditioning differs from intervening by > 0.05). In 4-node graphs with in-degree <= 2 those rules
reject almost every two-intervention query, so the frozen split holds 0-53 do() items per fold.
This script draws each query type to its own quota under exactly the same rules, then evaluates
every finished checkpoint on it. No retraining; the frozen splits are left untouched and still
reported.

    python scripts/reeval_e2.py --mode cpt --n-per-type 1000 [--folds 0 1 2 3 4] [--device auto] [--only comp25]

Writes results/e2/<cond>_seed<k>/metrics_balanced.json (+ balanced_preds.npz).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from witnesses.meta import get_device, utc_now
from witnesses.model import TinyGPT
from witnesses.tasks.base import kl_bernoulli_bits
from witnesses.tasks.e2_dags import E2Dags

ROOT = Path(__file__).resolve().parents[1]
STRICT_DO_TOPOS = (20, 24)


def balanced_sets(task: E2Dags, n: int, seed: int, max_rounds: int = 40) -> dict:
    """Combined-query test sets with up to n do() and n conditioning items each.

    strict  = the pre-registered rules (both elements move the answer by > 0.05 AND conditioning
              differs from intervening by > 0.05). Only 2 of 25 topologies (T20, T24) admit such
              do() queries at all, so some held-out sets have no do() items; we report the counts.
    relaxed = exploratory: both elements must move the answer, no do-vs-conditioning requirement.
    """
    path = ROOT / "runs" / "e2" / "eval_sets" / f"balanced2_{task.mode}_f{task.fold}_n{n}_s{seed}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    torch.manual_seed(seed)
    cpu = torch.device("cpu")
    out = {}
    for rule, require_disc in (("strict", True), ("relaxed", False)):
        for split, topos in (("train", task.train_topos), ("heldout", task.heldout_topos)):
            parts = []
            for p_cond in (0.0, 1.0):  # do() quota, then conditioning quota
                task.p_cond = p_cond
                tp = topos
                if require_disc and p_cond == 0.0:
                    # only these topologies can satisfy the strict do() rules at all (measured: 0
                    # acceptances in 4,000 draws for every other topology); restricting the draw
                    # changes efficiency, not the acceptance rules
                    tp = [t for t in topos if t in STRICT_DO_TOPOS]
                    if not tp:
                        print(f"  fold {task.fold} composite_{split}_{rule}: no topology admits strict do() queries")
                        continue
                part = task._make_split(n, cpu, tp, "composite", max_rounds=max_rounds,
                                        require_disc=require_disc)
                if part:
                    parts.append(part)
            task.p_cond = 0.5
            if not parts:
                continue
            merged = {}
            for k in parts[0]:
                vals = [pt[k] for pt in parts]
                merged[k] = torch.cat(vals) if isinstance(vals[0], torch.Tensor) else vals[0]
            name = f"composite_{split}_{rule}"
            out[name] = merged
            isc = merged["q_is_cond"].numpy()
            print(f"  fold {task.fold} {name}: {(~isc).sum()} do() + {isc.sum()} conditioning items", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("cpt", "samples"), default="cpt")
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-per-type", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2_000_003)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", default=None, help="re-score only checkpoints whose run name contains this")
    a = ap.parse_args()
    dev = get_device(a.device)
    for fold in a.folds:
        ckpts = sorted((ROOT / "runs" / "e2").glob(f"{a.mode}_f{fold}*_seed*/model.pt"))
        ckpts = [c for c in ckpts if "smoke" not in str(c) and (a.only is None or a.only in c.parent.name)]
        if not ckpts:
            print(f"fold {fold}: no finished checkpoints yet")
            continue
        cfg0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)["config"]
        task = E2Dags(**{k: v for k, v in cfg0["task_args"].items() if k != "n_eval_per_split"},
                      n_eval_per_split=a.n_per_type)
        sets = balanced_sets(task, a.n_per_type, a.seed)
        for ck in ckpts:
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            cfg = blob["config"]
            model = TinyGPT(task.vocab_size, task.max_len, {k: v.dim for k, v in task.heads.items()},
                            cont_dim=task.cont_dim, **cfg["model"]).to(dev)
            model.load_state_dict(blob["model"])
            res, preds = {}, {}
            for name, ep in sets.items():
                task._split_metrics(model, ep, dev, name, res, preds)
                isc = ep["q_is_cond"].numpy()
                res[f"n_{name}_do"], res[f"n_{name}_cond"] = int((~isc).sum()), int(isc.sum())
                # structure of the two pinned variables relative to the target's parents
                pm = preds[f"{name}_model"][:, -1].astype(np.float64)
                pa, A, B, T = (ep[k].numpy() for k in ("pa", "q_A", "q_B", "q_T"))
                both = np.array([(pa[i, T[i]] >> A[i] & 1) and (pa[i, T[i]] >> B[i] & 1) for i in range(len(A))], bool)
                # graph-aware Bayes answers at the last k (= the exact answers in cpt mode)
                right = np.where(isc, preds[f"{name}_l1"][:, -1], preds[f"{name}_l2"][:, -1]).astype(np.float64)
                kl = kl_bernoulli_bits(right, pm)
                for qn, qs in (("do", ~isc), ("cond", isc)):
                    for tag, ss in (("both_parents", both), ("other", ~both)):
                        sel = qs & ss
                        res[f"kl_{name}_{qn}_{tag}"] = float(kl[sel].mean()) if sel.any() else float("nan")
                        res[f"n_{name}_{qn}_{tag}"] = int(sel.sum())
            rd = ROOT / "results" / "e2" / ck.parent.name
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "metrics_balanced.json").write_text(json.dumps(
                {"n_per_type": a.n_per_type, "eval_seed": a.seed, "updated_utc": utc_now(), "final": res},
                indent=1, default=float))
            np.savez_compressed(rd / "balanced_preds.npz", **preds)
            g = lambda k: res.get(k, float("nan"))  # noqa: E731
            print(f"{ck.parent.name}: strict do GC train/held-out {g('gc_composite_train_strict_do'):.3f}/"
                  f"{g('gc_composite_heldout_strict_do'):.3f} | strict cond {g('gc_composite_train_strict_cond'):.3f}/"
                  f"{g('gc_composite_heldout_strict_cond'):.3f} | relaxed do {g('kl_composite_heldout_relaxed_do'):.4f} bits",
                  flush=True)


if __name__ == "__main__":
    main()
