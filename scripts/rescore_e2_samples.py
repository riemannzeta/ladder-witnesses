"""Repair NaN oracle entries in the frozen E2-full (samples-mode) eval sets, then re-score every
finished samples-mode checkpoint on them (PREDICTIONS.md, Addendum 2).

Why: the Monte Carlo L1 oracle and the conditional branch of the graph-blind baseline divided by
the probability of the conditioning event per posterior draw; Beta(1/4, 1/4) rows occasionally
underflow to exactly 0 or 1, giving 0/0 and a NaN entry. Only those entries are recomputed (with
the fixed oracle code); every item and every other entry is unchanged. The original eval-set files
are kept; repaired copies are written next to them.

    python scripts/rescore_e2_samples.py [--folds 0 1 2 3 4] [--device auto] [--only SUBSTR]

If a fold's repaired set already exists it is reused as is, so later checkpoints are scored on
exactly the same entries; --only restricts re-scoring to runs whose name contains SUBSTR.

Writes runs/e2/eval_sets/<orig>_repaired.pt and <orig>_repaired.json (log of repaired entries),
and results/e2/<run>/metrics_rescored.json + eval_preds_rescored.npz for each checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from witnesses.meta import get_device, utc_now
from witnesses.model import TinyGPT
from witnesses.tasks import dag_utils as du
from witnesses.tasks.e2_dags import EVAL_K, E2Dags
from witnesses.train import eval_cache_path, load_config

ROOT = Path(__file__).resolve().parents[1]


def _repair_split(task: E2Dags, ep: dict, pa_key: str, rng, log: list, where: str) -> None:
    q = {k[2:]: ep[k].numpy() for k in ep if k.startswith("q_")}
    pa_all = ep[pa_key].numpy()
    vals = ep["vals"].numpy()
    for field in ("oracle_l2", "oracle_l1", "oracle_blind"):
        arr = ep[field].numpy().copy()
        bad = np.argwhere(np.isnan(arr))
        for b, ki in bad:
            k = EVAL_K[ki]
            pa = tuple(int(x) for x in pa_all[b])
            qb = {kk: v[b:b + 1] for kk, v in q.items()}
            if field == "oracle_l1":
                n1, n0 = task._row_counts(pa, vals[b, :k])
                a = np.array([du.bdeu_ab(bin(p).count("1")) for p in pa])[:, None]
                draws = rng.beta(a + n1, a + n0, size=(task.mc, 4, 4))
                new = task._cond_mc(pa, draws, qb)
            elif field == "oracle_blind":
                new = task._bma(vals[b, :k], qb, rng)
            else:
                raise RuntimeError(f"unexpected NaN in exact oracle {field} at {where} item {b}")
            arr[b, ki] = new
            log.append(dict(split=where, field=field, item=int(b), k=int(k), new_value=float(new)))
        ep[field] = torch.as_tensor(arr, dtype=ep[field].dtype)
        assert not np.isnan(arr).any(), (where, field)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", default=None, help="re-score only runs whose name contains this")
    a = ap.parse_args()
    dev = get_device(a.device)
    for fold in a.folds:
        cfg = load_config(f"configs/e2/samples_f{fold}.yaml", 0, a.device, False, [])
        task = E2Dags(**cfg["task_args"])
        path = eval_cache_path(cfg, task)
        rpath = path.with_name(path.stem + "_repaired.pt")
        if rpath.exists():
            sets = torch.load(rpath, weights_only=False)
            log = json.loads(rpath.with_suffix(".json").read_text())["repaired"]
            print(f"fold {fold}: reusing {rpath.name} ({len(log)} repaired entries)", flush=True)
        else:
            sets = torch.load(path, weights_only=False)
            rng = np.random.default_rng(20261008 + fold)
            log = []
            for name, ep in sets.items():
                if name == "swap":
                    _repair_split(task, ep["orig"], "pa", rng, log, "swap/orig")
                    _repair_split(task, ep["alt"], "pa_alt", rng, log, "swap/alt")
                else:
                    _repair_split(task, ep, "pa", rng, log, name)
            torch.save(sets, rpath)
            rpath.with_suffix(".json").write_text(json.dumps(
                {"source": path.name, "fold": fold, "updated_utc": utc_now(), "repaired": log}, indent=1))
            print(f"fold {fold}: {path.name} -> {rpath.name}, repaired {len(log)} entries", flush=True)

        ckpts = sorted((ROOT / "runs" / "e2").glob(f"samples_f{fold}*_seed*/model.pt"))
        for ck in [c for c in ckpts if "smoke" not in str(c) and (a.only is None or a.only in c.parent.name)]:
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            model = TinyGPT(task.vocab_size, task.max_len, {k: v.dim for k, v in task.heads.items()},
                            cont_dim=task.cont_dim, **blob["config"]["model"]).to(dev)
            model.load_state_dict(blob["model"])
            model.eval()
            res, preds = task.evaluate(model, sets, dev, return_preds=True)
            rd = ROOT / "results" / "e2" / ck.parent.name
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "metrics_rescored.json").write_text(json.dumps(
                {"source": "scripts/rescore_e2_samples.py", "eval_set": rpath.name,
                 "n_repaired_entries": len(log), "updated_utc": utc_now(), "final": res},
                indent=1, default=float))
            np.savez_compressed(rd / "eval_preds_rescored.npz", **preds)
            print(f"  {ck.parent.name}: composite cond GC train/held-out "
                  f"{res.get('gc_composite_train_cond', float('nan')):.3f}/"
                  f"{res.get('gc_composite_heldout_cond', float('nan')):.3f}  "
                  f"KL {res.get('kl_composite_train_cond', float('nan')):.4f}/"
                  f"{res.get('kl_composite_heldout_cond', float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
