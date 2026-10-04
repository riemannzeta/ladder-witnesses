"""Measure training throughput per experiment config (no evaluation) and project wave durations.

Usage: python scripts/benchmark.py [--steps 300] [--configs configs/e1/aware.yaml ...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from witnesses.meta import get_device, set_seed
from witnesses.model import TinyGPT
from witnesses.tasks import get_task
from witnesses.tasks.base import head_loss
from witnesses.train import make_optimizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["configs/e1/aware.yaml", "configs/e3/aware.yaml", "configs/e4/main.yaml",
           "configs/e2/cpt_f0.yaml", "configs/e2/samples_f0.yaml", "configs/e2c/main.yaml",
           "configs/e5/restart.yaml", "configs/e5/cf_abducted.yaml"]


def bench(cfg_path: str, steps: int, device) -> dict:
    cfg = yaml.safe_load((ROOT / cfg_path).read_text())
    set_seed(0)
    task = get_task(cfg["task"], **cfg.get("task_args", {}))
    model = TinyGPT(task.vocab_size, task.max_len, {k: v.dim for k, v in task.heads.items()},
                    cont_dim=task.cont_dim, **cfg["model"]).to(device)
    opt = make_optimizer(model, {"lr": 3e-4, "weight_decay": 0.01})
    bs = cfg.get("optim", {}).get("batch_size", 256)
    amp = device.type == "cuda"
    t_data = t_all = 0.0
    for i in range(steps + 20):
        if i == 20:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            t_data = 0.0
        td = time.time()
        batch = task.sample_batch(bs, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_data += time.time() - td
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(batch["tokens"], batch.get("cont"))
        loss = sum(s.weight * head_loss(s, out[n], batch["targets"][n], batch["masks"][n])
                   for n, s in task.heads.items() if n in batch["targets"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_all = time.time() - t0
    return dict(config=cfg_path, params=sum(p.numel() for p in model.parameters()),
                steps_per_s=steps / t_all, tokens_per_s=steps * bs * task.max_len / t_all,
                data_frac=t_data / t_all, batch=bs, seq=task.max_len,
                base_steps=cfg.get("schedule", {}).get("base_steps"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--configs", nargs="*", default=DEFAULT)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = get_device(a.device)
    rows = []
    for c in a.configs:
        if not (ROOT / c).exists():
            print(f"skip missing {c}")
            continue
        r = bench(c, a.steps, dev)
        rows.append(r)
        print(f"{c:34s} {r['params'] / 1e6:5.2f}M  {r['steps_per_s']:7.1f} it/s  "
              f"{r['tokens_per_s'] / 1e6:6.2f} Mtok/s  data {100 * r['data_frac']:4.1f}%", flush=True)
    out = ROOT / "runs" / "benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"device": str(dev), "gpu": torch.cuda.get_device_name(0)
                               if dev.type == "cuda" else None, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
