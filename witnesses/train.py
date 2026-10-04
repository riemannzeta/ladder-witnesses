"""Generic trainer with convergence-based stopping.

Schedule is warmup -> stable (constant LR) -> linear cooldown ("WSD"). The cooldown starts when
the headline eval metric (lower is better) has improved by less than `plateau_rel` (relative)
over the last `plateau_window` fraction of steps, after at least `min_steps`; the run is then
marked converged. If the plateau never comes, the cooldown is placed so that the run ends at the
step cap (`base_steps * cap_mult`) and the run is marked *unconverged* -- such runs are never
counted as negative results downstream.

Usage:
    python -m witnesses.train --config configs/e1/aware.yaml --seed 0 [--device auto] [--smoke]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from witnesses.meta import REPO_ROOT, get_device, run_metadata, set_seed, utc_now
from witnesses.model import TinyGPT
from witnesses.tasks import get_task
from witnesses.tasks.base import head_loss

DEFAULT_SCHEDULE = dict(
    base_steps=30_000,
    cap_mult=4,
    min_steps=10_000,
    eval_every=2_000,
    plateau_window=0.25,
    plateau_rel=0.01,
    cooldown_frac=0.2,
    cooldown_min=2_000,
)
DEFAULT_OPTIM = dict(lr=3e-4, weight_decay=0.01, batch_size=256, warmup=1_000, grad_clip=1.0)

SMOKE_OVERRIDES = dict(
    schedule=dict(base_steps=300, cap_mult=2, min_steps=200, eval_every=100, cooldown_min=50),
    optim=dict(batch_size=64, warmup=50),
    eval=dict(n_episodes=256),
)


def deep_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def load_config(path: str, seed: int, device: str, smoke: bool, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("schedule", {})
    cfg.setdefault("optim", {})
    cfg.setdefault("eval", {})
    cfg["schedule"] = {**DEFAULT_SCHEDULE, **cfg["schedule"]}
    cfg["optim"] = {**DEFAULT_OPTIM, **cfg["optim"]}
    cfg["eval"] = {"seed": 1_000_003, "n_episodes": 10_000, **cfg["eval"]}
    if smoke:
        deep_update(cfg, copy.deepcopy(SMOKE_OVERRIDES))
        cfg["model"] = {**cfg.get("model", {}), **cfg.get("smoke_model", {})}
        for k in list(cfg.get("task_args", {})):
            if k.startswith("n_eval_per"):
                cfg["task_args"][k] = 64
    for ov in overrides:  # dotted.key=value (yaml-parsed value)
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    cfg["seed"] = seed
    cfg["device"] = device
    cfg["smoke"] = smoke
    cfg["config_path"] = str(path)
    tag = f"{cfg['condition']}_seed{seed}" + ("_smoke" if smoke else "")
    cfg["results_dir"] = str(REPO_ROOT / "results" / cfg["exp"] / tag)
    cfg["ckpt_dir"] = str(REPO_ROOT / "runs" / cfg["exp"] / tag)
    return cfg


def eval_cache_path(cfg: dict, task) -> Path:
    sig = json.dumps(
        {"task": cfg["task"], "sig": task.eval_signature(), "eval": cfg["eval"]}, sort_keys=True
    )
    h = hashlib.sha1(sig.encode()).hexdigest()[:12]
    return REPO_ROOT / "runs" / cfg["exp"] / "eval_sets" / f"{cfg['task']}_{h}.pt"


def get_eval_sets(cfg: dict, task) -> tuple[dict, str]:
    path = eval_cache_path(cfg, task)
    if path.exists():
        return torch.load(path, weights_only=False), path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        sets = task.make_eval_sets(seed=cfg["eval"]["seed"], n_episodes=cfg["eval"]["n_episodes"])
    tmp = path.with_suffix(".tmp")
    torch.save(sets, tmp)
    tmp.rename(path)
    return sets, path.name


def make_optimizer(model: torch.nn.Module, ocfg: dict) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and "emb" not in name else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": ocfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=ocfg["lr"], betas=(0.9, 0.95))


def lr_at(step: int, state: dict, ocfg: dict) -> float:
    base = ocfg["lr"]
    if step < ocfg["warmup"]:
        return base * (step + 1) / ocfg["warmup"]
    if state["phase"] == "cooldown":
        frac = (step - state["cooldown_start"]) / max(1, state["cooldown_len"])
        return base * max(0.0, 1.0 - frac)
    return base


def plateaued(history: list[tuple[int, float]], step: int, window: float, rel: float) -> bool:
    cut = step * (1.0 - window)
    prior = [v for s, v in history if s <= cut]
    recent = [v for s, v in history if s > cut]
    if not prior or not recent:
        return False
    best_prior, best_recent = min(prior), min(recent)
    return best_recent > best_prior - rel * abs(best_prior)


def train(cfg: dict) -> dict:
    results_dir, ckpt_dir = Path(cfg["results_dir"]), Path(cfg["ckpt_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mpath = results_dir / "metrics.json"
    if mpath.exists() and json.loads(mpath.read_text()).get("status") == "done":
        print(f"[skip] {results_dir} already done")
        return json.loads(mpath.read_text())

    device = get_device(cfg["device"])
    set_seed(cfg["seed"])
    task = get_task(cfg["task"], **cfg.get("task_args", {}))
    eval_sets, eval_name = get_eval_sets(cfg, task)
    mcfg = cfg["model"]
    model = TinyGPT(
        vocab_size=task.vocab_size,
        max_len=task.max_len,
        heads={k: v.dim for k, v in task.heads.items()},
        cont_dim=task.cont_dim,
        **mcfg,
    ).to(device)
    ocfg, scfg = cfg["optim"], cfg["schedule"]
    opt = make_optimizer(model, ocfg)
    cap = scfg["base_steps"] * scfg["cap_mult"]
    state = dict(step=0, phase="main", cooldown_start=None, cooldown_len=None, converged=None,
                 stop_reason=None, history=[], log=[], train_time=0.0)
    meta = run_metadata(cfg, device)
    meta.update(n_params=sum(p.numel() for p in model.parameters()), eval_set=eval_name)

    ck = ckpt_dir / "ckpt.pt"
    if ck.exists():
        blob = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        state = blob["state"]
        torch.set_rng_state(blob["rng_cpu"])
        if device.type == "cuda" and blob.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(blob["rng_cuda"])
        print(f"[resume] step {state['step']}")
    (results_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    amp = device.type == "cuda"
    if cfg.get("compile") and device.type == "cuda":
        model = torch.compile(model)
    bs = ocfg["batch_size"]
    tokens_per_step = bs * task.max_len
    run_loss, run_n, t0 = 0.0, 0, time.time()

    def save_ckpt():
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        blob = dict(model=raw.state_dict(), opt=opt.state_dict(), state=state,
                    rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state() if device.type == "cuda" else None)
        tmp = ck.with_suffix(".tmp")
        torch.save(blob, tmp)
        tmp.rename(ck)

    def write_metrics(status: str, final: dict | None = None):
        out = dict(status=status, exp=cfg["exp"], condition=cfg["condition"], seed=cfg["seed"],
                   task=cfg["task"], step=state["step"], converged=state["converged"],
                   stop_reason=state["stop_reason"], cap_steps=cap, log=state["log"],
                   train_time_s=state["train_time"], final=final, updated_utc=utc_now())
        tmp = mpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2, default=float))
        tmp.rename(mpath)

    while True:
        step = state["step"]
        lr = lr_at(step, state, ocfg)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = task.sample_batch(bs, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(batch["tokens"], batch.get("cont"))
        loss = sum(
            spec.weight * head_loss(spec, out[name], batch["targets"][name], batch["masks"][name])
            for name, spec in task.heads.items()
            if name in batch["targets"]
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["grad_clip"])
        opt.step()
        state["step"] = step = step + 1
        run_loss += float(loss.detach())
        run_n += 1

        done = state["phase"] == "cooldown" and step >= state["cooldown_start"] + state["cooldown_len"]
        if step % scfg["eval_every"] == 0 or done:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            state["train_time"] += dt
            metrics = task.evaluate(model, eval_sets, device)
            h = metrics[task.headline]
            state["history"].append((step, h))
            state["log"].append(dict(step=step, lr=lr, train_loss=run_loss / max(1, run_n),
                                     steps_per_s=run_n / max(dt, 1e-9),
                                     tokens_per_s=run_n * tokens_per_step / max(dt, 1e-9),
                                     **metrics))
            print(f"[{cfg['exp']}/{cfg['condition']}/s{cfg['seed']}] step {step} lr {lr:.2e} "
                  f"loss {run_loss / max(1, run_n):.4f} {task.headline} {h:.5f} "
                  f"({run_n / max(dt, 1e-9):.1f} it/s)", flush=True)
            run_loss, run_n = 0.0, 0
            if state["phase"] == "main" and not done:
                if step >= scfg["min_steps"] and plateaued(
                    state["history"], step, scfg["plateau_window"], scfg["plateau_rel"]
                ):
                    state.update(phase="cooldown", cooldown_start=step, converged=True,
                                 stop_reason="plateau",
                                 cooldown_len=max(scfg["cooldown_min"],
                                                  int(scfg["cooldown_frac"] * step)))
                else:
                    cl = max(scfg["cooldown_min"], int(scfg["cooldown_frac"] * cap))
                    if step >= cap - cl:
                        state.update(phase="cooldown", cooldown_start=step, converged=False,
                                     stop_reason="cap", cooldown_len=max(1, cap - step))
            if done:
                break
            save_ckpt()
            write_metrics("running")
            t0 = time.time()

    final, preds = task.evaluate(model, eval_sets, device, return_preds=True)
    np.savez_compressed(results_dir / "eval_preds.npz", **preds)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model": raw.state_dict(), "config": cfg}, ckpt_dir / "model.pt")
    write_metrics("done", final)
    if ck.exists():
        ck.unlink()
    print(f"[done] {results_dir} converged={state['converged']} {task.headline}={final[task.headline]:.5f}")
    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="dotted.key=value overrides")
    a = ap.parse_args()
    train(load_config(a.config, a.seed, a.device, a.smoke, a.set))


if __name__ == "__main__":
    main()
