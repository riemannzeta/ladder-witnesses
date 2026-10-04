"""Resumable job queue: run (config, seed) jobs with N concurrent workers on one GPU.

Each job is `python -m witnesses.train --config C --seed S`; the trainer itself skips finished
runs (status == done) and resumes from its checkpoint, so re-launching a queue is always safe.
A failed job is retried once; failures are recorded in runs/queues/<name>/_FAILED and never block
other jobs. When the queue drains, runs/queues/<name>/_DONE is written and, with --shutdown, the
machine is powered off (on EC2 an EBS-backed instance then STOPS, which ends billing).

Queue file (YAML):
    name: wave1
    workers: 3
    jobs:
      - {config: configs/e1/aware.yaml, seeds: [0, 1, 2, 3, 4]}
      - {config: configs/e2/cpt_f0.yaml, seeds: [0, 1, 2], set: ["schedule.base_steps=40000"]}

Usage:  python scripts/run_queue.py configs/queues/wave1.yaml [--workers 3] [--shutdown]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str, fh) -> None:
    line = f"[{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def job_done(cfg_path: str, seed: int, overrides: list[str]) -> bool:
    cfg = yaml.safe_load((ROOT / cfg_path).read_text())
    cond = cfg["condition"]
    for ov in overrides:
        if ov.startswith("condition="):
            cond = ov.split("=", 1)[1]
    m = ROOT / "results" / cfg["exp"] / f"{cond}_seed{seed}" / "metrics.json"
    return m.exists() and json.loads(m.read_text()).get("status") == "done"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--shutdown", action="store_true")
    a = ap.parse_args()
    q = yaml.safe_load(Path(a.queue).read_text())
    name = q["name"]
    workers = a.workers or q.get("workers", 2)
    qdir = ROOT / "runs" / "queues" / name
    qdir.mkdir(parents=True, exist_ok=True)
    for f in ("_DONE", "_FAILED"):
        (qdir / f).unlink(missing_ok=True)
    jobs = [(j["config"], s, j.get("set", [])) for j in q["jobs"] for s in j["seeds"]]
    lock = threading.Lock()
    failures: list[str] = []
    fh = open(qdir / "queue.log", "a")
    log(f"queue {name}: {len(jobs)} jobs, {workers} workers", fh)

    def worker(wid: int) -> None:
        while True:
            with lock:
                if not jobs:
                    return
                cfg, seed, ov = jobs.pop(0)
            tag = f"{Path(cfg).parent.name}_{Path(cfg).stem}_s{seed}" + ("_" + "_".join(o.split("=")[0].split(".")[-1] + o.split("=")[1] for o in ov) if ov else "")
            if job_done(cfg, seed, ov):
                log(f"[w{wid}] skip (done) {tag}", fh)
                continue
            for attempt in (1, 2):
                logf = ROOT / "runs" / "logs" / f"{tag}.log"
                logf.parent.mkdir(parents=True, exist_ok=True)
                log(f"[w{wid}] start {tag} (attempt {attempt})", fh)
                t0 = time.time()
                cmd = [sys.executable, "-m", "witnesses.train", "--config", cfg, "--seed", str(seed)]
                if ov:
                    cmd += ["--set", *ov]
                with open(logf, "a") as lf:
                    rc = subprocess.call(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
                log(f"[w{wid}] end {tag} rc={rc} ({(time.time() - t0) / 60:.1f} min)", fh)
                if rc == 0:
                    break
            else:
                with lock:
                    failures.append(tag)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
        time.sleep(20)  # stagger starts (eval-set builds, GPU warmup)
    for t in threads:
        t.join()
    summary = {"name": name, "failures": failures,
               "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    (qdir / ("_FAILED" if failures else "_DONE")).write_text(json.dumps(summary, indent=2))
    log(f"queue {name} finished; failures: {failures}", fh)
    if a.shutdown:
        log("shutting down in 2 minutes", fh)
        subprocess.call(["sudo", "shutdown", "-h", "+2"])


if __name__ == "__main__":
    main()
