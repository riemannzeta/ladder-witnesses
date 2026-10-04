"""E6 runner: build the Message Batches requests, estimate their cost, submit them, collect results.

    python -m witnesses.llm_probe.run_batch --config configs/e6/probe.yaml [--pilot K]
        [--models M ...] [--include-fable]
        (--dry-run | --submit | --collect [BATCH_ID ...])
        [--env-file PATH] [--max-cost USD] [--out-dir DIR]

--dry-run   writes items (results/e6/items/) and requests (results/e6/requests/<model>.jsonl) and
            prints an offline cost ESTIMATE. It never imports the SDK and never touches the network.
--submit    same, then refuses if the estimate exceeds --max-cost (default: config max_cost_usd);
            otherwise creates one batch per model and records the ids in results/e6/batches.json.
--collect   polls the given batches (default: the latest submission in batches.json) until they
            end, streams their results and writes results/e6/responses/<model>.jsonl.

The API key is read ONLY from --env-file PATH (just its ANTHROPIC_API_KEY= line) or, failing that,
from the ANTHROPIC_API_KEY environment variable, and is passed to the client explicitly (so the
SDK consults no other credential source). The key is never printed or logged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from witnesses.llm_probe.items import CONDITIONS, generate_items, write_items
from witnesses.llm_probe.prompts import build_request, decode_custom_id, parse_answer


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_models(cfg: dict, models: list[str] | None, include_fable: bool) -> list[str]:
    known = cfg["models"]
    chosen = list(models) if models else [m for m, mc in known.items() if not mc.get("opt_in")]
    unknown = [m for m in chosen if m not in known]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; configured: {list(known)}")
    if include_fable:
        chosen += [m for m, mc in known.items() if mc.get("opt_in") and m not in chosen]
    else:
        opted = [m for m in chosen if known[m].get("opt_in")]
        if opted:
            raise SystemExit(f"{opted} require --include-fable")
    return chosen


def build_requests(items: dict[str, list[dict]], model: str, cfg: dict) -> list[dict]:
    mc = cfg["models"][model]
    return [build_request(it, model, mc, cfg["max_tokens"])
            for probe in items for it in items[probe]]


def estimate_cost(requests: list[dict], model: str, cfg: dict) -> dict:
    """Offline heuristic: input = chars / chars_per_token (system + user + schema); output = a
    fixed per-item guess for the model; batch price = batch_discount x list price."""
    mc = cfg["models"][model]
    chars = 0
    for r in requests:
        p = r["params"]
        chars += len(p["system"]) + sum(len(m["content"]) for m in p["messages"])
        chars += len(json.dumps(p["output_config"]["format"]["schema"]))
    in_tok = chars / float(cfg["chars_per_token"])
    out_tok = len(requests) * float(mc["est_output_tokens"])
    p_in, p_out = (float(v) for v in mc["price_per_mtok"])
    cost = (in_tok * p_in + out_tok * p_out) / 1e6 * float(cfg["batch_discount"])
    return {"model": model, "n_requests": len(requests), "input_tokens": in_tok,
            "output_tokens": out_tok, "cost_usd": cost}


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------- credentials / client


def load_api_key(env_file: str | None) -> tuple[str, str]:
    """(key, description of its source). Never prints the key."""
    if env_file:
        key = None
        for line in Path(env_file).read_text().splitlines():
            s = line.strip()
            if s.startswith("export "):
                s = s[len("export "):].lstrip()
            if s.startswith("ANTHROPIC_API_KEY="):
                v = s.split("=", 1)[1].strip()
                if v[:1] in ("'", '"'):
                    v = v[1:].split(v[0], 1)[0]
                else:
                    v = v.split(" #", 1)[0].strip()
                key = v
        if not key:
            raise SystemExit(f"no ANTHROPIC_API_KEY= line in {env_file}")
        return key, f"--env-file {env_file}"
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("no API key: set ANTHROPIC_API_KEY or pass --env-file PATH")
    return key, "the ANTHROPIC_API_KEY environment variable"


def make_client(env_file: str | None):
    import anthropic  # imported here so --dry-run never loads the SDK

    key, source = load_api_key(env_file)
    print(f"using the API key from {source}")
    # An explicit api_key means the SDK consults no other credential source.
    return anthropic.Anthropic(api_key=key)


# ---------------------------------------------------------------- submit / collect


def submit(client, requests_by_model: dict[str, list[dict]], record: dict, batches_path: Path):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    log = json.loads(batches_path.read_text()) if batches_path.exists() else {"submissions": []}
    log["submissions"].append(record)
    for model, reqs in requests_by_model.items():
        batch = client.messages.batches.create(requests=[
            Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**r["params"]))
            for r in reqs
        ])
        record["batches"][model] = {"batch_id": batch.id, "n_requests": len(reqs),
                                    "processing_status": batch.processing_status}
        batches_path.write_text(json.dumps(log, indent=2) + "\n")  # persist after every batch
        print(f"  {model}: batch {batch.id} ({len(reqs)} requests, {batch.processing_status})")


def wait_for_batch(client, batch_id: str, poll_interval: float, no_wait: bool):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended" or no_wait:
            return batch
        rc = batch.request_counts
        print(f"  {batch_id}: {batch.processing_status} (processing {rc.processing}, "
              f"succeeded {rc.succeeded}, errored {rc.errored}); next poll in {poll_interval:.0f}s",
              flush=True)
        time.sleep(poll_interval)


def result_record(result, batch_id: str) -> dict:
    """One results-stream entry -> a flat JSON record (raw text, parsed JSON, usage, stop_reason)."""
    probe, condition, item_id, model = decode_custom_id(result.custom_id)
    rec = {"custom_id": result.custom_id, "batch_id": batch_id, "probe": probe,
           "condition": condition, "item_id": item_id, "model": model,
           "result_type": result.result.type, "category": None, "stop_reason": None,
           "stop_details": None, "text": None, "thinking": None, "parsed": None,
           "parse_error": None, "usage": None, "error": None, "message_id": None,
           "served_model": None}
    rtype = result.result.type
    if rtype == "succeeded":
        msg = result.result.message
        texts = [b.text for b in msg.content if b.type == "text"]
        thinking = [b.thinking for b in msg.content if b.type == "thinking" and b.thinking]
        rec.update(message_id=msg.id, served_model=msg.model, stop_reason=msg.stop_reason,
                   stop_details=msg.stop_details.to_dict() if msg.stop_details is not None else None,
                   usage=msg.usage.to_dict(), text="".join(texts) if texts else None,
                   thinking="\n\n".join(thinking) if thinking else None)
        if msg.stop_reason == "refusal":
            rec["category"] = "refusal"
        else:
            parsed, err = parse_answer(probe, texts[0] if texts else "")
            rec.update(parsed=parsed, parse_error=err,
                       category="ok" if parsed is not None else "parse_failure")
    elif rtype == "errored":
        rec["error"] = result.result.error.to_dict()
        rec["category"] = "errored"
    else:  # canceled / expired: never sent to the model, not billed
        rec["category"] = rtype
    return rec


def collect(client, batch_ids: list[str], out_dir: Path, poll_interval: float, no_wait: bool):
    by_model: dict[str, list[dict]] = defaultdict(list)
    for bid in batch_ids:
        batch = wait_for_batch(client, bid, poll_interval, no_wait)
        if batch.processing_status != "ended":
            print(f"  {bid}: {batch.processing_status}; not collected (--no-wait)")
            continue
        n = 0
        for result in client.messages.batches.results(bid):  # any order: keyed by custom_id
            rec = result_record(result, bid)
            by_model[rec["model"]].append(rec)
            n += 1
        print(f"  {bid}: {n} results")
    for model, recs in by_model.items():
        recs.sort(key=lambda r: r["custom_id"])
        path = out_dir / "responses" / f"{model}.jsonl"
        write_jsonl(path, recs)
        cats = Counter(r["category"] for r in recs)
        stops = Counter(r["stop_reason"] for r in recs if r["stop_reason"])
        print(f"{model}: {len(recs)} records -> {path}; categories {dict(cats)}; "
              f"stop_reason {dict(stops)}")


# ---------------------------------------------------------------- main


def _print_items_summary(items: dict[str, list[dict]], n: int, seed: int, items_dir: Path):
    print(f"E6 items: seed {seed}, {n} contexts per condition -> {items_dir}")
    for probe, its in items.items():
        for cond in CONDITIONS[probe]:
            sel = [it for it in its if it["condition"] == cond]
            extra = ""
            if probe == "V1" and cond == "stated":
                ctx = {it["context_id"] for it in sel}
                n_rev = len({it["context_id"] for it in sel if it["reversal"]})
                extra = (f" ({len(ctx)} contexts x {len(sel) // max(1, len(ctx))} orderings; "
                         f"{n_rev} Simpson-reversal contexts)")
            elif probe == "V2":
                extra = f" ({sum(it['crossing'] for it in sel)} crossing)"
            print(f"  {probe} {cond:<10} {len(sel):4d} prompts{extra}")


def _print_cost_table(estimates: list[dict], cfg: dict, max_cost: float) -> float:
    print(f"\nCost ESTIMATE (offline heuristic, no API call: input tokens = chars/"
          f"{cfg['chars_per_token']}; output tokens = per-model guess per item; Batches API = "
          f"{cfg['batch_discount']:.0%} of list price)")
    print(f"  {'model':<20}{'requests':>9}{'in_tok':>12}{'out_tok':>12}{'$/MTok in/out':>15}"
          f"{'est_usd':>10}")
    total = 0.0
    for e in estimates:
        p_in, p_out = cfg["models"][e["model"]]["price_per_mtok"]
        print(f"  {e['model']:<20}{e['n_requests']:>9}{e['input_tokens']:>12,.0f}"
              f"{e['output_tokens']:>12,.0f}{f'{p_in:g}/{p_out:g}':>15}{e['cost_usd']:>10.2f}")
        total += e["cost_usd"]
    print(f"  {'TOTAL':<20}{sum(e['n_requests'] for e in estimates):>9}{'':>39}{total:>10.2f}")
    print(f"  (--submit refuses above --max-cost = ${max_cost:.2f})")
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--pilot", type=int, metavar="K", help="K contexts per condition")
    ap.add_argument("--models", nargs="+", metavar="MODEL")
    ap.add_argument("--include-fable", action="store_true")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--collect", nargs="*", metavar="BATCH_ID")
    ap.add_argument("--env-file", metavar="PATH")
    ap.add_argument("--max-cost", type=float, metavar="USD")
    ap.add_argument("--out-dir", help="default: config out_dir")
    ap.add_argument("--poll-interval", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true", help="--collect: do not wait for batches")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir or cfg["out_dir"])
    batches_path = out_dir / "batches.json"

    if args.collect is not None:
        ids = list(args.collect)
        if not ids:
            if not batches_path.exists():
                raise SystemExit(f"no batch ids given and no {batches_path}")
            last = json.loads(batches_path.read_text())["submissions"][-1]
            ids = [b["batch_id"] for b in last["batches"].values()]
        client = make_client(args.env_file)
        collect(client, ids, out_dir, args.poll_interval, args.no_wait)
        return 0

    n = args.pilot if args.pilot is not None else int(cfg["n_per_condition"])
    if n < 1:
        raise SystemExit("--pilot must be >= 1")
    models = resolve_models(cfg, args.models, args.include_fable)
    max_cost = float(args.max_cost if args.max_cost is not None else cfg["max_cost_usd"])

    items = generate_items(cfg, n)
    items_dir = out_dir / "items"
    write_items(items, items_dir, {"seed": cfg["seed"], "n_per_condition": n,
                                   "pilot": args.pilot is not None,
                                   "config": {k: cfg[k] for k in ("v1", "v2", "v3")}})
    _print_items_summary(items, n, cfg["seed"], items_dir)

    requests_by_model, estimates = {}, []
    for model in models:
        reqs = build_requests(items, model, cfg)
        write_jsonl(out_dir / "requests" / f"{model}.jsonl", reqs)
        requests_by_model[model] = reqs
        estimates.append(estimate_cost(reqs, model, cfg))
    print(f"requests -> {out_dir / 'requests'}/<model>.jsonl")
    total = _print_cost_table(estimates, cfg, max_cost)

    if args.dry_run:
        print("\n--dry-run: nothing was sent.")
        return 0

    if total > max_cost:
        print(f"\nREFUSED: estimated ${total:.2f} exceeds --max-cost ${max_cost:.2f}; "
              "nothing was sent.")
        return 2
    client = make_client(args.env_file)
    record = {"submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "seed": cfg["seed"], "n_per_condition": n, "pilot": args.pilot,
              "est_total_usd": round(total, 4), "max_cost_usd": max_cost, "batches": {}}
    print(f"\nsubmitting {len(models)} batch(es):")
    submit(client, requests_by_model, record, batches_path)
    print(f"batch ids -> {batches_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
