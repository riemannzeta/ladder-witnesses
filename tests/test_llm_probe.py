"""E6 (Claude verbal probe): items, oracles, prompts, request shapes and scoring. No API calls."""

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from witnesses.llm_probe import prompts, run_batch
from witnesses.llm_probe.items import (CONDITIONS, generate_items, item_key, nonsense_names,
                                       read_items, v2_oracle, write_items)
from witnesses.llm_probe.prompts import (CUSTOM_ID_RE, SCHEMAS, build_request, decode_custom_id,
                                         encode_custom_id, parse_answer, validate_params)
from witnesses.llm_probe.score import score_model
from witnesses.tasks.e1_simpson import (ADJ, CRUDE, EST_NAMES, EST_TABLE, FUNC_P, PERMS,
                                        functionals_posterior)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "e6" / "probe.yaml"
CFG = yaml.safe_load(CONFIG.read_text())

COMMON_WORDS = set("""
the be to of and a in that have it for not on with he as you do at this but his by from they we
say her she or an will my one all would there their what so up out if about who get which go me
when make can like time no just him know take people into year your good some could them see other
than then now look only come its over think also back after use two how our work first well way
even new want because any these give day most us table count number value study cause effect
data sample unit model noise water paper money music river robot lemon melon tiger zebra pizza
banana potato tomato radio piano salad pilot hotel motel metal medal pedal rival total vital local
vocal panic magic logic topic tonic comic relax remix matrix prefix paradox topaz latex reflex
duplex kodak felix batik kapok borax index annex complex vortex fox box tax mix fix wax zinc
talk walk milk silk desk risk task mask park dark work fork pink link bank tank rock lock block
stock truck stick trick brick snack track black smoke drive prove brave grave move love above
""".split())


def _items(n=12, seed=None):
    cfg = dict(CFG)
    if seed is not None:
        cfg["seed"] = seed
    return generate_items(cfg, n)


def _by_key(items):
    return {item_key(it): it for its in items.values() for it in its}


# ---------------------------------------------------------------- items


def test_items_deterministic_given_seed_and_prefix_stable():
    a, b = _items(6), _items(6)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = _items(6, seed=CFG["seed"] + 1)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)
    small = _items(3)  # a pilot of K contexts is the first K contexts of a larger run
    for probe in small:
        big = [it for it in a[probe] if it["context_id"] < 3]
        assert json.dumps(small[probe], sort_keys=True) == json.dumps(big, sort_keys=True)


def test_item_counts_per_condition():
    items = _items(10)
    v1 = items["V1"]
    assert sum(it["condition"] == "blind" for it in v1) == 10
    assert sum(it["condition"] == "stated" for it in v1) == 30  # one adj, crude, marg each
    for probe in ("V2", "V3"):
        for cond in CONDITIONS[probe]:
            assert sum(it["condition"] == cond for it in items[probe]) == 10


def test_nonsense_names_are_not_common_english_words():
    pat = re.compile(r"^[A-Z][a-z]{3,7}$")
    names = []
    for s in range(300):
        names += nonsense_names(np.random.default_rng(s), 3)
    per_context = {}  # witness twins share their names, so count each context once
    for seed in (0, 1):
        for its in _items(8, seed=seed).values():
            for it in its:
                assert len(set(it["names"])) == len(it["names"])
                per_context[(seed, it["probe"], it["context_id"])] = it["names"]
    names += [n for v in per_context.values() for n in v]
    assert all(pat.match(n) for n in names), [n for n in names if not pat.match(n)][:5]
    assert not {n.lower() for n in names} & COMMON_WORDS
    assert len(set(names)) > 0.95 * len(names)  # varied, not a small fixed pool


def test_v1_oracle_equals_functionals_posterior():
    items = _items(12)
    alpha = CFG["v1"]["alpha"]
    for it in items["V1"]:
        x, y, c = (it["roles"][k] for k in ("treatment", "outcome", "covariate"))
        # rebuild (c, x, y) counts from the 8-cell table independently of cxy_index
        n = np.zeros((2, 2, 2), int)
        for cell, cnt in enumerate(it["counts"]):
            v = ((cell >> 2) & 1, (cell >> 1) & 1, cell & 1)  # values of (L0, L1, L2)
            n[v[c], v[x], v[y]] += cnt
        assert n.tolist() == it["counts_cxy"] and n.sum() == it["n_samples"]
        assert n.sum(-1).min() >= CFG["v1"]["min_stratum_count"]
        F = functionals_posterior(n, alpha)
        mix = np.tensordot(np.array(FUNC_P), F, axes=(0, 0))
        if it["condition"] == "stated":
            e = int(EST_TABLE[it["ordering_index"], x, y])
            assert tuple(it["ordering"]) == PERMS[it["ordering_index"]]
            assert it["estimand"] == EST_NAMES[e]
            assert np.allclose([it["oracle"]["p_do_x0"], it["oracle"]["p_do_x1"]], F[e])
            assert it["oracle"]["determinable"] is True
        else:
            assert np.allclose([it["oracle"]["p_do_x0"], it["oracle"]["p_do_x1"]], mix)
            assert it["oracle"]["determinable"] is False
        ate_a, ate_c = F[ADJ, 1] - F[ADJ, 0], F[CRUDE, 1] - F[CRUDE, 0]
        rev = np.sign(ate_a) != np.sign(ate_c) and min(abs(ate_a), abs(ate_c)) > 0.03
        assert it["reversal"] == bool(rev)
        if it["kind"] == "reversal":
            assert it["reversal"]
    blind = [it for it in items["V1"] if it["condition"] == "blind"]
    assert sum(it["kind"] == "reversal" for it in blind) == 6  # half the contexts
    stated = [it for it in items["V1"] if it["condition"] == "stated"]
    for cid in range(12):
        ests = sorted(it["estimand"] for it in stated if it["context_id"] == cid)
        assert ests == ["adj", "crude", "marg"]


def test_v2_oracle_matches_monte_carlo():
    rng = np.random.default_rng(0)
    sigma, w, M = 0.5, 1.3, 1_000_000
    for rho in (-1.0, -0.5, 0.0, 0.5, 1.0, None):
        for x_star, x_prime in ((0.8, -1.2), (-0.7, 1.5), (0.4, 1.1), (-0.3, -1.6)):
            u1, u2 = rng.standard_normal((2, M))
            r = rng.uniform(-1, 1, M) if rho is None else np.full(M, rho)  # unknown rho: U(-1,1)

            def noise(xv):
                return sigma * u1 if xv > 0 else sigma * (r * u1 + np.sqrt(1 - r * r) * u2)

            y_obs = w * x_star + noise(x_star)
            y_cf = w * x_prime + noise(x_prime)
            y_star = w * x_star + 0.2
            sel = np.abs(y_obs - y_star) < 0.01
            assert sel.sum() > 5000
            mc = y_cf[sel].mean()
            assert abs(mc - v2_oracle(w, x_star, y_star, x_prime, rho)) < 0.02, (rho, x_star, mc)


def test_v2_items_structure():
    items = _items(15)["V2"]
    for it in items:
        assert min(abs(it["x_star"]), abs(it["x_prime"])) >= 0.1
        crossing = (it["x_star"] > 0) != (it["x_prime"] > 0)
        assert it["crossing"] == crossing == (it["context_id"] % 5 != 4)
        rho = it["rho"] if it["condition"] == "stated" else None
        o = v2_oracle(it["w"], it["x_star"], it["y_star"], it["x_prime"], rho)
        assert math.isclose(it["oracle"]["counterfactual_y"], o)
        if it["condition"] == "stated":
            assert it["rho"] in CFG["v2"]["rho_values"] and it["oracle"]["determinable"]
        else:
            assert it["rho"] is None and it["oracle"]["determinable"] == (not crossing)
            if crossing:
                assert math.isclose(o, it["w"] * it["x_prime"])
    stated_rhos = [it["rho"] for it in items if it["condition"] == "stated" and it["crossing"]]
    assert sorted(set(stated_rhos)) == sorted(CFG["v2"]["rho_values"])


def test_v3_items_oracles_and_bounds():
    for it in _items(10)["V3"]:
        n = np.array(it["counts_zxy"])
        N = n.sum()
        F = functionals_posterior(n, CFG["v3"]["alpha"])
        assert math.isclose(it["adjusted_posterior"], F[ADJ, 1])
        lo, hi = n[:, 1, 1].sum() / N, (n[:, 1, 1].sum() + n[:, 0, :].sum()) / N
        assert np.allclose(it["natural_bounds"], [lo, hi])
        assert lo <= it["crude"] <= hi
        assert np.array(it["counts_xy"]).tolist() == n.sum(0).tolist()
        if it["condition"] == "measured":
            assert it["oracle"]["identifiable"] and math.isclose(
                it["oracle"]["point_estimate"], F[ADJ, 1])
        else:
            assert not it["oracle"]["identifiable"] and it["oracle"]["point_estimate"] is None
            assert np.allclose([it["oracle"]["lower_bound"], it["oracle"]["upper_bound"]],
                               [lo, hi])


def test_witness_pairs_differ_only_in_the_key_sentence():
    items = _by_key(_items(6))
    for (probe, cond, iid), it in items.items():
        if probe == "V1" and cond == "stated":
            blind = items[("V1", "blind", iid.split("-")[0])]
            sentence = prompts.ordering_sentence(it)
            assert sentence in it["prompt"] and "affect" not in blind["prompt"]
            assert it["prompt"].replace("\n\n" + sentence, "", 1) == blind["prompt"]
        if probe == "V2" and cond == "stated":
            twin = items[("V2", "unstated", iid)]
            a = it["prompt"].replace(prompts.rho_sentence(it), "<RHO>")
            b = twin["prompt"].replace(prompts.rho_sentence(twin), "<RHO>")
            assert a == b and "<RHO>" in a
        if probe == "V3" and cond == "measured":
            twin = items[("V3", "unmeasured", iid)]
            pa, pb = it["prompt"].split("\n\n"), twin["prompt"].split("\n\n")
            assert pa[0] == pb[0] and pa[-2:] == pb[-2:] and pa[1:-2] != pb[1:-2]
            assert it["z_name"] in pa[2] and it["z_name"] not in pb[2]  # Z column dropped


def test_items_roundtrip_through_jsonl(tmp_path):
    items = _items(3)
    write_items(items, tmp_path, {"seed": CFG["seed"]})
    back = read_items(tmp_path)
    assert back == _by_key(json.loads(json.dumps(items)))
    assert json.loads((tmp_path / "manifest.json").read_text())["files"]["V1.jsonl"]["n_items"] == 12


# ---------------------------------------------------------------- prompts / requests


def test_custom_id_roundtrip():
    items = _items(4)
    seen = set()
    for model in CFG["models"]:
        for its in items.values():
            for it in its:
                cid = encode_custom_id(it["probe"], it["condition"], it["item_id"], model)
                assert CUSTOM_ID_RE.match(cid) and len(cid) <= 64
                assert decode_custom_id(cid) == (it["probe"], it["condition"], it["item_id"], model)
                seen.add(cid)
    assert len(seen) == len(CFG["models"]) * sum(len(v) for v in items.values())
    for bad in (("V1", "stated", "c0_1", "m"), ("V1", "a|b", "c", "m"), ("V1", "s", "c", "x" * 60)):
        with pytest.raises(ValueError):
            encode_custom_id(*bad)
    with pytest.raises(ValueError):
        decode_custom_id("V1_stated_c000")


def test_request_shapes_built_without_sending():
    items = _items(2)
    for model, mc in CFG["models"].items():
        reqs = run_batch.build_requests(items, model, CFG)
        assert len(reqs) == sum(len(v) for v in items.values())
        for r in reqs:
            assert set(r) == {"custom_id", "params"}
            p = r["params"]
            probe = decode_custom_id(r["custom_id"])[0]
            assert p["model"] == model and p["max_tokens"] == 16000
            assert p["system"] == prompts.SYSTEM_PROMPT
            assert p["messages"] == [{"role": "user", "content": p["messages"][0]["content"]}]
            assert p["output_config"]["format"] == {"type": "json_schema",
                                                    "schema": SCHEMAS[probe]}
            assert not {"fallbacks", "betas", "stream", "speed", "temperature"} & set(p)
            if model == "claude-haiku-4-5":
                assert p["thinking"] == {"type": "enabled", "budget_tokens": 4000}
                assert "effort" not in p["output_config"]
            elif model == "claude-fable-5-1":
                assert "thinking" not in p and p["output_config"]["effort"] == "high"
            else:
                assert p["thinking"] == {"type": "adaptive"}
                assert p["output_config"]["effort"] == "high"
            json.dumps(r)  # JSONL-serializable


def test_schemas_are_strict_and_parse():
    anthropic = pytest.importorskip("anthropic")
    for probe, schema in SCHEMAS.items():
        assert anthropic.transform_schema(schema) == schema  # already in the API's strict form
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    ok, err = parse_answer("V1", '{"p_do_x1": 0.4, "p_do_x0": 0.6, "determinable": true, '
                                 '"reasoning_summary": "Adjust."}')
    assert err is None and ok["p_do_x1"] == 0.4
    ok, err = parse_answer("V3", '{"identifiable": false, "point_estimate": null, "lower_bound": '
                                 '0.1, "upper_bound": 0.6, "reasoning_summary": "x"}')
    assert err is None and ok["point_estimate"] is None
    for probe, text in (("V1", "not json"), ("V2", '{"counterfactual_y": "1"}'),
                        ("V2", '{"counterfactual_y": true, "determinable": true, '
                               '"reasoning_summary": ""}'),
                        ("V3", '{"identifiable": 1, "point_estimate": null, "lower_bound": null,'
                               ' "upper_bound": null, "reasoning_summary": ""}')):
        assert parse_answer(probe, text)[0] is None


def test_validate_params_rejects_documented_400s():
    it = _items(1)["V1"][0]
    base = build_request(it, "claude-opus-5-5", CFG["models"]["claude-opus-5-5"], 16000)["params"]
    for mutate in (
        lambda p: p.update(thinking={"type": "enabled", "budget_tokens": 4000}),
        lambda p: p.update(thinking={"type": "disabled"}),
        lambda p: p.update(fallbacks="default"),
        lambda p: p.update(temperature=0.0),
        lambda p: p.update(model="claude-haiku-4-5"),  # haiku rejects effort
        lambda p: p.update(model="claude-unknown-1"),
        lambda p: p["output_config"].pop("format"),
    ):
        p = json.loads(json.dumps(base))
        mutate(p)
        with pytest.raises(ValueError):
            validate_params(p)
    haiku = {**CFG["models"]["claude-haiku-4-5"], "thinking": {"type": "adaptive"}}
    with pytest.raises(ValueError):
        build_request(it, "claude-haiku-4-5", haiku, 16000)


# ---------------------------------------------------------------- scoring


def _records(items_by_key, answer):
    return [{"probe": it["probe"], "condition": it["condition"], "item_id": it["item_id"],
             "model": "m", "category": "ok", "stop_reason": "end_turn", "parsed": answer(it)}
            for it in items_by_key.values()]


def _oracle_answer(it):
    return {**it["oracle"], "reasoning_summary": ""}


def _l2_only_answer(it):
    """Rung-2 answers: V2 ignores the residual entirely (E[Y | do(X = x')]); V1 ignores the
    ordering (always the mixture); V3 always reports the crude conditional."""
    if it["probe"] == "V1":
        p0, p1 = it["blind_mixture"]
        return {"p_do_x0": p0, "p_do_x1": p1, "determinable": True, "reasoning_summary": ""}
    if it["probe"] == "V2":
        return {"counterfactual_y": it["l2_answer"], "determinable": True,
                "reasoning_summary": ""}
    return {"identifiable": True, "point_estimate": it["crude"], "lower_bound": None,
            "upper_bound": None, "reasoning_summary": ""}


SCFG = {"n_boot": 200, "seed": 0, "split_min_spread": 0.02, "rho_values": CFG["v2"]["rho_values"],
        "bound_tol": 0.03}


def test_scoring_perfect_answers():
    items = _by_key(_items(40))
    s = score_model(items, _records(items, _oracle_answer), SCFG)
    v1s, v1b = s["V1"]["stated"], s["V1"]["blind"]
    assert abs(v1s["split_slope"]["value"] - 1) < 1e-9 and v1s["split_r2"]["value"] > 0.999
    assert v1s["split_slope"]["n"] >= 35
    assert v1s["mae"]["value"] < 1e-12 and v1s["determinable_true"]["value"] == 1
    assert v1s["reversal_agree_both"]["value"] == 1 and v1s["reversal_agree_both"]["n"] >= 20
    assert v1b["determinable_false"]["value"] == 1 and v1b["mae_vs_mixture"]["value"] < 1e-12
    for r in CFG["v2"]["rho_values"]:
        e = s["V2"]["stated"][f"slope_rho={float(r):g}"]
        assert abs(e["value"] - r) < 1e-9 and e["n"] >= 6
    assert abs(s["V2"]["stated"]["rho_tracking_slope"]["value"] - 1) < 1e-9
    assert abs(s["V2"]["unstated"]["slope_crossing"]["value"]) < 1e-9
    assert s["V2"]["unstated"]["determinable_true_crossing"]["value"] == 0
    assert s["V2"]["unstated"]["determinable_true_same_side"]["value"] == 1
    assert s["V3"]["measured"]["identifiable_acc"]["value"] == 1
    assert s["V3"]["measured"]["point_mae"]["value"] < 1e-12
    u = s["V3"]["unmeasured"]
    assert u["identifiable_acc"]["value"] == 1 and u["bounds_acc"]["value"] == 1
    assert u["point_given"]["value"] == 0
    assert all(c["ok"] > 0 and c["missing"] == 0 for c in s["categories"].values())


def test_scoring_l2_only_and_blind_answers():
    items = _by_key(_items(40))
    s = score_model(items, _records(items, _l2_only_answer), SCFG)
    for r in CFG["v2"]["rho_values"]:
        assert abs(s["V2"]["stated"][f"slope_rho={float(r):g}"]["value"]) < 1e-9
    assert abs(s["V2"]["stated"]["rho_tracking_slope"]["value"]) < 1e-9
    assert abs(s["V1"]["stated"]["split_slope"]["value"]) < 1e-9  # ordering-blind
    assert s["V3"]["unmeasured"]["identifiable_acc"]["value"] == 0
    assert s["V3"]["unmeasured"]["point_given"]["value"] == 1


def test_scoring_counts_refusals_and_parse_failures_separately():
    items = _by_key(_items(4))
    recs = _records(items, _oracle_answer)
    recs[0].update(category="refusal", parsed=None, stop_reason="refusal")
    recs[1].update(category="parse_failure", parsed=None)
    recs.pop()  # one item without a response
    s = score_model(items, recs, SCFG)
    tot = {c: sum(v[c] for v in s["categories"].values()) for c in ("ok", "refusal",
                                                                   "parse_failure", "missing")}
    assert tot == {"ok": len(items) - 3, "refusal": 1, "parse_failure": 1, "missing": 1}
    assert s["stop_reasons"]["refusal"] == 1


# ---------------------------------------------------------------- runner (offline)


class _NoClient:
    def __init__(self, *a, **k):
        raise AssertionError("no API client may be constructed in this test")


def test_dry_run_and_cost_cap_never_construct_a_client(tmp_path, monkeypatch, capsys):
    anthropic = pytest.importorskip("anthropic")
    monkeypatch.setattr(anthropic, "Anthropic", _NoClient)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = run_batch.main(["--config", str(CONFIG), "--dry-run", "--pilot", "2",
                         "--out-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out and "nothing was sent" in out
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"):
        lines = (tmp_path / "requests" / f"{model}.jsonl").read_text().splitlines()
        assert len(lines) == 2 * 4 + 2 * 2 + 2 * 2
    assert not (tmp_path / "requests" / "claude-fable-5-1.jsonl").exists()
    rc = run_batch.main(["--config", str(CONFIG), "--submit", "--pilot", "2", "--max-cost", "0.01",
                         "--out-dir", str(tmp_path)])
    assert rc == 2 and "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "batches.json").exists()
    with pytest.raises(SystemExit):
        run_batch.main(["--config", str(CONFIG), "--dry-run", "--models", "claude-fable-5-1",
                        "--out-dir", str(tmp_path)])


def test_cost_estimate_arithmetic():
    reqs = [{"params": {"system": "s" * 35, "messages": [{"role": "user", "content": "u" * 315}],
                        "output_config": {"format": {"schema": {}}}}}] * 10
    est = run_batch.estimate_cost(reqs, "claude-opus-5-5", CFG)
    in_tok = 10 * (35 + 315 + len("{}")) / 3.5
    assert math.isclose(est["input_tokens"], in_tok)
    assert math.isclose(est["cost_usd"], (in_tok * 4 + 10 * 2500 * 20) / 1e6 * 0.5)


def test_env_file_key_parsing_only_reads_the_key_line(tmp_path, monkeypatch, capsys):
    f = tmp_path / "keys.env"
    f.write_text("OTHER_SECRET=abc\n# comment\nexport ANTHROPIC_API_KEY='sk-test-123'\nX=1\n")
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    assert run_batch.load_api_key(str(f))[0] == "sk-test-123"
    assert "OTHER_SECRET" not in __import__("os").environ
    f.write_text("ANTHROPIC_API_KEY=sk-test-456 # note\n")
    assert run_batch.load_api_key(str(f))[0] == "sk-test-456"
    f.write_text("NOTHING=here\n")
    with pytest.raises(SystemExit):
        run_batch.load_api_key(str(f))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        run_batch.load_api_key(None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-env")
    assert run_batch.load_api_key(None)[0] == "sk-test-env"

    anthropic = pytest.importorskip("anthropic")
    seen = {}

    class FakeClient:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    run_batch.make_client(None)
    assert seen == {"api_key": "sk-test-env"}
    assert "sk-test" not in capsys.readouterr().out  # the key is never printed


class _FakeBatches:
    """Stands in for client.messages.batches: records create() calls, serves canned results."""

    def __init__(self):
        self.created, self.results_by_id = {}, {}

    def create(self, requests):
        bid = f"msgbatch_fake{len(self.created)}"
        self.created[bid] = list(requests)
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        return SimpleNamespace(id=bid, processing_status="ended",
                               request_counts=SimpleNamespace(processing=0, succeeded=1, errored=0))

    def results(self, bid):
        return iter(self.results_by_id[bid])


def test_submit_and_collect_paths_with_a_fake_client(tmp_path, monkeypatch):
    pytest.importorskip("anthropic")
    batches = _FakeBatches()
    fake = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    monkeypatch.setattr(run_batch, "make_client", lambda env_file, workspace_id=None: fake)
    args = ["--config", str(CONFIG), "--pilot", "1", "--out-dir", str(tmp_path)]
    assert run_batch.main(args + ["--submit"]) == 0
    log = json.loads((tmp_path / "batches.json").read_text())
    sub = log["submissions"][-1]
    assert set(sub["batches"]) == {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5"}
    for model, info in sub["batches"].items():
        reqs = batches.created[info["batch_id"]]
        assert len(reqs) == info["n_requests"] == 4 + 2 + 2
        assert all(decode_custom_id(r["custom_id"])[3] == model for r in reqs)
        assert all(r["params"]["model"] == model and "fallbacks" not in r["params"] for r in reqs)
        written = (tmp_path / "requests" / f"{model}.jsonl").read_text().splitlines()
        assert [json.loads(line) for line in written] == [dict(r) for r in reqs]
        # canned results: every request answered with its oracle, in reverse order
        items = read_items(tmp_path / "items")
        res = []
        for r in reversed(reqs):
            probe, cond, iid, _ = decode_custom_id(r["custom_id"])
            text = json.dumps({**items[(probe, cond, iid)]["oracle"], "reasoning_summary": "r"})
            res.append(_fake_result(r["custom_id"], "succeeded",
                                    [SimpleNamespace(type="text", text=text)]))
        batches.results_by_id[info["batch_id"]] = res
    assert run_batch.main(args + ["--collect"]) == 0  # no ids: the latest submission
    for model in sub["batches"]:
        recs = [json.loads(line) for line in
                (tmp_path / "responses" / f"{model}.jsonl").read_text().splitlines()]
        assert len(recs) == 8 and all(r["category"] == "ok" for r in recs)


def _fake_result(cid, rtype, content=(), stop_reason="end_turn"):
    to_dict = {"to_dict": lambda: {"x": 1}}
    if rtype == "succeeded":
        msg = SimpleNamespace(id="msg_1", model="m", content=list(content), stop_reason=stop_reason,
                              stop_details=SimpleNamespace(**to_dict) if stop_reason == "refusal"
                              else None, usage=SimpleNamespace(**to_dict))
        res = SimpleNamespace(type=rtype, message=msg)
    elif rtype == "errored":
        res = SimpleNamespace(type=rtype, error=SimpleNamespace(**to_dict))
    else:
        res = SimpleNamespace(type=rtype)
    return SimpleNamespace(custom_id=cid, result=res)


def test_result_records_categories():
    cid = encode_custom_id("V2", "stated", "c000", "claude-opus-5-5")
    good = SimpleNamespace(type="text", text='{"counterfactual_y": 1.5, "determinable": true, '
                                             '"reasoning_summary": "r"}')
    think = SimpleNamespace(type="thinking", thinking="")
    cases = [
        (_fake_result(cid, "succeeded", [think, good]), "ok"),
        (_fake_result(cid, "succeeded", [SimpleNamespace(type="text", text="I can't")],
                      stop_reason="refusal"), "refusal"),
        (_fake_result(cid, "succeeded", [SimpleNamespace(type="text", text='{"counterf')],
                      stop_reason="max_tokens"), "parse_failure"),
        (_fake_result(cid, "errored"), "errored"),
        (_fake_result(cid, "expired"), "expired"),
    ]
    for res, cat in cases:
        rec = run_batch.result_record(res, "msgbatch_x")
        assert rec["category"] == cat
        assert (rec["probe"], rec["condition"], rec["item_id"], rec["model"]) == (
            "V2", "stated", "c000", "claude-opus-5-5")
    rec = run_batch.result_record(cases[0][0], "b")
    assert rec["parsed"]["counterfactual_y"] == 1.5 and rec["usage"] == {"x": 1}
