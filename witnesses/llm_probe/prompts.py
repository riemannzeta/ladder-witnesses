"""E6 prompt text, JSON output schemas and Message Batches request construction.

Request shapes follow the claude-api skill docs (python/claude-api/batches.md, tool-use.md
"Structured Outputs", README "Extended Thinking") and the anthropic 1.x SDK types:

* each prompt is one batch entry ``{"custom_id": ..., "params": {...}}``; ``run_batch`` wraps it as
  ``Request(custom_id=..., params=MessageCreateParamsNonStreaming(**params))``;
* structured outputs via ``output_config={"format": {"type": "json_schema", "schema": ...}}``
  (works with the Batches API and with extended thinking); ``output_config.effort`` where the
  model accepts it;
* thinking: ``{"type": "adaptive"}`` on Opus 5.5 / Sonnet 5.5 (``budget_tokens`` and
  ``disabled`` are 400s there), omitted on Fable 5.1 (always on), ``{"type": "enabled",
  "budget_tokens": 4000}`` on Haiku 4.5, which also rejects ``effort``;
* never ``fallbacks`` (Batches reject it, and an eval must measure each model), ``speed``,
  ``stream`` or sampling parameters.

Batch ``custom_id`` values must match ``^[a-zA-Z0-9_-]{1,64}$`` (batch-processing docs), so the
``|`` separator is not allowed: the four parts are joined with ``_`` and each part is restricted
to ``[A-Za-z0-9-]``, which makes the encoding unambiguous.
"""

from __future__ import annotations

import json
import re

SYSTEM_PROMPT = (
    "You are answering a quantitative causal-inference question. Use only the information given "
    "in the question. Give your answer in the required JSON format."
)

# ---------------------------------------------------------------- custom_id

CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_PART_RE = re.compile(r"^[A-Za-z0-9-]+$")
CID_SEP = "_"


def encode_custom_id(probe: str, condition: str, item_id: str, model: str) -> str:
    parts = (probe, condition, item_id, model)
    for p in parts:
        if not _PART_RE.match(p):
            raise ValueError(f"custom_id part {p!r} must match [A-Za-z0-9-]+")
    cid = CID_SEP.join(parts)
    if not CUSTOM_ID_RE.match(cid):
        raise ValueError(f"custom_id {cid!r} violates ^[a-zA-Z0-9_-]{{1,64}}$")
    return cid


def decode_custom_id(cid: str) -> tuple[str, str, str, str]:
    parts = cid.split(CID_SEP)
    if len(parts) != 4 or not all(_PART_RE.match(p) for p in parts):
        raise ValueError(f"not an E6 custom_id: {cid!r}")
    return parts[0], parts[1], parts[2], parts[3]


# ---------------------------------------------------------------- output schemas

_SUMMARY = {"type": "string", "description": "At most two sentences."}
_NULLABLE_NUMBER = {"anyOf": [{"type": "number"}, {"type": "null"}]}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


SCHEMAS: dict[str, dict] = {
    "V1": _obj({
        "p_do_x1": {"type": "number"},
        "p_do_x0": {"type": "number"},
        "determinable": {"type": "boolean"},
        "reasoning_summary": _SUMMARY,
    }),
    "V2": _obj({
        "counterfactual_y": {"type": "number"},
        "determinable": {"type": "boolean"},
        "reasoning_summary": _SUMMARY,
    }),
    "V3": _obj({
        "identifiable": {"type": "boolean"},
        "point_estimate": dict(_NULLABLE_NUMBER),
        "lower_bound": dict(_NULLABLE_NUMBER),
        "upper_bound": dict(_NULLABLE_NUMBER),
        "reasoning_summary": _SUMMARY,
    }),
}

# field -> allowed python types after json.loads (bool is excluded from numbers explicitly)
_FIELD_KINDS = {
    "V1": {"p_do_x1": "num", "p_do_x0": "num", "determinable": "bool", "reasoning_summary": "str"},
    "V2": {"counterfactual_y": "num", "determinable": "bool", "reasoning_summary": "str"},
    "V3": {"identifiable": "bool", "point_estimate": "num?", "lower_bound": "num?",
           "upper_bound": "num?", "reasoning_summary": "str"},
}


def parse_answer(probe: str, text: str) -> tuple[dict | None, str | None]:
    """json.loads the structured-output text and check it against the probe's schema."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"invalid JSON: {e}"
    if not isinstance(obj, dict):
        return None, "top-level JSON is not an object"
    kinds = _FIELD_KINDS[probe]
    missing = [k for k in kinds if k not in obj]
    extra = [k for k in obj if k not in kinds]
    if missing or extra:
        return None, f"missing={missing} extra={extra}"
    for k, kind in kinds.items():
        v = obj[k]
        if kind.startswith("num"):
            if v is None and kind == "num?":
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf"):
                return None, f"{k} is not a finite number: {v!r}"
        elif kind == "bool" and not isinstance(v, bool):
            return None, f"{k} is not a boolean: {v!r}"
        elif kind == "str" and not isinstance(v, str):
            return None, f"{k} is not a string: {v!r}"
    return obj, None


# ---------------------------------------------------------------- wording variants
# One entry per sentence slot; items.py draws an index per slot and context, so the two members
# of a witness pair share every sentence except the one that carries the higher-rung information.

V1_WORDING: dict[str, tuple[str, ...]] = {
    "intro": (
        "A study recorded three binary variables, {a}, {b} and {c} (each takes the value 0 or 1), "
        "on {n} units sampled independently from a large population. The table gives the number "
        "of units with each combination of values.",
        "Researchers measured three yes/no variables coded 0 or 1, called {a}, {b} and {c}, on {n} "
        "randomly sampled members of a large population. Each row of the table shows how many "
        "units had that combination of values.",
        "Data are available on {n} units drawn at random from a large population. For each unit, "
        "three binary (0/1) variables were recorded: {a}, {b} and {c}. The table counts the units "
        "with each combination of values.",
    ),
    "count_header": ("Count", "Number of units", "Units"),
    "ordering": (
        "{p} directly affects {q} and {r}, and {q} directly affects {r}. There are no other causes "
        "shared between them.",
        "{p} is a direct cause of both {q} and {r}; {q} is a direct cause of {r}. The three "
        "variables share no other common causes.",
        "The causal relationships are {p} → {q}, {p} → {r} and {q} → {r} (each arrow is a direct "
        "causal effect), and no other causes are shared between the variables.",
    ),
    "question": (
        "Estimate P({o} = 1 | do({t} = 1)) and P({o} = 1 | do({t} = 0)): the probability that "
        "{o} = 1 if {t} were set by an external intervention to 1, respectively to 0, for the "
        "whole population.",
        "Suppose an outside intervention forced {t} to equal 1 for every unit in the population. "
        "What fraction of units would have {o} = 1? Answer the same with {t} forced to 0. That "
        "is, give P({o} = 1 | do({t} = 1)) and P({o} = 1 | do({t} = 0)).",
        "What are the interventional probabilities P({o} = 1 | do({t} = 1)) and "
        "P({o} = 1 | do({t} = 0)), i.e. the chance that {o} equals 1 when {t} is set by "
        "intervention to 1 or to 0 throughout the population?",
    ),
}
V1_ANSWER = (
    "Answer fields: p_do_x1 = P({o} = 1 | do({t} = 1)) and p_do_x0 = P({o} = 1 | do({t} = 0)), "
    "as probabilities between 0 and 1; determinable = true if these two quantities can be "
    "determined from the information given (up to sampling error in the counts), false if they "
    "cannot (if false, still give your best estimates); reasoning_summary = at most two sentences."
)

V2_WORDING: dict[str, tuple[str, ...]] = {
    "mechanism": (
        "Two quantities, {x} and {y}, are related by the causal mechanism\n\n"
        "{y} = {w}·{x} + noise,\n\n"
        "where the noise equals σ·U1 when {x} > 0 and σ·(ρ·U1 + √(1 − ρ²)·U2) when {x} ≤ 0, with "
        "σ = 0.5. U1 and U2 are independent standard normal background variables that are "
        "specific to each unit and are not affected by {x}.",
        "{y} is produced from {x} by the structural equation {y} = {w}·{x} + N. The noise term is "
        "N = σ·U1 if {x} > 0 and N = σ·(ρ·U1 + √(1 − ρ²)·U2) if {x} ≤ 0, where σ = 0.5 and U1, "
        "U2 are independent standard normal unit-level background factors that do not depend "
        "on {x}.",
        "Consider the structural causal model {y} := {w}·{x} + σ·E with σ = 0.5, where E = U1 "
        "when {x} > 0 and E = ρ·U1 + √(1 − ρ²)·U2 when {x} ≤ 0. U1 and U2 are independent "
        "standard normal background variables attached to each unit; {x} does not affect them.",
    ),
    "rho_stated": (
        "Here ρ = {rho}.",
        "The parameter ρ equals {rho}.",
        "The value of ρ is {rho}.",
    ),
    "rho_unstated": (
        "The value of ρ is unknown; it could be anywhere between -1 and 1.",
        "ρ has not been disclosed: it is some number between -1 and 1.",
        "Nothing is known about ρ except that it lies between -1 and 1.",
    ),
    "observation": (
        "A particular unit was observed with {x} = {xs} and {y} = {ys}.",
        "For one specific unit, we observed {x} = {xs} and {y} = {ys}.",
        "One unit had {x} = {xs} and {y} = {ys}.",
    ),
    "question": (
        "For this same unit, what would {y} have been had {x} been {xp}? Give the expected value "
        "given all of the information above.",
        "Had {x} been {xp} for this very unit, what would its value of {y} have been? Report the "
        "expected value given the information above.",
        "What value would {y} have taken for this unit if {x} had instead been {xp}? Give the "
        "expected value given everything stated.",
    ),
}
V2_ANSWER = (
    "Answer fields: counterfactual_y = that expected value; determinable = true if this expected "
    "value is fully determined by the information given, false if it depends on something not "
    "given (if false, still give your best estimate); reasoning_summary = at most two sentences."
)

V3_WORDING: dict[str, tuple[str, ...]] = {
    "intro": (
        "A study examined three binary variables (each 0 or 1), {z}, {x} and {y}, in a large "
        "population, using {n} independently sampled units.",
        "Researchers studied three yes/no variables coded 0 or 1, called {z}, {x} and {y}, in a "
        "random sample of {n} units from a large population.",
        "Consider {n} units sampled at random from a large population, and three binary (0/1) "
        "variables {z}, {x} and {y}.",
    ),
    "graph": (
        "{z} affects both {x} and {y}, and {x} affects {y}. There are no other causes shared "
        "between them.",
        "{z} is a direct cause of {x} and of {y}; {x} is a direct cause of {y}. The variables "
        "share no other common causes.",
        "The causal relationships are {z} → {x}, {z} → {y} and {x} → {y}, and no other causes are "
        "shared between the variables.",
    ),
    "measured": (
        "All three variables were measured. The table gives the number of units with each "
        "combination of values.",
        "{z}, {x} and {y} were all recorded; the table counts the units with each combination of "
        "values.",
        "Every unit's {z}, {x} and {y} were measured. Counts for each combination of values:",
    ),
    "unmeasured": (
        "{z} was not measured in this study; only {x} and {y} were recorded. The table gives the "
        "number of units with each combination of {x} and {y}.",
        "Only {x} and {y} were recorded; {z} was not measured. The table counts the units with "
        "each combination of {x} and {y}.",
        "{z} was never measured. For {x} and {y}, the counts for each combination of values are:",
    ),
    "count_header": ("Count", "Number of units", "Units"),
    "question": (
        "Is P({y} = 1 | do({x} = 1)), the probability that {y} = 1 if {x} were set to 1 by an "
        "intervention, identifiable from the information given? If it is, give a point "
        "estimate. If it is not, give the tightest bounds on it that the data support.",
        "Can P({y} = 1 | do({x} = 1)), the fraction of the population with {y} = 1 if everyone's "
        "{x} were set to 1, be determined from what is given here? If so, estimate it. If not, "
        "report the tightest bounds the data allow.",
        "Consider P({y} = 1 | do({x} = 1)), the chance that {y} = 1 when {x} is set to 1 by an "
        "external intervention. Is it identifiable from this information? If yes, give a point "
        "estimate; if no, give the tightest bounds supported by the data.",
    ),
}
V3_ANSWER = (
    "Answer fields: identifiable = true or false. If identifiable, give point_estimate and set "
    "lower_bound and upper_bound to null. If not identifiable, set point_estimate to null and "
    "give the tightest lower_bound and upper_bound. reasoning_summary = at most two sentences."
)

# Slots drawn once per context -> number of variants. A condition-specific slot ("rho",
# "measurement") indexes both twins' sentence tables with the same draw.
WORDING_SLOTS = {
    "V1": {k: len(v) for k, v in V1_WORDING.items()},
    "V2": {"mechanism": len(V2_WORDING["mechanism"]),
           "rho": min(len(V2_WORDING["rho_stated"]), len(V2_WORDING["rho_unstated"])),
           "observation": len(V2_WORDING["observation"]),
           "question": len(V2_WORDING["question"])},
    "V3": {"intro": len(V3_WORDING["intro"]), "graph": len(V3_WORDING["graph"]),
           "measurement": min(len(V3_WORDING["measured"]), len(V3_WORDING["unmeasured"])),
           "count_header": len(V3_WORDING["count_header"]),
           "question": len(V3_WORDING["question"])},
}


# ---------------------------------------------------------------- rendering


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def fmt2(v: float) -> str:
    return f"{round(float(v), 2) + 0.0:.2f}"  # + 0.0 turns -0.0 into 0.0


def fmt_rho(r: float) -> str:
    return format(float(r) + 0.0, "g")


def ordering_sentence(item: dict) -> str:
    p, q, r = (item["names"][i] for i in item["ordering"])
    return V1_WORDING["ordering"][item["wording"]["ordering"]].format(p=p, q=q, r=r)


def render_v1(item: dict) -> str:
    wd, names = item["wording"], item["names"]
    a, b, c = names
    t, o = item["treatment"], item["outcome"]
    counts = item["counts"]  # cell index 4*v(L0) + 2*v(L1) + v(L2)
    rows = [[(k >> 2) & 1, (k >> 1) & 1, k & 1, counts[k]] for k in range(8)]
    parts = [
        V1_WORDING["intro"][wd["intro"]].format(a=a, b=b, c=c, n=item["n_samples"]),
        md_table([a, b, c, V1_WORDING["count_header"][wd["count_header"]]], rows),
    ]
    if item["condition"] == "stated":
        parts.append(ordering_sentence(item))
    parts.append(V1_WORDING["question"][wd["question"]].format(t=t, o=o))
    parts.append(V1_ANSWER.format(t=t, o=o))
    return "\n\n".join(parts)


def rho_sentence(item: dict) -> str:
    k = item["wording"]["rho"]
    if item["condition"] == "stated":
        return V2_WORDING["rho_stated"][k].format(rho=fmt_rho(item["rho"]))
    return V2_WORDING["rho_unstated"][k]


def render_v2(item: dict) -> str:
    wd = item["wording"]
    x, y = item["names"]
    parts = [
        V2_WORDING["mechanism"][wd["mechanism"]].format(x=x, y=y, w=fmt2(item["w"])),
        rho_sentence(item),
        V2_WORDING["observation"][wd["observation"]].format(
            x=x, y=y, xs=fmt2(item["x_star"]), ys=fmt2(item["y_star"]))
        + " " + V2_WORDING["question"][wd["question"]].format(x=x, y=y, xp=fmt2(item["x_prime"])),
        V2_ANSWER,
    ]
    return "\n\n".join(parts)


def render_v3(item: dict) -> str:
    wd = item["wording"]
    z, x, y = item["names"]
    head = V3_WORDING["count_header"][wd["count_header"]]
    if item["condition"] == "measured":
        n = item["counts_zxy"]
        table = md_table([z, x, y, head], [[i, j, k, n[i][j][k]] for i in (0, 1) for j in (0, 1)
                                          for k in (0, 1)])
        sentence = V3_WORDING["measured"][wd["measurement"]]
    else:
        n = item["counts_xy"]
        table = md_table([x, y, head], [[j, k, n[j][k]] for j in (0, 1) for k in (0, 1)])
        sentence = V3_WORDING["unmeasured"][wd["measurement"]]
    parts = [
        V3_WORDING["intro"][wd["intro"]].format(z=z, x=x, y=y, n=item["n_samples"]) + " "
        + V3_WORDING["graph"][wd["graph"]].format(z=z, x=x, y=y),
        sentence.format(z=z, x=x, y=y),
        table,
        V3_WORDING["question"][wd["question"]].format(x=x, y=y),
        V3_ANSWER,
    ]
    return "\n\n".join(parts)


RENDERERS = {"V1": render_v1, "V2": render_v2, "V3": render_v3}


def render_prompt(item: dict) -> str:
    return RENDERERS[item["probe"]](item)


# ---------------------------------------------------------------- requests

# Documented per-model constraints (claude-api skill, "Thinking & Effort" table).
ADAPTIVE_ONLY_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"}
BUDGET_THINKING_MODELS = {"claude-haiku-4-5"}  # pre-4.6 style thinking; `effort` is rejected
EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}
FORBIDDEN_PARAMS = {"fallbacks", "betas", "speed", "stream", "temperature", "top_p", "top_k"}


def request_params(item: dict, model: str, model_cfg: dict, max_tokens: int) -> dict:
    """Messages API params for one item (a MessageCreateParamsNonStreaming-shaped dict)."""
    params: dict = {
        "model": model,
        "max_tokens": int(max_tokens),
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": item["prompt"]}],
    }
    if model_cfg.get("thinking") is not None:
        params["thinking"] = dict(model_cfg["thinking"])
    output_config: dict = {}
    if model_cfg.get("effort") is not None:
        output_config["effort"] = model_cfg["effort"]
    output_config["format"] = {"type": "json_schema", "schema": SCHEMAS[item["probe"]]}
    params["output_config"] = output_config
    validate_params(params)
    return params


def validate_params(params: dict) -> None:
    """Offline guard against requests the docs say the API rejects (no request is sent)."""
    model = params["model"]
    bad = FORBIDDEN_PARAMS & set(params)
    if bad:
        raise ValueError(f"{model}: parameters not allowed in E6 batch requests: {sorted(bad)}")
    max_tokens = params["max_tokens"]
    if not 1 <= max_tokens <= 16000:
        raise ValueError(f"{model}: max_tokens {max_tokens} outside [1, 16000]")
    thinking = params.get("thinking")
    oc = params.get("output_config") or {}
    effort = oc.get("effort")
    if effort is not None and effort not in EFFORT_LEVELS:
        raise ValueError(f"{model}: unknown effort {effort!r}")
    if model in ADAPTIVE_ONLY_MODELS:
        if thinking is not None and thinking.get("type") != "adaptive":
            raise ValueError(f"{model}: thinking must be adaptive or omitted, got {thinking}")
    elif model in BUDGET_THINKING_MODELS:
        if effort is not None:
            raise ValueError(f"{model}: output_config.effort is not supported")
        if thinking is not None:
            if thinking.get("type") != "enabled":
                raise ValueError(f"{model}: thinking must be {{type: enabled, budget_tokens}}")
            budget = thinking.get("budget_tokens")
            if not (isinstance(budget, int) and 1024 <= budget < max_tokens):
                raise ValueError(f"{model}: budget_tokens must be >= 1024 and < max_tokens")
    else:
        raise ValueError(f"model {model!r} has no documented E6 settings")
    fmt = oc.get("format") or {}
    if fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
        raise ValueError(f"{model}: output_config.format must be a json_schema")


def build_request(item: dict, model: str, model_cfg: dict, max_tokens: int) -> dict:
    return {
        "custom_id": encode_custom_id(item["probe"], item["condition"], item["item_id"], model),
        "params": request_params(item, model, model_cfg, max_tokens),
    }
