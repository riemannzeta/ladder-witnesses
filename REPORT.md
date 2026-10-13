# Report: witness pairs on Pearl's ladder

Pre-registration: [`PREDICTIONS.md`](PREDICTIONS.md). The main document was committed before any training (`b48faf9`); Addenda 1–4 record every later change, each written before the runs it governs. Every pre-registered criterion is reported here exactly as defined, with its verdict. Anything that was not pre-registered is labeled **post hoc**.

**Status (2026-10-13): all runs complete.**
- wave 1 (77 runs) and wave 2 (48 runs);
- post-hoc arms 1–3 (5 + 5 + 5 runs), plus wave2c: arm 3, folds 0–2 re-run to the 240k cap (3 runs, Addendum 4). The 240k runs are the primary arm-3 rows; the 36–38k runs are kept as a training-length comparison;
- E6 (800 items × 4 Claude models).

Figures: `results/figures/` (PNG + PDF). Full tables: the generated block at the end of this file, also written to `results/report_tables.md` and `results/summary.json` by `python -m witnesses.analyze`.

## Verdicts at a glance

| | Role | Pre-registered result | In one line |
|---|---|---|---|
| E1 | demonstration | **9/9 PASS** | Given the ordering token, the answers move across the six orderings exactly as Bayes says (split slope 1.000). With the token scrambled they don't move at all (0.000). |
| E2-0 | test | **Interventions: all PASS** (single, unseen graphs, combined, graph swap). **Matched L1 two-variable conditioning: FAIL** (GC 0.38 / 0.40). | Given graph + CPTs, the model computes untried single and combined interventions on unseen graphs (GC 1.000). Its weak spot is a composite *observational* query. |
| E2-full | test | **Single queries, unseen graphs, graph swap: PASS.** **Combined interventions: FAIL** (GC 0.22 / 0.11); **two-variable conditioning: FAIL** (≈ 0.28 bits) | Given graph + 128 samples, single queries transfer to unseen graphs; nothing composite does. |
| E2c | test | **3/3 PASS** | With no graph, given data from tried do(A) and do(B), the model answers untried do(A, B) and do(C) close to ideal Bayesian model averaging (GC 0.971 and 0.954 against the model-free "1¾" learner). |
| E3 | demonstration | **6/6 PASS** | Counterfactual residual transfer tracks ρ, including the held-out band. Both controls collapse to the interventional answer. |
| E4 | demonstration | **Slopes 4/4 PASS. sd ratio FAIL** (OBS-CONF). **Coverage as implemented FAIL** (4/4) | Posterior width shrinks like the oracle's where the effect is identified and plateaus where it isn't. The far tail of the softmax head is too heavy. |
| E5 | demonstration of transfer | **GC_TV PASS** (all conditions, seen programs). **Flags-zeroed FAIL. Held-out-program prediction wrong.** | The same witness pairs work in Misra's recurrence on seen programs. On held-out programs even factual prediction fails, so there is no L3-specific deficit to report. |
| E6 | descriptive | Opus/Sonnet predictions held; Haiku beat 4 of 6 predictions | — |

## Reading E2: what the combined-query failures are

The pre-registered outcome table maps the E2 results as follows:
- **Held-out topologies: H_info in every fold of both E2-0 and E2-full.** GC on held-out do() is ≥ 0.97 in every fold, and the matched COND control is ≥ 0.81.
- **Combined queries, E2-0:** the L2 test passes (GC 1.000 on the frozen and balanced sets, on unseen graphs) while the matched L1 control fails. The table has no cell for that. The model composes two interventions it never saw combined in training, but not two conditioning variables.
- **Combined queries, E2-full:** both fail. By the table this is "H_arch-like (matched L1 also fails)" in folds 0 and 2, the only folds whose frozen held-out split contains combined do() items.
  - The pre-registered safeguard, "no negative conclusion unless the matched COND head passes", therefore blocks an H_1¾ reading.

**Post hoc: the combined-query weakness of the pre-registered runs is a gap in the training distribution, not in capacity or training length.** With samples, closing that gap transfers to most unseen graphs but not all (next subsection). Three pieces of evidence:
1. **Not training length.**
   - In the pre-registered runs, combined-conditioning KL is flat between 20k and 240k steps. For example, E2-full fold 1 seed 0 goes 0.315 → 0.314 → 0.308 → 0.301 → 0.303 bits at 20k / 60k / 120k / 180k / 240k.
   - The 10 E2-full runs that ran to the 240k cap end at 0.27 bits on both topology sets.
   - The convergence labels are unchanged: those runs are "unconverged" under the pre-registered rule, which watches the single-query headline.
2. **Not capacity.** The sweeps (E2-0: 0.8M–14M parameters, 9 sizes; E2-full: 0.8M / 4.8M / 14M) show no trend in combined-query error.
3. **Training on the query type fixes it** (arms 1 and 3). Replacing 25% of training queries with queries whose two elements both matter leaves the eval sets provably unchanged, and:
   - **E2-0:** two-variable conditioning GC 0.98 on training and on unseen graphs (from 0.37 / 0.40). Relaxed combined-do() KL where both pinned variables are parents of the target drops from 0.071 to 0.0001 bits.
   - **E2-full (arm 3), all five folds at 240k steps:**
     - **Training topologies, every fold:** combined-conditioning KL 0.003–0.005 bits (from about 0.28), and balanced strict combined do() GC 0.91–0.94 (pre-registered runs: mean −0.04).
     - **Held-out topologies:** combined-conditioning KL ≤ 0.009 bits in folds 0, 2, 3 and 4. Fold 1 stays at 0.110 bits (slope 0.76).
     - **Held-out strict combined do()**, which only folds 0 and 2 can test: GC 0.89 in fold 0 and 0.25 in fold 2 (balanced sets).
     - **Training length:** the 240k re-runs cut composite error on training topologies about 5× relative to the 36–38k runs. The two held-out failures did not shrink: fold 1's combined conditioning went 0.092 → 0.110 bits (frozen set), and fold 2's combined do() 0.143 → 0.123 bits (balanced set).
   - By the Addendum 3 criteria, 4 of 5 folds pass combined conditioning. The pooled held-out criteria fail: KL 0.026 against ≤ 0.02, driven by fold 1, and combined do() GC 0.51 against ≥ 0.8 (generated tables).

**What this changes:** once trained on two-element queries, a combined query is no longer an *untrained query type*. The held-out graphs are still unseen. With probability tables (arm 1), the two-element computation transfers to every held-out graph. With samples (arm 3), it transfers to most held-out graphs, but not all. Two-variable conditioning is within 0.03 bits for 20 of the 22 held-out topologies that have such items. Combined do() transfers in one of the two folds that can test it.

### Where two-element training does not transfer (post hoc, exploratory)

The by-topology table in the generated block breaks the held-out error down. The two failures are each confined to specific graph shapes, and neither shrank with 6× more training.

- **Fold 1, two-variable conditioning: T9 and T15** carry the error, at 0.32 and 0.09 bits. The fold's other held-out topologies, T11 and T14, are at 0.009 and 0.024.
  - T9 and T15 are the only two of the 25 topologies in which an unshielded collider has a child (0 → 2 ← 1, with 2 → 3).
  - Fold 1 holds out both, so its training set contains no graph with that pattern. That is consistent with the model failing on a structural pattern it never saw. It is an observation, not a tested explanation.
  - The CPT-token model (E2-0 arm 1) handles the same held-out graphs at 0.008 and 0.035 bits.
  - Fold 1's held-out composite error *rises* during training: 0.04–0.06 bits over the first 24k steps (minimum 0.040), then 0.110 at 240k (figure `e2_two_element`, right panel).
- **Fold 2, combined do(): T24.**
  - Across the whole design, exactly one strict combined do() query type exists: target 3 under do(0), do(2), where 1 is a mediator from 0 and a confounder of 2 and 3. It occurs in T20 and T24 only.
  - Both graphs give the same truncated-factorization answer, Σ₁ P(1 | 0) P(3 | 1, 2). They differ only by T24's extra edge 0 → 2, into an intervened node.
  - Fold 0 trains on the T24 version and generalizes to T20 (GC 0.89). Fold 2 trains on the T20 version and fails on T24 (GC 0.25). Its answers sit about a quarter of the way from the conditioning answer to the interventional one (slope 0.23).
  - On those same T24 graphs, two-variable conditioning passes (0.002 bits). The CPT-token model gets the T24 items right (0.0001 bits).
  - So in this one fold, the pattern the pre-registered outcome table assigns to H_1¾ appears: combined interventions fail on an unseen graph while the matched conditioning query passes and in-distribution passes. It is one seed, one topology, one query type, in a post-hoc arm, so it is reported as an observation, not as a verdict.

**What it doesn't say:** nothing here is about learning causal structure from observational data. In E2 and E2-0 the graph is given, and training targets are the true interventional quantities.

## Reading E2c

E2c is the closest test of Pearl's "1¾" remark: no graph, data from two tried experiments, and questions about an untried combination and an untried single intervention. The transformer's KL to the ideal learner is about 0.003 bits on the untried combination. That compares with 0.092 for the best model-free learner, and 0.014 for a learner that knows the *true* graph and plugs in posterior-mean parameters (a different estimator from the Bayes average over graphs, hence its nonzero distance).

The causal assumptions entered through the training distribution: a prior over DAGs with in-degree ≤ 2 and BDeu parameters, with true interventional targets. In Pearl's terms the model was given a *model class*, not a model. It did not get causal knowledge from data alone, which the CHT rules out for any learner.

## Convergence

| | Runs | Converged (plateau) | Hit the cap |
|---|---|---|---|
| E1 aware / shuffled | 5 / 5 | 2 / 5 | 3 / 0 |
| E2-0 / E2-full | 15 / 15 | 4 / 5 | 11 / 10 |
| E2c | 3 | 0 | 3 |
| E3 (3 conditions) | 15 | 15 | 0 |
| E4 main / bins256 | 5 / 5 | 4 / 4 | 1 / 1 |
| E5a / E5b | 6 / 6 | 0 | 6 / 6 |

Under the pre-registered policy, capped runs are never counted as negative evidence, so the passes of capped runs stand.

The plateau rule watches each experiment's headline metric. For E2 that is single-query KL, which levels off long before composite-query performance does. Two consequences:
- it stopped arm 3's folds 0–2 early, so they were re-run to the cap (Addendum 4);
- it makes "converged" uninformative about composite queries. That is why the flat composite trajectories above are reported alongside the labels.

## Deviations from the plan, and why

1. **E2 combined sets (Addendum 1 A).** The frozen combined split holds 0–53 do() items per fold, and none in held-out folds 1, 3 and 4, because only 2 of 25 topologies admit a do() query passing the pre-registered acceptance rules. Balanced re-evaluation sets were added (`scripts/reeval_e2.py`); the frozen-set numbers remain the pre-registered result.
2. **E2-full oracle bug (Addendum 2).** Beta(¼, ¼) underflow gave 0/0 in the Monte Carlo conditional oracles for 1 of 1,000 combined items in folds 2–3. This was fixed, and only the NaN entries were recomputed (`scripts/rescore_e2_samples.py`). All E2-full numbers here use the repaired sets.
3. **E4 coverage (Addendum 1 A).** Bin-edge interval ends over-cover even for the exact oracle. The interpolated version is reported as a labeled diagnostic; the as-implemented FAIL stands.
4. **Capacity sweep seeds (Addendum 1 B).** 2 seeds, unconditionally.
5. **Arm-3 stopping rule (Addendum 4).** Folds 0–2 were re-run with `min_steps` = cap (wave2c). As fixed in advance, the 240k runs are the primary rows whatever they showed, and the 36–38k runs are kept as a comparison.
6. **Correction to Addendum 2.** With the main analysis's aggregation, the 256-bin coverage prediction failed in 3 of 4 cells, not 2 (HID-MED by 0.0001).

## Predictions that failed

- **E2-0:** the matched L1 control for combined queries, two-variable conditioning (GC 0.38 / 0.40 against ≥ 0.8).
- **E2-full:** combined interventions (GC 0.22 / 0.11) and two-variable conditioning (KL ≈ 0.28 bits).
- **E4:**
  - OBS-CONF sd ratio, max |log ratio| 0.163 against ≤ 0.140; the excess is concentrated at k = 256;
  - as-implemented coverage in all four cells.
- **E4 arm 2:** the 256-bin head *increased* the sd excess (1.62 against 1.18). The posterior core matches the oracle (80%-width and IQR ratios 0.99–1.06), so the excess is a far-tail floor (0.5–0.7% of mass beyond 5 oracle sd). As-implemented coverage stayed outside the band in 3 of 4 cells.
- **E5a:** with flags zeroed, KL to flag-blind Bayes-L1 is 0.47 (restart) and 0.50 (clock) bits against ≤ 0.05.
- **E5b, held-out programs:** I predicted factual prediction would generalize. It did not: the KL ratio held-out / seen is about 170, against ≤ 2.
  - The model matches the Bayes oracle *under the training prior*, which itself gives held-out programs zero prior mass.
  - So the pre-registered L3-specific-deficit reading does not apply.
- **E6, Haiku 4.5:** it did better than predicted on V1-blind, V2-stated, V2-unstated and V3 "not identifiable".
- **Arm 3 (E2-full two-element training, post hoc, 240k steps):** pooled held-out combined-conditioning KL 0.026 against ≤ 0.02, because of fold 1 (0.110; the other four folds are ≤ 0.009). Pooled held-out combined do() GC 0.51 against ≥ 0.8: fold 0 0.88, fold 2 0.13 (frozen sets; balanced 0.89 / 0.25). Addendum 4's expectation for folds 0 and 2 (KL ≤ 0.01 and slope ≥ 0.95 on combined conditioning) held.

## E6: Claude verbal probe (descriptive)

800 items per model, 100 contexts per condition. All four models answered all but a handful (1 refusal, 3 parse failures in total). Full table: [`results/e6/scores.md`](results/e6/scores.md).

| Metric | Ideal | Opus 5.5 | Sonnet 5.5 | Haiku 4.5 | Fable 5.1 |
|---|---|---|---|---|---|
| V1 ordering stated: split slope | 1 | 1.011 | 1.001 | 0.776 | 1.016 |
| V1 ordering stated: split R² | 1 | 0.998 | 0.919 | 0.251 | 0.934 |
| V1 no ordering: says "not determinable" | 1 | 1.00 | 1.00 | 0.47 | 1.00 |
| V2 ρ stated: ρ-tracking slope | 1 | 1.000 | 0.903 | 1.000 | 0.951 |
| V2 ρ unstated: crossing slope (should be 0) | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| V3 Z unmeasured: says "not identifiable" | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| V3 Z unmeasured: bounds correct | 1 | 1.00 | 1.00 | 0.03 | 1.00 |

These are verbal answers to stated problems. They say nothing about how any model represents causal structure internally, and the small-model results above say nothing about LLMs.

## Claims this report does not make

- It does not refute Pearl or the CHT; the design relies on the CHT.
- The models did not learn causal effects from observational data; the training targets were true higher-rung quantities.
- Nothing is "resolved" about Simpson's paradox; the model is told the ordering.
- Nothing about internal abduction or self-knowledge; only input–output behavior is measured.
- Nothing confirms Mossé–Ibeling–Icard, whose result is about satisfiability complexity.
- Nothing beyond the truncated factorization, beyond the training range of N, or about LLMs from the small-model results.

## Reproduce

```
uv sync --group dev && uv run pytest
uv run python -m witnesses.train --config configs/<exp>/<cond>.yaml --seed 0   # --smoke for a 1-minute check
uv run python scripts/run_queue.py configs/queues/wave1.yaml                    # the queues used here
uv run python -m witnesses.analyze && uv run python -m witnesses.plots
```

Checkpoints are not in the repository. Every number above is recomputable from `results/*/*/metrics*.json` and `eval_preds*.npz`.

## Generated tables

<!-- BEGIN GENERATED TABLES -->
Generated by `python -m witnesses.analyze`. Estimates are means over seeds; brackets are 95% bootstrap CIs.

### E1 — Simpson pair over six orderings (demonstration)

Runs: {'aware': '5 runs (2 converged, 3 hit the cap)', 'shuffled': '5 runs (5 converged, 0 hit the cap)'}

| criterion | estimate | threshold | verdict |
|---|---|---|---|
| split slope (aware) | 1.000 [0.999, 1.001] | 0.9–1.1 | PASS |
| split R² (aware) | 0.999 [0.999, 1.000] | ≥ 0.9 | PASS |
| GC_L2 vs ordering-blind Bayes (aware) | 0.999 [0.999, 0.999] | ≥ 0.85 | PASS |
| split slope (shuffled-ordering control) | 0.000 [-0.000, 0.000] | |·| ≤ 0.05 | PASS |
| Simpson-reversal sign agreement, ADJ ordering | 0.995 [0.992, 0.997] | ≥ 0.95 | PASS |
| Simpson-reversal sign agreement, CRUDE ordering | 1.000 [1.000, 1.000] | ≥ 0.95 | PASS |
| L2 − FUNC KL, ADJ (bits) | -0.0000 [-0.0000, -0.0000] | CI contains 0 | PASS |
| L2 − FUNC KL, CRUDE (bits) | 0.0000 [0.0000, 0.0000] | CI contains 0 | PASS |
| L2 − FUNC KL, MARG (bits) | 0.0000 [0.0000, 0.0000] | CI contains 0 | PASS |

### E2-0 — graph + CPTs given (the test)

| fold | GC in-dist | GC held-out (do) | GC held-out (COND) | GC combined held-out | swap slope | outcome (held-out) | outcome (combined) |
|---|---|---|---|---|---|---|---|
| 0 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.993 [0.991, 0.995] | 1.000 [1.000, 1.000] | 0.999 [0.998, 1.000] | H_info | H_info |
| 1 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.995 [0.992, 0.997] | n/a | 1.000 [0.999, 1.001] | H_info | inconclusive |
| 2 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.996 [0.994, 0.996] | 1.000 [1.000, 1.000] | 1.000 [0.999, 1.000] | H_info | H_info |
| 3 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.989 [0.985, 0.992] | n/a | 1.000 [1.000, 1.000] | H_info | inconclusive |
| 4 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.993 [0.985, 0.998] | n/a | 1.000 [1.000, 1.000] | H_info | inconclusive |

| criterion | estimate | threshold | verdict |
|---|---|---|---|
| GC, in-distribution do() (pooled over folds) | 1.000 [1.000, 1.000] | ≥ 0.85 | PASS |
| GC, held-out topologies do() (pooled over folds) | 1.000 [1.000, 1.000] | ≥ 0.8 | PASS |
| GC, held-out topologies, matched L1 (COND) (pooled over folds) | 0.993 [0.991, 0.995] | ≥ 0.8 | PASS |
| GC, combined interventions (training topologies) (pooled over folds) | 1.000 [1.000, 1.000] | ≥ 0.8 | PASS |
| GC, combined interventions (held-out topologies) (pooled over folds) | 1.000 [1.000, 1.000] | ≥ 0.8 | PASS |
| slope (model−L1) on (L2−L1), held-out (pooled over folds) | 1.000 [1.000, 1.000] | ≥ 0.8 | PASS |
| slope (model−L1) on (L2−L1), combined, held-out (pooled over folds) | 0.999 [0.998, 1.000] | ≥ 0.8 | PASS |
| GC, combined, matched L1 (two-variable conditioning), training topologies (pooled over folds) | 0.378 [0.320, 0.441] | ≥ 0.8 | FAIL |
| GC, combined, matched L1 (two-variable conditioning), held-out topologies (pooled over folds) | 0.398 [0.334, 0.453] | ≥ 0.8 | FAIL |
| same-data Markov-equivalent graph swap: split slope (pooled over folds) | 1.000 [0.999, 1.000] | ≥ 0.8 | PASS |
| KL to Bayes, combined two-variable conditioning, training topologies (bits) | 0.4182 [0.3708, 0.4636] | (reported) |  |
| KL to Bayes, combined two-variable conditioning, held-out topologies (bits) | 0.4063 [0.3681, 0.4550] | (reported) |  |
| [balanced] combined do(A,B), strict rules, training topologies (~1000 items/run) | 1.000 [1.000, 1.000] | ≥ 0.8 | PASS |
| [balanced] matched L1: conditioning on two variables, training topologies | 0.371 [0.304, 0.445] | ≥ 0.8 | FAIL |
| [balanced] combined do(A,B), strict rules, held-out topologies (~400 items/run) | 1.000 [1.000, 1.000] | ≥ 0.8 | PASS |
| [balanced] matched L1: conditioning on two variables, held-out topologies | 0.397 [0.342, 0.451] | ≥ 0.8 | FAIL |
| [exploratory] relaxed combined do(), train: both pinned variables are parents of T — KL (bits) | 0.0723 [0.0446, 0.1003] | (reported) |  |
| [exploratory] relaxed combined do(), train: other structures — KL (bits) | 0.0073 [0.0040, 0.0110] | (reported) |  |
| [exploratory] relaxed combined do(), heldout: both pinned variables are parents of T — KL (bits) | 0.0706 [0.0458, 0.0953] | (reported) |  |
| [exploratory] relaxed combined do(), heldout: other structures — KL (bits) | 0.0044 [0.0013, 0.0081] | (reported) |  |

### E2-full — graph + samples given (the test)

Scored on the frozen sets with the NaN oracle entries repaired (PREDICTIONS.md, Addendum 2; 15/15 runs re-scored).

| fold | GC in-dist | GC held-out (do) | GC held-out (COND) | GC combined held-out | swap slope | outcome (held-out) | outcome (combined) |
|---|---|---|---|---|---|---|---|
| 0 | 0.989 [0.974, 0.997] | 0.983 [0.969, 0.991] | 0.860 [0.645, 0.971] | 0.450 [0.186, 0.583] | 0.971 [0.957, 0.977] | H_info | H_arch-like (matched L1 also fails) |
| 1 | 0.982 [0.961, 0.994] | 0.978 [0.967, 0.984] | 0.811 [0.809, 0.814] | n/a | 0.995 [0.994, 0.997] | H_info | inconclusive |
| 2 | 0.996 [0.995, 0.996] | 0.987 [0.982, 0.993] | 0.976 [0.973, 0.978] | -0.238 [-0.490, 0.110] | 0.985 [0.983, 0.989] | H_info | H_arch-like (matched L1 also fails) |
| 3 | 0.974 [0.931, 0.996] | 0.984 [0.957, 0.998] | 0.939 [0.866, 0.977] | n/a | 0.981 [0.948, 0.999] | H_info | inconclusive |
| 4 | 0.994 [0.991, 0.996] | 0.993 [0.987, 0.998] | 0.974 [0.970, 0.977] | n/a | 0.994 [0.988, 0.997] | H_info | inconclusive |

| criterion | estimate | threshold | verdict |
|---|---|---|---|
| GC, in-distribution do() (pooled over folds) | 0.987 [0.977, 0.995] | ≥ 0.85 | PASS |
| GC, held-out topologies do() (pooled over folds) | 0.985 [0.979, 0.991] | ≥ 0.8 | PASS |
| GC, held-out topologies, matched L1 (COND) (pooled over folds) | 0.912 [0.861, 0.956] | ≥ 0.8 | PASS |
| GC, combined interventions (training topologies) (pooled over folds) | 0.221 [0.074, 0.368] | ≥ 0.8 | FAIL |
| GC, combined interventions (held-out topologies) (pooled over folds) | 0.106 [-0.225, 0.430] | ≥ 0.8 | FAIL |
| slope (model−L1) on (L2−L1), held-out (pooled over folds) | 0.997 [0.995, 1.000] | ≥ 0.8 | PASS |
| slope (model−L1) on (L2−L1), combined, held-out (pooled over folds) | 0.428 [0.272, 0.597] | ≥ 0.8 | FAIL |
| GC, combined, matched L1 (two-variable conditioning), training topologies (pooled over folds) | -6.205 [-6.601, -5.791] | ≥ 0.8 | FAIL |
| GC, combined, matched L1 (two-variable conditioning), held-out topologies (pooled over folds) | -5.842 [-6.269, -5.434] | ≥ 0.8 | FAIL |
| same-data Markov-equivalent graph swap: split slope (pooled over folds) | 0.985 [0.978, 0.992] | ≥ 0.8 | PASS |
| KL to Bayes, combined two-variable conditioning, training topologies (bits) | 0.2781 [0.2621, 0.2925] | (reported) |  |
| KL to Bayes, combined two-variable conditioning, held-out topologies (bits) | 0.2758 [0.2528, 0.3010] | (reported) |  |
| [balanced] combined do(A,B), strict rules, training topologies (~300 items/run) | -0.039 [-0.206, 0.144] | ≥ 0.8 | FAIL |
| [balanced] matched L1: conditioning on two variables, training topologies | -5.755 [-6.170, -5.297] | ≥ 0.8 | FAIL |
| [balanced] combined do(A,B), strict rules, held-out topologies (~120 items/run) | 0.175 [0.021, 0.340] | ≥ 0.8 | FAIL |
| [balanced] matched L1: conditioning on two variables, held-out topologies | -6.262 [-6.773, -5.663] | ≥ 0.8 | FAIL |
| [exploratory] relaxed combined do(), train: both pinned variables are parents of T — KL (bits) | 0.4142 [0.3370, 0.4889] | (reported) |  |
| [exploratory] relaxed combined do(), train: other structures — KL (bits) | 0.2255 [0.1955, 0.2552] | (reported) |  |
| [exploratory] relaxed combined do(), heldout: both pinned variables are parents of T — KL (bits) | 0.4088 [0.3402, 0.4776] | (reported) |  |
| [exploratory] relaxed combined do(), heldout: other structures — KL (bits) | 0.1964 [0.1666, 0.2295] | (reported) |  |

### E2c — no graph; tried do(A), do(B) (the test)

Runs: 3 runs (0 converged, 3 hit the cap)

| criterion | estimate | threshold | verdict |
|---|---|---|---|
| GC vs model-free '1¾' baseline, combo | 0.971 [0.969, 0.974] | ≥ 0.8 | PASS |
|   … sharp half (graph-posterior sd) | 0.980 [0.979, 0.982] | (reported) |  |
|   … diffuse half (graph-posterior sd) | 0.961 [0.955, 0.964] | (reported) |  |
| GC vs model-free '1¾' baseline, untried | 0.954 [0.949, 0.956] | ≥ 0.8 | PASS |
|   … sharp half (graph-posterior sd) | 0.957 [0.955, 0.958] | (reported) |  |
|   … diffuse half (graph-posterior sd) | 0.952 [0.948, 0.956] | (reported) |  |
| tried regimes: KL(model) − KL(baseline) (bits) | -0.0409 [-0.0410, -0.0407] | ≤ 0 | PASS |

### E3 — residual twin with continuous ρ (demonstration)

Runs: {'aware': '5 runs (5 converged, 0 hit the cap)', 'shuffled': '5 runs (5 converged, 0 hit the cap)', 'l2trained': '5 runs (5 converged, 0 hit the cap)'}

| criterion | estimate | threshold | verdict |
|---|---|---|---|
| mean |slope − ρ|, training ρ bins | 0.009 [0.005, 0.013] | ≤ 0.05 | PASS |
| mean |slope − ρ|, held-out ρ bins (|ρ|<0.3) | 0.033 [0.011, 0.074] | ≤ 0.10 | PASS |
| same-side slope | 0.999 [0.996, 1.001] | 1 ± 0.05 | PASS |
| normalized excess MSE (cf head) | 0.057 [0.042, 0.073] | ≤ 0.10 | PASS |
| control 'shuffled': max |slope| over ρ bins | 0.005 [0.004, 0.006] | ≤ 0.05 | PASS |
| control 'l2trained': max |slope| over ρ bins | 0.003 [0.003, 0.004] | ≤ 0.05 | PASS |

### E4 — identifiability tracking (demonstration)

Runs: 5 runs (4 converged, 1 hit the cap)

| criterion | estimate | threshold | verdict |
|---|---|---|---|
| log-log slope of posterior sd vs k, OBS_CONF (oracle -0.48) | -0.412 [-0.419, -0.405] | [-0.6, -0.35] | PASS |
| max |log sd ratio| over k ≥ 16, OBS_CONF | 0.163 [0.146, 0.185] | ≤ 0.140 (ratio in 0.85–1.15) | FAIL |
| 80% coverage over k ≥ 16, OBS_CONF — as implemented (bin-edge rounding) | 0.821 – 0.866 | within 0.77–0.83 | FAIL |
| 80% coverage over k ≥ 16, OBS_CONF — interpolated (oracle: 0.789–0.811) | 0.790 – 0.815 | within 0.77–0.83 | PASS |
| log-log slope of posterior sd vs k, OBS_MED (oracle -0.48) | -0.426 [-0.431, -0.419] | [-0.6, -0.35] | PASS |
| max |log sd ratio| over k ≥ 16, OBS_MED | 0.124 [0.112, 0.138] | ≤ 0.140 (ratio in 0.85–1.15) | PASS |
| 80% coverage over k ≥ 16, OBS_MED — as implemented (bin-edge rounding) | 0.816 – 0.870 | within 0.77–0.83 | FAIL |
| 80% coverage over k ≥ 16, OBS_MED — interpolated (oracle: 0.782–0.808) | 0.785 – 0.813 | within 0.77–0.83 | PASS |
| log-log slope of posterior sd vs k, HID_MED (oracle -0.48) | -0.429 [-0.433, -0.421] | [-0.6, -0.35] | PASS |
| max |log sd ratio| over k ≥ 16, HID_MED | 0.125 [0.114, 0.140] | ≤ 0.140 (ratio in 0.85–1.15) | PASS (CI crosses) |
| 80% coverage over k ≥ 16, HID_MED — as implemented (bin-edge rounding) | 0.825 – 0.865 | within 0.77–0.83 | FAIL |
| 80% coverage over k ≥ 16, HID_MED — interpolated (oracle: 0.793–0.810) | 0.794 – 0.809 | within 0.77–0.83 | PASS |
| log-log slope of posterior sd vs k, HID_CONF (oracle -0.13) | -0.122 [-0.125, -0.118] | [-0.25, 0.0] | PASS |
| max |log sd ratio| over k ≥ 16, HID_CONF | 0.020 [0.014, 0.029] | ≤ 0.140 (ratio in 0.85–1.15) | PASS |
| 80% coverage over k ≥ 16, HID_CONF — as implemented (bin-edge rounding) | 0.829 – 0.850 | within 0.77–0.83 | FAIL |
| 80% coverage over k ≥ 16, HID_CONF — interpolated (oracle: 0.805–0.812) | 0.803 – 0.811 | within 0.77–0.83 | PASS |

### E5 — witness pairs in Misra's task

| criterion | estimate | threshold | verdict |
|---|---|---|---|
| E5a restart: GC_TV (programs, a≠0) | 0.992 [0.990, 0.996] | ≥ 0.85 | PASS |
| E5a restart: flags zeroed → KL to Bayes-L1 (bits) | 0.472 [0.296, 0.686] | ≤ 0.05 | FAIL |
| E5a clock: GC_TV (programs, a≠0) | 0.998 [0.998, 0.998] | ≥ 0.85 | PASS |
| E5a clock: flags zeroed → KL to Bayes-L1 (bits) | 0.498 [0.350, 0.588] | ≤ 0.05 | FAIL |
| E5b cf_abducted: GC_TV noise episodes | 1.000 [1.000, 1.000] | ≥ 0.85 | PASS |
| E5b cf_abducted: GC_TV program episodes | 0.997 [0.994, 0.998] | ≥ 0.85 | PASS |
| E5b cf_abducted: GC_TV held-out programs | -0.448 [-1.198, 0.234] | (reported) |  |
| E5b cf_abducted: factual KL held-out / seen | 174.742 [163.429, 182.117] | ≤ 2 (L1 generalizes) | FAIL |
| E5b cf_fresh: GC_TV noise episodes | 0.999 [0.999, 0.999] | ≥ 0.85 | PASS |
| E5b cf_fresh: GC_TV program episodes | 0.988 [0.986, 0.990] | ≥ 0.85 | PASS |
| E5b cf_fresh: GC_TV held-out programs | -0.078 [-0.391, 0.123] | (reported) |  |
| E5b cf_fresh: factual KL held-out / seen | 166.242 [153.162, 187.718] | ≤ 2 (L1 generalizes) | FAIL |

### Post hoc (Addendum 1 C) — E2-0 with two-element training queries

Runs: 5 runs (1 converged, 4 hit the cap); one seed per fold, CIs resample runs

| criterion | estimate | threshold | verdict |
|---|---|---|---|
| [balanced] two-variable conditioning GC, training topologies (base E2-0: 0.371 [0.305, 0.442]) | 0.979 [0.976, 0.982] | ≥ 0.9 | prediction held |
| [balanced] two-variable conditioning GC, held-out topologies (base E2-0: 0.397 [0.341, 0.451]) | 0.981 [0.977, 0.984] | ≥ 0.9 | prediction held |
| [balanced] relaxed combined do(), both pinned variables parents of T, held-out — KL bits (base E2-0: 0.0706 [0.0450, 0.0943]) | 0.0001 [0.0001, 0.0001] | ≤ 0.01 | prediction held |
| no single-query regression: GC, in-distribution do() | 1.000 [1.000, 1.000] | ≥ 0.99 | prediction held |
| no single-query regression: GC, held-out do() | 1.000 [1.000, 1.000] | ≥ 0.99 | prediction held |
| no single-query regression: GC, held-out COND | 0.990 [0.985, 0.995] | ≥ 0.99 | prediction held (CI crosses) |

### Post hoc (Addenda 3–4) — E2-full with two-element training queries

Runs: 5 runs (0 converged, 5 hit the cap)

Primary runs: 4/5 folds meet the Addendum 3 combined-conditioning criteria. One seed per fold; CIs resample runs. Second table (post hoc, exploratory): balanced held-out two-element KL by topology. 'collider with a child' = an unshielded collider whose middle node has a child; 'in training' = some training topology of that fold has one too.

| fold | run | steps (stop) | cond slope train / held-out | cond KL train / held-out (bits) | combined do() GC train / held-out | [balanced] strict combined do() GC train / held-out | single-query GC in-dist / held-out do() | Addendum 3 cond criteria |
|---|---|---|---|---|---|---|---|---|
| 0 | 240k re-run (samples_f0_comp25full) | 240k (cap) | 0.994 / 0.986 | 0.0033 / 0.0086 | 0.95 / 0.88 | 0.94 / 0.89 | 0.996 / 0.951 | PASS |
| 0 | arm 3 (samples_f0_comp25) | 36k (plateau) | 0.973 / 0.971 | 0.0166 / 0.0228 | -0.34 / 0.39 | 0.05 / 0.38 | 0.970 / 0.941 | miss |
| 1 | 240k re-run (samples_f1_comp25full) | 240k (cap) | 0.994 / 0.761 | 0.0037 / 0.1097 | 0.95 / nan | 0.94 / nan | 0.992 / 0.988 | miss |
| 1 | arm 3 (samples_f1_comp25) | 36k (plateau) | 0.960 / 0.812 | 0.0227 / 0.0919 | 0.56 / nan | 0.22 / nan | 0.923 / 0.951 | miss |
| 2 | 240k re-run (samples_f2_comp25full) | 240k (cap) | 0.992 / 0.991 | 0.0039 / 0.0042 | 0.96 / 0.13 | 0.92 / 0.25 | 0.993 / 0.989 | PASS |
| 2 | arm 3 (samples_f2_comp25) | 38k (plateau) | 0.964 / 0.951 | 0.0197 / 0.0216 | 0.57 / -0.27 | 0.33 / 0.13 | 0.969 / 0.956 | miss |
| 3 | arm 3 (samples_f3_comp25) | 240k (cap) | 0.989 / 0.984 | 0.0041 / 0.0061 | 0.92 / nan | 0.93 / nan | 0.989 / 0.992 | PASS |
| 4 | arm 3 (samples_f4_comp25) | 240k (cap) | 0.992 / 0.994 | 0.0046 / 0.0030 | 0.85 / nan | 0.91 / nan | 0.994 / 0.983 | PASS |

| fold | held-out topology | collider with a child | query | n | KL arm 3 stopped at 36–38k (bits) | KL at 240k (bits) | E2-0 arm 1, CPT tokens (bits) |
|---|---|---|---|---|---|---|---|
| 0 | T13 (3 edges) | no | two-variable conditioning | 60 | 0.0293 | 0.0085 | 0.0020 |
| 0 | T19 (4 edges) | no | two-variable conditioning | 129 | 0.0207 | 0.0085 | 0.0223 |
| 0 | T20 (4 edges) | no | two-variable conditioning | 111 | 0.0174 | 0.0058 | 0.0170 |
| 0 | T20 (4 edges) | no | strict combined do() | 300 | 0.0942 | 0.0162 | 0.0001 |
| 1 | T9 (3 edges) | yes (not in training) | two-variable conditioning | 72 | 0.2644 | 0.3211 | 0.0076 |
| 1 | T11 (3 edges) | no | two-variable conditioning | 43 | 0.0300 | 0.0087 | 0.0014 |
| 1 | T14 (4 edges) | no | two-variable conditioning | 104 | 0.0285 | 0.0242 | 0.0057 |
| 1 | T15 (4 edges) | yes (not in training) | two-variable conditioning | 81 | 0.0738 | 0.0904 | 0.0351 |
| 2 | T5 (2 edges) | no | two-variable conditioning | 25 | 0.0407 | 0.0022 | 0.0033 |
| 2 | T8 (3 edges) | no | two-variable conditioning | 83 | 0.0284 | 0.0088 | 0.0218 |
| 2 | T12 (3 edges) | no | two-variable conditioning | 56 | 0.0209 | 0.0054 | 0.0009 |
| 2 | T21 (4 edges) | no | two-variable conditioning | 60 | 0.0179 | 0.0035 | 0.0060 |
| 2 | T24 (5 edges) | no | two-variable conditioning | 76 | 0.0077 | 0.0021 | 0.0154 |
| 2 | T24 (5 edges) | no | strict combined do() | 300 | 0.1427 | 0.1226 | 0.0001 |
| 3 | T3 (2 edges) | no | two-variable conditioning | 27 | — | 0.0022 | 0.0026 |
| 3 | T6 (3 edges) | no | two-variable conditioning | 79 | — | 0.0278 | 0.0103 |
| 3 | T10 (3 edges) | no | two-variable conditioning | 43 | — | 0.0048 | 0.0035 |
| 3 | T16 (4 edges) | no | two-variable conditioning | 75 | — | 0.0031 | 0.0133 |
| 3 | T23 (5 edges) | no | two-variable conditioning | 76 | — | 0.0016 | 0.0142 |
| 4 | T2 (2 edges) | no | two-variable conditioning | 56 | — | 0.0027 | 0.0133 |
| 4 | T7 (3 edges) | no | two-variable conditioning | 46 | — | 0.0009 | 0.0213 |
| 4 | T17 (4 edges) | no | two-variable conditioning | 67 | — | 0.0071 | 0.0101 |
| 4 | T18 (4 edges) | no | two-variable conditioning | 61 | — | 0.0028 | 0.0108 |
| 4 | T22 (5 edges) | no | two-variable conditioning | 70 | — | 0.0020 | 0.0071 |

| criterion | estimate | threshold | verdict |
|---|---|---|---|
| combined conditioning slope (model−L2) on (L1−L2), training topologies (base E2-full: 0.369 [0.343, 0.394]) | 0.992 [0.990, 0.994] | ≥ 0.9 | prediction held |
| combined conditioning slope (model−L2) on (L1−L2), held-out topologies (base E2-full: 0.393 [0.358, 0.423]) | 0.943 [0.852, 0.991] | ≥ 0.9 | prediction held (CI crosses) |
| combined conditioning KL to Bayes (bits), training topologies (base E2-full: 0.2781 [0.2617, 0.2931]) | 0.0039 [0.0036, 0.0043] | ≤ 0.02 | prediction held |
| combined conditioning KL to Bayes (bits), held-out topologies (base E2-full: 0.2758 [0.2549, 0.2983]) | 0.0263 [0.0041, 0.0684] | ≤ 0.02 | prediction failed (CI crosses) |
| combined do() GC, training topologies (folds whose frozen split has do() items) | 0.926 [0.884, 0.953] | ≥ 0.8 | prediction held |
| combined do() GC, held-out topologies (folds whose frozen split has do() items) | 0.505 [0.129, 0.880] | ≥ 0.8 | prediction failed (CI crosses) |
| no single-query regression: GC, in-distribution do() | 0.993 [0.991, 0.995] | ≥ 0.85 | prediction held |
| no single-query regression: GC, held-out do() | 0.981 [0.966, 0.990] | ≥ 0.8 | prediction held |
| [balanced] two-variable conditioning, training topologies — KL bits (base E2-full: 0.2878 [0.2679, 0.3062]) | 0.0037 [0.0031, 0.0044] | (reported) |  |
| [balanced] strict combined do(A, B) GC, training topologies (base E2-full: -0.039 [-0.210, 0.140]) | 0.926 [0.918, 0.936] | (reported) |  |
| [balanced] two-variable conditioning, held-out topologies — KL bits (base E2-full: 0.2728 [0.2523, 0.2931]) | 0.0272 [0.0047, 0.0692] | (reported) |  |
| [balanced] strict combined do(A, B) GC, held-out topologies (base E2-full: 0.175 [0.012, 0.336]) | 0.573 [0.252, 0.894] | (reported) |  |
| (control) pre-registered E2-full runs that also ran to the 240k cap (10): combined conditioning KL, training topologies (bits) | 0.2750 [0.2560, 0.2930] | (reported) |  |
| (control) pre-registered E2-full runs that also ran to the 240k cap (10): combined conditioning KL, held-out topologies (bits) | 0.2658 [0.2405, 0.2970] | (reported) |  |

### Capacity sweep — E2-0

Folds 0-1, 2 seeds per size (Addendum 1 B); means over runs.

| size | params | runs (converged) | GC in-dist do() | GC held-out do() | GC held-out COND | combined cond KL train / held-out (bits) | [balanced] two-variable cond held-out: GC / KL (bits) |
|---|---|---|---|---|---|---|---|
| d128L4 | 0.8M | 4 (1) | 0.998 | 0.998 | 0.939 | 0.3857 / 0.3342 | 0.470 / 0.3481 |
| d128L6 | 1.2M | 4 (0) | 1.000 | 1.000 | 0.981 | 0.4402 / 0.4061 | 0.434 / 0.3732 |
| d128L8 | 1.6M | 4 (0) | 1.000 | 0.996 | 0.973 | 0.5006 / 0.4715 | 0.267 / 0.4843 |
| d256L4 | 3.2M | 4 (1) | 0.999 | 0.997 | 0.965 | 0.3394 / 0.3184 | 0.524 / 0.3133 |
| d256L6 (base) | 4.8M | 4 (2) | 1.000 | 1.000 | 0.993 | 0.4406 / 0.4089 | 0.361 / 0.4222 |
| d256L8 | 6.3M | 4 (1) | 1.000 | 1.000 | 0.992 | 0.5002 / 0.4346 | 0.345 / 0.4313 |
| d384L4 | 7.1M | 4 (2) | 1.000 | 0.999 | 0.974 | 0.3950 / 0.3732 | 0.435 / 0.3741 |
| d384L6 | 10.7M | 4 (1) | 1.000 | 0.998 | 0.987 | 0.4261 / 0.4078 | 0.377 / 0.4134 |
| d384L8 | 14.2M | 4 (1) | 1.000 | 1.000 | 0.992 | 0.4548 / 0.4039 | 0.404 / 0.3949 |

### Capacity sweep — E2-full

Folds 0-1, 2 seeds per size (Addendum 1 B); means over runs.

| size | params | runs (converged) | GC in-dist do() | GC held-out do() | GC held-out COND | combined cond KL train / held-out (bits) | [balanced] two-variable cond held-out: GC / KL (bits) |
|---|---|---|---|---|---|---|---|
| d128L4 | 0.8M | 4 (3) | 0.962 | 0.961 | 0.847 | 0.2672 / 0.2628 | -5.771 / 0.2759 |
| d256L6 (base) | 4.8M | 4 (2) | 0.981 | 0.977 | 0.809 | 0.2995 / 0.2847 | -6.184 / 0.2927 |
| d384L8 | 14.3M | 4 (2) | 0.984 | 0.969 | 0.839 | 0.3063 / 0.2863 | -6.334 / 0.2986 |

### Post hoc (Addendum 1 D) — E4 with a 256-bin head

Diagnostics at k = 256, means over 5 seeds; widths from interpolated quantiles. Coverage uses the main E4 analysis's aggregation (each seed's min/max over k, then the mean over seeds); Addendum 2 averaged over seeds first and so reported 2 of 4 cells outside the band, not 3 (HID-MED: 0.8292 vs 0.8301).

| head | cell | steps (stop) | sd ratio k=256 | 80%-width ratio | IQR ratio | mass beyond 5 oracle sd (model / oracle) | posterior-mean RMSE ratio |
|---|---|---|---|---|---|---|---|
| 64 bins (main) | OBS_CONF | 38k–120k (cap, plateau) | 1.15–1.22 | 1.021 | 1.011 | 0.18% / 0.01% | 1.007 |
| 64 bins (main) | OBS_MED | 38k–120k (cap, plateau) | 1.11–1.17 | 0.999 | 0.989 | 0.14% / 0.01% | 1.000 |
| 64 bins (main) | HID_MED | 38k–120k (cap, plateau) | 1.12–1.17 | 0.994 | 0.985 | 0.13% / 0.01% | 1.000 |
| 64 bins (main) | HID_CONF | 38k–120k (cap, plateau) | 1.01–1.04 | 0.987 | 1.025 | 0.12% / 0.03% | 1.000 |
| 256 bins | OBS_CONF | 14k–120k (cap, plateau) | 1.45–1.76 | 1.064 | 1.046 | 0.73% / 0.01% | 1.014 |
| 256 bins | OBS_MED | 14k–120k (cap, plateau) | 1.44–1.68 | 1.024 | 1.005 | 0.66% / 0.01% | 1.006 |
| 256 bins | HID_MED | 14k–120k (cap, plateau) | 1.44–1.68 | 1.015 | 0.996 | 0.66% / 0.01% | 1.003 |
| 256 bins | HID_CONF | 14k–120k (cap, plateau) | 1.11–1.15 | 0.988 | 1.035 | 0.46% / 0.03% | 1.001 |

| criterion | estimate | threshold | verdict |
|---|---|---|---|
| OBS-CONF sd ratio at k = 256, 256 bins (64 bins: 1.177 [1.157, 1.201]) | 1.620 [1.519, 1.721] | > 1.10 (excess persists) | prediction held |
| as-implemented 80% coverage over k ≥ 16, OBS_CONF, 256 bins | 0.8002 – 0.8417 | within 0.77–0.83 | prediction failed |
| as-implemented 80% coverage over k ≥ 16, OBS_MED, 256 bins | 0.7936 – 0.8376 | within 0.77–0.83 | prediction failed |
| as-implemented 80% coverage over k ≥ 16, HID_MED, 256 bins | 0.8027 – 0.8301 | within 0.77–0.83 | prediction failed |
| as-implemented 80% coverage over k ≥ 16, HID_CONF, 256 bins | 0.8112 – 0.8208 | within 0.77–0.83 | prediction held |

<!-- END GENERATED TABLES -->
