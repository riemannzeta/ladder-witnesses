# Causal Wind Tunnel: Can a Transformer Learn Layer 2 and Layer 3 Reasoning?

You are setting up a controlled experiment. The goal is to test whether a small transformer, trained with an appropriate input interface and training signal, answers interventional (Layer 2) and counterfactual (Layer 3) queries in Pearl's causal hierarchy. It must match analytic Bayes-optimal answers, including on interventions it never saw in training.

The design must answer Judea Pearl's 2023 objection to reinforcement learning ("level one and three-fourths"). He said such a system cannot infer anything about a move it has never tried, and cannot combine interventions to predict doing both A and B. So the tests that matter most are generalization to **untried intervention positions** and to **composite interventions**.

Work in stages. Stop and report at each checkpoint marked **CHECKPOINT** before continuing.

---

## 0. Background and prior code

Two existing repositories contain the task and model you will build on. Clone them read-only into a scratch directory. **Do not modify them.**

- `https://github.com/riemannzeta/auxiliary-causal`: vendors the recurrence task and `RecurrenceTransformer` (`auxcausal/recurrence_bwt.py`), plus a training loop and an interchange-intervention harness (`auxcausal/interchange.py`).
- `https://github.com/vishalmisra/bayesian-wind-tunnel`: the original task (`experiments/construction_boundary/recurrence_bwt.py`).

**The task.** Sequences over Z_p with p = 17 and length T = 16. Each episode is drawn from one of two hypotheses:
- **H_P (program), prior π:** draw (a, b) uniformly and x₀ uniformly, then x_{t+1} = (a·x_t + b) mod p.
- **H_R (noise), prior 1 − π:** each x_t is i.i.d. uniform.

Before writing any oracle, read the vendored generator and **verify** the exact value of π (expected 0.5), the range of a (expected 0..p−1, including 0), the range of b, and how x₀ is drawn. If anything differs from the above, use the code's actual values and report the difference.

**Why the input interface matters.** An earlier interchange-intervention test on this task returned uniform outputs. The cause was a confound, not a property of the network. Swapping a token at position p creates a transition x_{p−1} → Y that no affine recurrence can produce, so a Bayes-optimal model conditioning on that input must conclude "noise" and predict 1/17 everywhere. The do() answer, a·Y + b, is correct only if the model knows the value was *set* rather than *observed*. This experiment supplies that information through an explicit regime flag, following Dawid's regime indicator.

**Avoid the loss-horizon confound.** In the earlier "wall" experiment, loss was applied only to positions 1–5. As a result, rows 6 and up of the learned positional embedding never received gradient. Every position in this experiment must be supervised, so that no parameter specific to a position goes untrained.

---

## 1. Project setup

- Create a new repository named `causal-wind-tunnel` with `uv` (Python ≥ 3.11). Dependencies: `torch`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `pytest`, `tqdm`.
- Layout:
  ```
  cwt/
    data.py           # generators for observational, interventional, counterfactual episodes
    oracle.py         # exact Bayes oracles: L1, L2 (do), L3 (counterfactual)
    model.py          # RecurrenceTransformer adapted from auxiliary-causal (vendored, with attribution)
    train.py          # training loop, eval hooks, checkpointing
    evaluate.py       # all test conditions; writes metrics JSON
    plots.py          # figures
  configs/            # YAML per condition, plus smoke configs
  scripts/            # run_smoke.sh, run_l2.sh, run_l3.sh
  tests/
  PREDICTIONS.md      # pre-registered predictions (Stage 2)
  REPORT.md           # results (Stage 6)
  ```
- Vendor the model class from `auxiliary-causal` with an attribution comment. Keep its default architecture: confirm d_model, layers, heads, and learned absolute positional embeddings from the source and report them. Add only what this spec requires.
- Detect the hardware (CUDA, MPS, or CPU). Before launching any full run, report the estimated wall-clock time per run and for the whole suite.
- Every run records: config, seed, git commit hash, torch version, device, and the final and best metrics. Save checkpoints at the end and at each evaluation.

---

## 2. Pre-registration — CHECKPOINT

Before writing training code, create `PREDICTIONS.md` and commit it. It must state, for each test in Stages 4 and 5:
- what the **Bayes-L2 / Bayes-L3** answer is,
- what a model reasoning only at **Layer 1** (flag-blind) would output,
- what **Pearl's "1¾" hypothesis** predicts: success on seen interventions, failure on untried and composite ones,
- the numeric pass criterion (below),
- which outcome would falsify which hypothesis.

**Pass criteria.** Report all metrics in bits, as means over 3 seeds with 95% bootstrap CIs.
- A model **tracks** a reference if its mean KL(reference ‖ model) ≤ 0.05 bits at the evaluated positions.
- A model **discriminates** L2 from L1 (or L3 from L2) if its KL to the correct reference is at least 10× smaller than its KL to the wrong one.

Stop and show me `PREDICTIONS.md`.

---

## 3. Bayes oracles and data generators — CHECKPOINT

### 3a. Regime flag interface (Layer 2)

Each position carries a token x_t ∈ Z_p and a binary flag f_t ∈ {0, 1}, where f_t = 1 means x_t was set externally by do(). The model input at position t is `token_emb(x_t) + pos_emb(t) + flag_emb(f_t)`, where `flag_emb` is a new learned 2-row embedding. Sequence length and positions are unchanged; **no inserted tokens.**

**Generative semantics under intervention.** Let I be the set of intervened positions, with I ⊂ {1, …, T−2}.
- **H_P:** for t ∈ I, x_t = v_t (the set value). For t ∉ I, x_t = a·x_{t−1} + b. The recurrence continues from the set value.
- **H_R:** for t ∈ I, x_t = v_t. Otherwise x_t is i.i.d. uniform. The intervention has no downstream effect.

**Critical:** each v_t must be drawn **uniformly and independently of everything**, including the value the process would naturally have produced. Do **not** exclude the natural value. Excluding it would make the experimenter's choice depend on the hidden (a, b), and the cut-arrow oracle would then be wrong. This is the randomization assumption behind do(). Write a test that fails if the generator conditions v on the natural value.

**Loss masking.** The token at an intervened position is exogenous, so mask the loss on the prediction *of* x_t for t ∈ I. All other positions, including those immediately after an intervention, are supervised.

### 3b. Oracles

Implement exact, vectorized numpy oracles by enumerating all p² programs plus H_R.

- **Bayes-L1 (flag-blind):** the standard posterior predictive P(x_{t+1} | x_{0..t}), treating every transition as evidence.
- **Bayes-L2 (do):** the same, except that transitions *into* intervened positions (x_{t−1} → x_t for t ∈ I) contribute a likelihood factor of 1 under both hypotheses. Transitions *out of* intervened positions remain evidence.
  - Likelihood under H_P(a, b): (1/p) · ∏_{t ≥ 1, t ∉ I} 𝟙[x_t = a·x_{t−1} + b].
  - Likelihood under H_R: ∏_{t ∉ I} (1/p).
  - Predictive for x_{t+1} when t+1 ∉ I: the posterior-weighted mixture of point masses (programs) and the uniform distribution (noise).
- **Bayes-L3 (counterfactual):** given a complete factual sequence (no interventions), a query position q, and a counterfactual value v:
  - Compute the posterior over hypotheses from the **entire** factual sequence.
  - Under each consistent H_P(a, b): y_k = f^k(v), where f(x) = a·x + b.
  - Under H_R: y_k = x_{q+k}, the factual value, because the exogenous noise is held fixed.
  - The answer is the posterior-weighted mixture.
- **Bayes-L2-for-CF (control reference):** the same query answered with fresh noise. Under H_R, y_k is uniform. Under H_P, use the posterior from the prefix x_{0..q} only.

### 3c. Required tests (`pytest`)

1. On clean sequences, Bayes-L1 matches the posterior predictive implied by the vendored code's ground-truth functions, if any exist; otherwise test against an independent brute-force implementation.
2. **Regression test for the earlier confound:** for 1,000 program-sequence pairs, substitute the token at position p ∈ {3, 4} with a different value Y and evaluate Bayes-L1. Every target must get 1/17 ± 1e−6, with P(program) ≈ 0. Evaluate Bayes-L2 on the same input with f_p = 1: probability ≈ 1 on a·Y + b for identified programs.
3. Under H_R, Bayes-L3 puts its mass on the factual token once P(H_R | sequence) ≈ 1.
4. For program episodes with q ∈ {0, 1}, Bayes-L3 is sharp while Bayes-L2-for-CF is diffuse.
5. The v-independence test from 3a.
6. All oracle outputs are normalized distributions; loss masks align with I.

Stop and report the test results plus a few printed example episodes with their oracle distributions.

---

## 4. Layer 2 experiments

### Training conditions (3 seeds each)

| Model | Training data | Flags |
|---|---|---|
| **M_obs** | 100% observational (I = ∅) | always 0 |
| **M_flag** | 50% observational, 50% with exactly one intervention at a position in S_train | true flags |
| **M_hidden** | same mixture as M_flag | always 0 (interventions present but unmarked) |

- S_train = {2, 4, 6, 8, 10, 12}; S_test = {3, 5, 7, 9, 11, 13}.
- The hypothesis split (π) applies within both observational and interventional episodes.
- Optimizer and schedule should match `auxiliary-causal` defaults (expected: AdamW, lr 3e-4, weight decay 0.01, batch 64, 150k steps). Verify and report. Evaluate every 10k steps.
- Smoke config: d_model 64, 2 layers, 3k steps, CPU-feasible.

### Test conditions

Use 5,000 fresh episodes per condition, half program and half noise.

| Test | Description | Purpose |
|---|---|---|
| **T0** | observational episodes | sanity: all models track Bayes-L1 |
| **T1** | one intervention at a position in S_train, flag on | seen interventions |
| **T2** | one intervention at a position in S_test, flag on | **untried interventions** (Pearl) |
| **T3** | two interventions, both in S_train, flags on, never seen jointly | **composite interventions** (Pearl) |
| **T4** | two interventions, one in S_train and one in S_test | composite and untried together |
| **T5** | T1 episodes with flags forced to 0 | does M_flag use the flag? It should revert to Bayes-L1 |

For M_obs, run T1–T4 with flags forced to 0, since `flag_emb` row 1 is untrained. M_obs should then track Bayes-L1, which is the Causal Hierarchy Theorem's prediction.

### Metrics per (model, test, position)

- KL(Bayes-L2 ‖ model) and KL(Bayes-L1 ‖ model), in bits, at all supervised positions after the first intervention.
- Entropy MAE against Bayes-L2.
- On program episodes: the probability mass on the do-continuation f^k(v) at positions v+1, v+2, ….
- A discrimination ratio: KL to Bayes-L1 divided by KL to Bayes-L2.

---

## 5. Layer 3 experiments

### Format

```
[x_0 … x_15] [CF] [Q_q] [v] [y_1] [y_2] [y_3]
```

- `CF` is a separator token. `Q_0 … Q_15` are 16 query-position tokens. v ∈ Z_p with v ≠ x_q.
- The y_k are the counterfactual values of x_{q+k}, with q ≤ 12 so that three exist.
- The vocabulary grows by 17 tokens, and the context length is 22. All 22 positions get supervision wherever a target exists: factual next-token loss on positions 0–15, plus loss on y₁–y₃. No loss on predicting `CF`, `Q_q`, or v.
- **Labels are hard targets from the true SCM:** for program episodes, use the true (a, b); for noise episodes, use the factual tokens. The cross-entropy minimizer is then Bayes-L3.

### Training conditions (3 seeds each)

| Model | CF labels |
|---|---|
| **M_cf** | true counterfactuals (noise episodes: copy the factual tokens) |
| **M_cf_fresh** (control) | noise episodes: freshly resampled uniform tokens; program episodes unchanged |

- Q_train = {0, 2, 4, 6, 8, 10, 12}; Q_test = {1, 3, 5, 7, 9, 11}.
- **Held-out programs:** with a fixed seed, exclude 20% of (a, b) pairs from training program episodes. Evaluate these only on episodes where the factual sequence uniquely identifies (a, b), so that the prior cannot matter.

### Test conditions

| Test | Description |
|---|---|
| **C1** | q ∈ Q_train, programs seen in training |
| **C2** | q ∈ Q_test (untried query positions) |
| **C3** | held-out (a, b), q ∈ Q_train |
| **C4** | held-out (a, b), q ∈ Q_test |

### Key discriminants

- **D1 (noise episodes):** mass on the factual copy x_{q+k} versus uniform. Bayes-L3 says copy; the L2-style answer says uniform. M_cf should copy and M_cf_fresh should not.
- **D2 (program episodes, q ∈ {0, 1}):** Bayes-L3 is sharp because it abducts (a, b) from facts *after* q, while Bayes-L2-for-CF is diffuse. Report KL to each.
- **C3/C4** distinguish abduction of (a, b) as an algorithm from memorization of specific programs.

### Stretch goal, only if time permits

Add composite counterfactuals with two edits, where training uses only single edits.

---

## 6. Analysis, figures, report — CHECKPOINT

### Figures (matplotlib, following Tufte)

- No chart junk: no heavy gridlines, no boxes, no legends where direct labels work. Use range frames and muted reference lines.
- **Small multiples:** one panel per test condition, x = position, y = KL in bits (log scale if the range spans orders of magnitude). Show each model as a directly labeled line, with Bayes-L1 and Bayes-L2 or L3 as thin gray references.
- One summary dot plot: discrimination ratio per (model, test), with seed CIs as thin whiskers.
- Save as both PNG and PDF.

### `REPORT.md`

- Restate each pre-registered prediction with its outcome: pass or fail against the numeric criteria, with CIs.
- State plainly which hypothesis each result supports:
  - **"Interface + data suffice":** M_flag passes T2–T4, and M_cf passes C2–C4 and D1–D2.
  - **"Pearl's 1¾":** passes T1 but fails T2/T3.
  - **"Architecture-limited":** fails even T1 or C1 despite the correct interface.
- Report the training curves for T2/T3 and C3/C4. A late jump would indicate grokking-like dynamics, which bears on whether failures reflect undertraining.
- List the remaining confounds and what would rule each out.
- Do not soften negative results.

Stop and show me the report, figures, and the metrics table before any further runs.

---

## Working rules

- Run the smoke suite end to end before any full run.
- Use fixed seeds everywhere; generate evaluation sets once, save them to disk, and reuse them across models.
- If any oracle test fails, or any generator assumption differs from this spec, stop and report. Do not paper over it.
- Keep commits small, with descriptive messages. Never push to the two source repositories.
