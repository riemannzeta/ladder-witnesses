# Pre-registration: ladder-witnesses

Registered 2026-10-03, **before any GPU training run**. The criteria come from the approved plan (`PLAN.md`), which was itself fixed after an independent numerical review of the design. The git commit containing this file is the registration.

**Runs before registration, disclosed.**
- Smoke runs of every experiment (≤ 600 steps, 2-layer d=64 models), to test the pipeline.
- One E1 feasibility run on an Apple M1: 5,000+ steps, d=128 / 4 layers, batch 128, so about 1/8 of the planned training. It checked that the task is learnable at all. By step 5,000 the model had GC 0.94, split slope 0.94 (R² 0.95), and matched-L1 parity on the adjusted estimand (L2-ADJ KL 0.029 vs FUNC-ADJ KL 0.031 bits).
- Nothing below was changed after seeing that run.

**One later addition, from the E5 implementation review.** The E5 criteria use a total-variation gap closed instead of KL. E5's oracles are near-point-masses, and KL between them saturates, so a KL ratio reads ~0.4 even for an untrained model.

## Hypotheses

| | Statement | Prediction pattern |
|---|---|---|
| **H_info** | Given the information the Causal Hierarchy Theorem says is necessary, a small transformer computes the higher-rung answer, including on structures and query combinations it never trained on. | E1–E5 pass; E2/E2c held-out and combined queries pass |
| **H_1¾** | (Pearl's "level one and three-fourths") The model learns the rung-2 answers it was trained on but cannot extend them to untried interventions, unseen graphs, or combinations. | in-distribution passes; held-out / combined fail (GC ≤ 0.5) while the matched L1 control passes |
| **H_arch** | The architecture cannot compute rung-2/3 answers even when given the information. | fails in-distribution despite the bits |

E1, E3 and E4 are *demonstrations*: every hypothesis predicts they pass, because the witness pair supplies the bit and a fixed computation follows. A failure there is a bug or undertraining, not evidence. **E2 and E2c are the tests that can come out against H_info.**

Nothing here is read as a claim about all transformers: a failure licenses only "this model at this scale and training".

## Frozen design

- **Training targets:** PFN-style. Each episode's target is the TRUE higher-rung quantity, so the loss minimizer is the Bayes posterior under the training prior.
- **Convergence:**
  - Warmup, then constant LR, then linear cooldown.
  - Cooldown starts when the headline eval metric improves < 1% (relative) over the last 25% of steps, after `min_steps`, or else at the cap of 4× `base_steps`.
  - Runs that hit the cap are labeled **unconverged** and are never counted as negative results.
- **Seeds:**
  - E1, E3, E4: 5 per condition.
  - E2: 3 per fold.
  - E2c, E5: 3 per condition.
  - Capacity sweeps: 2 per point.
- **Eval sets:** generated once from seed 1,000,003, disjoint from training draws, and saved to disk.
- **Metrics:** means over seeds with 95% bootstrap CIs (over episodes and seeds).
- **E1:** θ ~ Dirichlet(0.3·1₈), N = 128, 6 complete orderings; d=128, 4 layers, 4 heads.
- **E2:** 4 nodes, in-degree ≤ 2 (25 topologies, 443 labeled DAGs), BDeu with ESS 2.
  - E2-0 = CPT-token mode, exact targets.
  - E2-full = 128 samples.
  - d=256, 6 layers, 8 heads.
  - Capacity sweep:
    - E2-0: d ∈ {128, 256, 384} × layers ∈ {4, 6, 8} on folds 0–1.
    - E2-full: the diagonal (128,4), (256,6), (384,8) on folds 0–1.
  - Folds (`configs/e2/folds.json`, seed 20261003, stratified by edge count): every held-out topology that has a Markov-equivalent partner keeps at least one partner in training.

| Fold | Held-out topologies (internal representative edges) |
|---|---|
| 0 | T0: (none); T4: 0→2, 1→3; T13: 0→1, 1→2, 2→3; T19: 0→1, 1→2, 0→3, 2→3; T20: 0→1, 1→2, 1→3, 2→3 |
| 1 | T1: 0→3; T9: 0→2, 1→2, 2→3; T11: 0→1, 0→2, 1→3; T14: 0→2, 1→2, 0→3, 1→3; T15: 0→2, 1→2, 0→3, 2→3 |
| 2 | T5: 0→2, 2→3; T8: 0→2, 1→3, 2→3; T12: 0→1, 1→2, 1→3; T21: 0→1, 0→2, 1→2, 2→3; T24: 0→1, 0→2, 1→2, 1→3, 2→3 |
| 3 | T3: 0→2, 0→3; T6: 0→2, 0→3, 1→3; T10: 0→1, 0→2, 0→3; T16: 0→1, 0→2, 0→3, 1→3; T23: 0→1, 0→2, 1→2, 0→3, 2→3 |
| 4 | T2: 0→3, 1→3; T7: 0→2, 0→3, 2→3; T17: 0→1, 0→2, 1→3, 2→3; T18: 0→1, 1→2, 0→3, 1→3; T22: 0→1, 0→2, 1→2, 0→3, 1→3 |

- **E2 composite eval queries:** both elements must change the answer by > 0.05, AND conditioning must differ from intervening by > 0.05. Composite queries never appear in training: training queries have one effective element plus a NULL or *inert* second slot.
- **E2c:** all 25 topologies. Context = 64 observational + 64 do(A) + 64 do(B) samples, regime-tagged. Queries: untried do(A,B), untried do(C), tried do(A)/do(B). d=256, 6 layers.
- **E3:** w ~ N(0,1), σ = 0.5, training ρ ~ U([−1,−0.3] ∪ [0.3,1]), up to 32 pairs. d=128, 4 layers.
- **E4:** θ ~ Dirichlet(0.5·1₈), N ≤ 256, 64-bin posterior head. d=128, 4 layers.
- **E5:** Misra's task (p=17, T=16, π=0.5, a ∈ 0..16). d=192, 6 layers, 6 heads (his `RecurrenceTransformer` size).
  - E5a: one intervention at a uniformly random position in 50% of episodes.
  - E5b: ε ~ Bern(0.2); 20% of programs held out (seed 12345).

## Pass criteria and my point predictions

**GC (gap closed)** = 1 − E[KL(oracle ‖ model)] / E[KL(oracle ‖ baseline)]. The baseline is the best answer *without* the higher-rung bit. Averages use positions with k ≥ 8 samples (E1, E2-full) or k ≥ 24 (E2c).

### E1 — Simpson pair over all six orderings (demonstration)
- **Pass:**
  - split slope in [0.9, 1.1] with R² ≥ 0.9 (episodes with oracle spread ≥ 0.02);
  - GC_L2 ≥ 0.85 against the ordering-mixture (blind) oracle;
  - the shuffled-ordering control has |split slope| ≤ 0.05;
  - reversal sign agreement ≥ 95% under the ADJ ordering and under the CRUDE ordering;
  - per-estimand L2 KL within the bootstrap CI of the matching FUNC (named-functional) query.
- **Prediction:** pass on all. The adjusted estimand converges last, and L2-ADJ ≈ FUNC-ADJ throughout.

### E2-0 and E2-full — Pearl's "1¾" with the graph supplied (the test)
- **In-distribution pass:** GC ≥ 0.85.
  - E2-0's baseline is the other rung's answer (conditioning for do-queries, intervening for conditioning queries).
  - E2-full's baseline is the graph-blind model average over all 443 DAGs.
- **Held-out topologies and combined interventions pass:** GC ≥ 0.8, and slope of (model − L1) on (L2 − L1) ≥ 0.8 over discriminative episodes.
- **Fail:** GC ≤ 0.5.
- **Same-data graph swap** between Markov-equivalent graphs: split slope reported. Prediction ≥ 0.8.

| Outcome | Held-out / combined L2 | Matched L1-with-graph (COND) |
|---|---|---|
| H_info | GC ≥ 0.8 | GC ≥ 0.8 |
| H_1¾ | GC ≤ 0.5 with in-distribution passing | GC ≥ 0.8 |
| H_arch | fails, or in-distribution also fails | fails |
| Inconclusive | 0.5 < GC < 0.8 → convergence + capacity sweep, then report final status | — |

Verdicts are reported per fold and pooled.

**Predictions:**
- E2-0: in-distribution passes. Held-out topologies pass in ≥ 4 of 5 folds. Combined interventions: **genuinely uncertain**; my central guess is GC ≈ 0.7–0.9.
- E2-full: in-distribution passes. Held-out passes but is weaker than E2-0. Combined: uncertain.

### E2c — no graph; tried do(A), do(B); untried do(A,B) and do(C) (the test)
- **Pass (H_info):** GC ≥ 0.8 on the combined and untried queries.
  - The baseline is the model-free "1¾" learner: tried regimes from their frequencies; the untried combination from the two nearest tried regimes conditioned on the other variable, averaged; untried single from the observational conditional.
  - The reference is ideal Bayesian model averaging over the 443 DAGs with Cooper–Yoo interventional likelihoods.
- **H_1¾:** GC ≤ 0.5 on combined/untried while tried queries "track", i.e. KL(Bayes ‖ model) ≤ KL(Bayes ‖ baseline).
- Reported split by graph-posterior sharpness (below/above the median of the answer's posterior sd across graphs).
- **Prediction:** pass overall. GC is lower on the diffuse half than on the sharp half.

### E3 — residual twin with continuous ρ (demonstration)
- **Pass:**
  - on crossing queries, the residual-transfer slope tracks ρ: mean |slope − ρ̄| ≤ 0.05 over training-band ρ bins and ≤ 0.1 over held-out bins (|ρ| < 0.3);
  - same-side slope 1 ± 0.05;
  - shuffled-ρ and L2-trained controls have |slope| ≤ 0.05 in every bin;
  - normalized excess MSE (excess over the Bayes error, divided by the Bayes error) ≤ 0.1.
- **Prediction:** pass, including the held-out ρ band.

### E4 — identifiability tracking, 2×2 (demonstration)
- **Pass:**
  - model/oracle posterior-sd ratio in [0.85, 1.15] at every k ∈ {16 … 256};
  - log-log slope of median sd vs k over k ≥ 32 in [−0.6, −0.35] for OBS-CONF, OBS-MED and HID-MED, and in [−0.25, 0] for HID-CONF;
  - 80%-interval coverage of the true p₁ in [0.77, 0.83].
- **Prediction:** pass. The plateau appears only in HID-CONF, even though HID-MED uses identical tokens.

### E5 — the same witness pairs in Misra's task (demonstration of transfer)
- **E5a (restart vs clock) pass:**
  - for each model, `gc_tv_prog` ≥ 0.85, where the baseline is the other semantics' oracle (program episodes, a ≠ 0, discriminative positions);
  - with flags zeroed, `abl_kl_l1` ≤ 0.05 bits (both collapse to flag-blind Bayes-L1).
- **E5b (abducted vs fresh noise) pass:**
  - `gc_tv_noise` ≥ 0.85 and `gc_tv_prog` ≥ 0.85 for both conditions.
  - Held-out programs: if factual next-token KL generalizes (`kl_fact_heldout` ≤ 2× seen) while `gc_tv_heldout` < 0.5, that is an L3-specific generalization deficit; otherwise not.
- **Prediction:**
  - E5a and E5b pass on seen programs.
  - Held-out programs: factual prediction generalizes, as in your earlier work; the counterfactual on held-out programs is uncertain, GC_TV ≈ 0.6–0.9.

### E6 — Claude verbal probe (descriptive; predictions only)
Pilot of 20 contexts per condition first, then sized by power analysis at CP1.

| Model | V1 stated: split slope | V1 blind: says "not determinable" | V2 stated: pooled ρ-tracking slope | V2 unstated: crossing answer ≈ w·x′ | V3 unmeasured: says "not identifiable" / bounds correct |
|---|---|---|---|---|---|
| claude-opus-5-5 | ≥ 0.9 | ≥ 50% | 0.9–1.1 | ≥ 60% | ≥ 80% / ≥ 50% |
| claude-sonnet-5-5 | ≥ 0.85 | ≥ 40% | 0.85–1.1 | ≥ 50% | ≥ 70% / ≥ 40% |
| claude-haiku-4-5 | < 0.8 | < 40% | weaker | < 50% | < 60% / < 30% |

## Claims discipline (binding on the write-up)
Never say any of the following:
- that the results "refute Pearl" or "refute the CHT" (the design relies on the CHT, and giving the graph is Pearl's own recipe);
- that the model "learned causal effects from observational data" (its training targets were the true L2/L3 quantities);
- that it "resolves Simpson's paradox" (it is told the ordering);
- that it "performs abduction internally" or "knows it can't know" (only input-output behavior is measured, and the height of the E4 plateau is set by the prior);
- that the results "confirm Mossé–Ibeling–Icard" (theirs is a satisfiability-complexity result).

Also nothing about do-calculus beyond the truncated factorization, about N beyond the training range, or about LLMs from the small-model results.

---

## Addendum 1 (2026-10-06): changes after wave 1, written before any wave-2 run

Everything above this line was committed before any training (commit `3f434c8`). This addendum records every change made after seeing the wave-1 results (E1, E3, E4, E2-0). Every pre-registered metric is still reported exactly as defined above; nothing here replaces one.

### A. Evaluation changes made after seeing wave-1 results (post hoc)
1. **E2 combined-query sets held too few do() items.** The frozen combined split draws do() and conditioning queries 50/50, then applies the pre-registered acceptance rules (both elements move the answer by > 0.05, and conditioning differs from intervening by > 0.05). In 4-node graphs with in-degree ≤ 2, only 2 of the 25 topologies (T20, T24) admit any do() query that passes, so the frozen split holds 0–53 do() items per fold and none at all in held-out folds 1, 3 and 4.
   - `scripts/reeval_e2.py` re-scores finished checkpoints on **balanced** sets that draw each query type to its own quota (up to 1,000) under the same rules (*strict*).
   - It also scores an exploratory *relaxed* set that drops the do-vs-conditioning rule.
   - The frozen-split numbers remain the pre-registered result.
2. **Structural breakdown (exploratory).** Combined-set KL is split by whether both pinned variables are parents of the target.
3. **E4 coverage, interpolated.** The pre-registered coverage reads interval ends off bin edges, which widens every interval by up to a bin on each side and over-covers even for the exact oracle.
   - The as-implemented metric remains the pre-registered result and is reported as FAIL (0.82–0.87).
   - The interpolated version (model 0.79–0.82; oracle 0.78–0.81) is reported as a labeled diagnostic.

### B. Wave-2 decisions
1. **The diagonal capacity sweep runs 2 seeds.** The rule above says 2 seeds "if any fold comes out inconclusive or negative", but that can't be checked before the sweep starts, because the sweep shares the queue with the E2-full folds. We take the larger option unconditionally, because in E2-0 the matched two-condition control failed and the seed-to-seed spread exceeded any capacity effect.
2. Everything else in wave 2 runs as pre-registered: E2-full 5 folds × 3 seeds, E2c 3 seeds, E5 4 conditions × 3 seeds.

### C. Post-hoc arm 1: E2-0 trained with two-effective-element queries (exploratory)
- **Why.** In E2-0 every pre-registered interventional criterion passed (GC 1.000). The matched control with two conditioning variables failed: GC 0.37 [0.30, 0.44] on training topologies and 0.40 [0.34, 0.45] on held-out ones. On the relaxed do() sets, KL was 10–18× higher (0.07 vs. 0.004–0.007 bits) when both pinned variables are parents of the target.
  - Training never contained a query in which both slots matter: the second slot was always empty or inert.
  - The capacity sweep (0.8M–14M parameters) showed no trend.
- **Question.** Is the weakness a gap in the training distribution, or something that more examples of the query type don't fix?
- **Design.** Identical to `cpt_f<k>`, except that 25% of training queries are replaced by queries whose two elements are both structurally effective:
  - **do-pairs:** each element has a directed path to the target that avoids the other;
  - **conditioning pairs:** neither element is d-separated from the target given the other.
  - Folds 0–4, seed 0 (5 runs), configs `configs/e2/cpt_comp_f<k>.yaml`, scored on the same frozen and balanced sets as the pre-registered runs. The eval splits are provably unchanged (`tests/test_e2.py::test_composite_training_mix_is_training_only`).
- **What changes in meaning.** For this arm, a combined query is no longer a query type never seen in training. The held-out topologies are still never seen. The arm asks whether two-element computations transfer to unseen graphs once the query type itself has been trained.
- **Predictions (written before the runs):**
  - strict two-variable-conditioning GC ≥ 0.9 on training topologies **and** ≥ 0.9 on held-out topologies;
  - relaxed do() KL with both pinned variables parents of the target ≤ 0.01 bits on held-out topologies;
  - no regression on the pre-registered single-query criteria (GC ≥ 0.99).
- **Reading:**

  | Outcome | Reading |
  |---|---|
  | Training and held-out both ≥ 0.9 | The E2-0 weakness was a gap in the training distribution |
  | Training ≥ 0.9, held-out ≤ 0.5 | Two-element computations were learned but do not transfer to unseen graphs |
  | Both ≤ 0.5 | Training examples of the query type do not fix it at this size and budget |

### D. Post-hoc arm 2: E4 with a 256-bin posterior head (diagnostic)
- **Why.** E4's sd-ratio criterion failed only for OBS-CONF at k = 256: ratio 1.15–1.22 across seeds, against ≤ 1.07 at every k ≤ 128.
  - Measured before this run: binning into 64 bins widens the *oracle's* own posterior sd by only 0.8% at k = 256 (Sheppard's correction).
  - So discretization arithmetic can't account for a 15–22% excess. The open question is whether the head's resolution matters in some other way, for example through the difficulty of fitting a sharp peak with 64 logits.
- **Design.** `configs/e4/bins256.yaml`: `main.yaml` with `n_bins: 256`, 5 seeds, everything else unchanged.
- **Predictions:**
  - **The OBS-CONF excess persists:** the mean sd ratio at k = 256 stays > 1.10. If it falls into [0.85, 1.15], the head's resolution was responsible after all.
  - As-implemented 80% coverage over k ≥ 16 falls within [0.77, 0.83] in all four cells, because bin-edge rounding shrinks fourfold.

---

## Addendum 2 (2026-10-08): oracle bug fix and diagnostics, written mid-wave-2 (21/48 runs done)

1. **Bug fix in the E2-full oracles: no change to any definition.** The Monte Carlo L1 oracle and the conditional branch of the graph-blind baseline divide by the probability of the conditioning event, computed for each posterior draw.
   - Beta(¼, ¼) rows for unseen parent configurations occasionally underflow to exactly 0 or 1 in float64, which makes that probability exactly 0 and the draw 0/0.
   - In the frozen sets this produced NaN for 1 of 1,000 combined-conditioning items in folds 2 and 3 (and a NaN baseline in fold 2), so those means were NaN.
   - Fix: drop such draws, a probability-zero event affecting at most a few of the 2,048 draws (`tests/test_e2.py::test_conditional_oracle_survives_beta_underflow`).
   - After wave 2, every E2-full run is re-scored on the same frozen items, with only the NaN entries recomputed. The eval-set files are not touched while the queue is running.
2. **E4 256-bin diagnostic (Addendum 1 D), recorded as measured.**
   - **The excess got larger, not smaller:** OBS-CONF sd ratio at k = 256 was 1.45–1.76 against 1.15–1.22 with 64 bins. In both arms it shrinks monotonically with training steps.
   - **Post-hoc analysis:**
     - the model's posterior mean is as accurate as the oracle's (RMSE ratio 1.00–1.02);
     - its central 80% width and IQR match the oracle's (ratios 0.99–1.06);
     - the sd excess comes from a diffuse far tail: 0.1–0.2% of mass beyond 5 oracle sd with 64 bins, and 0.5–0.7% with 256 bins.
   - **The prediction "excess persists (> 1.10)" held, but the head's resolution is involved in the opposite direction from the one I anticipated:** more bins means more tail floor.
   - **The coverage prediction failed in 2 of 4 cells** (as implemented, OBS-CONF reached 0.840 and OBS-MED 0.838).
   - The pre-registered E4 sd-ratio result stays FAIL.

---

## Addendum 3 (2026-10-08): post-hoc arm 3, E2-full with two-effective-element training (exploratory)

- **Why.** E2-full (11/15 runs done) passes every single-query criterion, but fails combined queries in the same way E2-0 did.
  - Combined-conditioning KL is 0.22–0.33 bits, against 0.0003–0.006 for single queries.
  - The slope of (model − L2) on (L1 − L2) is 0.30–0.45.
  - Arm 1 (Addendum 1 C) showed that in E2-0 this was a gap in the training distribution.
- **Design.** `samples_f<k>` with `p_composite_train: 0.25` (same PAIRS rule as arm 1). Folds 0–4, seed 0 (5 runs), configs `configs/e2/samples_comp_f<k>.yaml`, queue `wave2b`, run after wave 2.
  - Scored on the same frozen sets as the pre-registered E2-full runs, after the Addendum 2 repair.
  - Scored also on balanced sets built with the fixed oracles.
- **Predictions (written before the runs), on the frozen combined sets:**
  - combined conditioning: slope ≥ 0.9 and KL ≤ 0.02 bits, on training and on held-out topologies;
  - combined do(): GC ≥ 0.8 wherever the split has do() items;
  - no regression on the single-query criteria (in-distribution do() GC ≥ 0.85, held-out ≥ 0.8).
- **Reading:** as in the table under Addendum 1 C. Additionally, if E2-0 recovers but E2-full does not, the obstacle is the combination of estimating from data and composing two elements, not composition alone.
