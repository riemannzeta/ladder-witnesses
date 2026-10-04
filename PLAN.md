# Plan: `ladder-witnesses`, witness-pair experiments on Pearl's ladder → public repo + symmetrybroken.com post

## Context

The `~/causal-wind-tunnel` prompt tested L2/L3 inside Misra's mod-17 recurrence. Its tests can't separate the hypotheses:
- With randomized values, an intervention looks exactly like a flagged random restart.
- T2/T3 measure generalization to new positions.
- The L3 tokens Q₁, Q₃, … are never trained.
- L3 is trivial in the deterministic program branch.

The replacement uses **witness pairs**, the standard device behind the Causal Hierarchy Theorem (CHT):
- Two causal models agree on rung k and differ on rung k+1.
- Anything using only rung-k information must answer both the same way.
- We give the model the distinguishing information through its input and test whether its answers diverge exactly as an exact Bayes oracle says.
- A format-matched control gets the same input with that information scrambled. Any difference between the two is attributable to the information, not to the architecture.

**Your decisions:**
- Repo name: `ladder-witnesses`.
- Scope: core E1–E4, plus E5 (Misra-task port) and E6 (Claude probe).
- I pause for your review before any GPU spend.

**Deliverables**
1. A public repo `github.com/riemannzeta/ladder-witnesses`: code, tests, configs, `PREDICTIONS.md`, `REPORT.md`, results, figures.
2. A Ghost-ready Markdown post whose math is verified to render on symmetrybroken.com.

**Your adjustment: there is no GPU-time limit; correctness comes first.** Concretely:
- Every run trains **to convergence**, not to a fixed budget.
- More seeds.
- Before any negative result is reported, E2 goes through capacity and training-length sweeps and cross-validation over held-out topology folds.
- E2c, the closest test of Pearl's actual claim, moves from stretch goal to full experiment.

**Design review folded in.** A separate review checked every oracle numerically. All are correct except E2's L1 reference, which now uses Monte Carlo. Adopted fixes:
- Gap-closed and slope criteria instead of absolute KL. The blind oracle is only about 0.017 bits from the correct one, so a 0.05-bit threshold would pass a model that ignores the graph.
- A score-equivalent BDeu prior in E2. The earlier Beta(.5,.5) prior leaked edge orientation through the data.
- A format-matched test for combined interventions.
- A continuous cross-world correlation ρ in E3.
- A 2×2 control in E4.
- Role randomization and a matched L1 control in E1.

**Framing that follows from the review:**
- E1, E3 and E4 are *demonstrations*. Every hypothesis predicts they pass, so a failure there means a bug.
- **E2 and E2c are the real tests** of Pearl's "1¾" remark. E2 asks whether the model can *use* a supplied graph. E2c asks whether it can combine interventions *without* one.

## Compute: keep the existing A10G (g5.xlarge); a bigger GPU isn't needed
- **The models are tiny** (≤ ~5M parameters, ≤ ~260 tokens), so a faster GPU would mostly shorten the calendar, not change any result.
- **Your account can't scale out without a quota increase anyway:** the on-demand G/VT quota is **4 vCPUs** (exactly one g5.xlarge) and the spot quota is 0.
  - If you later want results sooner, a quota request lets several seeds run in parallel. I won't file it without your OK.
- **Throughput work:**
  - on-GPU sampling and exact vectorized oracles;
  - bf16 with SDPA attention;
  - soft targets;
  - 2–3 small runs sharing the GPU.
  - For comparison, your old runs managed ~43 steps/s at 1k tokens/step because per-sequence Python oracles were the bottleneck.
- **Estimate:** roughly **80–110 GPU-hours, i.e. 4–5 days**, ≈ $85–120 at ~$1/hr. The first benchmark refines this. E2-full is about half of it.
- **Runs are unattended and resumable:**
  - the queue skips finished runs;
  - checkpoints include optimizer state;
  - `run_suite.sh` stops the instance when the queue drains (`shutdown -h`, which stops rather than terminates), so billing ends even if I'm not watching.

## Experiments
**Shared setup**
- Training is PFN-style: the targets are the episode's *true* higher-rung quantity, so the loss minimizer is the Bayes posterior predictive under the training prior.
- Every context position has a readout.
- Causal attention.
- Train and eval seeds are disjoint, and eval sets (10k episodes each) are frozen to disk.

**Convergence policy (applies to every run):**
- Evaluate every 2k steps.
- A run counts as converged when its headline metric (GC, KL or MSE) improves by less than 1% relative over the last 25% of steps. It stops then, or at the step cap (4× the base budget).
- Learning curves are saved and published.
- Any run that hits the cap before converging is labeled "unconverged" in `REPORT.md` and never counted as a negative result.

**Seeds:** 5 per condition for E1, E3 and E4. 3 per fold for E2, and 3 per condition for E2c and E5. 2 per point of the capacity sweeps.

**Leak probes** (tests): a classifier given only the data must predict the hidden bit (graph, ordering, ρ, orientation within a Markov class) no better than chance.

**E1: L2 demonstration ("one bit buys the rung"), Pearl's Simpson pair, generalized**
- **Episode setup:**
  - θ ~ Dirichlet(0.3·1₈) over binary (A,B,C), with α fixed.
  - N=128 samples.
  - Roles: the treatment and outcome are drawn from the three labels.
  - Graph = one of the **6 complete orderings**, drawn independently of the data. Under the Dirichlet joint the data carry zero information about it (Geiger–Heckerman).
  - Each (ordering, query) pair needs one of three estimands: adjusted Σ_c P(y|x,c)P(c), crude P(y|x), or marginal P(y).
- **Matched L1 control:** half the episodes instead ask for the same functional *by name* (FUNC_ADJ / FUNC_CRUDE / FUNC_MARG). This measures whether the causal step adds any difficulty over the identical arithmetic.
- **Oracles:** closed form, checked against Monte Carlo to 1e-3.
- **Controls:**
  - a shuffled-ordering model (format-matched blind);
  - the blind Bayes answer, i.e. the ordering-mixture (1/6)·adj + (2/6)·crude + (3/6)·marg.
- **Readouts:**
  - the split of the model's answers across orderings on identical data, against the oracle's split;
  - gap closed, GC = 1 − KL(p*‖model)/KL(p*‖p_blind);
  - Simpson-reversal sign agreement;
  - L2 accuracy vs. the accuracy on the matching FUNC query.
- **Model:** d=128, 4 layers.

**E2: Pearl's "1¾" (untried and combined interventions on unseen graphs) — the actual test**
- **Graphs and prior:**
  - 4 binary nodes; DAGs with in-degree ≤ 2: 25 unlabeled topologies, 443 labeled DAGs, at most 5 edges.
  - Labels are permuted per episode.
  - **BDeu prior with ESS 2:** Beta(1,1) for roots, (.5,.5) rows for one parent, (.25,.25) rows for two parents. It is score-equivalent, so the data carry no orientation information within a Markov class.
- **Prefix:**
  - 5 edge slots, shuffled, with padding.
  - **2 do-slots, shuffled.** In training, one do() is effective and the other is NULL or *inert* (on a non-ancestor of the target).
  - A query type: DO, or COND (the matched L1-with-graph control).
  - A target.
- **Held-out topologies, cross-validated:**
  - The 25 topologies are split into **5 folds**, stratified by edge count and frozen by script in `PREDICTIONS.md`.
  - Where a held-out topology has Markov-equivalent partners, at least one partner stays in training, which enables same-data graph swaps on unseen graphs.
  - Each fold trains its own model, so no conclusion depends on one lucky or unlucky choice of held-out graphs.
- **E2-0, run first:** instead of samples, the input has up to 11 CPT-row tokens (node, parent configuration, value θ). The targets are exact. This isolates graph → truncated factorization from statistical estimation.
  - Runs: 5 folds × 3 seeds, plus a full capacity sweep: d ∈ {128, 256, 384} × layers ∈ {4, 6, 8}, on 2 folds × 2 seeds. These runs are cheap.
- **E2-full:** the same prefix plus 128 samples (16-symbol tokens), with an auxiliary next-sample head.
  - Runs: 5 folds × 3 seeds at the base size.
  - Plus a diagonal capacity sweep, (128,4) / (256,6) / (384,8) on 2 folds. It uses 2 seeds if any fold comes out inconclusive or negative, and 1 seed otherwise (for the scaling figure).
- **No negative conclusion without the checks:** a failure is reported only if it holds at the largest converged setting **and** the matched COND (L1-with-graph) head passes on the same graphs.
- **Oracles:**
  - Bayes-L2 is the truncated factorization at the posterior-mean CPTs (exact).
  - Bayes-L1 (COND) is computed by Monte Carlo with 2–4k draws.
  - The **blind baseline** is Bayesian model averaging over all 443 DAGs, using closed-form BDeu marginal likelihoods. It is the denominator for GC.
- **Tests:**
  - in-distribution;
  - held-out topologies;
  - **combined interventions**, where both do()'s change the answer — never seen in training;
  - a same-data graph swap on held-out graphs.
- **Model:** d=256, 6 layers.

**E3: L3 demonstration, the residual twin with continuous cross-world correlation ρ**
- **Mechanism:** w~N(0,1), x~N(0,1), and Y = wx + σ[1{x>0}U₁ + 1{x≤0}(ρU₁ + √(1−ρ²)U₂)], with σ=0.5.
  - L1 and L2 are identical for every ρ; ρ=+1 is additive noise and ρ=−1 is the flip.
  - Training uses ρ~U(−1,1) with **|ρ|<0.3 held out**.
- **Sequence:** `[BOS][ρ][unit (x*,y*,x′)]` followed by up to 32 (x,y) tokens.
- **Heads:**
  - a counterfactual head;
  - an L2 head, m·x′, which separates errors in estimating w from errors in the counterfactual step.
- **Oracle:** for crossing queries, m·x′ + ρ(y*−m·x*); for same-side queries, slope 1.
- **Controls:**
  - a shuffled-ρ model, whose Bayes answer is slope 0, i.e. **exactly the L2 answer**;
  - an L2-trained model (fresh-noise targets).
- **Readouts:**
  - the fitted crossing slope against ρ, which should lie on the identity line, including held-out ρ;
  - normalized excess MSE against N.
  - Queries with |x*| or |x′| < 0.1 are excluded.
- **Model:** d=128, 4 layers.

**E4: the CHT as something measurable (identifiability tracking), 2×2**
- **Episode:** θ~Dirichlet(0.5·1₈). Tokens for visibility (OBS-Z / HID-Z) × graph (CONF / MED). N ≤ 256.
  - HID-MED is identified and uses the *same* 4-symbol tokens as HID-CONF, which controls for token format.
- **Head:** 64-bin categorical over p₁ = P(Y=1|do(X=1)). The target is the bin of the true p₁, so the loss minimizer is the posterior.
- **Oracles:** exact samplers (verified). For HID-CONF, the (x,y) margin is Dirichlet(2α+n) and the z-splits stay at their Beta(α,α) prior.
- **Readouts:**
  - the model/oracle posterior-sd ratio in each N bin;
  - log-log slope of sd against N: the expected oracle values are about −0.47 for OBS and HID-MED, and about −0.12 for HID-CONF;
  - coverage of 80% intervals;
  - sd divided by the width of the natural bounds.

**E5: the same witness pairs in Misra's task** (3 seeds; architecture d=192, 6 layers, 6 heads, learned PE, matching `RecurrenceTransformer`)
- **E5a, L2:** restart (x_{t+1}=f(v)) vs. clock (x_t:=f^t(x₀)) semantics.
  - Their observational sequences are identical.
  - Interventions happen at **uniformly random positions**, so no position goes untrained.
  - Each model should track its own oracle. With flags zeroed, both should collapse to Bayes-L1.
- **E5b, L3:** per-step noise x_{t+1}=ax_t+b+ε_t, with ε~Bern(0.2).
  - The query is marked in place with flag state 2, and q is uniform.
  - Reusing the inferred ε vs. resampling it gives two models with identical L1/L2 and different L3.
  - 20% of programs are held out, with the same model's L1 predictions on them as the matched control.
- Oracles enumerate all 289 programs on the GPU. I'll port upstream `verify_bayesian_calculations` (`bayesian-wind-tunnel/experiments/construction_boundary/recurrence_bwt.py:791-918`) to pytest.

**E2c: the closest test of Pearl's actual claim (full experiment, 3 seeds).**
- No edge tokens.
- The context mixes observational samples with do(A) and do(B) samples, each tagged with a Dawid regime token. This is exactly the experience of Pearl's chess-playing RL agent.
- The query is the untried do(A,B), plus untried single do(C).
- The oracle is Bayesian model averaging over the 443 DAGs using Cooper–Yoo interventional likelihoods under BDeu, so the transformer is compared against the ideal learner with the same information.
- **A "1¾" baseline** is model-free. It answers tried regimes from their observed frequencies, and for an untried regime it falls back on the observational conditional. That is the most a learner without a causal model can do.
- **Readout:** GC of the transformer between that baseline and the Bayes oracle, on queries where the oracle is sharp and on queries where it isn't.

**E6: Claude verbal probe (runs from the Mac during the GPU runs)**
- **Items:** verbal versions of E1, E3 and E4, generated from the same code with nonsense variable names and randomized wording. 50 contexts per condition, about 400 prompts per model.
  - **V1:** a count table, plus one sentence stating the causal ordering, or no sentence.
  - **V2:** the mechanism stated with a known w and a stated ρ, or ρ unstated.
  - **V3:** Z unmeasured vs. measured. Ask whether the effect is identifiable, then give a point estimate or the natural bounds.
- **Models:**
  - `claude-opus-5-5` (effort high);
  - `claude-sonnet-5-5` (effort high);
  - `claude-haiku-4-5` (thinking budget 4k);
  - `claude-fable-5-1` only if you opt in at CP1 (about 2.5× the cost of Opus).
- **API settings:**
  - Message Batches API, at a 50% discount.
  - Structured JSON output.
  - Refusals are recorded as their own category; no fallbacks, since we're measuring each model and Batches rejects them anyway.
- **Sizing and cost:**
  - At CP1, a pilot of 20 items per condition sets the item count by power analysis. The target is a CI half-width of ≤ 0.05 on slopes and ≤ 5 points on rates.
  - The projected cost goes to you for approval. The current guess is $20–50, with a hard cap enforced in code.
- **Credentials:** `ANTHROPIC_API_KEY` from `~/.env.convergence`, loaded only into the probe process. **I'll confirm this key with you at CP1.**

## Pre-registration (`PREDICTIONS.md`, committed before any training; final numbers fixed at CP1)
**Hypotheses**
- **H_info:** with the bits supplied, the model tracks the oracle and generalizes.
- **H_1¾:** the model tracks the oracle on training structures but fails on held-out graphs and combined interventions.
- **H_arch:** the model fails even in-distribution.

**Pass criteria** (means over seeds, 95% bootstrap CIs)
- **E1:**
  - split slope 0.9–1.1 with R² ≥ 0.9 on episodes with |Δ*| ≥ 0.02;
  - GC ≥ 0.85;
  - shuffled-control |slope| ≤ 0.05;
  - reversal-sign agreement ≥ 95% where both |ATE| > 0.02;
  - L2 accuracy within the CI of the matching FUNC query.
- **E2:**
  - in-distribution: GC ≥ 0.85 against model averaging;
  - held-out and combined: GC ≥ 0.8, and slope of (model−L1) on (L2−L1) ≥ 0.8 on discriminative episodes;
  - **fail** if GC ≤ 0.5.
- **E3:**
  - crossing slopes within ±0.05 of ρ;
  - held-out-ρ slope within ±0.1;
  - controls |slope| ≤ 0.05;
  - normalized excess MSE ≤ 0.1.
- **E4:**
  - sd ratio 0.85–1.15 in every N bin;
  - log-log slopes: OBS and HID-MED in [−0.6, −0.35], HID-CONF in [−0.25, 0];
  - 80% coverage 0.77–0.83.

**Outcome mapping for E2**

| Outcome | Held-out / combined L2 | Matched L1-with-graph (COND) |
|---|---|---|
| H_info | GC ≥ 0.8 | GC ≥ 0.8 |
| H_1¾ | GC ≤ 0.5, while in-distribution passes | GC ≥ 0.8 |
| H_arch | fails, or in-distribution also fails | fails |
| Inconclusive | GC 0.5–0.8 → train to convergence (up to the 4× cap) and run the capacity sweep, then report the final status | — |

- Verdicts are reported per fold and pooled.
- **E2c** uses the same table: the ideal-Bayes oracle replaces Bayes-L2, and the "1¾" baseline replaces blind model averaging.
- E1, E3 and E4 failures are treated as bugs, not evidence.
- Nothing is read as a claim about all transformers.

## Claims discipline for the post
Avoid saying:
- that this "refutes Pearl/the CHT". The design *relies* on the CHT; giving the graph is Pearl's own recipe; "1¾" was about model-free RL.
- that the model "learned causal effects from observational data". Its training targets were the true L2/L3 quantities.
- that it "resolves" Simpson's paradox. It is told the ordering.
- that it "performs abduction internally" or "knows it can't know". Only the input-output behavior matches, and the height of the HID plateau is set by the prior.
- that this "confirms Mossé–Ibeling–Icard". That result is about satisfiability complexity.
- anything beyond the truncated factorization, beyond the training range of N, or about LLMs from the small-model results.

## Repo layout (`~/ladder-witnesses/`, a new git repo; `~/causal-wind-tunnel` stays untouched and its prompt is copied to `docs/original-plan.md`)
```
README.md  PLAN.md  PREDICTIONS.md  REPORT.md  LICENSE (MIT)  CITATION.cff  pyproject.toml  uv.lock
witnesses/
  model.py       TinyGPT: pre-LN, SDPA causal attention, learned PE, discrete and continuous
                 inputs, multiple readout heads
                 (adapted from introspection/introspection/models/transformer.py:16-143)
  train.py       on-GPU sampling; soft-target CE/MSE; AdamW + warmup/cosine; clip 1.0; bf16;
                 resumable checkpoints (including optimizer state)
  meta.py        records seed, device, git hash, torch/CUDA/GPU, UTC times, steps/s
  tasks/         e1_simpson.py  e2_dags.py  e3_twin.py  e4_identify.py  e5_recurrence.py
                 (each: generator, exact vectorized oracles, eval sets)
  analyze.py     metrics and bootstrap CIs → results/summary.json
  plots.py       Tufte style: range frames, direct labels, small multiples, PNG+PDF
                 (built from prolate-claude/plot.py:20-46)
  llm_probe/     items.py  run_batch.py  score.py
configs/<exp>/<cond>.yaml (+ smoke.yaml)
scripts/
  run_suite.sh   nohup/tmux, UTC log, skips finished runs, writes _DONE/_FAILED
                 (pattern from auxiliary-causal/scripts/run_full_suite.sh)
  benchmark.py  aws_up.sh  aws_down.sh  sync_up.sh  sync_down.sh
tests/           oracles vs brute force / Monte Carlo; witness equivalence and leak probes;
                 generator invariants; train smoke tests
results/<exp>/<cond>_seed<k>/{metrics.json, eval_preds.npz, meta.json}
results/figures/                (checkpoints go to runs/, which is gitignored)
blog/            post.src.md  build_ghost.mjs  lint.mjs  post.ghost.md  preview.html  figures/
```

## Blog pipeline (checked against the live site and Ghost's source)
**What the site does:**
- symmetrybroken.com is Ghost 6.68 with KaTeX 0.16.9 auto-render. The math delimiters are `$$…$$`, `\[…\]` and `\(…\)`; a single `$` is not a delimiter.
- Your posts go through a Markdown card: `markdown-it` 14.3.0 with `{html:true, breaks:true, linkify:true}` plus footnote, sub, sup, mark and lazy-headers. This is TryGhost/Koenig `kg-markdown-html-renderer`.

**Four ways math breaks**, all reproduced locally with that exact configuration:
1. `\{` → `{`. Your published post shows `\({V_i^A}\)` without its braces.
2. `e^{-x^2}` → `e<sup>{-x</sup>2}`, from the sup plugin.
3. Underscores in two separate inline spans pair up into `<em>`.
4. A `<div class="math">` that doesn't start its own line gets parsed as Markdown, which drops `\,`.

**Pipeline:**
- Write `post.src.md` in plain `$…$` / `$$…$$`. It renders on GitHub as-is.
- `build_ghost.mjs` converts it:
  - inline math becomes `\\(…\\)` with every Markdown-active character escaped;
  - display math becomes `<div class="math">$$…$$</div>` on its own lines, with no blank lines inside;
  - each paragraph is a single line;
  - `[^n]` footnotes are kept.
- `lint.mjs` renders the result with Ghost's exact markdown-it version and plugins. It **fails** unless every math span survives byte-for-byte and parses in KaTeX 0.16.9.
- It also writes `preview.html` with the site's KaTeX settings, so you can check it in a browser.
- Figures are 1600-px PNGs, linked from `raw.githubusercontent.com/riemannzeta/ladder-witnesses/main/blog/figures/`, or uploaded to Ghost image cards.

**Post plan:** about 3–3.5k words, in your first person, with an optional AI-assistance note for you to keep or cut. Outline:
1. "Stuck on rung one?" Misra's wall, Pearl's 1¾, and Pearl's 2023 reconsideration.
2. What the CHT actually says.
3. A wind tunnel for the ladder.
4. E1.
5. E2, the real test.
6. E3: the blind counterfactual is exactly the interventional answer.
7. E4.
8. E5: back in Misra's tunnel.
9. E6: and Claude?
10. What this does and doesn't show.

Footnotes carry the citations.

## Workflow and checkpoints
**Phase A — local build on the M1, no GPU spend.**
- Scaffold the repo with uv and Python 3.12+.
- Write the tasks, oracles, tests and model.
- Smoke-test every experiment end-to-end on MPS/CPU.
- Freeze the E2 held-out topologies.
- Write `PREDICTIONS.md` and commit it locally.
- Generate the E6 items and run a 5-item dry run (about $0.10).

**CP1 — I stop.** You review `PREDICTIONS.md`, the test results, example episodes with oracle values, the smoke curves, and the E6 samples with a cost estimate. You also confirm the API key, and whether to include Fable.

**Phase B1 — GPU, first wave (about 1 day): benchmark → E1, E3, E4 (5 seeds each) → E2-0 (all folds, 3 seeds).**
- Start `i-REDACTED`.
- Connect over SSH: Tailscale `REDACTED-TAILSCALE-IP`, or the public IP with `~/ml-dev.pem` if the Tailscale key has expired.
- Run `uv sync`, then the queue in tmux.
- Pull results down as they land. Crashed runs are re-queued.
- The instance stops itself when the queue drains.
- The E6 batches run from the Mac in parallel, once you approve their size at CP1.

**CP1.5 — I stop.** I report results from the first wave. E2-0 shows whether the graph → truncated-factorization step generalizes before we spend days on E2-full. You approve or adjust the long queue.

**Phase B2 — GPU, long queue (about 2–4 days): E2-full (5 folds × 3 seeds) → capacity sweep → E2c → E5.**
- Unattended and resumable, with automatic stop at the end.
- I check in periodically, and whenever you resume the session.

**Phase C — analysis and writing.**
- `analyze.py` → `REPORT.md`, giving every pre-registered prediction a pass or fail with CIs and no softening.
- Figures, the README, and the post through build, lint and preview.

**CP2 — I stop.** You review the results, figures, post and repo.

**Phase D — publish, only on your explicit go.**
- `gh repo create riemannzeta/ladder-witnesses --public --source . --push`. You're already authenticated.
- Check that the figure URLs resolve.
- Hand you `post.ghost.md` to paste into a Ghost Markdown card.

## Risks
- **E2 learnability.** Handled by the convergence policy, the capacity sweep and the folds. The fallbacks (N ≤ 64, then k=3) apply only if even the largest converged models fail in-distribution, and that would be reported explicitly.
- **A multi-day unattended queue.** Mitigated by:
  - resumable checkpoints;
  - skipping finished runs;
  - `_FAILED` markers with logs;
  - the instance stopping itself.

  A crash costs one run, not the queue.
- **Expired Tailscale key.** Use the public IP instead.
- **g5 capacity at start.** Retry.
- **E6 refusals and parse failures** are reported as their own category.

## Verification
- `uv run pytest` passes: oracles, leak probes, invariants.
- The smoke suite runs end-to-end on the Mac.
- Each run writes `metrics.json` and `meta.json`.
- The tables in `REPORT.md` are regenerated from `results/summary.json`.
- `node blog/lint.mjs` passes, and `preview.html` renders.
- Figure URLs resolve after the push.
- `aws ec2 describe-instances` shows the instance `stopped`.
