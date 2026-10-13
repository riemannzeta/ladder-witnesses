# ladder-witnesses

**Can a transformer climb Pearl's causal ladder when it is handed exactly the information the Causal Hierarchy Theorem says is necessary?**

The Causal Hierarchy Theorem (CHT) says that data from a lower rung (observation, intervention) almost never determine answers on a higher rung (intervention, counterfactual). The theorem constrains information, not learners. It does not say which architectures can compute higher-rung answers once the missing information is supplied.

This repo tests that with **witness pairs**. These are two causal models that agree on rung *k* and disagree on rung *k+1*, which is the device behind the theorem. Any reasoner limited to rung *k* must answer both models identically. A small transformer is given the distinguishing information in its input, and we check whether its answers diverge exactly as an exact Bayes oracle says they should. A format-matched control gets the same input with that information scrambled.

| ID | Rung | Witness pair | Question |
|---|---|---|---|
| E1 | L2 | same joint distribution, six complete causal orderings | one bit buys the rung? |
| E2 | L2 | 4-node DAGs, labels shuffled, 5-fold held-out topologies | Pearl's "1¾": untried and combined interventions on unseen graphs |
| E2c | L2 | no graph; tried do(A), do(B) | combine interventions without being given a model |
| E3 | L3 | Y=wX+noise with cross-world correlation ρ (identical L1/L2 for all ρ) | counterfactual residual transfer |
| E4 | CHT | Z observed vs hidden, confounder vs mediator | does posterior uncertainty stop shrinking exactly when the query is not identified? |
| E5 | L2/L3 | restart vs clock semantics; abducted vs fresh noise, in Misra's mod-17 recurrence | same witnesses in the "Bayesian wind tunnel" |
| E6 | L2/L3 | verbal versions of E1/E3/E4 | how do Claude models do? |

## Results in brief

Full verdicts, with confidence intervals: [`REPORT.md`](REPORT.md). Pre-registration: [`PREDICTIONS.md`](PREDICTIONS.md), with Addenda 1–4 for every later change.

- **Given the distinguishing information, the model computes the higher-rung answer.**
  - E1: split slope 1.000 across the six orderings; 0.000 with the ordering scrambled.
  - E3: counterfactual slope within 0.009 of ρ, and within 0.033 in the held-out ρ band.
  - E2: untried interventions on unseen graphs close 100% of the gap with CPTs and 98.5% with samples.
- **Without a graph (E2c):** given data from tried do(A) and do(B), the untried do(A, B) closes 97% of the gap between the best model-free learner and ideal Bayesian model averaging.
- **Composite queries failed in the pre-registered runs:** two-variable conditioning in E2-0, and combined interventions as well as conditioning in E2-full. Neither capacity (0.8M–14M parameters) nor training length helped.
  - **Post hoc:** adding two-element queries to training fixed it. With CPTs (E2-0), it transferred to every unseen graph (GC 0.98). With samples (E2-full, all folds at 240k steps), every fold's training graphs passed and most unseen graphs did too.
  - **Two exceptions, both with samples, neither improved by longer training:**
    - two-variable conditioning on the only two graph shapes where a collider has a child (fold 1, which never trains on that pattern);
    - combined do() on the 5-edge graph T24 (fold 2), where the matched conditioning query passes.
- **E4:** posterior width plateaus exactly where the effect is not identified (log-log slope −0.12, against −0.13 for the exact posterior). The sd-ratio and as-implemented coverage criteria failed (a far-tail softmax floor, and a flawed coverage metric).
- **E5:** the same witness pairs work in Misra's recurrence on seen programs (GC_TV ≥ 0.99). On held-out programs, even factual prediction fails.
- **E6:** Opus 5.5, Sonnet 5.5 and Fable 5.1 answer verbal versions correctly and say "not determinable" when the information is missing. Haiku 4.5 often does not.

None of this refutes the CHT, which the design relies on. Training targets were true higher-rung quantities, the models are tiny, and nothing here is evidence about LLMs.

- **Plan:** [`PLAN.md`](PLAN.md). **Figures:** `results/figures/`. **Blog post source:** `blog/post.src.md`.
- **Reproduce:** `uv sync --group dev && uv run pytest`, then `python -m witnesses.train --config configs/<exp>/<cond>.yaml --seed 0` (add `--smoke` for a 1-minute pipeline check).
- **History:** AWS instance identifiers were redacted from the git history before publication. Commit dates are unchanged; [`COMMIT_MAP.tsv`](COMMIT_MAP.tsv) maps original commit IDs to published ones. The GPU helper scripts read machine-specific values from an untracked `scripts/aws_env.local.sh`.

MIT licensed. © 2026 Michael Frank Martin.
