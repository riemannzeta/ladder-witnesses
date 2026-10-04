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

- **Status:** under construction. The plan is in [`PLAN.md`](PLAN.md); pre-registered criteria are in [`PREDICTIONS.md`](PREDICTIONS.md); results will be in `REPORT.md`.
- **Reproduce:** `uv sync --group dev && uv run pytest`, then `python -m witnesses.train --config configs/<exp>/<cond>.yaml --seed 0` (add `--smoke` for a 1-minute pipeline check).

MIT licensed. © 2026 Michael Frank Martin.
