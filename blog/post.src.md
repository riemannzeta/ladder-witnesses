# Witnesses on the Ladder

*What the Causal Hierarchy Theorem does — and doesn't — say about transformers, tested in a wind tunnel.*

<!-- DRAFT. Sections marked RESULT PENDING are filled in from REPORT.md after the runs finish. -->

## Stuck on rung one?

Judea Pearl's ladder of causation has three rungs. On the first you *see*: you can estimate associations like $P(y \mid x)$ from data. On the second you *do*: you can say what happens to $Y$ if you reach in and set $X$, written $P(y \mid do(x))$. On the third you *imagine*: you can ask what *would have* happened to this particular case had things been different, $P(y_x \mid x', y')$.

A familiar argument says that large language models are stuck on the first rung. They learn from text, text is observational data, and the Causal Hierarchy Theorem (CHT) says observational data cannot answer interventional questions. Vishal Misra's "wall between Shannon and Kolmogorov" experiment was offered in the same spirit: a transformer trained on a modular recurrence predicts beautifully where it was trained and falls off a cliff where it wasn't, which he read as pattern-matching rather than a learned algorithm. When I followed up on that wall, it turned out to be a *calibration* barrier rather than a missing algorithm: give the network a single scalar hint about how uncertain to be, and the "missing" algorithm was there all along.[^wall]

Pearl himself put the reinforcement-learning version of the argument memorably in 2023. RL, he said, is "at level one and three-fourths. Reinforcement learning trains machines on interventions. … However, they cannot infer from this anything about a third move they haven't tried. They also cannot combine interventions to infer what will happen if they do both A and B." In the same interview he said something that gets quoted less often: "I have had to reconsider my proof that one cannot get any answer to any causal or counterfactual query from observational studies," because "text may contain information on levels two and three."[^pearl]

That second remark is the key to this post. The CHT is a theorem about **information**, not about **learners**.

## What the Causal Hierarchy Theorem actually says

The theorem (Bareinboim, Correa, Ibeling and Icard) says that for almost every structural causal model, the distributions on a lower rung leave the answers on the higher rung undetermined.[^cht] The proof device is a *witness pair*: two causal models that agree on everything at rung $k$ and disagree at rung $k+1$.

Here is the textbook pair at rung two. Take three binary variables and the same joint distribution $P(z, x, y)$. In one story $Z$ is a confounder ($Z \to X$, $Z \to Y$, $X \to Y$); in the other $Z$ is a mediator ($X \to Z \to Y$, $X \to Y$). Both graphs are complete, so both fit *any* joint distribution, and no amount of observational data can tell them apart. Yet the effect of intervening differs:

$$
P(y \mid do(x)) \;=\;
\begin{cases}
\sum_z P(y \mid x, z)\, P(z) & \text{if } Z \text{ is a confounder,}\\
P(y \mid x) & \text{if } Z \text{ is a mediator.}
\end{cases}
$$

When these two numbers have opposite signs you get Simpson's paradox, and the data alone cannot resolve it.

And here is a pair at rung three. Let $N$ be a fair coin and compare $Y := X \oplus N$ with $Y := N$. Under every intervention $Y$ is a fair coin in both models, so they are indistinguishable at rung two. But if we saw $X = 1, Y = 1$ and ask what $Y$ would have been had $X$ been $0$, the first model says $0$ and the second says $1$.

Any reasoner that uses only rung-$k$ information must give the *same* answer to both members of a witness pair. That is true of transformers, of people, and of do-calculus engines alike. So the theorem by itself cannot tell us that transformers are stuck: a transformer that fails to climb from rung one without extra information is merely illustrating the theorem, and so would any other learner. The interesting questions are different:

1. **Given** the information that distinguishes the pair, does the model compute the higher-rung answer?
2. Does it generalize that computation to structures and combinations it never saw?
3. And when the information is *not* given, does its uncertainty reflect that?

There is no complexity-theoretic reason to expect question 1 to be hard. An intervention is just "cut the arrows into $X$, then do ordinary inference", and a counterfactual is ordinary inference on a doubled copy of the model. Mossé, Ibeling and Icard proved that causal reasoning problems reduce to probabilistic ones with no jump in computational complexity.[^mii] How *much* information the higher rungs need can be large — Emadi recently quantified the gap in bits[^emadi] — but that is a statement about what must be supplied, not about who can use it.

## A wind tunnel for the ladder

Misra's other idea is the one I borrow: a "Bayesian wind tunnel", a synthetic task where the exact Bayes-optimal answer is computable, so a small transformer's predictions can be graded against ground truth to fractions of a bit.[^misra]

Every experiment below has the same shape:

- A **witness pair**: two causal models that agree one rung down and disagree on the rung being tested.
- The **distinguishing information** is given to the model as a token (a causal ordering, a graph, a mechanism parameter).
- An **exact oracle** computes the Bayes-optimal answer for each member of the pair.
- A **format-matched control** sees identical inputs with that token scrambled. By the theorem, its best possible answer is the same for both members.

Training is in the style of prior-fitted networks:[^pfn] each training episode's target is the *true* higher-rung quantity of that episode's causal model, so the loss is minimized exactly by the Bayes posterior under the training distribution. To be clear about what this does and does not test: the model is shown higher-rung targets during training. It is not learning causal effects from observational data — the CHT says nobody can. The question is whether, *given the bits*, it computes the right thing and generalizes it.

Everything was pre-registered before any GPU training: the hypotheses, the pass/fail thresholds, the held-out graphs, and the claims I would not make.[^prereg] Every run trains to convergence; runs that hit the step cap are labeled unconverged and never counted as negative evidence.

## Experiment 1: one bit buys the rung

Three binary variables, labels shuffled, a random joint distribution, and 128 samples. The query names a treatment and an outcome and asks for $P(Y = 1 \mid do(X = x))$ under one of the **six** complete causal orderings, given as a token. Depending on the ordering, the right answer is the adjusted formula, the crude conditional $P(y \mid x)$, or the marginal $P(y)$. The data carry no information about the ordering whatsoever.[^gh]

As a matched control on the arithmetic, half of the episodes instead ask for the *same formula by name*, with no causal language at all.

> **RESULT PENDING (E1):** split slope across orderings, gap closed vs the ordering-blind Bayes answer, Simpson-reversal sign agreement, and causal query vs the same formula asked by name.

![Same data, six orderings: the transformer's answers move exactly as the exact Bayes answers do when the ordering token is given, and not at all when it is scrambled.](figures/e1_witness.png)

## Experiment 2: Pearl's "one and three-fourths", with the graph supplied

Pearl's objection is about untried moves and combinations. Four binary variables; a random causal graph with in-degree at most two (25 shapes, 443 labeled graphs); labels shuffled every episode. The model is given the graph and either the conditional probability tables (E2-0) or 128 samples (E2-full), and is asked for $P(T = 1 \mid do(\cdot))$.

Every query is "untried" in Pearl's sense: the context contains no interventional data at all. Beyond that, three tests:

- **Unseen graphs.** Five-fold cross-validation over the 25 graph shapes; each fold's model never saw its five held-out shapes.
- **Combined interventions.** During training the second intervention slot is always empty or *inert* (it cannot affect the target). At test time both interventions matter, and conditioning gives a different answer from intervening.
- **Same data, swapped graph.** Two Markov-equivalent graphs fit the same data equally well but imply different interventions; does the model's answer move by the right amount?

As a matched control, the same model also answers *conditioning* questions on the same graphs.

> **RESULT PENDING (E2-0, E2-full):** gap closed per fold for in-distribution, held-out graphs, combined interventions; matched conditioning control; graph-swap slope; capacity sweep.

## Experiment 2c: combining interventions without being given a model

This is the closest test of Pearl's actual claim. No graph is given. The context holds observational samples plus samples from two *tried* randomized experiments, $do(A)$ and $do(B)$ — exactly the experience of his chess-playing RL agent. The queries are the untried combination $do(A, B)$ and the untried single intervention $do(C)$. The reference is the ideal Bayesian learner with the same information: model averaging over all 443 graphs. The baseline is the best a model-free learner can do: answer tried moves from their frequencies, and untried ones from the nearest tried experiment or from observation.

> **RESULT PENDING (E2c):** gap closed between the model-free baseline and the ideal learner, split by how sharply the data pin down the graph.

## Experiment 3: counterfactuals and the residual twin

Rung three needs information that interventions cannot supply. Let

$$
Y = wX + \sigma\left[\mathbb{1}\{X > 0\}\, U_1 + \mathbb{1}\{X \le 0\}\left(\rho\, U_1 + \sqrt{1 - \rho^2}\, U_2\right)\right],
$$

with $U_1, U_2$ independent standard normal noise private to each unit. For every value of $\rho$ the noise is $N(0, \sigma^2)$ and independent of $X$, so all observational *and* interventional distributions are identical across $\rho$: it is pure rung-three information. For a unit observed at $(x^*, y^*)$, the counterfactual "had $X$ been $x'$" has expectation

$$
E[Y_{x'} \mid x^*, y^*] = w x' + r\,(y^* - w x^*), \qquad r = \begin{cases} 1 & \text{same side of } 0,\\ \rho & \text{crossing.} \end{cases}
$$

At $\rho = 1$ the unit's residual carries over; at $\rho = -1$ it flips; at $\rho = 0$ the counterfactual equals the interventional answer $w x'$. The model gets $\rho$ as a number (training never shows $|\rho| < 0.3$) and must estimate $w$ from the context. A control sees a scrambled $\rho$, and its Bayes-optimal counterfactual collapses *exactly* to the interventional answer.

> **RESULT PENDING (E3):** residual-transfer slope vs $\rho$, including the held-out band; controls; excess error over the Bayes error.

![Counterfactual slope against the cross-world correlation ρ.](figures/e3_rho.png)

## Experiment 4: knowing what you can't know

The last demonstration turns the theorem itself into a measurement. With $Z$ observed, the effect of $X$ on $Y$ is identified and the exact posterior over it narrows like $1/\sqrt{N}$. With $Z$ hidden and confounding, it is not identified, and the posterior stops narrowing no matter how much data arrives. The control that makes this a clean test: with $Z$ hidden but acting as a *mediator*, the effect is identified again — using exactly the same input tokens.

> **RESULT PENDING (E4):** posterior width vs samples for the four cells; coverage.

![Posterior width of P(Y=1 | do(X=1)) as evidence accumulates, in four cells.](figures/e4_identifiability.png)

## Back in Misra's wind tunnel

> **RESULT PENDING (E5):** restart-vs-clock interventions and abducted-vs-fresh counterfactual noise in the modular recurrence.

## And Claude?

> **RESULT PENDING (E6):** verbal versions of the witness pairs posed to Claude models (Opus 5.5, Sonnet 5.5, Haiku 4.5, Fable 5.1).

## What this does and doesn't show

> **RESULT PENDING:** summary, limitations, what would change my mind.

*Code, configs, pre-registration and every number in this post: [github.com/riemannzeta/ladder-witnesses](https://github.com/riemannzeta/ladder-witnesses).*

<!-- Optional: "This work was carried out with substantial help from Claude (Anthropic), which wrote much of the code and an early draft of this post." Keep or cut. -->

[^wall]: See my earlier wall-erosion experiments: [github.com/riemannzeta/wall-erosion-experiment](https://github.com/riemannzeta/wall-erosion-experiment).

[^pearl]: Interview with Dana Mackenzie, *Amstat News*, September 2023.

[^cht]: E. Bareinboim, J. Correa, D. Ibeling and T. Icard, "On Pearl's Hierarchy and the Foundations of Causal Inference" (2022). See also K. Xia, K.-Z. Lee, Y. Bengio and E. Bareinboim, "The Causal-Neural Connection" (NeurIPS 2021), which shows that the theorem holds for neural causal models however expressive they are.

[^mii]: M. Mossé, D. Ibeling and T. Icard, "Is Causal Reasoning Harder Than Probabilistic Reasoning?", *Review of Symbolic Logic* 17 (2024). Their result concerns the satisfiability of causal versus probabilistic logics.

[^emadi]: S. M. Emadi, "The Causal Description Gap: Information-Theoretic Separations Across Pearl's Hierarchy" (2026).

[^misra]: N. Agarwal, S. R. Dalal and V. Misra, "The Bayesian Geometry of Transformer Attention" (2025).

[^pfn]: S. Müller et al., "Transformers Can Do Bayesian Inference" (ICLR 2022). Related causal versions: Do-PFN (Robertson et al., NeurIPS 2025) and in-context counterfactual reasoning (Miller, Schölkopf and Guo, NeurIPS 2025).

[^prereg]: `PREDICTIONS.md` in the repository, committed before any GPU run.

[^gh]: Under a Dirichlet prior on the joint distribution, every complete ordering factorizes into independent conditional priors (Geiger and Heckerman 1997), so the ordering is exactly uninformative about the data.
