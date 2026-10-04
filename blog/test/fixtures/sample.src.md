# Ladder witnesses: a fixture for the Ghost math pipeline

This paragraph is hard-wrapped the way an editor wraps prose, so the
builder has to join these lines into one; otherwise Ghost's `breaks: true`
turns every newline into a `<br>`.

## The four failure modes

Failure mode 1, braces: the set of potential outcomes $\{V_i^A\}$ must keep
its escaped braces.

Failure mode 2, carets: the Gaussian kernel $e^{-x^2}$ has two carets with no
space between them.

Failure mode 3, underscores across spans: compare $\hat{\theta}_{zx}$ vs
$n_{z}$ in the same paragraph.

Failure mode 4, display math with `\,`, `\mid`, `<` and `&` in an aligned
environment:

$$
\begin{aligned}
P(Y = 1 \mid do(X = 1)) &= \sum_{z} P(Y = 1 \mid X = 1, Z = z)\, P(Z = z) \\

\hat{\tau}_{\text{naive}} &< \tau_{\max} \quad \text{if} \quad a \,\&\, b
\end{aligned}
$$

## Results for $\tau$ and the [$do$-calculus](https://en.wikipedia.org/wiki/Do-calculus)

The update $x_{t+1} = x_t - \eta \nabla f(x_t)$ uses the norm $\|g\|_2$ and
the intervention $\text{do}(x)$; the causal effect is $P(Y=1 \mid do(X=1))$.
A literal price is \$12, and `$x$` in inline code is not math. A literal
backslash \\$\alpha$ precedes math, and *an emphasised $\beta_1$* still
works, as does **bold $\gamma^{*}$**.

We rank this approach
#1 among the alternatives, and the year was
2019. Neither line may become a heading or a list.

#hashtags are not headings on GitHub, so they must not become one on Ghost.

A hard break is kept when the author asks for one:  
this line follows a deliberate `<br>`.

A display equation with explicit line breaks:

$$
f(x) = \begin{cases}
1 & x \ge 0 \\
0 & x < 0
\end{cases}
\qquad \|x\|_\infty = \max_i |x_i|
$$

| Quantity | Symbol | Note |
|---|:---:|---|
| Treatment | $X \in \{0, 1\}$ | binary |
| Norm | $\|v\|$ | escaped pipes inside a cell |
| Conditional | $P(a \mid b)$ | `\mid` |

- First item with $a_i^2$ and a continuation line that is
  indented, so it must be joined.
- Second item: $\sum_{i=1}^{n} w_i = 1$
  1. Nested item with $\mathbb{E}[Y_{x}]$.

     $$
     \mathbb{E}[Y_x] = \sum_u P(u)\, Y_x(u)
     $$

     Back in the nested item, $Y_x(u)$ again.
- Third item.

![A relative figure, alt text with $x$ is not typeset](figures/example.svg)

Footnotes work too.[^ci] And here is a second one.[^2]

> A blockquote with $\epsilon > 0$ and $\delta < \epsilon$
> keeps its line structure.

<details><summary>Proof sketch for $n_{z} \ge 1$</summary>

The bound holds because $\sum_z n_z = N$.

</details>

<!-- a comment: $x$ and $$ in here are ignored -->

```python
# fenced code: $$ and $x$ are not math here
price = "$$"
```

[^ci]: The interval $[\hat{\theta} - 1.96\,\hat{\sigma},\ \hat{\theta} + 1.96\,\hat{\sigma}]$
    is a footnote whose math survives the wrap.
[^2]: A short footnote with $\lambda_{\min}$.
