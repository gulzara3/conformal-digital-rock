# Methods in brief

This is a compact reference for the code in [`src/conformal_rock`](../src/conformal_rock). Equation numbers follow the paper.

## Setting

A predictor is trained on the training split; `n` further volumes are held out for calibration. For each property `j` and miscoverage level `α`, a conformal method returns an interval `C_j(x)` with the property-wise guarantee

```
P( Y_j ∈ C_j(X) ) ≥ 1 − α,     j = 1, …, 6                                   (1)
```

over the calibration and test draws, assuming calibration and test volumes are exchangeable. Eq. (1) does **not** imply joint coverage of several properties.

## Threshold rule: `conformal.conformal_quantile`

```
q̂ = s_(k),   k = ⌈(n + 1)(1 − α)⌉,   q̂ = +∞ if k > n                           (2)
```

Ties are kept, which keeps the guarantee and makes it slightly conservative.

## Conformal predictors: `conformal.interval`

| Method | Score `s_i` | Interval | Needs |
|---|---|---|---|
| `split` (3) | `|y − μ̂(x)|` | `[μ̂ − q̂, μ̂ + q̂]` | point prediction |
| `norm` (4) | `|y − μ̂(x)| / (σ̂(x) + ε)` | `[μ̂ − q̂(σ̂ + ε), μ̂ + q̂(σ̂ + ε)]` | plus a scale; `ε = 10⁻³ ×` training sd |
| `cqr` (5) | `max(q̂_lo − y, y − q̂_hi)` | `[q̂_lo − q̂, q̂_hi + q̂]` | quantile pair matched to `1 − α` |
| `mondrian` (6) | split score within group `g(x)` | split interval with `q̂_g` | groups fixed before calibration |

* **Mondrian groups** are quintiles of the property's own predicted value, with edges taken from **training-set predictions**. Calibration data never shape the partition, so `P(Y ∈ C(X) | g(X) = g) ≥ 1 − α` holds exactly. A group too small for a finite threshold would use the pooled threshold (never needed; smallest group ≈ 530).
* **CQR** has ten quantile heads (0.01, 0.025, 0.05, 0.10, 0.15 and complements) and uses the pair closest to the target. Crossed heads are sorted; a negative `q̂` narrows the band; an inverted band collapses to its midpoint and counts as a miss.

## Heuristic baselines

Gaussian head `μ̂ ± z σ̂`; MC-dropout with 30 passes in two forms (7): spread of the means only, `Var_t(μ̂_t)`, and total variance, `mean_t(σ̂_t²) + Var_t(μ̂_t)`; a five-member deep ensemble trained on the Gaussian likelihood. Batch normalisation always uses running statistics; dropout is active only during MC-dropout passes.

## Exact test of the implementation: `conformal.coverage_law`, `conformal.discrete_ks_test`

For a fixed predictor and a random split of a pool into `n` calibration and `m` test volumes, the number of covered test volumes is

```
H ~ BetaBinomial(m, k, n + 1 − k),   k = ⌈(n + 1)(1 − α)⌉                      (9)
```

for continuous scores. Observed values from 900 random partitions are compared with this law by a discrete Kolmogorov–Smirnov statistic with a Monte Carlo p-value.

## Joint coverage: `conformal.joint_box`, `conformal.bonferroni_box`

```
s_i = max_j |y_ij − μ̂_j(x_i)| / d_j(x_i),   C(x) = Π_j [ μ̂_j − q̂ d_j, μ̂_j + q̂ d_j ]   (8)
```

with `d_j` either the training residual sd (fixed) or `σ̂_j(x) + ε_j` (adaptive). Bonferroni runs split conformal at `α / P`.

## Out-of-distribution guard: `shift.novelty_scores`, `shift.conformal_pvalue`

Novelty is the mean distance to the ten nearest training volumes in standardised penultimate-layer features. With calibration novelty scores `s_i`,

```
p(x) = (1 + #{i : s_i ≥ s(x)}) / (n + 1),     P( p(X) ≤ δ ) ≤ δ                   (10)
```

Volumes with `p(x) ≤ δ = 0.05` are flagged and sent to simulation.

## Weighted conformal prediction: `shift.weighted_threshold`, `shift.density_ratio`

Calibration scores are reweighted by the density ratio `w(x)` (Tibshirani et al., 2019), with the test point's mass placed at `+∞` (11). The ratio is estimated by logistic regression between calibration descriptors and unlabelled deployment descriptors.

## Risk-controlled triage: `triage.ltt_threshold`, `triage.crc_threshold`

Accept the prediction when an uncertainty score `u(x) ≤ t`, otherwise simulate. A bad accept misses permeability by more than a factor `f`:

```
R(t) = P( u(X) ≤ t, |μ̂(X) − Y| > log10 f )                                    (12)
```

Learn-then-Test (Angelopoulos et al., 2025) tests a fixed ascending grid of thresholds (quantiles of the score on the monitoring slice) with exact binomial p-values and fixed-sequence testing, so `R(t̂) ≤ ε` with probability at least `1 − δ` (here `δ = 0.1`). Conformal risk control bounds the expected risk instead.
