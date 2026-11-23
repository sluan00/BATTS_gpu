# batts_gpu: the boosting of BATTS on a GPU

A PyTorch port of the boosting part of BATTS: `boots()` (gradient and forward-stagewise boosting,
with cross-validation of the number of trees and per-tree subsampling) and `eval_balance_weight()`
for boosting fits. The Bayesian `batts()` is not ported: its sampler changes one node of one tree at
a time, each step depending on the last, so a GPU has little to work on in parallel.

Given a sample from p0 (group 0) and a sample from p1 (group 1), it fits an additive tree estimate of
the balancing weight w = sqrt(p0 / p1). The log density ratio is log(p0 / p1) = 2 log(w).

**Which BATTS it follows.** BATTS 0.3.0, the R code in this repository (commit `9ade23e` of
[nawaya040/BATTS](https://github.com/nawaya040/BATTS)): the arguments and defaults are those of
`boots()` in `R/`, and the port has been checked against that R code (see below).

## Install

Python 3.8 or later with NumPy and a CUDA build of PyTorch 2.0 or later. Then, from this folder:

```sh
pip install -e .
```

or put this folder on `PYTHONPATH`. Without a GPU the package runs on the CPU, far slower than R.

## Use

```python
import numpy as np
from batts_gpu import boots

x = np.vstack([sample_p0, sample_p1])                    # observations in rows
g = np.repeat([0, 1], [len(sample_p0), len(sample_p1)])  # group 0 is the numerator

fit = boots(x, g, seed=1)             # defaults of BATTS 0.3.0
fit.num_trees                         # number of trees chosen by cross-validation
fit.balance_weight                    # w at the training points
fit.log_ratio(new_points)             # log(p0 / p1) at new points
fit.predict(new_points, outside="nan")   # w; NaN instead of an error outside the fitted box
```

Many fits on subsets of the rows and columns of one data set run as one batch, which is where a GPU
pays off for small data. For example the model on all variables and the models with one variable left
out, with the same folds and subsamples for all of them:

```python
from batts_gpu import Task, boots_many

d = x.shape[1]
tasks = [Task(seed=1)] + [Task(cols=[k for k in range(d) if k != j], seed=1) for j in range(d)]
fits = boots_many(x, g, tasks)        # fits[i] is boots(x[rows][:, cols], g[rows], seed=task.seed)
fits[3].log_ratio(new_points)         # new_points keeps all d columns
```

A `Task` can also restrict the rows (`rows=`, for cross-fitting) and fix the folds (`fold_labels=`).
Two tasks on the same rows with the same seed share their folds and per-tree subsamples.

### From R (BATTS 0.3.0) to Python

| R | Python |
|---|---|
| `boots(data, group_labels, ...)` | `boots(data, group_labels, ...)`, same argument names and defaults |
| `set.seed(s)` before the fit | `seed=s` (different draws from R's, see below) |
| `fit$balance_weight_boosting_data` | `fit.balance_weight` |
| `fit$loss_CV_store` | `fit.loss_cv` |
| `length(fit$tree_list)` | `fit.num_trees` |
| `fit$tree_list` | `fit.tree_list()` (same format) |
| `fit$c`, `fit$Omega` | `fit.c`, `fit.omega` |
| `eval_balance_weight(fit, points)$balancing_weight_boosting` | `fit.predict(points)` or `eval_balance_weight(fit, points)` |
| `2 * log(...)` of the above | `fit.log_ratio(points)` |

`n_ratio_per_node` (reserved in BATTS) is dropped. `n_min_obs_per_node` must be at least 1.

## How it differs from R

1. **Random draws.** The folds and subsamples follow BATTS's scheme (group-stratified folds; per tree,
   one observation from each group and the rest uniformly without replacement) but come from this
   package's own generator, so a seed does not reproduce an R fit. A seed gives the same draws on a
   CPU and on a GPU, alone or inside any batch.
2. **Ties between cuts.** Cuts often have exactly the same loss: in the first tree the loss depends
   only on the group counts of the two children, and later two cuts tie whenever they divide the
   tree's subsample in the same way. BATTS takes the floating-point minimum, so rounding noise picks
   among them. Here losses within `tie_tolerance` (default 1e-9, relative to the loss of the node)
   count as equal and the first in BATTS's order is taken, which is BATTS's rule in exact arithmetic.
   This keeps GPU fits reproducible (GPU sums are added in an order that changes from run to run).
   `tie_tolerance=0` gives BATTS's floating-point rule.
3. **The final fit is boosted next to the cross-validation folds**, for `num_trees_max` trees, and cut
   at the chosen number. This gives the same model as fitting the chosen number afterwards.
4. **Precision.** Everything is in double precision, as in R. Results agree with R up to rounding
   (see the next section), not bit for bit.

## Checked against R (BATTS 0.3.0)

The tests, the benchmarks and their job scripts are kept outside this repository. The results, of 23
Nov 2026, with R built from this repository:

The reference is 9 real `BATTS::boots()` fits (gradient and forward-stagewise, with and without
subsampling, depths 0 to 5, 2 to 8 variables, tied data, unequal groups), with the folds and
subsamples R drew and the trees of every model, CV folds included: 35 models and 1204 trees. The port
is given R's draws.

- **Replay.** Made to follow R's trees, the port reports how far each of R's choices is from what its
  own search allows. On all 35 models, every split R made minimises the port's loss and improves on
  its node, and no node R left as a leaf has an improving cut, to within 6e-16 (relative). The leaf
  values, held-out loss curves, chosen numbers of trees, weights and predictions are then R's to
  within 2e-15. This holds on the CPU and on the GPU (NVIDIA A100).
- **Free run.** Searching by itself, the port gives R's fit to 1e-9 in 6 of the 9 cases (CPU and GPU,
  with either tie rule). In the other 3 a tie fell differently somewhere; their held-out loss curves
  stay within 2% of R's and reach the same minimum to 0.5%.

Further tests cover a batch of `boots_many` against separate `boots` calls, CPU against GPU,
reproducibility, the subsample draws, prediction and input errors.

## Speed

Default settings (up to 1000 trees, 5-fold CV, depth 4) on simulated data with a known ratio: the two
groups differ in the first two variables, the others are noise. R is BATTS 0.3.0 on one core (AMD EPYC
7542), timed once. Times are wall clock for the whole call. RMSE is the error of the estimated
log(p0 / p1) against the truth on 5000 fresh points.

By sample size, 10 variables, port on one NVIDIA A100 (timed once):

| n | R (s) | GPU (s) | R / GPU | RMSE, R | RMSE, GPU |
|---|---|---|---|---|---|
| 2,000 | 12.3 | 5.3 | 2x | 0.504 | 0.497 |
| 20,000 | 105.5 | 5.8 | 18x | 0.262 | 0.263 |
| 200,000 | not run | 18.1 | | not run | 0.148 |

By number of variables d, port on one NVIDIA A40 (median of 3 fits):

| n | d | R (s) | GPU (s) | R / GPU | RMSE, R | RMSE, GPU |
|---|---|---|---|---|---|---|
| 2,000 | 10 | 12.3 | 7.1 | 1.7x | 0.466 | 0.481 |
| 2,000 | 20 | 21.2 | 7.1 | 3.0x | 0.511 | 0.519 |
| 2,000 | 30 | 28.7 | 7.1 | 4.1x | 0.487 | 0.499 |
| 2,000 | 40 | 37.6 | 7.0 | 5.3x | 0.529 | 0.538 |
| 2,000 | 50 | 46.7 | 7.0 | 6.6x | 0.489 | 0.498 |
| 20,000 | 10 | 109.5 | 7.0 | 16x | 0.263 | 0.263 |
| 20,000 | 20 | 185.0 | 8.2 | 23x | 0.305 | 0.314 |
| 20,000 | 30 | 268.1 | 10.4 | 26x | 0.301 | 0.301 |
| 20,000 | 40 | 252.8 | 12.6 | 20x | 0.310 | 0.307 |
| 20,000 | 50 | 312.9 | 14.8 | 21x | 0.313 | 0.313 |

Several fits at n = 2,000 and d = 10 as one batch, on the A100:

| Fits | R, one after another (s) | GPU, one batch (s) | R / GPU |
|---|---|---|---|
| 11: all variables, and each variable left out | 119.6 | 5.2 | 23x |
| 55: the same on each of 5 cross-fitting parts | not run | 10.0 | |

- A GPU fit has a fixed cost of 5 to 7 s, whatever d at n = 2,000 and whether the batch holds 1 fit
  or 11, while R's time grows with n and with d. So one small fit gains little, and the gain grows
  with the number of variables (at n = 2,000), the sample size and the number of fits boosted
  together. At n = 20,000 the GPU time grows with d too, and the gain levels off at 20x to 26x.
- The accuracy is R's. The two fits differ in their random draws, hence small differences either way.
- **On a CPU the port is much slower than R**: 334 s at n = 2,000 and d = 10 on 4 threads, against
  12 s. Use the R package when there is no GPU.

## License

MIT, as BATTS. BATTS is by Naoki Awaya, Yuliang Xu and Li Ma; this port follows its source
(`src/class_balancePM.cpp` and `R/BATTS_functions.R` at commit `9ade23e`).
