---
title: Task Plots
description: Ready-made task-specific plotting modules returned by `TaskAdapter.get_plots()`.
---

Task-specific plot APIs live under `tasks.plots.*`. In normal use you do not import them manually; you ask the active adapter for its plotting module:

```python
from tasks import get_adapter

adapter = get_adapter("two_afc")  # or "nuo_auditory", "mcdr"
plots = adapter.get_plots()
```

Available ready-made modules:

- `tasks.plots.two_afc`
- `tasks.plots.nuo_auditory`
- `tasks.plots.mcdr`

---

## Shared High-Level Surface

All task plotting modules expose the same core diagnostics used in the notebooks:

```python
plots.plot_emission_weights(...)
plots.plot_transition_matrix(...)
plots.plot_posterior_probs(...)
plots.plot_state_accuracy(...)
plots.plot_session_trajectories(...)
plots.plot_state_occupancy(...)
plots.plot_state_dwell_times_by_subject(...)
plots.plot_state_dwell_times_summary(...)
plots.plot_state_dwell_times(...)
plots.plot_session_deepdive(...)
```

These functions wrap the shared helpers in [`glmhmmt.plots_common`](/docs/api/common-plots) and inject task-specific column names, labels, and styling.

**Example**

```python
fig = plots.plot_state_accuracy(views, trial_df, thresh=0.6)
```

---

## Binary Task Modules

`tasks.plots.two_afc` and `tasks.plots.nuo_auditory` provide the same extra families of ready-made plots:

- `plot_emission_weights_by_subject`
- `plot_emission_weights_summary`
- `plot_transition_matrix_by_subject`
- `plot_categorical_performance_all`
- `plot_categorical_performance_all_by_state`
- `plot_regressor_psychometric_by_state`
- `plot_model_comparison`
- `plot_model_comparison_diffs`

They also keep lower-level primitives available for direct use:

- `plot_weights`
- `plot_weights_per_contrast`
- `plot_weights_boxplot`
- `plot_trans_mat`
- `plot_trans_mat_boxplots`
- `plot_occupancy`
- `plot_occupancy_boxplot`
- `plot_ll`

**Typical use**

```python
fig = plots.plot_emission_weights_summary(views=views, K=3)
fig = plots.plot_transition_matrix_by_subject(
    arrays_store=arrays_store,
    state_labels=state_labels,
    K=3,
    subjects=subjects,
)
```

---

## MCDR Task Module

`tasks.plots.mcdr` combines the shared diagnostics with MCDR-specific behavioural plots:

- `plot_categorical_performance_by_state`
- `plot_categorical_performance_all`
- `plot_categorical_strat_by_side`
- `plot_delay_binned_1d`
- `plot_tau_sweep`
- `plot_transition_weights`

The module also re-exports the standard diagnostics from `glmhmmt.model_plots`, so the notebook-facing plotting surface stays consistent across tasks.

---

## Emission Weight Convention In The MCDR Notebooks

The MCDR notebook summaries use a post-hoc collapse convention to turn the
stored multinomial emission tensor into task-level grouped summaries. This does
not change the fitted model. It is only an interpretation layer on top of the
softmax parameterisation.

```python
W.shape == (K, 2, n_features)
# rows: [Left vs Center, Right vs Center]
# Center is the reference class, so its logit is implicit 0.

logits_f = [W[k, 0, f], 0.0, W[k, 1, f]]
p_f = softmax(logits_f)
```

There are two baselines:

- logit baseline: the reference class has fixed logit `0`
- probability baseline: after reconstructing the full softmax, probabilities are compared to the uniform baseline `1 / C`, which is `1 / 3` here

So for one isolated feature `f` in state `k`, the aligned readouts are:

- Left-aligned effect: `P(L | f) - 1/3`
- Center-aligned effect: `P(C | f) - 1/3`
- Right-aligned effect: `P(R | f) - 1/3`

The grouped MCDR summaries average those aligned readouts over symmetric task
members. In practice:

- `mode=0` means read out `P(L) - 1/3`
- `mode=1` means read out `P(R) - 1/3`
- `mode="neg_mean"` means read out `P(C) - 1/3`
- `mode="mean"` means read out `((P(L) + P(R)) / 2) - 1/3`

This is the implementation used by the MCDR plot summaries in
`glmhmmt.model_plots`.

## Weight-Space Shortcut

The shorter notebook comment is a useful mnemonic, but it is only a shortcut:

- for symmetric L/R pairs, `mean(W[k, 0, feat_L], W[k, 1, feat_R])`
- for Center features, `-mean(W[k, 0, feat_C], W[k, 1, feat_C])`
- for shared scalars, `mean(...)` across the explicit rows

The sign flip for Center happens because Center has no explicit row in the
stored tensor. If a feature increases both `L-vs-C` and `R-vs-C` logits, it is
pushing probability away from Center, so the derived Center summary should go
down.

The probability-space collapse is more faithful than the raw-weight shortcut,
because each class probability depends on all logits, not just one stored row.

## Generalising Beyond 3 Choices

For any task with `C` choices, the same generic rule applies:

1. choose a reference class `r`
2. reconstruct the full logits with `z_r = 0`
3. compute `p = softmax(z)`
4. define task-specific symmetry groups such as `(feature, target_class)`
5. collapse with `mean(p[target_class] - 1 / C)`

If a group member maps to the reference class, use `p[r] - 1 / C`. If it maps
to a neutral competitor set rather than one class, average over that set before
subtracting `1 / C`.

For tasks with more than 3 choices, there is no single canonical agonist
collapse. The symmetry groups have to come from the task geometry, but the
probability-space rule above stays the same.

This convention is specific to the MCDR notebook plots and their grouped
readouts. The stored model weights themselves follow the general softmax
convention described in [`SoftmaxGLMHMM`](/docs/api/model).
