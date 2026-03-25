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

The MCDR notebook summaries use the following collapse convention when turning the stored emission tensor into interpretable grouped coefficients:

```python
# ── emission weights ───────────────────────────────────────────────────────
# W shape: (K, 2, n_features)  — axis-1: [L-choice=0, R-choice=1]
# Center = reference class (implicit weight 0).
#
# Agonist collapse: for symmetric L/R feature pairs, take
#   mean(W[k, 0, feat_L], W[k, 1, feat_R])  → one point per group per state
# For C features (no direct weight): -mean(W[k, 0, feat_C], W[k, 1, feat_C])
# For shared scalars: mean across both rows.
#
# Groups: (label, [(feat_name, class_idx), ...])
# class_idx int = direct weight; "neg_mean"/"mean" = derived from both rows
# Coherent = cue and choice on same side; Incoherent = opposite side
```

This convention is specific to the 3-choice MCDR notebook plots. The stored model weights themselves follow the general softmax convention described in [`SoftmaxGLMHMM`](/docs/api/model).
