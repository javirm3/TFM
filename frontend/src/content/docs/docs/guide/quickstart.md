---
title: Quickstart
description: Install glmhmmt and fit your first GLM-HMM in minutes.
---

## Installation

This repository is organised as a small workspace. The recommended setup is:

```bash
git clone https://github.com/javirm3/TFM
cd TFM/code
uv sync
uv pip install -e glmhmmt/
```

**Requirements:** Python ≥ 3.11, JAX ≥ 0.9, Dynamax ≥ 1.0.1.

:::tip[GPU / TPU acceleration]
Install the GPU build of JAX before installing glmhmmt for hardware-accelerated EM:
```bash
uv pip install "jax[cuda12]"
```
:::

## Working with marimo

Because `glmhmmt` is built on JAX, it pairs exceptionallly well with **[Marimo](https://marimo.io/)** — a reactive Python notebook environment. Unlike Jupyter, Marimo notebooks are pure Python scripts that execute reactively, meaning your state tracking and plots are always guaranteed to be consistent with your code.

To start an analysis notebook:
```bash
uv run marimo edit notebooks/glmhmmt_analysis.py
```

## Task-aware workflow

The codebase is task-aware rather than assuming a single dataset layout:

```python
from tasks import get_adapter

adapter = get_adapter("mcdr")  # or "two_afc"
plots = adapter.get_plots()
```

The adapter owns:
- data file selection
- subject/session filtering
- tensor construction
- state labels
- task-specific plots

## Prepare your data

Your trial data should be a `pandas.DataFrame` with one row per trial. Use `build_sequence_from_df` to convert it into the tensor format the model expects:

```python
from glmhmmt import build_sequence_from_df

y, X, U, names, extras = build_sequence_from_df(
    df_sub,
    tau=50.0,
)
```

## Fit the model

```python
import jax.numpy as jnp
from glmhmmt import SoftmaxGLMHMM

model = SoftmaxGLMHMM(
    num_states=3,
    num_classes=adapter.num_classes,
    emission_input_dim=X.shape[1],
    transition_input_dim=U.shape[1],
)

inputs_all = jnp.concatenate([X, U], axis=1)
fitted_params, lps = model.fit_em(
    params=params,
    props=props,
    emissions=y,
    inputs=inputs_all,
    num_iters=100,
)
```

For repository scripts, use the generic entry points instead:

```bash
uv run python scripts/fit_glm.py --task mcdr
uv run python scripts/fit_glmhmm.py --task mcdr --K 3
uv run python scripts/fit_glmhmmt.py --task two_afc --K 2
```

## Postprocess and visualise

```python
from glmhmmt import build_trial_df, build_emission_weights_df, build_views

views = build_views(fitted_params, df)
trial_df = build_trial_df(...)
weights_df = build_emission_weights_df(...)

fig, _ = plots.plot_categorical_performance_all(
    plots.prepare_predictions_df(trial_df),
    model_name="glmhmm_K3",
)
```

## Next steps

See the [framework guide](/docs/guide/framework) for the full repository flow and
[adding a task](/docs/guide/tasks) for the task adapter contract.
