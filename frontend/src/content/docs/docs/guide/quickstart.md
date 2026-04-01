---
title: Quickstart
description: Install glmhmmt and fit your first GLM-HMM in minutes.
---

## Installation

Clone the repository and install the thesis workspace with `uv`:

```bash
git clone https://github.com/javirm3/TFM
cd TFM/code
uv sync
```

If you also want the marimo notebooks and notebook-only dependencies:

```bash
cd TFM/code
uv sync --extra notebooks
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
from glmhmmt.tasks import get_adapter

adapter = get_adapter("mcdr")  # or "two_afc"
plots = adapter.get_plots()
```

The adapter owns:
- data file selection
- subject/session filtering
- tensor construction
- state labels
- task-specific plots

Repo-local task adapters live in `TFM/code/tasks`, and `glmhmmt.tasks` also
supports task folders discovered from `GLMHMMT_TASK_PATHS` or a project's own
`./tasks` directory.

If you are starting support for a new behavioural task, the repository also
ships an optional Codex skill, `$glmhmmt-task-adapter`, to speed up the first
pass. It helps scaffold or update the task adapter, the task-owned plot module,
and the related docs while keeping task semantics out of `glmhmmt`.

## Prepare your data

Load the cleaned dataset through the active adapter and let the adapter build
the design matrices:

```python
import polars as pl
from glmhmmt.runtime import get_runtime_paths

paths = get_runtime_paths()

df = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
df = adapter.subject_filter(df)

subject = df["subject"].unique().sort()[0]
df_sub = df.filter(pl.col("subject") == subject).sort(adapter.sort_col)

y, X, U, names = adapter.load_subject(df_sub, tau=50.0)
```

This keeps the notebook or script generic while letting each task decide how to
filter rows, derive regressors, and map behavioural columns into the shared
`(y, X, U, names)` contract.

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

For the packaged CLI, use the console entry points instead:

```bash
uv run glmhmmt-fit-glm --task MCDR
uv run glmhmmt-fit-glmhmm --task MCDR --K 3
uv run glmhmmt-fit-glmhmmt --task 2AFC --K 2
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
[adding a task](/docs/guide/tasks) for the task adapter contract, the plot
boundary, and the install path for `$glmhmmt-task-adapter`. The generic
adapter/factory reference lives in [tasks API](/docs/api/tasks).
