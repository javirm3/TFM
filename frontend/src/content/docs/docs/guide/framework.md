---
title: Framework
description: Repository structure, data flow, and the division between shared model code and task-owned logic.
---

## Overview

The repository is split into two layers:

- `glmhmmt`: the task-agnostic model package
- `tasks`: adapters and plots for each experimental task

The core package should never need to know whether the data comes from MCDR,
2AFC, or a future task. It only consumes tensors and returns fitted parameters,
posteriors, and diagnostic views.

In practice, the boundary is enforced through `TaskAdapter`. The adapter turns a
cleaned task dataframe into the `(y, X, U, names)` contract used by the shared
fit scripts and notebooks. That is the main design decision in the repository:
the package stays generic by pushing task semantics into adapters rather than
special-casing tasks inside the model code.

## Repository structure

```text
code/
├── glmhmmt/
│   └── src/glmhmmt/
│       ├── model.py
│       ├── features.py
│       ├── model_plots.py
│       ├── postprocess.py
│       └── views.py
├── tasks/
│   ├── __init__.py
│   ├── mcdr.py
│   ├── two_afc.py
│   └── plots/
│       ├── mcdr.py
│       └── two_afc.py
├── scripts/
├── notebooks/
└── paths.py
```

## Data flow

```text
raw data
  -> preprocessing notebook
  -> parquet dataset
  -> TaskAdapter.build_feature_df()
  -> TaskAdapter.load_subject()
  -> (y, X, U, names)
  -> fitting script / notebook
  -> fitted parameters + posteriors
  -> build_views / build_trial_df
  -> shared diagnostics + task-specific plots
```

## What belongs where

### `glmhmmt`

Put code here only if it is meaningful for any task:

- the model itself
- generic feature builders
- posterior and weight postprocessing
- fit result views
- shared diagnostics such as emission weights, posterior probabilities, state occupancy, and session trajectories

### `tasks`

Put code here if it depends on task semantics:

- file names and filtering rules
- column mappings
- feature-dataframe and design-matrix construction
- state naming rules
- psychometrics
- performance plots by stimulus or condition
- task-specific diagnostics

## Plot architecture

Plots follow the same split as the model code:

- shared diagnostics live in `glmhmmt.model_plots` and `glmhmmt.plots_common`
- task-owned plots live in `tasks.plots.<task>`

The shared diagnostics cover views that should make sense for any task once the
adapter contract is satisfied, such as emission weights, posterior
probabilities, state occupancy, and session trajectories.

Task-owned plot modules can also define custom plots for the task itself:
psychometrics, performance by condition, or any diagnostic that depends on the
task's own stimulus coding or behavioural interpretation.

This is why notebooks ask the active adapter for `adapter.get_plots()` instead
of importing a single global plotting module.

## Generic analysis pattern

```python
from tasks import get_adapter

adapter = get_adapter("mcdr")
plots = adapter.get_plots()

df = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
df = adapter.subject_filter(df)
y, X, U, names = adapter.load_subject(df_sub, tau=50.0)
```

This keeps notebooks and scripts generic while letting each task expose its own
plotting API.

## Extending the package

Adding a new task should usually mean adding task-owned modules rather than
editing `glmhmmt`:

1. preprocess the raw data into a cleaned parquet dataset
2. implement a `TaskAdapter` in `tasks/<task>.py`
3. implement a task plot module in `tasks/plots/<task>.py`
4. register the adapter and use the existing scripts and notebooks with `--task`

If a new task requires editing `glmhmmt.model.py`, the task boundary is
probably wrong.

This repository also includes an optional helper skill,
`$glmhmmt-task-adapter`, for creating or updating the adapter, its plot module,
and the supporting docs in a way that stays aligned with this architecture. The
skill is useful when you are adding tasks repeatedly, but the package design
does not depend on it.

## Running fits

Use the generic scripts:

```bash
uv run python scripts/fit_glm.py --task mcdr
uv run python scripts/fit_glmhmm.py --task mcdr --K 3
uv run python scripts/fit_glmhmmt.py --task two_afc --K 2
```

Use marimo notebooks for exploration:

```bash
uv run marimo edit notebooks/model_comparison.py
uv run marimo edit notebooks/glmhmm_analysis.py
uv run marimo edit notebooks/glmhmmt_analysis.py
```

## Design rule

The important boundary is:

- shared model code lives in `glmhmmt`
- task semantics and design matrices live behind `TaskAdapter`
- task-specific plots live in `tasks.plots.*`

That boundary is what lets you add a new task without rewriting the package.
