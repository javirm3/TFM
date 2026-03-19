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
- state naming rules
- psychometrics
- performance plots by stimulus or condition
- task-specific diagnostics

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
- task semantics live behind `TaskAdapter`
- task-specific plots live in `tasks.plots.*`

That boundary is what lets you add a new task without rewriting the package.
