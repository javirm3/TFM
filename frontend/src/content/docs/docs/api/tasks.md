---
title: Tasks API
description: Generic adapter and registry API exposed by the `glmhmmt.tasks` module.
---

The `glmhmmt.tasks` module is the generic extension surface between the shared
`glmhmmt` package and task-specific code. In normal use, notebooks and CLI
commands work with a `TaskAdapter` instance returned by `get_adapter()`.

## Import

```python
from glmhmmt.tasks import TaskAdapter, get_adapter, get_task_options
```

## `get_adapter`

Resolve the active task and return an instantiated adapter.

```python
get_adapter(task: str) -> TaskAdapter
```

The lookup is case-insensitive and normalises hyphens to underscores.

**Example**

```python
from glmhmmt.tasks import get_adapter

adapter = get_adapter("mcdr")
plots = adapter.get_plots()
```

If the task key is unknown, `get_adapter()` raises `ValueError` and reports the
registered keys.

## `get_task_options`

Return the available tasks in a UI-friendly format.

```python
get_task_options() -> list[dict[str, str]]
```

Each item has the form:

```python
{"value": "<task_key>", "label": "<task_label>"}
```

**Example**

```python
options = get_task_options()
# [{"value": "MCDR", "label": "MCDR"}, ...]
```

## `TaskAdapter`

`TaskAdapter` is the generic contract that every task implements.

### Required attributes

| Attribute | Description |
|---|---|
| `task_key` | Canonical task key used in selectors and CLI arguments. |
| `task_label` | Human-readable task name. |
| `num_classes` | Number of observable choice classes. |
| `data_file` | Dataset filename under the configured data path. |
| `sort_col` | Trial ordering column or columns. |
| `session_col` | Session identifier column. |

### Loading and design-matrix methods

| Method | Role |
|---|---|
| `subject_filter(df)` | Apply task-specific filtering to the full dataset. |
| `build_feature_df(df_sub, tau=50.0)` | Build the task-owned dataframe containing raw behavioural columns and derived regressors. |
| `load_subject(df_sub, tau=50.0, emission_cols=None, transition_cols=None)` | Return `(y, X, U, names)` for one subject slice. |
| `build_design_matrices(feature_df, emission_cols=None, transition_cols=None)` | Build `(y, X, U, names)` from an already prepared feature dataframe. |
| `default_emission_cols(df=None)` | Default ordered emission regressor names, optionally including dynamic task-owned columns inferred from `df`. |
| `default_transition_cols()` | Default ordered transition regressor names. |
| `available_emission_cols(df=None)` | Selectable emission regressors. Defaults to the emission defaults and can include dynamic task-owned columns from `df`. |
| `available_transition_cols()` | Selectable transition regressors. Defaults to the transition defaults. |
| `resolve_design_names(...)` | Resolve `X_cols` and `U_cols` without rebuilding arrays. |

### Plot and metadata methods

| Method or property | Role |
|---|---|
| `get_plots()` | Return the task-owned plot module. |
| `behavioral_cols` | Mapping from canonical behavioural names to dataset column names. |
| `choice_labels` | Human-readable choice labels aligned with the class indices. |
| `probability_columns` | Trial-level prediction column names aligned with `choice_labels`. |
| `get_correct_class(df)` | Return the correct-class index per trial. |
| `label_states(arrays_store, names, K, subjects)` | Return task-specific state labels and state ordering. |
| `cv_balance_labels(feature_df)` | Optional per-trial labels used for balanced cross-validation. |

## State-assignment scoring convention

Keep state-assignment scoring on the adapter via `_SCORING_OPTIONS` and
`scoring_key`, rather than hardcoding it in notebooks. For the current binary
2AFC-style default:

```python
_SCORING_OPTIONS: dict = {
    "stim_vals (-w)": [("stim_vals", "neg")],
    "stim_vals (|w|)": [("stim_vals", "abs")],
    "at_choice (|w|)": [("at_choice", "abs")],
    "wsls (|w|)": [("wsls", "abs")],
    "bias (|w|)": [("bias", "abs")],
}
scoring_key: str = "stim_vals (-w)"
```

Other tasks can use task-appropriate scoring pairs, but the adapter should own
the scoring rule and its default `scoring_key`.

### Typical use

```python
import polars as pl
from glmhmmt.runtime import get_runtime_paths
from glmhmmt.tasks import get_adapter

paths = get_runtime_paths()
adapter = get_adapter("mcdr")
df = pl.read_parquet(paths.data_dir / adapter.data_file)
df = adapter.subject_filter(df)

subject = df["subject"].unique().sort()[0]
df_sub = df.filter(pl.col("subject") == subject).sort(adapter.sort_col)

y, X, U, names = adapter.load_subject(df_sub, tau=50.0)
plots = adapter.get_plots()
```

The important point is that CLI commands and notebooks only depend on the generic
adapter surface. Task-specific semantics stay behind that boundary.

## Implementing a new task

For the full adapter contract and the task-owned plot boundary, see
[adding a task](/docs/guide/tasks).
