---
title: Features
description: Helper functions for building model-ready input tensors from DataFrames.
---

The `features` module converts raw trial-level `pandas` DataFrames into the JAX tensor format expected by `SoftmaxGLMHMM`.

## Import

```python
from glmhmmt import build_sequence_from_df, build_sequence_from_df_2afc
```

---

## `build_sequence_from_df`

General-purpose builder for arbitrary feature matrices and multi-class choices.

```python
build_sequence_from_df(
    df: pd.DataFrame,
    choice_col: str,
    feature_cols: list[str],
    subject_col: str = "subject",
    session_col: str = "session",
) -> tuple[list[Array], list[Array], list[Array]]
```

**Arguments**

| Name | Type | Description |
|---|---|---|
| `df` | `DataFrame` | One row per trial. Must contain all listed columns. |
| `choice_col` | `str` | Column with integer-encoded choices (0-indexed). |
| `feature_cols` | `list[str]` | Columns to use as GLM covariates. |
| `subject_col` | `str` | Column identifying subjects. |
| `session_col` | `str` | Column identifying sessions within a subject. |

**Returns** a three-tuple `(inputs, choices, masks)`:

| Return | Shape | Description |
|---|---|---|
| `inputs` | `list[(T, F)]` | Feature tensors per subject. |
| `choices` | `list[(T,)]` | Flattened choice arrays per subject. |
| `masks` | `list[(T,)]` | Boolean boundary mask — `False` at session boundaries. |

**Example**

```python
inputs, choices, masks = build_sequence_from_df(
    df,
    choice_col="response",
    feature_cols=["contrast_L", "contrast_R", "prev_choice", "prev_reward"],
    subject_col="mouse_id",
    session_col="session_date",
)
```

---

## `build_sequence_from_df_2afc`

Convenience wrapper for standard **2-Alternative Forced Choice (2AFC)** paradigms. Automatically constructs the signed contrast covariate and previous-choice history feature.

```python
build_sequence_from_df_2afc(
    df: pd.DataFrame,
    choice_col: str = "choice",
    contrast_L_col: str = "contrast_left",
    contrast_R_col: str = "contrast_right",
    subject_col: str = "subject",
    session_col: str = "session",
    history_lags: int = 1,
) -> tuple[list[Array], list[Array], list[Array]]
```

**Arguments**

| Name | Type | Description |
|---|---|---|
| `contrast_L_col` / `contrast_R_col` | `str` | Left and right stimulus contrast columns. |
| `history_lags` | `int` | Number of previous-choice lags to include as features. |

All other arguments are the same as `build_sequence_from_df`.

**Example**

```python
inputs, choices, masks = build_sequence_from_df_2afc(
    df,
    history_lags=2,
)
# Produces feature matrix with columns: [signed_contrast, prev_choice_t-1, prev_choice_t-2]
```
