---
title: Postprocessing
description: Utilities for extracting tidy DataFrames from fitted model parameters.
---

The `postprocess` module turns raw parameter objects returned by `SoftmaxGLMHMM.fit_per_subject` into tidy `pandas` DataFrames ready for downstream analysis and plotting.

## Import

```python
from glmhmmt import build_trial_df, build_emission_weights_df, build_posterior_df
```

---

## `build_trial_df`

Constructs a **trial-level DataFrame** that merges the original data with Viterbi-decoded state assignments and smoothed posterior probabilities.

```python
build_trial_df(
    fitted_params: list[Params],
    df: pd.DataFrame,
    model: SoftmaxGLMHMM,
    inputs: list[Array],
    choices: list[Array],
    masks: list[Array],
    subject_col: str = "subject",
) -> pd.DataFrame
```

**Returns** the original `df` augmented with:

| New column | Description |
|---|---|
| `state_viterbi` | Most likely state sequence (Viterbi). |
| `state_posterior_k` | Posterior probability of state *k* for each *k*. |

**Example**

```python
trial_df = build_trial_df(fitted_params, df, model, inputs, choices, masks)
# Group by subject and state to inspect dwell times
trial_df.groupby(["subject", "state_viterbi"]).size()
```

---

## `build_emission_weights_df`

Returns a **long-format DataFrame** of GLM emission weights (one row per state × feature combination), useful for heatmap visualisations.

```python
build_emission_weights_df(
    fitted_params: list[Params],
    feature_cols: list[str],
    subject_col: str = "subject",
) -> pd.DataFrame
```

| Column | Description |
|---|---|
| `subject` | Subject identifier. |
| `state` | Latent state index. |
| `feature` | Feature name (from `feature_cols`). |
| `weight` | GLM regression weight. |

**Example**

```python
w_df = build_emission_weights_df(fitted_params, feature_cols=["contrast", "prev_choice"])
import seaborn as sns
sns.heatmap(w_df.pivot(index="state", columns="feature", values="weight"))
```

---

## `build_posterior_df`

Returns the full smoothed posteriors `p(z_t | y_{1:T})` for every trial and subject as a long DataFrame.

```python
build_posterior_df(
    fitted_params: list[Params],
    model: SoftmaxGLMHMM,
    inputs: list[Array],
    choices: list[Array],
    masks: list[Array],
    subject_col: str = "subject",
) -> pd.DataFrame
```

| Column | Description |
|---|---|
| `subject` | Subject identifier. |
| `trial` | Trial index within subject. |
| `state` | Latent state index. |
| `posterior` | Smoothed posterior probability. |
