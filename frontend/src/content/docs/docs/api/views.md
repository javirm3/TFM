---
title: Views
description: Result container objects and batch view builder for per-subject fits.
---

The `views` module provides lightweight **result container objects** (`SubjectFitView`) that bundle a subject's fitted parameters with their raw data and expose convenience plotting methods.

## Import

```python
from glmhmmt import SubjectFitView, build_views
```

---

## `SubjectFitView`

A container for a single subject's fit result.

```python
@dataclass
class SubjectFitView:
    subject_id: Any
    params: Params
    inputs: Array       # (T, F)
    choices: Array      # (T,)
    masks: Array        # (T,)
    model: SoftmaxGLMHMM
```

### Methods

#### `.plot_state_occupancy(ax=None)`

Bar chart of the fraction of trials spent in each state (based on Viterbi decoding).

```python
view.plot_state_occupancy()
```

#### `.plot_emission_weights(feature_names=None, ax=None)`

Heatmap of GLM emission weights `(num_states × num_features)`.

```python
view.plot_emission_weights(feature_names=["contrast", "prev_choice", "prev_reward"])
```

#### `.plot_state_sequence(ax=None)`

Raster plot of the Viterbi state sequence across trials, colour-coded by state.

```python
view.plot_state_sequence()
```

#### `.plot_posterior(ax=None)`

Ribbon plot of smoothed posterior probabilities `p(z_t | y_{1:T})` across trials.

```python
view.plot_posterior()
```

#### `.transition_matrix() -> Array`

Returns the `(K, K)` transition matrix from the fitted parameters.

---

## `build_views`

Batch-creates a list of `SubjectFitView` from the outputs of `model.fit_per_subject`.

```python
build_views(
    fitted_params: list[Params],
    model: SoftmaxGLMHMM,
    inputs: list[Array],
    choices: list[Array],
    masks: list[Array],
    subject_ids: list[Any] | None = None,
) -> list[SubjectFitView]
```

**Example**

```python
views = build_views(fitted_params, model, inputs, choices, masks)

# Plot all subjects
import matplotlib.pyplot as plt
fig, axes = plt.subplots(len(views), 2, figsize=(12, 4 * len(views)))
for i, v in enumerate(views):
    v.plot_state_occupancy(ax=axes[i, 0])
    v.plot_emission_weights(ax=axes[i, 1])
plt.tight_layout()
plt.savefig("all_subjects.pdf")
```

---

## Constants

| Name | Value | Description |
|---|---|---|
| `_LABEL_RANK` | `dict` | Maps state index → human-readable label (e.g. `"Engaged"`, `"Lapse"`). |
| `_STATE_HEX` | `list[str]` | Default hex colour palette for states in plots. |
