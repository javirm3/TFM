---
title: Quickstart
description: Install glmhmmt and fit your first GLM-HMM in minutes.
---

## Installation

We highly recommend using [uv](https://github.com/astral-sh/uv) to manage your Python environments and dependencies. It is incredibly fast (often 10-100x faster than pip) and handles JAX and dynamax installations flawlessly.

Until `glmhmmt` is released on PyPI, install it directly from the repository in editable mode:

```bash
git clone https://github.com/javirm3/TFM
cd TFM/code/glmhmmt
uv pip install -e .
```

*Note: `glmhmmt` will be published to PyPI shortly, after which you'll be able to install it directly with `uv pip install glmhmmt`.*

**Requirements:** Python ≥ 3.11, JAX ≥ 0.9, Dynamax ≥ 1.0.1.

:::tip[GPU / TPU acceleration]
Install the GPU build of JAX before installing glmhmmt for hardware-accelerated EM:
```bash
uv pip install "jax[cuda12]"
```
:::

## Working with Marimo

Because `glmhmmt` is built on JAX, it pairs exceptionallly well with **[Marimo](https://marimo.io/)** — a reactive Python notebook environment. Unlike Jupyter, Marimo notebooks are pure Python scripts that execute reactively, meaning your state tracking and plots are always guaranteed to be consistent with your code.

To start a marimo session:
```bash
uv run marimo edit notebook.py
```

## Prepare your data

Your trial data should be a `pandas.DataFrame` with one row per trial. Use `build_sequence_from_df` to convert it into the tensor format the model expects:

```python
from glmhmmt import build_sequence_from_df

# df must contain at minimum:
#   - a column for choices (e.g. "choice")    — integer encoded
#   - feature columns (stimulus, history, …)
#   - a "subject" and "session" column

inputs, choices, masks = build_sequence_from_df(
    df,
    choice_col="choice",
    feature_cols=["contrast_left", "contrast_right", "prev_choice"],
    subject_col="subject",
    session_col="session",
)
```

## Fit the model

```python
from glmhmmt import SoftmaxGLMHMM

model = SoftmaxGLMHMM(
    num_states=3,       # number of latent strategies
    num_obs=2,           # number of choice options (e.g. Left / Right)
    num_features=3,      # must match len(feature_cols) above
)

# Per-subject EM: returns one fitted params object per subject
fitted_params = model.fit_per_subject(
    inputs,
    choices,
    masks,
    num_iters=100,
)
```

## Postprocess and visualise

```python
from glmhmmt import build_trial_df, build_emission_weights_df, build_views

# Reconstruct a tidy trial-level DataFrame with state assignments
trial_df = build_trial_df(fitted_params, df, subject_col="subject")

# Emission weights per state
weights_df = build_emission_weights_df(fitted_params, feature_cols=["contrast_left", "contrast_right", "prev_choice"])

# Build view objects for plotting
views = build_views(fitted_params, df)

# Plot diagnostics for one subject
views[0].plot_state_occupancy()
views[0].plot_emission_weights()
```

## Next steps

See the [API Reference](/docs/api/model) for the full signature of `SoftmaxGLMHMM` and all helper functions.
