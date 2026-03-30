---
title: Common Plots
description: Task-agnostic plotting helpers in `glmhmmt.plots_common`.
---

The `glmhmmt.plots_common` module contains the shared plotting primitives used by task-owned plotting modules. These helpers operate on two core inputs:

- `views`: a `{subject: SubjectFitView}` mapping
- `trial_df`: a trial-level pandas or Polars table with canonical behavioural columns

## Import

```python
from glmhmmt.plots_common import (
    plot_session_deepdive,
    plot_session_trajectories,
    plot_state_accuracy,
    plot_state_dwell_times_by_subject,
    plot_state_dwell_times_summary,
    plot_state_dwell_times,
    plot_state_occupancy,
)
```

---

## `plot_state_accuracy`

```python
plot_state_accuracy(
    views: dict,
    trial_df,
    *,
    thresh: float = 0.5,
    performance_candidates: Sequence[str] = ("correct_bool", "performance"),
    stim_candidates: Sequence[str] = ("stimd_n", "stimulus", "ILD"),
    chance_level: float | None = None,
    stim_label: str = "nonzero stimulus",
) -> tuple[plt.Figure, pd.DataFrame]
```

Computes per-state accuracy by thresholding posterior state probabilities and returns:

- a boxplot-style summary figure
- a `pandas.DataFrame` table with mean accuracy and total trials per state

**Example**

```python
fig, acc_table = plot_state_accuracy(views, trial_df, thresh=0.7)
```

---

## `plot_session_trajectories`

```python
plot_session_trajectories(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
) -> plt.Figure
```

Plots the average posterior state trajectory across sessions for each subject, aligned by trial index within session.

**Example**

```python
fig = plot_session_trajectories(views, trial_df, session_col="session")
```

---

## `plot_state_occupancy`

```python
plot_state_occupancy(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | None = None,
    switch_posterior_threshold: float | None = None,
) -> plt.Figure
```

Builds occupancy diagnostics across subjects and sessions, including:

- state occupancy by subject
- occupancy by session
- state-switch summaries

`switch_posterior_threshold` can be used to ignore weak posterior fluctuations when counting switches.

**Example**

```python
fig = plot_state_occupancy(
    views,
    trial_df,
    session_col="session",
    sort_col="trial_idx",
    switch_posterior_threshold=0.8,
)
```

---

## `plot_state_dwell_times_by_subject`

```python
plot_state_dwell_times_by_subject(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    max_dwell: int | None = 90,
    ci_level: float = 0.68,
) -> plt.Figure
```

Builds Ashwood-style dwell-time panels for each selected subject and state:

- solid line with markers: geometric dwell distribution from the fitted self-transition probability `A_kk`
- dashed line with markers: empirical dwell distribution from MAP posterior state assignments
- vertical error bars: Wilson confidence interval around the empirical probabilities

By default the display horizon is fixed to `90` trials, and both curves are shown in 10-trial bins up to that limit for Ashwood-style comparison.
Runs are split at session boundaries, so dwell episodes never span multiple sessions.

**Example**

```python
fig = plot_state_dwell_times_by_subject(
    views,
    trial_df,
    session_col="session",
    sort_col="trial_idx",
    ci_level=0.68,
)
```

---

## `plot_state_dwell_times_summary`

```python
plot_state_dwell_times_summary(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    max_dwell: int | None = 90,
    ci_level: float = 0.68,
) -> plt.Figure
```

Builds the subject-aggregated dwell-time summary using the same binning and 90-trial dwell horizon as the by-subject plot.
The summary pools empirical dwell lengths across subjects inside the function and uses the same y-axis limit as the by-subject panels for direct comparison.

**Example**

```python
fig = plot_state_dwell_times_summary(
    views,
    trial_df,
    session_col="session",
    sort_col="trial_idx",
    ci_level=0.68,
)
```

---

## `plot_state_dwell_times`

```python
plot_state_dwell_times(
    views: dict,
    trial_df,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    max_dwell: int | None = 90,
    ci_level: float = 0.68,
) -> plt.Figure
```

Alias for `plot_state_dwell_times_by_subject(...)`.

---

## `plot_session_deepdive`

```python
plot_session_deepdive(
    views: dict,
    trial_df,
    subj: str,
    sess,
    *,
    session_col: str = "session",
    sort_col: str | Sequence[str] | None = None,
    switch_posterior_threshold: float | None = None,
    performance_candidates: Sequence[str] = ("correct_bool", "performance"),
    stim_candidates: Sequence[str] = ("stimd_n", "ILD", "stimulus"),
    response_candidates: Sequence[str] = ("response", "Choice"),
    trace_x_candidates: Sequence[str] = ("A_R", "A_L", "A_C"),
    trace_u_candidates: Sequence[str] = ("A_plus", "A_minus"),
    choice_colors: dict[int, str] | None = None,
    choice_labels: dict[int, str] | None = None,
) -> plt.Figure
```

Creates a single-session diagnostic view combining:

- stacked posterior state probabilities
- trial-by-trial responses and rolling accuracy
- optional emission or transition regressor traces when available
- the session's state-change count in the title
- red markers at detected state-change trials

If `switch_posterior_threshold` is set, the title count only includes confident MAP changes where posterior is at least that threshold.
The red marker is drawn at the later detected trial and sits on the `P(engaged)` trace.

**Example**

```python
fig = plot_session_deepdive(views, trial_df, subj="A12", sess=3)
```

---

## Data Contract

These shared helpers assume:

- `views[subj].smoothed_probs` is aligned with the subject rows in `trial_df`
- `trial_df` contains a `subject` column
- session-aware plots can resolve a valid session column

Task-owned plot modules wrap these helpers and provide task-specific default column names, so notebook code will usually call the task module rather than `glmhmmt.plots_common` directly.
