---
title: Adding a Task
description: Add a new behavioural task by implementing one adapter, one task-owned plotting module, and optionally using the repo-hosted helper skill.
---

## Goal

A new task should plug into the existing fit scripts and notebooks without
editing the shared model code. In practice, that means:

1. add a `TaskAdapter` subclass
2. add a task-owned plotting module
3. drop the module into the repo task package
4. keep task semantics out of `glmhmmt`

The core package should continue to consume the same `(y, X, U, names)`
contract regardless of whether the data comes from MCDR, 2AFC, or a new task.

## Adapter contract

Each adapter should define:

- `task_key`
- `task_label`
- `num_classes`
- `data_file`
- `sort_col`
- `session_col`
- `subject_filter(df)`
- `build_feature_df(df_sub, tau)`
- `load_subject(df_sub, tau, emission_cols, transition_cols)`
- `build_design_matrices(feature_df, emission_cols, transition_cols)`
- `default_emission_cols(df=None)`
- `default_transition_cols()`
- `behavioral_cols`
- `get_correct_class(df)`
- `label_states(...)`
- adapter-level state-assignment scoring (`_SCORING_OPTIONS` and `scoring_key`)
- `get_plots()`

The adapter owns task-specific filtering, column mappings, feature construction,
and any interpretation rules that depend on the experiment. If the logic would
not make sense for every task, it probably belongs here rather than in
`glmhmmt`.

## Minimal example

```python
from glmhmmt.tasks import TaskAdapter


class MyTaskAdapter(TaskAdapter):
    task_key = "my_task"
    task_label = "My Task"
    num_classes = 2
    data_file = "my_task.parquet"
    sort_col = ["session", "trial"]
    session_col = "session"

    # State-assignment scoring used by the generic analysis notebooks.
    _SCORING_OPTIONS: dict = {
        "stim_vals (-w)": [("stim_vals", "neg")],
        "stim_vals (|w|)": [("stim_vals", "abs")],
        "at_choice (|w|)": [("at_choice", "abs")],
        "wsls (|w|)": [("wsls", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "stim_vals (-w)"

    def subject_filter(self, df):
        return df

    def build_feature_df(self, df_sub, tau=50.0):
        ...

    def load_subject(self, df_sub, tau=50.0, emission_cols=None, transition_cols=None):
        ...

    def build_design_matrices(self, feature_df, emission_cols=None, transition_cols=None):
        ...

    def default_emission_cols(self, df=None):
        return [...]

    def default_transition_cols(self):
        return [...]

    @property
    def behavioral_cols(self):
        return {...}

    def get_correct_class(self, df):
        ...

    def label_states(self, arrays_store, names, K, subjects):
        ...

    def get_plots(self):
        import tasks.plots.my_task as plots
        return plots
```

Adapt the feature names or sign modes to the task, but keep
`_SCORING_OPTIONS` and `scoring_key` on the adapter. The generic analysis
notebooks use them to rank and label states.

## Plot boundary

Create `tasks/plots/my_task.py`. It should expose two things:

- shared diagnostics reused from `glmhmmt.model_plots`
- task-specific functions implemented locally

Common diagnostics are the plots that should work for any task once the adapter
contract is satisfied, for example:

- `plot_emission_weights`
- `plot_transition_matrix`
- `plot_posterior_probs`
- `plot_state_accuracy`
- `plot_state_occupancy`
- `plot_session_trajectories`
- `plot_session_deepdive`

Task-specific functions depend on the semantics of that experiment. Typical
examples are:

- `prepare_predictions_df`
- `plot_categorical_performance_all`
- `plot_categorical_performance_by_state`
- `plot_psychometric`
- `plot_task_diagnostics`

The point is not to force every task into the same plot family. Reuse the
shared diagnostics where they fit, and define custom plots where the task needs
its own axes, groupings, or behavioural interpretation.

## Registration

No explicit registration file is needed for repo-local tasks. `glmhmmt.tasks`
auto-imports every non-private `*.py` module inside `code/tasks/`, and
each module registers itself via `@_register([...])`.

## Workflow after adding the task

1. preprocess raw data into a parquet dataset
2. implement the task-owned feature dataframe and adapter
3. implement the task plot module
4. save the adapter module in `code/tasks/`
5. run the generic CLI commands with `--task my_task`
6. open the generic analysis notebooks and select the new task

## Companion skill

This repository ships a project-specific companion skill:

- `$glmhmmt-task-adapter`

Use it when you want Codex to scaffold or update the adapter, its plot module,
and the supporting docs in a way that matches this repository's architecture.
You do not need the skill to understand or use the package. The skill is there
to keep repeated task additions consistent.

If you want the installable version, the repo-hosted path is:

- [https://github.com/javirm3/TFM/tree/main/.agents/skills/glmhmmt-task-adapter](https://github.com/javirm3/TFM/tree/main/.agents/skills/glmhmmt-task-adapter)

After installing it into Codex, ask Codex to use `$glmhmmt-task-adapter` when
adding a new task.

This repo-hosted skill is the public version. Keep it stable and generic. Any
personal feedback loop you use while developing new adapters should stay in
your local Codex skills directory, not in the published installable skill.

## Design check

If adding the new task requires editing `glmhmmt.model.py`, the task boundary is
probably wrong. First check whether the logic actually belongs in the adapter or
the task plot module instead.
