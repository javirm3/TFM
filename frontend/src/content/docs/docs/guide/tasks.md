---
title: Adding a Task
description: Add a new behavioural task by implementing one adapter and one task-owned plotting module.
---

## Goal

A new task should be added without editing the shared model code. In practice,
that means:

1. add a `TaskAdapter` subclass
2. add a task-owned plotting module
3. register the adapter

## Adapter contract

Each adapter should define:

- `num_classes`
- `data_file`
- `sort_col`
- `session_col`
- `subject_filter(df)`
- `build_feature_df(df_sub, tau)`
- `load_subject(df_sub, tau, emission_cols, transition_cols)`
- `default_emission_cols()`
- `default_transition_cols()`
- `behavioral_cols`
- `label_states(...)`
- `get_plots()`

## Minimal example

```python
from tasks import TaskAdapter, _register


@_register(["my_task"])
class MyTaskAdapter(TaskAdapter):
    num_classes = 2
    data_file = "my_task.parquet"
    sort_col = ["session", "trial"]
    session_col = "session"

    def subject_filter(self, df):
        return df

    def build_feature_df(self, df_sub, tau=50.0):
        ...

    def load_subject(self, df_sub, tau=50.0, emission_cols=None, transition_cols=None):
        ...

    def default_emission_cols(self):
        return [...]

    def default_transition_cols(self):
        return [...]

    @property
    def behavioral_cols(self):
        return {...}

    def label_states(self, arrays_store, names, K, subjects):
        ...

    def get_plots(self):
        import tasks.plots.my_task as plots
        return plots
```

## Plot module

Create `tasks/plots/my_task.py`. It should expose:

- shared diagnostics by importing from `glmhmmt.model_plots`
- task-specific functions implemented locally

Typical task-specific functions are:

- `prepare_predictions_df`
- `plot_categorical_performance_all`
- `plot_categorical_performance_by_state`
- `plot_psychometric`
- `plot_task_diagnostics`

## Registration

Make sure the adapter file is imported at the bottom of `tasks/__init__.py` so it
self-registers via the decorator.

## Workflow after adding the task

1. preprocess raw data into a parquet dataset
2. implement the task-owned feature dataframe and adapter
3. implement the task plot module
4. run the generic fit scripts with `--task my_task`
5. open the generic analysis notebooks and select the new task

## Install the companion skill

The skill is versioned in this repository and can be installed from its GitHub
path:

- [https://github.com/javirm3/TFM/tree/main/.agents/skills/glmhmmt-task-adapter](https://github.com/javirm3/TFM/tree/main/.agents/skills/glmhmmt-task-adapter)

If you are using Codex with the skill installer, install from that path and
restart Codex so the new skill is picked up. After that, ask Codex to use
`$glmhmmt-task-adapter` when adding a new task.

This repo-hosted skill is the public version. Keep it stable and generic. Any
personal feedback loop you use while developing new adapters should stay in
your local Codex skills directory, not in the published installable skill.

## Design check

If adding the new task requires editing `glmhmmt.model.py`, the task boundary is
probably wrong. First check whether the logic actually belongs in the adapter or
the task plot module instead.
