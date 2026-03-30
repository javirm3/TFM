---
name: glmhmmt-task-adapter
description: Build or update a glmhmmt task adapter, its task-owned plots, and the supporting docs. Use when adding a new behavioural task or porting a dataset into the TaskAdapter contract.
---

# glmhmmt task adapter

Use this skill when adding a task under `code/tasks/` or updating an existing
adapter. The goal is to keep `glmhmmt` task-agnostic and push task semantics
into the adapter and task-owned plot module.

## Start here

Read these files before coding:

- `frontend/src/content/docs/docs/guide/tasks.md`
- `frontend/src/content/docs/docs/guide/framework.md`
- `code/tasks/__init__.py`
- one close adapter pair from `code/tasks/*.py` and `code/tasks/plots/*.py`

## Workflow

1. Clarify the task contract from the cleaned dataframe.
   You need subject id, session id, trial order, observed choice, correct class
   or stimulus coding, performance, and candidate emission and transition
   regressors.
   Ask explicitly which evidence / regressor axes are continuous and which are
   categorical, because that determines whether psychometric summaries should
   be binned or shown at native levels.
2. Keep preprocessing separate from model code.
   Heavy ETL should end in a parquet file. The adapter should consume the
   cleaned dataframe, not recreate the full preprocessing pipeline.
3. Implement the adapter in `code/tasks/<task>.py`.
   Define `task_key`, `task_label`, `num_classes`, `data_file`, `sort_col`,
   `session_col`, `subject_filter`, `build_feature_df`, `load_subject`,
   defaults, `behavioral_cols`, `get_correct_class`, `label_states`,
   adapter-level state-assignment scoring (`_SCORING_OPTIONS` and
   `scoring_key`), `cv_balance_labels` when CV balancing is needed, and
   `get_plots`.
4. Implement task-owned plots in `code/tasks/plots/<task>.py`.
   Put psychometrics, task diagnostics, and performance-by-condition plots
   here. Reuse `glmhmmt.model_plots` for shared diagnostics, but do not
   re-export another task's plot module as the new task module.
5. Register the adapter in `code/tasks/__init__.py`.
   Add the decorator key(s) and the import at the bottom so `get_adapter()`
   can resolve it.
6. Verify the generic entry points.
   Use the existing fit scripts and generic notebooks with `--task`.
7. Update docs.
   Keep the public task guide aligned with the adapter contract and install
   path.

For binary tasks, keep feature construction inside the adapter boundary.
Reuse existing parser code only when the task dataframe genuinely matches that
parser's contract; otherwise implement a task-owned `build_feature_df(...)`.

## State-assignment scoring

Keep state-assignment scoring in the adapter. Define `_SCORING_OPTIONS` and
`scoring_key` there so the generic analysis notebooks can rank and label
states. For binary stimulus-following tasks, start from:

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

Adapt the feature names or sign modes to the task, but do not leave the
adapter without this scoring config.

## Design rules

- Do not move task semantics into `glmhmmt` unless the logic is genuinely
  task-agnostic.
- Prefer adapting the dataframe to the adapter contract over widening the
  shared API for one task.
- Only add canonical behavioral / sort columns needed by the adapter
  contract. Do not rename one task's raw columns into another task's private
  schema just to make copied code run.
- Never import or call another task module from inside a task module. If a new
  task needs similar logic, duplicate it into the new task file and adapt it
  there.
- Registry-driven task selectors are preferred over hard-coded task lists in
  widgets and notebooks.
- For tasks with history regressors, decide separately:
  the modeled response column, the choice column used for traces, whether
  action traces are included, and the decay constants used to build them.
- Use the selected modeled response consistently. If the task should model
  `last_choice`, do not leave `first_choice` wired into the adapter or plots.
- If the adapter exposes `tau`, the history features should respect that
  parameter. Do not silently hardcode fixed decay values inside one task.
- Task-owned plots should read task-specific axes and behavioral columns from
  the adapter or task-local constants, rather than assuming Alexis 2AFC names
  like `ILD`, `Choice`, or `Hit`.
- Do not inherit another task's emission-weight sign convention blindly.
  Verify whether positive stimulus weight should stay positive in the new
  task's native response coding before copying 2AFC sign flips into state
  labelling or weight plots.
- If you add balanced session-holdout CV for a task, expose
  `cv_balance_labels(feature_df)` from the adapter and base it on the task's
  own signed evidence / balancing variable.
- Decide plot aggregation from the regressor type:
  continuous axes should usually be binned before pooling, while categorical
  axes should usually stay at their native levels.
- Keep the task's categorical / psychometric plot family internally
  consistent. If the overall psychometric is binned for a continuous axis, the
  by-state version should follow the same binning pattern unless there is a
  task-specific reason not to.
- If a blocker is specific to one dataset, document it near the adapter or in
  task docs, not in the public skill.

## Validation

At minimum:

- `uv run python -c "from tasks import get_adapter; print(type(get_adapter('<task>')).__name__)"`
- run the relevant fit script for the new task
- open one generic notebook and confirm the task loads
