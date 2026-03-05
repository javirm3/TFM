# HOWTO — GLM-HMM Behavioural Modelling Framework

A guide for using this repository and porting a new experimental task.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Data Flow](#data-flow)
5. [Adding a New Task](#adding-a-new-task)
6. [Running Fits](#running-fits)
7. [Exploring Results](#exploring-results)
8. [Design Decisions](#design-decisions)

---

## Overview

This repository implements GLM-HMM and GLM-HMM-t models for analysing trial-by-trial
behavioural data. The core model (`glmhmmt` package) is task-agnostic — it takes a
design matrix and choices and returns fitted parameters. All task-specific knowledge
(data paths, column names, filtering criteria) is isolated in two files per task.

**Key principle:** the `glmhmmt` package never knows about experimental tasks.
Tasks never know about the model internals. Data flows in one direction:

```
raw data  →  preprocess_{task}.py  →  data/{task}.parquet
                                              ↓
                           TaskAdapter.load_subject()
                                              ↓
                              fit_model.py  →  results/fits/{task}/
                                              ↓
                           notebooks/analysis.py  →  figures
```

---

## Repository Structure

```
code/
├── glmhmmt/                        # installable Python package
│   └── src/glmhmmt/
│       ├── model.py                # SoftmaxGLMHMM — core model, EM fitting
│       ├── features.py             # shared feature builders (action traces, etc.)
│       ├── plots.py                # plots for 3AFC / MCDR tasks
│       └── plots_alexis.py         # plots for 2AFC tasks
│
├── tasks/                          # one file per experimental task
│   ├── base.py                     # TaskAdapter abstract base class
│   ├── mcdr.py                     # MCDRTask adapter
│   └── two_afc.py                  # TwoAFCTask adapter
│
├── scripts/
│   └── fit_model.py                # single generic fit script, all tasks/models
│
├── notebooks/
│   ├── preprocess_mcdr.py          # raw → data/mcdr.parquet  (run once)
│   ├── preprocess_two_afc.py       # raw → data/two_afc.parquet  (run once)
│   ├── glmhmmt_analysis.py         # interactive results explorer
│   └── model_comparison.py         # BIC / model selection across K
│
├── paths.py                        # all filesystem paths in one place
├── pyproject.toml
└── HOWTO.md                        # this file

data/
├── raw/                            # original files — never modified
│   ├── mcdr/
│   └── two_afc/
├── mcdr.parquet                    # clean, all subjects (generated)
└── two_afc.parquet                 # clean, all subjects (generated)

results/
└── fits/
    ├── mcdr/
    │   ├── glmhmmt_K2/
    │   │   ├── A83_K2_glmhmmt_metrics.parquet
    │   │   └── A83_K2_glmhmmt_arrays.npz
    │   └── glmhmmt_K3/
    └── two_afc/
```

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
cd code/
uv sync                        # creates .venv and installs all dependencies
uv pip install -e glmhmmt/     # install the glmhmmt package in editable mode
```

To run any script or notebook:

```bash
uv run python scripts/fit_model.py --task mcdr --model glmhmmt --K 2
uv run marimo edit notebooks/glmhmmt_analysis.py
```

---

## Data Flow

### Step 1 — Preprocess (run once per task)

Open the preprocessing notebook and run it top to bottom:

```bash
uv run marimo edit notebooks/preprocess_mcdr.py
```

This notebook:
- Globs all raw files (one per subject, possibly nested in experiment folders)
- Unifies column names across experiments / recording rigs
- Applies task-specific filtering (RT bounds, valid dates, minimum trials per session)
- Selects only the columns needed for modelling
- Writes `data/{task}.parquet` with zstd compression

The resulting parquet **is the documented dataset**. When a reviewer asks how trials
were filtered, you point to this notebook — the criteria are explicit and in one place.

**Partitioned data (>5M trials):** if the parquet becomes large, the preprocessing
notebook writes one file per subject into `data/{task}/`. The task adapter reads
either format without code changes — just swap one commented line in the adapter:

```python
DATA = paths.DATA_PATH / "mcdr.parquet"          # single file
# DATA = paths.DATA_PATH / "mcdr" / "*.parquet"  # partitioned — uncomment if large
```

### Step 2 — Fit

```bash
uv run python scripts/fit_model.py --task mcdr --model glmhmmt --K 2 3 4
```

Results are saved to `results/fits/{task}/{model}_K{K}/` as:
- `{subject}_K{K}_{model}_metrics.parquet` — scalar metrics (BIC, accuracy, LL/trial)
- `{subject}_K{K}_{model}_arrays.npz` — fitted parameters, smoothed probabilities, predictions

### Step 3 — Explore

```bash
uv run marimo edit notebooks/glmhmmt_analysis.py
```

The analysis notebook reads from `results/fits/` and re-renders reactively when you
change task / model / K sliders. It never refits — if a fit is missing it shows a
warning and waits.

---

## Adding a New Task

Adding a task requires **two files**: a preprocessing notebook and a task adapter.

### 1. Preprocessing notebook

Create `notebooks/preprocess_{task}.py`. A minimal template:

```python
import marimo as mo
import polars as pl
import paths

# ── load all raw files ────────────────────────────────────────────────────────
raw_files = list((paths.DATA_PATH / "raw" / "my_task").glob("**/*.csv"))

dfs = []
for f in raw_files:
    df = (pl.read_csv(f)
            .with_columns(pl.lit(f.stem).alias("subject")))
    dfs.append(df)

df_raw = pl.concat(dfs, how="diagonal")  # diagonal tolerates missing cols

# ── rename non-standard columns ───────────────────────────────────────────────
df_raw = df_raw.rename({"resp": "response", "reaction_time": "RT"})

# ── filter ────────────────────────────────────────────────────────────────────
KEEP_COLS = ["subject", "session", "date", "trial_idx", "response", "RT",
             "stimulus", "correct"]

df_clean = (df_raw
    .select(KEEP_COLS)
    .filter(pl.col("RT").is_between(0.15, 5.0))   # adjust per task / species
    .filter(pl.col("response").is_not_null()))

# ── save ──────────────────────────────────────────────────────────────────────
df_clean.write_parquet(paths.DATA_PATH / "my_task.parquet", compression="zstd")
mo.md(f"Saved {len(df_clean)} trials, {df_clean['subject'].n_unique()} subjects")
```

### 2. Task adapter

Create `tasks/my_task.py`:

```python
import polars as pl
import numpy as np
import paths
from tasks.base import TaskAdapter
from glmhmmt.features import build_sequence_from_df   # or build_sequence_from_df_2afc

DATA = paths.DATA_PATH / "my_task.parquet"
# DATA = paths.DATA_PATH / "my_task" / "*.parquet"  # uncomment if partitioned


class MyTask(TaskAdapter):
    """
    One-line description of the task and species.
    Data: data/my_task.parquet
    """
    num_classes  = 2                           # number of choice categories
    plots_module = "glmhmmt.plots_alexis"      # or "glmhmmt.plots" for 3AFC

    def list_subjects(self, cfg):
        return (pl.scan_parquet(DATA)
                  .select("subject").unique()
                  .collect()["subject"].sort().to_list())

    def load_subject(self, subject, cfg):
        df = (pl.scan_parquet(DATA)
                .filter(pl.col("subject") == subject)
                .sort("trial_idx")
                .collect())

        # any task-specific filtering NOT already in the parquet goes here
        df = df.filter(pl.col("block_type") == "test")

        y, X, U, names, _ = build_sequence_from_df(
            df,
            tau=cfg.get("tau", 50.0),
            emission_cols=cfg.get("emission_cols"),
            transition_cols=cfg.get("transition_cols"),
        )
        return dict(
            y=np.asarray(y), X=np.asarray(X), U=np.asarray(U),
            session_ids=df["session"].to_numpy(), names=names,
        )
```

### 3. Register the adapter

In `scripts/fit_model.py`, add one import at the top with the others:

```python
import tasks.mcdr        # noqa: F401
import tasks.two_afc     # noqa: F401
import tasks.my_task     # noqa: F401  ← add this line
```

Done. `fit_model.py` and all analysis notebooks now accept `--task my_task`.

---

## Running Fits

### Basic usage

```bash
uv run python scripts/fit_model.py --task mcdr --model glmhmmt --K 3
```

### Multiple K values

```bash
uv run python scripts/fit_model.py --task mcdr --model glmhmmt --K 1 2 3 4 5
```

### Specific subjects

```bash
uv run python scripts/fit_model.py --task mcdr --model glmhmmt --K 3 \
    --subjects A83 A84
```

### GLM-HMM without transition inputs

```bash
uv run python scripts/fit_model.py --task mcdr --model glmhmm --K 3
```

### Override default feature columns

```bash
uv run python scripts/fit_model.py --task mcdr --model glmhmmt --K 3 \
    --emission_cols bias SL SR \
    --transition_cols speed
```

### Cluster / batch jobs

```bash
for K in 1 2 3 4 5; do
    uv run python scripts/fit_model.py --task mcdr --model glmhmmt --K $K
done
```

All results land in `results/fits/{task}/{model}_K{K}/` automatically.

---

## Exploring Results

```bash
uv run marimo edit notebooks/glmhmmt_analysis.py   # per-subject inspection
uv run marimo edit notebooks/model_comparison.py    # BIC curves across K
```

Both notebooks are fully reactive — changing any dropdown immediately re-renders all
figures. They show a warning if the requested fits have not been computed yet, rather
than refitting interactively.

---

## Design Decisions

**Why pre-processed parquets instead of loading raw files each time?**
Parsing is slow, error-prone, and task-specific. Doing it once means the modelling
code never touches raw files. The parquet is also what you share as the dataset with
a paper.

**Why `scan_parquet` instead of `read_parquet`?**
`scan_parquet` is lazy — Polars pushes the subject filter down to the file read and
never loads the full dataset into memory. It accepts a glob so single-file and
partitioned datasets use identical code.

**Why a `TaskAdapter` class and not plain functions?**
Three methods (`list_subjects`, `load_subject`, `get_plots`) define a contract that
is easy to describe in a methods section and straightforward for a new collaborator
to implement. Task-specific loading logic (e.g. filtering by block type, handling
multiple cohorts) lives in the adapter without touching shared code.

**Why Marimo instead of Jupyter?**
Marimo notebooks are plain Python files (git diffs are readable), cells re-execute
reactively when their dependencies change (no stale outputs), and they run as scripts
from the command line. A `mo.stop` with a run button prevents accidental refitting
when working interactively.