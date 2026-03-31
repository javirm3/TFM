---
title: Introduction
description: What is glmhmmt and what problem does it solve?
---

## What is glmhmmt?

**glmhmmt** is a Python package that implements a **Softmax Generalised Linear Model Hidden Markov Model (GLM-HMM)** on top of [Dynamax](https://github.com/probml/dynamax) — Google DeepMind's JAX-based library for probabilistic state space models.

It was developed as part of a Master's thesis (TFM, MAMME) in collaboration with IDIBAPS to analyse *decision-making strategies* in cognitive neuroscience experiments.

## The scientific problem

In behavioural neuroscience, subjects (humans or animals) perform repeated decision tasks across many sessions. A key question is:

> *Does the subject's decision strategy change over time, and how many latent strategies are being used?*

A **Hidden Markov Model** captures this naturally — discrete hidden states represent latent strategies, and the model infers when and how often each strategy is active.

The **GLM emission** connects observable covariates (stimulus contrast, previous choice, reward history…) to the probability of each observable choice, making the emission model interpretable.

## What glmhmmt adds on top of Dynamax

| Feature | Dynamax base | glmhmmt |
|---|---|---|
| JAX/JIT acceleration | ✅ | ✅ |
| GLM-HMM model class | Partial | `SoftmaxGLMHMM` with softmax emissions |
| Per-subject session-aware EM | ❌ | ✅ |
| Task-aware data loading | ❌ | `TaskAdapter.build_feature_df()` + `load_subject()` |
| Postprocessing utilities | ❌ | `build_trial_df`, `build_emission_weights_df` |
| Shared diagnostics + task-owned plots | ❌ | `glmhmmt.model_plots` + project `tasks.plots.*` |

## Package structure

```text
code/
├── glmhmmt/
│   ├── config.toml
│   └── src/glmhmmt/
│       ├── cli/
│       ├── model.py
│       ├── features.py
│       ├── postprocess.py
│       ├── views.py
│       └── model_plots.py
├── notebooks/
└── tasks/
    └── plots/
```

The core package stays task-agnostic. All task semantics live in the adapter and
its task-owned plotting module.

## Next steps

- **[Quickstart →](/docs/guide/quickstart)** — install and fit your first model
- **[Framework guide →](/docs/guide/framework)** — understand the repository layout and data flow
- **[Adding a task →](/docs/guide/tasks)** — add a new experimental task cleanly
- **[Adapter + skill workflow →](/docs/guide/tasks)** — build a new task and update the reusable Codex skill in the same pass
- **[API Reference →](/docs/api/model)** — detailed class and function docs
