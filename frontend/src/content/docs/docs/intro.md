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
| Feature engineering helpers | ❌ | `build_sequence_from_df` |
| Postprocessing utilities | ❌ | `build_trial_df`, `build_emission_weights_df` |
| Rich diagnostic plots | ❌ | Full `plots.py` module |

## Package structure

```
glmhmmt/
├── model.py          # SoftmaxGLMHMM — core model class
├── features.py       # build_sequence_from_df — raw data → tensors
├── postprocess.py    # build_trial_df, build_emission_weights_df, …
├── views.py          # SubjectFitView, build_views — fit result containers
└── plots.py          # Matplotlib diagnostic figures
```

## Next steps

- **[Quickstart →](/docs/guide/quickstart)** — install and fit your first model
- **[API Reference →](/docs/api/model)** — detailed class and function docs
