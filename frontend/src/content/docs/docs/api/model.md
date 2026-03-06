---
title: SoftmaxGLMHMM
description: Core model class — a Softmax GLM-HMM built on Dynamax.
---

The `SoftmaxGLMHMM` class is the main entry point of the package. It defines a Hidden Markov Model whose emission distribution is a **softmax-linear (multinomial logistic) GLM** — one set of regression weights per hidden state.

## Import

```python
from glmhmmt import SoftmaxGLMHMM
```

## Constructor

```python
SoftmaxGLMHMM(
    num_states: int,
    num_obs: int,
    num_features: int,
)
```

| Parameter | Type | Description |
|---|---|---|
| `num_states` | `int` | Number of discrete latent states (strategies). |
| `num_obs` | `int` | Number of observable outcomes per trial (e.g. 2 for binary choice). |
| `num_features` | `int` | Dimensionality of the input feature vector on each trial. |

## Methods

### `fit_per_subject`

```python
model.fit_per_subject(
    inputs: list[Array],
    choices: list[Array],
    masks: list[Array],
    num_iters: int = 100,
) -> list[Params]
```

Runs the **session-aware EM algorithm** independently for each subject.
Sessions within a subject are treated as **independent sequences** — the forward-backward pass is applied per session and sufficient statistics are accumulated before each M-step.

**Arguments**

| Name | Shape | Description |
|---|---|---|
| `inputs` | `list[ (T, F) ]` | Per-subject list of feature tensors. |
| `choices` | `list[ (T,) ]` | Per-subject list of integer choice arrays. |
| `masks` | `list[ (T,) ]` | Boolean mask — `False` on session boundaries. |
| `num_iters` | `int` | Maximum EM iterations. |

**Returns** a list of fitted parameter objects (one per subject), each containing:
- `transitions` — `(K, K)` transition matrix
- `emissions` — `(K, C, F)` emission weight tensor
- `initial` — `(K,)` initial state distribution

---

### `fit`

```python
model.fit(inputs, choices, masks, num_iters=100) -> Params
```

Fits a **single shared model** across all subjects (pooled). Useful for model recovery experiments.

---

### `most_likely_states`

```python
model.most_likely_states(params, inputs, choices, masks) -> Array
```

Runs the **Viterbi algorithm** given fitted parameters and returns the most likely state sequence for each trial.

---

### `posterior_state_probs`

```python
model.posterior_state_probs(params, inputs, choices, masks) -> Array  # (T, K)
```

Returns the **smoothed posterior** `p(z_t | observations, params)` via the forward-backward algorithm.

## Example

```python
from glmhmmt import SoftmaxGLMHMM

model = SoftmaxGLMHMM(num_states=3, num_obs=2, num_features=4)
fitted = model.fit_per_subject(inputs, choices, masks, num_iters=200)

# Viterbi decoding for subject 0
states = model.most_likely_states(fitted[0], inputs[0], choices[0], masks[0])
```
