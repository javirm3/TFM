---
title: Interactive Notebooks
description: Explore the GLM-HMM models and results interactively.
---

These notebooks run entirely in your browser using WebAssembly (WASM) via [marimo](https://marimo.io). You can explore the data, adjust parameters, and visualize results without installing any local dependencies.

:::note
Model fitting is disabled in the web version as it requires heavy backends (JAX/Torch) not available in WASM. Please run the notebooks locally for model optimization.
:::

## Model Comparison

Compare fit metrics (Log-likelihood, BIC, Accuracy) across different model types (GLM, GLM-HMM, GLM-HMM-T) and state counts.

<iframe
  src="https://molab.marimo.io/github/javirm3/TFM/blob/main/code/notebooks/model_comparison.py/wasm?embed=true"
  width="100%"
  height="800px"
  frameborder="0"
  style="border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 2rem;"
></iframe>

## GLM-HMM Analysis

Deep dive into a single GLM-HMM fit, exploring emission weights, transition matrices, and posterior state probabilities.

<iframe
  src="https://molab.marimo.io/github/javirm3/TFM/blob/main/code/notebooks/glmhmm_analysis.py/wasm?embed=true"
  width="100%"
  height="800px"
  frameborder="0"
  style="border-radius: 8px; border: 1px solid #e2e8f0;"
></iframe>
