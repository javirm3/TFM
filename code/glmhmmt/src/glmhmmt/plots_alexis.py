"""
plots_alexis.py
───────────────
Plotting utilities for 2-AFC (binary) GLM-HMM results.

API mirrors glmhmmt.plots so analysis notebooks can swap imports without
changing any call sites:

    import glmhmmt.plots_alexis as plots   # 2AFC

vs

    import glmhmmt.plots as plots          # 3AFC / MCDR

High-level functions (same signatures as plots.py):
  - plot_emission_weights
  - plot_posterior_probs
  - plot_state_accuracy
  - plot_session_trajectories
  - plot_state_occupancy
  - plot_psychometric_all        (≡ plot_categorical_performance_all)
  - plot_psychometric_by_state   (≡ plot_categorical_performance_by_state)
  - plot_trans_mat               (already homologous)
  - plot_trans_mat_boxplots      (already homologous)
  - plot_model_comparison
  - plot_model_comparison_diffs
  - norm_ll

Low-level primitives are kept for direct use:
  - remap_states
  - plot_weights / plot_weights_per_contrast / plot_weights_boxplot
  - plot_occupancy / plot_occupancy_boxplot
  - plot_ll
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy.stats import sem, ttest_1samp
from typing import Dict, List, Optional, Sequence, Tuple

# ── default state colour palette (matches plots.py config colours) ────────────
_DEFAULT_COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]

_LABEL_RANK = {
    "Engaged": 0,
    "Disengaged": 1,
    **{f"Disengaged {i}": i for i in range(1, 10)},
}


def _state_colors(K: int) -> List[str]:
    return _DEFAULT_COLORS[:K]


def _state_color(label: str, fallback_idx: int = 0) -> str:
    rank = _LABEL_RANK.get(label, fallback_idx)
    return _DEFAULT_COLORS[rank % len(_DEFAULT_COLORS)]


def _build_state_palette(
    state_labels_per_subj: dict,
) -> Tuple[dict, list]:
    """Return (palette_dict, hue_order) from a {subj: {k: label}} mapping."""
    seen: dict[str, int] = {}
    for _slbls in state_labels_per_subj.values():
        for k, lbl in _slbls.items():
            if lbl not in seen:
                seen[lbl] = _LABEL_RANK.get(lbl, len(seen))
    ordered = sorted(seen, key=lambda l: seen[l])
    pal = {lbl: _state_color(lbl, seen[lbl]) for lbl in ordered}
    return pal, ordered


def _default_labels(K: int, C: int = 2) -> List[str]:
    """Auto-generate state labels like ['Disengaged','Engaged'] for K=2."""
    if K == 1:
        return ["State 0"]
    if K == 2:
        return ["Disengaged", "Engaged"]
    if K == 3:
        return ["Engaged", "Biased L", "Biased R"]
    return [f"State {k}" for k in range(K)]


# ─────────────────────────────────────────────────────────────────────────────
# State remapping
# ─────────────────────────────────────────────────────────────────────────────

def remap_states(
    weights: np.ndarray,
    trans_mat: np.ndarray,
    smoothed_probs: np.ndarray,
    stim_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Re-order states so the most stimulus-sensitive is last ('Engaged').

    For K=2: [disengaged, engaged]
    For K=3: [engaged, biased-left, biased-right]

    Args:
        weights:        (K, C-1, M) emission weight array.
        trans_mat:      (K, K) transition matrix.
        smoothed_probs: (T, K) posterior state probabilities.
        stim_idx:       Feature column index used to rank engagement.

    Returns:
        Remapped (weights, trans_mat, smoothed_probs, remap_indices).
    """
    K = weights.shape[0]
    stim_w = weights[:, 0, stim_idx]
    engaged = int(np.argmax(np.abs(stim_w)))

    if K == 2:
        order = [1 - engaged, engaged]
    elif K == 3:
        others = [k for k in range(K) if k != engaged]
        bias_w = weights[:, 0, :]
        biased_left  = others[int(np.argmin([bias_w[k, 0] for k in others]))]
        biased_right = others[int(np.argmax([bias_w[k, 0] for k in others]))]
        order = [engaged, biased_left, biased_right]
    else:
        order = list(range(K))

    o = np.array(order)
    return weights[o], trans_mat[np.ix_(o, o)], smoothed_probs[:, o], order


# ─────────────────────────────────────────────────────────────────────────────
# Low-level weight plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_weights(
    weights: np.ndarray,
    feature_names: Sequence[str],
    state_labels: Optional[Sequence[str]] = None,
    title: str = "GLM-HMM weights",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Bar chart of emission weights per state.

    For C-1=1 (binary) each state has one row W[k,0,:].
    Multiple contrasts are averaged.

    Args:
        weights:       (K, C-1, M) or (K, M) weight array.
        feature_names: Names of the M features.
        state_labels:  Per-state labels.
        title:         Figure title.
        figsize:       Figure size.
        ax:            Optional existing Axes.

    Returns:
        matplotlib Figure.
    """
    W = np.asarray(weights)
    if W.ndim == 2:
        W = W[:, None, :]
    K, C_m1, M = W.shape
    labels = list(state_labels) if state_labels else _default_labels(K, C_m1 + 1)
    colors = _state_colors(K)
    x = np.arange(M)
    width = 0.8 / K

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(5, 0.7 * M), 3.5))
    else:
        fig = ax.figure

    for k in range(K):
        w_k = W[k].mean(axis=0)
        offset = (k - (K - 1) / 2) * width
        ax.bar(x + offset, w_k, width, label=labels[k], color=colors[k], alpha=0.85)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_weights_per_contrast(
    weights: np.ndarray,
    feature_names: Sequence[str],
    contrast_names: Optional[Sequence[str]] = None,
    state_labels: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """One subplot per contrast (row of W), all states overlaid."""
    W = np.asarray(weights)
    if W.ndim == 2:
        W = W[:, None, :]
    K, C_m1, M = W.shape
    labels = list(state_labels) if state_labels else _default_labels(K, C_m1 + 1)
    cnames = list(contrast_names) if contrast_names else [f"Contrast {c}" for c in range(C_m1)]
    colors = _state_colors(K)
    x = np.arange(M)
    bar_w = 0.8 / K

    fig, axes = plt.subplots(
        1, C_m1, figsize=figsize or (max(5, 0.7 * M) * C_m1, 3.5), sharey=True
    )
    axes = np.atleast_1d(axes)
    for c, ax in enumerate(axes):
        for k in range(K):
            offset = (k - (K - 1) / 2) * bar_w
            ax.bar(x + offset, W[k, c], bar_w, label=labels[k], color=colors[k], alpha=0.85)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.set_title(cnames[c])
        sns.despine(ax=ax)
    axes[0].set_ylabel("Weight")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_weights_boxplot(
    all_weights: np.ndarray,
    feature_names: Sequence[str],
    state_labels: Optional[Sequence[str]] = None,
    title: str = "GLM-HMM weights (across subjects)",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Error-bar plot of weights per state across N subjects."""
    W = np.asarray(all_weights)
    if W.ndim == 3:
        W = W[:, :, None, :]
    N, K, C_m1, M = W.shape
    W_avg = W.mean(axis=2)

    labels = list(state_labels) if state_labels else _default_labels(K, C_m1 + 1)
    colors = _state_colors(K)
    x = np.arange(M)
    bar_w = 0.8 / K

    fig, ax = plt.subplots(figsize=figsize or (max(5, 0.7 * M), 4))
    ax.axhline(0, color="k", lw=0.8, ls="--")

    for k in range(K):
        offset = (k - (K - 1) / 2) * bar_w
        for n in range(N):
            ax.plot(x + offset, W_avg[n, k], "o", color=colors[k], alpha=0.25, ms=4, zorder=2)
        mean_w = W_avg[:, k].mean(axis=0)
        sem_w  = sem(W_avg[:, k], axis=0)
        ax.errorbar(x + offset, mean_w, yerr=sem_w, fmt="o-",
                    color=colors[k], lw=1.5, capsize=3, label=labels[k], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Transition matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_trans_mat(
    trans_mat: np.ndarray,
    state_labels: Optional[Sequence[str]] = None,
    title: str = "Transition matrix",
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Heatmap of a (possibly averaged) transition matrix."""
    A = np.asarray(trans_mat)
    if A.ndim == 3:
        A = A.mean(axis=0)
    K = A.shape[0]
    labels = list(state_labels) if state_labels else _default_labels(K, 2)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (3 + 0.4 * K, 3 + 0.4 * K))
    else:
        fig = ax.figure

    im = ax.imshow(A, vmin=0, vmax=1, cmap="bone", origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    color="w" if A[i, j] < 0.5 else "k", fontsize=9)
    ticks = range(K)
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_ylabel("State $t$")
    ax.set_xlabel("State $t+1$")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    return fig


def plot_trans_mat_boxplots(
    all_trans_mats: np.ndarray,
    state_labels: Optional[Sequence[str]] = None,
    title: str = "Diagonal transitions",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Box-plots of diagonal (self-transition) probabilities across subjects."""
    A = np.asarray(all_trans_mats)
    K = A.shape[1]
    labels = list(state_labels) if state_labels else _default_labels(K, 2)

    diag = np.stack([A[:, k, k] for k in range(K)], axis=1)
    df = pd.DataFrame(diag, columns=labels)
    df_melt = df.melt(var_name="State", value_name="P(stay)")

    fig, ax = plt.subplots(figsize=figsize or (2 + 0.8 * K, 3.5))
    sns.boxplot(x="State", y="P(stay)", data=df_melt,
                showfliers=False, showcaps=False, fill=False,
                palette=dict(zip(labels, _state_colors(K))), ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(stay)")
    ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Low-level occupancy plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_occupancy(
    smoothed_probs: np.ndarray,
    state_labels: Optional[Sequence[str]] = None,
    trial_range: Optional[Tuple[int, int]] = None,
    title: str = "Posterior state probabilities",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Line plot of smoothed state probabilities over trials."""
    P = np.asarray(smoothed_probs)
    if trial_range is not None:
        P = P[trial_range[0]:trial_range[1]]
    T, K = P.shape
    labels = list(state_labels) if state_labels else _default_labels(K, 2)
    colors = _state_colors(K)

    fig, ax = plt.subplots(figsize=figsize or (14, 2.5))
    for k in range(K):
        ax.plot(P[:, k], color=colors[k], label=labels[k], lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(state)")
    ax.set_title(title)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1))
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_occupancy_boxplot(
    all_smoothed_probs: Sequence[np.ndarray],
    state_labels: Optional[Sequence[str]] = None,
    title: str = "State occupancy",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Box-plot of fractional state occupancy across subjects.

    Returns:
        (Figure, occupancies) where occupancies is (N, K).
    """
    P_list = [np.asarray(p) for p in all_smoothed_probs]
    K = P_list[0].shape[1]
    labels = list(state_labels) if state_labels else _default_labels(K, 2)
    colors = _state_colors(K)

    occs = np.array([p.mean(axis=0) for p in P_list])
    df = pd.DataFrame(occs, columns=labels)
    df_melt = df.melt(var_name="State", value_name="Occupancy")

    fig, ax = plt.subplots(figsize=figsize or (2 + 0.8 * K, 3.5))
    sns.boxplot(x="State", y="Occupancy", data=df_melt,
                showfliers=False, showcaps=False, fill=False,
                palette=dict(zip(labels, colors)), ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Occupancy")
    ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, occs


# ─────────────────────────────────────────────────────────────────────────────
# Log-likelihood
# ─────────────────────────────────────────────────────────────────────────────

def norm_ll(
    lls: Sequence[float],
    n_trials: Sequence[int],
    ll_null: Optional[Sequence[float]] = None,
    to_bits: bool = True,
) -> np.ndarray:
    """Normalise log-likelihoods to bits/Trial (or nats/Trial).

    The 2AFC chance baseline is log(0.5) × T (Bernoulli null).

    Returns:
        Normalised LL array of length N.
    """
    lls_arr = np.array(lls, dtype=float)
    n_arr   = np.array(n_trials, dtype=float)
    if ll_null is not None:
        lls_arr = lls_arr - np.asarray(ll_null, dtype=float)
    else:
        lls_arr = lls_arr - np.log(0.5) * n_arr
    ll_norm = lls_arr / n_arr
    if to_bits:
        ll_norm /= np.log(2)
    return ll_norm


def plot_ll(
    lls: Sequence[float],
    n_trials: Sequence[int],
    ll_null: Optional[Sequence[float]] = None,
    to_bits: bool = True,
    ax: Optional[plt.Axes] = None,
    color: str = "k",
    label: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Strip + box plot of normalised log-likelihoods.

    Returns:
        (Figure, normalised_lls).
    """
    ll_norm = norm_ll(lls, n_trials, ll_null, to_bits)
    ylabel = f"LL ({'bits' if to_bits else 'nats'}/Trial)"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (2, 3.5))
    else:
        fig = ax.figure

    lbl = label or "model"
    df_ll = pd.DataFrame({"LL": ll_norm, "Model": lbl})
    sns.boxplot(x="Model", y="LL", data=df_ll, ax=ax,
                showfliers=False, showcaps=False, fill=False, color=color)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    sns.despine(ax=ax)
    fig.tight_layout()

    print(f"{lbl}: {ll_norm.mean():.3f} ± {ll_norm.std(ddof=1):.3f} {ylabel}")
    return fig, ll_norm


def plot_model_comparison(
    all_lls: dict,
    all_n_trials: Sequence[int],
    ll_null: Optional[Sequence[float]] = None,
    to_bits: bool = True,
    title: str = "Model comparison",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Line plot of mean ± SEM normalised LL vs number of states."""
    ns_list = sorted(all_lls.keys())
    means, sems = [], []
    for ns in ns_list:
        ll_n = norm_ll(all_lls[ns], all_n_trials, ll_null, to_bits)
        means.append(ll_n.mean())
        sems.append(sem(ll_n))

    fig, ax = plt.subplots(figsize=figsize or (4, 3))
    ax.errorbar(ns_list, means, yerr=sems, fmt="o-", color="k", capsize=4)
    ax.set_xlabel("N states")
    ax.set_ylabel(f"LL ({'bits' if to_bits else 'nats'}/Trial)")
    ax.set_title(title)
    ax.set_xticks(ns_list)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_model_comparison_diffs(
    all_lls: dict,
    all_n_trials: Sequence[int],
    ll_null: Optional[Sequence[float]] = None,
    to_bits: bool = True,
    title: str = "ΔLL per state step",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Strip-plot of consecutive ΔLL (n-vs-n-1 states) with t-test stars."""
    ns_list = sorted(all_lls.keys())
    norms = {ns: norm_ll(all_lls[ns], all_n_trials, ll_null, to_bits) for ns in ns_list}
    diffs  = [norms[ns_list[i + 1]] - norms[ns_list[i]] for i in range(len(ns_list) - 1)]
    labels = [f"{ns_list[i+1]}s–{ns_list[i]}s" for i in range(len(ns_list) - 1)]

    df = pd.DataFrame({
        "ΔLL": np.concatenate(diffs),
        "comparison": np.repeat(labels, [len(d) for d in diffs]),
    })

    fig, ax = plt.subplots(figsize=figsize or (2.5 * len(diffs), 3.5))
    sns.stripplot(x="comparison", y="ΔLL", data=df, color="k",
                  alpha=0.5, jitter=True, ax=ax)
    ax.axhline(0, color="k", ls="--", lw=0.8)

    for i, diff in enumerate(diffs):
        _, p = ttest_1samp(diff, 0)
        ymax = ax.get_ylim()[1]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(i, ymax * 0.95, star, ha="center", va="top", fontsize=11)

    ax.set_xlabel("N states comparison")
    ax.set_ylabel(f"ΔLL ({'bits' if to_bits else 'nats'}/Trial)")
    ax.set_title(title)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Psychometric helpers  (2AFC equivalent of categorical performance panels)
# ─────────────────────────────────────────────────────────────────────────────

_LABELED_ILDS = {-70, -8, 8, 70}


# ─────────────────────────────────────────────────────────────────────────────
# GLM grid evaluation  (smooth sigmoid for psychometric plots)
# ─────────────────────────────────────────────────────────────────────────────

def eval_glm_on_ild_grid(
    weights: np.ndarray,
    X_cols: Sequence[str],
    ild_max: float = 70.0,
    n_grid: int = 300,
    lapse_rates: Optional[np.ndarray] = None,
    X_data: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a fitted GLM P(right) on a dense ILD grid.

    **Partial-dependence mode** (recommended, requires ``X_data``):
    For each ILD grid point, the ``stim_vals`` column of ``X_data`` is
    replaced with the normalised ILD value and P(right) is averaged over
    all trials.  This correctly marginalises over the history covariates
    (action traces, bias, etc.) using their *actual* empirical distribution,
    producing the psychometric curve the model predicts.

    **Synthetic-grid mode** (fallback when ``X_data`` is *None*):
    Builds a synthetic feature matrix where ``stim_vals`` sweeps the grid,
    ``bias`` = 1 and all other features = 0.  This is fast but gives wrong
    results when history covariates have a non-zero mean.

    Args:
        weights:     ``(K, C-1, M)`` or ``(C-1, M)`` emission weight array.
        X_cols:      Ordered list of M feature names matching the weight columns.
        ild_max:     Maximum |ILD| used for normalisation in ``parse_glmhmm``.
                     Defaults to 70 dB.
        n_grid:      Number of ILD points in the evaluation grid.
        lapse_rates: ``[gamma_L, gamma_R]`` lapse parameters from fitting.
                     *None* means no lapse correction.
        X_data:      ``(T, M)`` actual feature matrix used during fitting.
                     When provided, partial-dependence mode is used.

    Returns:
        ild_grid : ``(n_grid,)`` ILD values in dB.
        p_right  : ``(n_grid,)`` model P(rightward) for K=1, or
                   ``(K, n_grid)`` for K>1.
    """
    W = np.asarray(weights, dtype=float)
    if W.ndim == 2:
        W = W[None, ...]          # (1, C-1, M)
    K, _C_m1, M = W.shape

    X_cols_list = list(X_cols)

    # Accept any of these as the stimulus / ILD column
    _STIM_NAMES = {"stim_vals", "stim_d", "ild_norm", "ILD", "ild", "stimulus"}
    ild_idx  = next((i for i, n in enumerate(X_cols_list) if n in _STIM_NAMES), None)
    bias_idx = next((i for i, n in enumerate(X_cols_list) if n == "bias"), None)

    print(f"[eval_glm_on_ild_grid] X_cols={X_cols_list}")
    print(f"  ild_idx={ild_idx}  bias_idx={bias_idx}  K={K}  M={M}  "
          f"X_data={'yes ('+str(np.asarray(X_data).shape)+')' if X_data is not None else 'NO'}")
    print(f"  W[0,0,:]={np.asarray(W)[0, 0, :].round(3).tolist()}")

    ild_grid = np.linspace(-ild_max, ild_max, n_grid)
    ild_norm = ild_grid / ild_max

    gL, gR = 0.0, 0.0
    if lapse_rates is not None:
        _lr = np.asarray(lapse_rates, dtype=float).ravel()
        if len(_lr) >= 2:
            gL, gR = float(_lr[0]), float(_lr[1])
        elif len(_lr) == 1:
            gL = gR = float(_lr[0])

    p_right = np.zeros((K, n_grid))

    if X_data is not None and ild_idx is not None:
        # ── partial-dependence: average over real trial features ──────────────
        X_base = np.asarray(X_data, dtype=float).copy()
        for k in range(K):
            w = W[k, 0, :]
            other_logit  = X_base @ w
            stim_contrib = X_base[:, ild_idx] * w[ild_idx]
            base_logit   = other_logit - stim_contrib
            for gi, sv in enumerate(ild_norm):
                logit  = base_logit + sv * w[ild_idx]
                # W[k,0,:] parameterises P(class-0 = LEFT); class-1 (RIGHT) is
                # the softmax reference (logit=0). So P(right) = sigmoid(-logit).
                p_left = 1.0 / (1.0 + np.exp(-logit))
                p_right[k, gi] = float(np.mean(gL + (1.0 - gL - gR) * (1.0 - p_left)))
    else:
        # ── fallback: sweep only stim, fix others at empirical mean ──────────
        if X_data is not None:
            col_means = np.asarray(X_data, dtype=float).mean(axis=0)
        else:
            col_means = np.zeros(M)
            if bias_idx is not None:
                col_means[bias_idx] = 1.0

        X_grid = np.tile(col_means, (n_grid, 1))
        if ild_idx is not None:
            X_grid[:, ild_idx] = ild_norm
        if bias_idx is not None:
            X_grid[:, bias_idx] = 1.0        # bias is always 1

        for k in range(K):
            logit  = X_grid @ W[k, 0, :]
            # W[k,0,:] is logit for class-0 (LEFT); P(right) = sigmoid(-logit)
            p_left = 1.0 / (1.0 + np.exp(-logit))
            p_right[k] = gL + (1.0 - gL - gR) * (1.0 - p_left)

    if K == 1:
        return ild_grid, p_right[0]
    return ild_grid, p_right


def _mean_glm_curve(
    arrays_store: dict,
    subjects: Sequence[str],
    X_cols: Optional[Sequence[str]],
    ild_max: float = 70.0,
    state_k: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Average the per-subject GLM partial-dependence curve over subjects.

    Uses :func:`eval_glm_on_ild_grid` in partial-dependence mode (passing
    each subject's real ``X`` matrix) so history covariates are correctly
    marginalised rather than fixed to 0.

    Returns:
        ``(ild_grid, mean_p_right)`` or *None* if no valid fits are found.
    """
    all_p: list[np.ndarray] = []
    ild_g: Optional[np.ndarray] = None

    for subj in subjects:
        if subj not in arrays_store:
            continue
        W = arrays_store[subj].get("emission_weights")
        if W is None:
            continue
        # Resolve feature names
        cols = X_cols
        if cols is None:
            raw = arrays_store[subj].get("X_cols")
            if raw is None:
                continue
            cols = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, str) else [raw]

        # Read actual X matrix for partial-dependence evaluation
        X_data = arrays_store[subj].get("X")
        if X_data is not None:
            X_data = np.asarray(X_data, dtype=float)
            # Sanity check: column count must match weight columns
            _M_w = np.asarray(W).shape[-1]
            if X_data.shape[1] != _M_w:
                print(f"[_mean_glm_curve] {subj}: X_data cols {X_data.shape[1]} ≠ W cols {_M_w}, dropping X_data")
                X_data = None

        # When computing the curve for a specific state, restrict partial-
        # dependence to trials assigned to that state so that history
        # covariates are marginalised over its empirical distribution only.
        if X_data is not None and state_k is not None:
            _gamma = arrays_store[subj].get("smoothed_probs")
            if _gamma is not None:
                _map_k = np.argmax(np.asarray(_gamma), axis=1)
                _mask  = _map_k == state_k
                if _mask.sum() > 0:
                    X_data = X_data[_mask]
                # if the state has no trials, fall through with full X_data

        print(f"[_mean_glm_curve] {subj}: cols={list(cols)}  X_data={'yes ('+str(np.asarray(X_data).shape)+')' if X_data is not None else 'NO'}  state_k={state_k}")

        try:
            _lr = arrays_store[subj].get("lapse_rates")
            if _lr is not None:
                _lr = np.asarray(_lr, dtype=float).ravel()
                if not np.any(_lr > 0):
                    _lr = None
            ig, pg = eval_glm_on_ild_grid(
                W, cols, ild_max=ild_max, lapse_rates=_lr, X_data=X_data
            )
        except Exception as e:
            print(f"[_mean_glm_curve] {subj} skipped: {e}")
            continue

        # pg is (n_grid,) for K=1, or (K, n_grid) for K>1
        if pg.ndim == 2 and state_k is not None:
            pg = pg[state_k]
        elif pg.ndim == 2:
            pg = pg.mean(axis=0)

        all_p.append(pg)
        ild_g = ig

    if not all_p or ild_g is None:
        return None
    return ild_g, np.mean(all_p, axis=0)


def _sparse_ild_labels(ilds: list) -> list:
    """Return tick labels that show only the extreme values and ±8; rest are empty."""
    lo, hi = min(ilds), max(ilds)
    labeled = _LABELED_ILDS | {lo, hi}
    return [str(int(v)) if float(v) in labeled else "" for v in ilds]


def _psych_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    ild_col: str = "ILD",
    choice_col: str = "Choice",
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    title: str = "",
    xlabel: str = "ILD (dB)",
    ylabel: Optional[str] = None,
    color: str = "k",
    smooth_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    """Draw a pooled psychometric curve (P(right) vs ILD) on ax.

    Style mirrors plot_pc_across_batches:
    - Per-subject individual traces drawn with low alpha.
    - Extreme ILD positions compressed so inner values are not squeezed.
    - Pooled mean ± SEM as error-bar markers; model as a solid black line.
    - axhline at 0.5, axvline at 0.
    """
    if df.empty:
        ax.set_title(title)
        return

    subj_agg = (
        df.groupby([subj_col, ild_col], observed=True)
          .agg(data_mean=(choice_col, "mean"), model_mean=(pred_col, "mean"))
          .reset_index()
    )

    ilds = sorted(subj_agg[ild_col].unique())
    xpos = np.array(ilds, dtype=float)

    agg = (
        subj_agg.groupby(ild_col)
                .agg(
                    md=("data_mean",  "mean"),
                    sd=("data_mean",  "std"),
                    nd=("data_mean",  "count"),
                    mm=("model_mean", "mean"),
                )
                .reindex(ilds)
    )

    md    = agg["md"].values
    sd    = agg["sd"].fillna(0).values
    nd    = agg["nd"].clip(lower=1).values
    mm    = agg["mm"].values
    sem_d = sd / np.sqrt(nd)

    # per-subject individual traces (low alpha)
    for subj, grp in subj_agg.groupby(subj_col):
        grp_ilds = [i for i in ilds if i in grp[ild_col].values]
        xi = np.array(grp_ilds, dtype=float)
        yi = grp.set_index(ild_col).reindex(grp_ilds)["data_mean"].values
        ax.plot(xi, yi, "-o", color=color, alpha=0.15, lw=1, ms=3, zorder=2)

    # model line: smooth sigmoid over dense ILD grid (if available) else aggregated p_pred
    if smooth_curve is not None:
        ild_g, p_g = smooth_curve
        # Clip to the observed ILD range so the curve doesn't extend beyond
        # the data and compress the visible sigmoid into a near-flat line.
        _x0, _x1 = float(xpos[0]), float(xpos[-1])
        _clip = (ild_g >= _x0) & (ild_g <= _x1)
        ax.plot(ild_g[_clip], p_g[_clip], "-", color="black", lw=2, label="Model", zorder=6)
    else:
        ax.plot(xpos, mm, "-", color="black", lw=2, label="Model", zorder=6)

    # pooled data mean ± SEM
    ax.errorbar(xpos, md, yerr=sem_d, fmt="o", color=color,
                ecolor=color, elinewidth=1, capsize=3, ms=5,
                label="Data", zorder=5)

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(xpos, labels=_sparse_ild_labels(ilds))
    ax.tick_params(axis="x", which="major", length=4, width=0.8)
    ax.set_xlim(xpos[0], xpos[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _psych_state_panel(
    ax: plt.Axes,
    df_state: pd.DataFrame,
    ild_col: str,
    choice_col: str,
    pred_col: str,
    subj_col: str,
    color: str,
    label: str,
    smooth_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple:
    """Draw state-specific psychometric on ax.  Returns (data_h, model_h)."""
    if df_state.empty:
        return None, None

    subj_agg = (
        df_state.groupby([subj_col, ild_col], observed=True)
                .agg(data_mean=(choice_col, "mean"), model_mean=(pred_col, "mean"))
                .reset_index()
    )
    ilds = sorted(subj_agg[ild_col].unique())
    xpos = np.array(ilds, dtype=float)

    agg = (
        subj_agg.groupby(ild_col)
                .agg(
                    md=("data_mean", "mean"),
                    sd=("data_mean", "std"),
                    nd=("data_mean", "count"),
                    mm=("model_mean", "mean"),
                )
                .reindex(ilds)
    )
    md    = agg["md"].values
    sd    = agg["sd"].fillna(0).values
    nd    = agg["nd"].clip(lower=1).values
    mm    = agg["mm"].values
    sem_d = sd / np.sqrt(nd)

    # per-subject individual traces (low alpha)
    for subj, grp in subj_agg.groupby(subj_col):
        grp_ilds = [i for i in ilds if i in grp[ild_col].values]
        xi = np.array(grp_ilds, dtype=float)
        yi = grp.set_index(ild_col).reindex(grp_ilds)["data_mean"].values
        ax.plot(xi, yi, "-o", color=color, alpha=0.15, lw=1, ms=3, zorder=2)

    data_h = ax.errorbar(xpos, md, yerr=sem_d, fmt="o",
                         color=color, ecolor=color, elinewidth=1.2,
                         capsize=3, ms=5, zorder=5, label=label)
    # smooth sigmoid model line (if available) else aggregated p_pred
    if smooth_curve is not None:
        ild_g, p_g = smooth_curve
        # Clip the dense grid to the observed ILD range so the sigmoid
        # doesn't extend far beyond the data and compress the visible area.
        _x0, _x1 = float(xpos[0]), float(xpos[-1])
        _clip = (ild_g >= _x0) & (ild_g <= _x1)
        (model_h,) = ax.plot(ild_g[_clip], p_g[_clip], "-", color=color, lw=2.2,
                             zorder=6, label="_nolegend_")
    else:
        (model_h,) = ax.plot(xpos, mm, "-", color=color, lw=2.2, zorder=6,
                             label="_nolegend_")

    ax.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(xpos, labels=_sparse_ild_labels(ilds))
    ax.tick_params(axis="x", which="major", length=4, width=0.8)
    ax.set_xlim(xpos[0], xpos[-1])
    return data_h, model_h


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame preparation  (mirrors plots.prepare_predictions_df)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_predictions_df(df_pred):
    """Prepare a 2-AFC Trial-level predictions DataFrame for plotting.

    Accepts a polars or pandas DataFrame that already contains the per-Trial
    model predictions (``pL``, ``pR``) produced by the fit script.

    Expected input columns
    ----------------------
    Side   : int  – correct Side (0 = left, 1 = right)
    Choice : int  – animal's Choice (0 = left, 1 = right)
    Hit    : int/bool – Trial correct (1) or incorrect (0)
    pL     : float – model P(left Choice)
    pR     : float – model P(right Choice)

    Added / ensured output columns
    ------------------------------
    correct_bool    : bool  – Trial accuracy
    p_pred          : float – model P(right)  → psychometric x-axis
    p_model_correct : float – model P(correct Side)
    stimulus        : int   – alias for ``Side``, used for state indexing

    Returns
    -------
    DataFrame of the same type as the input (polars or pandas).
    """
    try:
        import polars as pl
        _is_polars = hasattr(df_pred, "lazy")
    except ImportError:
        _is_polars = False

    if _is_polars:
        df = df_pred.clone()

        if "correct_bool" not in df.columns:
            if "Hit" in df.columns:
                df = df.with_columns(pl.col("Hit").cast(pl.Boolean).alias("correct_bool"))
            elif "performance" in df.columns:
                df = df.with_columns(pl.col("performance").cast(pl.Boolean).alias("correct_bool"))
            else:
                raise ValueError("No 'Hit', 'performance', or 'correct_bool' column found.")

        if "pL" not in df.columns or "pR" not in df.columns:
            raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")

        df = df.with_columns(
            pl.col("pR").alias("p_pred"),
            pl.when(pl.col("Side") == 0)
              .then(pl.col("pL"))
              .otherwise(pl.col("pR"))
              .alias("p_model_correct"),
        )

        if "stimulus" not in df.columns:
            if "Side" in df.columns:
                df = df.with_columns(pl.col("Side").cast(pl.Int32).alias("stimulus"))
            else:
                raise ValueError("No 'Side' column to derive 'stimulus' from.")

        return df

    else:
        # pandas path
        df = df_pred.copy()

        if "correct_bool" not in df.columns:
            if "Hit" in df.columns:
                df["correct_bool"] = df["Hit"].astype(bool)
            elif "performance" in df.columns:
                df["correct_bool"] = df["performance"].astype(bool)
            else:
                raise ValueError("No 'Hit', 'performance', or 'correct_bool' column found.")

        if "pL" not in df.columns or "pR" not in df.columns:
            raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")

        df["p_pred"] = df["pR"]
        df["p_model_correct"] = df.apply(
            lambda row: row["pL"] if row["Side"] == 0 else row["pR"], axis=1
        )

        if "stimulus" not in df.columns:
            if "Side" in df.columns:
                df["stimulus"] = df["Side"].astype(int)
            else:
                raise ValueError("No 'Side' column to derive 'stimulus' from.")

        return df


# ─────────────────────────────────────────────────────────────────────────────
# High-level API  — mirrors glmhmmt.plots exactly
# ─────────────────────────────────────────────────────────────────────────────

def plot_emission_weights(
    arrays_store: dict,
    state_labels: dict,
    names: dict,
    K: int,
    subjects: list,
    save_path=None,
) -> Tuple[plt.Figure, plt.Figure]:
    """Emission weights: per-subject bar charts + multi-subject error-bar figure.

    Mirrors plots.plot_emission_weights.

    Parameters
    ----------
    arrays_store : {subj: npz-dict with "emission_weights"}
    state_labels : {subj: {state_idx: label_str}}
    names        : dict with key "X_cols"
    K            : number of states
    subjects     : subject IDs to include
    save_path    : optional Path – per-subject figure saved there if provided

    Returns
    -------
    fig_single, fig_multi
    """
    feat_names = names.get("X_cols", [])
    _pal, _hue_order = _build_state_palette(state_labels)

    valid_subjs = [s for s in subjects if s in arrays_store
                   and arrays_store[s].get("emission_weights") is not None]

    if not valid_subjs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig, fig

    # ── per-subject panel ─────────────────────────────────────────────────────
    n_cols = min(3, len(valid_subjs))
    n_rows = int(np.ceil(len(valid_subjs) / n_cols))
    M = len(feat_names) or 1
    fig_single, axes_s = plt.subplots(
        n_rows, n_cols,
        figsize=(max(5, 0.7 * M) * n_cols, 3.5 * n_rows),
        sharey=True, squeeze=False,
    )

    all_w = []
    for idx, subj in enumerate(valid_subjs):
        ax = axes_s[idx // n_cols][idx % n_cols]
        W  = arrays_store[subj]["emission_weights"]
        slbls = state_labels.get(subj, {})
        lbls  = [slbls.get(k, f"S{k}") for k in range(K)]
        fn_subj = arrays_store[subj].get("X_cols", feat_names)
        plot_weights(W, fn_subj, state_labels=lbls, title=f"Subject {subj}", ax=ax)
        all_w.append(np.asarray(W))

    for idx in range(len(valid_subjs), n_rows * n_cols):
        axes_s[idx // n_cols][idx % n_cols].set_visible(False)

    fig_single.suptitle(f"Emission weights  (K={K})", y=1.01)
    sns.despine(fig=fig_single)
    fig_single.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig_single.savefig(save_path, dpi=150, bbox_inches="tight")

    # ── multi-subject summary ─────────────────────────────────────────────────
    if len(all_w) > 1:
        W_stack = np.stack(all_w, axis=0)
        if W_stack.ndim == 3:
            W_stack = W_stack[:, :, None, :]
        fn = feat_names or (arrays_store[valid_subjs[0]].get("X_cols") or [])
        fig_multi = plot_weights_boxplot(W_stack, fn,
                                         state_labels=_hue_order[:K],
                                         title=f"Emission weights – all subjects  (K={K})")
    else:
        fig_multi = fig_single

    return fig_single, fig_multi


def plot_posterior_probs(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
    t0: int = 0,
    t1: int = 199,
) -> plt.Figure:
    """Stacked-area posterior state probability plot.

    Mirrors plots.plot_posterior_probs.

    Returns
    -------
    fig
    """
    _selected = [s for s in subjects if s in arrays_store]
    if not _selected:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    colors = _state_colors(K)
    fig, axes = plt.subplots(len(_selected), 1,
                             figsize=(14, 3 * len(_selected)), squeeze=False)

    for i, subj in enumerate(_selected):
        ax = axes[i, 0]
        P     = np.asarray(arrays_store[subj]["smoothed_probs"])
        P_sub = P[t0:min(t1, len(P))]
        T_sub = P_sub.shape[0]

        ax.stackplot(np.arange(T_sub), P_sub.T, colors=colors[:K], alpha=0.8)
        slbls = state_labels.get(subj, {})
        legend_patches = [
            plt.matplotlib.patches.Patch(color=colors[k], label=slbls.get(k, f"S{k}"))
            for k in range(K)
        ]
        ax.legend(handles=legend_patches, frameon=False,
                  bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.set_xlim(0, T_sub)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(state)")
        ax.set_title(f"Subject {subj}")

    axes[-1, 0].set_xlabel("Trial")
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    sns.despine(fig=fig)
    return fig


def plot_state_accuracy(
    arrays_store: dict,
    state_labels: dict,
    df_all,
    K: int,
    subjects: list,
    thresh: float = 0.5,
    session_col: str = "Session",
    sort_col: str = "Trial",
    **kwargs,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Per-state accuracy bar chart.

    Mirrors plots.plot_state_accuracy.
    Accuracy = fraction of correct trials (Hit==1) assigned to each state
    with posterior probability ≥ thresh, on ILD≠0 trials only.

    Returns
    -------
    fig, summary_df
    """
    _label_order = (
        ["Engaged", "Disengaged"] if K == 2
        else ["Engaged"] + [f"Disengaged {i}" for i in range(1, K)]
    )
    _cmap = {"All": "#999999"}
    for ri, lbl in enumerate(_label_order):
        _cmap[lbl] = _state_color(lbl, ri)
    _x_labels = ["All"] + _label_order

    _acc_records = []
    for subj in subjects:
        if subj not in arrays_store:
            continue
        P = np.asarray(arrays_store[subj]["smoothed_probs"])  # (T, K)

        try:
            import polars as pl
            if hasattr(df_all, "filter"):
                df_sub = df_all.filter(pl.col("subject") == subj)
                hits = df_sub["Hit"].to_numpy().astype(float)
                ilds = df_sub["ILD"].to_numpy().astype(float)
            else:
                raise AttributeError
        except (ImportError, AttributeError):
            df_sub = df_all[df_all["subject"] == subj]
            hits = df_sub["Hit"].to_numpy().astype(float)
            ilds = df_sub["ILD"].to_numpy().astype(float)

        T = min(len(P), len(hits))
        P, hits, ilds = P[:T], hits[:T], ilds[:T]
        stim_mask = np.abs(ilds) > 0

        valid = stim_mask & np.isfinite(hits)
        if valid.sum() > 0:
            _acc_records.append({"subject": subj, "label": "All",
                                  "acc": hits[valid].mean() * 100, "n": valid.sum()})

        slbls = state_labels.get(subj, {})
        for k in range(K):
            lbl  = slbls.get(k, f"State {k}")
            mask = stim_mask & (P[:, k] >= thresh) & np.isfinite(hits)
            if mask.sum() > 0:
                _acc_records.append({"subject": subj, "label": lbl,
                                      "acc": hits[mask].mean() * 100, "n": mask.sum()})

    if not _acc_records:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig, pd.DataFrame()

    _df_acc = pd.DataFrame(_acc_records)
    _tbl = (
        _df_acc.groupby("label")[["acc", "n"]]
        .agg({"acc": "mean", "n": "sum"})
        .reindex(_x_labels)
        .rename(columns={"acc": "mean_acc (%)", "n": "total_trials"})
        .round(1)
    )

    fig, ax = plt.subplots(figsize=(2 + len(_x_labels) * 0.9, 4))
    rng = np.random.default_rng(42)
    for li, lbl in enumerate(_x_labels):
        rows = _df_acc[_df_acc["label"] == lbl]["acc"].dropna().values
        if len(rows) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(rows))
        ax.scatter(np.full(len(rows), li) + jitter, rows,
                   color=_cmap.get(lbl, "k"), alpha=0.6, s=30, zorder=3)
        ax.errorbar(li, rows.mean(), yerr=rows.std(ddof=1) / max(np.sqrt(len(rows)), 1),
                    fmt="o", color=_cmap.get(lbl, "k"), ms=8, capsize=4, lw=2, zorder=4)

    ax.axhline(50, color="black", linestyle="--", linewidth=0.9, alpha=0.5,
               label="Chance (50%)")
    ax.set_xticks(range(len(_x_labels)))
    ax.set_xticklabels(_x_labels, rotation=20, ha="right")
    ax.set_xlabel("State")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 105)
    ax.set_title(f"Per-state accuracy  (K={K},  posterior ≥ {thresh},  ILD≠0)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    sns.despine(fig=fig)
    return fig, _tbl


def plot_session_trajectories(
    arrays_store: dict,
    state_labels: dict,
    df_all,
    K: int,
    subjects: list,
    session_col: str = "Session",
    sort_col: str = "Trial",
    **kwargs,
) -> plt.Figure:
    """Average state-probability trajectories within a session (mean ± SEM).

    Mirrors plots.plot_session_trajectories.

    Returns
    -------
    fig
    """
    palette = _state_colors(K)
    fig, axes = plt.subplots(len(subjects), 1,
                             figsize=(10, 3.5 * len(subjects)), squeeze=False)

    for i, subj in enumerate(subjects):
        ax = axes[i, 0]
        if subj not in arrays_store:
            ax.set_title(f"Subject {subj} — no fit")
            continue

        P = np.asarray(arrays_store[subj]["smoothed_probs"])

        try:
            import polars as pl
            if hasattr(df_all, "filter"):
                df_sub = df_all.filter(pl.col("subject") == subj).sort(sort_col)
                _sess_arr = df_sub[session_col].to_numpy()
            else:
                raise AttributeError
        except (ImportError, AttributeError):
            df_sub = df_all[df_all["subject"] == subj].sort_values(sort_col)
            _sess_arr = df_sub[session_col].to_numpy()

        T = min(len(P), len(_sess_arr))
        P, _sess_arr = P[:T], _sess_arr[:T]

        sess_ids = np.unique(_sess_arr)
        sess_len = int(np.median([np.sum(_sess_arr == s) for s in sess_ids]))
        traj = np.full((len(sess_ids), sess_len, K), np.nan)
        for si, s in enumerate(sess_ids):
            idx = np.where(_sess_arr == s)[0]
            n   = min(len(idx), sess_len)
            traj[si, :n, :] = P[idx[:n], :]

        mean_traj = np.nanmean(traj, axis=0)
        sem_traj  = sem(traj, axis=0, nan_policy="omit")

        slbls = state_labels.get(subj, {})
        for k in range(K):
            lbl = slbls.get(k, f"S{k}")
            t   = np.arange(mean_traj.shape[0])
            ax.plot(t, mean_traj[:, k], color=palette[k], lw=2, label=lbl)
            ax.fill_between(t,
                            mean_traj[:, k] - sem_traj[:, k],
                            mean_traj[:, k] + sem_traj[:, k],
                            color=palette[k], alpha=0.2)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Trial within session")
        ax.set_ylabel("P(state)")
        ax.set_title(f"Subject {subj}")
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_state_occupancy(
    arrays_store: dict,
    state_labels: dict,
    df_all,
    K: int,
    subjects: list,
    session_col: str = "Session",
    sort_col: str = "Trial",
    **kwargs,
) -> plt.Figure:
    """Fractional occupancy bar chart + state-switch histogram per session.

    Mirrors plots.plot_state_occupancy.

    Returns
    -------
    fig
    """
    palette = _state_colors(K)
    fig, axes = plt.subplots(len(subjects), 2,
                             figsize=(10, 3.5 * len(subjects)), squeeze=False)

    for i, subj in enumerate(subjects):
        ax_occ, ax_chg = axes[i, 0], axes[i, 1]
        if subj not in arrays_store:
            ax_occ.set_title(f"Subject {subj} — no fit")
            continue

        P     = np.asarray(arrays_store[subj]["smoothed_probs"])
        slbls = state_labels.get(subj, {})
        labels = [slbls.get(k, f"S{k}") for k in range(K)]

        occ = P.mean(axis=0)
        ax_occ.bar(labels, occ, color=palette[:K], alpha=0.85)
        ax_occ.set_ylim(0, 1)
        ax_occ.set_ylabel("Fractional occupancy")
        ax_occ.set_title(f"Subject {subj} – occupancy")

        try:
            import polars as pl
            if hasattr(df_all, "filter"):
                df_sub = df_all.filter(pl.col("subject") == subj).sort(sort_col)
                _sess_arr = df_sub[session_col].to_numpy()
            else:
                raise AttributeError
        except (ImportError, AttributeError):
            df_sub = df_all[df_all["subject"] == subj].sort_values(sort_col)
            _sess_arr = df_sub[session_col].to_numpy()

        T = min(len(P), len(_sess_arr))
        viterbi = np.argmax(P[:T], axis=1)
        _sess_arr = _sess_arr[:T]

        changes_per_sess = []
        for s in np.unique(_sess_arr):
            v = viterbi[_sess_arr == s]
            changes_per_sess.append(int(np.sum(np.diff(v) != 0)))

        max_chg = max(changes_per_sess) if changes_per_sess else 1
        ax_chg.hist(changes_per_sess, bins=range(0, max_chg + 2),
                    color="#888888", alpha=0.75, edgecolor="white")
        ax_chg.set_xlabel("# state switches / session")
        ax_chg.set_ylabel("# sessions")
        ax_chg.set_title(f"Subject {subj} – state switches")

    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Session deep-dive
# ─────────────────────────────────────────────────────────────────────────────

def plot_session_deepdive(
    arrays_store: dict,
    state_labels: dict,
    df_all,
    names: dict,
    K: int,
    subj: str,
    sess,
    session_col: str = "Session",
    sort_col: str = "Trial",
    **kwargs,
) -> plt.Figure:
    """Session deep-dive: P(Engaged) + cumulative accuracy (twin axis) + action traces.

    2AFC equivalent of plots.plot_session_deepdive.

    Action traces are auto-detected from the saved X matrix; looks for
    columns whose names start with 'A_' (e.g. A_L, A_R, A_plus, A_minus).

    Returns
    -------
    fig
    """
    try:
        import polars as pl
        _has_pl = True
    except ImportError:
        _has_pl = False

    # Coerce sess to match the column dtype (dropdown widgets return strings)
    try:
        sess = int(sess)
    except (TypeError, ValueError):
        pass

    # ── filter df to subject, then to session ─────────────────────────────────
    if _has_pl and hasattr(df_all, "filter"):
        _df_sub_all = (
            df_all
            .filter(pl.col("subject") == subj)
            .sort(sort_col)
            .filter(pl.col(session_col).count().over(session_col) >= 2)
        )
        _sess_mask = _df_sub_all[session_col].to_numpy() == sess
        _df_sess = (
            df_all
            .filter((pl.col("subject") == subj) & (pl.col(session_col) == sess))
            .sort(sort_col)
        )
        _hit   = _df_sess["Hit"].to_numpy().astype(float)
        _ild   = _df_sess["ILD"].to_numpy().astype(float)
        _choice = _df_sess["Choice"].to_numpy().astype(int)
    else:
        _df_sub_all = (
            df_all[df_all["subject"] == subj]
            .sort_values(sort_col)
        )
        _sess_counts = _df_sub_all.groupby(session_col)[sort_col].transform("count")
        _df_sub_all = _df_sub_all[_sess_counts >= 2]
        _sess_mask  = _df_sub_all[session_col].to_numpy() == sess
        _df_sess = (
            df_all[(df_all["subject"] == subj) & (df_all[session_col] == sess)]
            .sort_values(sort_col)
        )
        _hit    = _df_sess["Hit"].to_numpy().astype(float)
        _ild    = _df_sess["ILD"].to_numpy().astype(float)
        _choice = _df_sess["Choice"].to_numpy().astype(int)

    _probs_all = arrays_store[subj]["smoothed_probs"]
    _probs     = _probs_all[_sess_mask]
    _T         = _probs.shape[0]
    # guard: align lengths in case session filter differs
    _T = min(_T, len(_hit))
    _probs, _hit, _ild, _choice = (
        _probs[:_T], _hit[:_T], _ild[:_T], _choice[:_T]
    )
    _x = np.arange(_T)

    # ── auto-detect action traces from X ─────────────────────────────────────
    _X_cols_s = arrays_store[subj].get("X_cols") or names.get("X_cols", [])
    _X_idx    = {f: i for i, f in enumerate(_X_cols_s)}
    _X_sess   = arrays_store[subj]["X"][_sess_mask][:_T]

    _trace_colors = {
        "A_plus": "royalblue", "A_minus": "gold",
        "A_L": "royalblue",    "A_C": "gold",    "A_R": "tomato",
    }
    _trace_sources = {
        tc: (_X_sess, idx)
        for tc, idx in _X_idx.items()
        if tc.startswith("A_")
    }

    # ── cumulative accuracy on ILD≠0 trials ──────────────────────────────────
    _nz      = _ild != 0
    _cum_acc = np.full(_T, np.nan)
    _cum_n, _cum_s = 0, 0.0
    for _ti in range(_T):
        if _nz[_ti]:
            _cum_s += _hit[_ti]; _cum_n += 1
        if _cum_n > 0:
            _cum_acc[_ti] = 100.0 * _cum_s / _cum_n

    # ── find "Engaged" state index ────────────────────────────────────────────
    _slbl = state_labels.get(subj, {k: f"State {k}" for k in range(K)})
    _engaged_k = next(
        (k for k in range(K) if _LABEL_RANK.get(_slbl.get(k, ""), k) == 0), 0
    )
    _palette = _DEFAULT_COLORS

    # ── figure ────────────────────────────────────────────────────────────────
    _n_rows = 2 if _trace_sources else 1
    _height_ratios = [2, 1.5] if _trace_sources else [1]
    fig, _axes = plt.subplots(
        _n_rows, 1,
        figsize=(14, 5 + 2.5 * (_n_rows - 1)),
        sharex=True,
        gridspec_kw={"height_ratios": _height_ratios},
    )
    _axes = np.atleast_1d(_axes)
    _ax1  = _axes[0]

    # top panel: P(Engaged) + Choice ticks
    _ax1.plot(_x, _probs[:, _engaged_k],
              color=_palette[0], lw=2,
              label=f"P({_slbl.get(_engaged_k, 'Engaged')})")

    _choice_cols = {0: "royalblue", 1: "tomato"}
    _choice_lbls = {0: "L", 1: "R"}
    for _resp, _c in _choice_cols.items():
        _m = _choice == _resp
        _ax1.scatter(
            _x[_m], np.ones(_m.sum()) * 1.03,
            c=_c, s=5, marker="|",
            label=_choice_lbls[_resp],
            transform=_ax1.get_xaxis_transform(),
            clip_on=False,
        )

    _ax1.set_ylim(0, 1)
    _ax1.set_ylabel("State probability")
    _ax1.set_title(f"Subject {subj}  —  session {sess}  ({_T} trials)")

    _ax1r = _ax1.twinx()
    _ax1r.plot(_x, _cum_acc, color="black", lw=1.8, linestyle="-", alpha=0.7,
               label="Cumul. accuracy")
    _ax1r.axhline(50, color="grey", lw=0.9, linestyle="--", alpha=0.5,
                  label="Chance (50%)")
    _ax1r.set_ylim(0, 105)
    _ax1r.set_ylabel("Accuracy (%)", color="black")

    _lines1,  _labs1  = _ax1.get_legend_handles_labels()
    _lines1r, _labs1r = _ax1r.get_legend_handles_labels()
    _ax1.legend(
        _lines1 + _lines1r, _labs1 + _labs1r,
        bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False,
    )

    # bottom panel: action traces (if any)
    if _trace_sources:
        _ax2 = _axes[1]
        for _tc, (_arr, _ci) in _trace_sources.items():
            _ax2.plot(_x, _arr[:, _ci],
                      label=_tc,
                      color=_trace_colors.get(_tc, "gray"),
                      lw=1.5, alpha=0.85)
        _ax2.set_ylabel("Action trace")
        _ax2.set_ylim(0, None)
        _ax2.set_xlabel("Trial within session")
        _ax2.legend(
            bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False,
        )
    else:
        _ax1.set_xlabel("Trial within session")

    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    sns.despine(fig=fig, right=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Psychometric performance  (≡ categorical performance in plots.py)
# ─────────────────────────────────────────────────────────────────────────────

def plot_categorical_performance_all(
    df,
    model_name: str,
    ild_col: str = "ILD",
    choice_col: str = "Choice",
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    cond_col: str = "condition",
    exp_col: str = "experiment",
    arrays_store: Optional[dict] = None,
    X_cols: Optional[Sequence[str]] = None,
    ild_max: float = 70.0,
) -> plt.Figure:
    """Overall psychometric + by-condition + by-experiment panels.

    2AFC equivalent of plots.plot_categorical_performance_all.

    Panels
    ------
    a) Overall      – P(right) vs ILD, all trials pooled
    b) By condition – separate curves per rest / saline / drug
                      (skipped if 'condition' column absent)
    c) By experiment – separate curves per experiment batch

    Parameters
    ----------
    df         : Polars or pandas DataFrame with Trial-level predictions.
                 Must contain: ILD, Choice (0/1), p_pred, subject.
    model_name : String for figure suptitle.

    Returns
    -------
    fig
    """
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy()

    has_cond = cond_col in df_pd.columns
    has_exp  = exp_col  in df_pd.columns
    n_panels = 1 + int(has_cond) + int(has_exp)

    # Pre-compute smooth GLM sigmoid averaged over all subjects
    _all_subjects = list(df_pd[subj_col].unique()) if subj_col in df_pd.columns else []
    _smooth_all = (
        _mean_glm_curve(arrays_store, _all_subjects, X_cols, ild_max=ild_max)
        if arrays_store is not None
        else None
    )

    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 4), sharey=True)
    axes = np.atleast_1d(axes)
    ax_idx = 0

    # a) Overall
    _psych_panel(
        axes[ax_idx], df_pd,
        ild_col=ild_col, choice_col=choice_col, pred_col=pred_col, subj_col=subj_col,
        title="a) Overall psychometric", xlabel="ILD (dB)",
        ylabel="P(rightward Choice)", color="#2b7bba",
        smooth_curve=_smooth_all,
    )
    ax_idx += 1

    # b) By condition
    if has_cond:
        conds = sorted(df_pd[cond_col].dropna().unique())
        cond_colors = {"rest": "#444444", "saline": "#1f77b4", "drug": "#d62728"}
        for cond in conds:
            _cond_subjs = list(df_pd[df_pd[cond_col] == cond][subj_col].unique())
            _smooth_cond = (
                _mean_glm_curve(arrays_store, _cond_subjs, X_cols, ild_max=ild_max)
                if arrays_store is not None
                else None
            )
            _psych_panel(
                axes[ax_idx], df_pd[df_pd[cond_col] == cond],
                ild_col=ild_col, choice_col=choice_col, pred_col=pred_col, subj_col=subj_col,
                title=f"b) {cond}", xlabel="ILD (dB)",
                color=cond_colors.get(cond, "k"),
                smooth_curve=_smooth_cond,
            )
        ax_idx += 1

    # c) By experiment
    if has_exp:
        exps = sorted(df_pd[exp_col].dropna().unique())
        exp_palette = sns.color_palette("Set2", len(exps))
        for ei, exp in enumerate(exps):
            _exp_subjs = list(df_pd[df_pd[exp_col] == exp][subj_col].unique())
            _smooth_exp = (
                _mean_glm_curve(arrays_store, _exp_subjs, X_cols, ild_max=ild_max)
                if arrays_store is not None
                else None
            )
            _psych_panel(
                axes[ax_idx], df_pd[df_pd[exp_col] == exp],
                ild_col=ild_col, choice_col=choice_col, pred_col=pred_col, subj_col=subj_col,
                title=f"c) {exp}", xlabel="ILD (dB)",
                color=exp_palette[ei],
                smooth_curve=_smooth_exp,
            )

    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    sns.despine(fig=fig)
    fig.suptitle(model_name, y=1.02)
    fig.tight_layout()
    return fig, None


def plot_categorical_performance_all_by_state(
    df,
    smoothed_probs: np.ndarray,
    state_labels: dict,
    model_name: str,
    state_assign: Optional[np.ndarray] = None,
    ild_col: str = "ILD",
    choice_col: str = "Choice",
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    arrays_store: Optional[dict] = None,
    X_cols: Optional[Sequence[str]] = None,
    ild_max: float = 70.0,
) -> plt.Figure:
    """Per-state psychometric grid (K panels, one per state).

    2AFC equivalent of plots.plot_categorical_performance_by_state.

    Each state gets its own panel showing P(right) vs ILD; data (markers) and
    model prediction (lines) are drawn in the state's colour.

    Parameters
    ----------
    df             : Trial-level DataFrame (polars or pandas).
    smoothed_probs : (T, K) posterior state probabilities, ignored when
                     ``state_assign`` is provided.
    state_labels   : {state_idx: label_str} for single-subject, or
                     {subj: {state_idx: label_str}} for multi-subject.
    model_name     : string used as figure suptitle.
    state_assign   : optional pre-computed (T,) int array of MAP states.

    Returns
    -------
    (fig, None)
    """
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas().reset_index(drop=True)
    else:
        df_pd = df.reset_index(drop=True)

    if state_assign is not None:
        _arr = np.asarray(state_assign)
        K    = int(_arr.max()) + 1
    else:
        _arr = np.argmax(np.asarray(smoothed_probs), axis=1)
        K    = np.asarray(smoothed_probs).shape[1]

    assert len(df_pd) == len(_arr), (
        f"df has {len(df_pd)} rows but state assignment has T={len(_arr)}"
    )

    df_pd = df_pd.copy()
    df_pd["_state_k"] = _arr

    # Resolve labels: accept {int: str} or {subj: {int: str}}
    first_val = next(iter(state_labels.values()), None)
    if isinstance(first_val, dict):
        slbls: dict[int, str] = {}
        for subj_lbl in state_labels.values():
            for k, lbl in subj_lbl.items():
                slbls.setdefault(int(k), lbl)
    else:
        slbls = {int(k): v for k, v in state_labels.items()}

    ilds = sorted(df_pd[ild_col].dropna().unique())
    panel_w = max(4, 0.4 * len(ilds))

    # ── K-panel grid ──────────────────────────────────────────────────────────
    _all_subjects = list(df_pd[subj_col].unique()) if subj_col in df_pd.columns else []

    # Pre-compute per-state smooth sigmoid curves if arrays_store provided
    _smooth_by_k: dict[int, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    if arrays_store is not None:
        _test_W = next(
            (arrays_store[s].get("emission_weights")
             for s in _all_subjects if s in arrays_store), None
        )
        _K_fit = int(np.asarray(_test_W).shape[0]) if _test_W is not None else 1
        _smooth_single = _mean_glm_curve(arrays_store, _all_subjects, X_cols,
                                         ild_max=ild_max, state_k=None)
        for k in range(K):
            if _K_fit == 1:
                _smooth_by_k[k] = _smooth_single
            else:
                _smooth_by_k[k] = _mean_glm_curve(
                    arrays_store, _all_subjects, X_cols,
                    ild_max=ild_max, state_k=k,
                )
    else:
        for k in range(K):
            _smooth_by_k[k] = None

    fig, axes = plt.subplots(
        1, K, figsize=(panel_w * K, 4), sharey=True, squeeze=False
    )

    for k, ax in enumerate(axes[0]):
        lbl   = slbls.get(k, f"State {k}")
        color = _state_color(lbl, k)
        sub   = df_pd[df_pd["_state_k"] == k]
        _psych_state_panel(ax, sub, ild_col, choice_col, pred_col, subj_col,
                           color=color, label=lbl,
                           smooth_curve=_smooth_by_k.get(k))
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("ILD (dB)")
        ax.set_title(lbl)
        if k == 0:
            ax.set_ylabel("P(rightward Choice)")
        else:
            ax.set_ylabel("")

    fig.suptitle(model_name, y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, None


# Alias to match the plots.py API used in analysis notebooks
plot_categorical_performance_by_state = plot_categorical_performance_all_by_state
