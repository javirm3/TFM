"""
two_afc.py
──────────
Plotting utilities for 2-AFC (binary) GLM-HMM results.

This is the task-owned 2AFC plotting module exposed via
``TaskAdapter.get_plots()``. Its public API mirrors the task plot interface
used by the analysis notebooks.

High-level functions:
  - plot_emission_weights
  - plot_posterior_probs
  - plot_state_accuracy
  - plot_session_trajectories
  - plot_state_occupancy
  - plot_psychometric_all        (≡ plot_categorical_performance_all)
  - plot_psychometric_by_state   (≡ plot_categorical_performance_by_state)
  - plot_regressor_psychometric_by_state
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
from glmhmmt.plots_common import (
    plot_state_accuracy as _plot_state_accuracy_common,
    plot_session_trajectories as _plot_session_trajectories_common,
    plot_state_occupancy as _plot_state_occupancy_common,
    plot_session_deepdive as _plot_session_deepdive_common,
)
from glmhmmt.model_plots import plot_transition_weights

# ── default state colour palette ──────────────────────────────────────────────
_DEFAULT_COLORS = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]

_LABEL_RANK = {
    "Engaged": 0,
    "Disengaged": 1,
    "Biased L": 1,
    "Biased R": 2,
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
        biased_left = others[int(np.argmin([bias_w[k, 0] for k in others]))]
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

    fig, axes = plt.subplots(1, C_m1, figsize=figsize or (max(5, 0.7 * M) * C_m1, 3.5), sharey=True)
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
    """Lineplot and Boxplot of weights per state across N subjects with statistical annotations."""
    from scipy.stats import ttest_rel
    import itertools

    W = np.asarray(all_weights)
    if W.ndim == 3:
        W = W[:, :, None, :]
    N, K, C_m1, M = W.shape
    W_avg = W.mean(axis=2)

    labels = list(state_labels) if state_labels else _default_labels(K, C_m1 + 1)
    colors = _state_colors(K)
    x = np.arange(M)

    # Convert to DataFrame for seaborn
    records = []
    for n in range(N):
        for k in range(K):
            for m in range(M):
                records.append(
                    {
                        "Subject": n,
                        "State": labels[k],
                        "Feature": feature_names[m],
                        "Weight": W_avg[n, k, m],
                        "StateIdx": k,
                        "FeatureIdx": m,
                    }
                )
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=figsize or (8, 4), sharex=True)
    ax_line, ax_box = axes

    sns.lineplot( data=df, x="Feature", y="Weight", hue="State", palette=colors[:K], ax=ax_line, markers=True, marker="o", markersize=8, markeredgewidth=0, alpha=0.85, errorbar="se", legend=False,)
    ax_line.axhline(0, color="k", lw=0.8, ls="--")
    ax_line.set_ylabel("Weight")
    ax_line.set_xlabel("")
    ax_line.set_title(title)

    sns.boxplot(
        data=df,
        x="Feature",
        y="Weight",
        hue="State",
        palette=colors[:K],
        ax=ax_box,
        width=0.8,
        showfliers=False,
        boxprops={"alpha": 0.7},
    )
    sns.stripplot(
        data=df,
        x="Feature",
        y="Weight",
        hue="State",
        palette=colors[:K],
        ax=ax_box,
        dodge=True,
        alpha=0.5,
        zorder=1,
        legend=False,
    )
    ax_box.axhline(0, color="k", lw=0.8, ls="--")

    def get_star(pval):
        if pval < 0.001:
            return "***"
        elif pval < 0.01:
            return "**"
        elif pval < 0.05:
            return "*"
        return "ns"

    y_range = df["Weight"].max() - df["Weight"].min()
    if y_range == 0:
        y_range = 1

    state_pairs = list(itertools.combinations(range(K), 2))
    hue_width = 0.8 / K

    for m in range(M):
        y_max = df[df["FeatureIdx"] == m]["Weight"].max()
        y_offset_step = y_range * 0.05
        current_y_offset = y_max + y_offset_step

        for p1, p2 in state_pairs:
            w1 = W_avg[:, p1, m]
            w2 = W_avg[:, p2, m]

            try:
                stat, pval = ttest_rel(w1, w2)
                star = get_star(pval)
            except Exception:
                star = ""

            if star:
                offset_1 = (p1 - (K - 1) / 2) * hue_width
                offset_2 = (p2 - (K - 1) / 2) * hue_width
                x1 = m + offset_1
                x2 = m + offset_2

                h = y_range * 0.02
                ax_box.plot(
                    [x1, x1, x2, x2],
                    [current_y_offset, current_y_offset + h, current_y_offset + h, current_y_offset],
                    lw=1,
                    c="k",
                )
                ax_box.text((x1 + x2) / 2, current_y_offset + h, star, ha="center", va="bottom", color="k")
                current_y_offset += y_offset_step * 1.5

    ax_box.set_xticks(range(M))
    ax_box.set_xticklabels(feature_names, rotation=45, ha="right")
    ax_box.set_ylabel("Weight")

    handles, labels_lgd = ax_box.get_legend_handles_labels()
    if len(handles) >= K:
        ax_box.legend(handles[:K], labels_lgd[:K], frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        ax_box.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

    sns.despine(fig=fig)
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
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color="w" if A[i, j] < 0.5 else "k", fontsize=9)
    ticks = range(K)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
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
    sns.boxplot(
        x="State",
        y="P(stay)",
        data=df_melt,
        showfliers=False,
        showcaps=False,
        fill=False,
        palette=dict(zip(labels, _state_colors(K))),
        ax=ax,
    )
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
        P = P[trial_range[0] : trial_range[1]]
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
    sns.boxplot(
        x="State",
        y="Occupancy",
        data=df_melt,
        showfliers=False,
        showcaps=False,
        fill=False,
        palette=dict(zip(labels, colors)),
        ax=ax,
    )
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
    n_arr = np.array(n_trials, dtype=float)
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
    sns.boxplot(x="Model", y="LL", data=df_ll, ax=ax, showfliers=False, showcaps=False, fill=False, color=color)
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
    diffs = [norms[ns_list[i + 1]] - norms[ns_list[i]] for i in range(len(ns_list) - 1)]
    labels = [f"{ns_list[i + 1]}s-{ns_list[i]}s" for i in range(len(ns_list) - 1)]

    df = pd.DataFrame(
        {
            "ΔLL": np.concatenate(diffs),
            "comparison": np.repeat(labels, [len(d) for d in diffs]),
        }
    )

    fig, ax = plt.subplots(figsize=figsize or (2.5 * len(diffs), 3.5))
    sns.stripplot(x="comparison", y="ΔLL", data=df, color="k", alpha=0.5, jitter=True, ax=ax)
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

_LABELED_ILDS = {-8, 8}


def _resolve_ild_max(
    df: pd.DataFrame,
    ild_col: str,
    ild_max: Optional[float] = None,
) -> float:
    """Return an explicit |ILD| max or infer it from the plotted DataFrame."""
    if ild_max is not None:
        _ild_max = float(ild_max)
        if np.isfinite(_ild_max) and _ild_max > 0:
            return _ild_max

    if ild_col not in df.columns:
        return 1.0

    ild_vals = pd.to_numeric(df[ild_col], errors="coerce")
    ild_arr = np.asarray(ild_vals, dtype=float)
    if ild_arr.size == 0:
        return 1.0

    finite = ild_arr[np.isfinite(ild_arr)]
    if finite.size == 0:
        return 1.0

    inferred = float(np.max(np.abs(finite)))
    return inferred if inferred > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# GLM grid evaluation  (smooth sigmoid for psychometric plots)
# ─────────────────────────────────────────────────────────────────────────────


def eval_glm_on_ild_grid(
    weights: np.ndarray,
    X_cols: Sequence[str],
    ild_max: float,
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
        n_grid:      Number of ILD points in the evaluation grid.
        lapse_rates: ``[gamma_L, gamma_R]`` lapse parameters from fitting.
                     *None* means no lapse correction.
        X_data:      ``(T, M)`` actual feature matrix used during fitting.
                     When provided, partial-dependence mode is used.

    Returns:
        ild_grid : ``(n_grid,)`` ILD values in dB.
        p_right  : ``(n_grid,)`` model P(Right) for K=1, or
                   ``(K, n_grid)`` for K>1.
    """
    W = np.asarray(weights, dtype=float)
    if W.ndim == 2:
        W = W[None, ...]  # (1, C-1, M)
    K, _C_m1, M = W.shape

    X_cols_list = list(X_cols)

    # Accept any of these as the stimulus / ILD column
    _STIM_NAMES = {"stim_vals", "stim_d", "ild_norm", "ILD", "ild", "stimulus"}
    ild_idx = next((i for i, n in enumerate(X_cols_list) if n in _STIM_NAMES), None)
    bias_idx = next((i for i, n in enumerate(X_cols_list) if n == "bias"), None)

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
            other_logit = X_base @ w
            stim_contrib = X_base[:, ild_idx] * w[ild_idx]
            base_logit = other_logit - stim_contrib
            for gi, sv in enumerate(ild_norm):
                logit = base_logit + sv * w[ild_idx]
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
            X_grid[:, bias_idx] = 1.0  # bias is always 1

        for k in range(K):
            logit = X_grid @ W[k, 0, :]
            # W[k,0,:] is logit for class-0 (LEFT); P(right) = sigmoid(-logit)
            p_left = 1.0 / (1.0 + np.exp(-logit))
            p_right[k] = gL + (1.0 - gL - gR) * (1.0 - p_left)

    if K == 1:
        return ild_grid, p_right[0]
    return ild_grid, p_right


def eval_glm_on_feature_grid(
    weights: np.ndarray,
    X_cols: Sequence[str],
    feature_name: str,
    grid_min: float,
    grid_max: float,
    n_grid: int = 300,
    lapse_rates: Optional[np.ndarray] = None,
    X_data: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a fitted GLM P(right) on a dense grid of any regressor.

    Uses the same partial-dependence idea as :func:`eval_glm_on_ild_grid`:
    sweep one feature over a dense grid while marginalising over the empirical
    distribution of the remaining regressors from ``X_data`` when available.
    """
    W = np.asarray(weights, dtype=float)
    if W.ndim == 2:
        W = W[None, ...]  # (1, C-1, M)
    K, _C_m1, M = W.shape

    X_cols_list = list(X_cols)
    feat_idx = next((i for i, n in enumerate(X_cols_list) if n == feature_name), None)
    bias_idx = next((i for i, n in enumerate(X_cols_list) if n == "bias"), None)
    if feat_idx is None:
        raise KeyError(f"Feature {feature_name!r} not found in X_cols.")

    grid = np.linspace(float(grid_min), float(grid_max), int(n_grid))

    gL, gR = 0.0, 0.0
    if lapse_rates is not None:
        _lr = np.asarray(lapse_rates, dtype=float).ravel()
        if len(_lr) >= 2:
            gL, gR = float(_lr[0]), float(_lr[1])
        elif len(_lr) == 1:
            gL = gR = float(_lr[0])

    p_right = np.zeros((K, len(grid)))

    if X_data is not None:
        X_base = np.asarray(X_data, dtype=float).copy()
        if X_base.ndim != 2 or X_base.shape[1] != M:
            X_base = None
    else:
        X_base = None

    if X_base is not None:
        for k in range(K):
            w = W[k, 0, :]
            other_logit = X_base @ w
            swept_contrib = X_base[:, feat_idx] * w[feat_idx]
            base_logit = other_logit - swept_contrib
            for gi, gv in enumerate(grid):
                logit = base_logit + gv * w[feat_idx]
                p_left = 1.0 / (1.0 + np.exp(-logit))
                p_right[k, gi] = float(np.mean(gL + (1.0 - gL - gR) * (1.0 - p_left)))
    else:
        col_means = np.zeros(M)
        if bias_idx is not None:
            col_means[bias_idx] = 1.0
        X_grid = np.tile(col_means, (len(grid), 1))
        X_grid[:, feat_idx] = grid
        if bias_idx is not None:
            X_grid[:, bias_idx] = 1.0
        for k in range(K):
            logit = X_grid @ W[k, 0, :]
            p_left = 1.0 / (1.0 + np.exp(-logit))
            p_right[k] = gL + (1.0 - gL - gR) * (1.0 - p_left)

    if K == 1:
        return grid, p_right[0]
    return grid, p_right


def _mean_glm_curve(
    arrays_store: dict,
    subjects: Sequence[str],
    X_cols: Optional[Sequence[str]],
    ild_max: float,
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
                X_data = None

        # When computing the curve for a specific state, restrict partial-
        # dependence to trials assigned to that state so that history
        # covariates are marginalised over its empirical distribution only.
        if X_data is not None and state_k is not None:
            _gamma = arrays_store[subj].get("smoothed_probs")
            if _gamma is not None:
                _map_k = np.argmax(np.asarray(_gamma), axis=1)
                _mask = _map_k == state_k
                if _mask.sum() > 0:
                    X_data = X_data[_mask]
                # if the state has no trials, fall through with full X_data

        try:
            _lr = arrays_store[subj].get("lapse_rates")
            if _lr is not None:
                _lr = np.asarray(_lr, dtype=float).ravel()
                if not np.any(_lr > 0):
                    _lr = None
            ig, pg = eval_glm_on_ild_grid(W, cols, ild_max=ild_max, lapse_rates=_lr, X_data=X_data)
        except Exception:
            continue

        # pg is (n_grid,) for K=1, or (K, n_grid) for K>1
        if pg.ndim == 2 and state_k is not None:
            pg = pg[state_k]
        elif pg.ndim == 2:
            # For the marginal GLM-HMM psychometric, average state-specific
            # curves using empirical state occupancy rather than equal weights.
            _gamma = arrays_store[subj].get("smoothed_probs")
            if _gamma is not None:
                _w = np.asarray(_gamma, dtype=float).mean(axis=0)
                _w_sum = float(_w.sum())
                if _w_sum > 0:
                    _w = _w / _w_sum
                    pg = np.average(pg, axis=0, weights=_w)
                else:
                    pg = pg.mean(axis=0)
            else:
                pg = pg.mean(axis=0)

        all_p.append(pg)
        ild_g = ig

    if not all_p or ild_g is None:
        return None
    return ild_g, np.mean(all_p, axis=0)


def _subject_glm_curves(
    arrays_store: dict,
    subjects: Sequence[str],
    X_cols: Optional[Sequence[str]],
    ild_max: float,
    state_k: Optional[int] = None,
) -> dict:
    """Return {subject: (ild_grid, p_right)} for per-subject psychometric backgrounds."""
    out: dict = {}
    for subj in subjects:
        curve = _mean_glm_curve(arrays_store, [subj], X_cols, ild_max=ild_max, state_k=state_k)
        if curve is not None:
            out[subj] = curve
    return out


def _mean_glm_feature_curve(
    arrays_store: dict,
    subjects: Sequence[str],
    X_cols: Optional[Sequence[str]],
    feature_name: str,
    grid_min: float,
    grid_max: float,
    state_k: Optional[int] = None,
    n_grid: int = 300,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Average a feature partial-dependence curve over subjects."""
    all_p: list[np.ndarray] = []
    feat_g: Optional[np.ndarray] = None

    for subj in subjects:
        if subj not in arrays_store:
            continue
        W = arrays_store[subj].get("emission_weights")
        if W is None:
            continue
        cols = X_cols
        if cols is None:
            raw = arrays_store[subj].get("X_cols")
            if raw is None:
                continue
            cols = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, str) else [raw]
        if feature_name not in cols:
            continue

        X_data = arrays_store[subj].get("X")
        if X_data is not None:
            X_data = np.asarray(X_data, dtype=float)
            if X_data.ndim != 2 or X_data.shape[1] != np.asarray(W).shape[-1]:
                X_data = None

        if X_data is not None and state_k is not None:
            _gamma = arrays_store[subj].get("smoothed_probs")
            if _gamma is not None:
                _map_k = np.argmax(np.asarray(_gamma), axis=1)
                _mask = _map_k == state_k
                if _mask.sum() > 0:
                    X_data = X_data[_mask]

        try:
            _lr = arrays_store[subj].get("lapse_rates")
            if _lr is not None:
                _lr = np.asarray(_lr, dtype=float).ravel()
                if not np.any(_lr > 0):
                    _lr = None
            fg, pg = eval_glm_on_feature_grid(
                W,
                cols,
                feature_name=feature_name,
                grid_min=grid_min,
                grid_max=grid_max,
                n_grid=n_grid,
                lapse_rates=_lr,
                X_data=X_data,
            )
        except Exception:
            continue

        if pg.ndim == 2 and state_k is not None:
            pg = pg[state_k]
        elif pg.ndim == 2:
            _gamma = arrays_store[subj].get("smoothed_probs")
            if _gamma is not None:
                _w = np.asarray(_gamma, dtype=float).mean(axis=0)
                _w_sum = float(_w.sum())
                if _w_sum > 0:
                    _w = _w / _w_sum
                    pg = np.average(pg, axis=0, weights=_w)
                else:
                    pg = pg.mean(axis=0)
            else:
                pg = pg.mean(axis=0)

        all_p.append(pg)
        feat_g = fg

    if not all_p or feat_g is None:
        return None
    return feat_g, np.mean(all_p, axis=0)


def _subject_glm_feature_curves(
    arrays_store: dict,
    subjects: Sequence[str],
    X_cols: Optional[Sequence[str]],
    feature_name: str,
    grid_min: float,
    grid_max: float,
    state_k: Optional[int] = None,
    n_grid: int = 300,
) -> dict:
    """Return {subject: (feature_grid, p_right)} for per-subject feature backgrounds."""
    out: dict = {}
    for subj in subjects:
        curve = _mean_glm_feature_curve(
            arrays_store,
            [subj],
            X_cols,
            feature_name=feature_name,
            grid_min=grid_min,
            grid_max=grid_max,
            state_k=state_k,
            n_grid=n_grid,
        )
        if curve is not None:
            out[subj] = curve
    return out


def _feature_label(feature_name: str) -> str:
    labels = {
        "at_choice": "Action trace",
        "at_error": "Error trace",
        "at_correct": "Correct trace",
        "reward_trace": "Reward trace",
        "stim_vals": "Stimulus",
        "bias": "Bias",
        "wsls": "WSLS",
    }
    return labels.get(feature_name, feature_name.replace("_", " ").title())


def _binned_feature_summary(
    df: pd.DataFrame,
    feature_col: str,
    choice_col: str,
    subj_col: str,
    n_bins: int = 9,
) -> Optional[Tuple[pd.DataFrame, list[float]]]:
    needed = [feature_col, choice_col, subj_col]
    d = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    if d.empty:
        return None

    d[feature_col] = pd.to_numeric(d[feature_col], errors="coerce")
    d[choice_col] = pd.to_numeric(d[choice_col], errors="coerce")
    d = d.dropna(subset=[feature_col, choice_col])
    if d.empty:
        return None

    unique_vals = np.sort(d[feature_col].unique())
    if len(unique_vals) <= max(6, n_bins):
        d["_x_bin"] = d[feature_col]
        centers = (
            d.groupby("_x_bin", observed=True)[feature_col]
            .median()
            .rename("center")
            .reset_index()
            .sort_values("center")
        )
    else:
        d["_x_bin"] = pd.qcut(d[feature_col], q=n_bins, duplicates="drop")
        centers = (
            d.groupby("_x_bin", observed=True)[feature_col]
            .median()
            .rename("center")
            .reset_index()
            .sort_values("center")
        )

    subj = (
        d.groupby(["_x_bin", subj_col], observed=True)
        .agg(data_mean=(choice_col, "mean"))
        .reset_index()
        .merge(centers, on="_x_bin", how="left")
    )
    if subj.empty:
        return None

    agg = (
        subj.groupby("_x_bin", observed=True)
        .agg(
            x=("center", "median"),
            md=("data_mean", "mean"),
            sd=("data_mean", "std"),
            nd=("data_mean", "count"),
        )
        .reset_index(drop=True)
        .sort_values("x")
    )
    agg["sd"] = agg["sd"].fillna(0.0)
    agg["sem"] = agg["sd"] / np.sqrt(agg["nd"].clip(lower=1))
    return subj, agg["x"].tolist()


def _sparse_ild_labels(ilds: list) -> list:
    """Return tick labels that show only the extreme values and ±8; rest are empty."""
    lo, hi = min(ilds), max(ilds)
    labeled = _LABELED_ILDS | {lo, hi}
    return [str(int(v)) if float(v) in labeled else "" for v in ilds]


def _resolve_ild_ticks(
    ilds: Sequence,
    tick_ilds: Optional[Sequence[float]] = None,
) -> list[float]:
    vals = tick_ilds if tick_ilds is not None else ilds
    ticks = sorted({float(v) for v in vals if pd.notna(v)})
    if ticks:
        return ticks
    return sorted({float(v) for v in ilds if pd.notna(v)})


def _apply_ild_axis_ticks(ax: plt.Axes, xticks: Sequence[float]) -> None:
    xticks = np.asarray(xticks, dtype=float)
    ax.set_xticks(xticks, labels=_sparse_ild_labels(list(xticks)))
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        top=False,
        direction="out",
        length=7,
        width=1.1,
        color="#111827",
        labelcolor="#111827",
        pad=4,
    )
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.spines["bottom"].set_color("#111827")


def _resolve_plot_col(df: pd.DataFrame, preferred: str, candidates: Sequence[str]) -> str:
    if preferred in df.columns:
        return preferred
    for cand in candidates:
        if cand in df.columns:
            return cand
    raise KeyError(f"None of the candidate columns exist: {[preferred, *candidates]}")


def _psych_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    ild_col: str = "ILD",
    choice_col: str = "response",
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    title: str = "",
    xlabel: str = "ILD (dB)",
    ylabel: Optional[str] = None,
    color: str = "k",
    smooth_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    background_style: str = "data",
    subject_curves: Optional[dict] = None,
    tick_ilds: Optional[Sequence[float]] = None,
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

    choice_col = _resolve_plot_col(df, choice_col, ("response", "Choice"))

    subj_agg = (
        df.groupby([subj_col, ild_col], observed=True)
        .agg(data_mean=(choice_col, "mean"), model_mean=(pred_col, "mean"))
        .reset_index()
    )

    ilds = sorted(subj_agg[ild_col].unique())
    xpos = np.array(ilds, dtype=float)
    xticks = np.array(_resolve_ild_ticks(ilds, tick_ilds), dtype=float)

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

    md = agg["md"].values
    sd = agg["sd"].fillna(0).values
    nd = agg["nd"].clip(lower=1).values
    mm = agg["mm"].values
    sem_d = sd / np.sqrt(nd)

    # per-subject background traces: empirical summaries or fitted curves
    if background_style == "data":
        for subj, grp in subj_agg.groupby(subj_col):
            grp_ilds = [i for i in ilds if i in grp[ild_col].values]
            xi = np.array(grp_ilds, dtype=float)
            yi = grp.set_index(ild_col).reindex(grp_ilds)["data_mean"].values
            ax.plot(xi, yi, "-o", color=color, alpha=0.12, lw=1, ms=3, zorder=2)
    elif background_style == "model" and subject_curves is not None:
        for subj, curve in subject_curves.items():
            if curve is None:
                continue
            xi, yi = curve
            ax.plot(xi, yi, "-", color=color, alpha=0.12, lw=1.2, zorder=2)

    # model line: smooth sigmoid over dense ILD grid (if available) else aggregated p_pred
    if smooth_curve is not None:
        ild_g, p_g = smooth_curve
        # Clip to the observed ILD range so the curve doesn't extend beyond
        # the data and compress the visible sigmoid into a near-flat line.
        _x0, _x1 = float(xticks[0]), float(xticks[-1])
        _clip = (ild_g >= _x0) & (ild_g <= _x1)
        ax.plot(ild_g[_clip], p_g[_clip], "-", color="black", lw=2, label="Model", zorder=6)
    else:
        ax.plot(xpos, mm, "-", color="black", lw=2, label="Model", zorder=6)

    # pooled data mean ± SEM
    ax.errorbar(xpos, md, yerr=sem_d, fmt="o", color=color, ecolor=color, elinewidth=1, ms=5, label="Data", zorder=5)

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    _apply_ild_axis_ticks(ax, xticks)
    ax.set_xlim(xticks[0], xticks[-1])
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
    show_subject_traces: bool = True,
    background_style: str = "data",
    subject_curves: Optional[dict] = None,
    weight_col: Optional[str] = None,
    tick_ilds: Optional[Sequence[float]] = None,
) -> Tuple:
    """Draw state-specific psychometric on ax.  Returns (data_h, model_h)."""
    if df_state.empty:
        return None, None

    choice_col = _resolve_plot_col(df_state, choice_col, ("response", "Choice"))
    if weight_col is not None and weight_col in df_state.columns:
        _rows = []
        for (subj, ild), grp in df_state.groupby([subj_col, ild_col], observed=True):
            w = pd.to_numeric(grp[weight_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(grp[choice_col], errors="coerce").to_numpy(dtype=float)
            m = pd.to_numeric(grp[pred_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(w) & np.isfinite(y) & np.isfinite(m) & (w > 0)
            if not np.any(mask):
                continue
            w = w[mask]
            w_sum = float(w.sum())
            if w_sum <= 0:
                continue
            _rows.append(
                {
                    subj_col: subj,
                    ild_col: ild,
                    "data_mean": float(np.dot(y[mask], w) / w_sum),
                    "model_mean": float(np.dot(m[mask], w) / w_sum),
                }
            )
        subj_agg = pd.DataFrame(_rows)
    else:
        subj_agg = (
            df_state.groupby([subj_col, ild_col], observed=True)
            .agg(data_mean=(choice_col, "mean"), model_mean=(pred_col, "mean"))
            .reset_index()
        )
    if subj_agg.empty:
        return None, None
    ilds = sorted(subj_agg[ild_col].unique())
    xpos = np.array(ilds, dtype=float)
    xticks = np.array(_resolve_ild_ticks(ilds, tick_ilds), dtype=float)

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
    md = agg["md"].values
    sd = agg["sd"].fillna(0).values
    nd = agg["nd"].clip(lower=1).values
    mm = agg["mm"].values
    sem_d = sd / np.sqrt(nd)

    if show_subject_traces and background_style == "data":
        # per-subject individual traces (low alpha)
        for subj, grp in subj_agg.groupby(subj_col):
            grp_ilds = [i for i in ilds if i in grp[ild_col].values]
            xi = np.array(grp_ilds, dtype=float)
            yi = grp.set_index(ild_col).reindex(grp_ilds)["data_mean"].values
            ax.plot(xi, yi, "-o", color=color, alpha=0.12, lw=1, ms=3, zorder=2)
    elif show_subject_traces and background_style == "model" and subject_curves is not None:
        for subj, curve in subject_curves.items():
            if curve is None:
                continue
            xi, yi = curve
            ax.plot(xi, yi, "-", color=color, alpha=0.12, lw=1.2, zorder=2)

    data_h = ax.errorbar(
        xpos, md, yerr=sem_d, fmt="o", color=color, ecolor=color, elinewidth=1.2, ms=5, zorder=5, label=label
    )
    # smooth sigmoid model line (if available) else aggregated p_pred
    if smooth_curve is not None:
        ild_g, p_g = smooth_curve
        # Clip the dense grid to the observed ILD range so the sigmoid
        # doesn't extend far beyond the data and compress the visible area.
        _x0, _x1 = float(xticks[0]), float(xticks[-1])
        _clip = (ild_g >= _x0) & (ild_g <= _x1)
        (model_h,) = ax.plot(ild_g[_clip], p_g[_clip], "-", color=color, lw=2.2, zorder=6, label="_nolegend_")
    else:
        (model_h,) = ax.plot(xpos, mm, "-", color=color, lw=2.2, zorder=6, label="_nolegend_")

    ax.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    _apply_ild_axis_ticks(ax, xticks)
    ax.set_xlim(xticks[0], xticks[-1])
    return data_h, model_h


def _regressor_state_panel(
    ax: plt.Axes,
    df_state: pd.DataFrame,
    feature_col: str,
    choice_col: str,
    subj_col: str,
    color: str,
    label: str,
    smooth_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    show_subject_traces: bool = True,
    background_style: str = "data",
    subject_curves: Optional[dict] = None,
    n_bins: int = 9,
) -> Tuple:
    """Draw state-specific P(right) vs arbitrary regressor on ax."""
    if df_state.empty:
        return None, None

    choice_col = _resolve_plot_col(df_state, choice_col, ("response", "Choice"))
    summary = _binned_feature_summary(df_state, feature_col, choice_col, subj_col, n_bins=n_bins)
    if summary is None:
        return None, None
    subj_agg, _x_ticks = summary

    agg = (
        subj_agg.groupby("_x_bin", observed=True)
        .agg(
            x=("center", "median"),
            md=("data_mean", "mean"),
            sd=("data_mean", "std"),
            nd=("data_mean", "count"),
        )
        .reset_index(drop=True)
        .sort_values("x")
    )
    x = agg["x"].to_numpy(dtype=float)
    md = agg["md"].to_numpy(dtype=float)
    sd = agg["sd"].fillna(0.0).to_numpy(dtype=float)
    nd = agg["nd"].clip(lower=1).to_numpy(dtype=float)
    sem_d = sd / np.sqrt(nd)

    if show_subject_traces and background_style == "data":
        for subj, grp in subj_agg.groupby(subj_col):
            grp = grp.sort_values("center")
            ax.plot(
                grp["center"].to_numpy(dtype=float),
                grp["data_mean"].to_numpy(dtype=float),
                "-o",
                color=color,
                alpha=0.15,
                lw=1,
                ms=3,
                zorder=2,
            )
    elif show_subject_traces and background_style == "model" and subject_curves is not None:
        for subj, curve in subject_curves.items():
            if curve is None:
                continue
            xi, yi = curve
            ax.plot(xi, yi, "-", color=color, alpha=0.12, lw=1.2, zorder=2)

    data_h = ax.errorbar(x, md, yerr=sem_d, fmt="o", color=color, ecolor=color, elinewidth=1.2, ms=5, zorder=5, label=label)
    if smooth_curve is not None:
        feat_g, p_g = smooth_curve
        (model_h,) = ax.plot(feat_g, p_g, "-", color=color, lw=2.2, zorder=6, label="_nolegend_")
    else:
        model_h = None

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
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
    Side / stimulus : int  - correct side (0 = left, 1 = right)
    Choice / response : int  - animal's choice (0 = left, 1 = right)
    Hit / performance : int/bool - trial correct (1) or incorrect (0)
    pL               : float - model P(left choice)
    pR               : float - model P(right choice)

    Added / ensured output columns
    ------------------------------
    correct_bool    : bool  - Trial accuracy
    p_pred          : float - model P(right)  → psychometric x-axis
    p_model_correct : float - model P(correct Side)
    stimulus / Side : int   - canonical aliases for correct side
    response / Choice : int - canonical aliases for animal choice

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

        if "stimulus" not in df.columns:
            if "Side" in df.columns:
                df = df.with_columns(pl.col("Side").cast(pl.Int64).alias("stimulus"))
            else:
                raise ValueError("No 'Side' or 'stimulus' column found.")
        if "Side" not in df.columns:
            df = df.with_columns(pl.col("stimulus").cast(pl.Int64).alias("Side"))

        if "response" not in df.columns:
            if "Choice" in df.columns:
                df = df.with_columns(pl.col("Choice").cast(pl.Int64).alias("response"))
            else:
                raise ValueError("No 'Choice' or 'response' column found.")
        if "Choice" not in df.columns:
            df = df.with_columns(pl.col("response").cast(pl.Int64).alias("Choice"))

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
            pl.when(pl.col("stimulus") == 0).then(pl.col("pL")).otherwise(pl.col("pR")).alias("p_model_correct"),
        )

        return df

    else:
        # pandas path
        df = df_pred.copy()

        if "stimulus" not in df.columns:
            if "Side" in df.columns:
                df["stimulus"] = df["Side"].astype(int)
            else:
                raise ValueError("No 'Side' or 'stimulus' column found.")
        if "Side" not in df.columns:
            df["Side"] = df["stimulus"].astype(int)

        if "response" not in df.columns:
            if "Choice" in df.columns:
                df["response"] = df["Choice"].astype(int)
            else:
                raise ValueError("No 'Choice' or 'response' column found.")
        if "Choice" not in df.columns:
            df["Choice"] = df["response"].astype(int)

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
        df["p_model_correct"] = df.apply(lambda row: row["pL"] if row["stimulus"] == 0 else row["pR"], axis=1)

        return df


# ─────────────────────────────────────────────────────────────────────────────
# High-level API used by the task plot facade
# ─────────────────────────────────────────────────────────────────────────────


def plot_emission_weights(
    views: dict,
    K: int,
    save_path=None,
) -> Tuple[plt.Figure, plt.Figure]:
    """Emission weights: per-subject bar charts + multi-subject error-bar figure.

    Mirrors plots.plot_emission_weights.

    Parameters
    ----------
    views     : {subj: SubjectFitView} as produced by build_views
    K         : number of states
    save_path : optional Path - per-subject figure saved there if provided

    Returns
    -------
    fig_single, fig_multi
    """
    valid_subjs = list(views.keys())
    feat_names = next(iter(views.values())).feat_names if views else []
    _pal, _hue_order = _build_state_palette({s: v.state_name_by_idx for s, v in views.items()})

    if not valid_subjs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig, fig

    # ── per-subject panel ─────────────────────────────────────────────────────
    n_cols = min(3, len(valid_subjs))
    n_rows = int(np.ceil(len(valid_subjs) / n_cols))
    M = len(feat_names) or 1
    fig_single, axes_s = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(5, 0.7 * M) * n_cols, 3.5 * n_rows),
        sharey=True,
        squeeze=False,
    )

    all_w = []
    for idx, subj in enumerate(valid_subjs):
        ax = axes_s[idx // n_cols][idx % n_cols]
        W_raw = -views[subj].emission_weights  # flip sign: raw W = logit(Left); -W = logit(Right) (intuitive)
        slbls = views[subj].state_name_by_idx
        # reorder by rank so color index 0 = Engaged = green, 1 = Disengaged = orange
        rank_order = views[subj].state_idx_order  # [Engaged_k, Disengaged_k, ...]
        W = W_raw[rank_order]
        lbls = [slbls.get(k, f"S{k}") for k in rank_order]
        fn_subj = views[subj].feat_names or feat_names
        plot_weights(W, fn_subj, state_labels=lbls, title=f"Subject {subj}", ax=ax)
        all_w.append(np.asarray(W))  # rank-reordered so position k=0 = Engaged across all subjects

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
        fn = feat_names or (views[valid_subjs[0]].feat_names or [])
        fig_multi = plot_weights_boxplot(
            W_stack, fn, state_labels=_hue_order[:K], title=f"Emission weights - all subjects  (K={K})"
        )
    else:
        fig_multi = fig_single

    return fig_single, fig_multi


def plot_posterior_probs(
    views: dict,
    K: int,
    t0: int = 0,
    t1: int = 199,
) -> plt.Figure:
    """Stacked-area posterior state probability plot.

    Mirrors plots.plot_posterior_probs.

    Returns
    -------
    fig
    """
    _selected = list(views.keys())
    if not _selected:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    colors = _state_colors(K)
    fig, axes = plt.subplots(len(_selected), 1, figsize=(14, 3 * len(_selected)), squeeze=False)

    for i, subj in enumerate(_selected):
        ax = axes[i, 0]
        P = np.asarray(views[subj].smoothed_probs)
        P_sub = P[t0 : min(t1, len(P))]
        T_sub = P_sub.shape[0]

        ax.stackplot(np.arange(T_sub), P_sub.T, colors=colors[:K], alpha=0.8)
        slbls = views[subj].state_name_by_idx
        legend_patches = [plt.matplotlib.patches.Patch(color=colors[k], label=slbls.get(k, f"S{k}")) for k in range(K)]
        ax.legend(handles=legend_patches, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
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
    views: dict,
    trial_df,
    thresh: float = 0.5,
    session_col: str = "Session",
    sort_col: str = "Trial",
    performance_col: str = "correct_bool",
    stim_col: str = "ILD",
    **kwargs,
) -> Tuple[plt.Figure, pd.DataFrame]:
    return _plot_state_accuracy_common(
        views,
        trial_df,
        thresh=thresh,
        performance_candidates=(performance_col, "performance"),
        stim_candidates=(stim_col, "stimulus", "ILD"),
        stim_label="ILD≠0",
    )


def plot_session_trajectories(
    views: dict,
    trial_df,
    session_col: str = "Session",
    sort_col: str = "Trial",
    **kwargs,
) -> plt.Figure:
    return _plot_session_trajectories_common(
        views,
        trial_df,
        session_col=session_col,
    )


def plot_state_occupancy(
    views: dict,
    trial_df,
    session_col: str = "Session",
    sort_col: str = "Trial",
    **kwargs,
) -> plt.Figure:
    return _plot_state_occupancy_common(
        views,
        trial_df,
        session_col=session_col,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session deep-dive
# ─────────────────────────────────────────────────────────────────────────────


def plot_session_deepdive(
    views: dict,
    trial_df,
    subj: str,
    sess,
    session_col: str = "Session",
    sort_col: str = "Trial",
    **kwargs,
) -> plt.Figure:
    return _plot_session_deepdive_common(
        views,
        trial_df,
        subj,
        sess,
        session_col=session_col,
        performance_candidates=("correct_bool", "performance"),
        stim_candidates=("ILD", "stimulus"),
        response_candidates=("response", "Choice"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Psychometric performance helpers
# ─────────────────────────────────────────────────────────────────────────────


def plot_categorical_performance_all(
    df,
    model_name: str,
    ild_col: str = "ILD",
    choice_col: str = "response",
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    cond_col: str = "condition",
    exp_col: str = "experiment",
    views: Optional[dict] = None,
    X_cols: Optional[Sequence[str]] = None,
    ild_max: Optional[float] = None,
    background_style: str = "data",
) -> plt.Figure:
    """Overall psychometric + by-condition + by-experiment panels.

    2AFC equivalent of plots.plot_categorical_performance_all.

    Panels
    ------
    a) Overall      - P(right) vs ILD, all trials pooled
    b) By condition - separate curves per rest / saline / drug
                      (skipped if 'condition' column absent)
    c) By experiment - separate curves per experiment batch

    Parameters
    ----------
    df         : Polars or pandas DataFrame with Trial-level predictions.
                 Must contain: ILD, response/Choice (0/1), p_pred, subject.
                 The default model overlay is ``p_pred`` because this figure
                 shows psychometrics, i.e. P(rightward choice), not accuracy.
    model_name : String for figure suptitle.
    ild_max    : Optional explicit normalisation scale. When omitted, the
                 maximum absolute ILD in ``df[ild_col]`` is used.

    Returns
    -------
    fig
    """
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy()
    ild_max = _resolve_ild_max(df_pd, ild_col, ild_max)
    ild_ticks = (
        sorted(pd.to_numeric(df_pd[ild_col], errors="coerce").dropna().unique()) if ild_col in df_pd.columns else []
    )

    has_cond = cond_col in df_pd.columns
    has_exp = exp_col in df_pd.columns
    n_panels = 1 + int(has_cond) + int(has_exp)

    # Pre-compute smooth GLM sigmoid averaged over all subjects
    _all_subjects = list(df_pd[subj_col].unique()) if subj_col in df_pd.columns else []

    # Build arrays_store-compatible dict for _mean_glm_curve.
    # Reorder axes so that state index == rank (0=Engaged, …) for consistency.
    def _rank_ordered_as(v):
        _order = sorted(v.state_rank_by_idx, key=lambda ki: v.state_rank_by_idx[ki])
        return {
            "emission_weights": v.emission_weights[_order],
            "X_cols": v.feat_names,
            "X": v.X,
            "smoothed_probs": v.smoothed_probs[:, _order],
            "lapse_rates": v.lapse_rates,
        }

    _as = {s: _rank_ordered_as(v) for s, v in views.items()} if views is not None else None
    _smooth_all = _mean_glm_curve(_as, _all_subjects, X_cols, ild_max=ild_max) if _as is not None else None
    _subject_curves_all = (
        _subject_glm_curves(_as, _all_subjects, X_cols, ild_max=ild_max) if _as is not None and background_style == "model" else None
    )

    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 4), sharey=True)
    axes = np.atleast_1d(axes)
    ax_idx = 0

    # a) Overall
    _psych_panel(
        axes[ax_idx],
        df_pd,
        ild_col=ild_col,
        choice_col=choice_col,
        pred_col=pred_col,
        subj_col=subj_col,
        title="a) Overall psychometric",
        xlabel="ILD (dB)",
        ylabel="P(Right)",
        color="#2b7bba",
        smooth_curve=_smooth_all,
        background_style=background_style,
        subject_curves=_subject_curves_all,
        tick_ilds=ild_ticks,
    )
    ax_idx += 1

    # b) By condition
    if has_cond:
        conds = sorted(df_pd[cond_col].dropna().unique())
        cond_colors = {"rest": "#444444", "saline": "#1f77b4", "drug": "#d62728"}
        for cond in conds:
            _cond_subjs = list(df_pd[df_pd[cond_col] == cond][subj_col].unique())
            _smooth_cond = _mean_glm_curve(_as, _cond_subjs, X_cols, ild_max=ild_max) if _as is not None else None
            _subject_curves_cond = (
                _subject_glm_curves(_as, _cond_subjs, X_cols, ild_max=ild_max)
                if _as is not None and background_style == "model"
                else None
            )
            _psych_panel(
                axes[ax_idx],
                df_pd[df_pd[cond_col] == cond],
                ild_col=ild_col,
                choice_col=choice_col,
                pred_col=pred_col,
                subj_col=subj_col,
                title=f"b) {cond}",
                xlabel="ILD (dB)",
                color=cond_colors.get(cond, "k"),
                smooth_curve=_smooth_cond,
                background_style=background_style,
                subject_curves=_subject_curves_cond,
                tick_ilds=ild_ticks,
            )
        ax_idx += 1

    # c) By experiment
    if has_exp:
        exps = sorted(df_pd[exp_col].dropna().unique())
        exp_palette = sns.color_palette("Set2", len(exps))
        for ei, exp in enumerate(exps):
            _exp_subjs = list(df_pd[df_pd[exp_col] == exp][subj_col].unique())
            _smooth_exp = _mean_glm_curve(_as, _exp_subjs, X_cols, ild_max=ild_max) if _as is not None else None
            _subject_curves_exp = (
                _subject_glm_curves(_as, _exp_subjs, X_cols, ild_max=ild_max)
                if _as is not None and background_style == "model"
                else None
            )
            _psych_panel(
                axes[ax_idx],
                df_pd[df_pd[exp_col] == exp],
                ild_col=ild_col,
                choice_col=choice_col,
                pred_col=pred_col,
                subj_col=subj_col,
                title=f"c) {exp}",
                xlabel="ILD (dB)",
                color=exp_palette[ei],
                smooth_curve=_smooth_exp,
                background_style=background_style,
                subject_curves=_subject_curves_exp,
                tick_ilds=ild_ticks,
            )

    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
    sns.despine(fig=fig)
    fig.suptitle(model_name, y=1.02)
    fig.tight_layout()
    return fig, None


def plot_categorical_performance_all_by_state(
    df,
    views: dict,
    model_name: str,
    ild_col: str = "ILD",
    choice_col: str = "response",
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    X_cols: Optional[Sequence[str]] = None,
    ild_max: Optional[float] = None,
    background_style: str = "data",
) -> plt.Figure:
    """Per-state psychometric grid (K panels, one per state).

    2AFC equivalent of plots.plot_categorical_performance_by_state.

    Each state gets its own panel showing P(right) vs ILD; data (markers) and
    model prediction (lines) are drawn in the state's colour.

    Parameters
    ----------
    df     : Trial-level DataFrame (polars or pandas). Must contain a
             ``state_rank`` column (rank 0 = Engaged) produced by
             :func:`~glmhmmt.postprocess.build_trial_df`.
    views  : {subj: SubjectFitView} as produced by build_views.
    model_name : string used as figure suptitle.
    ild_max : Optional explicit normalisation scale. When omitted, the
              maximum absolute ILD in ``df[ild_col]`` is used.

    Returns
    -------
    (fig, None)
    """
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas().reset_index(drop=True)
    else:
        df_pd = df.reset_index(drop=True)
    ild_max = _resolve_ild_max(df_pd, ild_col, ild_max)
    ild_ticks = (
        sorted(pd.to_numeric(df_pd[ild_col], errors="coerce").dropna().unique()) if ild_col in df_pd.columns else []
    )

    K = next(iter(views.values())).K if views else 2

    # State assignment from trial_df (state_rank: 0=Engaged, 1=Disengaged, …)
    if "state_rank" in df_pd.columns:
        _arr = df_pd["state_rank"].to_numpy().astype(int)
    elif "_state_k" in df_pd.columns:
        _arr = df_pd["_state_k"].to_numpy().astype(int)
    else:
        raise ValueError("df must contain a 'state_rank' column (output of build_trial_df)")

    df_pd = df_pd.copy()
    df_pd["_state_k"] = _arr

    # Resolve labels: {rank: label} merged across all subjects
    slbls: dict[int, str] = {}
    for v in views.values():
        for k, lbl in v.state_name_by_idx.items():
            rank = v.state_rank_by_idx.get(int(k), int(k))
            slbls.setdefault(rank, lbl)

    # Build arrays_store-compatible dict for _mean_glm_curve.
    # Reorder axes so that state index == rank (0=Engaged, 1=Disengaged, …)
    # so that state_k=0 means Engaged for every subject regardless of fit order.
    def _rank_ordered_as(v):
        _order = sorted(v.state_rank_by_idx, key=lambda ki: v.state_rank_by_idx[ki])
        return {
            "emission_weights": v.emission_weights[_order],
            "X_cols": v.feat_names,
            "X": v.X,
            "smoothed_probs": v.smoothed_probs[:, _order],
        }

    _as = {s: _rank_ordered_as(v) for s, v in views.items()}

    ilds = sorted(df_pd[ild_col].dropna().unique())
    panel_w = max(2, 0.3 * len(ilds))

    # ── K-panel grid ──────────────────────────────────────────────────────────
    _all_subjects = list(df_pd[subj_col].unique()) if subj_col in df_pd.columns else []

    # Pre-compute per-state smooth sigmoid curves from views
    _smooth_by_k: dict[int, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    _test_W = next((v.emission_weights for v in views.values()), None)
    _K_fit = int(np.asarray(_test_W).shape[0]) if _test_W is not None else 1
    _smooth_single = _mean_glm_curve(_as, _all_subjects, X_cols, ild_max=ild_max, state_k=None)
    for k in range(K):
        if _K_fit == 1:
            _smooth_by_k[k] = _smooth_single
        else:
            _smooth_by_k[k] = _mean_glm_curve(_as, _all_subjects, X_cols, ild_max=ild_max, state_k=k)
    _subject_curves_by_k = (
        {k: _subject_glm_curves(_as, _all_subjects, X_cols, ild_max=ild_max, state_k=None if _K_fit == 1 else k) for k in range(K)}
        if background_style == "model"
        else {}
    )

    _include_overlay = K > 1
    _n_panels = K + int(_include_overlay)
    fig, axes = plt.subplots(1, _n_panels, figsize=(panel_w * _n_panels, 4), sharey=True)
    axes = np.atleast_1d(axes)

    if _include_overlay:
        _ax_overlay = axes[0]
        for k in range(K):
            lbl = slbls.get(k, f"State {k}")
            color = _state_color(lbl, k)
            _df_state = df_pd[df_pd["_state_k"] == k]
            _psych_state_panel(
                _ax_overlay,
                _df_state,
                ild_col,
                choice_col,
                pred_col,
                subj_col,
                color=color,
                label=lbl,
                smooth_curve=_smooth_by_k[k],
                show_subject_traces=False,
                background_style=background_style,
                subject_curves=_subject_curves_by_k.get(k),
                tick_ilds=ild_ticks,
            )
        _ax_overlay.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        _ax_overlay.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
        _ax_overlay.set_ylim(0, 1)
        _ax_overlay.set_yticks([0, 0.5, 1])
        _ax_overlay.set_xlabel("ILD (dB)")
        _ax_overlay.set_ylabel("P(Right)")
        _ax_overlay.set_title("All states")
        _ax_overlay.legend(frameon=False, fontsize=8)

    for k, ax in enumerate(axes[int(_include_overlay) :]):
        lbl = slbls.get(k, f"State {k}")
        color = _state_color(lbl, k)
        _df_state = df_pd[df_pd["_state_k"] == k]
        _psych_state_panel(
            ax,
            _df_state,
            ild_col,
            choice_col,
            pred_col,
            subj_col,
            color=color,
            label=lbl,
            smooth_curve=_smooth_by_k[k],
            background_style=background_style,
            subject_curves=_subject_curves_by_k.get(k),
            tick_ilds=ild_ticks,
        )
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("ILD (dB)")
        ax.set_title(lbl)
        if k == 0:
            ax.set_ylabel("P(Right)")
        else:
            ax.set_ylabel("")

    fig.suptitle(model_name, y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, None


# Alias used by the analysis notebooks
plot_categorical_performance_by_state = plot_categorical_performance_all_by_state


def plot_regressor_psychometric_by_state(
    df,
    views: dict,
    model_name: str,
    feature_col: str = "at_choice",
    choice_col: str = "response",
    subj_col: str = "subject",
    X_cols: Optional[Sequence[str]] = None,
    feature_min: Optional[float] = None,
    feature_max: Optional[float] = None,
    background_style: str = "data",
    n_bins: int = 9,
    n_grid: int = 300,
) -> plt.Figure:
    """Per-state partial-dependence plot for any emission regressor.

    The x-axis is the chosen regressor (for example ``at_choice``) instead of
    ILD. Empirical points are pooled within quantile bins of that regressor,
    while the model line sweeps the same regressor over a dense grid and
    marginalises over the empirical distribution of the remaining features.
    """
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas().reset_index(drop=True)
    else:
        df_pd = df.reset_index(drop=True)

    if feature_col not in df_pd.columns:
        raise ValueError(f"df must contain the regressor column {feature_col!r}.")

    df_pd = df_pd.copy()
    df_pd[feature_col] = pd.to_numeric(df_pd[feature_col], errors="coerce")
    df_pd = df_pd.dropna(subset=[feature_col])
    if df_pd.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No valid {feature_col} data", ha="center", va="center")
        ax.axis("off")
        return fig, None

    if "state_rank" in df_pd.columns:
        _arr = df_pd["state_rank"].to_numpy().astype(int)
    elif "_state_k" in df_pd.columns:
        _arr = df_pd["_state_k"].to_numpy().astype(int)
    else:
        raise ValueError("df must contain a 'state_rank' column (output of build_trial_df)")
    df_pd["_state_k"] = _arr

    if feature_min is None:
        feature_min = float(np.nanmin(df_pd[feature_col].to_numpy(dtype=float)))
    if feature_max is None:
        feature_max = float(np.nanmax(df_pd[feature_col].to_numpy(dtype=float)))
    if not np.isfinite(feature_min) or not np.isfinite(feature_max):
        raise ValueError(f"Could not infer finite range for {feature_col!r}.")
    if feature_min == feature_max:
        feature_min -= 1e-6
        feature_max += 1e-6

    K = next(iter(views.values())).K if views else int(df_pd["_state_k"].max()) + 1

    slbls: dict[int, str] = {}
    for v in views.values():
        for k, lbl in v.state_name_by_idx.items():
            rank = v.state_rank_by_idx.get(int(k), int(k))
            slbls.setdefault(rank, lbl)

    def _rank_ordered_as(v):
        _order = sorted(v.state_rank_by_idx, key=lambda ki: v.state_rank_by_idx[ki])
        return {
            "emission_weights": v.emission_weights[_order],
            "X_cols": v.feat_names,
            "X": v.X,
            "smoothed_probs": v.smoothed_probs[:, _order],
            "lapse_rates": v.lapse_rates,
        }

    _as = {s: _rank_ordered_as(v) for s, v in views.items()}
    _all_subjects = list(df_pd[subj_col].unique()) if subj_col in df_pd.columns else list(_as.keys())

    _smooth_by_k: dict[int, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    _test_W = next((v.emission_weights for v in views.values()), None)
    _K_fit = int(np.asarray(_test_W).shape[0]) if _test_W is not None else 1
    _smooth_single = _mean_glm_feature_curve(
        _as,
        _all_subjects,
        X_cols,
        feature_name=feature_col,
        grid_min=feature_min,
        grid_max=feature_max,
        state_k=None,
        n_grid=n_grid,
    )
    for k in range(K):
        if _K_fit == 1:
            _smooth_by_k[k] = _smooth_single
        else:
            _smooth_by_k[k] = _mean_glm_feature_curve(
                _as,
                _all_subjects,
                X_cols,
                feature_name=feature_col,
                grid_min=feature_min,
                grid_max=feature_max,
                state_k=k,
                n_grid=n_grid,
            )
    _subject_curves_by_k = (
        {
            k: _subject_glm_feature_curves(
                _as,
                _all_subjects,
                X_cols,
                feature_name=feature_col,
                grid_min=feature_min,
                grid_max=feature_max,
                state_k=None if _K_fit == 1 else k,
                n_grid=n_grid,
            )
            for k in range(K)
        }
        if background_style == "model"
        else {}
    )

    _include_overlay = K > 1
    _n_panels = K + int(_include_overlay)
    fig, axes = plt.subplots(1, _n_panels, figsize=(3.5 * _n_panels, 4), sharey=True)
    axes = np.atleast_1d(axes)

    xlabel = _feature_label(feature_col)

    if _include_overlay:
        _ax_overlay = axes[0]
        for k in range(K):
            lbl = slbls.get(k, f"State {k}")
            color = _state_color(lbl, k)
            _df_state = df_pd[df_pd["_state_k"] == k]
            _regressor_state_panel(
                _ax_overlay,
                _df_state,
                feature_col,
                choice_col,
                subj_col,
                color=color,
                label=lbl,
                smooth_curve=_smooth_by_k[k],
                show_subject_traces=False,
                background_style=background_style,
                subject_curves=_subject_curves_by_k.get(k),
                n_bins=n_bins,
            )
        _ax_overlay.set_xlabel(xlabel)
        _ax_overlay.set_ylabel("P(Right)")
        _ax_overlay.set_title("All states")
        _ax_overlay.legend(frameon=False, fontsize=8)

    for k, ax in enumerate(axes[int(_include_overlay) :]):
        lbl = slbls.get(k, f"State {k}")
        color = _state_color(lbl, k)
        _df_state = df_pd[df_pd["_state_k"] == k]
        _regressor_state_panel(
            ax,
            _df_state,
            feature_col,
            choice_col,
            subj_col,
            color=color,
            label=lbl,
            smooth_curve=_smooth_by_k[k],
            background_style=background_style,
            subject_curves=_subject_curves_by_k.get(k),
            n_bins=n_bins,
        )
        ax.set_xlabel(xlabel)
        ax.set_title(lbl)
        if k == 0:
            ax.set_ylabel("P(Right)")
        else:
            ax.set_ylabel("")

    fig.suptitle(f"{model_name} — {_feature_label(feature_col)} psychometric", y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, None
