"""
nuo_auditory.py
───────────────
Plotting utilities for the Nuo auditory 2AFC task.

This is the task-owned Nuo auditory plotting module exposed via
``TaskAdapter.get_plots()``. Its public API mirrors the task plot interface
used by the analysis notebooks.

High-level functions:
  - plot_emission_weights
  - plot_posterior_probs
  - plot_state_accuracy
  - plot_session_trajectories
  - plot_state_occupancy
  - plot_state_dwell_times
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
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from glmhmmt.model_plots import (
    norm_ll as _norm_ll,
    plot_binary_emission_weights as _plot_binary_emission_weights,
    plot_binary_emission_weights_by_subject as _plot_binary_emission_weights_by_subject,
    plot_binary_emission_weights_summary as _plot_binary_emission_weights_summary,
    plot_binary_emission_weights_summary_boxplot as _plot_binary_emission_weights_summary_boxplot,
    plot_binary_emission_weights_summary_lineplot as _plot_binary_emission_weights_summary_lineplot,
    plot_ll as _plot_ll,
    plot_model_comparison as _plot_model_comparison,
    plot_model_comparison_diffs as _plot_model_comparison_diffs,
    plot_occupancy as _plot_occupancy,
    plot_occupancy_boxplot as _plot_occupancy_boxplot,
    plot_change_triggered_posteriors_by_subject as _plot_change_triggered_posteriors_by_subject,
    plot_change_triggered_posteriors_summary as _plot_change_triggered_posteriors_summary,
    plot_session_deepdive as _plot_session_deepdive,
    plot_session_trajectories as _plot_session_trajectories,
    plot_state_accuracy as _plot_state_accuracy,
    plot_state_dwell_times as _plot_state_dwell_times,
    plot_state_dwell_times_by_subject as _plot_state_dwell_times_by_subject,
    plot_state_dwell_times_summary as _plot_state_dwell_times_summary,
    plot_state_occupancy as _plot_state_occupancy,
    plot_state_occupancy_overall_boxplot as _plot_state_occupancy_overall_boxplot,
    plot_state_posterior_count_kde as _plot_state_posterior_count_kde,
    plot_trans_mat as _plot_trans_mat,
    plot_trans_mat_boxplots as _plot_trans_mat_boxplots,
    plot_transition_matrix as _plot_transition_matrix,
    plot_transition_matrix_by_subject as _plot_transition_matrix_by_subject,
    plot_transition_weights,
)
from glmhmmt.views import get_state_color, get_state_palette
from ..nuo_auditory import _stim_bin_centers, _stim_param_weight_map, EMISSION_REGRESSOR_LABELS

_SESSION_COL = "session"
_SORT_COL = "trial_idx"
_RESPONSE_COL = "response"
_PERFORMANCE_COL = "performance"
_EVIDENCE_COL = "total_evidence_strength"
_CONDITION_COL = "difficulty"
_EXPERIMENT_COL = "stimulus_modality"
_PLOT_CHOICE_LEFT_COL = "_plot_choice_left"

# ── state colour palette ──────────────────────────────────────────────────────


def _state_colors(K: int) -> List[str]:
    return get_state_palette(K)[:K]


def _default_labels(K: int, C: int = 2) -> List[str]:
    """Auto-generate state labels like ['Disengaged','Engaged'] for K=2."""
    if K == 1:
        return ["State 0"]
    if K == 2:
        return ["Disengaged", "Engaged"]
    if K == 3:
        return ["Engaged", "Biased L", "Biased R"]
    return [f"State {k}" for k in range(K)]


def _with_plot_choice_left(
    df: pd.DataFrame,
    choice_col: str,
) -> Tuple[pd.DataFrame, str]:
    """Return a copy of *df* with a plotting column for P(left)."""
    out = df.copy()
    out[_PLOT_CHOICE_LEFT_COL] = 1.0 - pd.to_numeric(out[choice_col], errors="coerce")
    return out, _PLOT_CHOICE_LEFT_COL


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
    ax.set_xticklabels(_format_feature_labels(feature_names), rotation=0, ha="center")
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
        ax.set_xticklabels(_format_feature_labels(feature_names), rotation=0, ha="center")
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

    hue_width = 0.8 / K
    positions = []
    grouped_weights = []
    for m in range(M):
        for k in range(K):
            positions.append(m + (k - (K - 1) / 2) * hue_width)
            grouped_weights.append(W_avg[:, k, m])

    box = ax_box.boxplot(
        grouped_weights,
        positions=positions,
        widths=hue_width * 0.9,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "white", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.1},
        capprops={"color": "black", "linewidth": 0},
        medianprops={"linewidth": 2.0},
    )

    for patch in box["boxes"]:
        patch.set(facecolor="white", edgecolor="#666666", linewidth=1.1)
    for elem in ("whiskers", "caps"):
        for artist in box[elem]:
            artist.set(color="#666666", linewidth=1.0)
    for idx, median in enumerate(box["medians"]):
        median.set(color=colors[idx % K], linewidth=2.2)

    for m in range(M):
        for n in range(N):
            xs = []
            ys = []
            for k in range(K):
                xs.append(m + (k - (K - 1) / 2) * hue_width)
                ys.append(W_avg[n, k, m])
            ax_box.plot(xs, ys, color="#7A7A7A", alpha=0.18, lw=0.8, zorder=0)

    for m in range(M):
        for k in range(K):
            x_pos = m + (k - (K - 1) / 2) * hue_width
            ax_box.scatter(
                np.full(N, x_pos),
                W_avg[:, k, m],
                color=colors[k],
                alpha=0.8,
                s=22,
                zorder=3,
                linewidths=0,
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
    ax_box.set_xticklabels(_format_feature_labels(feature_names), rotation=0, ha="center")
    ax_box.set_ylabel("Weight")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=colors[k], markeredgecolor="none", markersize=7, label=labels[k])
        for k in range(K)
    ]
    ax_box.legend(legend_handles, labels[:K], frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


plot_trans_mat = _plot_trans_mat
plot_trans_mat_boxplots = _plot_trans_mat_boxplots
plot_occupancy = _plot_occupancy
plot_occupancy_boxplot = _plot_occupancy_boxplot
norm_ll = _norm_ll
plot_ll = _plot_ll
plot_model_comparison = _plot_model_comparison
plot_model_comparison_diffs = _plot_model_comparison_diffs


# ─────────────────────────────────────────────────────────────────────────────
# Psychometric helpers
# ─────────────────────────────────────────────────────────────────────────────

_LABELED_ILDS = {-8, 8}


def _legacy_square_panel_size(n_cols: int = 2) -> tuple[float, float]:
    """Return the legacy A4-derived square panel size used in older plots."""
    a4_size = np.array((8.27, 11.69), dtype=float)
    margins = 2.0
    usable = a4_size - margins
    panel_w = float(usable[0] / float(n_cols))
    return panel_w, panel_w


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
    trial_weights: Optional[np.ndarray] = None,
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
        ild_max:     Maximum absolute evidence value used for normalisation.
        n_grid:      Number of ILD points in the evaluation grid.
        lapse_rates: ``[gamma_L, gamma_R]`` lapse parameters from fitting.
                     *None* means no lapse correction.
        X_data:      ``(T, M)`` actual feature matrix used during fitting.
                     When provided, partial-dependence mode is used.
        trial_weights:
                     Optional non-negative weights of length ``T`` used to
                     average the trial-wise predictions in partial-dependence
                     mode. This is useful for state-specific curves, where the
                     natural weights are the HMM posteriors ``gamma_t(k)``.

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
    stim_bin_indices = {
        name: idx
        for idx, name in enumerate(X_cols_list)
        if isinstance(name, str) and name.startswith("stim_bin_")
    }
    stim_centers = _stim_bin_centers()
    stim_bin_names = [f"stim_bin_{idx:02d}" for idx in range(len(stim_centers))]
    stim_center_map = {
        name: float(center)
        for name, center in zip(stim_bin_names, stim_centers)
    }
    stim_param_idx = next((i for i, n in enumerate(X_cols_list) if n == "stim_param"), None)
    stim_param_weights = _stim_param_weight_map() if stim_param_idx is not None else {}

    # Accept any of these as the stimulus / ILD column
    _STIM_NAMES = {"stim_vals", "stim_d", "ild_norm", "ILD", "ild", "stimulus"}
    ild_idx = next((i for i, n in enumerate(X_cols_list) if n in _STIM_NAMES), None)
    bias_idx = next((i for i, n in enumerate(X_cols_list) if n == "bias"), None)
    if stim_bin_indices or stim_param_idx is not None:
        ild_grid = stim_centers.astype(float) * float(ild_max)
    else:
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

    weights_t = None
    if X_data is not None and trial_weights is not None:
        _w = np.asarray(trial_weights, dtype=float).reshape(-1)
        if _w.shape[0] == np.asarray(X_data).shape[0]:
            _w = np.where(np.isfinite(_w) & (_w > 0), _w, 0.0)
            if float(_w.sum()) > 0:
                weights_t = _w

    stim_feature_indices = sorted(
        set(
            ([ild_idx] if ild_idx is not None else [])
            + list(stim_bin_indices.values())
            + ([stim_param_idx] if stim_param_idx is not None else [])
        )
    )

    def _stim_bin_name_from_norm(stim_value: float) -> str:
        return min(
            stim_center_map,
            key=lambda name: abs(stim_value - stim_center_map[name]),
        )

    if X_data is not None and (ild_idx is not None or stim_bin_indices or stim_param_idx is not None):
        # ── partial-dependence: average over real trial features ──────────────
        X_base = np.asarray(X_data, dtype=float).copy()
        for k in range(K):
            w = W[k, 0, :]
            other_logit = X_base @ w
            stim_contrib = (
                X_base[:, stim_feature_indices] @ w[stim_feature_indices]
                if stim_feature_indices
                else 0.0
            )
            base_logit = other_logit - stim_contrib
            for gi, sv in enumerate(ild_norm):
                stim_logit = 0.0
                if ild_idx is not None:
                    stim_logit += sv * w[ild_idx]
                stim_bin_name = _stim_bin_name_from_norm(float(sv))
                for name, idx in stim_bin_indices.items():
                    stim_logit += (1.0 if name == stim_bin_name else 0.0) * w[idx]
                if stim_param_idx is not None:
                    stim_logit += float(stim_param_weights.get(stim_bin_name, 0.0)) * w[stim_param_idx]
                logit = base_logit + stim_logit
                # W[k,0,:] parameterises P(class-0 = LEFT); class-1 (RIGHT) is
                # the softmax reference (logit=0). So P(right) = sigmoid(-logit).
                p_left = 1.0 / (1.0 + np.exp(-logit))
                p_trial = gL + (1.0 - gL - gR) * (1.0 - p_left)
                if weights_t is not None:
                    p_right[k, gi] = float(np.average(p_trial, weights=weights_t))
                else:
                    p_right[k, gi] = float(np.mean(p_trial))
    else:
        # ── fallback: sweep only stim, fix others at empirical mean ──────────
        if X_data is not None:
            col_means = np.asarray(X_data, dtype=float).mean(axis=0)
        else:
            col_means = np.zeros(M)
            if bias_idx is not None:
                col_means[bias_idx] = 1.0

        X_grid = np.tile(col_means, (len(ild_grid), 1))
        if stim_feature_indices:
            X_grid[:, stim_feature_indices] = 0.0
        if ild_idx is not None:
            X_grid[:, ild_idx] = ild_norm
        for gi, sv in enumerate(ild_norm):
            stim_bin_name = _stim_bin_name_from_norm(float(sv))
            stim_bin_idx = stim_bin_indices.get(stim_bin_name)
            if stim_bin_idx is not None:
                X_grid[gi, stim_bin_idx] = 1.0
            if stim_param_idx is not None:
                X_grid[gi, stim_param_idx] = float(stim_param_weights.get(stim_bin_name, 0.0))
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
    trial_weights: Optional[np.ndarray] = None,
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

    weights_t = None
    if X_base is not None and trial_weights is not None:
        _w = np.asarray(trial_weights, dtype=float).reshape(-1)
        if _w.shape[0] == X_base.shape[0]:
            _w = np.where(np.isfinite(_w) & (_w > 0), _w, 0.0)
            if float(_w.sum()) > 0:
                weights_t = _w

    if X_base is not None:
        for k in range(K):
            w = W[k, 0, :]
            other_logit = X_base @ w
            swept_contrib = X_base[:, feat_idx] * w[feat_idx]
            base_logit = other_logit - swept_contrib
            for gi, gv in enumerate(grid):
                logit = base_logit + gv * w[feat_idx]
                p_left = 1.0 / (1.0 + np.exp(-logit))
                p_trial = gL + (1.0 - gL - gR) * (1.0 - p_left)
                if weights_t is not None:
                    p_right[k, gi] = float(np.average(p_trial, weights=weights_t))
                else:
                    p_right[k, gi] = float(np.mean(p_trial))
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
        ``(ild_grid, mean_p_left)`` or *None* if no valid fits are found.
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
            pg = 1.0 - np.asarray(pg, dtype=float)
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
    """Return {subject: (ild_grid, p_left)} for per-subject psychometric backgrounds."""
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
            pg = 1.0 - np.asarray(pg, dtype=float)
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
    """Return {subject: (feature_grid, p_left)} for per-subject feature backgrounds."""
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


def _mean_weighted_empirical_curve(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    subj_col: str,
    weight_col: Optional[str] = None,
    grid: Optional[np.ndarray] = None,
    grid_min: Optional[float] = None,
    grid_max: Optional[float] = None,
    n_grid: int = 300,
    bandwidth: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Average a subject-level kernel smoother of observed choices over subjects."""
    cols = [x_col, y_col, subj_col]
    if weight_col is not None:
        cols.append(weight_col)
    d = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    if d.empty:
        return None

    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    if weight_col is not None and weight_col in d.columns:
        d[weight_col] = pd.to_numeric(d[weight_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    if d.empty:
        return None

    x_all = d[x_col].to_numpy(dtype=float)
    finite_x = x_all[np.isfinite(x_all)]
    if finite_x.size == 0:
        return None

    if grid is None:
        lo = float(np.min(finite_x)) if grid_min is None else float(grid_min)
        hi = float(np.max(finite_x)) if grid_max is None else float(grid_max)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
        grid = np.linspace(lo, hi, int(n_grid))
    else:
        grid = np.asarray(grid, dtype=float)

    curves: list[np.ndarray] = []
    for _, grp in d.groupby(subj_col, observed=True):
        x = pd.to_numeric(grp[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(grp[y_col], errors="coerce").to_numpy(dtype=float)
        if weight_col is not None and weight_col in grp.columns:
            w = pd.to_numeric(grp[weight_col], errors="coerce").to_numpy(dtype=float)
        else:
            w = np.ones_like(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        if not np.any(mask):
            continue
        x = x[mask]
        y = y[mask]
        w = w[mask]
        if x.size == 0 or float(w.sum()) <= 0:
            continue

        bw = bandwidth
        if bw is None:
            xu = np.unique(np.sort(x))
            if xu.size >= 2:
                bw = float(np.median(np.diff(xu))) * 1.5
            else:
                span = float(np.max(x) - np.min(x))
                bw = span / 6.0 if span > 0 else 1.0
        bw = max(float(bw), 1e-6)

        z = (grid[:, None] - x[None, :]) / bw
        kernel = np.exp(-0.5 * z * z)
        kw = kernel * w[None, :]
        denom = kw.sum(axis=1)
        numer = kw @ y
        curve = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
        curves.append(curve)

    if not curves:
        return None
    return grid, np.nanmean(np.vstack(curves), axis=0)


def _feature_label(feature_name: str) -> str:
    return EMISSION_REGRESSOR_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def _format_feature_labels(feature_names: Sequence[str]) -> list[str]:
    return [_feature_label(name) for name in feature_names]


def _binned_feature_summary(
    df: pd.DataFrame,
    feature_col: str,
    choice_col: str,
    pred_col: str,
    subj_col: str,
    n_bins: int = 9,
    weight_col: Optional[str] = None,
) -> Optional[Tuple[pd.DataFrame, list[float]]]:
    needed = [feature_col, choice_col, pred_col, subj_col]
    d = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    if d.empty:
        return None

    d[feature_col] = pd.to_numeric(d[feature_col], errors="coerce")
    d[choice_col] = pd.to_numeric(d[choice_col], errors="coerce")
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce")
    d = d.dropna(subset=[feature_col, choice_col, pred_col])
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

    if weight_col is not None and weight_col in d.columns:
        rows = []
        for (x_bin, subj), grp in d.groupby(["_x_bin", subj_col], observed=True):
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
            rows.append(
                {
                    "_x_bin": x_bin,
                    subj_col: subj,
                    "data_mean": float(np.dot(y[mask], w) / w_sum),
                    "model_mean": float(np.dot(m[mask], w) / w_sum),
                }
            )
        subj = pd.DataFrame(rows)
        if not subj.empty:
            subj = subj.merge(centers, on="_x_bin", how="left")
    else:
        subj = (
            d.groupby(["_x_bin", subj_col], observed=True)
            .agg(data_mean=(choice_col, "mean"), model_mean=(pred_col, "mean"))
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
            mm=("model_mean", "mean"),
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
    labels: list[str] = []
    for v in ilds:
        if float(v) not in labeled:
            labels.append("")
            continue
        if float(v) == -20.0:
            labels.append("-70")
        elif float(v) == 20.0:
            labels.append("70")
        else:
            labels.append(str(int(v)))
    return labels


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


def _style_legacy_psych_axis(ax: plt.Axes, xticks: Sequence[float]) -> None:
    """Match the legacy categorical psychometric axis styling."""
    _apply_ild_axis_ticks(ax, xticks)
    ax.axhline(0.5, color="tab:gray", ls="--", lw=1.6)
    ax.axvline(0.0, color="tab:gray", ls="--", lw=1.6)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.5, 1], [0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _require_plot_col(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        raise KeyError(f"Missing required plotting column {col!r}.")
    return col


def _attach_rank_posterior_cols(
    df: pd.DataFrame,
    views: dict,
    subj_col: str = "subject",
) -> pd.DataFrame:
    """Attach `_p_state_rank_k` columns from `p_state_pred_*` using rank↔raw maps."""
    if df.empty or subj_col not in df.columns or not views:
        return df

    K = next(iter(views.values())).K
    out = df.copy()
    rank_cols = [f"_p_state_rank_{k}" for k in range(K)]
    if all(col in out.columns for col in rank_cols):
        return out

    idx_by_rank_by_subj: dict[object, dict[int, int]] = {}
    for subj, view in views.items():
        idx_by_rank_by_subj[subj] = {int(rank): int(raw_idx) for raw_idx, rank in view.state_rank_by_idx.items()}

    n_rows = len(out)
    for rank, rank_col in enumerate(rank_cols):
        if rank_col in out.columns:
            continue
        vals = np.full(n_rows, np.nan, dtype=float)
        for subj, idx in out.groupby(subj_col, observed=True).groups.items():
            raw_by_rank = idx_by_rank_by_subj.get(subj)
            if raw_by_rank is None:
                raw_by_rank = idx_by_rank_by_subj.get(str(subj))
            if raw_by_rank is None:
                continue
            raw_idx = raw_by_rank.get(rank)
            if raw_idx is None:
                continue
            src_col = f"p_state_pred_{raw_idx}"
            if src_col not in out.columns:
                raise KeyError(
                    f"Missing required predictive state column {src_col!r}. "
                    "Rebuild trial_df with the updated predictive-state export."
                )
            row_idx = np.asarray(idx, dtype=int)
            vals[row_idx] = pd.to_numeric(out.iloc[row_idx][src_col], errors="coerce").to_numpy(dtype=float)
        out[rank_col] = vals

    return out


def _attach_rank_state_model_cols(
    df: pd.DataFrame,
    views: dict,
    subj_col: str = "subject",
    base_col: str = "pL_state",
) -> pd.DataFrame:
    """Attach rank-aligned per-state model columns from raw-state trial_df columns."""
    if df.empty or subj_col not in df.columns or not views:
        return df

    K = next(iter(views.values())).K
    out = df.copy()
    rank_cols = [f"_{base_col}_rank_{k}" for k in range(K)]
    if all(col in out.columns for col in rank_cols):
        return out

    idx_by_rank_by_subj: dict[object, dict[int, int]] = {}
    for subj, view in views.items():
        idx_by_rank_by_subj[subj] = {int(rank): int(raw_idx) for raw_idx, rank in view.state_rank_by_idx.items()}

    n_rows = len(out)
    for rank, rank_col in enumerate(rank_cols):
        if rank_col in out.columns:
            continue
        vals = np.full(n_rows, np.nan, dtype=float)
        for subj, idx in out.groupby(subj_col, observed=True).groups.items():
            raw_by_rank = idx_by_rank_by_subj.get(subj)
            if raw_by_rank is None:
                raw_by_rank = idx_by_rank_by_subj.get(str(subj))
            if raw_by_rank is None:
                continue
            raw_idx = raw_by_rank.get(rank)
            if raw_idx is None:
                continue
            src_col = f"{base_col}_{raw_idx}"
            if src_col not in out.columns:
                raise KeyError(
                    f"Missing required state-conditional model column {src_col!r}. "
                    "Rebuild trial_df with the updated per-state prediction export."
                )
            row_idx = np.asarray(idx, dtype=int)
            vals[row_idx] = pd.to_numeric(out.iloc[row_idx][src_col], errors="coerce").to_numpy(dtype=float)
        out[rank_col] = vals

    return out


def _psych_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    ild_col: str = _EVIDENCE_COL,
    choice_col: str = _RESPONSE_COL,
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    title: str = "",
    xlabel: str = "Evidence strength",
    ylabel: Optional[str] = None,
    color: str = "k",
    smooth_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    background_style: str = "data",
    subject_curves: Optional[dict] = None,
    tick_ilds: Optional[Sequence[float]] = None,
) -> None:
    """Draw a pooled psychometric curve (P(left) vs evidence strength) on ax.

    Style mirrors plot_pc_across_batches:
    - Per-subject individual traces drawn with low alpha.
    - Extreme stimulus positions compressed so inner values are not squeezed.
    - Pooled mean ± SEM as error-bar markers; model as a solid black line.
    - axhline at 0.5, axvline at 0.
    """
    if df.empty:
        ax.set_title(title)
        return

    choice_col = _require_plot_col(df, choice_col)
    df_plot, choice_col = _with_plot_choice_left(df, choice_col)

    subj_agg = (
        df_plot.groupby([subj_col, ild_col], observed=True)
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

    # model line: smooth sigmoid over a dense stimulus grid (if available)
    if smooth_curve is not None:
        ild_g, p_g = smooth_curve
        x0, x1 = float(xticks[0]), float(xticks[-1])
        clip = (ild_g >= x0) & (ild_g <= x1)
        ax.plot(ild_g[clip], p_g[clip], "-", color="black", lw=2, label="Model", zorder=6)
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
    show_weighted_points: bool = True,
    show_data_smooth: bool = True,
    show_model_smooth: bool = True,
    model_line_mode: str = "smooth",
    bin_points: bool = False,
    n_bins: int = 9,
) -> Tuple:
    """Draw state-specific P(left) psychometric on ax. Returns (data_h, model_h)."""
    if df_state.empty:
        return None, None

    choice_col = _require_plot_col(df_state, choice_col)
    df_state, choice_col = _with_plot_choice_left(df_state, choice_col)
    empirical_smooth = None
    if weight_col is not None and weight_col in df_state.columns:
        empirical_smooth = _mean_weighted_empirical_curve(
            df_state,
            x_col=ild_col,
            y_col=choice_col,
            subj_col=subj_col,
            weight_col=weight_col,
            grid=smooth_curve[0] if smooth_curve is not None else None,
        )
    if bin_points:
        summary = _binned_feature_summary(
            df_state,
            feature_col=ild_col,
            choice_col=choice_col,
            pred_col=pred_col,
            subj_col=subj_col,
            n_bins=n_bins,
            weight_col=weight_col,
        )
        if summary is None:
            return None, None
        subj_agg, _ = summary
        agg = (
            subj_agg.groupby("_x_bin", observed=True)
            .agg(
                x=("center", "median"),
                md=("data_mean", "mean"),
                sd=("data_mean", "std"),
                nd=("data_mean", "count"),
                mm=("model_mean", "mean"),
            )
            .reset_index(drop=True)
            .sort_values("x")
        )
        x = agg["x"].to_numpy(dtype=float)
        md = agg["md"].to_numpy(dtype=float)
        sd = agg["sd"].fillna(0.0).to_numpy(dtype=float)
        nd = agg["nd"].clip(lower=1).to_numpy(dtype=float)
        mm = agg["mm"].to_numpy(dtype=float)
        sem_d = sd / np.sqrt(nd)
    else:
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
        x = np.array(ilds, dtype=float)
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
            if bin_points:
                grp = grp.sort_values("center")
                xi = grp["center"].to_numpy(dtype=float)
                yi = grp["data_mean"].to_numpy(dtype=float)
            else:
                grp_ilds = [i for i in ilds if i in grp[ild_col].values]
                xi = np.array(grp_ilds, dtype=float)
                yi = grp.set_index(ild_col).reindex(grp_ilds)["data_mean"].values
            ax.plot(xi, yi, "-o", color=color, alpha=0.14, lw=1.1, ms=4.0, zorder=2)
    elif show_subject_traces and background_style == "model" and subject_curves is not None:
        for subj, curve in subject_curves.items():
            if curve is None:
                continue
            xi, yi = curve
            ax.plot(xi, yi, "-", color=color, alpha=0.14, lw=1.2, zorder=2)

    if show_data_smooth and empirical_smooth is not None:
        x_emp, y_emp = empirical_smooth
        ax.plot(x_emp, y_emp, "--", color=color, lw=1.9, alpha=0.95, zorder=4, label="_nolegend_")

    data_h = None
    if show_weighted_points:
        data_h = ax.errorbar(
            x,
            md,
            yerr=sem_d,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=0,
            ms=5.8,
            zorder=5,
            label=label,
        )
    # smooth sigmoid model line (if available) else aggregated p_pred
    if show_model_smooth and model_line_mode == "smooth" and smooth_curve is not None:
        ild_g, p_g = smooth_curve
        # Clip the dense grid to the observed evidence range so the sigmoid
        # doesn't extend far beyond the data and compress the visible area.
        _x0, _x1 = float(np.nanmin(x)), float(np.nanmax(x))
        _clip = (ild_g >= _x0) & (ild_g <= _x1)
        (model_h,) = ax.plot(ild_g[_clip], p_g[_clip], "-", color=color, lw=2.3, zorder=6, label="_nolegend_")
    elif show_model_smooth:
        (model_h,) = ax.plot(x, mm, "-", color=color, lw=2.3, zorder=6, label="_nolegend_")
    else:
        model_h = None

    if bin_points:
        ax.axhline(0.5, color="tab:gray", ls="--", lw=1.6)
        ax.axvline(0.0, color="tab:gray", ls="--", lw=1.6)
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 0.5, 1], [0, 0.5, 1])
        ax.tick_params(axis="both", labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.title.set_size(13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    else:
        _style_legacy_psych_axis(ax, xticks)
    return data_h, model_h


def _regressor_state_panel(
    ax: plt.Axes,
    df_state: pd.DataFrame,
    feature_col: str,
    choice_col: str,
    pred_col: str,
    subj_col: str,
    color: str,
    label: str,
    smooth_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    show_subject_traces: bool = True,
    background_style: str = "data",
    subject_curves: Optional[dict] = None,
    n_bins: int = 9,
    weight_col: Optional[str] = None,
    show_weighted_points: bool = True,
    show_data_smooth: bool = True,
    show_model_smooth: bool = True,
    model_line_mode: str = "smooth",
) -> Tuple:
    """Draw state-specific P(left) vs arbitrary regressor on ax."""
    if df_state.empty:
        return None, None

    choice_col = _require_plot_col(df_state, choice_col)
    df_state, choice_col = _with_plot_choice_left(df_state, choice_col)
    empirical_smooth = None
    if weight_col is not None and weight_col in df_state.columns:
        empirical_smooth = _mean_weighted_empirical_curve(
            df_state,
            x_col=feature_col,
            y_col=choice_col,
            subj_col=subj_col,
            weight_col=weight_col,
            grid=smooth_curve[0] if smooth_curve is not None else None,
        )
    summary = _binned_feature_summary(
        df_state,
        feature_col,
        choice_col,
        pred_col,
        subj_col,
        n_bins=n_bins,
        weight_col=weight_col,
    )
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
            mm=("model_mean", "mean"),
        )
        .reset_index(drop=True)
        .sort_values("x")
    )
    x = agg["x"].to_numpy(dtype=float)
    md = agg["md"].to_numpy(dtype=float)
    sd = agg["sd"].fillna(0.0).to_numpy(dtype=float)
    nd = agg["nd"].clip(lower=1).to_numpy(dtype=float)
    mm = agg["mm"].to_numpy(dtype=float)
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
                lw=1.1,
                ms=4.0,
                zorder=2,
            )
    elif show_subject_traces and background_style == "model" and subject_curves is not None:
        for subj, curve in subject_curves.items():
            if curve is None:
                continue
            xi, yi = curve
            ax.plot(xi, yi, "-", color=color, alpha=0.14, lw=1.2, zorder=2)

    if show_data_smooth and empirical_smooth is not None:
        x_emp, y_emp = empirical_smooth
        ax.plot(x_emp, y_emp, "--", color=color, lw=1.9, alpha=0.95, zorder=4, label="_nolegend_")

    data_h = None
    if show_weighted_points:
        data_h = ax.errorbar(
            x,
            md,
            yerr=sem_d,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=0,
            ms=5.8,
            zorder=5,
            label=label,
        )
    if show_model_smooth and model_line_mode == "smooth" and smooth_curve is not None:
        feat_g, p_g = smooth_curve
        (model_h,) = ax.plot(feat_g, p_g, "-", color=color, lw=2.3, zorder=6, label="_nolegend_")
    elif show_model_smooth:
        (model_h,) = ax.plot(x, mm, "-", color=color, lw=2.3, zorder=6, label="_nolegend_")
    else:
        model_h = None

    ax.axhline(0.5, color="tab:gray", ls="--", lw=1.6)
    ax.axvline(0.0, color="tab:gray", ls="--", lw=1.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    return data_h, model_h


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame preparation  (mirrors plots.prepare_predictions_df)
# ─────────────────────────────────────────────────────────────────────────────


def prepare_predictions_df(df_pred):
    """Prepare a Nuo auditory trial-level predictions DataFrame for plotting.

    Accepts a polars or pandas DataFrame that already contains the per-Trial
    model predictions (``pL``, ``pR``) produced by the fit script.

    Expected input columns
    ----------------------
    stimulus             : int  - correct side (0 = left, 1 = right)
    response             : int  - subject choice from ``last_choice``
    performance          : int/bool - trial correct (1) or incorrect (0)
    total_evidence_strength : float - signed auditory evidence
    pL               : float - model P(left choice)
    pR               : float - model P(right choice)

    Added / ensured output columns
    ------------------------------
    correct_bool    : bool  - Trial accuracy
    p_pred          : float - model P(left) used on the psychometric y-axis
    p_model_correct : float - model P(correct stimulus)

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

        required = {"stimulus", _RESPONSE_COL, _PERFORMANCE_COL, _EVIDENCE_COL}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required Nuo auditory columns: {missing}")

        if "correct_bool" not in df.columns:
            df = df.with_columns(pl.col(_PERFORMANCE_COL).cast(pl.Boolean).alias("correct_bool"))

        if "pL" not in df.columns or "pR" not in df.columns:
            raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")

        df = df.with_columns(
            pl.col("pL").alias("p_pred"),
            pl.when(pl.col("stimulus") == 0).then(pl.col("pL")).otherwise(pl.col("pR")).alias("p_model_correct"),
        )

        return df

    else:
        # pandas path
        df = df_pred.copy()

        required = {"stimulus", _RESPONSE_COL, _PERFORMANCE_COL, _EVIDENCE_COL}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required Nuo auditory columns: {missing}")

        if "correct_bool" not in df.columns:
            df["correct_bool"] = df[_PERFORMANCE_COL].astype(bool)

        if "pL" not in df.columns or "pR" not in df.columns:
            raise ValueError("Missing 'pL' or 'pR' columns (model predictions).")

        df["p_pred"] = df["pL"]
        df["p_model_correct"] = df.apply(lambda row: row["pL"] if row["stimulus"] == 0 else row["pR"], axis=1)

        return df


# ─────────────────────────────────────────────────────────────────────────────
# High-level API used by the task plot facade
# ─────────────────────────────────────────────────────────────────────────────


def plot_emission_weights_by_subject(
    views: dict,
    K: int,
    save_path=None,
) -> plt.Figure:
    return _plot_binary_emission_weights_by_subject(
        views,
        K,
        weight_sign=1.0,
        feature_labeler=_feature_label,
        save_path=save_path,
    )


def plot_emission_weights_summary(
    views: dict,
    K: int,
) -> plt.Figure:
    return _plot_binary_emission_weights_summary(
        views,
        K,
        weight_sign=1.0,
        feature_labeler=_feature_label,
    )


def plot_emission_weights_summary_lineplot(
    views: dict,
    K: int,
) -> plt.Figure:
    return _plot_binary_emission_weights_summary_lineplot(
        views,
        K,
        weight_sign=1.0,
        feature_labeler=_feature_label,
    )


def plot_emission_weights_summary_boxplot(
    views: dict,
    K: int,
) -> plt.Figure:
    return _plot_binary_emission_weights_summary_boxplot(
        views,
        K,
        weight_sign=1.0,
        feature_labeler=_feature_label,
    )


def plot_emission_weights(
    views: dict,
    K: int,
    save_path=None,
) -> Tuple[plt.Figure, plt.Figure]:
    return _plot_binary_emission_weights(
        views,
        K,
        weight_sign=1.0,
        feature_labeler=_feature_label,
        save_path=save_path,
    )

plot_transition_matrix_by_subject = _plot_transition_matrix_by_subject
plot_transition_matrix = _plot_transition_matrix


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
    performance_col: str = _PERFORMANCE_COL,
    **kwargs,
) -> Tuple[plt.Figure, pd.DataFrame]:
    return _plot_state_accuracy(
        views,
        trial_df,
        thresh=thresh,
        performance_col=performance_col,
        **kwargs,
    )


def plot_session_trajectories(
    views: dict,
    trial_df,
    session_col: str = _SESSION_COL,
    sort_col: str = _SORT_COL,
    **kwargs,
) -> plt.Figure:
    return _plot_session_trajectories(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        **kwargs,
    )


def plot_state_posterior_count_kde(
    views: dict,
    thresh: float | None = None,
    bins: int = 40,
    **kwargs,
) -> plt.Figure:
    return _plot_state_posterior_count_kde(
        views,
        thresh=thresh,
        bins=bins,
        **kwargs,
    )


def plot_change_triggered_posteriors_summary(
    views: dict,
    trial_df,
    session_col: str = _SESSION_COL,
    sort_col: str = _SORT_COL,
    switch_posterior_threshold: float | None = None,
    window: int = 15,
    **kwargs,
) -> plt.Figure:
    return _plot_change_triggered_posteriors_summary(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        window=window,
        **kwargs,
    )


def plot_change_triggered_posteriors_by_subject(
    views: dict,
    trial_df,
    session_col: str = _SESSION_COL,
    sort_col: str = _SORT_COL,
    switch_posterior_threshold: float | None = None,
    window: int = 15,
    **kwargs,
) -> plt.Figure:
    return _plot_change_triggered_posteriors_by_subject(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        window=window,
        **kwargs,
    )


def plot_state_occupancy(
    views: dict,
    trial_df,
    session_col: str = _SESSION_COL,
    sort_col: str = _SORT_COL,
    **kwargs,
) -> plt.Figure:
    return _plot_state_occupancy(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        **kwargs,
    )


def plot_state_occupancy_overall_boxplot(
    views: dict,
    trial_df,
    session_col: str = _SESSION_COL,
    sort_col: str = _SORT_COL,
    **kwargs,
) -> plt.Figure:
    return _plot_state_occupancy_overall_boxplot(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        **kwargs,
    )


def plot_state_dwell_times_by_subject(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    max_dwell: int | None = None,
    ci_level: float = 0.68,
    **kwargs,
) -> plt.Figure:
    return _plot_state_dwell_times_by_subject(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
        **kwargs,
    )


def plot_state_dwell_times_summary(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    max_dwell: int | None = None,
    ci_level: float = 0.68,
    **kwargs,
) -> plt.Figure:
    return _plot_state_dwell_times_summary(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
        **kwargs,
    )


def plot_state_dwell_times(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    max_dwell: int | None = None,
    ci_level: float = 0.68,
    **kwargs,
) -> plt.Figure:
    return _plot_state_dwell_times(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session deep-dive
# ─────────────────────────────────────────────────────────────────────────────


def plot_session_deepdive(
    views: dict,
    trial_df,
    subj: str,
    sess,
    session_col: str = _SESSION_COL,
    sort_col: str = _SORT_COL,
    switch_posterior_threshold: float | None = None,
    **kwargs,
) -> plt.Figure:
    return _plot_session_deepdive(
        views,
        trial_df,
        subj,
        sess,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        performance_col="correct_bool",
        response_col=_RESPONSE_COL,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Psychometric performance helpers
# ─────────────────────────────────────────────────────────────────────────────


def plot_categorical_performance_all(
    df,
    model_name: str,
    ild_col: str = _EVIDENCE_COL,
    choice_col: str = _RESPONSE_COL,
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    cond_col: str = _CONDITION_COL,
    exp_col: str = _EXPERIMENT_COL,
    views: Optional[dict] = None,
    X_cols: Optional[Sequence[str]] = None,
    ild_max: Optional[float] = None,
    background_style: str = "data",
    n_bins: int = 9,
) -> plt.Figure:
    """Overall psychometric P(left) vs evidence strength.

    The Nuo non-state view is a single pooled panel. Empirical means use the
    modeled response column and the data are binned over the evidence axis
    before pooling across subjects.
    """
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy()
    df_pd, choice_col = _with_plot_choice_left(df_pd, choice_col)

    summary = _binned_feature_summary(
        df_pd,
        feature_col=ild_col,
        choice_col=choice_col,
        pred_col=pred_col,
        subj_col=subj_col,
        n_bins=n_bins,
    )
    if summary is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid psychometric data", ha="center", va="center")
        ax.axis("off")
        return fig, None

    subj_agg, _ = summary
    agg = (
        subj_agg.groupby("_x_bin", observed=True)
        .agg(
            x=("center", "median"),
            md=("data_mean", "mean"),
            sd=("data_mean", "std"),
            nd=("data_mean", "count"),
            mm=("model_mean", "mean"),
        )
        .reset_index(drop=True)
        .sort_values("x")
    )
    x = agg["x"].to_numpy(dtype=float)
    md = agg["md"].to_numpy(dtype=float)
    sd = agg["sd"].fillna(0.0).to_numpy(dtype=float)
    nd = agg["nd"].clip(lower=1).to_numpy(dtype=float)
    mm = agg["mm"].to_numpy(dtype=float)
    sem_d = sd / np.sqrt(nd)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharey=True)
    if background_style == "data":
        for _, grp in subj_agg.groupby(subj_col, observed=True):
            grp = grp.sort_values("center")
            ax.plot(
                grp["center"].to_numpy(dtype=float),
                grp["data_mean"].to_numpy(dtype=float),
                "-o",
                color="#2b7bba",
                alpha=0.15,
                lw=1.1,
                ms=4.0,
                zorder=2,
            )
    elif background_style == "model":
        for _, grp in subj_agg.groupby(subj_col, observed=True):
            grp = grp.sort_values("center")
            ax.plot(
                grp["center"].to_numpy(dtype=float),
                grp["model_mean"].to_numpy(dtype=float),
                "-",
                color="black",
                alpha=0.12,
                lw=1.2,
                zorder=2,
            )

    ax.plot(x, mm, "-", color="black", lw=2.3, zorder=6, label="Model")
    ax.errorbar(
        x,
        md,
        yerr=sem_d,
        fmt="o",
        color="#2b7bba",
        ecolor="#2b7bba",
        elinewidth=1.5,
        capsize=0,
        ms=5.8,
        zorder=5,
        label="Data",
    )
    ax.axhline(0.5, color="tab:gray", ls="--", lw=1.6)
    ax.axvline(0.0, color="tab:gray", ls="--", lw=1.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Evidence strength")
    ax.set_ylabel("P(Left)")
    ax.set_title("Overall psychometric")
    ax.legend(frameon=False, fontsize=8)
    sns.despine(fig=fig)
    fig.suptitle(model_name, y=1.02)
    fig.tight_layout()
    return fig, None


def plot_categorical_performance_all_by_state(
    df,
    views: dict,
    model_name: str,
    ild_col: str = _EVIDENCE_COL,
    choice_col: str = _RESPONSE_COL,
    pred_col: str = "p_pred",
    subj_col: str = "subject",
    X_cols: Optional[Sequence[str]] = None,
    ild_max: Optional[float] = None,
    background_style: str = "data",
    show_weighted_points: bool = True,
    show_data_smooth: bool = True,
    show_model_smooth: bool = True,
    figure_dpi: float = 80.0,
    overlay_only: bool = False,
    model_line_mode: str = "smooth",
    state_assignment_mode: str = "weighted",
    n_bins: int = 9,
) -> plt.Figure:
    """Per-state psychometric grid (K panels, one per state).

    Each state gets its own panel showing P(left) vs evidence strength; data (markers) and
    model prediction (lines) are drawn in the state's colour.

    Parameters
    ----------
    df     : Trial-level DataFrame (polars or pandas). Must contain a
             ``state_rank`` column (rank 0 = Engaged) produced by
             :func:`~glmhmmt.postprocess.build_trial_df`.
    views  : {subj: SubjectFitView} as produced by build_views.
    model_name : string used as figure suptitle.
    ild_max : Optional explicit normalisation scale. When omitted, the
              maximum absolute evidence value in ``df[ild_col]`` is used.

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
    if state_assignment_mode == "weighted":
        df_pd = _attach_rank_posterior_cols(df_pd, views, subj_col=subj_col)
        df_pd = _attach_rank_state_model_cols(df_pd, views, subj_col=subj_col, base_col="pL_state")

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
    panel_w = 3

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
    if overlay_only:
        _include_overlay = True
    _n_panels = K + int(_include_overlay)
    if overlay_only:
        _n_panels = 1
    _figsize = (3, 3) if overlay_only else (panel_w * _n_panels, 4)
    fig, axes = plt.subplots(1, _n_panels, figsize=_figsize, sharey=True, dpi=figure_dpi)
    axes = np.atleast_1d(axes)

    if _include_overlay:
        _ax_overlay = axes[0]
        for k in range(K):
            lbl = slbls.get(k, f"State {k}")
            color = get_state_color(lbl, k, K=K)
            _weight_col = (
                f"_p_state_rank_{k}" if state_assignment_mode == "weighted" and f"_p_state_rank_{k}" in df_pd.columns else None
            )
            _df_state = df_pd if _weight_col is not None else df_pd[df_pd["_state_k"] == k]
            _psych_state_panel(
                _ax_overlay,
                _df_state,
                ild_col,
                choice_col,
                pred_col=f"_pL_state_rank_{k}" if state_assignment_mode == "weighted" else pred_col,
                subj_col=subj_col,
                color=color,
                label=lbl,
                smooth_curve=_smooth_by_k[k],
                show_subject_traces=False,
                background_style=background_style,
                subject_curves=_subject_curves_by_k.get(k),
                weight_col=_weight_col,
                tick_ilds=ild_ticks,
                show_weighted_points=show_weighted_points,
                show_data_smooth=show_data_smooth,
                show_model_smooth=show_model_smooth,
                model_line_mode=model_line_mode,
                bin_points=True,
                n_bins=n_bins,
            )
        _ax_overlay.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        _ax_overlay.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
        _ax_overlay.set_ylim(0, 1)
        _ax_overlay.set_yticks([0, 0.5, 1])
        _ax_overlay.set_xlabel("Evidence strength")
        _ax_overlay.set_ylabel("p(left)")
        _ax_overlay.set_title("")
        _ax_overlay.legend(frameon=False, fontsize=8)

    if not overlay_only:
        for k, ax in enumerate(axes[int(_include_overlay) :]):
            lbl = slbls.get(k, f"State {k}")
            color = get_state_color(lbl, k, K=K)
            _weight_col = (
                f"_p_state_rank_{k}" if state_assignment_mode == "weighted" and f"_p_state_rank_{k}" in df_pd.columns else None
            )
            _df_state = df_pd if _weight_col is not None else df_pd[df_pd["_state_k"] == k]
            _psych_state_panel(
                ax,
                _df_state,
                ild_col,
                choice_col,
                pred_col=f"_pL_state_rank_{k}" if state_assignment_mode == "weighted" else pred_col,
                subj_col=subj_col,
                color=color,
                label=lbl,
                smooth_curve=_smooth_by_k[k],
                background_style=background_style,
                subject_curves=_subject_curves_by_k.get(k),
                weight_col=_weight_col,
                tick_ilds=ild_ticks,
                show_weighted_points=show_weighted_points,
                show_data_smooth=show_data_smooth,
                show_model_smooth=show_model_smooth,
                model_line_mode=model_line_mode,
                bin_points=True,
                n_bins=n_bins,
            )
            ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel("Evidence strength")
            ax.set_title(lbl)
            if k == 0:
                ax.set_ylabel("P(Left)")
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
    choice_col: str = _RESPONSE_COL,
    subj_col: str = "subject",
    X_cols: Optional[Sequence[str]] = None,
    feature_min: Optional[float] = None,
    feature_max: Optional[float] = None,
    background_style: str = "data",
    n_bins: int = 9,
    n_grid: int = 300,
    show_weighted_points: bool = True,
    show_data_smooth: bool = True,
    show_model_smooth: bool = True,
    figure_dpi: float = 80.0,
    overlay_only: bool = False,
    model_line_mode: str = "smooth",
    state_assignment_mode: str = "weighted",
) -> plt.Figure:
    """Per-state partial-dependence plot for any emission regressor.

    The x-axis is the chosen regressor (for example ``at_choice``). Empirical
    points are pooled within quantile bins of that regressor, while the model
    line sweeps the same regressor over a dense grid and marginalises over the
    empirical distribution of the remaining features.
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
    if state_assignment_mode == "weighted":
        df_pd = _attach_rank_posterior_cols(df_pd, views, subj_col=subj_col)
        df_pd = _attach_rank_state_model_cols(df_pd, views, subj_col=subj_col, base_col="pL_state")

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
    if overlay_only:
        _include_overlay = True
    _n_panels = K + int(_include_overlay)
    if overlay_only:
        _n_panels = 1
    panel_w, panel_h = _legacy_square_panel_size(n_cols=2)
    _figsize = (3, 3) if overlay_only else (panel_w * _n_panels, panel_h)
    fig, axes = plt.subplots(1, _n_panels, figsize=_figsize, sharey=True, dpi=figure_dpi)
    axes = np.atleast_1d(axes)

    xlabel = _feature_label(feature_col)

    if _include_overlay:
        _ax_overlay = axes[0]
        for k in range(K):
            lbl = slbls.get(k, f"State {k}")
            color = get_state_color(lbl, k, K=K)
            _weight_col = (
                f"_p_state_rank_{k}" if state_assignment_mode == "weighted" and f"_p_state_rank_{k}" in df_pd.columns else None
            )
            _df_state = df_pd if _weight_col is not None else df_pd[df_pd["_state_k"] == k]
            _regressor_state_panel(
                _ax_overlay,
                _df_state,
                feature_col,
                choice_col,
                pred_col=f"_pL_state_rank_{k}" if state_assignment_mode == "weighted" else "p_pred",
                subj_col=subj_col,
                color=color,
                label=lbl,
                smooth_curve=_smooth_by_k[k],
                show_subject_traces=False,
                background_style=background_style,
                subject_curves=_subject_curves_by_k.get(k),
                n_bins=n_bins,
                weight_col=_weight_col,
                show_weighted_points=show_weighted_points,
                show_data_smooth=show_data_smooth,
                show_model_smooth=show_model_smooth,
                model_line_mode=model_line_mode,
            )
        _ax_overlay.set_xlabel(xlabel)
        _ax_overlay.set_ylabel("p(left)")
        _ax_overlay.set_title("")
        _ax_overlay.legend(frameon=False, fontsize=8)

    if not overlay_only:
        for k, ax in enumerate(axes[int(_include_overlay) :]):
            lbl = slbls.get(k, f"State {k}")
            color = get_state_color(lbl, k, K=K)
            _weight_col = (
                f"_p_state_rank_{k}" if state_assignment_mode == "weighted" and f"_p_state_rank_{k}" in df_pd.columns else None
            )
            _df_state = df_pd if _weight_col is not None else df_pd[df_pd["_state_k"] == k]
            _regressor_state_panel(
                ax,
                _df_state,
                feature_col,
                choice_col,
                pred_col=f"_pL_state_rank_{k}" if state_assignment_mode == "weighted" else "p_pred",
                subj_col=subj_col,
                color=color,
                label=lbl,
                smooth_curve=_smooth_by_k[k],
                background_style=background_style,
                subject_curves=_subject_curves_by_k.get(k),
                n_bins=n_bins,
                weight_col=_weight_col,
                show_weighted_points=show_weighted_points,
                show_data_smooth=show_data_smooth,
                show_model_smooth=show_model_smooth,
                model_line_mode=model_line_mode,
            )
            ax.set_xlabel(xlabel)
            ax.set_title(lbl)
            if k == 0:
                ax.set_ylabel("P(Left)")
            else:
                ax.set_ylabel("")

    fig.suptitle(f"{model_name} — {_feature_label(feature_col)} psychometric", y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, None
