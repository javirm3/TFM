"""MCDR task-owned plots.

This module owns plots that depend on MCDR task semantics such as trial
difficulty, stimulus duration, delay duration, and side-stratified
performance. Shared model diagnostics are re-exported from
``glmhmmt.model_plots``.
"""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import t

import paths
from glmhmmt.model_plots import (
    _state_color,
    plot_emission_weights as _plot_emission_weights_generic,
    plot_emission_weights_by_subject as _plot_emission_weights_by_subject_generic,
    plot_posterior_probs,
    plot_change_triggered_posteriors_by_subject,
    plot_change_triggered_posteriors_summary,
    plot_session_deepdive,
    plot_session_trajectories,
    plot_state_accuracy,
    plot_state_dwell_times_by_subject,
    plot_state_dwell_times_summary,
    plot_state_dwell_times,
    plot_state_posterior_count_kde,
    plot_state_occupancy,
    plot_transition_matrix,
    plot_transition_matrix_by_subject,
    plot_tau_sweep,
    plot_transition_weights,
)

sns.set_style("white")

with paths.CONFIG.open("rb") as f:
    cfg = tomllib.load(f)


def _empty_plot(message: str = "No data") -> plt.Figure:
    """Return a minimal placeholder figure for empty selections."""
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    return fig


def _resolve_emission_plot_inputs(
    *,
    views: Optional[dict] = None,
    arrays_store: Optional[dict] = None,
    state_labels: Optional[dict] = None,
    names: Optional[dict] = None,
    subjects: Optional[Sequence[str]] = None,
) -> tuple[dict, dict, dict, list[str]]:
    """Normalize either `views` or legacy arrays inputs for emission plots."""
    if views is not None:
        arrays_from_views: dict = {}
        labels_from_views: dict = {}
        feat_names: list[str] = []

        for subj, view in views.items():
            if view is None or getattr(view, "emission_weights", None) is None:
                continue
            arrays_from_views[subj] = {
                "emission_weights": np.asarray(view.emission_weights),
                "X_cols": list(getattr(view, "feat_names", []) or []),
            }
            labels_from_views[subj] = {int(k): lbl for k, lbl in view.state_name_by_idx.items()}
            if not feat_names:
                feat_names = list(getattr(view, "feat_names", []) or [])

        return arrays_from_views, labels_from_views, {"X_cols": feat_names}, list(arrays_from_views.keys())

    if arrays_store is None:
        raise ValueError("Provide either `views` or `arrays_store` for emission plots.")
    if state_labels is None:
        raise ValueError("`state_labels` is required when `views` is not provided.")
    if names is None:
        raise ValueError("`names` is required when `views` is not provided.")

    resolved_subjects = list(subjects) if subjects is not None else list(arrays_store.keys())
    return arrays_store, state_labels, names, resolved_subjects


def _infer_emission_K(
    *,
    views: Optional[dict] = None,
    arrays_store: Optional[dict] = None,
    subjects: Optional[Sequence[str]] = None,
) -> int:
    """Infer K from views or the first available emission-weight array."""
    if views:
        first_view = next(iter(views.values()), None)
        if first_view is not None:
            return int(first_view.K)

    if arrays_store:
        candidate_subjects = list(subjects) if subjects is not None else list(arrays_store.keys())
        for subj in candidate_subjects:
            subj_arrays = arrays_store.get(subj, {})
            weights = subj_arrays.get("emission_weights")
            if weights is not None:
                return int(np.asarray(weights).shape[0])

    raise ValueError("Could not infer `K` for emission plots; pass it explicitly.")


def plot_emission_weights_by_subject(
    views: Optional[dict] = None,
    K: Optional[int] = None,
    save_path=None,
    *,
    arrays_store: Optional[dict] = None,
    state_labels: Optional[dict] = None,
    names: Optional[dict] = None,
    subjects: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Per-subject emission bars with either `views` or legacy arrays inputs."""
    arrays_store, state_labels, names, subjects = _resolve_emission_plot_inputs(
        views=views,
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        subjects=subjects,
    )
    if not subjects:
        return _empty_plot()

    K = int(K) if K is not None else _infer_emission_K(
        views=views,
        arrays_store=arrays_store,
        subjects=subjects,
    )
    return _plot_emission_weights_by_subject_generic(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        K=K,
        subjects=subjects,
        save_path=save_path,
    )


def plot_emission_weights_summary(
    views: Optional[dict] = None,
    K: Optional[int] = None,
    save_path=None,
    *,
    arrays_store: Optional[dict] = None,
    state_labels: Optional[dict] = None,
    names: Optional[dict] = None,
    subjects: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Notebook-friendly high-level emission summary, aligned with 2AFC API."""
    _ = save_path
    arrays_store, state_labels, names, subjects = _resolve_emission_plot_inputs(
        views=views,
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        subjects=subjects,
    )
    if not subjects:
        return _empty_plot()

    K = int(K) if K is not None else _infer_emission_K(
        views=views,
        arrays_store=arrays_store,
        subjects=subjects,
    )
    fig_summary, fig_detail = _plot_emission_weights_generic(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        K=K,
        subjects=subjects,
    )
    plt.close(fig_detail)
    return fig_summary


def plot_emission_weights(
    views: Optional[dict] = None,
    K: Optional[int] = None,
    save_path=None,
    *,
    arrays_store: Optional[dict] = None,
    state_labels: Optional[dict] = None,
    names: Optional[dict] = None,
    subjects: Optional[Sequence[str]] = None,
):
    """Emission summaries with `views` support and backward-compatible kwargs."""
    _ = save_path
    arrays_store, state_labels, names, subjects = _resolve_emission_plot_inputs(
        views=views,
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        subjects=subjects,
    )
    if not subjects:
        return _empty_plot(), _empty_plot()

    K = int(K) if K is not None else _infer_emission_K(
        views=views,
        arrays_store=arrays_store,
        subjects=subjects,
    )
    return _plot_emission_weights_generic(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        K=K,
        subjects=subjects,
    )


def truncate_colormap(cmap_name, minval=0.2, maxval=0.9, n=256):
    """Return a colormap truncated to a subrange."""
    cmap = cm.get_cmap(cmap_name, n)
    return colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )


def get_plot_path(subfolder: str, fname: str, model_name: str) -> Path:
    out_dir = Path("results") / "plots" / model_name / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname


def prepare_predictions_df(df_pred: pl.DataFrame) -> pl.DataFrame:
    df = df_pred.clone()

    if "correct_bool" not in df.columns:
        if "performance" in df.columns:
            df = df.with_columns(pl.col("performance").cast(pl.Boolean).alias("correct_bool"))
        else:
            raise ValueError("No encuentro 'performance' ni 'correct_bool' en df.")

    for col in ["pL", "pC", "pR"]:
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en df (predicciones por trial).")

    if "response" not in df.columns:
        raise ValueError("Falta la columna 'response' (0/1/2) en df.")

    if "p_model_correct" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("stimulus") == 0)
            .then(pl.col("pL"))
            .when(pl.col("stimulus") == 1)
            .then(pl.col("pC"))
            .when(pl.col("stimulus") == 2)
            .then(pl.col("pR"))
            .otherwise(None)
            .alias("p_model_correct")
        )

    if "stimd_c" not in df.columns:
        if "stimd_n" in df.columns:
            df = df.with_columns(pl.col("stimd_n").replace(cfg["encoding"]["stimd"], default=None).alias("stimd_c"))
        else:
            raise ValueError("Falta 'stimd_c' y no existe 'stimd_n' para mapear.")

    if "ttype_c" not in df.columns:
        if "ttype_n" in df.columns:
            df = df.with_columns(pl.col("ttype_n").replace(cfg["encoding"]["ttype"], default=None).alias("ttype_c"))
        else:
            raise ValueError("Falta 'ttype_c' y no existe 'ttype_n' para mapear.")

    return df


def plot_cat_panel(ax, df, group_col, order, title, xlabel, ylabel=None, palette=None, labels=None):
    subj = (
        df.filter(pl.col(group_col).is_in(order))
        .group_by([group_col, "subject"])
        .agg(
            [
                pl.col("correct_bool").mean().alias("correct_mean"),
                pl.col("p_model_correct").mean().alias("model_mean"),
            ]
        )
    )
    if subj.height == 0:
        ax.set_visible(False)
        return

    g = (
        subj.group_by(group_col)
        .agg(
            [
                pl.col("correct_mean").mean().alias("md"),
                pl.col("correct_mean").std(ddof=1).alias("sd"),
                pl.col("correct_mean").count().alias("nd"),
                pl.col("model_mean").mean().alias("mm"),
                pl.col("model_mean").std(ddof=1).alias("sm"),
                pl.col("model_mean").count().alias("nm"),
            ]
        )
        .with_columns(
            [
                pl.col("nd").clip(lower_bound=1),
                pl.col("nm").clip(lower_bound=1),
            ]
        )
        .with_columns(pl.col(group_col).cast(pl.Categorical).alias(group_col))
    )

    rows = {r[group_col]: r for r in g.to_dicts()}
    cats = [c for c in order if c in rows]
    md = np.array([rows[c]["md"] for c in cats])
    sd = np.array([rows[c]["sd"] for c in cats])
    nd = np.array([rows[c]["nd"] for c in cats])
    mm = np.array([rows[c]["mm"] for c in cats])
    sm = np.array([rows[c]["sm"] for c in cats])

    ax.plot(np.arange(len(cats)), mm, "-", color="black", lw=2, label="Model")

    colors_used = palette if palette else ["black"] * len(cats)
    if df["subject"].n_unique() > 1:
        ax.fill_between(np.arange(len(cats)), mm - sm, mm + sm, color="black", alpha=0.12)
        for i, (xpos, yval, err) in enumerate(zip(np.arange(len(cats)), md, sd)):
            ax.errorbar(xpos, yval, yerr=err, fmt="o", color=colors_used[i], ms=7, capsize=3)
    else:
        for i, (xpos, yval) in enumerate(zip(np.arange(len(cats)), md)):
            ax.errorbar(xpos, yval, fmt="o", color=colors_used[i], ms=7, capsize=3)

    ax.set_xticks(np.arange(len(cats)))
    if labels:
        label_map = dict(zip(order, labels))
        tick_labels = [label_map.get(c, c) for c in cats]
    else:
        tick_labels = cats
    ax.set_xticklabels(tick_labels)

    ax.set_ylim(0.2, 1.05)
    ax.axhspan(0, 1 / 3, color="gray", alpha=0.15)
    ax.set_xlim(left=-0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _plot_state_panel(ax, df_state, group_col, order, color):
    subj = (
        df_state.filter(pl.col(group_col).is_in(order))
        .group_by([group_col, "subject"])
        .agg(
            [
                pl.col("correct_bool").mean().alias("acc"),
                pl.col("p_model_correct").mean().alias("model"),
            ]
        )
    )
    if subj.height == 0:
        return None, None

    agg = subj.group_by(group_col).agg(
        [
            pl.col("acc").mean().alias("md"),
            pl.col("acc").std(ddof=1).alias("sd"),
            pl.col("model").mean().alias("mm"),
            pl.col("model").std(ddof=1).alias("sm"),
        ]
    )
    rows = {r[group_col]: r for r in agg.to_dicts()}
    cats = [c for c in order if c in rows]
    if not cats:
        return None, None

    xpos = np.array([order.index(c) for c in cats])
    md = np.array([rows[c]["md"] for c in cats])
    sd = np.array([rows[c]["sd"] for c in cats])
    mm = np.array([rows[c]["mm"] for c in cats])
    sm = np.array([rows[c]["sm"] for c in cats])
    n_subj = subj["subject"].n_unique()

    data_h = None
    for i, (x, y) in enumerate(zip(xpos, md)):
        eb = ax.errorbar(
            x,
            y,
            yerr=sd[i] if n_subj > 1 else None,
            fmt="o",
            color=color,
            ms=7,
            capsize=3,
            alpha=0.55,
            zorder=5,
            label="_nolegend_",
        )
        if data_h is None:
            data_h = eb

    (model_h,) = ax.plot(
        xpos,
        mm,
        "-",
        color=color,
        lw=2.2,
        alpha=0.95,
        zorder=6,
        label="_nolegend_",
    )
    if n_subj > 1:
        ax.fill_between(xpos, mm - sm, mm + sm, color=color, alpha=0.10, zorder=3)

    return data_h, model_h


def plot_categorical_performance_by_state(
    df,
    views: dict,
    model_name: str,
    background_style: str = "data",
    show_weighted_points: bool = True,
    show_data_smooth: bool = True,
    show_model_smooth: bool = True,
    figure_dpi: float = 80.0,
    overlay_only: bool = False,
    model_line_mode: str = "smooth",
    state_assignment_mode: str = "weighted",
):
    # Accepted for notebook API compatibility; MCDR keeps its current plot style.
    _ = (
        background_style,
        show_weighted_points,
        show_data_smooth,
        show_model_smooth,
        figure_dpi,
        overlay_only,
        model_line_mode,
        state_assignment_mode,
    )
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    if "state_rank" not in df.columns:
        raise ValueError("df must contain 'state_rank' (from build_trial_df).")

    K = next(iter(views.values())).K if views else int(df["state_rank"].max()) + 1

    state_labels = {}
    for v in views.values():
        for raw_idx, lbl in v.state_name_by_idx.items():
            rank = v.state_rank_by_idx[int(raw_idx)]
            state_labels.setdefault(rank, lbl)

    df = df.with_columns(pl.col("state_rank").cast(pl.Int64).alias("_state_k"))

    state_colors = {k: _state_color(state_labels.get(k, f"State {k}"), k) for k in range(K)}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    ax1, ax2, ax3 = axes
    panels = [
        (
            ax1,
            df,
            "ttype_c",
            cfg["plots"]["ttype"]["order"],
            "a) Trial difficulty",
            "Trial difficulty",
            cfg["plots"]["ttype"]["labels"],
        ),
        (
            ax2,
            df.filter(pl.col("ttype_c") == "DS"),
            "stimd_c",
            cfg["plots"]["stimd"]["order"],
            "b) Stim duration",
            "Stimulus type",
            cfg["plots"]["stimd"]["labels"],
        ),
        (
            ax3,
            df.filter(pl.col("stimd_c") == "SS"),
            "ttype_c",
            cfg["plots"]["delay"]["order"],
            "c) Delay duration",
            "Delay type",
            cfg["plots"]["delay"]["labels"],
        ),
    ]

    for ax, df_panel, gcol, order, title, xlabel, labels in panels:
        for k in range(K):
            df_k = df_panel.filter(pl.col("_state_k") == k)
            _plot_state_panel(ax, df_k, gcol, order, color=state_colors[k])

        cats = [c for c in order if df_panel.filter(pl.col(gcol) == c).height > 0]
        if labels:
            label_map = dict(zip(order, labels))
            tick_labels = [label_map.get(c, c) for c in cats]
        else:
            tick_labels = cats
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0.2, 1.05)
        ax.axhspan(0, 1 / 3, color="gray", alpha=0.15)
        ax.set_xlim(left=-0.4)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ax is ax1:
            ax.set_ylabel("Accuracy")

    import matplotlib.lines as mlines

    legend_handles = []
    legend_labels = []
    for k in range(K):
        label = state_labels.get(k, f"State {k}")
        color = state_colors[k]
        legend_handles.append(mlines.Line2D([], [], marker="o", color=color, linestyle="None", ms=7, alpha=0.55))
        legend_labels.append(f"{label} data")
        legend_handles.append(mlines.Line2D([], [], color=color, lw=2.2, alpha=0.95))
        legend_labels.append(f"{label} model")

    ax3.legend(legend_handles, legend_labels, fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle(model_name, y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, axes


def plot_categorical_performance_all(
    df,
    model_name: str,
    views: Optional[dict] = None,
    X_cols: Optional[Sequence[str]] = None,
    ild_max: Optional[float] = None,
    background_style: str = "data",
):
    # Accepted for notebook API compatibility; MCDR keeps its current plot style.
    _ = (views, X_cols, ild_max, background_style)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    ax1, ax2, ax3 = axes
    df = df.drop("p_model_correct").rename({"p_model_correct_marginal": "p_model_correct"})

    plot_cat_panel(
        ax1,
        df.clone(),
        "ttype_c",
        cfg["plots"]["ttype"]["order"],
        title="a) Trial difficulty",
        xlabel="Trial difficulty",
        ylabel="Accuracy",
        palette=cfg["plots"]["ttype"]["palette"],
        labels=cfg["plots"]["ttype"]["labels"],
    )
    plot_cat_panel(
        ax2,
        df.filter(pl.col("ttype_c") == "DS"),
        "stimd_c",
        cfg["plots"]["stimd"]["order"],
        title="b) Stim duration",
        xlabel="Stimulus type",
        palette=cfg["plots"]["stimd"]["palette"],
        labels=cfg["plots"]["stimd"]["labels"],
    )
    plot_cat_panel(
        ax3,
        df.filter(pl.col("stimd_c") == "SS"),
        "ttype_c",
        cfg["plots"]["delay"]["order"],
        title="c) Delay duration",
        xlabel="Delay type",
        palette=cfg["plots"]["delay"]["palette"],
        labels=cfg["plots"]["delay"]["labels"],
    )
    sns.despine()
    fig.tight_layout()
    return fig, axes


def plot_delay_or_stim_1d_on_ax(ax, df, subject, n_bins, which):
    """Plot delay or stimulus duration accuracy for one subject."""
    df = df.to_pandas()
    df_delay = df[df["stimd_c"] == "SS"]
    df_stim = df.copy()

    if subject is not None:
        df_delay = df_delay[df_delay["subject"] == subject].copy()
        df_stim = df_stim[df_stim["subject"] == subject].copy()

    needed_cols = ["delay_d", "correct_bool", "p_model_correct", "subject", "stim_d"]
    df_delay = df_delay.dropna(subset=needed_cols)
    df_stim = df_stim.dropna(subset=needed_cols)

    if which == "delay":
        d = df_delay
        xcol = "delay_d"
        xlabel = "Delay duration"
        title_suffix = "Delay"
        band_floor = 1 / 3
        palette_data = truncate_colormap("Purples_r", 0, 0.7)
    elif which == "stim":
        d = df_stim
        xcol = "stim_d"
        xlabel = "Stimulus duration"
        title_suffix = "Stimulus"
        band_floor = 1 / 3
        palette_data = truncate_colormap("Oranges", 0.3, 1.0)
    else:
        raise ValueError("which must be 'delay' or 'stim'")

    if d.empty:
        ax.set_title(f"{subject} - {title_suffix}\n(no data)", fontsize=9)
        ax.axis("off")
        return False

    d = d.copy()
    d["x_bin"], _ = pd.qcut(d[xcol], q=n_bins, retbins=True, duplicates="drop")
    centers = d.groupby("x_bin", observed=True)[xcol].median().rename("center").reset_index().sort_values("center")

    subj = (
        d.groupby(["x_bin", "subject"], observed=True)
        .agg(data_acc=("correct_bool", "mean"), model_acc=("p_model_correct", "mean"))
        .reset_index()
        .merge(centers, on="x_bin", how="left")
    )

    plot_df = subj.melt(
        id_vars=["x_bin", "subject", "center"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_df["kind"] = plot_df["kind"].map({"data_acc": "Data", "model_acc": "Model"})

    sns.lineplot(
        data=plot_df[plot_df["kind"] == "Model"],
        x="center",
        y="acc",
        color="gray",
        linestyle="-",
        errorbar=("ci", 95),
        err_style="band",
        ax=ax,
    )
    sns.lineplot(
        data=plot_df[plot_df["kind"] == "Data"],
        x="center",
        y="acc",
        hue="center",
        palette=palette_data,
        marker="o",
        linewidth=0,
        errorbar=("ci", 95),
        err_style="bars",
        legend=False,
        ax=ax,
        zorder=10,
    )

    ax.axhspan(0, band_floor, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frac. correct responses", fontsize=12)
    ax.set_title(f"{subject}", fontsize=12)
    ax.tick_params(labelsize=12)
    sns.despine()
    return True


def plot_categorical_strat_by_side(
    df,
    subject,
    model_name,
    df_silent=None,
    cond_col="stimd_c",
    cond_order=("VG", "SL", "SM", "SS", "SIL"),
    cond_labels=("Visual", "Easy", "Medium", "Hard", "Silent"),
):
    df = df.to_pandas().copy()
    df["x_c"] = df["x_c"].astype("string").str.strip().str.upper()

    if cond_order is None:
        cond_order = sorted(df[cond_col].dropna().unique())
    if cond_labels is None:
        cond_labels = cond_order

    g = (
        df.groupby([cond_col, "x_c"], observed=True)
        .agg(
            data_mean=("correct_bool", "mean"),
            model_mean=("p_model_correct", "mean"),
            n=("correct_bool", "size"),
        )
        .reset_index()
    )
    g["data_sem"] = np.sqrt(g["data_mean"] * (1.0 - g["data_mean"]) / g["n"].clip(lower=1))

    if df_silent is not None:
        df_s = df_silent.copy()
        p_silent = {"L": df_s["pL_mean"], "C": df_s["pC_mean"], "R": df_s["pR_mean"]}

    cond_to_x = {c: i for i, c in enumerate(cond_order)}
    g["x_pos"] = g[cond_col].map(cond_to_x)

    side_palette = {"L": "#e41a1c", "C": "#4daf4a", "R": "#377eb8"}
    fig, ax = plt.subplots(figsize=(4, 4))

    for side in ["L", "C", "R"]:
        sub = g[g["x_c"] == side].dropna(subset=["x_pos"]).sort_values("x_pos")
        if sub.empty:
            continue
        ax.plot(
            sub["x_pos"],
            sub["model_mean"],
            "-",
            lw=2,
            color=side_palette.get(side, "gray"),
            label=f"Model {side}",
            zorder=2,
        )
        ax.errorbar(
            sub["x_pos"],
            sub["data_mean"],
            yerr=sub["data_sem"],
            fmt="o",
            ms=5,
            capsize=3,
            color=side_palette.get(side, "gray"),
            linestyle="none",
            label=f"Data {side}",
            zorder=3,
        )
        if df_silent is not None:
            ax.plot(
                len(cond_order) - 1,
                p_silent[side],
                marker="D",
                ms=7,
                color=side_palette[side],
                linestyle="none",
                zorder=4,
            )

    ax.axhspan(0, 1 / 3, color="gray", alpha=0.15, zorder=0)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_labels)
    ax.set_ylim(0.2, 1.05)
    ax.set_ylabel("Frac. correct responses")
    ax.set_xlabel("Trial difficulty")
    ax.set_title(f"{subject}")
    sns.despine()
    fig.tight_layout()
    return fig, ax


def plot_delay_binned_1d(df, model_name, subject=None, n_bins=7):
    df = df.to_pandas()
    df_delay = df[df["stimd_c"] == "SS"]
    df_stim = df[df["ttype_c"] == "DS"].copy()

    if subject is not None:
        df_delay = df_delay[df_delay["subject"] == subject].copy()
        df_stim = df_stim[df_stim["subject"] == subject].copy()

    needed_cols = ["delay_d", "correct_bool", "p_model_correct", "subject", "stim_d"]
    df_delay = df_delay.dropna(subset=needed_cols)
    df_stim = df_stim.dropna(subset=needed_cols)
    if df_delay.empty or df_stim.empty:
        return None

    df_delay["delay_bin"] = df_delay.groupby("ttype_c", observed=True)["delay_d"].transform(
        lambda s: pd.qcut(s, q=n_bins, duplicates="drop")
    )
    centers_delay = (
        df_delay.groupby(["ttype_c", "delay_bin"], observed=True)["delay_d"].median().rename("center").reset_index()
    )
    subj_delay = (
        df_delay.groupby(["ttype_c", "delay_bin", "subject"], observed=True)
        .agg(data_acc=("correct_bool", "mean"), model_acc=("p_model_correct", "mean"))
        .reset_index()
        .merge(centers_delay, on=["ttype_c", "delay_bin"], how="left")
    )

    df_stim["stim_bin"] = df_stim.groupby("stimd_c", observed=True)["stim_d"].transform(
        lambda s: pd.qcut(s, q=n_bins, duplicates="drop")
    )
    centers_stim = (
        df_stim.groupby(["stimd_c", "stim_bin"], observed=True)["stim_d"].median().rename("center").reset_index()
    )
    subj_stim = (
        df_stim.groupby(["stimd_c", "stim_bin", "subject"], observed=True)
        .agg(data_acc=("correct_bool", "mean"), model_acc=("p_model_correct", "mean"))
        .reset_index()
        .merge(centers_stim, on=["stimd_c", "stim_bin"], how="left")
    )

    plot_delay = subj_delay.melt(
        id_vars=["delay_bin", "subject", "ttype_c", "center"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_delay["kind"] = plot_delay["kind"].map({"data_acc": "Data", "model_acc": "Model"})

    plot_stim = subj_stim.melt(
        id_vars=["stim_bin", "subject", "center", "stimd_c"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_stim["kind"] = plot_stim["kind"].map({"data_acc": "Data", "model_acc": "Model"})

    fig_delay, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(
        data=plot_delay[plot_delay["kind"] == "Model"],
        x="center",
        y="acc",
        color="gray",
        hue="ttype_c",
        linestyle="-",
        errorbar=("ci", 95),
        err_style="band",
        ax=ax,
    )
    sns.lineplot(
        x="center",
        y="acc",
        hue="ttype_c",
        data=plot_delay[plot_delay["kind"] == "Data"],
        errorbar=("ci", 95),
        err_style="bars",
        marker="o",
        linewidth=0,
        ax=ax,
        zorder=10,
        legend=False,
    )
    ax.axhspan(0, 1 / 3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Delay duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")
    title_subj = subject if subject is not None else "All subjects"
    ax.set_title(f"{title_subj} - Delay (1D)")
    sns.despine()
    fig_delay.tight_layout()

    fig_stim, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(
        data=plot_stim[plot_stim["kind"] == "Model"],
        x="center",
        y="acc",
        color="gray",
        hue="stimd_c",
        linestyle="-",
        errorbar=("ci", 95),
        err_style="band",
        ax=ax,
    )
    sns.lineplot(
        x="center",
        y="acc",
        hue="stimd_c",
        data=plot_stim[plot_stim["kind"] == "Data"],
        errorbar=("ci", 95),
        err_style="bars",
        marker="o",
        linewidth=0,
        ax=ax,
        zorder=10,
        legend=False,
    )
    ax.axhspan(0, 1 / 3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Stimulus duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")
    ax.set_title(f"{title_subj} - Stimulus (1D)")
    sns.despine()
    fig_stim.tight_layout()

    return fig_delay, fig_stim
