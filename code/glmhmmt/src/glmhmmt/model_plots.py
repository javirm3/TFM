import paths

import math
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from scipy.special import softmax as _softmax
import tomllib
from matplotlib import cm, colors
from typing import Tuple
from glmhmmt.views import get_state_palette as _get_config_state_palette
from glmhmmt.plots_common import (
    plot_state_accuracy as _plot_state_accuracy_common,
    plot_change_triggered_posteriors_by_subject as _plot_change_triggered_posteriors_by_subject_common,
    plot_change_triggered_posteriors_summary as _plot_change_triggered_posteriors_summary_common,
    plot_state_posterior_count_kde as _plot_state_posterior_count_kde_common,
    plot_session_trajectories as _plot_session_trajectories_common,
    plot_state_occupancy as _plot_state_occupancy_common,
    plot_state_dwell_times_by_subject as _plot_state_dwell_times_by_subject_common,
    plot_state_dwell_times_summary as _plot_state_dwell_times_summary_common,
    plot_state_dwell_times as _plot_state_dwell_times_common,
    plot_session_deepdive as _plot_session_deepdive_common,
)
sns.set_style("white")

with paths.CONFIG.open("rb") as f:
        cfg = tomllib.load(f)

def truncate_colormap(cmap_name, minval=0.2, maxval=0.9, n=256):
    """Trunca un colormap a un subrango."""
    cmap = cm.get_cmap(cmap_name, n)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

def get_plot_path(subfolder: str, fname: str, model_name: str) -> Path:
    out_dir = Path("results") / "plots" / model_name / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname

def prepare_predictions_df(df_pred: pl.DataFrame) -> pl.DataFrame:
    df = df_pred.clone()

    if "correct_bool" not in df.columns:
        if "performance" in df.columns:
            df = df.with_columns(
                pl.col("performance").cast(pl.Boolean).alias("correct_bool")
            )
        else:
            raise ValueError("No encuentro 'performance' ni 'correct_bool' en df.")

    for col in ["pL", "pC", "pR"]:
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en df (predicciones por trial).")

    if "response" not in df.columns:
        raise ValueError("Falta la columna 'response' (0/1/2) en df.")

    if "p_model_correct" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("stimulus") == 0).then(pl.col("pL"))
            .when(pl.col("stimulus") == 1).then(pl.col("pC"))
            .when(pl.col("stimulus") == 2).then(pl.col("pR"))
            .otherwise(None)
            .alias("p_model_correct")
        )

    if "stimd_c" not in df.columns:
        if "stimd_n" in df.columns:
            df = df.with_columns(
                pl.col("stimd_n")
                .replace(cfg["encoding"]["stimd"], default=None)
                .alias("stimd_c")
            )
        else:
            raise ValueError("Falta 'stimd_c' y no existe 'stimd_n' para mapear.")

    if "ttype_c" not in df.columns:
        if "ttype_n" in df.columns:
            df = df.with_columns(
                pl.col("ttype_n")
                .replace(cfg["encoding"]["ttype"], default=None)
                .alias("ttype_c")
            )
        else:
            raise ValueError("Falta 'ttype_c' y no existe 'ttype_n' para mapear.")

    return df

def plot_cat_panel(ax, df, group_col, order, title, xlabel, ylabel=None, palette=None, labels=None):
    
    subj = (df.filter(pl.col(group_col).is_in(order)).group_by([group_col, "subject"])
            .agg([
                pl.col("correct_bool").mean().alias("correct_mean"),
                pl.col("p_model_correct").mean().alias("model_mean"),
                ]))
    if subj.height == 0:
        ax.set_visible(False)
        return

    g = (
        subj.group_by(group_col)
            .agg([
                pl.col("correct_mean").mean().alias("md"),
                pl.col("correct_mean").std(ddof=1).alias("sd"),
                pl.col("correct_mean").count().alias("nd"),
                pl.col("model_mean").mean().alias("mm"),
                pl.col("model_mean").std(ddof=1).alias("sm"),
                pl.col("model_mean").count().alias("nm"),
            ])
    )

    g = g.with_columns([
    pl.col("nd").clip(lower_bound=1),
    pl.col("nm").clip(lower_bound=1),
    ])

    # reordenar
    g = g.with_columns(pl.col(group_col).cast(pl.Categorical).alias(group_col))

    rows = {r[group_col]: r for r in g.to_dicts()}
    cats = [c for c in order if c in rows]
    md = np.array([rows[c]["md"] for c in cats])
    sd = np.array([rows[c]["sd"] for c in cats])
    nd = np.array([rows[c]["nd"] for c in cats])
    mm = np.array([rows[c]["mm"] for c in cats])
    sm = np.array([rows[c]["sm"] for c in cats])
    nm = np.array([rows[c]["nm"] for c in cats])

    # Si quieres también poner el modelo como línea:
    ax.plot(np.arange(len(cats)), mm, "-", color="black", lw=2, label="Model")
    

    colors = palette if palette else ["black"] * np.arange(len(cats))
    if (df["subject"].unique().shape[0] > 1 ):
        ax.fill_between(np.arange(len(cats)), mm-sm, mm+sm, color="black", alpha=0.12)
        sem_d = sd / np.sqrt(nd)
        sem_m = sm / np.sqrt(nm)
        ci_d  = sem_d * t.ppf(0.975, nd-1)
        ci_m  = sem_m * t.ppf(0.975, nm-1)
        for i, (xpos, yval, err) in enumerate(zip(np.arange(len(cats)), md, sd)):
            ax.errorbar(xpos, yval, yerr=err, fmt="o",
                        color=colors[i], ms=7, capsize=3)
    else: 
        for i, (xpos, yval) in enumerate(zip(np.arange(len(cats)), md)):
            ax.errorbar(xpos, yval, fmt="o",
                        color=colors[i], ms=7, capsize=3)

    ax.set_xticks(np.arange(len(cats)))
    # align labels to the subset of categories actually present in this panel
    if labels:
        _label_map = dict(zip(order, labels))
        _tick_labels = [_label_map.get(c, c) for c in cats]
    else:
        _tick_labels = cats
    ax.set_xticklabels(_tick_labels)

    ax.set_ylim(0.2, 1.05)
    ax.axhspan(0, 1/3, color="gray", alpha=0.15)
    ax.set_xlim(left=-0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _plot_state_panel(ax, df_state, group_col, order, color, label):
    """
    Draw:
      - Data dots  : mean(correct_bool) ± sd per category, MAP-assigned trials
      - Model line : mean(p_model_correct) per category, same trial subset
    Both in `color`.  Returns (data_errorbar_container, model_line) for legend
    building, or (None, None) if the subset is empty.
    """
    subj = (
        df_state
        .filter(pl.col(group_col).is_in(order))
        .group_by([group_col, "subject"])
        .agg([
            pl.col("correct_bool").mean().alias("acc"),
            pl.col("p_model_correct").mean().alias("model"),
        ])
    )
    if subj.height == 0:
        return None, None

    agg = (
        subj.group_by(group_col)
        .agg([
            pl.col("acc").mean().alias("md"),
            pl.col("acc").std(ddof=1).alias("sd"),
            pl.col("model").mean().alias("mm"),
            pl.col("model").std(ddof=1).alias("sm"),
        ])
    )
    rows   = {r[group_col]: r for r in agg.to_dicts()}
    cats   = [c for c in order if c in rows]
    if not cats:
        return None, None
    xpos = np.array([order.index(c) for c in cats])
    md     = np.array([rows[c]["md"] for c in cats])
    sd     = np.array([rows[c]["sd"] for c in cats])
    mm     = np.array([rows[c]["mm"] for c in cats])
    sm     = np.array([rows[c]["sm"] for c in cats])
    n_subj = subj["subject"].n_unique()

    # ── data dots ─────────────────────────────────────────────────────────────
    data_h = None
    for i, (x, y) in enumerate(zip(xpos, md)):
        eb = ax.errorbar(
            x, y,
            yerr=sd[i] if n_subj > 1 else None,
            fmt="o", color=color, ms=7, capsize=3,
            alpha=0.55, zorder=5, label="_nolegend_",
        )
        if data_h is None:
            data_h = eb

    # ── model prediction line ─────────────────────────────────────────────────
    (model_h,) = ax.plot(
        xpos, mm, "-", color=color, lw=2.2, alpha=0.95,
        zorder=6, label="_nolegend_",
    )
    if n_subj > 1:
        ax.fill_between(xpos, mm - sm, mm + sm, color=color, alpha=0.10, zorder=3)

    return data_h, model_h


def plot_categorical_performance_by_state(
    df,
    views: dict,
    model_name: str,
):
    """
    Plot per-state categorical performance: dots + line per state,  no pooled
    overlay.  Supports multi-subject DataFrames when `state_assign` is provided.

    Parameters
    ----------
    df            : prepared predictions DataFrame (prepare_predictions_df output)
    smoothed_probs: (T, K) array, ignored when state_assign is given
    state_labels  : {rank_idx: label_str}  e.g. {0: "Engaged", 1: "Disengaged"}
                    For pooled multi-subject calls use normalised rank indices
                    (0=Engaged, 1=Disengaged, …).
    model_name    : string for figure suptitle
    state_assign  : optional pre-computed (T,) int array of normalised state
                    ranks (0=Engaged, 1=Disengaged, …).  If provided,
                    smoothed_probs is ignored.
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    if "state_rank" not in df.columns:
        raise ValueError("df must contain 'state_rank' (from build_trial_df).")

    # resolve K
    K = next(iter(views.values())).K if views else int(df["state_rank"].max()) + 1

    # resolve labels by rank
    state_labels = {}
    for v in views.values():
        for raw_idx, lbl in v.state_name_by_idx.items():
            rank = v.state_rank_by_idx[int(raw_idx)]
            state_labels.setdefault(rank, lbl)

    df = df.with_columns(pl.col("state_rank").cast(pl.Int64).alias("_state_k"))

    _state_colors = {
        k: _state_color(state_labels.get(k, f"State {k}"), k)
        for k in range(K)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    ax1, ax2, ax3 = axes

    panels = [
        (ax1, df,                                   "ttype_c", cfg["plots"]["ttype"]["order"],
         "a) Trial difficulty", "Trial difficulty",  cfg["plots"]["ttype"]["labels"]),
        (ax2, df.filter(pl.col("ttype_c") == "DS"), "stimd_c", cfg["plots"]["stimd"]["order"],
         "b) Stim duration",    "Stimulus type",     cfg["plots"]["stimd"]["labels"]),
        (ax3, df.filter(pl.col("stimd_c") == "SS"), "ttype_c", cfg["plots"]["delay"]["order"],
         "c) Delay duration",   "Delay type",        cfg["plots"]["delay"]["labels"]),
    ]

    # collect handles for legend (first panel that has data for each state)
    _data_handles  = {}   # k -> first data errorbar container
    _model_handles = {}   # k -> first model line

    for ax, df_panel, gcol, order, title, xlabel, labels in panels:
        # per-state dots + line (no pooled layer)
        for k in range(K):
            df_k = df_panel.filter(pl.col("_state_k") == k)
            d_h, m_h = _plot_state_panel(
                ax, df_k, gcol, order,
                color=_state_colors[k],
                label=state_labels.get(k, f"State {k}"),
            )
            if k not in _data_handles and d_h is not None:
                _data_handles[k]  = d_h
                _model_handles[k] = m_h

        # axis decoration based on categories present across all states
        _cats = [c for c in order
                 if df_panel.filter(pl.col(gcol) == c).height > 0]
        if labels:
            _lmap = dict(zip(order, labels))
            _tick_labels = [_lmap.get(c, c) for c in _cats]
        else:
            _tick_labels = _cats
        ax.set_xticks(np.arange(len(_cats)))
        ax.set_xticklabels(_tick_labels)
        ax.set_ylim(0.2, 1.05)
        ax.axhspan(0, 1 / 3, color="gray", alpha=0.15)
        ax.set_xlim(left=-0.4)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ax is ax1:
            ax.set_ylabel("Accuracy")

    # ── shared legend: data dots then model lines, grouped by state ───────────
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    legend_handles = []
    legend_labels  = []
    for k in range(K):
        _lbl   = state_labels.get(k, f"State {k}")
        _color = _state_colors[k]
        legend_handles.append(
            mlines.Line2D([], [], marker="o", color=_color, linestyle="None",
                          ms=7, alpha=0.55, label=f"{_lbl} data")
        )
        legend_labels.append(f"{_lbl} data")
        legend_handles.append(
            mlines.Line2D([], [], color=_color, lw=2.2, alpha=0.95,
                          label=f"{_lbl} model")
        )
        legend_labels.append(f"{_lbl} model")

    ax3.legend(legend_handles, legend_labels, fontsize=8, frameon=False,
               bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle(model_name, y=1.02)
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig, axes


def plot_categorical_performance_all(df, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    ax1, ax2, ax3 = axes
    df = df.drop("p_model_correct").rename({"p_model_correct_marginal": "p_model_correct"})
    df_a = df.clone()
    
    plot_cat_panel(ax1, df_a, "ttype_c", cfg["plots"]["ttype"]["order"],
                    title="a) Trial difficulty",
                    xlabel="Trial difficulty",
                    ylabel="Accuracy",
                    palette=cfg["plots"]["ttype"]["palette"], labels=cfg["plots"]["ttype"]["labels"])
    # b) Stim duration (DS, SS/SM/SL)
    df_b = df.filter(pl.col("ttype_c") == "DS")
    plot_cat_panel(ax2, df_b, "stimd_c", cfg["plots"]["stimd"]["order"],
                    title="b) Stim duration",
                    xlabel="Stimulus type",
                    palette=cfg["plots"]["stimd"]["palette"], labels=cfg["plots"]["stimd"]["labels"])

    # c) Delay duration (SS)
    df_c = df.filter(pl.col("stimd_c") == "SS")
    plot_cat_panel(ax3, df_c, "ttype_c", cfg["plots"]["delay"]["order"],
                    title="c) Delay duration",
                    xlabel="Delay type",
                    palette=cfg["plots"]["delay"]["palette"], labels=cfg["plots"]["delay"]["labels"])
    sns.despine()
    fig.tight_layout()
    return fig, axes


def plot_delay_or_stim_1d_on_ax( ax, df, subject, n_bins, which):
    """
    Makes the delay or stim duration plot for a single subject on the given axis.
    - which: "delay" or "stim"
    returns True if it plotted something, False if no data for that subject/condition (in which case the panel is left blank with a title indicating no data).
    """
    df = df.to_pandas()
    df_delay = df[df["stimd_c"] == "SS"]
    df_stim  = df[df["ttype_c"] == "DS"].copy()
    df_stim = df.copy()

    if subject is not None:
        df_delay = df_delay[df_delay["subject"] == subject].copy()
        df_stim  = df_stim[df_stim["subject"] == subject].copy()

    needed_cols = ["delay_d", "correct_bool", "p_model_correct", "subject", "stim_d"]
    df_delay = df_delay.dropna(subset=needed_cols)
    df_stim  = df_stim.dropna(subset=needed_cols)

    if which == "delay":
        d = df_delay
        xcol = "delay_d"
        xlabel = "Delay duration"
        title_suffix = "Delay"
        band_floor = 1/3
        palette_data = truncate_colormap("Purples_r", 0, 0.7)
    elif which == "stim":
        d = df_stim
        xcol = "stim_d"
        xlabel = "Stimulus duration"
        title_suffix = "Stimulus"
        band_floor = 1/3
        palette_data = truncate_colormap("Oranges", 0.3, 1.0)
    else:
        raise ValueError("which must be 'delay' or 'stim'")

    if d.empty:
        ax.set_title(f"{subject} - {title_suffix}\n(no data)", fontsize=9)
        ax.axis("off")
        return False

    d = d.copy()
    d["x_bin"], edges = pd.qcut(d[xcol], q=n_bins, retbins=True, duplicates="drop")

    centers = (
        d.groupby("x_bin", observed=True)[xcol].median().rename("center").reset_index().sort_values("center")
    )
    order_bins = list(centers["x_bin"])

    subj = (
        d.groupby(["x_bin", "subject"], observed=True)
         .agg(
            data_acc=("correct_bool", "mean"),
            model_acc=("p_model_correct", "mean"),
         )
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

    sns.lineplot(data=plot_df[plot_df["kind"] == "Model"],x="center", y="acc",color="gray", linestyle="-",errorbar=("ci", 95), err_style="band",ax=ax)

    sns.lineplot(data=plot_df[plot_df["kind"] == "Data"], x="center", y="acc", hue="center", palette=palette_data, marker="o", linewidth=0,errorbar=("ci", 95), err_style="bars",legend=False,ax=ax,zorder=10,)

    ax.axhspan(0, band_floor, color="gray", alpha=0.15, zorder=0)

    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frac. correct responses", fontsize=12)
    ax.set_title(f"{subject}", fontsize=12)
    ax.tick_params(labelsize=12)
    sns.despine()

    return True


def plot_categorical_strat_by_side(df, subject, model_name, df_silent = None, cond_col="stimd_c",
                                   cond_order=['VG', 'SL', 'SM', 'SS', 'SIL'], cond_labels=['Visual', 'Easy', 'Medium', 'Hard', 'Silent']):
    df = df.to_pandas()
    df = df.copy()
    df["x_c"] = (df["x_c"].astype("string").str.strip().str.upper())

    if cond_order is None:
        cond_order = list(df[cond_col].dropna().unique())
        cond_order = sorted(cond_order)

    if cond_labels is None:
        cond_labels = cond_order

    g = (df.groupby([cond_col, "x_c"], observed=True).agg(data_mean=("correct_bool", "mean"), model_mean=("p_model_correct", "mean"), n=("correct_bool", "size")).reset_index())

    g["data_sem"] = np.sqrt(g["data_mean"] * (1.0 - g["data_mean"]) / g["n"].clip(lower=1))

    if df_silent is not None:
        df_s = df_silent.copy()
        p_silent = {"L": df_s["pL_mean"], "C": df_s["pC_mean"], "R": df_s["pR_mean"]}

    cond_to_x = {c: i for i, c in enumerate(cond_order)}
    g["x_pos"] = g[cond_col].map(cond_to_x)

    side_palette = {'L': '#e41a1c', 'C': '#4daf4a', 'R': '#377eb8'}

    fig, ax = plt.subplots(figsize=(4,4))

    for side in ["L", "C", "R"]:
        sub = g[g["x_c"] == side].dropna(subset=["x_pos"])
        if sub.empty:
            continue

        sub = sub.sort_values("x_pos")

        ax.plot( sub["x_pos"], sub["model_mean"], "-", lw=2, color=side_palette.get(side, "gray"), label=f"Model {side}", zorder=2)

        ax.errorbar( sub["x_pos"], sub["data_mean"], yerr=sub["data_sem"], fmt="o", ms=5, capsize=3, color=side_palette.get(side, "gray"), linestyle="none", label=f"Data {side}", zorder=3)

        if df_silent is not None:
            ax.plot(len(cond_order)-1, p_silent[side],marker="D", ms=7,color=side_palette[side],linestyle="none",zorder=4)

    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)

    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_labels)

    ax.set_ylim(0.2, 1.05)
    ax.set_ylabel("Frac. correct responses")
    ax.set_xlabel("Trial difficulty")
    ax.set_title(f"{subject}")

    # ax.legend(frameon=False, fontsize=8, ncol=2)
    sns.despine()
    fig.tight_layout()

    fname = f"fig_categorical_strat_by_side_{subject}.pdf"
    out_path = get_plot_path("strat_by_side", fname, model_name)

    return fig, ax

def plot_delay_binned_1d(df, model_name, subject=None, n_bins=7):
    # n_bins=3
    # df_delay = df[df['onset']==0.0].copy()
    df = df.to_pandas()
    df_delay = df[df['stimd_c'] == 'SS']
    # df_stim = df[df['ttype_c']!='VG'].copy()
    df_stim = df[df['ttype_c']=='DS'].copy()
    
    
    if subject is not None:
        df_delay = df_delay[df_delay["subject"] == subject].copy()
        df_stim = df_stim[df_stim["subject"] == subject].copy()

    needed_cols = ["delay_d", "correct_bool", "p_model_correct", "subject", 'stim_d']
    df_delay = df_delay.dropna(subset=needed_cols)
    df_stim = df_stim.dropna(subset=needed_cols)
    if df_delay.empty:
        print(f"  (sin datos válidos para delay 1D en {subject})")
        return
    elif df_stim.empty:
        print(f"  (sin datos válidos para stim 1D en {subject})")
        return
    
    # df_delay["delay_bin"], edges = pd.qcut(df_delay["delay_duration"], q=n_bins, retbins=True, duplicates="drop")
    # df_stim["stim_bin"], edges_stim = pd.qcut(df_stim["stim_duration"], q=n_bins, retbins=True, duplicates="drop")
    # centers_delay = (df_delay.groupby("delay_bin", observed=True)["delay_duration"].median().rename("center").reset_index().sort_values("center"))
    # centers_stim = (df_stim.groupby("stim_bin", observed=True)["stim_duration"].median().rename("center").reset_index().sort_values("center"))
    # order_bins_delay = list(centers_delay["delay_bin"])
    # order_bins_stim = list(centers_stim["stim_bin"])
    
    df_delay["delay_bin"] = (
    df_delay.groupby("ttype_c", observed=True)["delay_d"]
    .transform(lambda s: pd.qcut(s, q=n_bins, duplicates="drop"))
    )

    # centers por ttype_c y bin
    centers_delay = (
        df_delay.groupby(["ttype_c", "delay_bin"], observed=True)["delay_d"]
        .median()
        .rename("center")
        .reset_index()
    )

    # (opcional) order de bins dentro de cada ttype_c según center
    centers_delay["bin_order"] = centers_delay.groupby("ttype_c")["center"].rank(method="dense")
    order_bins_delay = list(centers_delay["delay_bin"])
    # agregación por bin+subject+ttype_c
    subj_delay = (
        df_delay.groupby(["ttype_c", "delay_bin", "subject"], observed=True)
        .agg(
            data_acc=("correct_bool", "mean"),
            model_acc=("p_model_correct", "mean"),
        )
        .reset_index()
        .merge(centers_delay, on=["ttype_c", "delay_bin"], how="left")
    )
    df_stim["stim_bin"] = (
    df_stim.groupby("stimd_c", observed=True)["stim_d"]
    .transform(lambda s: pd.qcut(s, q=n_bins, duplicates="drop"))   
    )

    centers_stim = (
        df_stim.groupby(["stimd_c", "stim_bin"], observed=True)["stim_d"]
        .median()
        .rename("center")
        .reset_index()
    )
    order_bins_stim = list(centers_stim["stim_bin"])
    subj_stim = (
        df_stim.groupby(["stimd_c", "stim_bin", "subject"], observed=True)
        .agg(
            data_acc=("correct_bool", "mean"),
            model_acc=("p_model_correct", "mean"),
        )
        .reset_index()
        .merge(centers_stim, on=["stimd_c", "stim_bin"], how="left")
    )

    plot_stim = subj_stim.melt(
        id_vars=["stimd_c", "stim_bin", "subject", "center"],
        value_vars=["data_acc", "model_acc"],
        var_name="kind",
        value_name="acc",
    )
    plot_stim["kind"] = plot_stim["kind"].map({"data_acc": "Data", "model_acc": "Model"})


    # subj_delay = (df_delay.groupby(["delay_bin", "subject", "ttype_c"], observed=True).agg(data_acc=("correct_bool", "mean"),model_acc=("p_model_correct", "mean"),).reset_index().merge(centers_delay, on="delay_bin", how="left"))
    plot_delay = subj_delay.melt(id_vars=["delay_bin", "subject", "ttype_c", "center"],value_vars=["data_acc", "model_acc"],var_name="kind",value_name="acc",)
    plot_delay["kind"] = plot_delay["kind"].map({"data_acc": "Data","model_acc": "Model"})

    # subj_stim = (df_stim.groupby(["stim_bin", "subject", "stimd_c"], observed=True).agg(data_acc=("correct_bool", "mean"),model_acc=("p_model_correct", "mean"),).reset_index().merge(centers_stim, on="stim_bin", how="left"))
    plot_stim = subj_stim.melt(id_vars=["stim_bin", "subject", "center", "stimd_c"],value_vars=["data_acc", "model_acc"],var_name="kind",value_name="acc",)
    plot_stim["kind"] = plot_stim["kind"].map({"data_acc": "Data","model_acc": "Model"})


    fig, ax = plt.subplots(figsize=(6, 6))

    sns.lineplot(data=plot_delay[plot_delay["kind"] == "Model"], x="center", y="acc",color="gray", hue='ttype_c', linestyle="-",errorbar=("ci", 95),err_style="band",ax=ax)
    sns.lineplot(x="center", y="acc", hue="ttype_c",data=plot_delay[plot_delay["kind"] == "Data"], errorbar=("ci", 95), err_style="bars",marker="o", linewidth=0, ax=ax, zorder=10, legend=False)

    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)

    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Delay duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")

    title_subj = subject if subject is not None else "All subjects"
    ax.set_title(f"{title_subj} - Delay (1D, {len(order_bins_delay)} bins)")

    sns.despine()
    fig.tight_layout()

    fname = f"fig_delay_1d_{title_subj}.pdf"
    out_path = get_plot_path("binning", fname, model_name)
    fig.savefig(out_path, dpi=300)

    plt.show

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(data=plot_stim[plot_stim["kind"] == "Model"], x="center", y="acc",color="gray", hue = "stimd_c", linestyle="-",errorbar=("ci", 95),err_style="band",ax=ax)
    sns.lineplot(x="center", y="acc", hue="stimd_c",data=plot_stim[plot_stim["kind"] == "Data"],errorbar=("ci", 95), err_style="bars",marker="o", linewidth=0, ax=ax, zorder=10, legend=False)
    ax.axhspan(0, 1/3, color="gray", alpha=0.15, zorder=0)
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Stimulus duration (s, binned)")
    ax.set_ylabel("Frac. correct responses")
    title_subj = subject if subject is not None else "All subjects"
    ax.set_title(f"{title_subj} - Stimulus (1D, {len(order_bins_stim)} bins)")
    sns.despine()
    fig.tight_layout()
    fname = f"fig_stim_1d_{title_subj}.pdf"
    out_path = get_plot_path("binning", fname, model_name)
    fig.savefig(out_path, dpi=300)
    plt.show()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# GLM-HMM / GLM-HMM-t notebook analysis helpers
# Shared plotting functions extracted from glmhmm_analysis.py and
# glmhmmt_analysis.py.  All functions return matplotlib Figure objects;
# marimo UI wrappers (mo.vstack etc.) stay in the notebooks.
# ─────────────────────────────────────────────────────────────────────────────

_AG_GROUPS = [
    # neutral intercept bias (constant 1): overall lateral vs centre preference
    ("$bias$", [("bias", "mean")]),
    # bias: L/R context indicators (no C equivalent)
    ("$bias_{coh}$",  [("biasL", 0), ("biasR", 1)]),
    ("$bias_{incoh}$", [("biasL", 1), ("biasR", 0)]),
    # onset — C side merged in: coh=P(C)|onsetC, incoh=(P(L)+P(R))/2|onsetC
    ("$onset_{coh}$",  [("onsetL", 0), ("onsetR", 1), ("onsetC", "neg_mean")]),
    ("$onset_{incoh}$", [("onsetL", 1), ("onsetR", 0), ("onsetC", "mean")]),
    # delay (shared scalar)
    ("delay", [("delay", "mean")]),
    # delay × side
    ("$D_{coh}$",  [("DL", 0), ("DR", 1)]),
    ("$D_{incoh}$", [("DL", 1), ("DR", 0)]),
    ("DC", [("DC", "neg_mean")]),
    # stimulus — SC merged in
    ("$S_{coh}$",  [("SL", 0), ("SR", 1), ("SC", "neg_mean")]),
    ("$S_{incoh}$", [("SL", 1), ("SR", 0), ("SC", "mean")]),
    # stimulus × delay — SCxdelay merged in
    ("$Sxd_{coh}$",  [("SLxdelay", 0), ("SRxdelay", 1), ("SCxdelay", "neg_mean")]),
    ("$Sxd_{incoh}$", [("SLxdelay", 1), ("SRxdelay", 0), ("SCxdelay", "mean")]),
    # stim interval × side — stim1…4 collapsed like SL/SC/SR
    ("$S1_{coh}$",   [("stim1L", 0), ("stim1R", 1), ("stim1C", "neg_mean")]),
    ("$S1_{incoh}$", [("stim1L", 1), ("stim1R", 0), ("stim1C", "mean")]),
    ("$S2_{coh}$",   [("stim2L", 0), ("stim2R", 1), ("stim2C", "neg_mean")]),
    ("$S2_{incoh}$", [("stim2L", 1), ("stim2R", 0), ("stim2C", "mean")]),
    ("$S3_{coh}$",   [("stim3L", 0), ("stim3R", 1), ("stim3C", "neg_mean")]),
    ("$S3_{incoh}$", [("stim3L", 1), ("stim3R", 0), ("stim3C", "mean")]),
    ("$S4_{coh}$",   [("stim4L", 0), ("stim4R", 1), ("stim4C", "neg_mean")]),
    ("$S4_{incoh}$", [("stim4L", 1), ("stim4R", 0), ("stim4C", "mean")]),
    # action history (perseveration vs alternation)
    ("$A_{coh}$",  [("A_L", 0), ("A_R", 1)]),
    ("$A_{incoh}$", [("A_L", 1), ("A_R", 0)]),
]


# NOTE: _LABEL_RANK and _STATE_HEX are also defined in views.py (the canonical
# source of truth).  These local definitions are kept for backward compatibility
# with existing call sites inside this module.  Both definitions must stay in sync.
_LABEL_RANK = {
    "Engaged": 0,
    "Disengaged": 1,
    "Disengaged L": 1,
    "Disengaged R": 2,
    "Disengaged C": 3,
    **{f"Disengaged {i}": i for i in range(1, 10)},
}

# ── canonical state colours from config (rank-indexed) ───────────────────────
_STATE_HEX: list[str] = _get_config_state_palette()


def _state_color(label: str, fallback_idx: int = 0, palette: list[str] | None = None) -> str:
    """Return the config-defined hex colour for a state label."""
    _palette = list(palette) if palette else _STATE_HEX
    rank = _LABEL_RANK.get(label, fallback_idx)
    return _palette[rank % len(_palette)]


def _build_state_palette(
    state_labels_per_subj: dict,
    K: int | None = None,
) -> tuple[dict[str, str], list[str]]:
    """
    Build a (palette_dict, hue_order) pair from a {subj: {k: label}} mapping.

    Both are rank-ordered (Engaged first) so every seaborn plot that receives
    them uses the same colour and ordering regardless of K or subject set.
    When an exact-K palette is configured in ``config.toml``, it is preferred.
    """
    seen: dict[str, int] = {}
    for _slbls in state_labels_per_subj.values():
        for _k, _lbl in _slbls.items():
            if _lbl not in seen:
                seen[_lbl] = _LABEL_RANK.get(_lbl, _k)
    ordered = sorted(seen, key=lambda l: seen[l])
    _palette = _get_config_state_palette(K if K is not None else len(ordered))
    pal = {lbl: _state_color(lbl, seen[lbl], palette=_palette) for lbl in ordered}
    return pal, ordered


def _collect_emission_weight_frames(
    arrays_store: dict,
    state_labels: dict,
    names: dict,
    subjects: list,
):
    _CLS_LABELS  = ["Left (vs C)", "Right (vs C)"]
    _records     = []
    _ag_records  = []
    _feat_names  = names.get("X_cols", [])

    for subject in subjects:
        if subject not in arrays_store:
            continue
        _W      = arrays_store[subject]["emission_weights"]   # (K, 2, n_feat)
        _n      = _W.shape[2]
        _fnames = (arrays_store[subject].get("X_cols") or names.get("X_cols", []))[:_n]
        _f2i    = {f: i for i, f in enumerate(_fnames)}
        _feat_names = _fnames   # keep last subject's list for per-class axis labels

        for _k in range(_W.shape[0]):
            _slbl = state_labels.get(subject, {}).get(_k, f"State {_k}")
            for _c in range(_W.shape[1]):
                for _fi, _fn in enumerate(_fnames):
                    _records.append({
                        "subject":     subject,
                        "state":       _slbl,
                        "class":       _c,
                        "class_label": _CLS_LABELS[_c] if _c < len(_CLS_LABELS) else f"Class {_c}",
                        "feature":     _fn,
                        "weight":      float(_W[_k, _c, _fi]),
                    })
            # ── agonist: softmax ΔP (same convention as fit_glm.py) ──────────
            # logits = [W[k,0,f], 0.0, W[k,1,f]]  →  p = [P(L), P(C), P(R)]
            # mode=0 (L-vs-C weight, coh L feat) → P(L) − 1/3
            # mode=1 (R-vs-C weight, coh R feat) → P(R) − 1/3
            # mode="neg_mean" (C-only)           → P(C) − 1/3
            # mode="mean"    (neutral)           → (P(L)+P(R))/2 − 1/3
            _BASE = 1.0 / 3.0
            for _grp_label, _members in _AG_GROUPS:
                _vals = []
                for _fn, _mode in _members:
                    if _fn not in _f2i:
                        continue
                    _fi = _f2i[_fn]
                    _logits = [float(_W[_k, 0, _fi]), 0.0, float(_W[_k, 1, _fi])]
                    _p = _softmax(_logits)
                    if _mode == 0:
                        _vals.append(_p[0] - _BASE)                # P(L) − baseline
                    elif _mode == 1:
                        _vals.append(_p[2] - _BASE)                # P(R) − baseline
                    elif _mode == "neg_mean":
                        _vals.append(_p[1] - _BASE)                # P(C) − baseline
                    else:  # "mean" — neutral
                        _vals.append((_p[0] + _p[2]) / 2 - _BASE) # avg lateral
                if _vals:
                    _ag_records.append({
                        "subject": subject,
                        "state":   _slbl,
                        "feature": _grp_label,
                        "weight":  float(np.mean(_vals)),
                    })

    if not _records:
        raise ValueError("No emission weights found for the selected subjects.")

    _df_w     = pd.DataFrame(_records)
    _df_ag    = pd.DataFrame(_ag_records)
    _ag_order = [g for g, _ in _AG_GROUPS if g in _df_ag["feature"].values]
    _state_pal, _state_hue_order = _build_state_palette(state_labels)
    return _df_w, _df_ag, _feat_names, _ag_order, _state_pal, _state_hue_order, _CLS_LABELS


def plot_emission_weights_by_subject(
    arrays_store: dict,
    state_labels: dict,
    names: dict,
    K: int,
    subjects: list,
    save_path=None,
):
    """Per-subject barplots of emission weights."""
    _df_w, _, _feat_names, _, _state_pal, _state_hue_order, _cls_labels = _collect_emission_weight_frames(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        subjects=subjects,
    )

    _selected = [s for s in subjects if s in arrays_store and "emission_weights" in arrays_store[s]]
    _n_classes = max(1, _df_w["class"].nunique())
    _n_cols = min(3, max(1, len(_selected)))
    _n_panels = max(1, len(_selected) * _n_classes)
    _n_rows = int(math.ceil(_n_panels / _n_cols))
    _fig_w = max(6, len(_feat_names) * 0.8) * _n_cols
    _fig_h = max(3.4, 3.2 * _n_rows)
    fig_bar, axes = plt.subplots(
        _n_rows,
        _n_cols,
        figsize=(_fig_w, _fig_h),
        sharey=True,
        squeeze=False,
    )

    _x = np.arange(len(_feat_names))
    _bar_w = 0.8 / max(1, len(_state_hue_order))

    for subj_idx, subj in enumerate(_selected):
        for class_idx in range(_n_classes):
            panel_idx = subj_idx * _n_classes + class_idx
            ax = axes[panel_idx // _n_cols, panel_idx % _n_cols]
            _sub = _df_w[(_df_w["subject"] == subj) & (_df_w["class"] == class_idx)]
            for state_pos, state_name in enumerate(_state_hue_order):
                _state_sub = (
                    _sub[_sub["state"] == state_name]
                    .set_index("feature")
                    .reindex(_feat_names)
                    .reset_index()
                )
                _offset = (state_pos - (len(_state_hue_order) - 1) / 2) * _bar_w
                ax.bar(
                    _x + _offset,
                    _state_sub["weight"].to_numpy(dtype=float),
                    _bar_w,
                    label=state_name,
                    color=_state_pal[state_name],
                    alpha=0.85,
                )
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_xticks(_x)
            ax.set_xticklabels(_feat_names, rotation=35, ha="right")
            ax.set_title(
                f"Subject {subj} — {_cls_labels[class_idx] if class_idx < len(_cls_labels) else f'Class {class_idx}'}"
            )
            if panel_idx % _n_cols == 0:
                ax.set_ylabel("Weight")

    for panel_idx in range(_n_panels, _n_rows * _n_cols):
        axes[panel_idx // _n_cols, panel_idx % _n_cols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig_bar.suptitle(f"Emission weights by subject  (K={K})", y=1.01)
    fig_bar.tight_layout()
    sns.despine(fig=fig_bar)
    if save_path is not None:
        fig_bar.savefig(save_path, dpi=300)
    return fig_bar


def plot_emission_weights(
    arrays_store: dict,
    state_labels: dict,
    names: dict,
    K: int,
    subjects: list,
    save_path=None,
):
    """
    Emission-weight summaries: collapsed agonist view + per-choice-class panels.

    The per-subject barplots live in ``plot_emission_weights_by_subject`` so
    notebooks can render or comment them independently.
    """
    _df_w, _df_ag, _feat_names, _ag_order, _state_pal, _state_hue_order, _CLS_LABELS = _collect_emission_weight_frames(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        subjects=subjects,
    )

    # ── 1. Agonist (collapsed) figure ─────────────────────────────────────────
    fig_ag, axes_ag = plt.subplots(1, 2, figsize=(len(_ag_order) * 2, 4), sharex=True)
    ax_ag_line, ax_ag_box = axes_ag
    
    sns.lineplot(
        data=_df_ag, x="feature", y="weight", hue="state", ax=ax_ag_line,
        markers=True, marker="o", markersize=8, markeredgewidth=0,
        alpha=0.85, errorbar="se",
        palette=_state_pal, hue_order=_state_hue_order,
    )
    ax_ag_line.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_ag_line.set_ylabel("ΔP (from 1/3 baseline)")
    ax_ag_line.set_xlabel("")
    ax_ag_line.set_title(f"Emission weights - collapsed view  (K={K})")
    ax_ag_line.get_legend().set_title("")
    ax_ag_line.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

    sns.boxplot(
        data=_df_ag, x="feature", y="weight", hue="state", ax=ax_ag_box,
        palette=_state_pal, hue_order=_state_hue_order,
        width=0.8, showfliers=False, boxprops={'alpha': 0.7}
    )
    sns.stripplot(
        data=_df_ag, x="feature", y="weight", hue="state", ax=ax_ag_box,
        palette=_state_pal, hue_order=_state_hue_order,
        dodge=True, alpha=0.5, zorder=1, legend=False
    )
    ax_ag_box.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # Statistical annotations for agonist figure
    from scipy.stats import ttest_rel
    import itertools

    def get_star(pval):
        if pval < 0.001: return "***"
        elif pval < 0.01: return "**"
        elif pval < 0.05: return "*"
        return "ns"

    K_ag = len(_state_hue_order) # Number of states
    state_pairs = list(itertools.combinations(range(K_ag), 2))
    hue_width = 0.8 / K_ag

    y_range_ag = _df_ag["weight"].max() - _df_ag["weight"].min()
    if pd.isna(y_range_ag) or y_range_ag == 0:
        y_range_ag = 1

    for m, feat in enumerate(_ag_order):
        feat_df = _df_ag[_df_ag["feature"] == feat]
        if feat_df.empty: continue
        
        y_max = feat_df["weight"].max()
        y_offset_step = y_range_ag * 0.05
        current_y_offset = y_max + y_offset_step

        for p1, p2 in state_pairs:
            s1 = _state_hue_order[p1]
            s2 = _state_hue_order[p2]
            
            # Align by subject for paired t-test
            df1 = feat_df[feat_df["state"] == s1].set_index("subject")["weight"]
            df2 = feat_df[feat_df["state"] == s2].set_index("subject")["weight"]
            
            common_subjs = df1.index.intersection(df2.index)
            if len(common_subjs) < 2: continue
            
            w1 = df1.loc[common_subjs].values
            w2 = df2.loc[common_subjs].values

            try:
                stat, pval = ttest_rel(w1, w2)
                star = get_star(pval)
            except Exception:
                star = ""

            if star:
                offset_1 = (p1 - (K_ag - 1) / 2) * hue_width
                offset_2 = (p2 - (K_ag - 1) / 2) * hue_width
                x1 = m + offset_1
                x2 = m + offset_2
                
                h = y_range_ag * 0.02
                ax_ag_box.plot([x1, x1, x2, x2], [current_y_offset, current_y_offset+h, current_y_offset+h, current_y_offset], lw=1, c='k')
                ax_ag_box.text((x1+x2)/2, current_y_offset+h, star, ha='center', va='bottom', color='k')
                current_y_offset += y_offset_step * 1.5


    ax_ag_box.set_xticks(range(len(_ag_order)))
    ax_ag_box.set_xticklabels(_ag_order)
    ax_ag_box.set_xlabel("")
    ax_ag_box.set_ylabel("ΔP (from 1/3 baseline)")
    
    handles, labels_lgd = ax_ag_box.get_legend_handles_labels()
    if len(handles) >= K_ag:
        ax_ag_box.legend(handles[:K_ag], labels_lgd[:K_ag], frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        ax_ag_box.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
        
    fig_ag.tight_layout()
    sns.despine(fig=fig_ag)

    # ── 2. Per-class figure ────────────────────────────────────────────────────
    _n_classes = _df_w["class"].nunique()
    fig_cls, axes_cls_grid = plt.subplots(
        2, _n_classes, figsize=(6 * _n_classes, 8), sharex=True, squeeze=False
    )
    
    y_range_cls = _df_w["weight"].max() - _df_w["weight"].min()
    if pd.isna(y_range_cls) or y_range_cls == 0:
        y_range_cls = 1
        
    for _c in range(_n_classes):
        _ax_line = axes_cls_grid[0, _c]
        _ax_box = axes_cls_grid[1, _c]
        _sub = _df_w[_df_w["class"] == _c]
        
        sns.lineplot(
            data=_sub, x="feature", y="weight", hue="state", ax=_ax_line,
            markers=True, marker="o", markersize=8, markeredgewidth=0,
            alpha=0.8, errorbar="se",
            palette=_state_pal, hue_order=_state_hue_order,
            legend=False
        )
        _ax_line.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        _ax_line.set_title(_CLS_LABELS[_c] if _c < len(_CLS_LABELS) else f"Class {_c}")
        _ax_line.set_xlabel("")
        _ax_line.set_ylabel("Weight" if _c == 0 else "")

        sns.boxplot(
            data=_sub, x="feature", y="weight", hue="state", ax=_ax_box,
            palette=_state_pal, hue_order=_state_hue_order,
            width=0.8, showfliers=False, boxprops={'alpha': 0.7}
        )
        sns.stripplot(
            data=_sub, x="feature", y="weight", hue="state", ax=_ax_box,
            palette=_state_pal, hue_order=_state_hue_order,
            dodge=True, alpha=0.5, zorder=1, legend=False
        )
        _ax_box.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        
        # Annotations
        for m, feat in enumerate(_feat_names):
            feat_df = _sub[_sub["feature"] == feat]
            if feat_df.empty: continue
            
            y_max = feat_df["weight"].max()
            y_offset_step = y_range_cls * 0.05
            current_y_offset = y_max + y_offset_step
            
            for p1, p2 in state_pairs:
                s1 = _state_hue_order[p1]
                s2 = _state_hue_order[p2]
                
                df1 = feat_df[feat_df["state"] == s1].set_index("subject")["weight"]
                df2 = feat_df[feat_df["state"] == s2].set_index("subject")["weight"]
                
                common_subjs = df1.index.intersection(df2.index)
                if len(common_subjs) < 2: continue
                
                w1 = df1.loc[common_subjs].values
                w2 = df2.loc[common_subjs].values
    
                try:
                    stat, pval = ttest_rel(w1, w2)
                    star = get_star(pval)
                except Exception:
                    star = ""
    
                if star:
                    offset_1 = (p1 - (K_ag - 1) / 2) * hue_width
                    offset_2 = (p2 - (K_ag - 1) / 2) * hue_width
                    x1 = m + offset_1
                    x2 = m + offset_2
                    
                    h = y_range_cls * 0.02
                    _ax_box.plot([x1, x1, x2, x2], [current_y_offset, current_y_offset+h, current_y_offset+h, current_y_offset], lw=1, c='k')
                    _ax_box.text((x1+x2)/2, current_y_offset+h, star, ha='center', va='bottom', color='k')
                    current_y_offset += y_offset_step * 1.5

        _ax_box.set_xticks(range(len(_feat_names)))
        _ax_box.set_xticklabels(_feat_names, rotation=35, ha="right")
        _ax_box.set_xlabel("")
        _ax_box.set_ylabel("Weight" if _c == 0 else "")
        
        handles, labels_lgd = _ax_box.get_legend_handles_labels()
        if len(handles) >= K_ag:
            _ax_box.legend(handles[:K_ag], labels_lgd[:K_ag], frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
        else:
            _ax_box.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig_cls.suptitle(f"Emission weights per choice  (K={K})", y=1.02)
    fig_cls.tight_layout()
    sns.despine(fig=fig_cls)

    return fig_ag, fig_cls


def plot_transition_matrix_by_subject(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
):
    """Per-subject transition-matrix heatmaps."""
    def _resolve_matrix(subj: str) -> np.ndarray | None:
        _arr = arrays_store.get(subj, {})
        if "transition_matrix" in _arr:
            return np.asarray(_arr["transition_matrix"])
        if "transition_bias" in _arr:
            _bias = np.asarray(_arr["transition_bias"])
            _exp = np.exp(_bias - _bias.max(axis=-1, keepdims=True))
            return _exp / _exp.sum(axis=-1, keepdims=True)
        return None

    _selected = [s for s in subjects if _resolve_matrix(s) is not None]
    if not _selected:
        raise ValueError("No transition matrices found for selected subjects.")

    _n_cols = min(3, len(_selected))
    _n_rows = int(math.ceil(len(_selected) / _n_cols))
    fig, axes = plt.subplots(
        _n_rows,
        _n_cols,
        figsize=(4.2 * _n_cols, 3.4 * _n_rows),
        squeeze=False,
    )

    for idx, subj in enumerate(_selected):
        ax = axes[idx // _n_cols, idx % _n_cols]
        _A = _resolve_matrix(subj)
        _slbl = state_labels.get(subj, {k: f"S{k}" for k in range(K)})
        _tick_labels = [_slbl.get(k, f"S{k}") for k in range(K)]
        sns.heatmap(
            _A,
            ax=ax,
            cmap="bone",
            annot=True,
            fmt=".2f",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            xticklabels=_tick_labels,
            yticklabels=_tick_labels,
            cbar=idx == 0,
            cbar_kws={"shrink": 0.8, "label": "probability"},
        )
        ax.set_title(f"Subject {subj}")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")

    for idx in range(len(_selected), _n_rows * _n_cols):
        axes[idx // _n_cols, idx % _n_cols].set_visible(False)

    fig.tight_layout()
    return fig


def plot_transition_matrix(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
):
    """Mean transition-matrix heatmap across selected subjects."""
    def _resolve_matrix(subj: str) -> np.ndarray | None:
        _arr = arrays_store.get(subj, {})
        if "transition_matrix" in _arr:
            return np.asarray(_arr["transition_matrix"])
        if "transition_bias" in _arr:
            _bias = np.asarray(_arr["transition_bias"])
            _exp = np.exp(_bias - _bias.max(axis=-1, keepdims=True))
            return _exp / _exp.sum(axis=-1, keepdims=True)
        return None

    _selected = [s for s in subjects if _resolve_matrix(s) is not None]
    if not _selected:
        raise ValueError("No transition matrices found for selected subjects.")

    _A_mean = np.mean([_resolve_matrix(s) for s in _selected], axis=0)
    _first_labels = state_labels.get(_selected[0], {k: f"S{k}" for k in range(K)})
    _tick_labels = [_first_labels.get(k, f"S{k}") for k in range(K)]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    sns.heatmap(
        _A_mean,
        ax=ax,
        cmap="bone",
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        xticklabels=_tick_labels,
        yticklabels=_tick_labels,
        cbar_kws={"shrink": 0.8, "label": "probability"},
    )
    ax.set_title(f"Mean transition matrix  (n={len(_selected)} subjects)")
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    fig.tight_layout()
    return fig


def plot_posterior_probs(
    arrays_store: dict,
    state_labels: dict,
    K: int,
    subjects: list,
    t0: int = 0,
    t1: int = 199,
):
    """
    Stacked-area posterior state probability plot with choice tick marks.

    Returns
    -------
    fig
    """
    _selected = [s for s in subjects if s in arrays_store]
    if not _selected:
        raise ValueError("No fitted arrays for selected subjects.")

    _colors        = _STATE_HEX
    _choice_colors = {0: "royalblue", 1: "gold", 2: "tomato"}
    _choice_labels = {0: "L", 1: "C", 2: "R"}

    fig, axes = plt.subplots(len(_selected), 1,
                             figsize=(14, 3 * len(_selected)), squeeze=False)

    for _i, _subj in enumerate(_selected):
        _ax    = axes[_i, 0]
        _probs = arrays_store[_subj]["smoothed_probs"][t0: t1 + 1]
        _y     = arrays_store[_subj]["y"].astype(int)[t0: t1 + 1]
        _T_w   = _probs.shape[0]
        _x     = np.arange(t0, t0 + _T_w)

        _bottom = np.zeros(_T_w)
        _slbl   = state_labels.get(_subj, {k: f"State {k}" for k in range(K)})
        for _k in range(K):
            _rank = _LABEL_RANK.get(_slbl.get(_k, ""), _k)
            _ax.fill_between(
                _x, _bottom, _bottom + _probs[:, _k],
                alpha=0.7, color=_colors[_rank % len(_colors)],
                label=_slbl.get(_k, f"State {_k}"),
            )
            _bottom += _probs[:, _k]

        for _resp, _col in _choice_colors.items():
            _mask = _y == _resp
            _ax.scatter(
                _x[_mask], np.ones(_mask.sum()) * 1.03, c=_col, s=4, marker="|",
                label=_choice_labels[_resp],
                transform=_ax.get_xaxis_transform(), clip_on=False,
            )

        _ax.set_xlim(t0, t0 + _T_w - 1)
        _ax.set_ylim(0, 1)
        _ax.set_ylabel("State probability")
        _ax.set_title(f"Subject {_subj}")
        _ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
                   fontsize=8, ncol=1, frameon=False)

    axes[-1, 0].set_xlabel("Trial")
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    sns.despine(fig=fig)
    return fig


from matplotlib.collections import PathCollection
def strip_darken(ax, factor=0.7, lw=1.5):
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            fc = coll.get_facecolors()
            if len(fc) == 0:
                continue

            new_ec = fc.copy()
            new_ec[:, :3] *= factor

            coll.set_edgecolors(new_ec)
            coll.set_linewidth(lw)

def plot_state_accuracy(
    views: dict,
    trial_df,
    thresh: float = 0.5,
    session_col: str = "Session",
    sort_col: str = "Trial",
    performance_col: str = "correct_bool",
    stim_col: str = "stimd_n",
    **kwargs,
) -> Tuple[plt.Figure, pd.DataFrame]:
    return _plot_state_accuracy_common(
        views,
        trial_df,
        thresh=thresh,
        performance_candidates=(performance_col, "performance"),
        stim_candidates=(stim_col, "stimulus", "ILD"),
        stim_label="nonzero stimulus",
    )


def plot_session_trajectories(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    **kwargs,
):
    return _plot_session_trajectories_common(
        views,
        trial_df,
        session_col=session_col,
    )


def plot_state_posterior_count_kde(
    views: dict,
    thresh: float | None = None,
    bins: int = 40,
    **kwargs,
):
    return _plot_state_posterior_count_kde_common(
        views,
        thresh=thresh,
        bins=bins,
    )


def plot_change_triggered_posteriors_summary(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    switch_posterior_threshold: float | None = None,
    window: int = 15,
    **kwargs,
):
    return _plot_change_triggered_posteriors_summary_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        window=window,
    )


def plot_change_triggered_posteriors_by_subject(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    switch_posterior_threshold: float | None = None,
    window: int = 15,
    **kwargs,
):
    return _plot_change_triggered_posteriors_by_subject_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        window=window,
    )


def plot_state_occupancy(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    **kwargs,
):
    return _plot_state_occupancy_common(
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
):
    return _plot_state_dwell_times_by_subject_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
    )


def plot_state_dwell_times_summary(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    max_dwell: int | None = None,
    ci_level: float = 0.68,
    **kwargs,
):
    return _plot_state_dwell_times_summary_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
    )


def plot_state_dwell_times(
    views: dict,
    trial_df,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    max_dwell: int | None = None,
    ci_level: float = 0.68,
    **kwargs,
):
    return _plot_state_dwell_times_common(
        views,
        trial_df,
        session_col=session_col,
        sort_col=sort_col,
        max_dwell=max_dwell,
        ci_level=ci_level,
    )


def plot_session_deepdive(
    views: dict,
    trial_df,
    subj: str,
    sess: int,
    session_col: str = "session",
    sort_col: str = "trial_idx",
    switch_posterior_threshold: float | None = None,
    stimd_col: str = "stimd_n",
    perf_col: str = "performance",
    resp_col: str = "response",
    **kwargs,
):
    return _plot_session_deepdive_common(
        views,
        trial_df,
        subj,
        sess,
        session_col=session_col,
        sort_col=sort_col,
        switch_posterior_threshold=switch_posterior_threshold,
        performance_candidates=(perf_col, "correct_bool", "performance"),
        stim_candidates=(stimd_col, "stimulus", "ILD"),
        response_candidates=(resp_col, "response", "Choice"),
    )


def plot_tau_sweep(sweep_path, subjects: list, K: int):
    """
    BIC vs τ and LL/trial vs τ plots from a tau-sweep parquet file.

    Parameters
    ----------
    sweep_path : path-like pointing to tau_sweep_summary.parquet
    subjects   : list of subject IDs to plot
    K          : number of states (used to filter the sweep dataframe)

    Returns
    -------
    fig, best_df (polars DataFrame: best τ per subject sorted by subject & K)
    """
    _df_sweep = pl.read_parquet(sweep_path)
    _subjects = [s for s in subjects
                 if s in _df_sweep["subject"].unique().to_list()]
    if not _subjects:
        raise ValueError("No sweep data found for the selected subjects.")

    fig, (_ax_bic, _ax_ll) = plt.subplots(1, 2, figsize=(12, 4))
    _palette = sns.color_palette("tab10", n_colors=len(_subjects))

    for _i, _subj in enumerate(_subjects):
        _d    = _df_sweep.filter(
            (pl.col("subject") == _subj) & (pl.col("K") == K)
        ).sort("tau")
        _tau  = _d["tau"].to_numpy()
        _bic  = _d["bic"].to_numpy()
        _ll   = _d["ll_per_trial"].to_numpy()
        _c    = _palette[_i]
        _ax_bic.plot(_tau, _bic, "-o", ms=3, color=_c, label=_subj)
        _ax_ll .plot(_tau, _ll,  "-o", ms=3, color=_c, label=_subj)
        _best  = int(np.argmin(_bic))
        _ax_bic.axvline(_tau[_best], color=_c, lw=0.8, linestyle="--", alpha=0.6)

    for _ax, _ylabel, _title in [
        (_ax_bic, "BIC",        "BIC vs τ  (lower is better)"),
        (_ax_ll,  "LL / trial", "Log-likelihood per trial vs τ"),
    ]:
        _ax.set_xlabel("τ (action-trace half-life)")
        _ax.set_ylabel(_ylabel)
        _ax.set_title(_title)
        _ax.legend(fontsize=8, frameon=False)
        sns.despine(ax=_ax)

    fig.tight_layout()

    best_df = (
        _df_sweep
        .filter(pl.col("subject").is_in(_subjects) & (pl.col("K") == K))
        .sort("bic")
        .group_by(["subject", "K"])
        .first()
        .select(["subject", "K", "tau", "bic", "ll_per_trial", "acc"])
        .sort(["subject", "K"])
    )
    return fig, best_df


def plot_transition_weights(
    arrays_store: dict | None = None,
    names: dict | None = None,
    K: int | None = None,
    subjects: list | None = None,
    state_labels: dict | None = None,
    views: dict | None = None,
):
    """
    Input-dependent transition weights (glmhmm-t only).

    Produces two figures:
      fig_line – standardised lineplot (mean-centred across states)
      fig_box  – standardised boxplot by feature and state

    Parameters
    ----------
    state_labels : {subj: {state_idx: label_str}} – if provided, semantic
                   labels (e.g. "Engaged") and config-driven colours are used
                   consistently across all three figures.
    views : {subj: SubjectFitView} – preferred source. When provided, state
            labels, transition weights, and transition feature names are read
            directly from the views.

    Returns
    -------
    fig_line, fig_box
    """
    import itertools as _it
    from scipy import stats as _stats

    _selected = list(subjects or [])
    if views is not None:
        if not _selected:
            _selected = list(views.keys())
        _selected = [
            s for s in _selected
            if s in views and getattr(views[s], "transition_weights", None) is not None
        ]
        if K is None and _selected:
            K = int(views[_selected[0]].K)
        _slbls_map = {
            s: dict(views[s].state_name_by_idx) for s in _selected
        }
        _state_pal, _states_order = _build_state_palette(_slbls_map, K=K)
        _D_first = views[_selected[0]].transition_weights.shape[2]
        _U_cols = list(views[_selected[0]].U_cols)[:_D_first]

        def _get_tw(subj: str) -> np.ndarray:
            return np.asarray(views[subj].transition_weights)

        def _get_ucols(subj: str, D: int) -> list[str]:
            return list(views[subj].U_cols)[:D]
    else:
        if arrays_store is None:
            raise ValueError("Provide either views or arrays_store.")
        _selected = [
            s for s in _selected
            if s in arrays_store and "transition_weights" in arrays_store[s]
        ]
        if K is None:
            raise ValueError("K is required when views are not provided.")
        if names is None:
            names = {}

        _slbls_map: dict = {}
        for _subj in _selected:
            _slbls_map[_subj] = (
                (state_labels or {}).get(_subj) or {k: f"State {k}" for k in range(K)}
            )

        _state_pal, _states_order = _build_state_palette(_slbls_map, K=K)
        _D_first = arrays_store[_selected[0]]["transition_weights"].shape[2]
        _U_cols = (arrays_store[_selected[0]].get("U_cols") or names.get("U_cols", []))[:_D_first]

        def _get_tw(subj: str) -> np.ndarray:
            return np.asarray(arrays_store[subj]["transition_weights"])

        def _get_ucols(subj: str, D: int) -> list[str]:
            return list((arrays_store[subj].get("U_cols") or names.get("U_cols", []))[:D])

    if not _selected:
        raise ValueError("No transition weights found for selected subjects.")

    _state_pairs = list(_it.combinations(_states_order, 2))

    # ── standardised records ──────────────────────────────────────────────────
    _std_records = []
    for _subj in _selected:
        _W_raw    = _get_tw(_subj)                              # (K, K, D)
        _D        = _W_raw.shape[2]
        _U_cols_s = _get_ucols(_subj, _D)
        _W_avg    = _W_raw.mean(axis=0)                         # (K, D)
        _W_aug    = np.vstack([_W_avg, np.zeros((1, _W_avg.shape[1]))])
        _v1       = -np.mean(_W_aug, axis=0)
        _W_std    = np.array(_W_aug, copy=True)
        _W_std[-1] = _v1
        for _k in range(K):
            _W_std[_k] = _v1 + _W_avg[_k]
        for _k in range(K):
            _lbl_k = _slbls_map[_subj].get(_k, f"State {_k}")
            for _fi, _fname in enumerate(_U_cols_s):
                _std_records.append({
                    "subject": _subj, "state": _lbl_k,
                    "feature": _fname, "weight": float(_W_std[_k, _fi]),
                })

    _df_std = pd.DataFrame(_std_records)

    def _sig_label(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    _sig_results = {}
    for _feat in _U_cols:
        for _st_a, _st_b in _state_pairs:
            _va = (_df_std[(_df_std["feature"] == _feat) & (_df_std["state"] == _st_a)]
                   .set_index("subject")["weight"])
            _vb = (_df_std[(_df_std["feature"] == _feat) & (_df_std["state"] == _st_b)]
                   .set_index("subject")["weight"])
            _common = _va.index.intersection(_vb.index)
            if len(_common) >= 2:
                _, _p = _stats.ttest_rel(_va[_common], _vb[_common])
            else:
                _p = float("nan")
            _sig_results[(_feat, _st_a, _st_b)] = _p

    _feat_xpos = {f: i for i, f in enumerate(_U_cols)}
    _states_str = " / ".join(_states_order)

    # ── fig_line: standardised lineplot ───────────────────────────────────────
    fig_line, ax_line = plt.subplots(figsize=(4, max(3, K * 1.0)))
    sns.lineplot(
        data=_df_std, x="feature", y="weight", hue="state", ax=ax_line,
        markers=True, marker="o", markersize=9, markeredgewidth=0,
        alpha=0.85, errorbar="se",
        palette=_state_pal, hue_order=_states_order,
    )
    for _subj_s in _selected:
        _sub_df = _df_std[_df_std["subject"] == _subj_s]
        for _st in _states_order:
            _sub_st = _sub_df[_sub_df["state"] == _st]
            ax_line.plot(_sub_st["feature"].tolist(), _sub_st["weight"].tolist(),
                         color=_state_pal[_st], alpha=0.25, linewidth=0.8)
    _lxr   = abs(ax_line.get_xlim()[1] - ax_line.get_xlim()[0])
    for _pi, (_st_a, _st_b) in enumerate(_state_pairs):
        for _feat in _U_cols:
            _lbl = _sig_label(_sig_results[(_feat, _st_a, _st_b)])
            if _lbl == "ns":
                continue
            _xp  = _feat_xpos[_feat]
            _va_m = _df_std[(_df_std["feature"] == _feat) & (_df_std["state"] == _st_a)]["weight"].mean()
            _vb_m = _df_std[(_df_std["feature"] == _feat) & (_df_std["state"] == _st_b)]["weight"].mean()
            ax_line.text(_xp, max(_va_m, _vb_m) + _lxr * 0.04 * (_pi + 1),
                         _lbl, ha="center", va="bottom", fontsize=10, color="black")
    ax_line.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_line.set_xticks(range(len(_U_cols)))
    ax_line.set_xticklabels(_U_cols, rotation=20, ha="right")
    ax_line.set_xlabel("")
    ax_line.set_ylabel("Transition weight")
    ax_line.set_title(f"glmhmm-t K={K} — transition weights by state ({_states_str})")
    if ax_line.get_legend() is not None:
        ax_line.get_legend().set_title("")
    fig_line.tight_layout()
    sns.despine(fig=fig_line)

    # ── fig_box: emission-style boxplot with paired significance ─────────────
    fig_box, ax_box = plt.subplots(figsize=(max(5, len(_U_cols) * 1.4), max(3, K * 1.0)))
    sns.boxplot(
        data=_df_std, x="feature", y="weight", hue="state", ax=ax_box,
        palette=_state_pal, hue_order=_states_order,
        width=0.8, showfliers=False, boxprops={"alpha": 0.75},
    )
    sns.stripplot(
        data=_df_std, x="feature", y="weight", hue="state", ax=ax_box,
        palette=_state_pal, hue_order=_states_order,
        dodge=True, alpha=0.4, zorder=1, legend=False,
    )
    _n_states = max(1, len(_states_order))
    _state_pairs_idx = list(_it.combinations(range(_n_states), 2))
    _hue_width = 0.8 / _n_states
    _y_range = _df_std["weight"].max() - _df_std["weight"].min()
    if pd.isna(_y_range) or _y_range == 0:
        _y_range = 1.0
    for _m, _feat in enumerate(_U_cols):
        _feat_df = _df_std[_df_std["feature"] == _feat]
        if _feat_df.empty:
            continue
        _y_max = _feat_df["weight"].max()
        _y_offset_step = _y_range * 0.05
        _current_y_offset = _y_max + _y_offset_step
        for _p1, _p2 in _state_pairs_idx:
            _s1 = _states_order[_p1]
            _s2 = _states_order[_p2]
            _df1 = _feat_df[_feat_df["state"] == _s1].set_index("subject")["weight"]
            _df2 = _feat_df[_feat_df["state"] == _s2].set_index("subject")["weight"]
            _common = _df1.index.intersection(_df2.index)
            if len(_common) < 2:
                continue
            _, _pval = _stats.ttest_rel(_df1.loc[_common].values, _df2.loc[_common].values)
            _star = _sig_label(_pval)
            _offset_1 = (_p1 - (_n_states - 1) / 2) * _hue_width
            _offset_2 = (_p2 - (_n_states - 1) / 2) * _hue_width
            _x1 = _m + _offset_1
            _x2 = _m + _offset_2
            _h = _y_range * 0.02
            ax_box.plot(
                [_x1, _x1, _x2, _x2],
                [_current_y_offset, _current_y_offset + _h, _current_y_offset + _h, _current_y_offset],
                lw=1,
                c="k",
            )
            ax_box.text(
                (_x1 + _x2) / 2,
                _current_y_offset + _h,
                _star,
                ha="center",
                va="bottom",
                color="k",
            )
            _current_y_offset += _y_offset_step * 1.5
    ax_box.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_box.set_xticks(range(len(_U_cols)))
    ax_box.set_xticklabels(_U_cols, rotation=20, ha="right")
    ax_box.set_xlabel("")
    ax_box.set_ylabel("Transition weight")
    ax_box.set_title(f"glmhmm-t K={K} — transition weight boxplots ({_states_str})")
    _handles_box, _labels_box = ax_box.get_legend_handles_labels()
    if len(_handles_box) >= len(_states_order):
        ax_box.legend(
            _handles_box[: len(_states_order)],
            _labels_box[: len(_states_order)],
            title="State",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=False,
        )
    fig_box.tight_layout()
    sns.despine(fig=fig_box)
    return fig_line, fig_box
