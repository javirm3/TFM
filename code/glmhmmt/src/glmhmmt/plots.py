import paths

import math
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import tomllib
from matplotlib import cm, colors
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
    """
    Prepares the predictions DataFrame for plotting.
      - subject
      - response (0/1/2)
      - pL, pC, pR
      - performance (0/1)
      - stimd_c y ttype_c

    returns a Dataframe:
      - correct_bool
      - p_model_correct
    """
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

    df = df.with_columns(
        pl.when(pl.col("stimulus") == 0).then(pl.col("pL"))
        .when(pl.col("stimulus") == 1).then(pl.col("pC"))
        .when(pl.col("stimulus") == 2).then(pl.col("pR"))
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
    xpos   = np.arange(len(cats))
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
    df: pl.DataFrame,
    smoothed_probs,
    state_labels: dict,
    model_name: str,
    state_assign=None,
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
    if state_assign is not None:
        _arr = np.asarray(state_assign, dtype=int)
        T    = len(_arr)
        K    = int(_arr.max()) + 1
    else:
        T, K = smoothed_probs.shape
        _arr = np.argmax(smoothed_probs, axis=1).astype(int)

    assert df.height == T, (
        f"df has {df.height} rows but state assignment has T={T}"
    )
    df = df.with_columns(pl.Series("_state_k", _arr))

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

    fig, ax = plt.subplots(figsize=(5,5))

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
    plt.show()

    return True

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
    # bias: L/R context indicators
    ("$bias_{coh}$", [("biasL", 0), ("biasR", 1)]),
    ("$bias_{incoh}$", [("biasL", 1), ("biasR", 0)]),
    # onset
    ("$onset_{coh}$", [("onsetL", 0), ("onsetR", 1)]),
    ("$onset_{incoh}$", [("onsetL", 1), ("onsetR", 0)]),
    ("onsetC", [("onsetC", "neg_mean")]),
    # delay (shared scalar)
    ("delay", [("delay", "mean")]),
    # delay × side
    ("$D_{coh}$", [("DL", 0), ("DR", 1)]),
    ("$D_{incoh}$", [("DL", 1), ("DR", 0)]),
    ("DC", [("DC", "neg_mean")]),
    # stimulus
    ("$S_{coh}$", [("SL", 0), ("SR", 1)]),
    ("$S_{incoh}$", [("SL", 1), ("SR", 0)]),
    ("SC", [("SC", "neg_mean")]),
    # stimulus × delay
    ("$Sxd_{coh}$", [("SLxdelay", 0), ("SRxdelay", 1)]),
    ("$Sxd_{incoh}$", [("SLxdelay", 1), ("SRxdelay", 0)]),
    ("SCxd", [("SCxdelay", "neg_mean")]),
    # action history (perseveration vs alternation)
    ("$A_{coh}$", [("A_L", 0), ("A_R", 1)]),
    ("$A_{incoh}$", [("A_L", 1), ("A_R", 0)]),
]


_LABEL_RANK = {
    "Engaged": 0,
    "Disengaged": 1,
    **{f"Disengaged {i}": i for i in range(1, 10)},
}

# ── canonical state colours from config (rank-indexed) ───────────────────────
_STATE_HEX: list[str] = cfg.get("palettes", {}).get("states_hex", [
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02",
])


def _state_color(label: str, fallback_idx: int = 0) -> str:
    """Return the config-defined hex colour for a state label."""
    rank = _LABEL_RANK.get(label, fallback_idx)
    return _STATE_HEX[rank % len(_STATE_HEX)]


def _build_state_palette(
    state_labels_per_subj: dict,
) -> tuple[dict[str, str], list[str]]:
    """
    Build a (palette_dict, hue_order) pair from a {subj: {k: label}} mapping.

    Both are rank-ordered (Engaged first) so every seaborn plot that receives
    them uses the same colour and ordering regardless of K or subject set.
    """
    seen: dict[str, int] = {}
    for _slbls in state_labels_per_subj.values():
        for _k, _lbl in _slbls.items():
            if _lbl not in seen:
                seen[_lbl] = _LABEL_RANK.get(_lbl, _k)
    ordered = sorted(seen, key=lambda l: seen[l])
    pal = {lbl: _state_color(lbl, seen[lbl]) for lbl in ordered}
    return pal, ordered


def plot_emission_weights(
    arrays_store: dict,
    state_labels: dict,
    names: dict,
    K: int,
    subjects: list,
    save_path=None,
):
    """
    Emission weights: collapsed agonist view (fig_ag) + per-choice-class (fig_cls).

    Parameters
    ----------
    arrays_store : {subj: npz-dict with "emission_weights"}
    state_labels : {subj: {state_idx: label_str}}
    names        : dict with key "X_cols"
    K            : number of states
    subjects     : subject IDs to include
    save_path    : optional Path – agonist figure is saved there if provided

    Returns
    -------
    fig_ag, fig_cls
    """
    _CLS_LABELS  = ["Left (vs C)", "Right (vs C)"]
    _records     = []
    _ag_records  = []
    _feat_names  = names.get("X_cols", [])

    for _subj in subjects:
        if _subj not in arrays_store:
            continue
        _W      = arrays_store[_subj]["emission_weights"]   # (K, 2, n_feat)
        _n      = _W.shape[2]
        _fnames = (arrays_store[_subj].get("X_cols") or names.get("X_cols", []))[:_n]
        _f2i    = {f: i for i, f in enumerate(_fnames)}
        _feat_names = _fnames   # keep last subject's list for per-class axis labels

        for _k in range(_W.shape[0]):
            _slbl = state_labels.get(_subj, {}).get(_k, f"State {_k}")
            for _c in range(_W.shape[1]):
                for _fi, _fn in enumerate(_fnames):
                    _records.append({
                        "subject":     _subj,
                        "state":       _slbl,
                        "class":       _c,
                        "class_label": _CLS_LABELS[_c] if _c < len(_CLS_LABELS) else f"Class {_c}",
                        "feature":     _fn,
                        "weight":      float(_W[_k, _c, _fi]),
                    })
            for _grp_label, _members in _AG_GROUPS:
                _vals = []
                for _fn, _mode in _members:
                    if _fn not in _f2i:
                        continue
                    _fi = _f2i[_fn]
                    if isinstance(_mode, int):
                        _vals.append(float(_W[_k, _mode, _fi]))
                    elif _mode == "neg_mean":
                        _vals.append(-float(np.mean(_W[_k, :, _fi])))
                    else:
                        _vals.append(float(np.mean(_W[_k, :, _fi])))
                if _vals:
                    _ag_records.append({
                        "subject": _subj,
                        "state":   _slbl,
                        "feature": _grp_label,
                        "weight":  float(np.mean(_vals)),
                    })

    if not _records:
        raise ValueError("No emission weights found for the selected subjects.")

    _df_w     = pd.DataFrame(_records)
    _df_ag    = pd.DataFrame(_ag_records)
    _ag_order = [g for g, _ in _AG_GROUPS if g in _df_ag["feature"].values]

    # ── canonical palette (rank-ordered, config-driven) ───────────────────────
    _state_pal, _state_hue_order = _build_state_palette(state_labels)

    # ── 1. Agonist (collapsed) figure ─────────────────────────────────────────
    fig_ag, ax_ag = plt.subplots(figsize=(max(4, len(_ag_order) * 0.75), 4))
    sns.lineplot(
        data=_df_ag, x="feature", y="weight", hue="state", ax=ax_ag,
        markers=True, marker="o", markersize=8, markeredgewidth=0,
        alpha=0.85, errorbar="se",
        palette=_state_pal, hue_order=_state_hue_order,
    )
    ax_ag.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_ag.set_xticks(range(len(_ag_order)))
    ax_ag.set_xticklabels(_ag_order)
    ax_ag.set_xlabel("")
    ax_ag.set_ylabel("Agonist weight")
    ax_ag.set_title(f"Emission weights - collapsed view  (K={K})")
    ax_ag.get_legend().set_title("")
    ax_ag.legend(frameon=False)
    fig_ag.tight_layout()
    sns.despine(fig=fig_ag)
    if save_path is not None:
        fig_ag.savefig(save_path, dpi=300)

    # ── 2. Per-class figure ────────────────────────────────────────────────────
    _n_classes = _df_w["class"].nunique()
    fig_cls, axes_cls = plt.subplots(
        1, _n_classes, figsize=(6 * _n_classes, 4), sharey=True
    )
    axes_cls = np.atleast_1d(axes_cls)
    for _c, _ax in enumerate(axes_cls):
        _sub = _df_w[_df_w["class"] == _c]
        sns.lineplot(
            data=_sub, x="feature", y="weight", hue="state", ax=_ax,
            markers=True, marker="o", markersize=8, markeredgewidth=0,
            alpha=0.8, errorbar="se",
            palette=_state_pal, hue_order=_state_hue_order,
        )
        _ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        _ax.set_title(_CLS_LABELS[_c] if _c < len(_CLS_LABELS) else f"Class {_c}")
        _ax.set_xticks(range(len(_feat_names)))
        _ax.set_xticklabels(_feat_names, rotation=35, ha="right")
        _ax.set_xlabel("")
        _ax.set_ylabel("Weight" if _c == 0 else "")
        if _ax.get_legend() is not None:
            _ax.get_legend().set_title("")
    fig_cls.suptitle(f"Emission weights per choice  (K={K})", y=1.02)
    fig_cls.tight_layout()
    sns.despine(fig=fig_cls)

    return fig_ag, fig_cls


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


def plot_state_accuracy(
    arrays_store: dict,
    state_labels: dict,
    df_all,
    K: int,
    subjects: list,
    thresh: float = 0.5,
):
    """
    Per-state accuracy bar chart (Ashwood et al. 2022 method).

    Returns
    -------
    fig, summary_df (pandas DataFrame with mean_acc (%) and total_trials)
    """
    _label_order = (
        ["Engaged", "Disengaged"] if K == 2
        else ["Engaged"] + [f"Disengaged {i}" for i in range(1, K)]
    )
    _cmap     = {"All": "#999999"}
    for _ri, _lbl in enumerate(_label_order):
        _cmap[_lbl] = _STATE_HEX[_ri % len(_STATE_HEX)]
    _x_labels = ["All"] + _label_order

    _acc_records = []
    for _subj in subjects:
        if _subj not in arrays_store:
            continue
        _gamma = arrays_store[_subj].get("smoothed_probs")
        if _gamma is None:
            continue
        _df_sub = (
            df_all
            .filter(pl.col("subject") == _subj)
            .sort("trial_idx")
            .filter(pl.col("session").count().over("session") >= 2)
            .select(["stimd_n", "performance"])
        )
        _stim  = _df_sub["stimd_n"].to_numpy()
        _perf  = _df_sub["performance"].to_numpy()
        _T     = min(len(_stim), _gamma.shape[0])
        _stim  = _stim[:_T];  _perf = _perf[:_T];  _gamma = _gamma[:_T]
        _nz    = _stim != 0
        if _nz.sum() == 0:
            continue
        _acc_records.append({
            "subject": _subj, "label": "All",
            "acc": float(_perf[_nz].mean() * 100), "n": int(_nz.sum()),
        })
        for _k in range(K):
            _lbl_k  = state_labels[_subj][_k]
            _mask_k = _nz & (_gamma[:, _k] >= thresh)
            _n_k    = int(_mask_k.sum())
            _acc_records.append({
                "subject": _subj, "label": _lbl_k,
                "acc": float(_perf[_mask_k].mean() * 100) if _n_k > 0 else float("nan"),
                "n": _n_k,
            })

    _df_acc = pd.DataFrame(_acc_records)
    _tbl = (
        _df_acc.groupby("label")[["acc", "n"]]
        .agg({"acc": "mean", "n": "sum"})
        .reindex(_x_labels)
        .rename(columns={"acc": "mean_acc (%)", "n": "total_trials"})
        .round(1)
    )
    _state_rows = _tbl.loc[_label_order].dropna()
    _total_n    = _state_rows["total_trials"].sum()
    if _total_n > 0:
        _wavg = (_state_rows["mean_acc (%)"] * _state_rows["total_trials"]).sum() / _total_n
        _tbl.loc["States (E+D)"] = [round(float(_wavg), 1), int(_total_n)]

    fig, ax = plt.subplots(figsize=(2 + len(_x_labels) * 0.9, 4))
    _rng = np.random.default_rng(42)
    for _li, _lbl in enumerate(_x_labels):
        _vals = _df_acc[_df_acc["label"] == _lbl]["acc"].dropna().values
        if len(_vals) == 0:
            continue
        _mean = float(_vals.mean())
        _sem  = float(_vals.std(ddof=1) / np.sqrt(len(_vals))) if len(_vals) > 1 else 0.0
        ax.bar(_li, _mean, color=_cmap.get(_lbl, "#999999"),
               yerr=_sem, error_kw={"linewidth": 1.2, "capsize": 4},
               width=0.6, alpha=0.9, zorder=2)
        ax.text(_li, _mean + _sem + 1.2, f"{_mean:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        _jitter = _rng.uniform(-0.15, 0.15, size=len(_vals))
        ax.scatter(_li + _jitter, _vals, color="black", s=20, zorder=5, alpha=0.6)

    ax.axhline(100 / 3, color="black", linestyle="--",
               linewidth=0.9, alpha=0.5, label="Chance (33%)")
    ax.set_xticks(range(len(_x_labels)))
    ax.set_xticklabels(_x_labels, rotation=20, ha="right")
    ax.set_xlabel("State")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(30, 105)
    ax.set_title(f"Per-state accuracy  (K={K},  posterior ≥ {thresh},  non-zero stim)")
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
):
    """
    Average state-probability trajectories within a session (mean ± s.e.m. across sessions).

    Returns
    -------
    fig
    """
    _palette = _STATE_HEX
    fig, axes = plt.subplots(len(subjects), 1,
                             figsize=(10, 3.5 * len(subjects)), squeeze=False)

    for _i, _subj in enumerate(subjects):
        _ax    = axes[_i, 0]
        _probs = arrays_store[_subj]["smoothed_probs"]
        _df_sub = (
            df_all
            .filter(pl.col("subject") == _subj)
            .sort("trial_idx")
            .filter(pl.col("session").count().over("session") >= 2)
            .select(["session", "trial"])
        )
        _T        = min(_probs.shape[0], _df_sub.height)
        _probs_t  = _probs[:_T]
        _sessions = _df_sub["session"].to_numpy()[:_T]
        _trials   = _df_sub["trial"].to_numpy()[:_T]

        _sess_ids = np.unique(_sessions)
        _max_len  = max(int((_sessions == _s).sum()) for _s in _sess_ids)
        _mat = np.full((_sess_ids.size, _max_len, K), np.nan)
        for _si, _s in enumerate(_sess_ids):
            _mask  = _sessions == _s
            _p_s   = _probs_t[_mask]
            _order = np.argsort(_trials[_mask])
            _mat[_si, : _p_s.shape[0], :] = _p_s[_order]

        _mean  = np.nanmean(_mat, axis=0)
        _n_obs = np.sum(~np.isnan(_mat[:, :, 0]), axis=0)
        _sem   = np.nanstd(_mat, axis=0, ddof=1) / np.maximum(_n_obs[:, None] ** 0.5, 1)
        _x     = np.arange(_max_len)

        _slbl = state_labels.get(_subj, {k: f"State {k}" for k in range(K)})
        for _k in range(K):
            _rank  = _LABEL_RANK.get(_slbl.get(_k, ""), _k)
            _col   = _palette[_rank % len(_palette)]
            _valid = ~np.isnan(_mean[:, _k])
            _ax.plot(_x[_valid], _mean[_valid, _k], color=_col, lw=2,
                     label=_slbl.get(_k, f"State {_k}"))
            _ax.fill_between(
                _x[_valid],
                (_mean[:, _k] - _sem[:, _k])[_valid],
                (_mean[:, _k] + _sem[:, _k])[_valid],
                color=_col, alpha=0.25,
            )
        _ax.set_ylim(0, 1)
        _ax.set_xlabel("Trial within session")
        _ax.set_ylabel("State probability")
        _ax.set_title(
            f"Subject {_subj} — avg. state trajectory  (n={_sess_ids.size} sessions)"
        )
        _ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_state_occupancy(
    arrays_store: dict,
    state_labels: dict,
    df_all,
    K: int,
    subjects: list,
):
    """
    Fractional occupancy bar chart + state-change histogram per session.

    Returns
    -------
    fig
    """
    _palette = _STATE_HEX

    def _rank_lbl(lbl: str) -> int:
        if lbl in _LABEL_RANK:
            return _LABEL_RANK[lbl]
        if lbl.startswith("Disengaged "):
            _tail = lbl.split("Disengaged ", 1)[1]
            if _tail.isdigit():
                return int(_tail)
        return K + 100

    fig, axes = plt.subplots(len(subjects), 2,
                             figsize=(10, 3.5 * len(subjects)), squeeze=False)

    for _i, _subj in enumerate(subjects):
        _ax_bar  = axes[_i, 0]
        _ax_hist = axes[_i, 1]
        _probs   = arrays_store[_subj]["smoothed_probs"]
        _df_sub  = (
            df_all
            .filter(pl.col("subject") == _subj)
            .sort("trial_idx")
            .filter(pl.col("session").count().over("session") >= 2)
            .select(["session", "trial"])
        )
        _T            = min(_probs.shape[0], _df_sub.height)
        _probs_t      = _probs[:_T]
        _sessions     = _df_sub["session"].to_numpy()[:_T]
        _state_assign = np.argmax(_probs_t, axis=1)

        _slbl           = state_labels.get(_subj, {k: f"State {k}" for k in range(K)})
        _fracs           = np.array([np.mean(_state_assign == _k) for _k in range(K)])
        _bar_labels_raw  = [_slbl.get(_k, f"State {_k}") for _k in range(K)]
        _order           = np.argsort([_rank_lbl(_l) for _l in _bar_labels_raw], kind="stable")
        _fracs_ord       = _fracs[_order]
        _bar_labels      = [_bar_labels_raw[_j] for _j in _order]
        _bar_colors      = [
            _palette[_rank_lbl(_bar_labels_raw[_j]) % len(_palette)] for _j in _order
        ]

        _ax_bar.bar(range(K), _fracs_ord, color=_bar_colors, width=0.6, alpha=0.9)
        for _xi, _fv in enumerate(_fracs_ord):
            _ax_bar.text(_xi, _fv + 0.01, f"{_fv:.2f}", ha="center", va="bottom", fontsize=9)
        _ax_bar.set_xticks(range(K))
        _ax_bar.set_xticklabels(_bar_labels, rotation=15, ha="right")
        _ax_bar.set_ylim(0, 1.15)
        _ax_bar.set_ylabel("Fractional occupancy")
        _ax_bar.set_title(f"Subject {_subj} — state occupancy")

        _sess_ids  = np.unique(_sessions)
        _n_changes = [
            int(np.sum(np.diff(_state_assign[_sessions == _s]) != 0))
            for _s in _sess_ids
        ]
        _max_ch = max(_n_changes) if _n_changes else 0
        _ax_hist.hist(
            _n_changes, bins=_max_ch + 1, range=(-0.5, _max_ch + 0.5),
            color=_palette[0], edgecolor="white", alpha=0.85,
        )
        _ax_hist.set_xlabel("State changes per session")
        _ax_hist.set_ylabel("Number of sessions")
        _ax_hist.set_title(f"Subject {_subj} — state changes / session")

    fig.tight_layout()
    sns.despine(fig=fig)
    return fig


def plot_session_deepdive(
    arrays_store: dict,
    state_labels: dict,
    df_all,
    names: dict,
    K: int,
    subj: str,
    sess: int,
):
    """
    Session deep-dive: P(Engaged) + cumulative accuracy (twin axis) + action traces.

    Action traces are auto-detected: prefers A_plus / A_minus from U (glmhmmt),
    then falls back to A_L / A_C / A_R from X (glmhmm).

    Returns
    -------
    fig
    """
    _df_all_sub = (
        df_all
        .filter(pl.col("subject") == subj)
        .sort("trial_idx")
        .filter(pl.col("session").count().over("session") >= 2)
    )
    _sess_mask = _df_all_sub["session"].to_numpy() == sess
    _df_sub = (
        df_all
        .filter((pl.col("subject") == subj) & (pl.col("session") == sess))
        .sort("trial_idx")
    )

    _probs_all = arrays_store[subj]["smoothed_probs"]
    _probs     = _probs_all[_sess_mask]
    _y         = _df_sub["performance"].to_numpy()
    _stim      = _df_sub["stimd_n"].to_numpy()
    _response  = _df_sub["response"].to_numpy().astype(int)
    _T         = _probs.shape[0]
    _x         = np.arange(_T)

    # ── auto-detect action traces ─────────────────────────────────────────────
    _X_cols_s = arrays_store[subj].get("X_cols") or names.get("X_cols", [])
    _X_idx    = {f: i for i, f in enumerate(_X_cols_s)}
    _X_sess   = arrays_store[subj]["X"][_sess_mask]

    _trace_sources = {}
    _U_raw = arrays_store[subj].get("U")
    if _U_raw is not None:
        _U_cols_s = arrays_store[subj].get("U_cols") or names.get("U_cols", [])
        _U_idx    = {f: i for i, f in enumerate(_U_cols_s)}
        _U_sess   = _U_raw[_sess_mask]
        for _tc in ["A_plus", "A_minus"]:
            if _tc in _U_idx:
                _trace_sources[_tc] = (_U_sess, _U_idx[_tc])
    for _tc in ["A_R", "A_L", "A_C"]:
        if _tc in _X_idx and _tc not in _trace_sources:
            _trace_sources[_tc] = (_X_sess, _X_idx[_tc])

    _trace_colors = {
        "A_plus": "royalblue", "A_minus": "gold",
        "A_L": "royalblue", "A_C": "gold", "A_R": "tomato",
    }

    # ── cumulative accuracy (non-zero-stim trials) ────────────────────────────
    _nz = _stim != 0
    _cum_acc = np.full(_T, np.nan)
    _cum_n, _cum_s = 0, 0.0
    for _ti in range(_T):
        if _nz[_ti]:
            _cum_s += _y[_ti]; _cum_n += 1
        if _cum_n > 0:
            _cum_acc[_ti] = 100.0 * _cum_s / _cum_n

    # ── figure ────────────────────────────────────────────────────────────────
    _palette   = _STATE_HEX
    _slbl      = state_labels.get(subj, {k: f"State {k}" for k in range(K)})
    _engaged_k = next(
        (k for k in range(K) if _LABEL_RANK.get(_slbl.get(k, ""), k) == 0), 0
    )

    fig, (_ax1, _ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5]},
    )

    _ax1.plot(_x, _probs[:, _engaged_k], color=_palette[0], lw=2,
              label=f"P({_slbl.get(_engaged_k, 'Engaged')})")
    _choice_cols = {0: "royalblue", 1: "gold", 2: "tomato"}
    _choice_lbls = {0: "L", 1: "C", 2: "R"}
    for _resp, _c in _choice_cols.items():
        _m = _response == _resp
        _ax1.scatter(_x[_m], np.ones(_m.sum()) * 1.03, c=_c, s=5, marker="|",
                     label=_choice_lbls[_resp],
                     transform=_ax1.get_xaxis_transform(), clip_on=False)
    _ax1.set_ylim(0, 1)
    _ax1.set_ylabel("State probability")
    _ax1.set_title(f"Subject {subj}  —  session {sess}  ({_T} trials)")

    _ax1r = _ax1.twinx()
    _ax1r.plot(_x, _cum_acc, color="black", lw=1.8, linestyle="-", alpha=0.7,
               label="Cumul. accuracy")
    _ax1r.axhline(100 / 3, color="grey", lw=0.9, linestyle="--", alpha=0.5)
    _ax1r.set_ylim(0, 105)
    _ax1r.set_ylabel("Accuracy (%)", color="black")
    _lines1, _labs1   = _ax1.get_legend_handles_labels()
    _lines1r, _labs1r = _ax1r.get_legend_handles_labels()
    _ax1.legend(_lines1 + _lines1r, _labs1 + _labs1r,
                bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)

    if _trace_sources:
        for _tc, (_arr, _ci) in _trace_sources.items():
            _ax2.plot(_x, _arr[:, _ci], label=_tc,
                      color=_trace_colors.get(_tc, "gray"), lw=1.5, alpha=0.85)
    else:
        _ax2.text(0.5, 0.5, "No action-trace features found",
                  ha="center", va="center", transform=_ax2.transAxes)
    _ax2.set_ylabel("Action trace")
    _ax2.set_ylim(0, None)
    _ax2.set_xlabel("Trial within session")
    _ax2.legend(bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(right=0.82)
    sns.despine(fig=fig, right=False)
    return fig


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
    arrays_store: dict,
    names: dict,
    K: int,
    subjects: list,
    state_labels: dict | None = None,
):
    """
    Input-dependent transition weights (glmhmm-t only).

    Produces three figures:
      fig_line – standardised lineplot (mean-centred across states)
      fig_std  – horizontal barplot with significance brackets
      fig_raw  – raw per-transition lineplot

    Parameters
    ----------
    state_labels : {subj: {state_idx: label_str}} – if provided, semantic
                   labels (e.g. "Engaged") and config-driven colours are used
                   consistently across all three figures.

    Returns
    -------
    fig_line, fig_std, fig_raw
    """
    import copy as _copy
    from scipy import stats as _stats

    _selected = [s for s in subjects
                 if s in arrays_store and "transition_weights" in arrays_store[s]]
    if not _selected:
        raise ValueError("No transition weights found for selected subjects.")

    # ── resolve per-subject and global labels ─────────────────────────────────
    _slbls_map: dict = {}
    for _subj in _selected:
        _slbls_map[_subj] = (
            (state_labels or {}).get(_subj) or {k: f"State {k}" for k in range(K)}
        )

    # Build a consensus label per state index (first subject that has it)
    _resolved: dict[int, str] = {}
    for _k in range(K):
        for _subj in _selected:
            _lbl = _slbls_map[_subj].get(_k)
            if _lbl:
                _resolved[_k] = _lbl
                break
        if _k not in _resolved:
            _resolved[_k] = f"State {_k}"

    # Rank-ordered list + palette dict (consistent with all other plots)
    _states_order = [
        _resolved[k]
        for k in sorted(_resolved, key=lambda k: _LABEL_RANK.get(_resolved[k], k))
    ]
    _state_pal: dict[str, str] = {
        lbl: _state_color(lbl, i) for i, lbl in enumerate(_states_order)
    }
    _state_pairs = [
        (_states_order[a], _states_order[b])
        for a in range(len(_states_order))
        for b in range(a + 1, len(_states_order))
    ]

    _D_first = arrays_store[_selected[0]]["transition_weights"].shape[2]
    _U_cols  = (
        arrays_store[_selected[0]].get("U_cols") or names.get("U_cols", [])
    )[:_D_first]

    # ── standardised records ──────────────────────────────────────────────────
    _std_records = []
    for _subj in _selected:
        _W_raw    = arrays_store[_subj]["transition_weights"]   # (K, K, D)
        _D        = _W_raw.shape[2]
        _U_cols_s = (arrays_store[_subj].get("U_cols") or names.get("U_cols", []))[:_D]
        _W_avg    = _W_raw.mean(axis=0)                         # (K, D)
        _W_aug    = np.vstack([_W_avg, np.zeros((1, _W_avg.shape[1]))])  # (K+1, D)
        _v1       = -np.mean(_W_aug, axis=0)
        _W_std    = _copy.deepcopy(_W_aug)
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
    _states_str = " / ".join(_states_order)
    ax_line.set_title(f"glmhmm-t K={K} — transition weights by state ({_states_str})")
    if ax_line.get_legend() is not None:
        ax_line.get_legend().set_title("")
    fig_line.tight_layout()
    sns.despine(fig=fig_line)

    # ── fig_std: horizontal barplot with significance brackets ─────────────────
    _bar_height    = 0.8 / K
    _group_offsets = np.linspace(-(K - 1) / 2, (K - 1) / 2, K) * _bar_height
    _n_feats       = len(_U_cols)

    fig_std, ax_std = plt.subplots(figsize=(6, max(3, _n_feats * K * 0.45)))
    for _fi, _feat in enumerate(_U_cols):
        for _ki, _st in enumerate(_states_order):
            _vals = _df_std[
                (_df_std["feature"] == _feat) & (_df_std["state"] == _st)
            ]["weight"].values
            _mean = _vals.mean() if len(_vals) else 0.0
            _sem  = _vals.std(ddof=1) / np.sqrt(len(_vals)) if len(_vals) > 1 else 0.0
            _y    = _fi + _group_offsets[_ki]
            ax_std.barh(
                _y, _mean, height=_bar_height * 0.85,
                xerr=_sem, error_kw={"linewidth": 1.2, "capsize": 3},
                color=_state_pal[_st], alpha=0.85,
                label=_st if _fi == 0 else "_nolegend_",
            )
            ax_std.scatter(_vals, np.full(len(_vals), _y),
                           color="black", s=16, zorder=5, alpha=0.5)

    _xlim_r  = ax_std.get_xlim()[1]
    _xdr     = abs(ax_std.get_xlim()[1] - ax_std.get_xlim()[0])
    _bgap    = _xdr * 0.06
    for _fi, _feat in enumerate(_U_cols):
        for _pi, (_st_a, _st_b) in enumerate(_state_pairs):
            _lbl = _sig_label(_sig_results[(_feat, _st_a, _st_b)])
            _y_a = _fi + _group_offsets[_states_order.index(_st_a)]
            _y_b = _fi + _group_offsets[_states_order.index(_st_b)]
            _bx  = _xlim_r + _bgap * (1 + _pi)
            ax_std.plot([_bx, _bx], [_y_a, _y_b], color="black", lw=1.1, clip_on=False)
            _tick = _xdr * 0.01
            ax_std.plot([_bx - _tick, _bx], [_y_a, _y_a], color="black", lw=1.1, clip_on=False)
            ax_std.plot([_bx - _tick, _bx], [_y_b, _y_b], color="black", lw=1.1, clip_on=False)
            ax_std.text(_bx + _xdr * 0.015, (_y_a + _y_b) / 2, _lbl,
                        va="center", ha="left", fontsize=9, clip_on=False,
                        color="black" if _lbl != "ns" else "grey")
    ax_std.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_std.set_yticks(range(_n_feats))
    ax_std.set_yticklabels(_U_cols)
    ax_std.set_xlabel("Transition weight")
    ax_std.set_ylabel("")
    ax_std.set_title(f"glmhmm-t K={K} — transition weights ({_states_str})")
    ax_std.legend(title="State", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig_std.tight_layout()
    sns.despine(fig=fig_std)

    # ── fig_raw: raw per-transition lineplot ───────────────────────────────────
    _raw_records = []
    for _subj in _selected:
        _W_raw = arrays_store[_subj]["transition_weights"]
        _slbl  = _slbls_map[_subj]
        for _kf in range(_W_raw.shape[0]):
            for _kt in range(_W_raw.shape[1]):
                _from = _slbl.get(_kf, f"State {_kf}")
                _to   = _slbl.get(_kt, f"State {_kt}")
                for _fi, _fname in enumerate(_U_cols):
                    _raw_records.append({
                        "subject": _subj, "transition": f"{_from}→{_to}",
                        "feature": _fname, "weight": float(_W_raw[_kf, _kt, _fi]),
                    })
    _df_raw      = pd.DataFrame(_raw_records)
    _transitions = sorted(_df_raw["transition"].unique())
    fig_raw, axes_raw = plt.subplots(
        1, len(_transitions), figsize=(4 * len(_transitions), 4), sharey=True
    )
    axes_raw = np.atleast_1d(axes_raw)
    for _ax_r, _trans in zip(axes_raw, _transitions):
        # derive color from the "from" state label
        _from_lbl = _trans.split("→")[0]
        _t_color  = _state_pal.get(_from_lbl, "steelblue")
        sns.lineplot(
            data=_df_raw[_df_raw["transition"] == _trans],
            x="feature", y="weight", ax=_ax_r,
            markers=True, marker="o", markersize=10,
            markeredgewidth=0, alpha=0.75, color=_t_color, errorbar="se",
        )
        _ax_r.set_title(_trans)
        _ax_r.set_xticks(range(len(_U_cols)))
        _ax_r.set_xticklabels(_U_cols, rotation=20, ha="right")
        _ax_r.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        _ax_r.set_xlabel("")
        _ax_r.set_ylabel("Weight" if _ax_r is axes_raw[0] else "")
    fig_raw.suptitle(f"glmhmm-t K={K} — raw transition weights per state pair", y=1.02)
    fig_raw.tight_layout()
    sns.despine(fig=fig_raw)

    return fig_line, fig_std, fig_raw
