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

    _state_palette = sns.color_palette("tab10", n_colors=K)
    _label_rank = {"Engaged": 0, "Disengaged": 1,
                   **{f"Disengaged {i}": i for i in range(1, K)}}
    _state_colors = {
        k: _state_palette[_label_rank.get(state_labels.get(k, ""), k % len(_state_palette))]
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

