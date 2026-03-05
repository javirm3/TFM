import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys, os
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import paths
    from tasks import get_adapter

    sns.set_style("white")
    return Path, get_adapter, mo, np, paths, pl, plt, sns


# ── Task & folder configuration ──────────────────────────────────────────────


@app.cell
def _(get_adapter, mo, paths):
    ui_task = mo.ui.dropdown(
        options=["MCDR", "2AFC"],
        value="MCDR",
        label="Task",
    )
    adapter = get_adapter(ui_task.value)

    def _model_aliases(task: str, kind: str) -> list:
        p = paths.RESULTS / "fits" / task / kind
        if not p.exists():
            return []
        return sorted([d.name for d in p.iterdir() if d.is_dir()])

    ui_glm_dir = mo.ui.dropdown(
        options=_model_aliases(ui_task.value, "glm"),
        value=None,
        label="GLM alias",
    )
    ui_glmhmm_dir = mo.ui.dropdown(
        options=_model_aliases(ui_task.value, "glmhmm"),
        value=None,
        label="GLMHMM alias",
    )
    ui_glmhmmt_dir = mo.ui.dropdown(
        options=_model_aliases(ui_task.value, "glmhmmt"),
        value=None,
        label="GLMHMM-T alias",
    )

    mo.vstack([
        mo.md("### Model Comparison — Configuration"),
        mo.md(
            "Select the model alias for each model kind. "
            "Set to **None** to skip that model."
        ),
        mo.hstack([ui_task]),
        mo.hstack([ui_glm_dir, ui_glmhmm_dir, ui_glmhmmt_dir]),
    ])
    return adapter, ui_glm_dir, ui_glmhmm_dir, ui_glmhmmt_dir, ui_task


# ── Subject & K range ─────────────────────────────────────────────────────────


@app.cell
def _(adapter, mo, paths, pl):
    _df = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    _df = adapter.subject_filter(_df)
    _all_subjects = _df["subject"].unique().sort().to_list()

    ui_subjects = mo.ui.multiselect(
        options=_all_subjects,
        value=_all_subjects,
        label="Subjects",
    )
    ui_K_range = mo.ui.range_slider(
        start=1, stop=10, step=1, value=[1, 5],
        full_width=True, label="K range",
    )

    mo.vstack([
        mo.hstack([ui_subjects]),
        mo.hstack([mo.md("K range:"), ui_K_range]),
    ])
    return ui_K_range, ui_subjects


# ── Load metrics from cached _metrics.parquet files ──────────────────────────


@app.cell
def _(mo, paths, pl, ui_glm_dir, ui_glmhmm_dir, ui_glmhmmt_dir, ui_task):
    def _load_dir(folder_name, expected_model_kind):
        """Scan a fit dir for *_metrics.parquet files and concat them."""
        if not folder_name:
            return None
        d = paths.RESULTS / "fits" / ui_task.value / expected_model_kind / folder_name
        if not d.exists():
            return None
        files = list(d.glob("*_metrics.parquet"))
        if not files:
            return None
        frames = []
        for f in files:
            try:
                frames.append(pl.read_parquet(f))
            except Exception:
                pass
        if not frames:
            return None
        df = pl.concat(frames, how="diagonal")
        # Normalise: glm writes nll+n_trials; glmhmm/t writes ll_per_trial
        if "nll" in df.columns and "ll_per_trial" not in df.columns:
            df = df.with_columns(
                (-pl.col("nll") / pl.col("n_trials")).alias("ll_per_trial")
            )
        if "K" not in df.columns:
            df = df.with_columns(pl.lit(1).alias("K"))
        if "model_kind" not in df.columns:
            df = df.with_columns(pl.lit(expected_model_kind).alias("model_kind"))
        keep = ["subject", "K", "model_kind", "ll_per_trial", "bic", "acc"]
        return df.select([c for c in keep if c in df.columns])

    _parts = []
    for _name, _kind in [
        (ui_glm_dir.value,    "glm"),
        (ui_glmhmm_dir.value, "glmhmm"),
        (ui_glmhmmt_dir.value, "glmhmmt"),
    ]:
        _p = _load_dir(_name, _kind)
        if _p is not None:
            _parts.append(_p)

    if _parts:
        results_long = pl.concat(_parts, how="diagonal")
    else:
        results_long = pl.DataFrame(
            schema={
                "subject": pl.Utf8, "K": pl.Int64, "model_kind": pl.Utf8,
                "ll_per_trial": pl.Float64, "bic": pl.Float64, "acc": pl.Float64,
            }
        )

    mo.stop(
        results_long.is_empty(),
        mo.md("⚠️  No metrics loaded — select at least one fit folder above."),
    )
    mo.md(
        f"Loaded **{results_long.height}** fit rows from "
        f"**{len(_parts)}** model folder(s)."
    )
    return (results_long,)


# ── Filter to selected subjects & K range ────────────────────────────────────


@app.cell
def _(pl, results_long, ui_K_range, ui_subjects):
    K_min, K_max = ui_K_range.value
    results_filtered = results_long.filter(
        pl.col("subject").is_in(ui_subjects.value)
        & pl.col("K").is_between(K_min, K_max)
    )
    results_filtered
    return (results_filtered,)


# ── Aggregate: mean ± SEM per (model_kind, K) ────────────────────────────────


@app.cell
def _(pl, results_filtered):
    agg = (
        results_filtered.group_by(["model_kind", "K"])
        .agg([
            pl.len().alias("n_subjects"),
            pl.mean("ll_per_trial").alias("ll_mean"),
            pl.std("ll_per_trial").alias("ll_std"),
            pl.mean("bic").alias("bic_mean"),
            pl.std("bic").alias("bic_std"),
            pl.mean("acc").alias("acc_mean"),
        ])
        .with_columns([
            (pl.col("ll_std")  / pl.col("n_subjects").sqrt()).alias("ll_sem"),
            (pl.col("bic_std") / pl.col("n_subjects").sqrt()).alias("bic_sem"),
        ])
        .sort(["model_kind", "K"])
    )
    agg
    return (agg,)


# ── BIC & LL/trial comparison curves ─────────────────────────────────────────


@app.cell
def _(agg, plt, sns):
    _MODEL_STYLES = {
        "glm":      {"color": "#4C72B0", "marker": "s", "label": "GLM (K=1)"},
        "glmhmm":   {"color": "#55A868", "marker": "o", "label": "GLMHMM"},
        "glmhmmt":  {"color": "#C44E52", "marker": "^", "label": "GLMHMM-T"},
    }

    fig_cmp, (ax_ll, ax_bic) = plt.subplots(1, 2, figsize=(12, 4))

    for _kind_tup, _group in agg.group_by("model_kind"):
        _kind = _kind_tup[0]
        _g = _group.sort("K").to_pandas()
        _st = _MODEL_STYLES.get(_kind, {"color": "grey", "marker": "o", "label": _kind})
        _x = _g["K"].values
        ax_ll.errorbar(
            _x, _g["ll_mean"], yerr=_g["ll_sem"],
            color=_st["color"], marker=_st["marker"],
            label=_st["label"], capsize=3, linewidth=1.5,
        )
        ax_bic.errorbar(
            _x, _g["bic_mean"], yerr=_g["bic_sem"],
            color=_st["color"], marker=_st["marker"],
            label=_st["label"], capsize=3, linewidth=1.5,
        )

    _K_all = agg["K"].unique().sort().to_list()
    for _ax, _ylabel, _title in [
        (ax_ll,  "Log-likelihood / trial", "LL / trial  (higher = better)"),
        (ax_bic, "BIC",                    "BIC  (lower = better)"),
    ]:
        _ax.set_xlabel("Number of states K")
        _ax.set_ylabel(_ylabel)
        _ax.set_title(_title)
        _ax.set_xticks(_K_all)
        _ax.legend(frameon=False)
        sns.despine(ax=_ax)

    fig_cmp.tight_layout()
    fig_cmp
    return (fig_cmp,)


# ── Per-subject LL/trial heatmap ──────────────────────────────────────────────


@app.cell
def _(mo, pl, plt, results_filtered, sns):
    _pivot_df = (
        results_filtered
        .with_columns(
            (pl.col("model_kind") + "_K" + pl.col("K").cast(pl.Utf8)).alias("model_K")
        )
        .pivot(index="subject", on="model_K", values="ll_per_trial")
        .to_pandas()
        .set_index("subject")
    )

    mo.stop(_pivot_df.empty, mo.md("No data to plot."))

    _fig_heat, _ax_h = plt.subplots(
        figsize=(max(6, _pivot_df.shape[1] * 0.9), max(4, _pivot_df.shape[0] * 0.4))
    )
    sns.heatmap(
        _pivot_df, ax=_ax_h, cmap="RdYlGn",
        annot=True, fmt=".3f", linewidths=0.3,
        cbar_kws={"label": "LL / trial"},
    )
    _ax_h.set_title("Log-likelihood per trial — subject × model/K")
    _ax_h.set_xlabel("")
    _fig_heat.tight_layout()
    _fig_heat
    return


# ── Accuracy comparison ───────────────────────────────────────────────────────


@app.cell
def _(agg, plt, sns):
    _MODEL_STYLES = {
        "glm":      {"color": "#4C72B0", "marker": "s", "label": "GLM (K=1)"},
        "glmhmm":   {"color": "#55A868", "marker": "o", "label": "GLMHMM"},
        "glmhmmt":  {"color": "#C44E52", "marker": "^", "label": "GLMHMM-T"},
    }

    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
    for _kind_tup, _group in agg.group_by("model_kind"):
        _kind = _kind_tup[0]
        _g = _group.sort("K").to_pandas()
        _st = _MODEL_STYLES.get(_kind, {"color": "grey", "marker": "o", "label": _kind})
        ax_acc.plot(
            _g["K"], _g["acc_mean"],
            color=_st["color"], marker=_st["marker"],
            label=_st["label"], linewidth=1.5,
        )

    ax_acc.set_xlabel("Number of states K")
    ax_acc.set_ylabel("Accuracy (mean over subjects)")
    ax_acc.set_title("Model accuracy vs K")
    ax_acc.legend(frameon=False)
    sns.despine(ax=ax_acc)
    fig_acc.tight_layout()
    fig_acc
    return (fig_acc,)


# ── Weight visualisation — load arrays from .npz ─────────────────────────────


@app.cell
def _(mo, paths, ui_task):
    def _model_aliases_viz(task: str, kind: str) -> list:
        p = paths.RESULTS / "fits" / task / kind
        if not p.exists():
            return []
        return sorted([d.name for d in p.iterdir() if d.is_dir()])

    ui_viz_model = mo.ui.dropdown(
        options=["glm", "glmhmm", "glmhmmt"],
        value="glmhmm",
        label="Model kind",
    )
    ui_viz_alias = mo.ui.dropdown(
        options=_model_aliases_viz(ui_task.value, ui_viz_model.value),
        value=None,
        label="Model alias",
    )
    ui_viz_K = mo.ui.slider(start=1, stop=8, value=2, label="K (for GLMHMM/T)")

    mo.vstack([
        mo.md("### Emission weights from cached fits"),
        mo.hstack([ui_viz_model, ui_viz_alias, ui_viz_K]),
    ])
    return ui_viz_K, ui_viz_alias, ui_viz_model


@app.cell
def _(
    adapter,
    mo,
    np,
    paths,
    pl,
    ui_subjects,
    ui_task,
    ui_viz_K,
    ui_viz_alias,
    ui_viz_model,
):
    mo.stop(
        not ui_viz_alias.value,
        mo.md("Select a model alias above to visualise weights."),
    )

    _kind = ui_viz_model.value
    _viz_dir = paths.RESULTS / "fits" / ui_task.value / _kind / ui_viz_alias.value
    _K = ui_viz_K.value
    _suffix = {"glm": "glm", "glmhmm": "glmhmm", "glmhmmt": "glmhmmt"}[_kind]

    _arrays_store = {}
    for _s in ui_subjects.value:
        _f = _viz_dir / f"{_s}_K{_K}_{_suffix}_arrays.npz"
        if not _f.exists() and _kind == "glm":
            _f = _viz_dir / f"{_s}_glm_arrays.npz"
        if _f.exists():
            _arrays_store[_s] = dict(np.load(_f, allow_pickle=True))

    mo.stop(
        not _arrays_store,
        mo.md(
            f"No `*_{_suffix}_arrays.npz` files found for K={_K} "
            f"in `{ui_viz_alias.value}`."
        ),
    )

    # Feature names from adapter
    _df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    _df_all = adapter.subject_filter(_df_all)
    _subj0 = next(iter(_arrays_store))
    _df_sub = _df_all.filter(pl.col("subject") == _subj0).sort(adapter.sort_col)
    _, _, _, _names = adapter.load_subject(_df_sub, tau=0.3, emission_cols=adapter.default_emission_cols())

    _plots = adapter.get_plots()
    _state_labels = {s: {k: f"State {k+1}" for k in range(_K)} for s in _arrays_store}

    try:
        _fig_ag, _fig_cls = _plots.plot_emission_weights(
            arrays_store=_arrays_store,
            state_labels=_state_labels,
            names=_names,
            K=_K,
            subjects=list(_arrays_store.keys()),
        )
        mo.vstack([
            mo.md(f"**{_kind}  K={_K}**  —  {ui_viz_alias.value}"),
            _fig_ag,
            _fig_cls,
        ])
    except Exception as _e:
        mo.md(f"⚠️  Could not render weight plot: `{_e}`")
    return


# ── Optional re-fit ───────────────────────────────────────────────────────────


@app.cell
def _(mo):
    refit_button = mo.ui.run_button(
        label="⚠️  Re-fit selected (overwrites cached metrics)"
    )
    mo.vstack([
        mo.md("---\n### Re-fit (optional)"),
        mo.md(
            "> Runs the fit scripts for the selected task / subjects / K range "
            "and overwrites `_metrics.parquet` files in the chosen folders.  \n"
            "> Reload the page afterward to see updated metrics."
        ),
        refit_button,
    ])
    return (refit_button,)


@app.cell
def _(
    mo,
    paths,
    refit_button,
    ui_K_range,
    ui_glmhmm_dir,
    ui_glmhmmt_dir,
    ui_subjects,
    ui_task,
):
    mo.stop(
        not refit_button.value,
        mo.md("Press the button above to trigger re-fitting."),
    )

    import sys as _sys, os as _os
    _sys.path.append(_os.path.join(_os.path.dirname(__file__), ".."))
    from scripts.fit_glmhmm  import main as _fit_glmhmm_main
    from scripts.fit_glmhmmt import main as _fit_glmhmmt_main

    _K_min, _K_max = ui_K_range.value
    _K_list = list(range(max(2, _K_min), _K_max + 1))

    with mo.status.spinner(title="Re-fitting GLMHMM…"):
        if ui_glmhmm_dir.value:
            _fit_glmhmm_main(
                subjects=ui_subjects.value,
                K_list=_K_list,
                out_dir=paths.RESULTS / "fits" / ui_task.value / "glmhmm" / ui_glmhmm_dir.value,
                task=ui_task.value,
            )

    with mo.status.spinner(title="Re-fitting GLMHMM-T…"):
        if ui_glmhmmt_dir.value:
            _fit_glmhmmt_main(
                subjects=ui_subjects.value,
                K_list=_K_list,
                out_dir=paths.RESULTS / "fits" / ui_task.value / "glmhmmt" / ui_glmhmmt_dir.value,
                task=ui_task.value,
            )

    mo.md("✅  Re-fit complete. Reload the notebook to refresh cached metrics.")
    return


if __name__ == "__main__":
    app.run()
