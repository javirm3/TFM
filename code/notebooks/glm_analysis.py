import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys, os
    from pathlib import Path
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Path setup
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import paths
    from scripts.fit_glm import main as fit_main, generate_model_id

    sns.set_style("white")
    return fit_main, generate_model_id, mo, np, paths, pl, plt, sns


@app.cell
def _(mo):
    ui_task = mo.ui.dropdown(
        options=["2AFC", "MCDR"],
        value="2AFC",
        label="Task:",
    )
    ui_task
    return (ui_task,)


@app.cell
def _(paths, pl, ui_task):
    df_all = pl.read_parquet(paths.DATA_PATH / "df_filtered.parquet")
    df_all = df_all.filter(pl.col("subject") != "A84")

    if ui_task.value == "2AFC":
        df_all = pl.read_parquet(paths.DATA_PATH / "alexis_combined.parquet")
        # Filter typical experiments if needed
        # df_all = df_all.filter(pl.col("experiment").is_in(['2AFC_2', '2AFC_3', '2AFC_4', '2AFC_6']))
        import glmhmmt.plots_alexis as plots
    else:
        import glmhmmt.plots as plots
    return df_all, plots


@app.cell(disabled=True)
def _(df_all, pl, plt, sns):
    import pandas as pd

    df_plot = (
        df_all.filter(pl.col("ttype_n") == 1)
        .with_columns(
            # ((pl.col("onset") / 1).floor() * 1).round(2).alias("onset_bin"),
            # ((pl.col("stim_d") / 1).floor() * 1).round(2).cast(pl.Utf8).alias("stim_bin"),
             pl.col("stim_d").qcut(4).alias("stim_bin"),  # 4 quantile bins
            (1/((pl.col("timepoint_3")-pl.col("timepoint_2")))).round(2).qcut(4).alias("speed"),
              pl.col("stimd_n").cast(pl.Int32),
            # Replace zeros with null so qcut computes quantiles only from non-zero values
            pl.when(pl.col("onset") == 0).then(None).otherwise(pl.col("onset")).alias("_onset_nz")
        ).with_columns(
            pl.when(pl.col("onset") == 0)
            .then(pl.lit("0"))
            .otherwise(
                pl.col("_onset_nz")
                .qcut(3, labels=["low", "mid", "high"])
                .cast(pl.Utf8)
            )
            .alias("onset_bin")

        ).drop("_onset_nz")
    )
    df_plot = (
        df_plot
        .group_by(["stimd_n", "onset_bin"])
        .agg(pl.col("performance").mean())
        .sort(["onset_bin", "stimd_n"])  # sort so lineplot connects correctly
    )
    print(df_plot.pivot(index="stimd_n", on="onset_bin", values="performance"))
    fig, ax = plt.subplots(figsize=(5,4))
    sns.lineplot(
        data=df_plot,
        x="stimd_n",
        y="performance",
        hue="onset_bin",
        hue_order=["0", "low", "mid", "high"],
        palette = "viridis"
    )
    sns.despine()
    ax.set_xticks(sorted(df_plot["stimd_n"].unique()))
    ax.legend(title = "Onset", frameon=False,bbox_to_anchor=(1.02, 1), )
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df_all, mo, ui_task):
    from glmhmmt.features import _ALL_EMISSION_COLS, _ALL_2AFC_EMISSION_COLS, _SF_COL_PREFIX

    is_2afc = (ui_task.value == "2AFC")
    task_name = ui_task.value

    if is_2afc:
        # Load columns dynamically if possible, or use standard lists
        _sf_cols = [c for c in df_all.columns if c.startswith(_SF_COL_PREFIX)]
        emission_cols_opts = [c for c in _ALL_2AFC_EMISSION_COLS if c != "stim_strength"] + _sf_cols
    else:
        emission_cols_opts = _ALL_EMISSION_COLS

    ui_subjects = mo.ui.multiselect(
        value=df_all["subject"].unique().to_list(),
        options=df_all["subject"].unique().to_list(),
        label="Subjects",
    )

    ui_tau = mo.ui.slider(
        start=1, stop=100, value=5, step=1, label="τ (History)"
    )

    ui_lapse = mo.ui.checkbox(value=False, label="Fit lapse rates γ_L, γ_R")

    ui_lapse_max = mo.ui.slider(
        start=0.01, stop=0.5, value=0.2, step=0.01, label="Max lapse"
    )

    ui_emission_cols = mo.ui.multiselect(
        options=emission_cols_opts,
        value=emission_cols_opts[:10], # Default selection
        label="Regressors (X)",
    )
    return is_2afc, task_name, ui_emission_cols, ui_lapse, ui_lapse_max, ui_subjects, ui_tau


@app.cell
def _(
    generate_model_id,
    mo,
    paths,
    task_name,
    ui_emission_cols,
    ui_lapse,
    ui_subjects,
    ui_tau,
):

    # Compute Hash based on current selection
    current_hash = generate_model_id(task_name, ui_tau.value, ui_emission_cols.value, lapse=ui_lapse.value)

    # Existing Models Logic
    _fits_path = paths.RESULTS / "fits" / task_name
    _existing_opts = []
    if _fits_path.exists():
        # List dirs that have config.json (valid fits)
        _existing_opts = sorted([
            d.name for d in _fits_path.iterdir() 
            if d.is_dir() and (d / "config.json").exists()
        ])
    ui_existing = mo.ui.dropdown(
        options=_existing_opts,
        value=None,
        label="Load Existing Model (Select to Override Params)",
    )

    ui_alias = mo.ui.text(
        value="",
        label="Custom Alias (Optional)",
        placeholder="e.g. my_best_fit"
    )

    fit_button = mo.ui.run_button(label="Run GLM Fit")

    mo.vstack([
        mo.md("### GLM Configuration"),
        mo.hstack([ui_subjects, ui_tau]),
        mo.hstack([ui_lapse, ui_lapse_max]),
        ui_emission_cols,
        mo.md(f"**Current Params Hash:** `{current_hash}`"),
        mo.hstack([ui_alias, ui_existing]),
        fit_button,
    ])
    return current_hash, fit_button, ui_alias, ui_existing


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(
    fit_button,
    fit_main,
    is_2afc,
    mo,
    ui_alias,
    ui_emission_cols,
    ui_lapse,
    ui_lapse_max,
    ui_subjects,
    ui_task,
    ui_tau,
):
    mo.stop(not fit_button.value, mo.md("Configure parameters and press **Run GLM Fit**."))

    with mo.status.spinner(title=f"Fitting GLM for {len(ui_subjects.value)} subjects..."):
        fit_main(
            subjects=ui_subjects.value,
            out_dir=None,
            tau=ui_tau.value,
            emission_cols=ui_emission_cols.value,
            num_classes=2 if is_2afc else 3,
            task=ui_task.value,
            model_alias=ui_alias.value if ui_alias.value else None,
            lapse=ui_lapse.value,
            lapse_max=ui_lapse_max.value,
        )

    mo.md("✅ Fit complete. Plots updating...")
    return


@app.cell
def _(
    current_hash,
    df_all,
    is_2afc,
    mo,
    np,
    paths,
    pl,
    ui_alias,
    ui_emission_cols,
    ui_existing,
    ui_subjects,
    ui_task,
    ui_tau,
):
    selected = ui_subjects.value

    if ui_existing.value:
        selected_model_id = ui_existing.value
    elif ui_alias.value:
        selected_model_id = ui_alias.value
    else:
        selected_model_id = current_hash 

    OUT = paths.RESULTS / "fits" / ui_task.value / selected_model_id

    # Feature names fallback
    _df_sel = df_all.filter(pl.col("subject").is_in(selected))
    if len(_df_sel) > 0:
        if is_2afc:
            from glmhmmt.features import build_sequence_from_df_2afc
            names = [
        "bias",       # constant 1.0
        "stim_vals",  # ILD normalised to [-1, 1] per session
        "at_choice",  # EWMA of signed choice history
        "at_error",   # EWMA of error-weighted signed choice
        "at_correct", # EWMA of correct-weighted signed choice
        "prev_choice",# previous choice
        "wsls",       # win-stay-lose-switch
    ]
        else:
            from glmhmmt.features import build_sequence_from_df
            _df_sel = _df_sel.sort("trial_idx")
            res = build_sequence_from_df(_df_sel, tau=ui_tau.value, emission_cols=ui_emission_cols.value)
            names = res[-2] if len(res) == 5 else res[-1]
    else:
        names = {}

    arrays_store = {}
    for _subj in selected:
        # Match filename format from fit_glm.py: {subj}_glm_arrays.npz
        _f = OUT / f"{_subj}_glm_arrays.npz"
        if _f.exists():
            _d = dict(np.load(_f, allow_pickle=True))
            # decode column names saved as string arrays; fall back to build output
            _d["X_cols"] = (
                list(_d["X_cols"]) if "X_cols" in _d else names.get("X_cols", [])
            )
            arrays_store[_subj] = _d

    mo.md(f"Loaded {len(arrays_store)} subjects. (From: `{selected_model_id}`)")
    return arrays_store, selected


@app.cell
def _(arrays_store, mo, paths, plots, selected):
    # Plot Weights (Folded / Agonist)
    # GLM is essentially K=1.
    K = 1

    # State Labels Trivial
    state_labels = {s: {0: "GLM"} for s in selected}

    if not arrays_store:
        mo.stop(True, mo.md("No results loaded."))

    fig_ag, fig_cls = plots.plot_emission_weights(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=arrays_store[selected[0]] if selected else {},
        K=K,
        subjects=selected,
        save_path=paths.RESULTS / "plots/GLM/emissions_coefs.png"
    )
    return fig_ag, fig_cls


@app.cell
def _(
    arrays_store,
    df_all,
    fig_ag,
    fig_cls,
    is_2afc,
    mo,
    np,
    pl,
    plots,
    selected,
    ui_task,
    ui_tau,
):
    from scripts.alexis_functions import filter_behavior
    # Psychometrics
    if not arrays_store:
        mo.stop(True)

    _sort_col = "Trial" if is_2afc else "trial_idx"
    _frames = []

    for _subj in selected:
        if _subj not in arrays_store: continue

        _p_pred = arrays_store[_subj]["p_pred"]
        _n_classes = _p_pred.shape[1]

        print(len(_p_pred))
        if is_2afc:
            _df_sub = df_all.filter(pl.col("subject") == _subj)
        else:
            _df_sub = (
                df_all.filter(pl.col("subject") == _subj)
                .sort(_sort_col)
                # Filter valid sessions length logic might apply
                .filter(pl.col("session").count().over("session") >= 2)
            )

        # Ensure length match
        if len(_df_sub) != len(_p_pred):
            # This happens if fit logic filtered differently (e.g. min session length)
            # fit_glm filters min length implicitly if build_sequence checks it?
            # Or fit_glmhmm filters explicitly.
            # My fit_glm didn't filter explicitly for length inside fit_subject except 0 checks.
            # So df_sub matches full filter.
            # Warning: build_sequence_from_df doesn't drop short sessions by default?
            # Adjust if needed.
            pass

        _cols = [pl.Series("pred_choice", np.argmax(_p_pred, axis=1).astype(int))]
        if _n_classes == 2:
            _cols += [pl.Series("pL", _p_pred[:, 0]), pl.Series("pR", _p_pred[:, 1])]
        else:
             _cols += [pl.Series("pL", _p_pred[:, 0]), pl.Series("pC", _p_pred[:, 1]), pl.Series("pR", _p_pred[:, 2])]

        if len(_df_sub) == len(_p_pred):
            _df_sub = _df_sub.with_columns(_cols)
            _frames.append(_df_sub)

    _df_all_pred = pl.concat(_frames)
    _plot_df = plots.prepare_predictions_df(_df_all_pred)
    _fig_all, _ = plots.plot_categorical_performance_all(
        _plot_df,
        f"GLM (tau={ui_tau.value})",
        arrays_store=arrays_store,   # enables smooth sigmoid via eval_glm_on_ild_grid
    )

    if ui_task == "MCDR":
        _fig_strat, _ = plots.plot_categorical_strat_by_side(_plot_df, subject="All", model_name="GLM")
    else:
        _fig_strat = None
    mo.vstack([
         mo.md("### Model Performance"),
         _fig_all
    ])
    mo.vstack([
         mo.md("### Model Performance"),
          mo.hstack([_fig_all,_fig_strat]),
         fig_ag, fig_cls
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
