import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


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
    from tasks import get_adapter
    from widgets import ModelManagerWidget, CoefTweakerWidget

    sns.set_style("white")
    return (
        CoefTweakerWidget,
        ModelManagerWidget,
        fit_main,
        generate_model_id,
        get_adapter,
        mo,
        np,
        paths,
        pl,
        plt,
        sns,
    )


@app.cell
def _(mo):
    ui_task = mo.ui.dropdown(
        options=["2AFC", "MCDR"],
        value="MCDR",
        label="Task:",
    )
    ui_task
    return (ui_task,)


@app.cell
def _(get_adapter, paths, pl, ui_task):
    adapter = get_adapter(ui_task.value)
    df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df_all = adapter.subject_filter(df_all)
    plots = adapter.get_plots()
    return adapter, df_all, plots


@app.cell
def _(paths, ui_task):
    import json as _json
    _fits_path = paths.RESULTS / "fits" / ui_task.value / "glm"
    _existing_opts = []
    if _fits_path.exists():
        _existing_opts = sorted([
            d.name for d in _fits_path.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ])
    return


@app.cell
def _(paths, ui_task):
    import json as _json
    def load_cfg_for_model(model_name):
        loaded_cfg = {}
        if model_name:
            _cfg_path = (
                paths.RESULTS / "fits" / ui_task.value / "glm" / model_name / "config.json"
            )
            if _cfg_path.exists():
                loaded_cfg = _json.loads(_cfg_path.read_text())
        return loaded_cfg

    return


@app.cell
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
def _(ModelManagerWidget, adapter, df_all, mo, ui_task):
    is_2afc = adapter.num_classes == 2
    task_name = ui_task.value

    emission_cols_opts = (
        adapter.default_emission_cols() + adapter.sf_cols(df_all)
        if is_2afc
        else adapter.default_emission_cols()
    )

    _subjects = df_all["subject"].unique().to_list()

    # Preload config if there's an existing model selected
    # But since the widget handles the state, we just pass initial values
    _existing_opts = []
    mm_widget = ModelManagerWidget(
        model_type="glm",
        is_2afc=is_2afc,
        subjects_list=_subjects,
        existing_models=_existing_opts,
        subjects=_subjects,
        k_options=[1], # GLM is K=1
        K=1,
        tau=5,
        lapse=False,
        lapse_max=0.2,
        emission_cols_options=emission_cols_opts,
        emission_cols=emission_cols_opts[:10],
        transition_cols_options=[],
        transition_cols=[]
    )

    # We use mo.ui.anywidget to wrap it so it integrates with Marimo
    ui_model_manager = mo.ui.anywidget(mm_widget)

    # We display it
    ui_model_manager
    return is_2afc, task_name, ui_model_manager


@app.cell
def _(generate_model_id, task_name, ui_model_manager):
    _val = ui_model_manager.value
    current_hash = generate_model_id(
        task_name, 
        _val["tau"], 
        _val["emission_cols"], 
        lapse=_val["lapse"]
    )
    return (current_hash,)


@app.cell
def _():
    return


@app.cell
def _(fit_main, mo, ui_model_manager, ui_task):
    _val = ui_model_manager.value
    _clicks = _val["run_fit_clicks"]

    mo.stop(_clicks == 0, mo.md("Configure parameters and press **RUN FIT 🚀**."))

    with mo.status.spinner(title=f"Fitting GLM for {len(_val['subjects'])} subjects..."):
        fit_main(
            subjects=_val["subjects"],
            out_dir=None,
            tau=_val["tau"],
            emission_cols=_val["emission_cols"],
            task=ui_task.value,
            model_alias=_val["alias"] if _val["alias"] else None,
            lapse=_val["lapse"],
            lapse_max=_val["lapse_max"],
        )

    mo.md("✅ Fit complete. Plots updating...")
    return


@app.cell
def _(
    adapter,
    current_hash,
    df_all,
    mo,
    np,
    paths,
    pl,
    ui_model_manager,
    ui_task,
):
    _val = ui_model_manager.value
    selected = _val["subjects"]

    if _val["existing_model"]:
        selected_model_id = _val["existing_model"]
    elif _val["alias"]:
        selected_model_id = _val["alias"]
    else:
        selected_model_id = current_hash 

    OUT = paths.RESULTS / "fits" / ui_task.value / "glm" / selected_model_id

    # Feature names from adapter (uniform for both tasks)
    _df_sel = df_all.filter(pl.col("subject").is_in(selected))
    if len(_df_sel) > 0:
        _df_sel = _df_sel.sort(adapter.sort_col)
        _, _, _, names = adapter.load_subject(
            _df_sel, tau=_val["tau"], emission_cols=_val["emission_cols"]
        )
    else:
        names = {"X_cols": [], "U_cols": []}

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
            # ── Backward-compatibility: old fit_glm.py saved W_R at index 0.
            # New convention stores W_L (negative stim weight) at index 0.
            # Detect old files by sign of stim weight and negate to W_L.
            _W = _d.get("emission_weights")
            if _W is not None:
                _stim_names = {"stim_vals", "stim_d", "ild_norm"}
                _stim_idx = next(
                    (i for i, c in enumerate(_d["X_cols"]) if c in _stim_names), None
                )
                if _stim_idx is not None and float(_W[0, 0, _stim_idx]) > 0:
                    _d["emission_weights"] = -_W  # W_R → W_L (negate)
            arrays_store[_subj] = _d

    mo.md(f"Loaded {len(arrays_store)} subjects from `{selected_model_id}`")
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
    adapter,
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
    ui_model_manager,
):
    from scripts.alexis_functions import filter_behavior
    from scipy.special import log_softmax, softmax
    # Psychometrics
    if not arrays_store:
        mo.stop(True)

    _val_tau = ui_model_manager.value["tau"]

    _sort_col = adapter.sort_col
    _frames = []




    for _subj in selected:
        if _subj not in arrays_store: continue

        _p_pred = arrays_store[_subj]["p_pred"]
        _n_classes = _p_pred.shape[1]

        _df_sub = (
            df_all.filter(pl.col("subject") == _subj)
            .sort(_sort_col)
        )
        if not is_2afc:
            _df_sub = _df_sub.filter(pl.col("session").count().over("session") >= 2)

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
    # arrays_store kwarg only exists in plots_alexis (2AFC); MCDR version doesn't accept it
    _perf_kwargs = {"arrays_store": arrays_store} if is_2afc else {}
    _fig_all, _ = plots.plot_categorical_performance_all(
        _plot_df,
        f"GLM (tau={_val_tau})",
        **_perf_kwargs,
    )

    if not is_2afc:
        _fig_strat, _ = plots.plot_categorical_strat_by_side(_plot_df, subject="All", model_name="GLM")
    else:
        _fig_strat = None

    _row = [_fig_all] + ([_fig_strat] if _fig_strat is not None else [])
    mo.vstack([
        mo.md("### Model Performance"),
        mo.hstack(_row),
        fig_ag, fig_cls
    ])
    return (softmax,)


@app.cell
def _(mo):
    reset_button = mo.ui.run_button(label="Reset to A92 values")
    reset_button
    return (reset_button,)


@app.cell
def _(CoefTweakerWidget, arrays_store, mo, np, reset_button):
    _ = reset_button.value  # re-run this cell (resetting sliders) when button is clicked
    _params = arrays_store["A95"]
    _X_cols = _params["X_cols"]
    _W = _params["emission_weights"][0]  # shape (2, M): W_L at [0], W_R at [1]

    ui_coef_tweaker = mo.ui.anywidget(
        CoefTweakerWidget(
            features=list(_X_cols),
            w_L=list(np.round(np.array(_W[0, :], dtype=float), 2)),
            w_R=list(np.round(np.array(_W[1, :], dtype=float), 2)),
        )
    )

    ui_coef_tweaker
    return (ui_coef_tweaker,)


@app.cell
def _(
    adapter,
    arrays_store,
    df_all,
    is_2afc,
    mo,
    np,
    pl,
    plots,
    softmax,
    ui_coef_tweaker,
    ui_model_manager,
):
    _sort_col = adapter.sort_col
    _frames = []

    _params_a92 = arrays_store["A95"]
    _X_cols = _params_a92["X_cols"]
    _val_tau = ui_model_manager.value["tau"]

    # Build weight vectors directly from widget values
    _W_L = np.array(ui_coef_tweaker.value["w_L"])
    _W_R = np.array(ui_coef_tweaker.value["w_R"])

    _df_sub = (
        df_all.filter(pl.col("subject") == "A95")
        .sort(_sort_col)
    )
    y, X, _, _ = adapter.load_subject(_df_sub, tau=_val_tau, emission_cols=_X_cols)
    T, M = X.shape
    X_np = np.asarray(X, dtype=float)
    logits = np.stack([X_np @ _W_L, np.zeros(T), X_np @ _W_R], axis=1)
    p_pred = softmax(logits, axis=1)

    _df_sub = (
        df_all.filter(pl.col("subject") == "A95")
        .sort(_sort_col)
    )
    if not is_2afc:
        _df_sub = _df_sub.filter(pl.col("session").count().over("session") >= 2)

    _cols = [pl.Series("pred_choice", np.argmax(p_pred, axis=1).astype(int))]
    _cols += [pl.Series("pL", p_pred[:, 0]), pl.Series("pC", p_pred[:, 1]), pl.Series("pR", p_pred[:, 2])]

    _df_sub = _df_sub.with_columns(_cols)
    _frames.append(_df_sub)

    _df_all_pred = pl.concat(_frames)
    _plot_df = plots.prepare_predictions_df(_df_all_pred)
    _perf_kwargs = {"arrays_store": arrays_store} if is_2afc else {}
    _fig_all, _ = plots.plot_categorical_performance_all(
        _plot_df,
        f"GLM tweaked (tau={_val_tau})",
        **_perf_kwargs,
    )
    _fig_all_cat, _ = plots.plot_categorical_strat_by_side(
        _plot_df,
        f"GLM tweaked (tau={_val_tau})",
        **_perf_kwargs, model_name = ""
    )

    mo.vstack([
        mo.hstack([_fig_all, _fig_all_cat]),
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
