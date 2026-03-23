import marimo

__generated_with = "0.21.0"
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
    from tasks import get_adapter
    from widgets import ModelManagerWidget
    from figure_save_utils import make_plot_saver
    from coefficient_editor_widget import CoefficientEditorWidget
    from coefficient_editor_utils import (
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
    )

    sns.set_style("white")
    return (
        CoefficientEditorWidget,
        ModelManagerWidget,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
        fit_main,
        generate_model_id,
        get_adapter,
        make_plot_saver,
        mo,
        np,
        paths,
        pl,
        plt,
        sns,
    )


@app.cell
def _(get_adapter, paths, pl, ui_model_manager):
    task_name = ui_model_manager.value.get("task", "MCDR")
    adapter = get_adapter(task_name)
    df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df_all = adapter.subject_filter(df_all)
    plots = adapter.get_plots()
    return adapter, df_all, plots, task_name


@app.cell
def _(ModelManagerWidget, mo):
    mm_widget = ModelManagerWidget(
        model_type="glm",
        task="MCDR",
        tau=50,
        lapse=False,
        lapse_max=0.2,
    )
    ui_model_manager = mo.ui.anywidget(mm_widget)
    return (ui_model_manager,)


@app.cell
def _(df_all, mo, pl, plt, sns):
    import pandas as pd

    _required_cols = {
        "ttype_n",
        "stim_d",
        "timepoint_2",
        "timepoint_3",
        "stimd_n",
        "onset",
        "performance",
    }
    mo.stop(
        not _required_cols.issubset(set(df_all.columns)),
        mo.md("Task-specific MCDR onset diagnostic is unavailable for this task."),
    )

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
def _(adapter, ui_model_manager):
    class _V:
        def __init__(self, value):
            self.value = value

    _val = ui_model_manager.value
    is_2afc = adapter.num_classes == 2

    ui_existing = _V(None if _val.get("existing_model") in ("", "__default__") else _val.get("existing_model"))
    ui_alias = _V(_val.get("alias", ""))
    ui_subjects = _V(_val.get("subjects", []))
    ui_tau = _V(_val.get("tau", 5))
    ui_lapse = _V(_val.get("lapse", False))
    ui_lapse_max = _V(_val.get("lapse_max", 0.2))
    ui_emission_cols = _V(_val.get("emission_cols", []))
    return (
        is_2afc,
        ui_alias,
        ui_emission_cols,
        ui_existing,
        ui_lapse,
        ui_lapse_max,
        ui_subjects,
        ui_tau,
    )


@app.cell
def _(
    generate_model_id,
    mo,
    task_name,
    ui_emission_cols,
    ui_lapse,
    ui_model_manager,
    ui_tau,
):
    current_hash = generate_model_id(task_name, ui_tau.value, ui_emission_cols.value, lapse=ui_lapse.value)

    mo.vstack([
        mo.md("### GLM Configuration"),
        ui_model_manager,
        mo.md(f"**Current params hash:** `{current_hash}`"),
    ])
    return (current_hash,)


@app.cell
def _(
    fit_main,
    mo,
    task_name,
    ui_alias,
    ui_emission_cols,
    ui_lapse,
    ui_lapse_max,
    ui_model_manager,
    ui_subjects,
    ui_tau,
):
    _clicks = ui_model_manager.value.get("run_fit_clicks", 0)
    mo.stop(_clicks == 0, mo.md("Configure parameters and press **Run GLM Fit**."))

    with mo.status.spinner(title=f"Fitting GLM for {len(ui_subjects.value)} subjects..."):
        fit_main(
            subjects=ui_subjects.value,
            out_dir=None,
            tau=ui_tau.value,
            emission_cols=ui_emission_cols.value,
            task=task_name,
            model_alias=ui_alias.value if ui_alias.value else None,
            lapse=ui_lapse.value,
            lapse_max=ui_lapse_max.value,
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
    task_name,
    ui_alias,
    ui_emission_cols,
    ui_existing,
    ui_subjects,
    ui_tau,
):
    if ui_existing.value:
        selected_model_id = ui_existing.value
    elif ui_alias.value:
        selected_model_id = ui_alias.value
    else:
        selected_model_id = current_hash 

    OUT = paths.RESULTS / "fits" / task_name / "glm" / selected_model_id

    # Feature names from adapter (uniform for both tasks)
    _df_sel = df_all.filter(pl.col("subject").is_in(ui_subjects.value))
    if len(_df_sel) > 0:
        _df_sel = _df_sel.sort(adapter.sort_col)
        _, _, _, names = adapter.load_subject(
            _df_sel, tau=ui_tau.value, emission_cols=ui_emission_cols.value
        )
    else:
        names = {"X_cols": [], "U_cols": []}

    arrays_store = {}
    for _f in sorted(OUT.glob("*_glm_arrays.npz")):
        _subj = _f.name.removesuffix("_glm_arrays.npz")
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
    return arrays_store, selected_model_id


@app.cell
def _(make_plot_saver, mo, paths, selected_model_id, task_name):
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name=task_name,
        model_id=selected_model_id,
    )
    return (save_plot,)


@app.cell
def _(adapter, arrays_store, mo, ui_subjects):
    # ── Build SubjectFitViews + derive state_labels / state_order for backward compat ──
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))
    from glmhmmt.views import build_views
    from glmhmmt.postprocess import build_trial_df, build_emission_weights_df
    K = 1
    views = build_views(arrays_store, adapter, K, _selected)

    state_labels = {s: v.state_name_by_idx for s, v in views.items()}
    state_order  = {s: v.state_idx_order   for s, v in views.items()}
    return K, build_emission_weights_df, build_trial_df, build_views, views


@app.cell
def _(adapter, arrays_store, build_views):
    editor_views = build_views(arrays_store, adapter, 1, list(arrays_store.keys()))
    return (editor_views,)


@app.cell
def _(
    adapter,
    build_emission_weights_df,
    build_trial_df,
    df_all,
    mo,
    pl,
    views,
):
    # ── Build canonical trial-level DataFrame ────────────────────────────────────────────────────────
    # One row per trial per subject.  Columns include:
    #   p_state_k         → HMM posterior (direct copy of smoothed_probs[:, k])
    #   state_idx/rank/label → MAP state assignment
    #   pL / pC / pR      → marginal class probabilities from p_pred
    #   p_model_correct   → MAP-state emission P(correct class)
    #   p_model_correct_marginal → marginal P(correct class)
    #   correct_bool      → bool(performance)
    # All task-specific behavioral columns (stimd_n, ttype_n, …) are preserved.
    _sort_col = adapter.sort_col
    _ses_col  = adapter.session_col
    _bcols    = adapter.behavioral_cols
    _trial_frames = []
    for _subj, _view in views.items():
        _df_sub = (
            df_all
            .filter(pl.col("subject") == _subj)
            .sort(_sort_col)
            # .filter(pl.col(_ses_col).count().over(_ses_col) >= 2)
        )
        if _df_sub.height != _view.T:
            print(f"⚠️  {_subj}: row mismatch ({_df_sub.height} vs {_view.T}), skipping")
            continue
        _trial_frames.append(build_trial_df(_view, adapter, _df_sub, _bcols))

    mo.stop(not _trial_frames, mo.md("No subjects with matching data lengths."))
    trial_df = pl.concat(_trial_frames)

    # Emit emission-weights long DF for downstream use
    weights_df = build_emission_weights_df(views)
    return (trial_df,)


@app.cell
def _(K, arrays_store, mo, paths, plots, ui_subjects, views):
    # Plot Weights (Folded / Agonist)
    # GLM is essentially K=1.
    # State Labels Trivial

    if not arrays_store:
        mo.stop(True, mo.md("No results loaded."))
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    _save_path = paths.RESULTS / "plots/GLMHMM/emissions_coefs.png"
    _fig_ag, _fig_cls = plots.plot_emission_weights(
        views={s: views[s] for s in _selected}, K=K, save_path=_save_path,
    )
    mo.vstack([mo.md("### Emission weights"), _fig_ag, _fig_cls])
    return


@app.cell
def _(is_2afc, mo, pl, plots, save_plot, trial_df, ui_subjects, views):
    _selected = [s for s in ui_subjects.value if s in views]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _views_sel = {s: views[s] for s in _selected}
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(_selected))

    mo.stop(_trial_df_sel.height == 0, mo.md("No subjects with matching data lengths."))

    _plot_df_all = plots.prepare_predictions_df(_trial_df_sel)
    _perf_kwargs = {"views": _views_sel} if is_2afc else {}
    _fig_all, _ = plots.plot_categorical_performance_all(
        _plot_df_all,
        "glm",
        # background_style=ui_psychometric_background.value,
        **_perf_kwargs,
    )
    _plot_df_state = plots.prepare_predictions_df(_trial_df_sel)
    _fig_state, _ = plots.plot_categorical_performance_by_state(
        df=_plot_df_state,
        views=_views_sel,
        model_name="glm — per state",
        # background_style=ui_psychometric_background.value,
    )
    mo.vstack(
        [
            mo.md("### Categorical plots for accuracy"),
            mo.hstack(
                [
                    mo.vstack([_fig_all], align="center"),
                    mo.vstack(
                        [
                            save_plot(_fig_all, "overall psychometric", stem="categorical_overall"),
                        ],
                        align="start",
                    ),
                ],
                justify="space-between",
                align="center",
                widths=[4, 1],
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(editor_views, mo):
    subjects = list(editor_views.keys())
    mo.stop(not _subjects, mo.md("No fitted subjects available for coefficient editing."))
    ui_editor_subject = mo.ui.dropdown(
        options=_subjects,
        value=_subjects[0],
        label="Subject",
    )
    ui_editor_subject
    return (ui_editor_subject,)


@app.cell
def _(editor_views, mo, ui_editor_subject):
    _view = editor_views[ui_editor_subject.value]
    _state_options = [
        f"{_k} — {_view.state_name_by_idx.get(_k, f'State {_k}')}"
        for _k in _view.state_idx_order
    ]
    ui_editor_state = mo.ui.dropdown(
        options=_state_options,
        value=_state_options[0],
        label="State",
    )
    ui_editor_state
    return (ui_editor_state,)


@app.cell
def _(adapter, mo):
    if adapter.num_classes != 2:
        ui_editor_side = None
    else:
        _choices = [str(label) for label in adapter.choice_labels]
        ui_editor_side = mo.ui.dropdown(
            options=_choices,
            value=_choices[0],
            label="Side",
        )
        ui_editor_side
    return (ui_editor_side,)


@app.cell
def _(
    CoefficientEditorWidget,
    adapter,
    build_editor_payload,
    editor_views,
    mo,
    np,
    ui_editor_side,
    ui_editor_state,
    ui_editor_subject,
):
    _subj = ui_editor_subject.value
    _view = editor_views[_subj]
    coef_state_idx = int(ui_editor_state.value.split(" — ", 1)[0])
    coef_state_label = _view.state_name_by_idx.get(
        coef_state_idx, f"State {coef_state_idx}"
    )
    _stored_weights = np.asarray(_view.emission_weights[coef_state_idx], dtype=float)
    _choice_labels = [str(label) for label in adapter.choice_labels]
    _stored_class_indices = [0] if _view.num_classes == 2 else [0, 2]
    _reference_class_idx = 1 if _view.num_classes > 2 else (_view.num_classes - 1)
    if _view.num_classes == 2 and ui_editor_side is not None:
        _display_class_idx = _choice_labels.index(ui_editor_side.value)
        _display_reference_class_idx = next(
            idx for idx in range(_view.num_classes) if idx != _display_class_idx
        )
    else:
        _display_reference_class_idx = None
    _payload = build_editor_payload(
        _stored_weights,
        choice_labels=_choice_labels,
        stored_class_indices=_stored_class_indices,
        reference_class_idx=_reference_class_idx,
        display_reference_class_idx=_display_reference_class_idx,
    )

    coef_editor = mo.ui.anywidget(
        CoefficientEditorWidget(
            title="Coefficient editor",
            subtitle=_payload["subtitle"],
            features=list(_view.feat_names),
            channel_labels=_payload["channel_labels"],
            weights=_payload["weights"].tolist(),
            original_weights=_payload["weights"].tolist(),
            slider_min=-6.0,
            slider_max=6.0,
            slider_step=0.05,
        )
    )
    _controls = [ui_editor_subject, ui_editor_state]
    if ui_editor_side is not None:
        _controls.append(ui_editor_side)

    coef_editor_panel = mo.vstack(
        [
            mo.md("### Interactive coefficient editor"),
            mo.md(
                "The edited state is recomputed live and the categorical plots "
                "below use the updated probabilities."
            ),
            mo.hstack(_controls),
            coef_editor,
        ],
        align="center",
    )
    coef_editor_panel
    coef_editor_explicit_class_indices = _payload["explicit_class_indices"]
    coef_editor_reference_class_idx = _payload["reference_class_idx"]
    coef_editor_stored_class_indices = _payload["stored_class_indices"]
    coef_editor_stored_reference_class_idx = _payload["stored_reference_class_idx"]
    return (
        coef_editor,
        coef_editor_explicit_class_indices,
        coef_editor_reference_class_idx,
        coef_editor_stored_class_indices,
        coef_editor_stored_reference_class_idx,
        coef_state_idx,
        coef_state_label,
    )


@app.cell
def _(
    adapter,
    build_trial_df,
    df_all,
    editor_views,
    mo,
    pl,
    ui_editor_subject,
):
    _subj = ui_editor_subject.value
    _view = editor_views[_subj]
    _sort_col = adapter.sort_col
    _ses_col = adapter.session_col
    _bcols = adapter.behavioral_cols
    _df_sub = (
        df_all
        .filter(pl.col("subject") == _subj)
        .sort(_sort_col)
        .filter(pl.col(_ses_col).count().over(_ses_col) >= 2)
    )
    mo.stop(_df_sub.height != _view.T, mo.md(f"Subject {_subj} does not match the loaded fit arrays."))
    editor_trial_df = build_trial_df(_view, adapter, _df_sub, _bcols)
    editor_view = _view
    return editor_trial_df, editor_view


@app.cell
def _(
    adapter,
    apply_state_tweak_to_trial_df,
    apply_state_tweak_to_view,
    coef_editor,
    coef_editor_explicit_class_indices,
    coef_editor_reference_class_idx,
    coef_editor_stored_class_indices,
    coef_editor_stored_reference_class_idx,
    coef_state_idx,
    coef_state_label,
    editor_trial_df,
    editor_view,
    mo,
    np,
    plots,
    save_plot,
    ui_editor_subject,
    ui_psychometric_background,
):
    _subj = ui_editor_subject.value
    _view = editor_view
    _trial_df_sub = editor_trial_df
    _edited_weights = np.asarray(coef_editor.value["weights"], dtype=float)

    _trial_df_tweaked = apply_state_tweak_to_trial_df(
        _trial_df_sub,
        adapter=adapter,
        view=_view,
        state_idx=coef_state_idx,
        edited_weights=_edited_weights,
        original_weights=np.asarray(coef_editor.value["original_weights"], dtype=float),
        explicit_class_indices=list(coef_editor_explicit_class_indices),
        reference_class_idx=int(coef_editor_reference_class_idx),
    )
    _view_tweaked = apply_state_tweak_to_view(
        _view,
        state_idx=coef_state_idx,
        edited_weights=_edited_weights,
        explicit_class_indices=list(coef_editor_explicit_class_indices),
        reference_class_idx=int(coef_editor_reference_class_idx),
        stored_class_indices=list(coef_editor_stored_class_indices),
        stored_reference_class_idx=int(coef_editor_stored_reference_class_idx),
    )
    _plot_df_tweaked = plots.prepare_predictions_df(_trial_df_tweaked)

    _title = f"{_subj} — tweaked {coef_state_label}"
    _fig_all_tweaked, _ = plots.plot_categorical_performance_all(
        _plot_df_tweaked,
        _title,
        background_style=ui_psychometric_background.value,
    )
    _fig_state_tweaked, _ = plots.plot_categorical_performance_by_state(
        df=_plot_df_tweaked,
        views={_subj: _view_tweaked},
        model_name=f"{_title} — per state",
        background_style=ui_psychometric_background.value,
    )
    _side_plot_fn = getattr(plots, "plot_categorical_strat_by_side", None)
    if _side_plot_fn is None:
        _side_section = mo.md("This task does not expose a side-stratified categorical plot.")
    else:
        _fig_side_tweaked, _ = plots.plot_categorical_strat_by_side(
            _plot_df_tweaked,
            subject=_subj,
            model_name=f"{_subj}_tweaked_{coef_state_idx}",
        )
        _side_section = mo.vstack(
            [
                mo.md("### Tweaked categorical performance by stimulus side"),
                _fig_side_tweaked,
            ]
        )

    mo.vstack(
        [
            mo.md("### Tweaked categorical plots"),
            mo.hstack(
                [
                    mo.vstack([_fig_all_tweaked], align="center"),
                    mo.vstack(
                        [
                            ui_psychometric_background,
                            save_plot(
                                _fig_all_tweaked,
                                "tweaked overall psychometric",
                                stem="tweaked_categorical_overall",
                            ),
                        ],
                        align="start",
                    ),
                ],
                justify="space-between",
                align="center",
                widths=[4,1],
            ),
            mo.md("### Tweaked per-state categorical performance"),
            mo.hstack(
                [
                    mo.vstack([_fig_state_tweaked], align="center"),
                    save_plot(
                        _fig_state_tweaked,
                        "tweaked per-state psychometric",
                        stem="tweaked_categorical_by_state",
                    ),
                ],
                justify="space-between",
                align="center",
                widths=[4, 1],
            ),
            _side_section,
        ],
        align="center",
    )
    return


if __name__ == "__main__":
    app.run()
