import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from glmhmmt.notebook_support import (
        CoefficientEditorWidget,
        ModelManagerWidget,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
        make_plot_saver,
        model_cfg as ModelCfg,
    )
    from glmhmmt.notebook_support.analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
        resolve_selected_model_id,
        select_subject_behavior_df,
    )
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from glmhmmt.runtime import get_runtime_paths
    try:
        from glmhmmt.cli.fit_glmhmmt import main as fit_main
        _FITTING_AVAILABLE = True
    except ImportError:
        fit_main = None
        _FITTING_AVAILABLE = False
    paths = get_runtime_paths()
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from glmhmmt.postprocess import build_trial_df, build_emission_weights_df
    sns.set_style("white")
    return (
        CoefficientEditorWidget,
        ModelCfg,
        ModelManagerWidget,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
        build_trial_and_weights_df,
        build_trial_df,
        build_views,
        fit_main,
        get_adapter,
        load_fit_arrays,
        make_plot_saver,
        mo,
        np,
        paths,
        pd,
        pl,
        plt,
        resolve_selected_model_id,
        select_subject_behavior_df,
        sns,
    )


@app.cell
def _(get_adapter, model_cfg, paths, pl):
    task_name = model_cfg.task
    adapter = get_adapter(task_name)
    df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df_all = adapter.subject_filter(df_all)
    is_2afc = adapter.num_classes == 2
    plots = adapter.get_plots()
    return adapter, df_all, is_2afc, plots, task_name


@app.cell
def _(ModelManagerWidget, mo):
    mm_widget = ModelManagerWidget(
        model_type="glmhmmt",
        task="MCDR",
        K=2,
        tau=50,
    )
    ui_model_manager = mo.ui.anywidget(mm_widget)
    return mm_widget, ui_model_manager


@app.cell
def _(ModelCfg, ui_model_manager):
    model_cfg = ModelCfg.from_value(ui_model_manager.value)
    return (model_cfg,)


@app.cell
def _(mo):
    get_last_fit_click, set_last_fit_click = mo.state(0)
    return get_last_fit_click, set_last_fit_click


@app.cell
def _(model_cfg, task_name):
    from glmhmmt.cli.fit_glmhmmt import generate_model_id as _gen_id

    current_hash = _gen_id(
        task_name,
        model_cfg.K,
        model_cfg.tau,
        model_cfg.emission_cols,
        model_cfg.transition_cols,
        model_cfg.frozen_emissions,
        model_cfg.cv_mode,
        model_cfg.cv_repeats,
    )
    return (current_hash,)


@app.cell
def _(current_hash, make_plot_saver, model_cfg, mo, paths, resolve_selected_model_id, task_name):
    selected_model_id = resolve_selected_model_id(
        current_hash,
        model_cfg.existing,
        model_cfg.alias,
    )
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name=task_name,
        model_id=selected_model_id,
    )
    return save_plot, selected_model_id


@app.cell
def _(current_hash, mo, save_plot, ui_model_manager):
    mo.vstack([
        mo.md("### Model Configuration"),
        ui_model_manager,
        save_plot.save_all_widget(label="Save all model plots"),
        mo.md(f"**Hash:** `{current_hash}`"),
    ])
    return


@app.cell
def _(
    current_hash,
    fit_main,
    get_last_fit_click,
    model_cfg,
    mm_widget,
    mo,
    paths,
    set_last_fit_click,
    task_name,
):
    _last_fit_click = get_last_fit_click()
    mo.stop(
        model_cfg.run_fit_clicks <= _last_fit_click,
        mo.md("Configure parameters and press **Run fit**."),
    )
    set_last_fit_click(model_cfg.run_fit_clicks)

    _n_restarts = 1
    _cv_repeats = int(model_cfg.cv_repeats) if model_cfg.cv_mode != "none" else 0
    _selected_id = model_cfg.existing or (model_cfg.alias if model_cfg.alias else current_hash)
    _OUT = paths.RESULTS / "fits" / task_name / "glmhmmt" / _selected_id

    def _progress_title(info: dict) -> str:
        return (
            f"Fitting GLM-HMM-T K={info['K']} "
            f"subject {info['subject_index']}/{info['subject_total']}: {info['subject']}"
        )

    def _progress_subtitle(info: dict) -> str:
        _base = f"Restart {info['restart_index']}/{info['restart_total']}"
        if info.get("event") == "restart_complete":
            return f"{_base} complete"
        return _base

    _total_progress = max(
        1,
        len(model_cfg.subjects) * (_cv_repeats if model_cfg.cv_mode != "none" else _n_restarts),
    )
    mm_widget.is_running = True
    try:
        with mo.status.progress_bar(
            total=_total_progress,
            title=f"Fitting GLM-HMM-T K={model_cfg.K}",
            subtitle=(
                f"{len(model_cfg.subjects)} subjects × {_cv_repeats} CV repeat(s)"
                if model_cfg.cv_mode != "none"
                else f"{len(model_cfg.subjects)} subjects × {_n_restarts} restart(s)"
            ),
            completion_title="Fit complete",
            completion_subtitle=f"Saved under {_selected_id}",
        ) as _bar:
            def _on_progress(info: dict) -> None:
                if info.get("event") == "cv_repeat_start":
                    _bar.update(
                        increment=0,
                        title=_progress_title(info),
                        subtitle=f"CV repeat {info['cv_repeat_index']}/{info['cv_repeat_total']}",
                    )
                    return
                if info.get("event") == "cv_repeat_complete":
                    _bar.update(
                        increment=1,
                        title=_progress_title(info),
                        subtitle=f"CV repeat {info['cv_repeat_index']}/{info['cv_repeat_total']} complete",
                    )
                    return
                if info.get("event") == "restart_start":
                    _bar.update(
                        increment=0,
                        title=_progress_title(info),
                        subtitle=_progress_subtitle(info),
                    )
                    return
                if info.get("event") == "restart_complete":
                    _bar.update(
                        increment=0 if model_cfg.cv_mode != "none" else 1,
                        title=_progress_title(info),
                        subtitle=_progress_subtitle(info),
                    )

            fit_main(
                subjects=model_cfg.subjects,
                K_list=[model_cfg.K],
                out_dir=_OUT,
                emission_cols=model_cfg.emission_cols or None,
                transition_cols=model_cfg.transition_cols or None,
                frozen_emissions=model_cfg.frozen_emissions or None,
                tau=model_cfg.tau,
                task=task_name,
                cv_mode=model_cfg.cv_mode,
                cv_repeats=_cv_repeats,
                n_restarts=_n_restarts,
                verbose=False,
                progress_callback=_on_progress,
            )
        mm_widget.saved_model_name = _selected_id
        mm_widget.alias_error = ""
        mm_widget.alias_status = ""
        if not model_cfg.alias:
            mm_widget.alias = _selected_id
        mm_widget._update_options()
        if _selected_id in mm_widget.existing_models:
            mm_widget.existing_model = _selected_id
    finally:
        mm_widget.is_running = False
    mo.md("✅ Fit complete — plots below update automatically.")
    return


@app.cell
def _(mo, model_cfg, paths, pd, pl, plt, selected_model_id, sns, task_name):
    import json

    mo.stop(model_cfg.cv_mode == "none", mo.md(""))
    out_dir = paths.RESULTS / "fits" / task_name / "glmhmmt" / selected_model_id
    repeat_files = sorted(out_dir.glob("*_cv_repeats.parquet"))
    mo.stop(not repeat_files, mo.md("No CV repeat diagnostics found yet."))

    repeats_df = pl.concat([pl.read_parquet(path) for path in repeat_files], how="diagonal_relaxed")
    repeats_pd = repeats_df.to_pandas()

    count_rows = []
    for row in repeats_pd.to_dict(orient="records"):
        for split_name, counts_key in (
            ("train", "train_label_counts_json"),
            ("test", "test_label_counts_json"),
        ):
            counts = json.loads(row.get(counts_key) or "{}")
            for ild, count in counts.items():
                count_rows.append(
                    {
                        "subject": row["subject"],
                        "repeat_index": int(row["repeat_index"]),
                        "subject_repeat": f"{row['subject']} r{int(row['repeat_index'])}",
                        "split": split_name,
                        "ild": float(ild),
                        "count": int(count),
                    }
                )

    count_pd = pd.DataFrame(count_rows)
    if not count_pd.empty:
        count_pd = count_pd.sort_values(["ild", "subject", "repeat_index"])

    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.55 * max(len(repeats_pd), 1)), 12))

    sns.lineplot(
        data=repeats_pd,
        x="repeat_index",
        y="test_ll_per_trial",
        hue="subject",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("CV Test Log-Likelihood by Repeat")
    axes[0].set_xlabel("Repeat")
    axes[0].set_ylabel("Test LL / trial")

    for ax_idx, split_name in enumerate(["train", "test"], start=1):
        split_pd = count_pd[count_pd["split"] == split_name]
        if split_pd.empty:
            axes[ax_idx].set_visible(False)
            continue
        pivot = (
            split_pd.pivot_table(
                index="ild",
                columns="subject_repeat",
                values="count",
                fill_value=0,
            )
            .sort_index()
        )
        sns.heatmap(pivot, cmap="Blues", cbar=True, ax=axes[ax_idx])
        axes[ax_idx].set_title(f"Signed ILD Counts in {split_name.capitalize()} Split")
        axes[ax_idx].set_xlabel("Subject / Repeat")
        axes[ax_idx].set_ylabel("Signed ILD")

    plt.tight_layout()

    summary_cols = [
        "subject",
        "repeat_index",
        "balance_score",
        "train_session_count",
        "test_session_count",
        "test_ll_per_trial",
        "test_acc",
    ]
    return


@app.cell
def _(
    adapter,
    df_all,
    load_fit_arrays,
    model_cfg,
    paths,
    task_name,
    selected_model_id,
):
    K = model_cfg.K
    OUT = paths.RESULTS / "fits" / task_name / "glmhmmt" / selected_model_id
    arrays_store, names = load_fit_arrays(
        out_dir=OUT,
        arrays_suffix="glmhmmt_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=list(model_cfg.subjects),
        emission_cols=list(model_cfg.emission_cols),
        transition_cols=list(model_cfg.transition_cols),
        k=K,
    )
    _ = names
    return K, arrays_store


@app.cell
def _(adapter, mo):
    _opts = list(adapter._SCORING_OPTIONS.keys()) if hasattr(adapter, "_SCORING_OPTIONS") else ["default"]
    _default_key = getattr(adapter, "scoring_key", _opts[0])
    if _default_key not in _opts:
        _default_key = _opts[0]
    ui_scoring_key = mo.ui.dropdown(
        options=_opts,
        value=_default_key,
        label="State scoring regressor (Engaged = highest score)",
    )
    mo.vstack([mo.md("### State labelling regressor"), ui_scoring_key])
    return (ui_scoring_key,)


@app.cell
def _(K, adapter, arrays_store, build_views, mo, model_cfg, ui_scoring_key):
    selected = [s for s in model_cfg.subjects if s in arrays_store]
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))

    if hasattr(adapter, "scoring_key"):
        adapter.scoring_key = ui_scoring_key.value
    views = build_views(arrays_store, adapter, K, selected)
    state_labels = {s: v.state_name_by_idx for s, v in views.items()}
    return selected, state_labels, views


@app.cell
def _(K, adapter, arrays_store, build_views, ui_scoring_key):
    if hasattr(adapter, "scoring_key"):
        adapter.scoring_key = ui_scoring_key.value
    editor_views = build_views(arrays_store, adapter, K, list(arrays_store.keys()))
    return (editor_views,)


@app.cell
def _(adapter, build_trial_and_weights_df, df_all, mo, views):
    trial_df, weights_df = build_trial_and_weights_df(
        df_all,
        views=views,
        adapter=adapter,
        min_session_length=2,
    )
    mo.stop(trial_df.height == 0, mo.md("No subjects with matching data lengths."))
    return (trial_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("### Emission weights").center()
    return


@app.cell
def _(K, mo, paths, plots, save_plot, selected, views):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    _save_path = paths.RESULTS / "plots/GLMHMMT/emissions_coefs.png"
    _views_sel = {s: views[s] for s in selected}
    _fig_by_subject = plots.plot_emission_weights_by_subject(
        views=_views_sel,
        K=K,
        save_path=_save_path,
    )

    _subject_figs, _summary_figs = plots.plot_emission_weights(views=_views_sel, K=K)

    mo.vstack([
               # _subject_figs,
               _summary_figs,
               mo.hstack([save_plot(_summary_figs, f"Emission Weights lineplot",
                                    stem=f"emissions_lineplot",), 
                          save_plot(_summary_figs, f"Emission Weights boxplot",
                                    stem=f"emissions_boxplot",),
             ], gap = "15"), ], align="center")
    return


@app.cell
def _(K, mo, plots, selected, views):
    mo.stop(
        not selected or getattr(views.get(selected[0]), "transition_weights", None) is None,
        mo.md("No transition weights found — run the glmhmm-t fit first."),
    )
    _fig_line, _fig_box = plots.plot_transition_weights(views=views, K=K, subjects=selected)
    mo.vstack([
        mo.md("### Transition weights"),
        mo.hstack([_fig_line, _fig_box]),
    ])
    return


@app.cell
def _(K, arrays_store, mo, plots, selected, state_labels):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    _fig_by_subject = plots.plot_transition_matrix_by_subject(
        arrays_store=arrays_store,
        state_labels=state_labels,
        K=K,
        subjects=selected,
    )
    _fig_summary = plots.plot_transition_matrix(
        arrays_store=arrays_store,
        state_labels=state_labels,
        K=K,
        subjects=selected,
    )
    mo.vstack([
        mo.md(f"### Transition matrices — bias-only component  (K={K})"),
        mo.md("#### By subject"),
        # _fig_by_subject,
        mo.md("#### Summary"),
        _fig_summary,
    ])
    return


@app.cell
def _(mo):
    ui_psychometric_background = mo.ui.radio(
        options={
            "Data traces": "data",
            "Model curves": "model",
            "None": "none",
        },
        value="Data traces",
        inline=False,
        label="Psychometric background",
    )
    ui_psychometric_background
    return (ui_psychometric_background,)


@app.cell
def _(mo):
    ui_state_show_weighted_points = mo.ui.checkbox(value=True, label="Weighted dots")
    ui_state_show_data_smooth = mo.ui.checkbox(value=True, label="Data smooth")
    ui_state_assignment_mode = mo.ui.radio(
        options={
            "Predictive weights": "weighted",
            "MAP state": "map",
        },
        value="Predictive weights",
        inline=False,
        label="State assignment",
    )
    ui_state_model_line_mode = mo.ui.radio(
        options={
            "Smooth curve": "smooth",
            "Trial-matched line": "trial_matched",
            "None": "none",
        },
        value="Smooth curve",
        inline=False,
        label="Model line",
    )
    return (
        ui_state_assignment_mode,
        ui_state_model_line_mode,
        ui_state_show_data_smooth,
        ui_state_show_weighted_points,
    )


@app.cell
def _(is_2afc, mo, views):
    _feature_names = []
    if is_2afc and views:
        for _view in views.values():
            for _feat in list(getattr(_view, "feat_names", []) or []):
                if _feat not in _feature_names:
                    _feature_names.append(_feat)
    if not _feature_names:
        _feature_names = ["at_choice"]
    _default_feature = "at_choice" if "at_choice" in _feature_names else _feature_names[0]
    ui_psychometric_regressor = mo.ui.dropdown(
        options=_feature_names,
        value=_default_feature,
        label="Regressor",
    )
    ui_psychometric_regressor
    return (ui_psychometric_regressor,)


@app.cell
def _(
    K,
    is_2afc,
    model_cfg,
    mo,
    pl,
    plots,
    save_plot,
    trial_df,
    ui_psychometric_background,
    ui_psychometric_regressor,
    ui_state_assignment_mode,
    ui_state_model_line_mode,
    ui_state_show_data_smooth,
    ui_state_show_weighted_points,
    views,
):
    _selected = [s for s in model_cfg.subjects if s in views]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _views_sel = {s: views[s] for s in _selected}
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(_selected))

    mo.stop(_trial_df_sel.height == 0, mo.md("No subjects with matching data lengths."))

    _plot_df_all = plots.prepare_predictions_df(_trial_df_sel)
    _perf_kwargs = {"views": _views_sel} if is_2afc else {}
    _fig_all, _ = plots.plot_categorical_performance_all(
        _plot_df_all,
        f"glmhmmt K={K}",
        background_style=ui_psychometric_background.value,
        **_perf_kwargs,
    )

    _plot_df_state = plots.prepare_predictions_df(_trial_df_sel)
    _fig_state, _ = plots.plot_categorical_performance_by_state(
        df=_plot_df_state,
        views=_views_sel,
        model_name=f"glmhmmt K={K} — per state",
        background_style=ui_psychometric_background.value,
        show_weighted_points=ui_state_show_weighted_points.value,
        show_data_smooth=ui_state_show_data_smooth.value,
        show_model_smooth=ui_state_model_line_mode.value != "none",
        model_line_mode=ui_state_model_line_mode.value,
        state_assignment_mode=ui_state_assignment_mode.value,
        figure_dpi=80,
    )
    _reg_plot_fn = getattr(plots, "plot_regressor_psychometric_by_state", None)
    if is_2afc and _reg_plot_fn is not None:
        _fig_reg_state, _ = _reg_plot_fn(
            df=_plot_df_state,
            views=_views_sel,
            model_name=f"glmhmmt K={K}",
            feature_col=ui_psychometric_regressor.value,
            background_style=ui_psychometric_background.value,
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
        )
        _reg_section = mo.vstack([
            mo.hstack([mo.md("### Per-state psychometric by regressor"), ui_psychometric_regressor], justify="space-between"),
            mo.vstack(
                [
                    _fig_reg_state,
                    save_plot(
                        _fig_reg_state,
                        f"{ui_psychometric_regressor.value} by state",
                        stem=f"regressor_by_state_{ui_psychometric_regressor.value}",
                    ),
                ],
                align="center",
            ),
        ], align="center")
    else:
        _reg_section = mo.md("This task does not expose a regressor psychometric plot.")

    mo.vstack([
        mo.md("### Categorical plots for accuracy"),
        mo.hstack(
            [
                mo.vstack(
                    [
                        _fig_all,
                        save_plot(_fig_all, "overall psychometric", stem="categorical_overall"),
                    ],
                    align="center",
                ),
                mo.vstack(
                    [
                        ui_psychometric_background,
                        ui_state_show_weighted_points,
                        ui_state_show_data_smooth,
                        ui_state_assignment_mode,
                        ui_state_model_line_mode,
                    ],
                    align="start",
                ),
            ],
            justify="space-between",
            align="center",
            widths=[4,1],
        ),
        mo.md("### Per-state categorical performance"),
        mo.vstack(
            [
                _fig_state,
                save_plot(_fig_state, "per-state psychometric", stem="categorical_by_state"),
            ],
            align="center",
        ),
        _reg_section,
    ], align="center")
    return


@app.cell
def _(editor_views, mo):
    _subjects = sorted(editor_views.keys(), key=str)
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
    _stored_class_indices = list(range(_view.num_classes - 1))
    _reference_class_idx = _view.num_classes - 1
    if _view.num_classes == 2 and ui_editor_side is not None:
        _display_class_idx = _choice_labels.index(ui_editor_side.value)
        _display_reference_class_idx = next(
            idx for idx in range(_view.num_classes) if idx != _display_class_idx
        )
    else:
        _display_reference_class_idx = 1 if _view.num_classes == 3 else _reference_class_idx
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
                "Only the selected state's emission coefficients are edited. "
                "The categorical plots below update with the edited state."
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
    select_subject_behavior_df,
    ui_editor_subject,
):
    _subj = ui_editor_subject.value
    _view = editor_views[_subj]

    _df_sub = select_subject_behavior_df(
        df_all,
        subject=_subj,
        sort_col=adapter.sort_col,
        session_col=adapter.session_col,
        min_session_length=2,
    )
    mo.stop(_df_sub.height != _view.T, mo.md(f"Subject {_subj} does not match the loaded fit arrays."))
    editor_trial_df = build_trial_df(_view, adapter, _df_sub, adapter.behavioral_cols)
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
    ui_psychometric_regressor,
    ui_state_assignment_mode,
    ui_state_model_line_mode,
    ui_state_show_data_smooth,
    ui_state_show_weighted_points,
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
        show_weighted_points=ui_state_show_weighted_points.value,
        show_data_smooth=ui_state_show_data_smooth.value,
        show_model_smooth=ui_state_model_line_mode.value != "none",
        model_line_mode=ui_state_model_line_mode.value,
        state_assignment_mode=ui_state_assignment_mode.value,
        figure_dpi=80,
    )
    _reg_plot_fn = getattr(plots, "plot_regressor_psychometric_by_state", None)
    if _reg_plot_fn is None:
        _reg_section = mo.md("This task does not expose a regressor psychometric plot.")
    else:
        _fig_reg_state_tweaked, _ = _reg_plot_fn(
            df=_plot_df_tweaked,
            views={_subj: _view_tweaked},
            model_name=_title,
            feature_col=ui_psychometric_regressor.value,
            background_style=ui_psychometric_background.value,
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
        )
        _reg_section = mo.vstack(
            [
                mo.hstack([mo.md("### Tweaked per-state psychometric by regressor"), ui_psychometric_regressor], justify="space-between"),
                mo.vstack(
                    [
                        _fig_reg_state_tweaked,
                        save_plot(
                            _fig_reg_state_tweaked,
                            f"tweaked {ui_psychometric_regressor.value} by state",
                            stem=f"tweaked_regressor_by_state_{ui_psychometric_regressor.value}",
                        ),
                    ],
                    align="center",
                ),
            ],
            align="center",
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
                    mo.vstack(
                        [
                            _fig_all_tweaked,
                            save_plot(
                                _fig_all_tweaked,
                                "tweaked overall psychometric",
                                stem="tweaked_categorical_overall",
                            ),
                        ],
                        align="center",
                    ),
                    mo.vstack(
                        [
                            ui_psychometric_background,
                            ui_state_show_weighted_points,
                            ui_state_show_data_smooth,
                            ui_state_assignment_mode,
                            ui_state_model_line_mode,
                        ],
                        align="start",
                    ),
                ],
                justify="space-between",
                align="center",
                widths=[4,1],
            ),
            mo.md("### Tweaked per-state categorical performance"),
            mo.vstack(
                [
                    _fig_state_tweaked,
                    save_plot(
                        _fig_state_tweaked,
                        "tweaked per-state psychometric",
                        stem="tweaked_categorical_by_state",
                    ),
                ],
                align="center",
            ),
            _reg_section,
            _side_section,
        ],
        align="center",
    )
    return


@app.cell
def _(mo):
    from wigglystuff import TangleSlider

    THRESH_ui = mo.ui.anywidget(
        TangleSlider(
            amount=0.9,
            min_value=0.0,
            max_value=1,
            step=0.01,
            digits=2,
        )
    )
    return (THRESH_ui,)


@app.cell
def _(THRESH_ui, adapter, mo, model_cfg, plots, trial_df, views):
    _selected = [s for s in model_cfg.subjects if s in views]
    mo.stop(not _selected, mo.md("No fitted subjects available."))
    _fig_acc, _tbl = plots.plot_state_accuracy(
        views={s: views[s] for s in _selected},
        trial_df=trial_df,
        thresh=THRESH_ui.amount,
        session_col=adapter.session_col,
        sort_col=adapter.sort_col,
    )
    _fig_post = plots.plot_state_posterior_count_kde(
        views={s: views[s] for s in _selected},
        thresh=THRESH_ui.amount,
    )
    mo.vstack([
        mo.hstack([
            mo.vstack([
                mo.md("### Accuracy by state"),
                _fig_acc,
            ]),
            mo.vstack([
                mo.md("### Posterior histogram by state"),
                _fig_post,
            ]),
        ], align="start"),
        mo.md(
            f"> **Accuracy**: **All** = full nonzero-stim pool · **State bars** = subsets where posterior ≥ {THRESH_ui}. "
            f"**Histogram**: pooled posterior percentages by state from `views`, using 0.05-wide posterior bins and the same threshold marked by the dashed line."
        ),
        mo.md("**Trial counts & mean accuracy per label:**"),
        mo.plain_text(_tbl.to_string()),
    ])
    return


@app.cell
def _(df_all, mo):
    # ── controls for session-trajectory & occupancy plots ─────────────────────
    ui_subjects_traj = mo.ui.multiselect(
        options=sorted(df_all["subject"].unique().to_list(), key=str),
        label="Subjects (session trajectories & occupancy)",
    )
    mo.vstack([mo.md("### Session trajectory & occupancy"), ui_subjects_traj])
    return (ui_subjects_traj,)


@app.cell
def _(K, mo, plots, trial_df, ui_subjects_traj, views):
    # ── c. Average state-probability trajectories within a session ────────────
    _selected_traj = [s for s in ui_subjects_traj.value if s in views]
    mo.stop(not _selected_traj, mo.md("Select subjects above to view session trajectories."))
    _fig_traj = plots.plot_session_trajectories(
        views={s: views[s] for s in _selected_traj},
        trial_df=trial_df,
        session_col="session",
        sort_col="trial_idx",
    )
    mo.vstack([
        mo.md(f"### c. Average state-probability trajectories within a session  (K={K})"),
        mo.md("> Mean ± 1 s.e.m. across sessions for the selected subjects."),
        _fig_traj,
    ], align="center")
    return


@app.cell
def _(K, THRESH_ui, mo, plots, trial_df, ui_subjects_traj, views):
    # ── d. Fractional occupancy & state-change histogram ─────────────────────
    _selected_occ = [s for s in ui_subjects_traj.value if s in views]
    mo.stop(not _selected_occ, mo.md("Select subjects above."))
    _fig_occ = plots.plot_state_occupancy(
        views={s: views[s] for s in _selected_occ},
        trial_df=trial_df,
        session_col="session",
        sort_col="trial_idx",
        switch_posterior_threshold=THRESH_ui.amount,
    )
    mo.vstack([
        mo.md(f"### d. Fractional occupancy & state changes per session  (K={K})"),
        mo.md(
            "> **Top row**: all selected subjects pooled. Left = posterior fractional occupancy boxplot by state; "
            "middle = per-session occupancy pooled across subjects; right = histogram of state switches per session.  \n"
            "> **Rows below**: one row per subject. Left = posterior mean occupancy by state; middle = per-session "
            "occupancy boxplots; right = histogram of inferred state switches per session."
        ),
        mo.md(
            f"> Switch counts use the same posterior threshold slider as the state-accuracy panel and only count "
            f"changes between confident MAP assignments with posterior ≥ {THRESH_ui.amount:.2f}."
        ),
        _fig_occ,
    ], align="center")
    return


@app.cell
def _(K, THRESH_ui, mo, plots, trial_df, ui_subjects_traj, views):
    _selected_change = [s for s in ui_subjects_traj.value if s in views]
    mo.stop(not _selected_change, mo.md("Select subjects above."))
    _views_sel = {s: views[s] for s in _selected_change}
    _fig_change_summary = plots.plot_change_triggered_posteriors_summary(
        views=_views_sel,
        trial_df=trial_df,
        session_col="session",
        sort_col="trial_idx",
        switch_posterior_threshold=THRESH_ui.amount,
    )
    _fig_change_by_subject = plots.plot_change_triggered_posteriors_by_subject(
        views=_views_sel,
        trial_df=trial_df,
        session_col="session",
        sort_col="trial_idx",
        switch_posterior_threshold=THRESH_ui.amount,
    )
    mo.vstack([
        mo.md(f"### e. Change-triggered posteriors  (K={K})"),
        mo.md(
            f"> Change events use the same confident MAP switch rule as the histogram above: posterior ≥ {THRESH_ui.amount:.2f}. "
            "> Trial 0 is the later confident trial in each detected change, and the mean traces are split into non-engaged -> engaged versus engaged -> non-engaged."
        ),
        _fig_change_summary,
        _fig_change_by_subject,
    ], align="center")
    return


@app.cell
def _(mo, model_cfg, views):
    # ── Session deep-dive controls ─────────────────────────────────────────────
    _selected = sorted((s for s in model_cfg.subjects if s in views), key=str)
    _subj_opts = _selected if _selected else ["(no fitted subjects)"]

    ui_session_subj = mo.ui.dropdown(
        options=_subj_opts,
        value=_subj_opts[0],
        label="Subject",
    )
    return (ui_session_subj,)


@app.cell
def _(mo, pl, trial_df, ui_session_subj, views):
    _sess_opts = (
        sorted(
            trial_df.filter(pl.col("subject") == ui_session_subj.value)["session"]
            .unique()
            .to_list()
        )
        if ui_session_subj.value in views
        else [0]
    )
    _sess_opts = _sess_opts or [0]
    ui_session_id = mo.ui.dropdown(
        options=[str(s) for s in _sess_opts],
        value=str(_sess_opts[0]),
        label="Session",
    )
    mo.vstack([
        mo.md("### Session deep-dive"),
        mo.hstack([ui_session_subj, ui_session_id]),
    ])
    return (ui_session_id,)


@app.cell
def _(K, THRESH_ui, mo, plots, trial_df, ui_session_id, ui_session_subj, views):
    # ── Session deep-dive plot ─────────────────────────────────────────────────
    _subj = ui_session_subj.value
    mo.stop(
        _subj not in views,
        mo.md("No fitted arrays for this subject — run the fit first."),
    )

    _sess = ui_session_id.value
    _fig = plots.plot_session_deepdive(
        views={_subj: views[_subj]},
        trial_df=trial_df,
        subj=_subj,
        sess=_sess,
        session_col="session",
        sort_col="trial_idx",
        switch_posterior_threshold=THRESH_ui.amount,
    )
    mo.vstack([
        mo.md(f"### Session deep-dive  (K={K})"),
        _fig,
    ], align="center")
    return


@app.cell
def _(K, current_hash, mo, model_cfg, paths, plots):
    # ── τ sweep analysis ────────────────────────────────────────────────────────
    _sweep_path = paths.RESULTS / "fits" / "tau_sweep" / f"glmhmmt_K{K}" / "tau_sweep_summary.parquet"
    mo.stop(
        not _sweep_path.exists(),
        mo.md(
            f"**τ sweep results not found.**  \
     Run the sweep first:\n```\n"
            f"uv run glmhmmt-fit-tau-sweep --model glmhmmt --K {K}\n```"
        ),
    )
    _subjects = list(model_cfg.subjects)
    _fig_sweep, _best = plots.plot_tau_sweep(
        sweep_path=_sweep_path,
        subjects=_subjects,
        K=K,
    )
    mo.vstack([
        mo.md(f"### τ sweep results — {model_cfg.alias or current_hash} K={K}"),
        _fig_sweep,
        mo.md("**Best τ per subject (min BIC):**"),
        mo.plain_text(_best.to_pandas().to_string(index=False)),
    ], align="center")
    return


if __name__ == "__main__":
    app.run()
