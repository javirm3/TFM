import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "glmhmmt", "src"))
    from analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
        resolve_selected_model_id,
        select_subject_behavior_df,
    )
    import paths
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    try:
        from scripts.fit_glmhmm import main as fit_main
        _FITTING_AVAILABLE = True
    except ImportError:
        fit_main = None
        _FITTING_AVAILABLE = False
    from tasks import get_adapter
    from glmhmmt.views import build_views
    from glmhmmt.postprocess import build_trial_df, build_emission_weights_df
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
def _(get_adapter, paths, pl, ui_model_manager):
    task_name = ui_model_manager.value["task"]
    adapter = get_adapter(task_name)
    df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df_all = adapter.subject_filter(df_all)
    is_2afc = adapter.num_classes == 2
    plots = adapter.get_plots()
    return adapter, df_all, is_2afc, plots, task_name


@app.cell
def _(ModelManagerWidget, mo):
    mm_widget = ModelManagerWidget(
        model_type="glmhmm",
        task="2AFC",
        K=2,
        tau=50,
    )
    ui_model_manager = mo.ui.anywidget(mm_widget)
    return mm_widget, ui_model_manager


@app.cell
def _(mo):
    get_last_fit_click, set_last_fit_click = mo.state(0)
    return get_last_fit_click, set_last_fit_click


@app.cell
def _(mo, task_name, ui_model_manager):
    from scripts.fit_glmhmm import generate_model_id as _gen_id

    class _V:
        def __init__(self, value):
            self.value = value

    _val = ui_model_manager.value
    current_hash = _gen_id(
        task_name,
        _val["K"],
        _val["tau"],
        _val["emission_cols"],
        _val.get("frozen_emissions", {}),
        _val.get("cv_mode", "none"),
        _val.get("cv_repeats", 0),
    )
    ui_existing = _V(None if _val.get("existing_model") in ("", "__default__") else _val.get("existing_model"))
    ui_alias = _V(_val.get("alias", ""))
    ui_K = _V(_val["K"])
    ui_subjects = _V(_val["subjects"])
    ui_tau = _V(_val["tau"])
    ui_emission_cols = _V(_val["emission_cols"])
    ui_frozen_emissions = _V(_val.get("frozen_emissions", {}))
    ui_cv_mode = _V(_val.get("cv_mode", "none"))
    ui_cv_repeats = _V(_val.get("cv_repeats", 5))
    fit_clicks = _val.get("run_fit_clicks", 0)

    mo.vstack(
        [
            mo.md("### Configuration"),
            ui_model_manager,
            mo.md(f"**Current params hash:** `{current_hash}`"),
        ],
        align="center",
    )
    return (
        current_hash,
        fit_clicks,
        ui_K,
        ui_alias,
        ui_cv_mode,
        ui_cv_repeats,
        ui_emission_cols,
        ui_existing,
        ui_frozen_emissions,
        ui_subjects,
        ui_tau,
    )


@app.cell
def _(
    current_hash,
    fit_clicks,
    fit_main,
    get_last_fit_click,
    mm_widget,
    mo,
    paths,
    set_last_fit_click,
    task_name,
    ui_K,
    ui_alias,
    ui_cv_mode,
    ui_cv_repeats,
    ui_emission_cols,
    ui_existing,
    ui_frozen_emissions,
    ui_subjects,
    ui_tau,
):
    _last_fit_click = get_last_fit_click()
    mo.stop(
        fit_clicks <= _last_fit_click,
        mo.md("Configure parameters and press **Run fit**."),
    )
    set_last_fit_click(fit_clicks)

    _n_restarts = 1 if ui_cv_mode.value != "none" else 5
    _cv_repeats = int(ui_cv_repeats.value) if ui_cv_mode.value != "none" else 0

    _selected_id = ui_existing.value or (ui_alias.value if ui_alias.value else current_hash)
    _OUT = paths.RESULTS / "fits" / task_name / "glmhmm" / _selected_id

    def _progress_title(info: dict) -> str:
        return (
            f"Fitting GLM-HMM K={info['K']} "
            f"subject {info['subject_index']}/{info['subject_total']}: {info['subject']}"
        )

    def _progress_subtitle(info: dict) -> str:
        _base = f"Restart {info['restart_index']}/{info['restart_total']}"
        if info.get("event") == "restart_complete":
            return f"{_base} complete"
        return _base

    _total_progress = max(
        1,
        len(ui_subjects.value) * (_cv_repeats if ui_cv_mode.value != "none" else _n_restarts),
    )
    mm_widget.is_running = True
    try:
        with mo.status.progress_bar(
            total=_total_progress,
            title=f"Fitting GLM-HMM K={ui_K.value}",
            subtitle=(
                f"{len(ui_subjects.value)} subjects × {_cv_repeats} CV repeat(s)"
                if ui_cv_mode.value != "none"
                else f"{len(ui_subjects.value)} subjects × {_n_restarts} restart(s)"
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
                        increment=0 if ui_cv_mode.value != "none" else 1,
                        title=_progress_title(info),
                        subtitle=_progress_subtitle(info),
                    )

            fit_main(
                subjects=ui_subjects.value,
                K_list=[ui_K.value],
                out_dir=_OUT,
                tau=ui_tau.value,
                emission_cols=ui_emission_cols.value,
                frozen_emissions=ui_frozen_emissions.value or None,
                task=task_name,
                cv_mode=ui_cv_mode.value,
                cv_repeats=_cv_repeats,
                n_restarts=_n_restarts,
                verbose=False,
                progress_callback=_on_progress,
            )
        mm_widget.saved_model_name = _selected_id
        mm_widget.alias_error = ""
        mm_widget.alias_status = ""
        if not ui_alias.value:
            mm_widget.alias = _selected_id
        mm_widget._update_options()
        if _selected_id in mm_widget.existing_models:
            mm_widget.existing_model = _selected_id
    finally:
        mm_widget.is_running = False
    mo.md("✅ Fit complete — plots below update automatically.")
    return


@app.cell
def _(OUT, mo, pd, pl, plt, sns, ui_cv_mode):
    import json
    mo.stop(ui_cv_mode.value == "none", mo.md(""))

    repeat_files = sorted(OUT.glob("*_cv_repeats.parquet"))
    mo.stop(not repeat_files, mo.md("No CV repeat diagnostics found yet."))

    repeats_df = pl.concat([pl.read_parquet(path) for path in repeat_files], how="diagonal_relaxed")

    repeats_pd = repeats_df.to_pandas()

    count_rows = []
    for _row in repeats_pd.to_dict(orient="records"):
        print(_row)
        for split_name, counts_key in (
            ("train", "train_label_counts_json"),
            ("test", "test_label_counts_json"),
        ):
            counts = json.loads(_row.get(counts_key) or "{}")
            for ild, count in counts.items():
                count_rows.append(
                    {
                        "subject": _row["subject"],
                        "repeat_index": int(_row["repeat_index"]),
                        "subject_repeat": f"{_row['subject']} r{int(_row['repeat_index'])}",
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
    plt.show()
    # summary_cols = [
    #     "subject",
    #     "repeat_index",
    #     "balance_score",
    #     "train_session_count",
    #     "test_session_count",
    #     "test_ll_per_trial",
    #     "test_acc",
    # ]
    return


@app.cell
def _(
    adapter,
    current_hash,
    df_all,
    load_fit_arrays,
    paths,
    resolve_selected_model_id,
    task_name,
    ui_K,
    ui_alias,
    ui_emission_cols,
    ui_existing,
    ui_subjects,
):
    K = ui_K.value

    selected_model_id = resolve_selected_model_id(
        current_hash,
        ui_existing.value,
        ui_alias.value,
    )
    OUT = paths.RESULTS / "fits" / task_name / "glmhmm" / selected_model_id
    arrays_store, names = load_fit_arrays(
        out_dir=OUT,
        arrays_suffix="glmhmm_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=list(ui_subjects.value),
        emission_cols=list(ui_emission_cols.value),
        k=K,
    )
    return K, OUT, arrays_store, names, selected_model_id


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
def _(adapter, mo):
    # ── State-scoring regressor selector ─────────────────────────────────────
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
def _(K, adapter, arrays_store, build_views, mo, ui_scoring_key, ui_subjects):
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    if hasattr(adapter, "scoring_key"):
        adapter.scoring_key = ui_scoring_key.value
    views = build_views(arrays_store, adapter, K, _selected)
    state_labels = {s: v.state_name_by_idx for s, v in views.items()}
    return state_labels, views


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


@app.cell
def _(
    K,
    arrays_store,
    is_2afc,
    mo,
    names,
    paths,
    plots,
    state_labels,
    ui_subjects,
    views,
):
    # ── emission weights ───────────────────────────────────────────────────────
    # W shape: (K, 2, n_features)  — axis-1: [L-choice=0, R-choice=1]
    # Center = reference class (implicit weight 0).
    #
    # Agonist collapse: for symmetric L/R feature pairs, take
    #   mean(W[k, 0, feat_L], W[k, 1, feat_R])  → one point per group per state
    # For C features (no direct weight): -mean(W[k, 0, feat_C], W[k, 1, feat_C])
    # For shared scalars: mean across both rows.
    #
    # Groups: (label, [(feat_name, class_idx), ...])
    # class_idx int = direct weight; "neg_mean"/"mean" = derived from both rows
    # Coherent = cue and choice on same side; Incoherent = opposite side

    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))
    _save_path = paths.RESULTS / "plots/GLMHMM/emissions_coefs.png"
    _views_sel = {s: views[s] for s in _selected}
    if is_2afc:
        _fig_by_subject = plots.plot_emission_weights_by_subject(
            views=_views_sel,
            K=K,
            save_path=_save_path,
        )
        _summary_figs = [plots.plot_emission_weights_summary(views=_views_sel, K=K)]
    else:
        _fig_by_subject = plots.plot_emission_weights_by_subject(
            arrays_store=arrays_store,
            state_labels=state_labels,
            names=names,
            K=K,
            subjects=_selected,
            save_path=_save_path,
        )
        _summary_figs = list(
            plots.plot_emission_weights(
                arrays_store=arrays_store,
                state_labels=state_labels,
                names=names,
                K=K,
                subjects=_selected,
            )
        )
    mo.vstack([mo.md("### Emission weights"), mo.md("#### By subject"), _fig_by_subject, mo.md("#### Summary"), *_summary_figs])
    return


@app.cell
def _(K, arrays_store, mo, plots, state_labels, ui_subjects):
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))
    _fig_by_subject = plots.plot_transition_matrix_by_subject(
        arrays_store=arrays_store,
        state_labels=state_labels,
        K=K,
        subjects=_selected,
    )
    _fig_summary = plots.plot_transition_matrix(
        arrays_store=arrays_store,
        state_labels=state_labels,
        K=K,
        subjects=_selected,
    )
    mo.vstack([mo.md(f"### Transition matrices  (K={K})"), mo.md("#### By subject"), _fig_by_subject, mo.md("#### Summary"), _fig_summary])
    return


@app.cell
def _(mo, ui_subjects, views):
    # ── trial-window slider (shared across all posterior plots) ──────────────
    _selected = [s for s in ui_subjects.value if s in views]
    _T_max = max((views[s].T for s in _selected), default=200)
    ui_trial_range = mo.ui.range_slider( start=0, stop=_T_max - 1, value=[0, min(_T_max - 1, 199)], label="Trial window", step=1,)
    mo.vstack([mo.md("### Trial window"), ui_trial_range])
    return (ui_trial_range,)


@app.cell
def _(K, mo, np, plt, sns, ui_subjects, ui_trial_range, views):
    # ── posterior state probabilities ─────────────────────────────────────────
    _selected = [s for s in ui_subjects.value if s in views]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _t0, _t1 = ui_trial_range.value
    _n_subj = len(_selected)
    _fig_p, _axes_p = plt.subplots( _n_subj, 1, figsize=(14, 3 * _n_subj), squeeze=False )

    for _i, _subj in enumerate(_selected):
        _ax = _axes_p[_i, 0]
        _view = views[_subj]
        _probs = np.asarray(_view.smoothed_probs)[_t0 : _t1 + 1]
        _y = np.asarray(_view.y).astype(int)[_t0 : _t1 + 1]
        _T_w = _probs.shape[0]
        _x = np.arange(_t0, _t0 + _T_w)

        # stacked area — color by label rank so Engaged is always palette[0]
        _colors = ["tab:green", "tab:grey", *sns.color_palette("tab10", n_colors=max(0, K - 2))]
        _bottom = np.zeros(_T_w)
        _slbl = _view.state_name_by_idx
        sorted_states = list(_view.state_idx_order)

        for _k in sorted_states:
            _rank = _view.state_rank_by_idx.get(int(_k), int(_k))
            _col = _colors[_rank] if _rank < len(_colors) else sns.color_palette("tab10", n_colors=K)[_rank % K]
            _ax.fill_between( _x, _bottom, _bottom + _probs[:, _k], alpha=0.7, color=_col, label=_slbl.get(_k, f"State {_k}"),)
            _bottom += _probs[:, _k]

        # choice markers on top
        _choice_colors = {0: "royalblue", 1: "gold", 2: "tomato"}
        _choice_labels = {0: "L", 1: "C", 2: "R"}
        for _resp, _col in _choice_colors.items():
            _mask = _y == _resp
            _ax.scatter( _x[_mask], np.ones(_mask.sum()) * 1.03, c=_col, s=4, marker="|", label=_choice_labels[_resp], transform=_ax.get_xaxis_transform(), clip_on=False,)

        _ax.set_xlim(_t0, _t0 + _T_w - 1)
        _ax.set_ylim(0, 1)
        _ax.set_ylabel("State probability")
        _ax.set_title(f"Subject {_subj}")
        _ax.legend( bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, ncol=1, frameon=False,)

    _axes_p[-1, 0].set_xlabel("Trial")
    _fig_p.tight_layout()
    _fig_p.subplots_adjust(right=0.85)
    sns.despine(fig=_fig_p)
    mo.vstack(
        [
            mo.md(f"### Posterior state probabilities  (K={K})"),
            _fig_p,
        ], align="center",
    )
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
    ui_subjects,
    views,
):
    _selected = [s for s in ui_subjects.value if s in views]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _views_sel = {s: views[s] for s in _selected}
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(_selected))

    mo.stop(_trial_df_sel.height == 0, mo.md("No subjects with matching data lengths."))

    _plot_df_all = plots.prepare_predictions_df(_trial_df_sel)
    _perf_kwargs = {"views": _views_sel} if is_2afc else {}
    _fig_all, _ = plots.plot_categorical_performance_all(
        _plot_df_all,
        f"glmhmm K={K}",
        background_style=ui_psychometric_background.value,
        **_perf_kwargs,
    )
    for _ax_idx, _ax in enumerate(_fig_all.axes):
        _ax.set_title("")
        _ax.set_ylabel(r"$\mathit{p}(\mathrm{right})$" if _ax_idx == 0 else "")
    if _fig_all._suptitle is not None:
        _fig_all._suptitle.set_text("")
    _fig_all.tight_layout()

    _plot_df_state = plots.prepare_predictions_df(_trial_df_sel)
    _fig_state, _ = plots.plot_categorical_performance_by_state(
        df=_plot_df_state,
        views=_views_sel,
        model_name=f"glmhmm K={K} — per state",
        background_style=ui_psychometric_background.value,
        show_weighted_points=ui_state_show_weighted_points.value,
        show_data_smooth=ui_state_show_data_smooth.value,
        show_model_smooth=ui_state_model_line_mode.value != "none",
        model_line_mode=ui_state_model_line_mode.value,
        state_assignment_mode=ui_state_assignment_mode.value,
        figure_dpi=80,
    )
    _fig_state_overlay, _ = plots.plot_categorical_performance_by_state(
        df=_plot_df_state,
        views=_views_sel,
        model_name=f"glmhmm K={K} — all states",
        background_style=ui_psychometric_background.value,
        show_weighted_points=ui_state_show_weighted_points.value,
        show_data_smooth=ui_state_show_data_smooth.value,
        show_model_smooth=ui_state_model_line_mode.value != "none",
        model_line_mode=ui_state_model_line_mode.value,
        state_assignment_mode=ui_state_assignment_mode.value,
        figure_dpi=80,
        overlay_only=True,
    )
    _reg_plot_fn = getattr(plots, "plot_regressor_psychometric_by_state", None)
    if is_2afc and _reg_plot_fn is not None:
        _fig_reg_state, _ = _reg_plot_fn(
            df=_plot_df_state,
            views=_views_sel,
            model_name=f"glmhmm K={K}",
            feature_col=ui_psychometric_regressor.value,
            background_style=ui_psychometric_background.value,
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
        )
        _fig_reg_overlay, _ = _reg_plot_fn(
            df=_plot_df_state,
            views=_views_sel,
            model_name=f"glmhmm K={K}",
            feature_col=ui_psychometric_regressor.value,
            background_style=ui_psychometric_background.value,
            show_weighted_points=ui_state_show_weighted_points.value,
            show_data_smooth=ui_state_show_data_smooth.value,
            show_model_smooth=ui_state_model_line_mode.value != "none",
            model_line_mode=ui_state_model_line_mode.value,
            state_assignment_mode=ui_state_assignment_mode.value,
            figure_dpi=80,
            overlay_only=True,
        )
        _reg_section = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.md("### Per-state psychometric by regressor"),
                        ui_psychometric_regressor,
                    ],
                    justify="space-between",
                ),
                mo.vstack(
                    [
                        mo.vstack([_fig_reg_state], align="center"),
                        save_plot(
                            _fig_reg_state,
                            f"{ui_psychometric_regressor.value} by state",
                            stem=f"regressor_by_state_{ui_psychometric_regressor.value}",
                        ),
                        mo.vstack([_fig_reg_overlay], align="center"),
                        save_plot(
                            _fig_reg_overlay,
                            f"all states {ui_psychometric_regressor.value}",
                            stem=f"regressor_all_states_{ui_psychometric_regressor.value}",
                        ),
                    ],
                    justify="space-between",
                    align="center",
                ),
            ],
            align="center",
        )
    else:
        _reg_section = mo.md("This task does not expose a regressor psychometric plot.")

    mo.vstack(
        [
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
                widths=[4, 1],
            ),
            mo.md("### Per-state categorical performance"),
            mo.vstack(
                [
                    mo.vstack([_fig_state_overlay], align="center"),
                    save_plot(_fig_state_overlay, "all states psychometric", stem="categorical_all_states"),
                    mo.vstack([_fig_state], align="center"),
                    save_plot(_fig_state, "per-state psychometric", stem="categorical_by_state"),
                ],
                justify="space-between",
                align="center",
            ),
            _reg_section,
        ],
        align="center",
    )
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
                "The overall and per-state categorical plots update using the edited state."
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
    for _ax_idx, _ax in enumerate(_fig_all_tweaked.axes):
        _ax.set_title("")
        _ax.set_ylabel(r"$\mathit{p}(\mathrm{right})$" if _ax_idx == 0 else "")
    if _fig_all_tweaked._suptitle is not None:
        _fig_all_tweaked._suptitle.set_text("")
    _fig_all_tweaked.tight_layout()
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
                        ],
                        align="start",
                    ),
                ],
                justify="space-between",
                align="center",
                widths=[4, 1],
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
            coef_editor
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
                    ))
    return (THRESH_ui,)


@app.cell
def _(
    THRESH_ui,
    adapter,
    arrays_store,
    is_2afc,
    mo,
    plots,
    trial_df,
    ui_subjects,
    views,
):
    # ── Per-state accuracy — Ashwood et al. 2022 method ────────────────────────────────────────────────
    # All     : mean(performance) on nonzero-stim trials — the full pool
    # State k : mean(performance) on the SUBSET where posterior[:,k] >= thresh
    #           AND stimd_n != 0
    # "All" is the weighted average of the state bars (plus ambiguous trials).
    # Colors assigned by rank: Engaged=palette[0], Disengaged=palette[1], …

    _selected_acc = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected_acc, mo.md("No fitted subjects available."))
    if is_2afc:
        _fig_acc, _tbl = plots.plot_state_accuracy(
            views={s: views[s] for s in _selected_acc},
            trial_df=trial_df,
            thresh=THRESH_ui.amount,
            session_col=adapter.session_col,
            sort_col=adapter.sort_col,
        )
    else:
        _fig_acc, _tbl = plots.plot_state_accuracy(
            views={s: views[s] for s in _selected_acc},
            trial_df=trial_df,
            thresh=THRESH_ui.amount,
            session_col=adapter.session_col,
            sort_col=adapter.sort_col,

        )

    mo.vstack([
        mo.md("### Accuracy by state"),
        _fig_acc,
        mo.md(f"> **All** = full nonzero-stim pool · **State bars** = subsets where posterior ≥ {THRESH_ui}"),
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
    selected_traj = [s for s in ui_subjects_traj.value if s in views]
    mo.stop(not selected_traj, mo.md("Select subjects above to view session trajectories."))
    _fig_traj = plots.plot_session_trajectories(
        views={s: views[s] for s in selected_traj},
        trial_df=trial_df,
        session_col="session",
        sort_col="trial_idx",
    )
    mo.vstack([
        mo.md(f"### c. Average state-probability trajectories within a session  (K={K})"),
        _fig_traj,
        mo.md("> Mean ± 1 s.e.m. across sessions for the selected subjects."),
    ], align="center")
    return


@app.cell
def _(K, THRESH_ui, mo, plots, trial_df, ui_subjects_traj, views):
    selected_occ = [s for s in ui_subjects_traj.value if s in views]
    mo.stop(not selected_occ, mo.md("Select subjects above."))
    _fig_occ = plots.plot_state_occupancy(
        views={s: views[s] for s in selected_occ},
        trial_df=trial_df,
        session_col="session",
        sort_col="trial_idx",
        switch_posterior_threshold=THRESH_ui.amount,
    )
    mo.vstack([
        mo.md(f"### d. Fractional occupancy & state changes per session  (K={K})"),
        _fig_occ,
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
    ], align="center")
    return


@app.cell
def _(mo, ui_subjects, views):
    # ── Session deep-dive controls ─────────────────────────────────────────────
    _selected = sorted((s for s in ui_subjects.value if s in views), key=str)
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
    mo.vstack(
        [
            mo.md("### Session deep-dive"),
            mo.hstack([ui_session_subj, ui_session_id]),
        ]
    )
    return (ui_session_id,)


@app.cell
def _(K, mo, plots, trial_df, ui_session_id, ui_session_subj, views):
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
        sort_col="trial",
    )
    mo.vstack([
        mo.md(f"### Session statistics  (K={K})"),
        mo.hstack([ui_session_subj, ui_session_id]),
        _fig,
    ], align="center")
    return


@app.cell
def _(K, df_all, mo, np, paths, pl, plt, sns, ui_subjects):
    # ── τ sweep analysis ────────────────────────────────────────────────────────
    # Loads results produced by:
    #   uv run python scripts/fit_tau_sweep.py --model glmhmm --K <K>
    # Expects: RESULTS/fits/tau_sweep/glmhmm_K<K>/tau_sweep_summary.parquet

    _sweep_path = (
        paths.RESULTS
        / "fits"
        / "tau_sweep"
        / f"glmhmm_K{K}"
        / "tau_sweep_summary.parquet"
    )
    mo.stop(
        not _sweep_path.exists(),
        mo.md(
            f"**τ sweep results not found.**  \
     Run the sweep first:\n```\n"
            f"uv run python scripts/fit_tau_sweep.py --model glmhmm --K {K}\n```"
        ),
    )

    _df_sweep = pl.read_parquet(_sweep_path)
    _subjects = [
        s
        for s in ui_subjects.value
        if s in _df_sweep["subject"].unique().to_list()
    ]
    mo.stop(not _subjects, mo.md("No sweep data for selected subjects."))

    # ── BIC vs τ plot ────────────────────────────────────────────────────
    _fig_sweep, _axes_sw = plt.subplots(1, 2, figsize=(12, 4))
    _ax_bic, _ax_ll = _axes_sw
    _palette = sns.color_palette("tab10", n_colors=len(_subjects))
    n_trials = df_all.group_by("subject").agg(pl.len().alias("n_trials"))

    for _i, _subj in enumerate(_subjects):
        _d = _df_sweep.filter(
            (pl.col("subject") == _subj) & (pl.col("K") == K)
        ).sort("tau")
        _tau = _d["tau"].to_numpy()
        _bic = _d["bic"].to_numpy()
        _ll = _d["ll_per_trial"].to_numpy()
        _c = _palette[_i]
        _ax_bic.plot(_tau, _bic, "-o", ms=3, color=_c, label=_subj)
        _ax_ll.plot(_tau, _ll, "-o", ms=3, color=_c, label=_subj)
        # mark best τ
        _best_idx = int(np.argmin(_bic))
        _ax_bic.axvline(
            _tau[_best_idx], color=_c, lw=0.8, linestyle="--", alpha=0.6
        )
    4
    for _ax, _ylabel, _title in [
        (_ax_bic, "BIC", "BIC vs τ  (lower is better)"),
        (_ax_ll, "LL / trial", "Log-likelihood per trial vs τ"),
    ]:
        _ax.set_xlabel("τ (action-trace half-life)")
        _ax.set_ylabel(_ylabel)
        _ax.set_title(_title)
        _ax.legend(fontsize=8, frameon=False)
        sns.despine(ax=_ax)

    _fig_sweep.tight_layout()

    # ── best τ table ────────────────────────────────────────────────────────
    _best = (
        _df_sweep.filter(pl.col("subject").is_in(_subjects) & (pl.col("K") == K))
        .sort("bic")
        .group_by(["subject", "K"])
        .first()
        .select(["subject", "K", "tau", "bic", "ll_per_trial", "acc"])
        .sort(["subject", "K"])
    )

    _best_all = (
        _df_sweep.filter(pl.col("subject").is_in(_subjects) & (pl.col("K") == K))
        .join(n_trials, on="subject", how="left")
        .group_by("tau")
        .agg(
            [
                (pl.col("bic") * pl.col("n_trials")).sum().alias("bic_wsum"),
                (pl.col("ll_per_trial") * pl.col("n_trials"))
                .sum()
                .alias("llpt_wsum"),
                (pl.col("acc") * pl.col("n_trials")).sum().alias("acc_wsum"),
                pl.col("n_trials").sum().alias("n_total"),
                pl.n_unique("subject").alias("n_subjects"),
            ]
        )
        .with_columns(
            [
                (pl.col("bic_wsum") / pl.col("n_total")).alias("bic_mean_w"),
                (pl.col("llpt_wsum") / pl.col("n_total")).alias(
                    "ll_per_trial_mean_w"
                ),
                (pl.col("acc_wsum") / pl.col("n_total")).alias("acc_mean_w"),
            ]
        )
        .select(
            [
                "tau",
                "bic_mean_w",
                "ll_per_trial_mean_w",
                "acc_mean_w",
                "n_subjects",
                "n_total",
            ]
        )
        .sort("bic_mean_w")
    )

    mo.vstack(
        [
            mo.md(f"### τ sweep results — glmhmm K={K}"),
            _fig_sweep,
            mo.md("**Best τ per subject (min BIC):**"),
            mo.plain_text(_best.to_pandas().to_string(index=False)),
            mo.ui.dataframe(_best_all),
        ],
        align="center",
    )
    return


@app.cell
def _(mo, task_name):

    # ── SSM GLM-HMM safety check (2AFC only) ──────────────────────────────────
    mo.stop(
        task_name != "2AFC",
        mo.md("ℹ️ **SSM safety check is only available for the 2AFC task.** Switch task to 2AFC above."),
    )
    ssm_run_btn = mo.ui.run_button(label="▶ Run SSM safety check")
    mo.vstack([
        mo.md("### SSM GLM-HMM safety check (2AFC)"),
        mo.md(
            "Fits a K-state GLM-HMM using the **SSM library** (`input_driven_obs`, `standard` "
            "transitions) with the exact same covariates as the custom model.  \n"
        ),
        ssm_run_btn,
    ])
    return (ssm_run_btn,)


@app.cell
def _(
    K,
    adapter,
    build_trial_df,
    build_views,
    df_all,
    mo,
    np,
    paths,
    pl,
    plots,
    selected_model_id,
    ssm_run_btn,
    task_name,
    trial_df,
    ui_subjects,
    views,
):
    # ── SSM fit + comparison tables ────────────────────────────────────────────
    mo.stop(not ssm_run_btn.value, mo.md("Press **▶ Run SSM safety check** above to fit."))
    try:
        import ssm as ssm_lib
    except Exception as exc:
        mo.stop(
            True,
            mo.md(
                "SSM could not be imported in the current environment, so the SSM vs custom log-likelihood comparison cannot run. "
                f"Import error: `{type(exc).__name__}: {exc}`. "
                "In this project, that usually means `ssm` is installed but incompatible with the currently resolved "
                "`autograd`/`numpy` versions, not that `uv` is using the wrong environment."
            ),
        )
    from scripts.fit_common import valid_trial_mask

    ssm_subjects = [subject for subject in ui_subjects.value if subject in views]
    mo.stop(not ssm_subjects, mo.md("No fitted arrays found — run the custom fit first."))

    ssm_arrays = {}
    cmp_rows = []
    missing_metric_subjects = []
    out_dir = paths.RESULTS / "fits" / task_name / "glmhmm" / selected_model_id


    def load_custom_metrics(subject: str, n_trials: int):
        candidates = [
            out_dir / f"{subject}_K{K}_glmhmm_metrics.parquet",
            out_dir / f"{subject}_glmhmm_metrics.parquet",
            *sorted(out_dir.glob(f"{subject}*_glmhmm_metrics.parquet")),
        ]
        for path in dict.fromkeys(candidates):
            if not path.exists():
                continue
            metrics_df = pl.read_parquet(path)
            if metrics_df.height == 0:
                continue
            row = metrics_df.row(0, named=True)
            raw_ll = row.get("raw_ll")
            ll_per_trial = row.get("ll_per_trial")
            if raw_ll is None and ll_per_trial is not None:
                raw_ll = float(ll_per_trial) * n_trials
            if ll_per_trial is None and raw_ll is not None:
                ll_per_trial = float(raw_ll) / max(n_trials, 1)
            if raw_ll is None or ll_per_trial is None:
                continue
            return float(raw_ll), float(ll_per_trial), path.name
        return np.nan, np.nan, None


    def ssm_data_loglik(model, choices_list, inputs_list):
        if hasattr(model, "log_likelihood"):
            return float(model.log_likelihood(choices_list, inputs=inputs_list)), "log_likelihood"
        if hasattr(model, "log_probability"):
            return float(model.log_probability(choices_list, inputs=inputs_list)), "log_probability"
        raise AttributeError("SSM HMM object exposes neither log_likelihood nor log_probability.")


    def stable_softmax_np(logits: np.ndarray) -> np.ndarray:
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


    with mo.status.spinner(title="Fitting SSM GLM-HMM…"):
        for _subject in ssm_subjects:
            view = views[_subject]
            X = np.asarray(view.X)  # (T, n_feat) — already session-filtered
            y = np.asarray(view.y)  # (T,)

            # Reconstruct session ids with same mask as fit_subject()
            subject_df = df_all.filter(pl.col("subject") == _subject).sort(adapter.sort_col)
            session_ids_raw = subject_df[adapter.session_col].to_numpy()
            valid_mask = valid_trial_mask(session_ids_raw)
            session_ids = session_ids_raw[valid_mask]

            # Split into per-session lists — SSM expects list of arrays
            unique_sessions = list(dict.fromkeys(session_ids.tolist()))
            choices_list, inputs_list = [], []
            for session_id in unique_sessions:
                idx = np.where(session_ids == session_id)[0]
                choices_list.append(y[idx].reshape(-1, 1).astype(int))
                inputs_list.append(X[idx].astype(float))

            # Initialise and fit
            obs_dim = 1
            n_cats = 2
            n_feat = X.shape[1]
            glmhmm_ssm = ssm_lib.HMM(
                K,
                obs_dim,
                n_feat,
                observations="input_driven_obs",
                observation_kwargs=dict(C=n_cats),
                transitions="standard",
            )
            glmhmm_ssm.fit(
                choices_list,
                inputs=inputs_list,
                method="em",
                num_iters=200,
                tolerance=1e-4,
            )

            W_ssm = glmhmm_ssm.observations.params  # (K, C-1, n_feat); flip sign
            transition_matrix_ssm = glmhmm_ssm.transitions.transition_matrix  # (K, K)
            smoothed_probs_ssm = np.vstack(
                [glmhmm_ssm.expected_states(data=data, input=inp)[0] for data, inp in zip(choices_list, inputs_list)]
            )
            initial_state_distn_ssm = np.asarray(glmhmm_ssm.init_state_distn.initial_state_distn, dtype=float)
            p_pred_ssm_parts = []
            for data, inp in zip(choices_list, inputs_list):
                filtered_probs = np.asarray(glmhmm_ssm.filter(data=data, input=inp), dtype=float)  # (T_s, K)
                n_trials_session = int(inp.shape[0])
                pred_z_session = (
                    np.vstack(
                        [
                            initial_state_distn_ssm[None, :],
                            filtered_probs[:-1] @ transition_matrix_ssm,
                        ]
                    )
                    if n_trials_session > 1
                    else initial_state_distn_ssm[None, :]
                )
                logits_ce_session = np.einsum("kcf,tf->tkc", W_ssm, np.asarray(inp, dtype=float))
                logits_session = np.concatenate(
                    [
                        logits_ce_session,
                        np.zeros((n_trials_session, K, 1), dtype=float),
                    ],
                    axis=-1,
                )
                p_y_given_z_session = stable_softmax_np(logits_session)  # (T_s, K, C)
                p_pred_ssm_parts.append(np.einsum("tk,tkc->tc", pred_z_session, p_y_given_z_session))
            p_pred_ssm = np.concatenate(p_pred_ssm_parts, axis=0)
            ssm_raw_ll, ssm_ll_source = ssm_data_loglik(glmhmm_ssm, choices_list, inputs_list)
            _n_trials = int(y.shape[0])
            ssm_ll_per_trial = ssm_raw_ll / max(_n_trials, 1)
            custom_raw_ll, custom_ll_per_trial, metric_file = load_custom_metrics(_subject, _n_trials)
            if metric_file is None:
                missing_metric_subjects.append(_subject)

            cmp_rows.append(
                {
                    "subject": _subject,
                    "n_trials": _n_trials,
                    "custom_raw_ll": custom_raw_ll,
                    "ssm_raw_ll": ssm_raw_ll,
                    "delta_raw_ll_ssm_minus_custom": ssm_raw_ll - custom_raw_ll,
                    "custom_ll_per_trial": custom_ll_per_trial,
                    "ssm_ll_per_trial": ssm_ll_per_trial,
                    "delta_ll_per_trial_ssm_minus_custom": ssm_ll_per_trial - custom_ll_per_trial,
                    "custom_metrics_file": metric_file,
                    "ssm_ll_source": ssm_ll_source,
                }
            )

            ssm_arrays[_subject] = {
                "smoothed_probs": smoothed_probs_ssm,
                "emission_weights": W_ssm,
                "transition_matrix": transition_matrix_ssm,
                "X": X,
                "y": y,
                "X_cols": np.array(list(view.feat_names), dtype=object),
                "p_pred": p_pred_ssm,
            }

    ssm_views = build_views(ssm_arrays, adapter, K, ssm_subjects)
    views_sel = {subject: views[subject] for subject in ssm_subjects}
    ssm_views_sel = {subject: ssm_views[subject] for subject in ssm_subjects}
    trial_df_custom_sel = trial_df.filter(pl.col("subject").is_in(ssm_subjects))
    sort_col = adapter.sort_col
    session_col = adapter.session_col
    behavioral_cols = adapter.behavioral_cols
    trial_frames_ssm = []
    for _subject, view in ssm_views_sel.items():
        subject_df = (
            df_all.filter(pl.col("subject") == _subject)
            .sort(sort_col)
            .filter(pl.col(session_col).count().over(session_col) >= 2)
        )
        if subject_df.height != view.T:
            continue
        trial_frames_ssm.append(build_trial_df(view, adapter, subject_df, behavioral_cols))
    trial_df_ssm = pl.concat(trial_frames_ssm) if trial_frames_ssm else pl.DataFrame()

    ssm_psych_fig_custom = None
    ssm_psych_fig_ssm = None
    if trial_df_custom_sel.height > 0 and trial_df_ssm.height > 0:
        plot_df_custom = plots.prepare_predictions_df(trial_df_custom_sel)
        ssm_psych_fig_custom, _ = plots.plot_categorical_performance_all(
            plot_df_custom,
            f"Dynamax glmhmm K={K}",
            views=views_sel,
        )
        for _ax_idx, _ax in enumerate(ssm_psych_fig_custom.axes):
            _ax.set_title("")
            _ax.set_ylabel(r"$\mathit{p}(\mathrm{right})$" if _ax_idx == 0 else "")
        if ssm_psych_fig_custom._suptitle is not None:
            ssm_psych_fig_custom._suptitle.set_text("")
        ssm_psych_fig_custom.tight_layout()
        plot_df_ssm = plots.prepare_predictions_df(trial_df_ssm)
        ssm_psych_fig_ssm, _ = plots.plot_categorical_performance_all(
            plot_df_ssm,
            f"SSM glmhmm K={K}",
            views=ssm_views_sel,
        )
        for _ax_idx, _ax in enumerate(ssm_psych_fig_ssm.axes):
            _ax.set_title("")
            _ax.set_ylabel(r"$\mathit{p}(\mathrm{right})$" if _ax_idx == 0 else "")
        if ssm_psych_fig_ssm._suptitle is not None:
            ssm_psych_fig_ssm._suptitle.set_text("")
        ssm_psych_fig_ssm.tight_layout()

    ssm_cmp_df = pl.DataFrame(cmp_rows)
    contrast_labels = list(adapter.choice_labels[:-1]) or ["contrast_0"]
    coef_rows = []
    for _subject in ssm_subjects:
        custom_view = views[_subject]
        _ssm_view = ssm_views[_subject]
        custom_feat_names = list(custom_view.feat_names)
        ssm_feat_names = list(_ssm_view.feat_names)
        feat_names = (
            custom_feat_names
            if custom_feat_names == ssm_feat_names
            else [
                custom_feat_names[i] if i < len(custom_feat_names) else ssm_feat_names[i]
                for i in range(min(len(custom_feat_names), len(ssm_feat_names)))
            ]
        )

        for state_rank, (custom_k, ssm_k) in enumerate(zip(custom_view.state_idx_order, _ssm_view.state_idx_order, strict=False)):
            custom_label = custom_view.state_name_by_idx.get(int(custom_k), f"State {custom_k}")
            ssm_label = _ssm_view.state_name_by_idx.get(int(ssm_k), f"State {ssm_k}")
            state_label = custom_label if custom_label == ssm_label else f"{custom_label} | {ssm_label}"
            custom_w = np.asarray(custom_view.emission_weights[int(custom_k)], dtype=float)
            ssm_w = np.asarray(_ssm_view.emission_weights[int(ssm_k)], dtype=float)
            n_contrasts = min(custom_w.shape[0], ssm_w.shape[0], len(contrast_labels))
            n_features = min(custom_w.shape[1], ssm_w.shape[1], len(feat_names))

            for contrast_idx in range(n_contrasts):
                for feature_idx in range(n_features):
                    custom_coef = float(custom_w[contrast_idx, feature_idx])
                    ssm_coef = -float(ssm_w[contrast_idx, feature_idx])
                    coef_rows.append(
                        {
                            "subject": _subject,
                            "state_rank": int(state_rank),
                            "state_label": state_label,
                            "custom_state_idx": int(custom_k),
                            "ssm_state_idx": int(ssm_k),
                            "contrast": contrast_labels[contrast_idx],
                            "feature": feat_names[feature_idx],
                            "dynamax_coef": custom_coef,
                            "ssm_coef": ssm_coef,
                            "delta_ssm_minus_dynamax": abs(ssm_coef + custom_coef),
                        }
                    )

    ssm_coef_df = (
        pl.DataFrame(coef_rows)
        if coef_rows
        else pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "state_rank": pl.Int64,
                "state_label": pl.Utf8,
                "custom_state_idx": pl.Int64,
                "ssm_state_idx": pl.Int64,
                "contrast": pl.Utf8,
                "feature": pl.Utf8,
                "dynamax_coef": pl.Float64,
                "ssm_coef": pl.Float64,
                "delta_ssm_minus_dynamax": pl.Float64,
            }
        )
    )
    ssm_coef_df = ssm_coef_df.sort(["subject", "state_rank", "contrast", "feature"])
    ssm_coef_display = ssm_coef_df.select(
        [
            "subject",
            "state_rank",
            "state_label",
            "custom_state_idx",
            "ssm_state_idx",
            "contrast",
            "feature",
            "dynamax_coef",
            "ssm_coef",
            "delta_ssm_minus_dynamax",
        ]
    )

    cmp_valid = ssm_cmp_df.filter(pl.col("custom_raw_ll").is_finite())
    if cmp_valid.height > 0:
        custom_total_raw = float(cmp_valid["custom_raw_ll"].sum())
        ssm_total_raw = float(cmp_valid["ssm_raw_ll"].sum())
        total_trials = int(cmp_valid["n_trials"].sum())
        ssm_summary_md = "\n".join(
            [
                "### Log-likelihood comparison",
                "",
                f"- Compared on **{cmp_valid.height} subject(s)** and **{total_trials} trials**.",
                f"- **Custom / Dynamax total raw LL:** `{custom_total_raw:.3f}`",
                f"- **SSM total raw LL:** `{ssm_total_raw:.3f}`",
                f"- **Δ raw LL (SSM - custom):** `{ssm_total_raw - custom_total_raw:.3f}`",
                f"- **Custom / Dynamax LL per trial:** `{custom_total_raw / max(total_trials, 1):.6f}`",
                f"- **SSM LL per trial:** `{ssm_total_raw / max(total_trials, 1):.6f}`",
                f"- **Δ LL per trial (SSM - custom):** `{(ssm_total_raw - custom_total_raw) / max(total_trials, 1):.6f}`",
            ]
        )
    else:
        ssm_summary_md = (
            "### Log-likelihood comparison\n\n"
            "No matching saved custom metrics were found for the selected fit, so only the "
            "SSM posterior overlay is shown below."
        )

    notes = []
    if missing_metric_subjects:
        notes.append("Missing custom metrics for: " + ", ".join(sorted(dict.fromkeys(missing_metric_subjects))))
    ssm_sources = sorted(dict.fromkeys(ssm_cmp_df["ssm_ll_source"].to_list())) if ssm_cmp_df.height > 0 else []
    if ssm_sources and ssm_sources != ["log_likelihood"]:
        notes.append("SSM LL used fallback method(s): " + ", ".join(ssm_sources))
    ssm_notes_md = (
        "  \n".join(f"- {note}" for note in notes)
        if notes
        else "- `raw_ll` is the data log-likelihood from the saved custom fit metrics.  \n"
        "- `delta` columns are defined as **SSM - custom / Dynamax**."
    )
    ssm_notes_md += (
        "  \n- Emission coefficients are compared after each model's states are reordered by the notebook's "
        "semantic state labelling (`state_idx_order`), not by raw fitted state index."
    )

    mo.vstack(
        [
            mo.md("### SSM GLM-HMM fit summary"),
            mo.md(ssm_summary_md),
            mo.md(ssm_notes_md),
            mo.ui.dataframe(ssm_cmp_df),
            mo.md("### Emission coefficients — SSM vs Dynamax"),
            mo.md(
                "Each row below is one fitted emission coefficient for one subject, aligned by the notebook's "
                "state order. `delta_ssm_minus_dynamax > 0` means the SSM coefficient is larger."
            ),
            mo.ui.dataframe(ssm_coef_display),
        ],
        align="center",
    )
    return (
        cmp_valid,
        ssm_coef_df,
        ssm_psych_fig_custom,
        ssm_psych_fig_ssm,
        ssm_subjects,
        ssm_views,
    )


@app.cell
def _(
    K,
    adapter,
    cmp_valid,
    np,
    plt,
    sns,
    ssm_coef_df,
    ssm_subjects,
    ssm_views,
    ui_trial_range,
    views,
):
    def choice_meta(num_classes: int):
        if num_classes == 2:
            return {0: "royalblue", 1: "tomato"}
        return {0: "royalblue", 1: "gold", 2: "tomato"}


    def choice_short_labels(labels):
        return {int(i): str(label)[0].upper() for i, label in enumerate(labels)}


    def posterior_color(rank: int):
        palette = ["tab:green", "tab:grey", *sns.color_palette("tab10", n_colors=max(0, K - 2))]
        if rank < len(palette):
            return palette[rank]
        return sns.color_palette("tab10", n_colors=K)[rank % K]


    def plot_view_posterior(
        ax,
        view,
        title: str,
        t0_plot: int,
        t1_plot: int,
        overlay_line=None,
        overlay_label: str | None = None,
    ):
        probs = np.asarray(view.smoothed_probs)[t0_plot : t1_plot + 1]
        y_window = np.asarray(view.y).astype(int)[t0_plot : t1_plot + 1]
        n_trials_window = probs.shape[0]
        x_window = np.arange(t0_plot, t0_plot + n_trials_window)
        bottom = np.zeros(n_trials_window)

        for state_idx in list(view.state_idx_order):
            rank = view.state_rank_by_idx.get(int(state_idx), int(state_idx))
            color = posterior_color(rank)
            ax.fill_between(
                x_window,
                bottom,
                bottom + probs[:, state_idx],
                alpha=0.7,
                color=color,
                label=view.state_name_by_idx.get(state_idx, f"State {state_idx}"),
            )
            bottom += probs[:, state_idx]

        engaged_state = view.engaged_k()
        engaged_label = view.state_name_by_idx.get(engaged_state, f"State {engaged_state}")
        ax.plot(
            x_window,
            probs[:, engaged_state],
            color="black",
            lw=1.4,
            alpha=0.95,
            label=f"P({engaged_label})",
        )
        if overlay_line is not None:
            ax.plot(
                x_window,
                np.asarray(overlay_line)[:n_trials_window],
                color="darkorange",
                lw=2,
                alpha=0.95,
                linestyle="--",
                label=overlay_label or "Overlay",
            )

        choice_colors = choice_meta(view.num_classes)
        choice_labels = choice_short_labels(adapter.choice_labels)
        for response, color in choice_colors.items():
            mask = y_window == response
            if not np.any(mask):
                continue
            ax.scatter(
                x_window[mask],
                np.ones(mask.sum()) * 1.03,
                c=color,
                s=4,
                marker="|",
                label=choice_labels.get(response, str(response)),
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

        ax.set_xlim(t0_plot, t0_plot + n_trials_window - 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("State probability")
        ax.set_title(title)
        ax.legend(
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=8,
            ncol=1,
            frameon=False,
        )


    ssm_ll_fig = None
    if cmp_valid.height > 0:
        import plotly.graph_objects as go

        cmp_pd = cmp_valid.select(["subject", "custom_ll_per_trial", "ssm_ll_per_trial"]).to_pandas()
        ssm_ll_fig = go.Figure()

        for row in cmp_pd.itertuples(index=False):
            ssm_ll_fig.add_trace(
                go.Scatter(
                    x=["Dynamax", "SSM"],
                    y=[row.custom_ll_per_trial, row.ssm_ll_per_trial],
                    mode="lines+markers",
                    line=dict(color="rgba(120, 120, 120, 0.22)", width=1.2),
                    marker=dict(color="rgba(0, 0, 0, 0.65)", size=7),
                    customdata=[row.subject, row.subject],
                    hovertemplate="Subject: %{customdata}<br>Model: %{x}<br>LL/trial: %{y:.6f}<extra></extra>",
                    showlegend=False,
                )
            )

        ssm_ll_fig.add_trace(
            go.Box(
                x=["Dynamax"] * len(cmp_pd),
                y=cmp_pd["custom_ll_per_trial"],
                name="Dynamax",
                marker_color="rgba(180, 180, 180, 0.9)",
                fillcolor="rgba(217, 217, 217, 0.6)",
                line=dict(color="rgba(90, 90, 90, 0.9)"),
                boxpoints=False,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        ssm_ll_fig.add_trace(
            go.Box(
                x=["SSM"] * len(cmp_pd),
                y=cmp_pd["ssm_ll_per_trial"],
                name="SSM",
                marker_color="rgba(180, 180, 180, 0.9)",
                fillcolor="rgba(217, 217, 217, 0.6)",
                line=dict(color="rgba(90, 90, 90, 0.9)"),
                boxpoints=False,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        ssm_ll_fig.update_layout(
            title="Per-subject LL comparison",
            xaxis_title=None,
            yaxis_title="Log-likelihood per trial",
            template="simple_white",
            width=560,
            height=420,
            margin=dict(l=60, r=20, t=60, b=50),
        )
        ssm_ll_fig.update_yaxes(zeroline=False)
        ssm_ll_fig.update_xaxes(categoryorder="array", categoryarray=["Dynamax", "SSM"])

    ssm_coef_fig = None
    if ssm_coef_df.height > 0:
        coef_pd = ssm_coef_df.to_pandas()
        panel_keys = (
            coef_pd[["state_rank", "state_label", "contrast"]]
            .drop_duplicates()
            .sort_values(["state_rank", "contrast"])
            .to_dict("records")
        )
        n_panels = len(panel_keys)
        n_cols = 1 if n_panels == 1 else min(2, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        ssm_coef_fig, coef_axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(8, 5.5 * n_cols), max(3.6, 3.6 * n_rows)),
            squeeze=False,
            sharey=True,
        )
        axes_flat = coef_axes.ravel()
        for ax, key in zip(axes_flat, panel_keys, strict=False):
            mask = (
                (coef_pd["state_rank"] == key["state_rank"])
                & (coef_pd["state_label"] == key["state_label"])
                & (coef_pd["contrast"] == key["contrast"])
            )
            panel_df = coef_pd.loc[mask].copy()
            sns.boxplot(
                data=panel_df,
                x="feature",
                y="delta_ssm_minus_dynamax",
                ax=ax,
                showfliers=False,
                color="#D9D9D9",
                boxprops={"alpha": 0.8},
            )
            sns.stripplot(
                data=panel_df,
                x="feature",
                y="delta_ssm_minus_dynamax",
                ax=ax,
                color="black",
                alpha=0.7,
                size=4,
                jitter=0.22,
            )
            ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.7)
            ax.set_title(f"{key['state_label']}  ({key['contrast']})")
            ax.set_xlabel("")
            ax.set_ylabel("SSM - Dynamax coefficient")
            ax.tick_params(axis="x", rotation=35)
            ax.set_yscale("log")
            sns.despine(ax=ax)
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)
        ssm_coef_fig.tight_layout()

    t0_ssm, t1_ssm = ui_trial_range.value
    n_subjects = len(ssm_subjects)
    ssm_posterior_fig, axes_ssm = plt.subplots(n_subjects, 1, figsize=(14, 3.4 * n_subjects), squeeze=False)

    for i, subject in enumerate(ssm_subjects):
        ssm_view = ssm_views[subject]
        ssm_engaged_probs = np.asarray(ssm_view.smoothed_probs)[t0_ssm : t1_ssm + 1, ssm_view.engaged_k()]
        plot_view_posterior(
            axes_ssm[i, 0],
            views[subject],
            f"Subject {subject} — Custom posterior + SSM line",
            t0_ssm,
            t1_ssm,
            overlay_line=ssm_engaged_probs,
            overlay_label="SSM P(Engaged)",
        )

    axes_ssm[-1, 0].set_xlabel("Trial")
    ssm_posterior_fig.tight_layout()
    ssm_posterior_fig.subplots_adjust(right=0.84)
    sns.despine(fig=ssm_posterior_fig)
    return ssm_coef_fig, ssm_ll_fig, ssm_posterior_fig


@app.cell
def _(ssm_posterior_fig):
    ssm_posterior_fig
    return


@app.cell
def _(
    K,
    mo,
    ssm_coef_fig,
    ssm_ll_fig,
    ssm_psych_fig_custom,
    ssm_psych_fig_ssm,
):
    mo.vstack([
        mo.md(f"### SSM GLM-HMM plots  (K={K})"),
        mo.md("### Log-likelihood comparison"),
        ssm_ll_fig if ssm_ll_fig is not None else mo.md("LL comparison unavailable because subject-level metrics are missing."),
        mo.md("### Categorical psychometrics — Dynamax vs SSM"),
        (
            mo.hstack(
                [
                    mo.vstack([mo.md("#### Dynamax"), ssm_psych_fig_custom], align="center"),
                    mo.vstack([mo.md("#### SSM"), ssm_psych_fig_ssm], align="center"),
                ],
                justify="start",
            )
            if ssm_psych_fig_custom is not None and ssm_psych_fig_ssm is not None
            else mo.md("Psychometric comparison unavailable because one of the trial-level prediction tables could not be built.")
        ),
        mo.md("### Coefficient differences"),
        ssm_coef_fig if ssm_coef_fig is not None else mo.md("No coefficient comparison available."),
    ], align="center")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
