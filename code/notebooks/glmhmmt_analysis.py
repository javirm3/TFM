import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "glmhmmt", "src"))
    import paths
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    try:
        from scripts.fit_glmhmmt import main as fit_main
        _FITTING_AVAILABLE = True
    except ImportError:
        fit_main = None
        _FITTING_AVAILABLE = False
    from tasks import get_adapter
    from glmhmmt.views import build_views
    from glmhmmt.postprocess import build_trial_df, build_emission_weights_df
    from widgets import ModelManagerWidget
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
        build_emission_weights_df,
        build_trial_df,
        build_views,
        fit_main,
        get_adapter,
        mo,
        np,
        paths,
        pl,
        plt,
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
        model_type="glmhmmt",
        task="MCDR",
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
    from scripts.fit_glmhmmt import generate_model_id as _gen_id

    class _V:
        def __init__(self, value):
            self.value = value

    _val = ui_model_manager.value
    current_hash = _gen_id(
        task_name,
        _val["K"],
        _val["tau"],
        _val["emission_cols"],
        _val.get("transition_cols", []),
        _val.get("frozen_emissions", {}),
    )
    ui_existing = _V(None if _val.get("existing_model") in ("", "__default__") else _val.get("existing_model"))
    ui_alias = _V(_val.get("alias", ""))
    ui_K = _V(_val["K"])
    ui_subjects = _V(_val["subjects"])
    ui_tau = _V(_val["tau"])
    ui_emission_cols = _V(_val["emission_cols"])
    ui_transition_cols = _V(_val["transition_cols"])
    ui_frozen_emissions = _V(_val.get("frozen_emissions", {}))
    fit_clicks = _V(_val.get("run_fit_clicks", 0))

    mo.vstack([
        mo.md("### Model Configuration"),
        ui_model_manager,
        mo.md(f"**Hash:** `{current_hash}`"),
    ])
    return (
        current_hash,
        fit_clicks,
        ui_K,
        ui_alias,
        ui_emission_cols,
        ui_existing,
        ui_frozen_emissions,
        ui_subjects,
        ui_tau,
        ui_transition_cols,
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
    ui_emission_cols,
    ui_existing,
    ui_frozen_emissions,
    ui_subjects,
    ui_tau,
    ui_transition_cols,
):
    _last_fit_click = get_last_fit_click()
    mo.stop(
        fit_clicks.value <= _last_fit_click,
        mo.md("Configure parameters and press **Run fit**."),
    )
    set_last_fit_click(fit_clicks.value)

    _n_restarts = 1
    _selected_id = ui_existing.value or (ui_alias.value if ui_alias.value else current_hash)
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

    _total_progress = max(1, len(ui_subjects.value) * _n_restarts)
    mm_widget.is_running = True
    try:
        with mo.status.progress_bar(
            total=_total_progress,
            title=f"Fitting GLM-HMM-T K={ui_K.value}",
            subtitle=f"{len(ui_subjects.value)} subjects × {_n_restarts} restart(s)",
            completion_title="Fit complete",
            completion_subtitle=f"Saved under {_selected_id}",
        ) as _bar:
            def _on_progress(info: dict) -> None:
                if info.get("event") == "restart_start":
                    _bar.update(
                        increment=0,
                        title=_progress_title(info),
                        subtitle=_progress_subtitle(info),
                    )
                    return
                if info.get("event") == "restart_complete":
                    _bar.update(
                        title=_progress_title(info),
                        subtitle=_progress_subtitle(info),
                    )

            fit_main(
                subjects=ui_subjects.value,
                K_list=[ui_K.value],
                out_dir=_OUT,
                emission_cols=ui_emission_cols.value or None,
                transition_cols=ui_transition_cols.value or None,
                frozen_emissions=ui_frozen_emissions.value or None,
                tau=ui_tau.value,
                task=task_name,
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
def _(
    adapter,
    current_hash,
    df_all,
    np,
    paths,
    pl,
    task_name,
    ui_K,
    ui_alias,
    ui_emission_cols,
    ui_existing,
    ui_subjects,
    ui_tau,
    ui_transition_cols,
):
    K = ui_K.value

    selected_model_id = ui_existing.value or (ui_alias.value if ui_alias.value else current_hash)
    OUT = paths.RESULTS / "fits" / task_name / "glmhmmt" / selected_model_id
    # load feature names from data (use first available subject for a representative build)
    _df_sel = df_all.filter(pl.col("subject").is_in(ui_subjects.value)).sort(adapter.sort_col)
    _, _, _, names = adapter.load_subject(
        _df_sel,
        tau=ui_tau.value,
        emission_cols=ui_emission_cols.value,
        transition_cols=ui_transition_cols.value,
    )

    arrays_store = {}
    _files = list(sorted(OUT.glob("*_glmhmmt_arrays.npz")))
    _files += [f for f in sorted(OUT.glob(f"*_K{K}_glmhmmt_arrays.npz")) if f not in _files]
    for _f in _files:
        _subj = _f.name.removesuffix("_glmhmmt_arrays.npz").removesuffix(f"_K{K}")
        _d = dict(np.load(_f, allow_pickle=True))
        _saved_names = {}
        if "names" in _d:
            _raw_names = _d["names"]
            if getattr(_raw_names, "shape", None) == ():
                _saved_names = _raw_names.item()
        # decode column names saved as string arrays; fall back to nested names,
        # then to the current build output for backward compatibility.
        _d["X_cols"] = (
            list(_d["X_cols"]) if "X_cols" in _d
            else list(_saved_names.get("X_cols", names["X_cols"]))
        )
        _d["U_cols"] = (
            list(_d["U_cols"]) if "U_cols" in _d
            else list(_saved_names.get("U_cols", names["U_cols"]))
        )
        arrays_store[_subj] = _d

    # arrays_store
    return K, arrays_store, names


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
def _(K, adapter, arrays_store, build_views, mo, ui_scoring_key, ui_subjects):
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    if hasattr(adapter, "scoring_key"):
        adapter.scoring_key = ui_scoring_key.value
    views = build_views(arrays_store, adapter, K, _selected)
    state_labels = {s: v.state_name_by_idx for s, v in views.items()}
    state_order = {s: v.state_idx_order for s, v in views.items()}
    return state_labels, views


@app.cell
def _(K, adapter, arrays_store, build_views, ui_scoring_key):
    if hasattr(adapter, "scoring_key"):
        adapter.scoring_key = ui_scoring_key.value
    editor_views = build_views(arrays_store, adapter, K, list(arrays_store.keys()))
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
    _sort_col = adapter.sort_col
    _ses_col = adapter.session_col
    _bcols = adapter.behavioral_cols
    _trial_frames = []
    for _subj, _view in views.items():
        _df_sub = (
            df_all
            .filter(pl.col("subject") == _subj)
            .sort(_sort_col)
            .filter(pl.col(_ses_col).count().over(_ses_col) >= 2)
        )
        if _df_sub.height != _view.T:
            print(f"⚠️  {_subj}: row mismatch ({_df_sub.height} vs {_view.T}), skipping")
            continue
        _trial_frames.append(build_trial_df(_view, adapter, _df_sub, _bcols))

    mo.stop(not _trial_frames, mo.md("No subjects with matching data lengths."))
    trial_df = pl.concat(_trial_frames)
    weights_df = build_emission_weights_df(views)
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
    _save_path = paths.RESULTS / "plots/GLMHMMT/emissions_coefs.png"
    if is_2afc:
        _fig_ag, _fig_cls = plots.plot_emission_weights(
            views={s: views[s] for s in _selected},
            K=K,
            save_path=_save_path,
        )
    else:
        _fig_ag, _fig_cls = plots.plot_emission_weights(
            arrays_store=arrays_store,
            state_labels=state_labels,
            names=names,
            K=K,
            subjects=_selected,
            save_path=_save_path,
        )
    mo.vstack([mo.md("### Emission weights"), _fig_ag, _fig_cls])
    return


@app.cell
def _(K, arrays_store, mo, np, plt, sns, state_labels, ui_subjects):
    # ── transition matrix heatmap — marimo grid (3 per row) ──────────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    _COLS = 3
    _figs_t = []
    for _subj in _selected:
        _arr = arrays_store[_subj]
        if "transition_matrix" in _arr:
            _A = _arr["transition_matrix"]
        else:
            _bias = _arr["transition_bias"]  # (K, K)
            _A = np.exp(_bias) / np.exp(_bias).sum(axis=-1, keepdims=True)
        _slbl = state_labels.get(_subj, {k: f"S{k}" for k in range(K)})
        _tick_labels = [_slbl.get(k, f"S{k}") for k in range(K)]
        _fig_t, _ax_t = plt.subplots(figsize=(3.2, 2.8))
        sns.heatmap(
            _A,
            ax=_ax_t,
            cmap="bone",
            annot=True, fmt=".2f",
            vmin=0, vmax=1,
            square=True,
            linewidths=0.5,
            xticklabels=_tick_labels,
            yticklabels=_tick_labels,
            cbar_kws={"shrink": 0.8, "label": "probability"},
        )
        _ax_t.set_title(f"Subject {_subj}")
        _ax_t.set_xlabel("To state")
        _ax_t.set_ylabel("From state")
        _fig_t.tight_layout()
        _figs_t.append(_fig_t)
    _rows_t = [
        mo.hstack(_figs_t[i : i + _COLS], justify="start")
        for i in range(0, len(_figs_t), _COLS)
    ]
    mo.vstack([
        mo.md(f"### Transition matrices — bias-only component  (K={K})"),
        *_rows_t,
    ])
    return


@app.cell
def _(arrays_store, mo, ui_subjects):
    # ── trial-window slider (shared across all posterior plots) ──────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    _T_max = (
        max(arrays_store[s]["smoothed_probs"].shape[0] for s in _selected)
        if _selected else 200
    )
    ui_trial_range = mo.ui.range_slider(
        start=0,
        stop=_T_max - 1,
        value=[0, min(_T_max - 1, 199)],
        label="Trial window",
        step=1,
    )
    mo.vstack([mo.md("### Trial window"), ui_trial_range])
    return (ui_trial_range,)


@app.cell
def _(
    K,
    arrays_store,
    is_2afc,
    mo,
    plots,
    state_labels,
    ui_subjects,
    ui_trial_range,
    views,
):
    # ── posterior state probabilities ─────────────────────────────────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))
    _t0, _t1 = ui_trial_range.value
    if is_2afc:
        _fig_p = plots.plot_posterior_probs(
            views={s: views[s] for s in _selected},
            K=K,
            t0=_t0,
            t1=_t1,
        )
    else:
        _fig_p = plots.plot_posterior_probs(
            arrays_store=arrays_store,
            state_labels=state_labels,
            K=K,
            subjects=_selected,
            t0=_t0,
            t1=_t1,
        )
    mo.vstack([
        mo.md(f"### Posterior state probabilities  (K={K})"),
        ui_trial_range,
        _fig_p,
    ], align="center")
    return

@app.cell
def _(mo):
    ui_psychometric_background = mo.ui.radio(
        options={
            "Data traces": "data",
            "Model curves": "model",
            "None": "none",
        },
        value="data",
        inline=False,
        label="Psychometric background",
    )
    ui_psychometric_background
    return (ui_psychometric_background,)


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
        options={_feat.replace("_", " ").title(): _feat for _feat in _feature_names},
        value=_default_feature,
        label="Regressor",
    )
    ui_psychometric_regressor
    return (ui_psychometric_regressor,)


@app.cell
def _(K, is_2afc, mo, pl, plots, trial_df, ui_psychometric_background, ui_psychometric_regressor, ui_subjects, views):
    _selected = [s for s in ui_subjects.value if s in views]
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
    )
    _reg_plot_fn = getattr(plots, "plot_regressor_psychometric_by_state", None)
    if is_2afc and _reg_plot_fn is not None:
        _fig_reg_state, _ = _reg_plot_fn(
            df=_plot_df_state,
            views=_views_sel,
            model_name=f"glmhmmt K={K}",
            feature_col=ui_psychometric_regressor.value,
            background_style=ui_psychometric_background.value,
        )
        _reg_section = mo.vstack([
            mo.hstack([mo.md("### Per-state psychometric by regressor"), ui_psychometric_regressor], justify="space-between"),
            _fig_reg_state,
        ], align="center")
    else:
        _reg_section = mo.md("This task does not expose a regressor psychometric plot.")

    mo.vstack([
        mo.md("### Categorical plots for accuracy"),
        mo.hstack([mo.vstack([_fig_all], align="center"), mo.vstack([ui_psychometric_background], align="center")], justify="space-between", align="center", widths=[4,1]),
        mo.md("### Per-state categorical performance"),
        _fig_state,
        _reg_section,
    ], align="center")
    return


@app.cell
def _(editor_views, mo):
    _subjects = list(editor_views.keys())
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
    ui_psychometric_background,
    ui_psychometric_regressor,
    ui_editor_subject,
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
        )
        _reg_section = mo.vstack(
            [
                mo.hstack([mo.md("### Tweaked per-state psychometric by regressor"), ui_psychometric_regressor], justify="space-between"),
                _fig_reg_state_tweaked,
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
            mo.hstack([mo.vstack([_fig_all_tweaked], align="center"), mo.vstack([ui_psychometric_background], align="center")], justify="space-between", align="center", widths=[4,1]),
            mo.md("### Tweaked per-state categorical performance"),
            _fig_state_tweaked,
            _reg_section,
            _side_section,
        ],
        align="center",
    )
    return


@app.cell
def _(K, mo, plots, ui_subjects, views):
    _selected = [s for s in ui_subjects.value if s in views]
    mo.stop(
        not _selected or getattr(views.get(_selected[0]), "transition_weights", None) is None,
        mo.md("No transition weights found — run the glmhmm-t fit first."),
    )
    _fig_line, _fig_box = plots.plot_transition_weights(views=views, K=K, subjects=_selected)
    mo.vstack([
        mo.md("### Transition weights"),
        mo.hstack([_fig_line, _fig_box]),
    ])
    return


@app.cell
def _(arrays_store, names):
    def _(K, mo, plots, ui_subjects, views):
        # ── Input-dependent transition weights ────────────────────────────────────
        _selected = [s for s in ui_subjects.value if s in arrays_store]
        _selected = [s for s in ui_subjects.value if s in views]
        mo.stop(
            not _selected or "transition_weights" not in arrays_store.get(_selected[0], {}),
            not _selected or getattr(views.get(_selected[0]), "transition_weights", None) is None,
            mo.md("No transition weights found — run the glmhmm-t fit first."),
        )
        _fig_line, _fig_box, _fig_std, _fig_raw = plots.plot_transition_weights(
            arrays_store=arrays_store,
            names=names,
            K=K,
            subjects=_selected,
        )
        _fig_line, _fig_box = plots.plot_transition_weights(views=views, K=K, subjects=_selected)
        mo.vstack([
            mo.md("### Transition weights"),
            mo.hstack([_fig_line, _fig_box]),
            _fig_std,
            _fig_raw,
        ])
        return

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
def _(THRESH_ui, adapter, mo, plots, trial_df, ui_subjects, views):
    _selected = [s for s in ui_subjects.value if s in views]
    mo.stop(not _selected, mo.md("No fitted subjects available."))
    _fig_acc, _tbl = plots.plot_state_accuracy(
        views={s: views[s] for s in _selected},
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
        options=df_all["subject"].unique().to_list(),
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
def _(K, mo, plots, trial_df, ui_subjects_traj, views):
    # ── d. Fractional occupancy & state-change histogram ─────────────────────
    _selected_occ = [s for s in ui_subjects_traj.value if s in views]
    mo.stop(not _selected_occ, mo.md("Select subjects above."))
    _fig_occ = plots.plot_state_occupancy(
        views={s: views[s] for s in _selected_occ},
        trial_df=trial_df,
        session_col="session",
        sort_col="trial_idx",
    )
    mo.vstack([
        mo.md(f"### d. Fractional occupancy & state changes per session  (K={K})"),
        mo.md(
            "> **Top row**: all selected subjects pooled. Left = posterior fractional occupancy boxplot by state; "
            "middle = per-session occupancy pooled across subjects; right = histogram of state switches per session.  \n"
            "> **Rows below**: one row per subject. Left = posterior mean occupancy by state; middle = per-session "
            "occupancy boxplots; right = histogram of inferred state switches per session."
        ),
        _fig_occ,
    ], align="center")
    return


@app.cell
def _(mo, ui_subjects, views):
    # ── Session deep-dive controls ─────────────────────────────────────────────
    _selected = [s for s in ui_subjects.value if s in views]
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
def _(K, mo, plots, trial_df, ui_session_id, ui_session_subj, views):
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
    )
    mo.vstack([
        mo.md(f"### Session deep-dive  (K={K})"),
        _fig,
    ], align="center")
    return


@app.cell
def _(K, current_hash, mo, paths, plots, ui_alias, ui_subjects):
    # ── τ sweep analysis ────────────────────────────────────────────────────────
    _sweep_path = paths.RESULTS / "fits" / "tau_sweep" / f"glmhmmt_K{K}" / "tau_sweep_summary.parquet"
    mo.stop(
        not _sweep_path.exists(),
        mo.md(
            f"**τ sweep results not found.**  \
     Run the sweep first:\n```\n"
            f"uv run python scripts/fit_tau_sweep.py --model glmhmmt --K {K}\n```"
        ),
    )
    _subjects = list(ui_subjects.value)
    _fig_sweep, _best = plots.plot_tau_sweep(
        sweep_path=_sweep_path,
        subjects=_subjects,
        K=K,
    )
    mo.vstack([
        mo.md(f"### τ sweep results — {ui_alias.value or current_hash} K={K}"),
        _fig_sweep,
        mo.md("**Best τ per subject (min BIC):**"),
        mo.plain_text(_best.to_pandas().to_string(index=False)),
    ], align="center")
    return


if __name__ == "__main__":
    app.run()
