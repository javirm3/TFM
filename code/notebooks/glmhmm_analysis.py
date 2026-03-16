import marimo

__generated_with = "0.20.4"
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
        from scripts.fit_glmhmm import main as fit_main
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
        build_editor_payload,
    )

    sns.set_style("white")
    return (
        CoefficientEditorWidget,
        ModelManagerWidget,
        apply_state_tweak_to_trial_df,
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
        model_type="glmhmm",
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
    from scripts.fit_glmhmm import generate_model_id as _gen_id

    class _V:
        def __init__(self, value):
            self.value = value

    _val = ui_model_manager.value
    current_hash = _gen_id(task_name, _val["K"], _val["tau"], _val["emission_cols"])
    ui_existing = _V(None if _val.get("existing_model") in ("", "__default__") else _val.get("existing_model"))
    ui_alias = _V(_val.get("alias", ""))
    ui_K = _V(_val["K"])
    ui_subjects = _V(_val["subjects"])
    ui_tau = _V(_val["tau"])
    ui_emission_cols = _V(_val["emission_cols"])
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
        ui_emission_cols,
        ui_existing,
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
    ui_emission_cols,
    ui_existing,
    ui_subjects,
    ui_tau,
):
    _last_fit_click = get_last_fit_click()
    mo.stop(
        fit_clicks <= _last_fit_click,
        mo.md("Configure parameters and press **Run fit**."),
    )
    set_last_fit_click(fit_clicks)

    _n_restarts = 5

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

    _total_progress = max(1, len(ui_subjects.value) * _n_restarts)
    mm_widget.is_running = True
    try:
        with mo.status.progress_bar(
            total=_total_progress,
            title=f"Fitting GLM-HMM K={ui_K.value}",
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
                tau=ui_tau.value,
                emission_cols=ui_emission_cols.value,
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
):
    K = ui_K.value

    selected_model_id = ui_existing.value or (ui_alias.value if ui_alias.value else current_hash)
    OUT = paths.RESULTS / "fits" / task_name / "glmhmm" / selected_model_id
    # load feature names via adapter
    _df_sel = df_all.filter(pl.col("subject").is_in(ui_subjects.value)).sort(adapter.sort_col)
    _, _, _, names = adapter.load_subject(_df_sel, tau=ui_tau.value, emission_cols=ui_emission_cols.value)

    arrays_store = {}
    _files = list(sorted(OUT.glob("*_glmhmm_arrays.npz")))
    _files += [f for f in sorted(OUT.glob(f"*_K{K}_glmhmm_arrays.npz")) if f not in _files]
    for _f in _files:
        _subj = _f.name.removesuffix("_glmhmm_arrays.npz").removesuffix(f"_K{K}")
        _d = dict(np.load(_f, allow_pickle=True))
        # decode column names saved as string arrays; fall back to adapter output
        _d["X_cols"] = (
            list(_d["X_cols"]) if "X_cols" in _d else names["X_cols"]
        )
        arrays_store[_subj] = _d

    # arrays_store
    return K, arrays_store, names


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
    # ── Build SubjectFitViews + derive state_labels / state_order for backward compat ──
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    if hasattr(adapter, "scoring_key"):
        adapter.scoring_key = ui_scoring_key.value
    views = build_views(arrays_store, adapter, K, _selected)
    state_labels = {s: v.state_name_by_idx for s, v in views.items()}
    state_order  = {s: v.state_idx_order   for s, v in views.items()}
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
            .filter(pl.col(_ses_col).count().over(_ses_col) >= 2)
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
    if is_2afc:
        _fig_ag, _fig_cls = plots.plot_emission_weights(
            views={s: views[s] for s in _selected}, K=K, save_path=_save_path,
        )
    else:
        _fig_ag, _fig_cls = plots.plot_emission_weights(
            arrays_store=arrays_store, state_labels=state_labels, names=names,
            K=K, subjects=_selected, save_path=_save_path,
        )
    mo.vstack([mo.md("### Emission weights"), _fig_ag, _fig_cls])
    return


@app.cell
def _(K, arrays_store, mo, plt, sns, state_labels, ui_subjects):
    # ── transition matrix heatmap — marimo grid (3 per row) ──────────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    _COLS = 3
    _figs_t = []
    for _subj in _selected:
        _A = arrays_store[_subj]["transition_matrix"]  # (K, K)
        _slbl = state_labels.get(_subj, {k: f"S{k}" for k in range(K)})
        _tick_labels = [_slbl.get(k, f"S{k}") for k in range(K)]
        _fig_t, _ax_t = plt.subplots(figsize=(3.2, 2.8))
        sns.heatmap(_A, ax=_ax_t, cmap="bone", annot=True, fmt=".2f", vmin=0, vmax=1, square=True, linewidths=0.5, xticklabels=_tick_labels,     
                    yticklabels=_tick_labels, cbar_kws={"shrink": 0.8, "label": "probability"},)
        _ax_t.set_title(f"Subject {_subj}")
        _ax_t.set_xlabel("To state")
        _ax_t.set_ylabel("From state")
        _fig_t.tight_layout()
        _figs_t.append(_fig_t)
    _rows_t = [
        mo.hstack(_figs_t[i : i + _COLS], justify="start")
        for i in range(0, len(_figs_t), _COLS)
    ]
    mo.vstack(
        [
            mo.md(f"### Transition matrices  (K={K})"),
            *_rows_t,
        ]
    )
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
def _(K, is_2afc, mo, pl, plots, trial_df, ui_subjects, views):
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
        **_perf_kwargs,
    )

    _plot_df_state = plots.prepare_predictions_df(_trial_df_sel)
    _fig_state, _ = plots.plot_categorical_performance_by_state(
        df=_plot_df_state,
        views=_views_sel,
        model_name=f"glmhmm K={K} — per state",
    )

    mo.vstack(
        [
            mo.md("### Categorical plots for accuracy"),
            _fig_all,
            mo.md("### Per-state categorical performance"),
            _fig_state,
        ],
        align="center",
    )
    return


@app.cell
def _(editor_views, mo):
    _subjects = list(editor_views.keys())
    mo.stop(not _subjects, mo.md("No fitted subjects available for coefficient editing."))
    ui_editor_subject = mo.ui.dropdown(
        options=_subjects,
        value=_subjects[0],
        label="Coefficient editor subject",
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
        label="Editable state",
    )
    return (ui_editor_state,)


@app.cell
def _(
    CoefficientEditorWidget,
    adapter,
    build_editor_payload,
    editor_views,
    mo,
    np,
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
    _stored_class_indices = list(range(_view.num_classes - 1))
    _reference_class_idx = _view.num_classes - 1
    _display_reference_class_idx = 1 if _view.num_classes == 3 else _reference_class_idx
    _payload = build_editor_payload(
        _stored_weights,
        choice_labels=list(adapter.choice_labels),
        stored_class_indices=_stored_class_indices,
        reference_class_idx=_reference_class_idx,
        display_reference_class_idx=_display_reference_class_idx,
    )

    coef_editor = mo.ui.anywidget(
        CoefficientEditorWidget(
            title=f"{_subj} · {coef_state_label}",
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

    mo.vstack(
        [
            mo.md("### Interactive coefficient editor"),
            mo.md(
                "Only the selected state's emission coefficients are edited. "
                "The overall and per-state categorical plots update using the edited state."
            ),
            mo.hstack([ui_editor_subject, ui_editor_state]),
            coef_editor,
        ],
        align="center",
    )
    coef_editor_explicit_class_indices = _payload["explicit_class_indices"]
    coef_editor_reference_class_idx = _payload["reference_class_idx"]
    return (
        coef_editor,
        coef_editor_explicit_class_indices,
        coef_editor_reference_class_idx,
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
    coef_editor,
    coef_editor_explicit_class_indices,
    coef_editor_reference_class_idx,
    coef_state_idx,
    coef_state_label,
    editor_trial_df,
    editor_view,
    mo,
    np,
    plots,
    ui_editor_subject,
):
    _subj = ui_editor_subject.value
    _view = editor_view
    _trial_df_sub = editor_trial_df

    _trial_df_tweaked = apply_state_tweak_to_trial_df(
        _trial_df_sub,
        adapter=adapter,
        view=_view,
        state_idx=coef_state_idx,
        edited_weights=np.asarray(coef_editor.value["weights"], dtype=float),
        original_weights=np.asarray(coef_editor.value["original_weights"], dtype=float),
        explicit_class_indices=list(coef_editor_explicit_class_indices),
        reference_class_idx=int(coef_editor_reference_class_idx),
    )
    _plot_df_tweaked = plots.prepare_predictions_df(_trial_df_tweaked)

    _title = f"{_subj} — tweaked {coef_state_label}"
    _fig_all_tweaked, _ = plots.plot_categorical_performance_all(
        _plot_df_tweaked,
        _title,
    )
    _fig_state_tweaked, _ = plots.plot_categorical_performance_by_state(
        df=_plot_df_tweaked,
        views={_subj: _view},
        model_name=f"{_title} — per state",
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
            _fig_all_tweaked,
            mo.md("### Tweaked per-state categorical performance"),
            _fig_state_tweaked,
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
        options=df_all["subject"].unique().to_list(),
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
def _(K, mo, plots, trial_df, ui_subjects_traj, views):
    selected_occ = [s for s in ui_subjects_traj.value if s in views]
    mo.stop(not selected_occ, mo.md("Select subjects above."))
    _fig_occ = plots.plot_state_occupancy(
        views={s: views[s] for s in selected_occ},
        trial_df=trial_df,
        session_col="session",
        sort_col="trial_idx",
    )
    mo.vstack([
        mo.md(f"### d. Fractional occupancy & state changes per session  (K={K})"),
        _fig_occ,
        mo.md(
            "> **Left**: fraction of all trials assigned to each state (argmax of posterior).  \n"
            "> **Middle**: per-session occupancy boxplots for each state.  \n"
            "> **Right**: histogram of inferred state changes per session."
        ),
    ], align="center")

    trial_df
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
            "SSM uses a different EM implementation (no custom stickiness prior, "
            "standard Baum-Welch) which often yields smoother posteriors — useful as a sanity check."
        ),
        ssm_run_btn,
    ])
    return (ssm_run_btn,)


@app.cell
def _(
    K,
    adapter,
    df_all,
    mo,
    np,
    pl,
    plt,
    sns,
    ssm_run_btn,
    ui_subjects,
    ui_trial_range,
    views,
):

    # ── SSM fit + posterior plot ───────────────────────────────────────────────
    mo.stop(not ssm_run_btn.value, mo.md("Press **▶ Run SSM safety check** above to fit."))

    import ssm as _ssm_lib
    from scripts.fit_glmhmm import _valid_trial_mask as _vtm

    _STIM_NAMES_SSM = {"stim_vals", "stim_d", "ild_norm", "ILD", "ild",
                       "stimulus", "net_ild", "stim_strength"}

    _ssm_subjects = [s for s in ui_subjects.value if s in views]
    mo.stop(not _ssm_subjects, mo.md("No fitted arrays found — run the custom fit first."))

    _ssm_results = {}

    with mo.status.spinner(title="Fitting SSM GLM-HMM…"):
        for _subj in _ssm_subjects:
            _view = views[_subj]
            _X   = np.asarray(_view.X)   # (T, n_feat) — already session-filtered
            _y   = np.asarray(_view.y)   # (T,)

            # Reconstruct session ids with same mask as fit_subject()
            _df_s    = df_all.filter(pl.col("subject") == _subj).sort(adapter.sort_col)
            _sess_raw = _df_s[adapter.session_col].to_numpy()
            _mask_s  = _vtm(_sess_raw)
            _sess_ids = _sess_raw[_mask_s]

            # Split into per-session lists — SSM expects list of arrays
            _uniq_sess = list(dict.fromkeys(_sess_ids.tolist()))
            _choices_list, _inputs_list = [], []
            for _sid in _uniq_sess:
                _idx = np.where(_sess_ids == _sid)[0]
                _choices_list.append(_y[_idx].reshape(-1, 1).astype(int))
                _inputs_list.append(_X[_idx].astype(float))

            # Initialise and fit
            _obs_dim   = 1
            _n_cats    = 2
            _n_feat    = _X.shape[1]
            _glmhmm_s  = _ssm_lib.HMM(
                K, _obs_dim, _n_feat,
                observations="input_driven_obs",
                observation_kwargs=dict(C=_n_cats),
                transitions="standard",
            )
            _glmhmm_s.fit(
                _choices_list, inputs=_inputs_list,
                method="em", num_iters=200, tolerance=1e-4,
            )

            # Extract quantities
            _W_ssm    = -_glmhmm_s.observations.params          # (K, C-1, n_feat); flip sign
            _trans_ssm = _glmhmm_s.transitions.transition_matrix  # (K, K)
            _gamma_ssm = np.vstack([
                _glmhmm_s.expected_states(data=d, input=inp)[0]
                for d, inp in zip(_choices_list, _inputs_list)
            ])  # (T, K)

            # Identify "Engaged" state: highest |stim weight| (W[:, 0, stim_idx])
            _feat_names_s = list(_view.feat_names)
            _stim_idx_s   = next(
                (i for i, n in enumerate(_feat_names_s) if n in _STIM_NAMES_SSM), None
            )
            if _stim_idx_s is not None and _W_ssm.ndim == 3:
                _scores_s = np.abs(_W_ssm[:, 0, _stim_idx_s])
            else:
                _scores_s = np.zeros(K)
            _engaged_ssm = int(np.argmin(_scores_s))

            _ssm_results[_subj] = {
                "gamma": _gamma_ssm,
                "W":     _W_ssm,
                "trans": _trans_ssm,
                "y":     _y,
                "engaged_k": _engaged_ssm,
            }

    # ── Plot: SSM posterior vs custom posterior (Engaged state) ──────────────
    _n_s   = len(_ssm_subjects)
    _t0_s, _t1_s = ui_trial_range.value
    _fig_ssm, _axes_ssm = plt.subplots(
        _n_s, 1, figsize=(14, 3 * _n_s), squeeze=False
    )

    for _i, _subj in enumerate(_ssm_subjects):
        _ax = _axes_ssm[_i, 0]

        # SSM engaged posterior (already identified above)
        _g_ssm = _ssm_results[_subj]["gamma"][_t0_s:_t1_s + 1]   # (window, K)
        _ek    = _ssm_results[_subj]["engaged_k"]
        _p_ssm_eng = _g_ssm[:, _ek]

        # Custom model's Engaged state for this subject
        _view = views[_subj]
        _ek_custom = _view.engaged_k()
        _g_custom = np.asarray(_view.smoothed_probs)[_t0_s:_t1_s + 1]
        _p_custom_eng = _g_custom[:, _ek_custom]

        _x_w = np.arange(_t0_s, _t0_s + len(_p_ssm_eng))

        _ax.plot(_x_w, _p_custom_eng, color="steelblue",  lw=1.2, alpha=0.85, label="Custom (stickiness)")
        _ax.plot(_x_w, _p_ssm_eng,    color="darkorange", lw=1.2, alpha=0.85, linestyle="--", label="SSM (standard)")

        # Choice rug
        _y_w = np.asarray(_view.y)[_t0_s:_t1_s + 1].astype(int)
        for _resp, _col, _lbl in [(0, "royalblue", "L"), (1, "gold", "R")]:
            _m = _y_w == _resp
            _ax.scatter(
                _x_w[_m], np.ones(_m.sum()) * 1.03,
                c=_col, s=4, marker="|",
                transform=_ax.get_xaxis_transform(), clip_on=False,
            )

        _ax.set_xlim(_t0_s, _t0_s + len(_p_ssm_eng) - 1)
        _ax.set_ylim(0, 1)
        _ax.set_ylabel("P(Engaged)")
        _ax.set_title(f"Subject {_subj}  — SSM state {_ek} vs Custom state {_ek_custom}")
        _ax.legend(fontsize=8, frameon=False, loc="upper right")

    _axes_ssm[-1, 0].set_xlabel("Trial")
    _fig_ssm.tight_layout()
    sns.despine(fig=_fig_ssm)

    mo.vstack([
        mo.md(f"### SSM GLM-HMM sanity check — P(Engaged)  (K={K})"),
        mo.md(
            "**Blue** = custom model posterior (with transition stickiness prior).  \n"
            "**Dashed orange** = SSM posterior (standard Baum-Welch, no stickiness).  \n"
            "SSM typically yields smoother posteriors because it lacks the stickiness "
            "prior that keeps the custom model in its current state, and because SSM's "
            "EM runs unconstrained for longer. Large discrepancies may indicate the "
            "stickiness prior is over-regularising state transitions."
        ),
        _fig_ssm,
    ], align="center")
    return


if __name__ == "__main__":
    app.run()
