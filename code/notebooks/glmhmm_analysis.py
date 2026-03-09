import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys, os
    from pathlib import Path

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import paths
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scripts.fit_glmhmm import main as fit_main
    from tasks import get_adapter
    from glmhmmt.views import build_views
    from glmhmmt.postprocess import build_trial_df, build_emission_weights_df
    from widgets import ModelManagerWidget

    sns.set_style("white")

    ui_task = mo.ui.dropdown(
        options=["2AFC", "MCDR"],
        value="MCDR",
        label="Task:",

    )
    ui_task
    return (
        ModelManagerWidget,
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
        ui_task,
    )


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
    _fits_path = paths.RESULTS / "fits" / ui_task.value / "glmhmm"
    _existing_opts = []
    return


@app.cell
def _(paths, ui_task):
    import json as _json
    return


@app.cell
def _(ModelManagerWidget, adapter, df_all, mo, ui_task):
    # ── controls ───────────────────────────────────────────────────────────────
    is_2afc = adapter.num_classes == 2
    _subjects = df_all["subject"].unique().to_list()

    mm_widget = ModelManagerWidget(
        model_type="glmhmm",
        task=ui_task.value,
        is_2afc=is_2afc,
        subjects=_subjects,
        K=2,
        tau=50,
        lapse=False,
        lapse_max=0.2,
    )

    ui_model_manager = mo.ui.anywidget(mm_widget)
    ui_model_manager

    return is_2afc, ui_model_manager


@app.cell
def _(mo, ui_model_manager, ui_task):
    from scripts.fit_glmhmm import generate_model_id as _gen_id
    
    _val = ui_model_manager.value
    # Use selected parameter traits to generate current hash string
    current_hash = _gen_id(ui_task.value, _val["K"], _val["tau"], _val["emission_cols"])

    return current_hash,


@app.cell
def _(
    current_hash,
    fit_main,
    mo,
    paths,
    ui_model_manager,
    ui_task,
):
    _val = ui_model_manager.value
    _clicks = _val["run_fit_clicks"]

    mo.stop(
        _clicks == 0, mo.md("Configure parameters and press **RUN FIT 🚀**.")
    )
    _selected_id = _val["existing_model"] or (_val["alias"] if _val["alias"] else current_hash)
    _OUT = paths.RESULTS / "fits" / ui_task.value / "glmhmm" / _selected_id
    with mo.status.spinner(
        title=f"Fitting GLM-HMM K={_val['K']} τ={_val['tau']} for {_val['subjects']}..."
    ):
        fit_main(
                subjects=_val["subjects"],
                K_list=[_val["K"]],
                out_dir=_OUT,
                tau=_val["tau"],
                emission_cols=_val["emission_cols"],
                task=ui_task.value,
                n_restarts=1,
            )
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
    ui_model_manager,
    ui_task,
):
    _val = ui_model_manager.value
    K = _val["K"]
    _selected_subjects = _val["subjects"]

    selected = _selected_subjects
    selected_model_id = _val["existing_model"] or (_val["alias"] if _val["alias"] else current_hash)
    OUT = paths.RESULTS / "fits" / ui_task.value / "glmhmm" / selected_model_id
    # load feature names via adapter
    _df_sel = df_all.filter(pl.col("subject").is_in(_selected_subjects)).sort(adapter.sort_col)
    _, _, _, names = adapter.load_subject(_df_sel, tau=_val["tau"], emission_cols=_val["emission_cols"])

    arrays_store = {}
    for _subj in _selected_subjects:
        _f = OUT / f"{_subj}_K{K}_glmhmm_arrays.npz"
        if _f.exists():
            _d = dict(np.load(_f, allow_pickle=True))
            # decode column names saved as string arrays; fall back to adapter output
            _d["X_cols"] = (
                list(_d["X_cols"]) if "X_cols" in _d else names["X_cols"]
            )
            arrays_store[_subj] = _d

    # arrays_store
    return K, arrays_store, names


@app.cell
def _(K, adapter, arrays_store, build_views, mo, ui_model_manager):
    # ── Build SubjectFitViews + derive state_labels / state_order for backward compat ──
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    views = build_views(arrays_store, adapter, K, _selected)
    state_labels = {s: v.state_name_by_idx for s, v in views.items()}
    state_order  = {s: v.state_idx_order   for s, v in views.items()}
    return state_labels, state_order, views


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
        _trial_frames.append(build_trial_df(_view, _df_sub, _bcols))

    mo.stop(not _trial_frames, mo.md("No subjects with matching data lengths."))
    trial_df = pl.concat(_trial_frames)

    # Emit emission-weights long DF for downstream use
    weights_df = build_emission_weights_df(views)
    return


@app.cell
def _(K, arrays_store, mo, names, paths, plots, state_labels, ui_model_manager):
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

    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))
    _fig_ag, _fig_cls = plots.plot_emission_weights( arrays_store=arrays_store, state_labels=state_labels, names=names, K=K, subjects=_selected, save_path=paths.RESULTS / "plots/GLMHMM/emissions_coefs.png",)
    mo.vstack([mo.md("### Emission weights"), _fig_ag, _fig_cls])
    return


@app.cell
def _(K, arrays_store, mo, plt, sns, state_labels, ui_model_manager):
    # ── transition matrix heatmap — marimo grid (3 per row) ──────────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
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
def _(arrays_store, mo, ui_model_manager):
    # ── trial-window slider (shared across all posterior plots) ──────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    _T_max = ( max(arrays_store[s]["smoothed_probs"].shape[0] for s in _selected) if _selected else 200 )
    ui_trial_range = mo.ui.range_slider( start=0, stop=_T_max - 1, value=[0, min(_T_max - 1, 199)], label="Trial window", step=1,)
    mo.vstack([mo.md("### Trial window"), ui_trial_range])
    return (ui_trial_range,)


@app.cell
def _(
    K,
    arrays_store,
    mo,
    np,
    plt,
    sns,
    state_labels,
    ui_model_manager,
    ui_trial_range,
):
    # ── posterior state probabilities ─────────────────────────────────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _t0, _t1 = ui_trial_range.value
    _n_subj = len(_selected)
    _fig_p, _axes_p = plt.subplots( _n_subj, 1, figsize=(14, 3 * _n_subj), squeeze=False )

    for _i, _subj in enumerate(_selected):
        _ax = _axes_p[_i, 0]
        _probs = arrays_store[_subj]["smoothed_probs"][_t0 : _t1 + 1]  # (window, K)
        _y = arrays_store[_subj]["y"].astype(int)[_t0 : _t1 + 1]  # (window,)
        _T_w = _probs.shape[0]
        _x = np.arange(_t0, _t0 + _T_w)

        # stacked area — color by label rank so Engaged is always palette[0]
        _colors = sns.color_palette("tab10", n_colors=K)
        _label_rank = { "Engaged": 0, "Disengaged": 1, **{f"Disengaged {i}": i for i in range(1, K)},}
        _bottom = np.zeros(_T_w)
        _slbl = state_labels.get(_subj, {k: f"State {k}" for k in range(K)})
        for _k in range(K):
            _rank = _label_rank.get(_slbl.get(_k, ""), _k)
            _ax.fill_between( _x, _bottom, _bottom + _probs[:, _k], alpha=0.7, color=_colors[_rank], label=_slbl.get(_k, f"State {_k}"),)
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
def _(
    K,
    adapter,
    arrays_store,
    df_all,
    is_2afc,
    mo,
    np,
    pl,
    plots,
    ui_model_manager,
    views,
):
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _sort_col = adapter.sort_col
    _ses_col = adapter.session_col

    _frames = []
    for _subj in _selected:
        _p_pred = arrays_store[_subj]["p_pred"]
        _n_classes = _p_pred.shape[1]
        _df_sub = (
            df_all.filter(pl.col("subject") == _subj)
            .sort(_sort_col)
            .filter(pl.col(_ses_col).count().over(_ses_col) >= 2)
        )
        _cols = [pl.Series("pred_choice", np.argmax(_p_pred, axis=1).astype(int))]
        if _n_classes == 2:
            _cols += [pl.Series("pL", _p_pred[:, 0]), pl.Series("pR", _p_pred[:, 1])]
        else:
            _cols += [
                pl.Series("pL", _p_pred[:, 0]),
                pl.Series("pC", _p_pred[:, 1]),
                pl.Series("pR", _p_pred[:, 2]),
            ]
        if len(_df_sub) == len(_p_pred):
            _frames.append(_df_sub.with_columns(_cols))
        else:
            print(f"\u26a0\ufe0f  {_subj}: length mismatch ({len(_df_sub)} vs {len(_p_pred)}), skipping")

    mo.stop(not _frames, mo.md("No subjects with matching data lengths."))
    _df_all_pred = pl.concat(_frames)
    _plot_df = plots.prepare_predictions_df(_df_all_pred)
    _perf_kwargs = {"arrays_store": arrays_store} if is_2afc else {}
    _fig_all, _ = plots.plot_categorical_performance_all(_plot_df, f"glmhmm K={K}", **_perf_kwargs)

    # ── per-state pool with normalised rank indices ───────────────────────────
    _lrank_map = {"Engaged": 0, "Disengaged": 1, **{f"Disengaged {i}": i for i in range(1, K)}}
    _pool_dfs = []
    _pool_assigns = []
    for _subj in _selected:
        _p_pred_s = arrays_store[_subj]["p_pred"]
        _nc_s = _p_pred_s.shape[1]
        _pred_cols_s = [pl.Series("pred_choice", np.argmax(_p_pred_s, axis=1).astype(int))]
        if _nc_s == 2:
            _pred_cols_s += [pl.Series("pL", _p_pred_s[:, 0]), pl.Series("pR", _p_pred_s[:, 1])]
        else:
            _pred_cols_s += [
                pl.Series("pL", _p_pred_s[:, 0]),
                pl.Series("pC", _p_pred_s[:, 1]),
                pl.Series("pR", _p_pred_s[:, 2]),
            ]
        _df_s = (
            df_all.filter(pl.col("subject") == _subj)
            .sort(_sort_col)
            .filter(pl.col(_ses_col).count().over(_ses_col) >= 2)
            .with_columns(_pred_cols_s)
        )
        if len(_df_s) != _p_pred_s.shape[0]:
            continue
        _plot_df_s = plots.prepare_predictions_df(_df_s)
        _gamma_s = arrays_store[_subj]["smoothed_probs"]
        _T_s = _gamma_s.shape[0]
        _map_k = np.argmax(_gamma_s, axis=1).astype(int)

        # ── per-state emission: replace marginal p_pred with state-specific P ──
        _W = np.asarray(arrays_store[_subj]["emission_weights"])  # (K, C-1, M)
        _X_s = np.asarray(arrays_store[_subj]["X"])               # (T, M)
        if is_2afc:
            # Binary: W[k,0,:] = logit for P(LEFT); P(right|k,t) = sigmoid(-logit)
            _logit_left = np.einsum("km,tm->tk", _W[:, 0, :], _X_s)  # (T, K)
            _p_right_per_state = 1.0 / (1.0 + np.exp(_logit_left))    # (T, K) P(right|state)
            _p_state_pR = _p_right_per_state[np.arange(_T_s), _map_k]  # (T,)
            _p_state_pL = 1.0 - _p_state_pR
            _plot_df_s = _plot_df_s.with_columns([
                pl.Series("pR", _p_state_pR.astype(np.float64)),
                pl.Series("pL", _p_state_pL.astype(np.float64)),
                pl.Series("p_pred", _p_state_pR.astype(np.float64)),
            ])
        else:
            _logits = np.einsum("kci,ti->tkc", _W, _X_s)
            _nc_logits = _logits.shape[2] + 1
            if _nc_logits == 2:
                _logits_full = np.concatenate([_logits, np.zeros((_T_s, K, 1))], axis=2)
            else:
                _logits_full = np.concatenate(
                    [_logits, np.zeros((_T_s, K, 1))],
                    axis=2,
                )
            _lse = _logits_full.max(axis=2, keepdims=True)
            _exp = np.exp(_logits_full - _lse)
            _p_state = _exp / _exp.sum(axis=2, keepdims=True)
            _stim = _plot_df_s["stimulus"].to_numpy().astype(int)
            _p_state_correct = _p_state[np.arange(_T_s), _map_k, _stim]
            _plot_df_s = _plot_df_s.with_columns(
                pl.Series("p_model_correct", _p_state_correct.astype(np.float64))
            )

        _pool_dfs.append(_plot_df_s)
        rank_by_idx = views[_subj].state_rank_by_idx
        _norm = np.array([rank_by_idx[int(k)] for k in _map_k], dtype=int)
        _pool_assigns.append(_norm)

    _df_state_pool = pl.concat(_pool_dfs)
    _assign_pool = np.concatenate(_pool_assigns)
    _state_lbl_global = {0: "Engaged", 1: "Disengaged", **{i: f"Disengaged {i}" for i in range(2, K)}}
    _fig_state, _ = plots.plot_categorical_performance_by_state(
        df=_df_state_pool,
        smoothed_probs=None,
        state_assign=_assign_pool,
        state_labels=_state_lbl_global,
        model_name=f"glmhmm K={K} \u2014 per state",
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
    K,
    THRESH_ui,
    adapter,
    arrays_store,
    df_all,
    mo,
    plots,
    state_labels,
    ui_model_manager,
):
    # ── Per-state accuracy — Ashwood et al. 2022 method ────────────────────────────────────────────────
    # All     : mean(performance) on nonzero-stim trials — the full pool
    # State k : mean(performance) on the SUBSET where posterior[:,k] >= thresh
    #           AND stimd_n != 0
    # "All" is the weighted average of the state bars (plus ambiguous trials).
    # Colors assigned by rank: Engaged=palette[0], Disengaged=palette[1], …

    _val = ui_model_manager.value
    _selected_acc = [s for s in _val["subjects"] if s in arrays_store]
    mo.stop(not _selected_acc, mo.md("No fitted subjects available."))
    _fig_acc, _tbl = plots.plot_state_accuracy(
        arrays_store=arrays_store,
        state_labels=state_labels,
        df_all=df_all,
        K=K,
        subjects=_selected_acc,
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
def _(
    K,
    adapter,
    arrays_store,
    df_all,
    mo,
    plots,
    state_labels,
    ui_subjects_traj,
):
    selected_traj = [s for s in ui_subjects_traj.value if s in arrays_store]
    mo.stop(not selected_traj, mo.md("Select subjects above to view session trajectories."))
    _fig_traj = plots.plot_session_trajectories(
        arrays_store=arrays_store,
        state_labels=state_labels,
        df_all=df_all,
        K=K,
        subjects=selected_traj,
        session_col=adapter.session_col,
        sort_col=adapter.sort_col,
    )
    mo.vstack([
        mo.md(f"### c. Average state-probability trajectories within a session  (K={K})"),
        _fig_traj,
        mo.md("> Mean ± 1 s.e.m. across sessions for the selected subjects."),
    ], align="center")
    return


@app.cell
def _(
    K,
    adapter,
    arrays_store,
    df_all,
    mo,
    plots,
    state_labels,
    ui_subjects_traj,
):
    selected_occ = [s for s in ui_subjects_traj.value if s in arrays_store]
    mo.stop(not selected_occ, mo.md("Select subjects above."))
    _fig_occ = plots.plot_state_occupancy(
        arrays_store=arrays_store,
        state_labels=state_labels,
        df_all=df_all,
        K=K,
        subjects=selected_occ,
        session_col=adapter.session_col,
        sort_col=adapter.sort_col,
    )
    mo.vstack([
        mo.md(f"### d. Fractional occupancy & state changes per session  (K={K})"),
        _fig_occ,
        mo.md(
            "> **Left**: fraction of all trials assigned to each state (argmax of posterior).  \n"
            "> **Right**: histogram of inferred state changes per session."
        ),
    ], align="center")
    return


@app.cell
def _(arrays_store, mo, ui_model_manager):
    # ── Session deep-dive controls ─────────────────────────────────────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    _subj_opts = _selected if _selected else ["(no fitted subjects)"]

    ui_session_subj = mo.ui.dropdown(
        options=_subj_opts,
        value=_subj_opts[0],
        label="Subject",
    )
    return (ui_session_subj,)


@app.cell
def _(adapter, arrays_store, df_all, mo, pl, ui_session_subj):
    _ses_col = adapter.session_col
    _sess_opts = (
        sorted(
            df_all.filter(pl.col("subject") == ui_session_subj.value)
            .filter(pl.col(_ses_col).count().over(_ses_col) >= 2)[_ses_col]
            .unique()
            .to_list()
        )
        if ui_session_subj.value in arrays_store
        else [0]
    )
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
def _(
    K,
    adapter,
    arrays_store,
    df_all,
    mo,
    plots,
    state_labels,
    ui_session_id,
    ui_session_subj,
):
    _subj = ui_session_subj.value
    mo.stop(
        _subj not in arrays_store,
        mo.md("No fitted arrays for this subject — run the fit first."),
    )

    _sess = ui_session_id.value
    _fig = plots.plot_session_deepdive(
        arrays_store=arrays_store,
        state_labels=state_labels,
        df_all=df_all,
        names=arrays_store[_subj],
        K=K,
        subj=_subj,
        sess=_sess,
        session_col=adapter.session_col,
        sort_col=adapter.sort_col,
    )
    mo.vstack([
        mo.md(f"### Session statistics  (K={K})"),
        mo.hstack([ui_session_subj, ui_session_id]),
        _fig,
    ], align="center")
    return


@app.cell
def _():
    return


@app.cell
def _(K, df_all, mo, np, paths, pl, plt, sns, ui_model_manager):
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
def _(mo, ui_task):

    # ── SSM GLM-HMM safety check (2AFC only) ──────────────────────────────────
    mo.stop(
        ui_task.value != "2AFC",
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
    arrays_store,
    df_all,
    mo,
    names,
    np,
    pl,
    plt,
    sns,
    ssm_run_btn,
    state_labels,
    state_order,
    ui_subjects,
    ui_trial_range,
):

    # ── SSM fit + posterior plot ───────────────────────────────────────────────
    mo.stop(not ssm_run_btn.value, mo.md("Press **▶ Run SSM safety check** above to fit."))

    import ssm as _ssm_lib
    from scripts.fit_glmhmm import _valid_trial_mask as _vtm

    _STIM_NAMES_SSM = {"stim_vals", "stim_d", "ild_norm", "ILD", "ild",
                       "stimulus", "net_ild", "stim_strength"}

    _ssm_subjects = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _ssm_subjects, mo.md("No fitted arrays found — run the custom fit first."))

    _ssm_results = {}

    with mo.status.spinner(title="Fitting SSM GLM-HMM…"):
        for _subj in _ssm_subjects:
            _X   = arrays_store[_subj]["X"]   # (T, n_feat) — already session-filtered
            _y   = arrays_store[_subj]["y"]   # (T,)

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
            _feat_names_s = list(arrays_store[_subj].get("X_cols", names.get("X_cols", [])))
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
        _slbl  = state_labels.get(_subj, {})
        _ek_custom = next(
            (k for k, lbl in _slbl.items() if lbl == "Engaged"),
            state_order.get(_subj, [0])[0],
        )
        _g_custom = arrays_store[_subj]["smoothed_probs"][_t0_s:_t1_s + 1]  # (window, K)
        _p_custom_eng = _g_custom[:, _ek_custom]

        _x_w = np.arange(_t0_s, _t0_s + len(_p_ssm_eng))

        _ax.plot(_x_w, _p_custom_eng, color="steelblue",  lw=1.2, alpha=0.85, label="Custom (stickiness)")
        _ax.plot(_x_w, _p_ssm_eng,    color="darkorange", lw=1.2, alpha=0.85, linestyle="--", label="SSM (standard)")

        # Choice rug
        _y_w = arrays_store[_subj]["y"][_t0_s:_t1_s + 1].astype(int)
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


@app.cell
def _(K, arrays_store, df_all, mo, np, pl, plots, state_labels, ui_subjects):
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _frames = []
    for _subj in _selected:
        _p_pred = arrays_store[_subj]["p_pred"]  # (T, 3): pL, pC, pR
        _p_pred_aux = _p_pred.copy()
        _df_sub = (
            df_all.filter(pl.col("subject") == _subj)
            .sort("Trial")
            .filter(pl.col("session").count().over("session") >= 2)
            .with_columns(
                [
                    pl.Series("pL", _p_pred[:, 0]),
                    pl.Series("pC", _p_pred[:, 1]),
                    pl.Series("pR", _p_pred[:, 2]),
                    pl.Series(
                        "pred_choice", np.argmax(_p_pred, axis=1).astype(int)
                    ),
                ]
            )
        )
        _frames.append(_df_sub)

    _df_all_pred = pl.concat(_frames)
    _plot_df = plots.prepare_predictions_df(_df_all_pred)
    _fig_all, _ = plots.plot_categorical_performance_all(_plot_df, f"glmhmm K={K}")

    # ── per-state overlay — pool all subjects with normalised state ranks ─────
    # Normalise: 0 = Engaged, 1 = Disengaged, … per-subject regardless of raw idx
    _lrank_map = { "Engaged": 0, "Disengaged": 1, **{f"Disengaged {i}": i for i in range(1, K)},}
    _pool_dfs = []
    _pool_assigns = []
    for _subj in _selected:
        _p_pred_s = arrays_store[_subj]["p_pred"]
        _df_s = (
            df_all.filter(pl.col("subject") == _subj)
            .sort("Trial")
            .filter(pl.col("Session").count().over("Session") >= 2)
            .with_columns(
                [
                    pl.Series("pL", _p_pred_s[:, 0]),
                    pl.Series("pC", _p_pred_s[:, 1]),
                    pl.Series("pR", _p_pred_s[:, 2]),
                    pl.Series(
                        "pred_choice", np.argmax(_p_pred_s, axis=1).astype(int)
                    ),
                ]
            )
        )
        _plot_df_s = plots.prepare_predictions_df(_df_s)
        _gamma_s = arrays_store[_subj]["smoothed_probs"]
        # Both must have the same length — if not, session filtering diverged
        # between the fit script and this notebook.
        _T_s = _gamma_s.shape[0]

        # ── per-state emission prediction: softmax(W_k × x) for MAP state k ───────
        # Using the marginal p_pred (blended over all states) makes every
        # state's model line look the same.  Instead look up the emission of
        # the MAP-assigned state directly from the saved weights.
        _W = np.asarray(
            arrays_store[_subj]["emission_weights"]
        )  # (K, C-1, n_feat)
        _X_s = np.asarray(arrays_store[_subj]["X"])  # (T, n_feat)
        _logits = np.einsum("kci,ti->tkc", _W, _X_s)  # (T, K, C-1)  → [L, R]
        _logits_full = np.concatenate(
            [_logits[:, :, :1], np.zeros((_T_s, K, 1)), _logits[:, :, 1:]],
            axis=2,  # (T, K, C) → [L, 0, R]
        )
        _lse = _logits_full.max(axis=2, keepdims=True)
        _exp = np.exp(_logits_full - _lse)
        _p_state = _exp / _exp.sum(axis=2, keepdims=True)  # (T, K, C)
        _map_k = np.argmax(_gamma_s, axis=1).astype(int)  # (T,)
        _stim = _plot_df_s["stimulus"].to_numpy().astype(int)  # (T,)
        _p_state_correct = _p_state[np.arange(_T_s), _map_k, _stim]  # (T,)
        # build per-state df with state-k emission replacing the marginal
        _plot_df_state_s = _plot_df_s.with_columns(
            pl.Series("p_model_correct", _p_state_correct.astype(np.float64))
        )
        _pool_dfs.append(_plot_df_state_s)
        _slbls = state_labels[_subj]
        _raw = _map_k  # reuse already-computed MAP assignment
        _norm = np.array([_lrank_map.get(_slbls.get(int(k), ""), k) for k in _raw])
        _pool_assigns.append(_norm)

    _df_state_pool = pl.concat(_pool_dfs)
    _assign_pool = np.concatenate(_pool_assigns)
    _state_lbl_global = {
        0: "Engaged",
        1: "Disengaged",
        **{i: f"Disengaged {i}" for i in range(2, K)},
    }
    _fig_state, _ = plots.plot_categorical_performance_by_state(df=_df_state_pool,smoothed_probs=None,state_assign=_assign_pool,
                                                                state_labels=_state_lbl_global,model_name=f"glmhmm K={K} — per state",)

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
def _():
    return


if __name__ == "__main__":
    app.run()
