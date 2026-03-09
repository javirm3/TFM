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
    from tasks import get_adapter
    from widgets import ModelManagerWidget
    sns.set_style("white")

    ui_task = mo.ui.dropdown(
        options=["2AFC", "MCDR"],
        value="MCDR",
        label="Task:",
    )
    ui_task
    return ModelManagerWidget, get_adapter, mo, np, paths, pl, plt, sns, ui_task


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
    _fits_path = paths.RESULTS / "fits" / ui_task.value / "glmhmmt"
    _existing_opts = []
    return


@app.cell
def _(paths, ui_task):
    import json as _json
    return


@app.cell
def _(ModelManagerWidget, adapter, df_all, mo, ui_task):
    # ── controls ──────────────────────────────────────────────────────────────
    is_2afc = adapter.num_classes == 2
    _subjects = df_all["subject"].unique().to_list()

    mm_widget = ModelManagerWidget(
        model_type="glmhmmt",
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
    from scripts.fit_glmhmmt import generate_model_id as _gen_id
    
    _val = ui_model_manager.value
    current_hash = _gen_id(ui_task.value, _val["K"], _val["tau"], _val["emission_cols"])

    return current_hash,


@app.cell
def _(
    current_hash,
    mo,
    paths,
    ui_model_manager,
    ui_task,
):
    from scripts.fit_glmhmmt import main as fit_main

    _val = ui_model_manager.value
    _clicks = _val["run_fit_clicks"]
    
    mo.stop(_clicks == 0, mo.md("Configure parameters and press **RUN FIT 🚀**."))

    _selected_id = _val["existing_model"] or (_val["alias"] if _val["alias"] else current_hash)
    _OUT = paths.RESULTS / "fits" / ui_task.value / "glmhmmt" / _selected_id
    with mo.status.spinner(title=f"Fitting {_selected_id} K={_val['K']} τ={_val['tau']} for {_val['subjects']}..."):
        fit_main(
            subjects=_val["subjects"],
            K_list=[_val["K"]],
            out_dir=_OUT,
            emission_cols=_val["emission_cols"] or None,
            transition_cols=_val["transition_cols"] or None,
            tau=_val["tau"],
            task=ui_task.value,
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

    selected_model_id = _val["existing_model"] or (_val["alias"] if _val["alias"] else current_hash)
    OUT = paths.RESULTS / "fits" / ui_task.value / "glmhmmt" / selected_model_id
    # load feature names from data (use first available subject for a representative build)
    _df_sel = df_all.filter(pl.col("subject").is_in(_selected_subjects)).sort(adapter.sort_col)
    _, _, _, names = adapter.load_subject(_df_sel, tau=_val["tau"])

    arrays_store = {}
    for _subj in _selected_subjects:
        _f = OUT / f"{_subj}_K{K}_glmhmmt_arrays.npz"
        if _f.exists():
            _d = dict(np.load(_f, allow_pickle=True))
            # decode column names saved as string arrays; fall back to build output
            _d["X_cols"] = list(_d["X_cols"]) if "X_cols" in _d else names["X_cols"]
            _d["U_cols"] = list(_d["U_cols"]) if "U_cols" in _d else names["U_cols"]
            arrays_store[_subj] = _d

    # arrays_store
    return K, arrays_store, names


@app.cell
def _(K, adapter, arrays_store, names, ui_model_manager):
    # ── State labelling — task-specific criteria via adapter ────────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    state_labels, state_order = adapter.label_states(arrays_store, names, K, _selected)

    #state_labels, state_order
    return (state_labels,)


@app.cell
def _(K, arrays_store, mo, names, paths, plots, state_labels, ui_model_manager):
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
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))
    _fig_ag, _fig_cls = plots.plot_emission_weights(
        arrays_store=arrays_store,
        state_labels=state_labels,
        names=names,
        K=K,
        subjects=_selected,
        save_path=paths.RESULTS / "plots/GLMHMMT/emissions_coefs.png",
    )
    mo.vstack([mo.md("### Emission weights"), _fig_ag, _fig_cls])
    return


@app.cell
def _(K, arrays_store, mo, np, plt, sns, state_labels, ui_model_manager):
    # ── transition matrix heatmap — marimo grid (3 per row) ──────────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
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
def _(arrays_store, mo, ui_model_manager):
    # ── trial-window slider (shared across all posterior plots) ──────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
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
def _(K, arrays_store, mo, plots, state_labels, ui_model_manager, ui_trial_range):
    # ── posterior state probabilities ─────────────────────────────────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))
    _t0, _t1 = ui_trial_range.value
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
def _(
    K,
    adapter,
    arrays_store,
    df_all,
    mo,
    np,
    pl,
    plots,
    state_labels,
    ui_model_manager,
):
    # ── predictions & categorical performance ────────────────────────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _frames = []
    for _subj in _selected:
        _p_pred = arrays_store[_subj]["p_pred"]        # (T, 3): pL, pC, pR
        _df_sub = (
            df_all
            .filter(pl.col("subject") == _subj)
            .sort(adapter.sort_col)
            .filter(pl.col(adapter.session_col).count().over(adapter.session_col) >= 2)
            .with_columns([
                pl.Series("pL", _p_pred[:, 0]),
                pl.Series("pC", _p_pred[:, 1]),
                pl.Series("pR", _p_pred[:, 2]),
                pl.Series("pred_choice", np.argmax(_p_pred, axis=1).astype(int)),
            ])
        )
        _frames.append(_df_sub)

    _df_all_pred = pl.concat(_frames)
    _plot_df = plots.prepare_predictions_df(_df_all_pred)
    _fig_all, _ = plots.plot_categorical_performance_all(_plot_df, f"glmhmmt K={K}")

    # ── per-state overlay — pool all subjects with normalised state ranks ─────
    # Normalise: 0 = Engaged, 1 = Disengaged, … per-subject regardless of raw idx
    _lrank_map = {"Engaged": 0, "Disengaged": 1,
                  **{f"Disengaged {i}": i for i in range(1, K)}}
    _pool_dfs     = []
    _pool_assigns = []
    for _subj in _selected:
        _p_pred_s = arrays_store[_subj]["p_pred"]
        _df_s = (
            df_all
            .filter(pl.col("subject") == _subj)
            .sort(adapter.sort_col)
            .filter(pl.col(adapter.session_col).count().over(adapter.session_col) >= 2)
            .with_columns([
                pl.Series("pL",          _p_pred_s[:, 0]),
                pl.Series("pC",          _p_pred_s[:, 1]),
                pl.Series("pR",          _p_pred_s[:, 2]),
                pl.Series("pred_choice", np.argmax(_p_pred_s, axis=1).astype(int)),
            ])
        )
        _plot_df_s = plots.prepare_predictions_df(_df_s)
        _gamma_s   = arrays_store[_subj]["smoothed_probs"]
        # Both must have the same length — if not, session filtering diverged
        # between the fit script and this notebook.
        assert _plot_df_s.height == _gamma_s.shape[0], (
            f"{_subj}: df has {_plot_df_s.height} rows but smoothed_probs has "
            f"{_gamma_s.shape[0]}. Check session-count filter consistency."
        )
        _T_s = _gamma_s.shape[0]

        # ── per-state emission prediction: softmax(W_k × x) for MAP state k ───────
        # Using the marginal p_pred (blended over all states) makes every
        # state's model line look the same.  Instead look up the emission of
        # the MAP-assigned state directly from the saved weights.
        _W = np.asarray(arrays_store[_subj]["emission_weights"])  # (K, C-1, n_feat)
        _X_s = np.asarray(arrays_store[_subj]["X"])               # (T, n_feat)
        _logits = np.einsum("kci,ti->tkc", _W, _X_s)              # (T, K, C-1)  → [L, R]
        _logits_full = np.concatenate(
                    [_logits, np.zeros((_T_s, K, 1))],
                    axis=2,
        )
        _lse = _logits_full.max(axis=2, keepdims=True)
        _exp = np.exp(_logits_full - _lse)
        _p_state = _exp / _exp.sum(axis=2, keepdims=True)         # (T, K, C)
        _map_k = np.argmax(_gamma_s, axis=1).astype(int)          # (T,)
        _stim = _plot_df_s["stimulus"].to_numpy().astype(int)      # (T,)
        _p_state_correct = _p_state[np.arange(_T_s), _map_k, _stim]  # (T,)
        # build per-state df with state-k emission replacing the marginal
        _plot_df_state_s = _plot_df_s.with_columns(
            pl.Series("p_model_correct", _p_state_correct.astype(np.float64))
        )
        _pool_dfs.append(_plot_df_state_s)
        _slbls  = state_labels[_subj]
        _raw    = _map_k  # reuse already-computed MAP assignment
        _norm   = np.array([_lrank_map.get(_slbls.get(int(k), ""), k) for k in _raw])
        _pool_assigns.append(_norm)

    _df_state_pool  = pl.concat(_pool_dfs)
    _assign_pool    = np.concatenate(_pool_assigns)
    _state_lbl_global = {0: "Engaged", 1: "Disengaged",
                         **{i: f"Disengaged {i}" for i in range(2, K)}}
    _fig_state, _   = plots.plot_categorical_performance_by_state(
        df=_df_state_pool,
        smoothed_probs=None,
        state_assign=_assign_pool,
        state_labels=_state_lbl_global,
        model_name=f"glmhmmt K={K} — per state",
    )

    mo.vstack([
        mo.md("### Categorical plots for accuracy"),
        _fig_all,
        mo.md("### Per-state categorical performance"),
        _fig_state,
    ], align="center")
    return


@app.cell
def _(K, arrays_store, mo, names, plots, ui_model_manager):
    # ── Input-dependent transition weights ────────────────────────────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    mo.stop(
        not _selected or "transition_weights" not in arrays_store.get(_selected[0], {}),
        mo.md("No transition weights found — run the glmhmm-t fit first."),
    )
    _fig_line, _fig_std, _fig_raw = plots.plot_transition_weights(
        arrays_store=arrays_store,
        names=names,
        K=K,
        subjects=_selected,
    )
    mo.vstack([mo.md("### Transition weights"), mo.hstack([_fig_line, _fig_std]), _fig_raw])
    return


@app.cell
def _(K, arrays_store, df_all, mo, plots, state_labels, ui_model_manager):
    # ── Per-state accuracy ────────────────────────────────────────────────────
    _val = ui_model_manager.value
    _selected = [s for s in _val["subjects"] if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted subjects available."))
    _fig_acc, _tbl = plots.plot_state_accuracy(
        arrays_store=arrays_store,
        state_labels=state_labels,
        df_all=df_all,
        K=K,
        subjects=_selected,
    )
    mo.vstack([
        mo.md("### Accuracy by state"),
        mo.md("> **All** = full nonzero-stim pool · **State bars** = subsets where posterior ≥ 0.5"),
        _fig_acc,
        mo.md("**Trial counts & mean accuracy per label:**"),
        mo.plain_text(_tbl.to_string()),
    ])
    return


@app.cell
def _():
    77791.0 + 68134.0
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
    # ── c. Average state-probability trajectories within a session ────────────
    _selected_traj = [s for s in ui_subjects_traj.value if s in arrays_store]
    mo.stop(not _selected_traj, mo.md("Select subjects above to view session trajectories."))
    _fig_traj = plots.plot_session_trajectories(
        arrays_store=arrays_store,
        state_labels=state_labels,
        df_all=df_all,
        K=K,
        subjects=_selected_traj,
        session_col=adapter.session_col,
    )
    mo.vstack([
        mo.md(f"### c. Average state-probability trajectories within a session  (K={K})"),
        mo.md("> Mean ± 1 s.e.m. across sessions for the selected subjects."),
        _fig_traj,
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
    # ── d. Fractional occupancy & state-change histogram ─────────────────────
    _selected_occ = [s for s in ui_subjects_traj.value if s in arrays_store]
    mo.stop(not _selected_occ, mo.md("Select subjects above."))
    _fig_occ = plots.plot_state_occupancy(
        arrays_store=arrays_store,
        state_labels=state_labels,
        df_all=df_all,
        K=K,
        subjects=_selected_occ,
        session_col=adapter.session_col,
    )
    mo.vstack([
        mo.md(f"### d. Fractional occupancy & state changes per session  (K={K})"),
        mo.md(
            "> **Left**: fraction of all trials assigned to each state (argmax of posterior).  \n"
            "> **Right**: histogram of inferred state changes per session."
        ),
        _fig_occ,
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
            .filter(pl.col(_ses_col).count().over(_ses_col) >= 2)
            [_ses_col].unique().to_list()
        )
        if ui_session_subj.value in arrays_store
        else [0]
    )
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
def _(
    K,
    adapter,
    arrays_store,
    df_all,
    mo,
    names,
    plots,
    state_labels,
    ui_session_id,
    ui_session_subj,
):
    # ── Session deep-dive plot ─────────────────────────────────────────────────
    _subj = ui_session_subj.value
    mo.stop(
        _subj not in arrays_store,
        mo.md("No fitted arrays for this subject — run the fit first."),
    )

    _sess = int(ui_session_id.value)
    _fig = plots.plot_session_deepdive(
        arrays_store=arrays_store,
        state_labels=state_labels,
        df_all=df_all,
        names=names,
        K=K,
        subj=_subj,
        sess=_sess,
        session_col=adapter.session_col,
    )
    mo.vstack([
        mo.md(f"### Session deep-dive  (K={K})"),
        _fig,
    ], align="center")
    return


@app.cell
def _(K, current_hash, mo, paths, plots, ui_model_manager):
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
    _val = ui_model_manager.value
    _subjects = list(_val["subjects"])
    _fig_sweep, _best = plots.plot_tau_sweep(
        sweep_path=_sweep_path,
        subjects=_subjects,
        K=K,
    )
    _alias = _val["alias"] or current_hash
    mo.vstack([
        mo.md(f"### τ sweep results — {_alias} K={K}"),
        _fig_sweep,
        mo.md("**Best τ per subject (min BIC):**"),
        mo.plain_text(_best.to_pandas().to_string(index=False)),
    ], align="center")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
