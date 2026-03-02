import marimo

__generated_with = "0.20.2"
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
    import glmhmmt.plots as plots
    sns.set_style("white")

    df_all = pl.read_parquet(paths.DATA_PATH / "df_filtered.parquet")
    df_all = df_all.filter(pl.col("subject") != "A84")
    return df_all, mo, np, paths, pl, plots, plt, sns


@app.cell(hide_code=True)
def _(df_all, mo):
    from glmhmmt.features import _ALL_EMISSION_COLS, _ALL_TRANSITION_COLS

    # ── controls ──────────────────────────────────────────────────────────────
    ui_K = mo.ui.slider(start=2, stop=6, value=2, label="K")
    ui_subjects = mo.ui.multiselect(
        value = df_all["subject"].unique(),
        options=df_all["subject"].unique(),
        label="Subjects",
    )

    ui_model_id = mo.ui.text(
        value="glmhmmt_2",
        label="Model ID (used as output folder name)",
        full_width=True,
    )

    ui_emission_cols = mo.ui.multiselect(
        options=_ALL_EMISSION_COLS,
        value=_ALL_EMISSION_COLS,
        label="Emission regressors (X)",
    )

    ui_transition_cols = mo.ui.multiselect(
        options=_ALL_TRANSITION_COLS,
        value=_ALL_TRANSITION_COLS,
        label="Transition regressors (U)",
    )

    ui_tau = mo.ui.slider(
        start=1, stop=200, value=50, step=1,
        label="τ action trace half-life",
    )

    fit_button = mo.ui.run_button(label="Run fit")

    mo.vstack([
        mo.md("### Model Configuration"),
        mo.hstack([ui_model_id, ui_K, ui_subjects]),
        mo.hstack([ui_tau, ui_emission_cols, ui_transition_cols]),
        mo.hstack([fit_button]),
    ], align="start")
    return (
        fit_button,
        ui_K,
        ui_emission_cols,
        ui_model_id,
        ui_subjects,
        ui_tau,
        ui_transition_cols,
    )


@app.cell
def _(
    fit_button,
    mo,
    paths,
    ui_K,
    ui_emission_cols,
    ui_model_id,
    ui_subjects,
    ui_tau,
    ui_transition_cols,
):
    from scripts.fit_glmhmmt import main as fit_main

    mo.stop(not fit_button.value, mo.md("Configure parameters and press **Run fit**."))

    _OUT = paths.RESULTS / "fits" / ui_model_id.value
    with mo.status.spinner(title=f"Fitting {ui_model_id.value} K={ui_K.value} τ={ui_tau.value} for {ui_subjects.value}..."):
        fit_main(
            subjects=ui_subjects.value,
            K_list=[ui_K.value],
            out_dir=_OUT,
            emission_cols=ui_emission_cols.value or None,
            transition_cols=ui_transition_cols.value or None,
            tau=ui_tau.value,
        )
    mo.md("✅ Fit complete — plots below update automatically.")
    return


@app.cell
def _(df_all, np, paths, pl, ui_K, ui_model_id, ui_subjects):
    from glmhmmt.features import build_sequence_from_df

    K = ui_K.value

    OUT = paths.RESULTS / "fits" / ui_model_id.value

    selected = [s for s in ui_subjects.value if s in ui_subjects.value]
    _df_sel = df_all.filter(pl.col("subject").is_in(selected)).sort("trial_idx")

    arrays_store = {}
    for _subj in selected:
        _f = OUT / f"{_subj}_K{K}_glmhmmt_arrays.npz"
        if _f.exists():
            _d = dict(np.load(_f, allow_pickle=True))
            _d["X_cols"] = list(_d["names"].item()["X_cols"])
            _d["U_cols"] = list(_d["names"].item()["U_cols"])
            arrays_store[_subj] = _d

    arrays_store
    return K, arrays_store, selected


@app.cell
def _(K, arrays_store, np, selected):
    # ── State labelling: Engaged / Disengaged per subject ────────────────────
    # S_coh score = mean(W[k, class_L, fi_SL], W[k, class_R, fi_SR])
    # The state with highest S_coh is "Engaged"; the rest are "Disengaged".
    # SC is excluded (no lateralised direction → not informative for engagement).

    def _scoh_score(W, feat_names):
        """Return S_coh engagement score per state (shape: K,)."""
        _name2fi = {n: i for i, n in enumerate(feat_names)}
        scores = np.zeros(W.shape[0])
        n_terms = 0
        if "SL" in _name2fi:
            scores += W[:, 0, _name2fi["SL"]]
            n_terms += 1
        if "SR" in _name2fi:
            scores += W[:, 1, _name2fi["SR"]]
            n_terms += 1
        return scores / max(1, n_terms)

    _feat_names = arrays_store[selected[0]]["X_cols"] + arrays_store[selected[0]]["U_cols"]
    state_labels = {}   # subj -> {state_idx: label_str}
    state_order  = {}   # subj -> [state_idx, ...] sorted by S_coh desc

    for _subj in selected:
        _W = arrays_store[_subj].get("emission_weights")
        if _W is None:
            state_labels[_subj] = {k: f"State {k+1}" for k in range(K)}
            state_order[_subj]  = list(range(K))
            continue
        # prefer column names saved alongside this subject's fit
        _feat_names_subj = arrays_store[_subj].get("X_cols", _feat_names)
        _scores  = _scoh_score(_W, _feat_names_subj)
        _ranking = list(np.argsort(_scores)[::-1])
        _labels  = {}
        _dis_idx = 1
        for _rank, _k in enumerate(_ranking):
            if _rank == 0:
                _labels[int(_k)] = "Engaged"
            else:
                _labels[int(_k)] = "Disengaged" if K == 2 else f"Disengaged {_dis_idx}"
                _dis_idx += 1
        state_labels[_subj] = _labels
        state_order[_subj]  = [int(k) for k in _ranking]

    #state_labels, state_order
    return (state_labels,)


@app.cell
def _(K, arrays_store, mo, paths, plots, selected, state_labels):
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
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    _fig_ag, _fig_cls = plots.plot_emission_weights( arrays_store=arrays_store, state_labels=state_labels, names=arrays_store[selected[0]], K=K, subjects=selected,
                                                    save_path=paths.RESULTS / "plots/GLMHMMT/emissions_coefs.png",)
    mo.vstack([mo.md("### Emission weights"), _fig_ag, _fig_cls])
    return


@app.cell
def _(K, arrays_store, mo, np, plt, selected, sns, state_labels):
    _figs_t = []
    for _subj in selected:
        _arr = arrays_store[_subj]
        if "transition_matrix" in _arr:
            _A = _arr["transition_matrix"]
        else:
            _bias = _arr["transition_bias"]  # (K, K)
            _A = np.exp(_bias) / np.exp(_bias).sum(axis=-1, keepdims=True)
        _slbl = state_labels.get(_subj, {k: f"S{k}" for k in range(K)})
        _tick_labels = [_slbl.get(k, f"S{k}") for k in range(K)]
        _fig_t, _ax_t = plt.subplots(figsize=(3.2, 2.8))
        sns.heatmap( _A, ax=_ax_t, cmap="bone", annot=True, fmt=".2f", vmin=0, vmax=1, square=True, linewidths=0.5, xticklabels=_tick_labels, yticklabels=_tick_labels, cbar_kws={"shrink": 0.8, "label": "probability"},)
        _ax_t.set_title(f"Subject {_subj}")
        _ax_t.set_xlabel("To state")
        _ax_t.set_ylabel("From state")
        _fig_t.tight_layout()
        _figs_t.append(_fig_t)
    _rows_t = [
        mo.hstack(_figs_t[i : i + 3], justify="start")
        for i in range(0, len(_figs_t), 3)
    ]
    mo.vstack([
        mo.md(f"### Transition matrices — bias-only component  (K={K})"),
        *_rows_t,
    ])
    return


@app.cell
def _(arrays_store, mo, selected):
    _T_max = (max(arrays_store[s]["smoothed_probs"].shape[0] for s in selected) if selected else 20 )
    ui_trial_range = mo.ui.range_slider( start=0, stop=_T_max - 1, value=[0, min(_T_max - 1, 199)], label="Trial window", step=1,)
    return (ui_trial_range,)


@app.cell
def _(K, arrays_store, mo, plots, selected, state_labels, ui_trial_range):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))
    _t0, _t1 = ui_trial_range.value
    _fig_p = plots.plot_posterior_probs( arrays_store=arrays_store, state_labels=state_labels, K=K, subjects=selected, t0=_t0, t1=_t1,)
    mo.vstack([
        mo.md(f"### Posterior state probabilities  (K={K})"),
        ui_trial_range,
        _fig_p,
    ], align="center")
    return


@app.cell
def _(K, arrays_store, df_all, mo, np, pl, plots, selected, state_labels):
    mo.stop(not selected, mo.md("No fitted arrays found — run the fit first."))

    _frames = []
    for _subj in selected:
        _p_pred = arrays_store[_subj]["p_pred"]        # (T, 3): pL, pC, pR
        _df_sub = (
            df_all
            .filter(pl.col("subject") == _subj)
            .sort("trial_idx")
            .filter(pl.col("session").count().over("session") >= 2)
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

    _lrank_map = {"Engaged": 0, "Disengaged": 1,
                  **{f"Disengaged {i}": i for i in range(1, K)}}
    _pool_dfs     = []
    _pool_assigns = []
    for _subj in selected:
        _p_pred_s = arrays_store[_subj]["p_pred"]
        _df_s = (
            df_all
            .filter(pl.col("subject") == _subj)
            .sort("trial_idx")
            .filter(pl.col("session").count().over("session") >= 2)
            .with_columns([
                pl.Series("pL",          _p_pred_s[:, 0]),
                pl.Series("pC",          _p_pred_s[:, 1]),
                pl.Series("pR",          _p_pred_s[:, 2]),
                pl.Series("pred_choice", np.argmax(_p_pred_s, axis=1).astype(int)),
            ])
        )
        _plot_df_s = plots.prepare_predictions_df(_df_s)
        _gamma_s   = arrays_store[_subj]["smoothed_probs"]
        _T_s = _gamma_s.shape[0]

        # ── per-state emission prediction: softmax(W_k × x) for MAP state k ───────
        # Using the marginal p_pred (blended over all states) makes every
        # state's model line look the same.  Instead look up the emission of
        # the MAP-assigned state directly from the saved weights.
        _W = np.asarray(arrays_store[_subj]["emission_weights"])  # (K, C-1, n_feat)
        _X_s = np.asarray(arrays_store[_subj]["X"])               # (T, n_feat)
        _logits = np.einsum("kci,ti->tkc", _W, _X_s)              # (T, K, C-1)  → [L, R]
        _logits_full = np.concatenate(
            [_logits[:, :, :1], np.zeros((_T_s, K, 1)), _logits[:, :, 1:]], axis=2  # (T, K, C) → [L, 0, R]
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
    states = {0: "Engaged", 1: "Disengaged",
                         **{i: f"Disengaged {i}" for i in range(2, K)}}
    _fig_state, _   = plots.plot_categorical_performance_by_state( df=_df_state_pool, smoothed_probs=None, state_assign=_assign_pool, state_labels=states, model_name=f"glmhmmt K={K} — per state",)

    mo.vstack([
        mo.md("### Categorical plots for accuracy"),
        _fig_all,
        mo.md("### Per-state categorical performance"),
        _fig_state,
    ], align="center")
    return


@app.cell
def _(K, arrays_store, mo, plots, selected, state_labels):
    mo.stop( not selected or "transition_weights" not in arrays_store.get(selected[0], {}), mo.md("No transition weights found — run the glmhmm-t fit first."))

    _fig_line, _fig_std, _fig_raw = plots.plot_transition_weights( arrays_store=arrays_store, names=arrays_store[selected[0]], K=K, subjects=selected, state_labels=state_labels,)

    mo.vstack([mo.md("### Transition weights"), mo.hstack([_fig_line, _fig_std]), _fig_raw])
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
def _(K, THRESH_ui, arrays_store, df_all, mo, plots, selected, state_labels):
    mo.stop(not selected, mo.md("No fitted subjects available."))
    _fig_acc, _tbl = plots.plot_state_accuracy(arrays_store=arrays_store, state_labels=state_labels, df_all=df_all, K=K, subjects=selected, thresh=THRESH_ui.amount)
    mo.vstack([
        mo.md("### Accuracy by state"),
        mo.md(f"> **All** = full nonzero-stim pool · **State bars** = subsets where posterior ≥ {THRESH_ui}"),
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
    ui_subjects_traj = mo.ui.multiselect( options=df_all["subject"].unique().to_list(), label="Subjects (session trajectories & occupancy)",)
    mo.vstack([mo.md("### Session trajectory & occupancy"), ui_subjects_traj])
    return (ui_subjects_traj,)


@app.cell
def _(K, arrays_store, df_all, mo, plots, state_labels, ui_subjects_traj):
    selected_traj = [s for s in ui_subjects_traj.value if s in arrays_store]
    mo.stop(not selected_traj, mo.md("Select subjects above to view session trajectories."))
    _fig_traj = plots.plot_session_trajectories( arrays_store=arrays_store, state_labels=state_labels, df_all=df_all, K=K, subjects=selected_traj,)
    mo.vstack([
        mo.md(f"### c. Average state-probability trajectories within a session  (K={K})"),
        mo.md("> Mean ± 1 s.e.m. across sessions for the selected subjects."),
        _fig_traj,
    ], align="center")
    return


@app.cell
def _(K, arrays_store, df_all, mo, plots, state_labels, ui_subjects_traj):
    selected_occ = [s for s in ui_subjects_traj.value if s in arrays_store]
    mo.stop(not selected_occ, mo.md("Select subjects above."))
    _fig_occ = plots.plot_state_occupancy(arrays_store=arrays_store,state_labels=state_labels,df_all=df_all,K=K,subjects=selected_occ,)
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
def _(mo, selected):
    _subj_opts = selected if selected else ["(no fitted subjects)"]

    ui_session_subj = mo.ui.dropdown( options=_subj_opts, value=_subj_opts[0], label="Subject",)
    return (ui_session_subj,)


@app.cell
def _(arrays_store, df_all, mo, pl, ui_session_subj):
    _sess_opts = (
        sorted(
            df_all.filter(pl.col("subject") == ui_session_subj.value)
            .filter(pl.col("session").count().over("session") >= 2)
            ["session"].unique().to_list()
        )
        if ui_session_subj.value in arrays_store
        else [0]
    )
    ui_session_id = mo.ui.dropdown( options=[str(s) for s in _sess_opts], value=str(_sess_opts[0]), label="Session",)
    return (ui_session_id,)


@app.cell
def _(
    K,
    arrays_store,
    df_all,
    mo,
    plots,
    selected,
    state_labels,
    ui_session_id,
    ui_session_subj,
):
    _subj = ui_session_subj.value
    mo.stop(
        _subj not in arrays_store,
        mo.md("No fitted arrays for this subject — run the fit first."),
    )

    _sess = int(ui_session_id.value)
    _fig = plots.plot_session_deepdive( arrays_store=arrays_store, state_labels=state_labels, df_all=df_all, names=arrays_store[selected[0]], K=K, subj=_subj, sess=_sess,)
    mo.vstack([
        mo.md(f"### Session statistics  (K={K})"),
        mo.hstack([ui_session_subj, ui_session_id]),
        _fig,
    ], align="center")
    return


@app.cell
def _(K, mo, paths, plots, selected, ui_model_id):
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
    _fig_sweep, _best = plots.plot_tau_sweep(sweep_path=_sweep_path,subjects=selected,K=K,)
    mo.vstack([
        mo.md(f"### τ sweep results — {ui_model_id.value} K={K}"),
        _fig_sweep,
        mo.md("**Best τ per subject (min BIC):**"),
        mo.plain_text(_best.to_pandas().to_string(index=False)),
    ], align="center")
    return


@app.cell
def _():
    from wigglystuff import ApiDoc

    return (ApiDoc,)


@app.cell
def tour(ApiDoc, mo, plots):
    mo.ui.anywidget(ApiDoc(plots.plot_categorical_performance_by_state))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
