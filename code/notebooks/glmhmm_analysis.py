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
    from glmhmmt.features import build_sequence_from_df
    from scripts.fit_glmhmm import main as fit_main

    sns.set_style("white")

    ui_task = mo.ui.dropdown(
        options=["2AFC", "MCDR"],
        value="MCDR",
        label="Task:",
    )
    ui_task
    return (
        build_sequence_from_df,
        fit_main,
        mo,
        np,
        paths,
        pl,
        plt,
        sns,
        ui_task,
    )


@app.cell
def _(paths, pl, ui_task):
    df_all = pl.read_parquet(paths.DATA_PATH / "df_filtered.parquet")
    df_all = df_all.filter(pl.col("subject") != "A84")
    if ui_task.value == "2AFC":
        df_all = pl.read_parquet(paths.DATA_PATH / "alexis_combined.parquet")
        df_all = df_all.filter(pl.col("experiment").is_in(['2AFC_2', '2AFC_3', '2AFC_4', '2AFC_6']))
        import glmhmmt.plots_alexis as plots
    else:
        import glmhmmt.plots as plots
    df_all
    return df_all, plots


@app.cell
def _(df_all, mo):
    from glmhmmt.features import _ALL_EMISSION_COLS, _ALL_TRANSITION_COLS, _ALL_2AFC_EMISSION_COLS, _SF_COL_PREFIX
    # ── controls ──────────────────────────────────────────────────────────────

    is_2afc = "experiment" in df_all.columns
    if is_2afc:
        _sf_cols = [c for c in df_all.columns if c.startswith(_SF_COL_PREFIX)]
        emission_cols = [c for c in _ALL_2AFC_EMISSION_COLS if c != "stim_strength"] + _sf_cols
    else:
        emission_cols = _ALL_EMISSION_COLS

    ui_K = mo.ui.slider(start=2, stop=6, value=2, label="K")
    ui_subjects = mo.ui.multiselect(
        value=df_all["subject"].unique(),
        options=df_all["subject"].unique(),  # replace with dynamic list
        label="Subjects",
    )
    ui_tau = mo.ui.slider(
        start=1,
        stop=200,
        value=50,
        step=1,
        label="τ action trace half-life",
    )
    ui_emission_cols = mo.ui.multiselect(
        options=emission_cols,
        value=emission_cols,
        label="Emission regressors (X)",
    )


    ui_model_id = mo.ui.text(
        value="glmhmm_2",
        label="Model ID (used as output folder name)",
        full_width=True,
    )



    fit_button = mo.ui.run_button(label="Run fit")
    mo.vstack(
        [
            mo.md("### Configuration"),
            mo.hstack([ui_K, ui_tau, ui_subjects, ui_emission_cols]),
            mo.hstack([fit_button]), ui_model_id
        ],
        align="center",
    )
    return (
        fit_button,
        is_2afc,
        ui_K,
        ui_emission_cols,
        ui_model_id,
        ui_subjects,
        ui_tau,
    )


@app.cell
def _(
    fit_button,
    fit_main,
    is_2afc,
    mo,
    paths,
    ui_K,
    ui_emission_cols,
    ui_model_id,
    ui_subjects,
    ui_task,
    ui_tau,
):
    mo.stop(
        not fit_button.value, mo.md("Configure parameters and press **Run fit**.")
    )
    _OUT =  paths.RESULTS / "fits" / ui_task.value / ui_model_id.value
    with mo.status.spinner(
        title=f"Fitting GLM-HMM K={ui_K.value} τ={ui_tau.value} for {ui_subjects.value}..."
    ):
        fit_main(
                subjects=ui_subjects.value,
                K_list=[ui_K.value],
                out_dir=_OUT,
                tau=ui_tau.value,
                emission_cols=ui_emission_cols.value,
                num_classes=2 if is_2afc else 3,
                task = ui_task.value,
            )
    mo.md("✅ Fit complete — plots below update automatically.")
    return


@app.cell
def _(
    build_sequence_from_df,
    df_all,
    is_2afc,
    np,
    paths,
    pl,
    ui_K,
    ui_emission_cols,
    ui_model_id,
    ui_subjects,
    ui_task,
    ui_tau,
):
    K = ui_K.value

    selected = ui_subjects.value
    OUT =  paths.RESULTS / "fits"/ ui_task.value / ui_model_id.value
    # load feature names from data
    _df_sel = df_all.filter(pl.col("subject").is_in(ui_subjects.value))
    if is_2afc:
        from glmhmmt.features import build_sequence_from_df_2afc
        _, _, names = build_sequence_from_df_2afc(_df_sel, emission_cols=ui_emission_cols.value)
    else:
        _df_sel = _df_sel.sort("trial_idx")
        _, _, _, names, _ = build_sequence_from_df(_df_sel, tau=ui_tau.value, emission_cols=ui_emission_cols.value)

    arrays_store = {}
    for _subj in ui_subjects.value:
        _f = OUT / f"{_subj}_K{K}_glmhmm_arrays.npz"
        if _f.exists():
            _d = dict(np.load(_f, allow_pickle=True))
            # decode column names saved as string arrays; fall back to build output
            _d["X_cols"] = (
                list(_d["X_cols"]) if "X_cols" in _d else names["X_cols"]
            )
            arrays_store[_subj] = _d

    arrays_store
    return K, arrays_store, names, selected


@app.cell
def _(K, arrays_store, names, np, ui_subjects):
    # ── State labelling: Engaged / Disengaged per subject ────────────────────
    # S_coh score = mean(W[k, class_L, fi_SL], W[k, class_R, fi_SR])
    # The state with highest S_coh is "Engaged"; the rest are "Disengaged".
    # SC is excluded (no lateralised direction → not informative for engagement).
    _selected = [s for s in ui_subjects.value if s in arrays_store]


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


    _feat_names = names.get("X_cols", [])  # fallback; per-subject override below
    state_labels = {}  # subj -> {state_idx: label_str}
    state_order = {}  # subj -> [state_idx, ...] sorted by S_coh desc

    for _subj in _selected:
        _W = arrays_store[_subj].get("emission_weights")
        if _W is None:
            state_labels[_subj] = {k: f"State {k + 1}" for k in range(K)}
            state_order[_subj] = list(range(K))
            continue
        # prefer column names saved alongside this subject's fit
        _feat_names_subj = arrays_store[_subj].get("X_cols", _feat_names)
        _scores = _scoh_score(_W, _feat_names_subj)
        _ranking = list(np.argsort(_scores)[::-1])
        _labels = {}
        _dis_idx = 1
        for _rank, _k in enumerate(_ranking):
            if _rank == 0:
                _labels[int(_k)] = "Engaged"
            else:
                _labels[int(_k)] = (
                    "Disengaged" if K == 2 else f"Disengaged {_dis_idx}"
                )
                _dis_idx += 1
        state_labels[_subj] = _labels
        state_order[_subj] = [int(k) for k in _ranking]
    return (state_labels,)


@app.cell
def _(K, arrays_store, mo, names, paths, plots, state_labels, ui_subjects):
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
    _fig_ag, _fig_cls = plots.plot_emission_weights( arrays_store=arrays_store, state_labels=state_labels, names=names, K=K, subjects=_selected, save_path=paths.RESULTS / "plots/GLMHMM/emissions_coefs.png",)
    mo.vstack([mo.md("### Emission weights"), _fig_ag, _fig_cls])
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
def _(arrays_store, mo, ui_subjects):
    # ── trial-window slider (shared across all posterior plots) ──────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
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
    ui_subjects,
    ui_trial_range,
):
    # ── posterior state probabilities ─────────────────────────────────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
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
    arrays_store,
    df_all,
    is_2afc,
    mo,
    np,
    pl,
    plots,
    state_labels,
    ui_subjects,
):
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))


    _sort_col = "trial" if is_2afc else "trial_idx"
    _frames = []
    for _subj in _selected:
        _p_pred = arrays_store[_subj]["p_pred"]  # (T, 2) for 2AFC, (T, 3) for 3AFC
        _n_classes = _p_pred.shape[1]
        _df_sub = (
            df_all.filter(pl.col("subject") == _subj)
            .sort(_sort_col)
            .filter(pl.col("session").count().over("session") >= 2)
        )
        _cols = [pl.Series("pred_choice", np.argmax(_p_pred, axis=1).astype(int))]
        if _n_classes == 2:
            _cols += [pl.Series("pL", _p_pred[:, 0]), pl.Series("pR", _p_pred[:, 1])]
        else:
            _cols += [pl.Series("pL", _p_pred[:, 0]), pl.Series("pC", _p_pred[:, 1]), pl.Series("pR", _p_pred[:, 2])]
        _df_sub = _df_sub.with_columns(_cols)
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
        _nc_s = _p_pred_s.shape[1]
        _pred_cols_s = [pl.Series("pred_choice", np.argmax(_p_pred_s, axis=1).astype(int))]
        if _nc_s == 2:
            _pred_cols_s += [pl.Series("pL", _p_pred_s[:, 0]), pl.Series("pR", _p_pred_s[:, 1])]
        else:
            _pred_cols_s += [pl.Series("pL", _p_pred_s[:, 0]), pl.Series("pC", _p_pred_s[:, 1]), pl.Series("pR", _p_pred_s[:, 2])]
        _df_s = (
            df_all.filter(pl.col("subject") == _subj)
            .sort(_sort_col)
            .filter(pl.col("session").count().over("session") >= 2)
            .with_columns(_pred_cols_s)
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
        _logits = np.einsum("kci,ti->tkc", _W, _X_s)  # (T, K, C-1)
        _nc_logits = _logits.shape[2] + 1  # actual number of classes
        if _nc_logits == 2:
            # binary: logits shape (T, K, 1) → append reference class 0
            _logits_full = np.concatenate(
                [_logits, np.zeros((_T_s, K, 1))], axis=2
            )  # (T, K, 2) → [L, ref=R]
        else:
            # 3-class: insert reference class C in middle
            _logits_full = np.concatenate(
                [_logits[:, :, :1], np.zeros((_T_s, K, 1)), _logits[:, :, 1:]],
                axis=2,
            )  # (T, K, 3) → [L, 0, R]
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
    arrays_store,
    df_all,
    mo,
    plots,
    state_labels,
    ui_subjects,
):
    # ── Per-state accuracy — Ashwood et al. 2022 method ──────────────────────
    # All     : mean(performance) on nonzero-stim trials — the full pool
    # State k : mean(performance) on the SUBSET where posterior[:,k] >= 0.9
    #           AND stimd_n != 0
    # "All" is the weighted average of the state bars (plus ambiguous trials).
    # Colors assigned by rank: Engaged=palette[0], Disengaged=palette[1], …

    _selected_acc = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected_acc, mo.md("No fitted subjects available."))
    _fig_acc, _tbl = plots.plot_state_accuracy(arrays_store=arrays_store, state_labels=state_labels, df_all=df_all, K=K, subjects=_selected_acc, thresh=THRESH_ui.amount)
    mo.vstack([
        mo.md("### Accuracy by state"),
        mo.md(f"> **All** = full nonzero-stim pool · **State bars** = subsets where posterior ≥ {THRESH_ui}"),
        _fig_acc,
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
def _(arrays_store, mo, ui_subjects):
    # ── Session deep-dive controls ─────────────────────────────────────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    _subj_opts = _selected if _selected else ["(no fitted subjects)"]

    ui_session_subj = mo.ui.dropdown(
        options=_subj_opts,
        value=_subj_opts[0],
        label="Subject",
    )
    return (ui_session_subj,)


@app.cell
def _(arrays_store, df_all, mo, pl, ui_session_subj):
    _sess_opts = (
        sorted(
            df_all.filter(pl.col("subject") == ui_session_subj.value)
            .filter(pl.col("session").count().over("session") >= 2)["session"]
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

    _sess = ui_session_id.value
    _sort_col_dd = "trial" if "2AFC" in str(df_all["experiment"][0]) else "trial_idx"
    _fig = plots.plot_session_deepdive( arrays_store=arrays_store, state_labels=state_labels, df_all=df_all, names=arrays_store[selected[0]], K=K, subj=_subj, sess=_sess,)
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
def _():
    return


if __name__ == "__main__":
    app.run()
