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

    sns.set_style("white")

    df_all = pl.read_parquet(paths.DATA_PATH / "df_filtered.parquet")
    return df_all, mo, np, paths, pd, pl, plt, sns


@app.cell
def _(df_all, mo):
    # ── controls ──────────────────────────────────────────────────────────────
    ui_K = mo.ui.slider(start=2, stop=6, value=2, label="K")
    ui_subjects = mo.ui.multiselect(
        value=df_all["subject"].unique(),
        options=df_all["subject"].unique(),  # replace with dynamic list
        label="Subjects",
    )
    fit_button = mo.ui.run_button(label="Run fit")
    mo.vstack(
        [mo.md("### Configuration"), mo.hstack([ui_K, ui_subjects, fit_button])],
        align="center",
    )
    return fit_button, ui_K, ui_subjects


@app.cell
def _(fit_button, mo, paths, ui_K, ui_subjects):
    from scripts.fit_glmhmm import main as fit_main

    mo.stop(
        not fit_button.value, mo.md("Configure parameters and press **Run fit**.")
    )

    with mo.status.spinner(
        title=f"Fitting glmhmm K={ui_K.value} for {ui_subjects.value}..."
    ):
        fit_main(
            subjects=ui_subjects.value,
            K_list=[ui_K.value],
            out_dir=paths.RESULTS / "fits/glmhmm",
        )
    mo.md("✅ Fit complete — plots below update automatically.")
    return


@app.cell
def _(df_all, np, paths, pl, ui_K, ui_subjects):
    from glmhmmt.features import build_sequence_from_df

    OUT = paths.RESULTS / "fits/glmhmm"
    K = ui_K.value

    # load feature names from data
    _df_sel = df_all.filter(pl.col("subject").is_in(ui_subjects.value)).sort(
        "trial_idx"
    )
    _, _, _, names, _ = build_sequence_from_df(_df_sel)

    arrays_store = {}
    for _subj in ui_subjects.value:
        _f = OUT / f"{_subj}_K{K}_glmhmm_arrays.npz"
        if _f.exists():
            arrays_store[_subj] = dict(np.load(_f))

    arrays_store
    return K, arrays_store, names


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


    _feat_names = names.get("X_cols", [])
    state_labels = {}  # subj -> {state_idx: label_str}
    state_order = {}  # subj -> [state_idx, ...] sorted by S_coh desc

    for _subj in _selected:
        _W = arrays_store[_subj].get("emission_weights")
        if _W is None:
            state_labels[_subj] = {k: f"State {k + 1}" for k in range(K)}
            state_order[_subj] = list(range(K))
            continue
        _scores = _scoh_score(_W, _feat_names)
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

    state_labels, state_order
    return (state_labels,)


@app.cell
def _(K, arrays_store, mo, names, np, pd, plt, sns, state_labels, ui_subjects):
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
    _AG_GROUPS = [
        # bias: L/R context indicators
        ("bias_coh", [("biasL", 0), ("biasR", 1)]),
        ("bias_incoh", [("biasL", 1), ("biasR", 0)]),
        # onset
        ("onset_coh", [("onsetL", 0), ("onsetR", 1)]),
        ("onset_incoh", [("onsetL", 1), ("onsetR", 0)]),
        ("onsetC", [("onsetC", "neg_mean")]),
        # delay (shared scalar)
        ("delay", [("delay", "mean")]),
        # delay × side
        ("D_coh", [("DL", 0), ("DR", 1)]),
        ("D_incoh", [("DL", 1), ("DR", 0)]),
        ("DC", [("DC", "neg_mean")]),
        # stimulus
        ("S_coh", [("SL", 0), ("SR", 1)]),
        ("S_incoh", [("SL", 1), ("SR", 0)]),
        ("SC", [("SC", "neg_mean")]),
        # stimulus × delay
        ("Sxd_coh", [("SLxdelay", 0), ("SRxdelay", 1)]),
        ("Sxd_incoh", [("SLxdelay", 1), ("SRxdelay", 0)]),
        ("SCxd", [("SCxdelay", "neg_mean")]),
        # action history (perseveration vs alternation)
        ("A_coh", [("A_L", 0), ("A_R", 1)]),
        ("A_incoh", [("A_L", 1), ("A_R", 0)]),
    ]
    _CLS_LABELS_ALL = ["Left (vs C)", "Right (vs C)"]

    _records = []  # raw per-class records (for 3-panel plot)
    _ag_records = []  # collapsed agonist records (for single-panel plot)

    for _subj in ui_subjects.value:
        if _subj not in arrays_store:
            continue
        _W = arrays_store[_subj]["emission_weights"]  # (K, 2, n_feat)
        _n_fit_feat = _W.shape[2]
        _feat_names = names["X_cols"][:_n_fit_feat]
        _fname2idx = {f: i for i, f in enumerate(_feat_names)}

        for _k in range(_W.shape[0]):
            # per-class records (keep as-is)
            for _c in range(_W.shape[1]):
                for _fi, _fname in enumerate(_feat_names):
                    _records.append(
                        {
                            "subject": _subj,
                            "state": state_labels.get(_subj, {}).get(
                                _k, f"State {_k}"
                            ),
                            "class": _c,
                            "class_label": _CLS_LABELS_ALL[_c]
                            if _c < len(_CLS_LABELS_ALL)
                            else f"Class {_c}",
                            "feature": _fname,
                            "weight": float(_W[_k, _c, _fi]),
                        }
                    )

            # agonist collapse per group
            for _grp_label, _members in _AG_GROUPS:
                _vals = []
                for _fname, _mode in _members:
                    if _fname not in _fname2idx:
                        continue
                    _fi = _fname2idx[_fname]
                    if isinstance(_mode, int):
                        _vals.append(float(_W[_k, _mode, _fi]))
                    elif _mode == "neg_mean":
                        _vals.append(-float(np.mean(_W[_k, :, _fi])))
                    else:  # "mean"
                        _vals.append(float(np.mean(_W[_k, :, _fi])))
                if _vals:
                    _ag_records.append(
                        {
                            "subject": _subj,
                            "state": state_labels.get(_subj, {}).get(
                                _k, f"State {_k}"
                            ),
                            "feature": _grp_label,
                            "weight": float(np.mean(_vals)),
                        }
                    )

    mo.stop(not _records, mo.md("No fitted arrays found — run the fit first."))

    _df_w = pd.DataFrame(_records)
    _df_ag = pd.DataFrame(_ag_records)
    # preserve group order
    _ag_order = [g for g, _ in _AG_GROUPS if g in _df_ag["feature"].values]

    # ── 1. Collapsed (agonist) plot ────────────────────────────────────────────
    _fig_ag, _ax_ag = plt.subplots(figsize=(len(_ag_order) * 0.75, 4))
    sns.lineplot(
        data=_df_ag,
        x="feature",
        y="weight",
        hue="state",
        ax=_ax_ag,
        markers=True,
        marker="o",
        markersize=8,
        markeredgewidth=0,
        alpha=0.85,
        errorbar="se",
    )
    _ax_ag.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    _ax_ag.set_xticks(range(len(_ag_order)))
    _ax_ag.set_xticklabels(_ag_order, rotation=35, ha="right")
    _ax_ag.set_xlabel("")
    _ax_ag.set_ylabel("Agonist weight")
    _ax_ag.set_title(f"Emission weights – agonist view  (K={K})")
    if _ax_ag.get_legend() is not None:
        _ax_ag.get_legend().set_title("")
    _fig_ag.tight_layout()
    sns.despine(fig=_fig_ag)

    # ── 2. Per-class plot (L / C / R) ─────────────────────────────────────────
    _n_classes = _df_w["class"].nunique()
    _fig_cls, _axes_cls = plt.subplots(
        1, _n_classes, figsize=(6 * _n_classes, 4), sharey=True
    )
    _axes_cls = np.atleast_1d(_axes_cls)

    for _c, _ax in enumerate(_axes_cls):
        _sub = _df_w[_df_w["class"] == _c]
        sns.lineplot(
            data=_sub,
            x="feature",
            y="weight",
            hue="state",
            ax=_ax,
            markers=True,
            marker="o",
            markersize=8,
            markeredgewidth=0,
            alpha=0.8,
            errorbar="se",
        )
        _ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        _ax.set_title(
            _CLS_LABELS_ALL[_c] if _c < len(_CLS_LABELS_ALL) else f"Class {_c}"
        )
        _ax.set_xticks(range(len(_feat_names)))
        _ax.set_xticklabels(_feat_names, rotation=35, ha="right")
        _ax.set_xlabel("")
        _ax.set_ylabel("Weight" if _c == 0 else "")
        if _ax.get_legend() is not None:
            _ax.get_legend().set_title("")

    _fig_cls.suptitle(f"Emission weights per choice  (K={K})", y=1.02)
    _fig_cls.tight_layout()
    sns.despine(fig=_fig_cls)

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
        sns.heatmap(
            _A,
            ax=_ax_t,
            cmap="bone",
            annot=True,
            fmt=".2f",
            vmin=0,
            vmax=1,
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
    _T_max = (
        max(arrays_store[s]["smoothed_probs"].shape[0] for s in _selected)
        if _selected
        else 200
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
    _fig_p, _axes_p = plt.subplots(
        _n_subj, 1, figsize=(14, 3 * _n_subj), squeeze=False
    )

    for _i, _subj in enumerate(_selected):
        _ax = _axes_p[_i, 0]
        _probs = arrays_store[_subj]["smoothed_probs"][
            _t0 : _t1 + 1
        ]  # (window, K)
        _y = arrays_store[_subj]["y"].astype(int)[_t0 : _t1 + 1]  # (window,)
        _T_w = _probs.shape[0]
        _x = np.arange(_t0, _t0 + _T_w)

        # stacked area — color by label rank so Engaged is always palette[0]
        _colors = sns.color_palette("tab10", n_colors=K)
        _label_rank = {
            "Engaged": 0,
            "Disengaged": 1,
            **{f"Disengaged {i}": i for i in range(1, K)},
        }
        _bottom = np.zeros(_T_w)
        _slbl = state_labels.get(_subj, {k: f"State {k}" for k in range(K)})
        for _k in range(K):
            _rank = _label_rank.get(_slbl.get(_k, ""), _k)
            _ax.fill_between(
                _x,
                _bottom,
                _bottom + _probs[:, _k],
                alpha=0.7,
                color=_colors[_rank],
                label=_slbl.get(_k, f"State {_k}"),
            )
            _bottom += _probs[:, _k]

        # choice markers on top
        _choice_colors = {0: "royalblue", 1: "gold", 2: "tomato"}
        _choice_labels = {0: "L", 1: "C", 2: "R"}
        for _resp, _col in _choice_colors.items():
            _mask = _y == _resp
            _ax.scatter(
                _x[_mask],
                np.ones(_mask.sum()) * 1.03,
                c=_col,
                s=4,
                marker="|",
                label=_choice_labels[_resp],
                transform=_ax.get_xaxis_transform(),
                clip_on=False,
            )

        _ax.set_xlim(_t0, _t0 + _T_w - 1)
        _ax.set_ylim(0, 1)
        _ax.set_ylabel("State probability")
        _ax.set_title(f"Subject {_subj}")
        _ax.legend(
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=8,
            ncol=1,
            frameon=False,
        )

    _axes_p[-1, 0].set_xlabel("Trial")
    _fig_p.tight_layout()
    _fig_p.subplots_adjust(right=0.85)
    sns.despine(fig=_fig_p)
    mo.vstack(
        [
            mo.md(f"### Posterior state probabilities  (K={K})"),
            _fig_p,
        ],
        align="center",
    )
    return


@app.cell
def _(K, arrays_store, df_all, mo, np, pl, state_labels, ui_subjects):
    import glmhmmt.plots as plots

    # ── predictions & categorical performance ────────────────────────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _frames = []
    for _subj in _selected:
        _p_pred = arrays_store[_subj]["p_pred"]  # (T, 3): pL, pC, pR
        _df_sub = (
            df_all.filter(pl.col("subject") == _subj)
            .sort("trial_idx")
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
    _lrank_map = {
        "Engaged": 0,
        "Disengaged": 1,
        **{f"Disengaged {i}": i for i in range(1, K)},
    }
    _pool_dfs = []
    _pool_assigns = []
    for _subj in _selected:
        _p_pred_s = arrays_store[_subj]["p_pred"]
        _df_s = (
            df_all.filter(pl.col("subject") == _subj)
            .sort("trial_idx")
            .filter(pl.col("session").count().over("session") >= 2)
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
        _T_s = min(_plot_df_s.height, _gamma_s.shape[0])
        _pool_dfs.append(_plot_df_s[:_T_s])
        _slbls = state_labels[_subj]
        _raw = np.argmax(_gamma_s[:_T_s], axis=1).astype(int)
        _norm = np.array([_lrank_map.get(_slbls.get(int(k), ""), k) for k in _raw])
        _pool_assigns.append(_norm)

    _df_state_pool = pl.concat(_pool_dfs)
    _assign_pool = np.concatenate(_pool_assigns)
    _state_lbl_global = {
        0: "Engaged",
        1: "Disengaged",
        **{i: f"Disengaged {i}" for i in range(2, K)},
    }
    _fig_state, _ = plots.plot_categorical_performance_by_state(
        df=_df_state_pool,
        smoothed_probs=None,
        state_assign=_assign_pool,
        state_labels=_state_lbl_global,
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
def _(
    K,
    arrays_store,
    df_all,
    mo,
    np,
    pd,
    pl,
    plt,
    sns,
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

    _THRESH = 0.9
    _palette = sns.color_palette("tab10", n_colors=K)

    _label_order = (
        ["Engaged", "Disengaged"]
        if K == 2
        else ["Engaged"] + [f"Disengaged {i}" for i in range(1, K)]
    )
    _cmap = {"All": "#999999"}
    for _ri, _lbl in enumerate(_label_order):
        _cmap[_lbl] = _palette[_ri]

    _x_labels = ["All"] + _label_order

    # ── collect per-subject accuracies ────────────────────────────────────────
    _acc_records = []
    for _subj in _selected_acc:
        _arr = arrays_store[_subj]
        _gamma = _arr.get("smoothed_probs")  # (T, K)
        if _gamma is None:
            continue

        _df_sub = (
            df_all.filter(pl.col("subject") == _subj)
            .sort("trial_idx")
            .filter(pl.col("session").count().over("session") >= 2)
            .select(["stimd_n", "performance"])
        )
        _stim = _df_sub["stimd_n"].to_numpy()  # signed, 0 = catch
        _perf = _df_sub["performance"].to_numpy()  # 0/1

        _T = min(len(_stim), _gamma.shape[0])
        _stim = _stim[:_T]
        _perf = _perf[:_T]
        _gamma = _gamma[:_T]

        # base mask: non-zero stimulus (same for All and every state)
        _nz = _stim != 0
        if _nz.sum() == 0:
            continue

        # ── All: full nonzero pool ────────────────────────────────────────────
        _acc_records.append(
            {
                "subject": _subj,
                "label": "All",
                "acc": float(_perf[_nz].mean() * 100),
                "n": int(_nz.sum()),
            }
        )

        # ── per state: subset of the same nonzero pool ────────────────────────
        _s_labels = state_labels[_subj]
        for _k in range(K):
            _lbl_k = _s_labels[_k]
            # subset: nonzero stim AND high-confidence in state k
            _mask_k = _nz & (_gamma[:, _k] >= _THRESH)
            _n_k = int(_mask_k.sum())
            _acc_k = (
                float(_perf[_mask_k].mean() * 100) if _n_k > 0 else float("nan")
            )

            _acc_records.append(
                {
                    "subject": _subj,
                    "label": _lbl_k,
                    "acc": _acc_k,
                    "n": _n_k,
                }
            )

    _df_acc = pd.DataFrame(_acc_records)

    # ── sanity check printed as marimo table ──────────────────────────────────
    # shows n_trials per state so you can verify the subset relationship
    _tbl = (
        _df_acc.groupby("label")[["acc", "n"]]
        .agg({"acc": "mean", "n": "sum"})
        .reindex(_x_labels)
        .rename(columns={"acc": "mean_acc (%)", "n": "total_trials"})
        .round(1)
    )

    # ── plot ──────────────────────────────────────────────────────────────────
    _fig_acc, _ax_acc = plt.subplots(figsize=(2 + len(_x_labels) * 0.9, 4))
    _rng = np.random.default_rng(42)

    for _li, _lbl in enumerate(_x_labels):
        _vals = _df_acc[_df_acc["label"] == _lbl]["acc"].dropna().values
        if len(_vals) == 0:
            continue
        _mean = float(_vals.mean())
        _sem = (
            float(_vals.std(ddof=1) / np.sqrt(len(_vals)))
            if len(_vals) > 1
            else 0.0
        )

        _ax_acc.bar(
            _li,
            _mean,
            color=_cmap.get(_lbl, "#999999"),
            yerr=_sem,
            error_kw={"linewidth": 1.2, "capsize": 4},
            width=0.6,
            alpha=0.9,
            zorder=2,
        )
        _ax_acc.text(
            _li,
            _mean + _sem + 1.2,
            f"{_mean:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        _jitter = _rng.uniform(-0.15, 0.15, size=len(_vals))
        _ax_acc.scatter(
            _li + _jitter,
            _vals,
            color="black",
            s=20,
            zorder=5,
            alpha=0.6,
        )

    _ax_acc.axhline(
        100 / 3,
        color="black",
        linestyle="--",
        linewidth=0.9,
        alpha=0.5,
        label="Chance (33%)",
    )
    _ax_acc.set_xticks(range(len(_x_labels)))
    _ax_acc.set_xticklabels(_x_labels, rotation=20, ha="right")
    _ax_acc.set_xlabel("State")
    _ax_acc.set_ylabel("Accuracy (%)")
    _ax_acc.set_ylim(30, 105)
    _ax_acc.set_title(
        f"Per-state accuracy  (K={K},  posterior ≥ {_THRESH},  non-zero stim)"
    )
    _ax_acc.legend(frameon=False, fontsize=8)
    _fig_acc.tight_layout()
    sns.despine(fig=_fig_acc)

    mo.vstack(
        [
            mo.md("### Accuracy by state"),
            mo.md(
                "> **All** = full nonzero-stim pool · **State bars** = subsets where posterior ≥ 0.9"
            ),
            _fig_acc,
            mo.md("**Trial counts & mean accuracy per label:**"),
            mo.plain_text(_tbl.to_string()),
        ]
    )
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
    arrays_store,
    df_all,
    mo,
    np,
    pl,
    plt,
    sns,
    state_labels,
    ui_subjects_traj,
):
    # ── c. Average state-probability trajectories within a session ────────────
    # For each selected subject: align smoothed_probs with (session, trial)
    # info, compute mean ± s.e.m. across sessions, plot one line per state.
    _selected_traj = [s for s in ui_subjects_traj.value if s in arrays_store]
    mo.stop(
        not _selected_traj,
        mo.md("Select subjects above to view session trajectories."),
    )

    _palette_traj = sns.color_palette("tab10", n_colors=K)
    _label_rank_traj = {
        "Engaged": 0,
        "Disengaged": 1,
        **{f"Disengaged {i}": i for i in range(1, K)},
    }

    _n_subj_traj = len(_selected_traj)
    _fig_traj, _axes_traj = plt.subplots(
        _n_subj_traj, 1, figsize=(10, 3.5 * _n_subj_traj), squeeze=False
    )

    for _i, _subj in enumerate(_selected_traj):
        _ax = _axes_traj[_i, 0]
        _probs = arrays_store[_subj]["smoothed_probs"]  # (T, K)
        _df_sub = (
            df_all.filter(pl.col("subject") == _subj)
            .sort("trial_idx")
            .filter(pl.col("session").count().over("session") >= 2)
            .select(["session", "trial"])
        )
        _sessions = _df_sub["session"].to_numpy()
        _trials = _df_sub["trial"].to_numpy()

        _sess_ids = np.unique(_sessions)
        _max_len = max(int((_sessions == _s).sum()) for _s in _sess_ids)

        # (n_sessions, max_len, K) — NaN-padded
        _mat = np.full((_sess_ids.size, _max_len, K), np.nan)
        for _si, _s in enumerate(_sess_ids):
            _mask = _sessions == _s
            _p_s = _probs[_mask]
            _order = np.argsort(_trials[_mask])
            _p_s = _p_s[_order]
            _mat[_si, : _p_s.shape[0], :] = _p_s

        _mean = np.nanmean(_mat, axis=0)  # (max_len, K)
        _n_obs = np.sum(~np.isnan(_mat[:, :, 0]), axis=0)  # (max_len,)
        _sem = np.nanstd(_mat, axis=0, ddof=1) / np.maximum(
            _n_obs[:, None] ** 0.5, 1
        )
        _x = np.arange(_max_len)

        _slbl = state_labels.get(_subj, {k: f"State {k}" for k in range(K)})
        for _k in range(K):
            _rank = _label_rank_traj.get(_slbl.get(_k, ""), _k)
            _col = _palette_traj[_rank % len(_palette_traj)]
            _lbl = _slbl.get(_k, f"State {_k}")
            _valid = ~np.isnan(_mean[:, _k])
            _ax.plot(_x[_valid], _mean[_valid, _k], color=_col, lw=2, label=_lbl)
            _ax.fill_between(
                _x[_valid],
                (_mean[:, _k] - _sem[:, _k])[_valid],
                (_mean[:, _k] + _sem[:, _k])[_valid],
                color=_col,
                alpha=0.25,
            )

        _ax.set_ylim(0, 1)
        _ax.set_xlabel("Trial within session")
        _ax.set_ylabel("State probability")
        _ax.set_title(
            f"Subject {_subj} — avg. state trajectory  "
            f"(n={_sess_ids.size} sessions)"
        )
        _ax.legend(
            bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False
        )

    _fig_traj.tight_layout()
    sns.despine(fig=_fig_traj)

    mo.vstack(
        [
            mo.md(
                f"### c. Average state-probability trajectories within a session  (K={K})"
            ),
            mo.md("> Mean ± 1 s.e.m. across sessions for the selected subjects."),
            _fig_traj,
        ],
        align="center",
    )
    return


@app.cell
def _(
    K,
    arrays_store,
    df_all,
    mo,
    np,
    pl,
    plt,
    sns,
    state_labels,
    ui_subjects_traj,
):
    # ── d. Fractional occupancy & state-change histogram ─────────────────────
    # Left panel : fraction of trials assigned to each state (argmax posterior).
    # Right panel: histogram of #state-changes per session.
    _selected_occ = [s for s in ui_subjects_traj.value if s in arrays_store]
    mo.stop(not _selected_occ, mo.md("Select subjects above."))

    _palette_occ = sns.color_palette("tab10", n_colors=K)
    _label_rank_occ = {
        "Engaged": 0,
        "Disengaged": 1,
        **{f"Disengaged {i}": i for i in range(1, K)},
    }

    _n_subj_occ = len(_selected_occ)
    _fig_occ, _axes_occ = plt.subplots(_n_subj_occ, 2, figsize=(10, 3.5 * _n_subj_occ), squeeze=False)

    for _i, _subj in enumerate(_selected_occ):
        _ax_bar = _axes_occ[_i, 0]
        _ax_hist = _axes_occ[_i, 1]

        _probs = arrays_store[_subj]["smoothed_probs"]  # (T, K)
        _df_sub = (
            df_all.filter(pl.col("subject") == _subj)
            .sort("trial_idx")
            .filter(pl.col("session").count().over("session") >= 2)
            .select(["session", "trial"])
        )
        _sessions = _df_sub["session"].to_numpy()
        _state_assign = np.argmax(_probs, axis=1)  # (T,) most-likely state

        _slbl = state_labels.get(_subj, {k: f"State {k}" for k in range(K)})

        # ── fractional occupancy bar plot ──────────────────────────────────
        _fracs = np.array([np.mean(_state_assign == _k) for _k in range(K)])
        _bar_labels = [_slbl.get(_k, f"State {_k}") for _k in range(K)]
        _bar_colors = [
            _palette_occ[
                _label_rank_occ.get(_slbl.get(_k, ""), _k) % len(_palette_occ)
            ]
            for _k in range(K)
        ]
        _rects = _ax_bar.bar(
            range(K), _fracs, color=_bar_colors, width=0.6, alpha=0.9
        )
        for _xi, _fv in enumerate(_fracs):
            _ax_bar.text(
                _xi,
                _fv + 0.01,
                f"{_fv:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        _ax_bar.set_xticks(range(K))
        _ax_bar.set_xticklabels(_bar_labels, rotation=15, ha="right")
        _ax_bar.set_ylim(0, 1.15)
        _ax_bar.set_ylabel("Fractional occupancy")
        _ax_bar.set_title(f"Subject {_subj} — state occupancy")

        # ── state-change histogram per session ──────────────────────────────
        _sess_ids = np.unique(_sessions)
        _n_changes = [
            int(np.sum(np.diff(_state_assign[_sessions == _s]) != 0))
            for _s in _sess_ids
        ]
        _max_changes = max(_n_changes) if _n_changes else 0
        _ax_hist.hist(
            _n_changes,
            bins=_max_changes + 1,
            range=(-0.5, _max_changes + 0.5),
            color=_palette_occ[0],
            edgecolor="white",
            alpha=0.85,
        )
        _ax_hist.set_xlabel("State changes per session")
        _ax_hist.set_ylabel("Number of sessions")
        _ax_hist.set_title(f"Subject {_subj} — state changes / session")

    _fig_occ.tight_layout()
    sns.despine(fig=_fig_occ)

    mo.vstack(
        [
            mo.md(
                f"### d. Fractional occupancy & state changes per session  (K={K})"
            ),
            mo.md(
                "> **Left**: fraction of all trials assigned to each state (argmax of posterior).  \n"
                "> **Right**: histogram of inferred state changes per session."
            ),
            _fig_occ,
        ],
        align="center",
    )
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
            .filter(pl.col("session").count().over("session") >= 2)
            ["session"].unique().to_list()
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
    arrays_store,
    df_all,
    mo,
    names,
    np,
    pl,
    plt,
    sns,
    state_labels,
    ui_session_id,
    ui_session_subj,
):
    # ── Session deep-dive plot ─────────────────────────────────────────────────
    # 3-panel figure for the selected subject × session:
    #   Panel 1 – smoothed state probabilities (Engaged / Disengaged)
    #   Panel 2 – action traces A_L, A_C, A_R
    #   Panel 3 – cumulative mean accuracy (non-zero-stim trials)

    _subj = ui_session_subj.value
    mo.stop(
        _subj not in arrays_store,
        mo.md("No fitted arrays for this subject — run the fit first."),
    )

    _sess = int(ui_session_id.value)
    _df_sub = (
        df_all.filter(
            (pl.col("subject") == _subj) & (pl.col("session") == _sess)
        )
        .sort("trial_idx")
    )
    mo.stop(len(_df_sub) == 0, mo.md("No trials found for this session."))

    # ── align smoothed_probs rows with this session ───────────────────────────
    _df_all_sub = (
        df_all.filter(pl.col("subject") == _subj)
        .sort("trial_idx")
        .filter(pl.col("session").count().over("session") >= 2)
    )
    _all_sessions = _df_all_sub["session"].to_numpy()
    _sess_mask = _all_sessions == _sess

    _probs_all = arrays_store[_subj]["smoothed_probs"]   # (T_total, K)
    _probs = _probs_all[_sess_mask]                        # (T_sess, K)

    _y = _df_sub["performance"].to_numpy()                 # 0/1 per trial
    _stim = _df_sub["stimd_n"].to_numpy()                  # 0 for catch
    _response = _df_sub["response"].to_numpy().astype(int) # 0=L,1=C,2=R
    _T = _probs.shape[0]
    _x = np.arange(_T)

    # ── action traces from X ─────────────────────────────────────────────────
    _X_all = arrays_store[_subj]["X"]                      # (T_total, n_feat)
    _X_sess = _X_all[_sess_mask]                            # (T_sess, n_feat)
    _feat_names = names.get("X_cols", [])
    _fname2idx = {f: i for i, f in enumerate(_feat_names)}
    _trace_cols = [c for c in ["A_L", "A_C", "A_R"] if c in _fname2idx]
    _trace_colors = {"A_L": "royalblue", "A_C": "gold", "A_R": "tomato"}

    # ── cumulative accuracy (non-zero stim) ───────────────────────────────────
    _nz = _stim != 0
    _cum_acc = np.full(_T, np.nan)
    _cum_n = 0
    _cum_sum = 0.0
    for _ti in range(_T):
        if _nz[_ti]:
            _cum_sum += _y[_ti]
            _cum_n += 1
        if _cum_n > 0:
            _cum_acc[_ti] = 100.0 * _cum_sum / _cum_n

    # ── figure ────────────────────────────────────────────────────────────────
    _palette = sns.color_palette("tab10", n_colors=K)
    _label_rank = {
        "Engaged": 0, "Disengaged": 1,
        **{f"Disengaged {i}": i for i in range(1, K)},
    }
    _slbl = state_labels.get(_subj, {k: f"State {k}" for k in range(K)})

    # Find which column index corresponds to Engaged
    _engaged_k = next(
        (k for k in range(K) if _label_rank.get(_slbl.get(k, ""), k) == 0), 0
    )
    _engaged_col = _palette[0]

    _fig, (_ax1, _ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5]},
    )

    # Panel 1 – P(Engaged) line + cumulative accuracy on twin axis
    _ax1.plot(_x, _probs[:, _engaged_k], color=_engaged_col, lw=2,
              label=f"P({_slbl.get(_engaged_k, 'Engaged')})")

    # choice tick marks
    _choice_cols = {0: "royalblue", 1: "gold", 2: "tomato"}
    _choice_lbls = {0: "L", 1: "C", 2: "R"}
    for _resp, _c in _choice_cols.items():
        _m = _response == _resp
        _ax1.scatter(_x[_m], np.ones(_m.sum()) * 1.03, c=_c, s=5, marker="|",
                     label=_choice_lbls[_resp],
                     transform=_ax1.get_xaxis_transform(), clip_on=False)
    _ax1.set_ylim(0, 1)
    _ax1.set_ylabel("State probability")
    _ax1.set_title(f"Subject {_subj}  —  session {_sess}  ({_T} trials)")
    # twin axis: cumulative accuracy
    _ax1r = _ax1.twinx()
    _ax1r.plot(_x, _cum_acc, color="black", lw=1.8, linestyle="-", alpha=0.7,
               label="Cumul. accuracy")
    _ax1r.axhline(100 / 3, color="grey", lw=0.9, linestyle="--", alpha=0.5)
    _ax1r.set_ylim(0, 105)
    _ax1r.set_ylabel("Accuracy (%)", color="black")
    # combined legend
    _lines1, _labs1 = _ax1.get_legend_handles_labels()
    _lines1r, _labs1r = _ax1r.get_legend_handles_labels()
    _ax1.legend(_lines1 + _lines1r, _labs1 + _labs1r,
                bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)

    # Panel 2 – action traces
    if _trace_cols:
        for _tc in _trace_cols:
            _ax2.plot(_x, _X_sess[:, _fname2idx[_tc]],
                      label=_tc, color=_trace_colors.get(_tc),
                      lw=1.5, alpha=0.85)
    else:
        _ax2.text(0.5, 0.5, "No action-trace features found in X",
                  ha="center", va="center", transform=_ax2.transAxes)
    _ax2.set_ylabel("Action trace")
    _ax2.set_ylim(0, None)
    _ax2.set_xlabel("Trial within session")
    _ax2.legend(bbox_to_anchor=(1.08, 1), loc="upper left", fontsize=8, frameon=False)

    _fig.tight_layout()
    _fig.subplots_adjust(right=0.82)
    sns.despine(fig=_fig, right=False)

    mo.vstack([
        mo.md(f"### Session deep-dive  (K={K})"),
        mo.hstack([ui_session_subj, ui_session_id]),
        _fig,
    ], align="center")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
