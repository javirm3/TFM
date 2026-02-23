import marimo

__generated_with = "0.19.11"
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
        options=df_all["subject"].unique(),  # replace with dynamic list
        label="Subjects",
    )
    fit_button = mo.ui.run_button(label="Run fit")
    mo.hstack([ui_K, ui_subjects, fit_button])
    return fit_button, ui_K, ui_subjects


@app.cell
def _(fit_button, mo, paths, ui_K, ui_subjects):
    from scripts.fit_glmhmm import main as fit_main

    mo.stop(not fit_button.value, mo.md("Configure parameters and press **Run fit**."))

    with mo.status.spinner(title=f"Fitting glmhmm K={ui_K.value} for {ui_subjects.value}..."):
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
    _df_sel = df_all.filter(pl.col("subject").is_in(ui_subjects.value)).sort("trial_idx")
    _, _, _, names, _ = build_sequence_from_df(_df_sel)

    arrays_store = {}
    for _subj in ui_subjects.value:
        _f = OUT / f"{_subj}_K{K}_glmhmm_arrays.npz"
        if _f.exists():
            arrays_store[_subj] = dict(np.load(_f))

    arrays_store
    return K, arrays_store, names


@app.cell
def _(K, arrays_store, mo, names, np, pd, plt, sns, ui_subjects):
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
        ("bias_coh",       [("biasL", 0), ("biasR", 1)]),
        ("bias_incoh",     [("biasL", 1), ("biasR", 0)]),
        # onset
        ("onset_coh",      [("onsetL", 0), ("onsetR", 1)]),
        ("onset_incoh",    [("onsetL", 1), ("onsetR", 0)]),
        ("onsetC",         [("onsetC", "neg_mean")]),
        # delay (shared scalar)
        ("delay",          [("delay",  "mean")]),
        # delay × side
        ("D_coh",          [("DL", 0), ("DR", 1)]),
        ("D_incoh",        [("DL", 1), ("DR", 0)]),
        ("DC",             [("DC",    "neg_mean")]),
        # stimulus
        ("S_coh",          [("SL", 0), ("SR", 1)]),
        ("S_incoh",        [("SL", 1), ("SR", 0)]),
        ("SC",             [("SC",    "neg_mean")]),
        # stimulus × delay
        ("Sxd_coh",        [("SLxdelay", 0), ("SRxdelay", 1)]),
        ("Sxd_incoh",      [("SLxdelay", 1), ("SRxdelay", 0)]),
        ("SCxd",           [("SCxdelay", "neg_mean")]),
        # action history (perseveration vs alternation)
        ("A_coh",          [("A_L", 0), ("A_R", 1)]),
        ("A_incoh",        [("A_L", 1), ("A_R", 0)]),
    ]
    _CLS_LABELS_ALL = ["Left (vs C)", "Right (vs C)"]

    _records = []    # raw per-class records (for 3-panel plot)
    _ag_records = [] # collapsed agonist records (for single-panel plot)

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
                    _records.append({
                        "subject": _subj,
                        "state": f"State {_k}",
                        "class": _c,
                        "class_label": _CLS_LABELS_ALL[_c] if _c < len(_CLS_LABELS_ALL) else f"Class {_c}",
                        "feature": _fname,
                        "weight": float(_W[_k, _c, _fi]),
                    })

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
                    _ag_records.append({
                        "subject": _subj,
                        "state": f"State {_k}",
                        "feature": _grp_label,
                        "weight": float(np.mean(_vals)),
                    })

    mo.stop(not _records, mo.md("No fitted arrays found — run the fit first."))

    _df_w  = pd.DataFrame(_records)
    _df_ag = pd.DataFrame(_ag_records)
    # preserve group order
    _ag_order = [g for g, _ in _AG_GROUPS if g in _df_ag["feature"].values]

    # ── 1. Collapsed (agonist) plot ────────────────────────────────────────────
    _fig_ag, _ax_ag = plt.subplots(figsize=(len(_ag_order) * 0.75, 4))
    sns.lineplot(
        data=_df_ag,
        x="feature", y="weight",
        hue="state",
        ax=_ax_ag,
        markers=True, marker="o",
        markersize=8, markeredgewidth=0,
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
    _fig_cls, _axes_cls = plt.subplots(1, _n_classes, figsize=(6 * _n_classes, 4), sharey=True)
    _axes_cls = np.atleast_1d(_axes_cls)

    for _c, _ax in enumerate(_axes_cls):
        _sub = _df_w[_df_w["class"] == _c]
        sns.lineplot(
            data=_sub,
            x="feature", y="weight",
            hue="state",
            ax=_ax,
            markers=True, marker="o",
            markersize=8, markeredgewidth=0,
            alpha=0.8,
            errorbar="se",
        )
        _ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        _ax.set_title(_CLS_LABELS_ALL[_c] if _c < len(_CLS_LABELS_ALL) else f"Class {_c}")
        _ax.set_xticks(range(len(_feat_names)))
        _ax.set_xticklabels(_feat_names, rotation=35, ha="right")
        _ax.set_xlabel("")
        _ax.set_ylabel("Weight" if _c == 0 else "")
        if _ax.get_legend() is not None:
            _ax.get_legend().set_title("")

    _fig_cls.suptitle(f"Emission weights per choice  (K={K})", y=1.02)
    _fig_cls.tight_layout()
    sns.despine(fig=_fig_cls)

    mo.vstack([_fig_ag, _fig_cls])
    return


@app.cell
def _(K, arrays_store, np, plt, sns, ui_subjects):
    # ── transition matrix heatmap ─────────────────────────────────────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    _n = len(_selected)
    if _n == 0:
        _fig_t = plt.figure()
    else:
        _fig_t, _axes_t = plt.subplots(1, _n, figsize=(3.8 * _n, 3.5))
        _axes_t = np.atleast_1d(_axes_t)
        for _ax_t, _subj in zip(_axes_t, _selected):
            _A = arrays_store[_subj]["transition_matrix"]  # (K, K)
            sns.heatmap(
                _A,
                ax=_ax_t,
                cmap="bone",
                annot=True, fmt=".2f",
                vmin=0, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "probability"},
            )
            _ax_t.set_title(f"Subject {_subj}")
            _ax_t.set_xlabel("To state")
            _ax_t.set_ylabel("From state")
        _fig_t.suptitle(f"Transition Matrix  (K={K})", y=1.02)
        _fig_t.tight_layout()
    _fig_t
    return


@app.cell
def _(K, arrays_store, mo, np, plt, sns, ui_subjects):
    # ── posterior state probabilities ─────────────────────────────────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _n_subj = len(_selected)
    _fig_p, _axes_p = plt.subplots(_n_subj, 1, figsize=(14, 3 * _n_subj), squeeze=False)

    for _i, _subj in enumerate(_selected):
        _ax = _axes_p[_i, 0]
        _probs = arrays_store[_subj]["smoothed_probs"]  # (T, K)
        _y    = arrays_store[_subj]["y"].astype(int)    # (T,)
        _T    = _probs.shape[0]
        _x    = np.arange(_T)

        # stacked area
        _colors = sns.color_palette("tab10", n_colors=K)
        _bottom = np.zeros(_T)
        for _k in range(K):
            _ax.fill_between(_x, _bottom, _bottom + _probs[:, _k],
                             alpha=0.7, color=_colors[_k], label=f"State {_k}")
            _bottom += _probs[:, _k]

        # choice markers on top
        _choice_colors = {0: "royalblue", 1: "gold", 2: "tomato"}
        _choice_labels = {0: "L", 1: "C", 2: "R"}
        for _resp, _col in _choice_colors.items():
            _mask = _y == _resp
            _ax.scatter(_x[_mask], np.ones(_mask.sum()) * 1.03,
                        c=_col, s=4, marker="|", label=_choice_labels[_resp],
                        transform=_ax.get_xaxis_transform(), clip_on=False)

        _ax.set_xlim(0, _T - 1)
        _ax.set_ylim(0, 1)
        _ax.set_ylabel("State probability")
        _ax.set_title(f"Subject {_subj}")
        _ax.legend(loc="upper right", fontsize=8, ncol=K + 3,
                   frameon=False)

    _axes_p[-1, 0].set_xlabel("Trial")
    _fig_p.suptitle(f"Posterior state probabilities  (K={K})", y=1.01)
    _fig_p.tight_layout()
    sns.despine(fig=_fig_p)
    _fig_p
    return


@app.cell
def _(K, arrays_store, df_all, mo, np, pl, ui_subjects):
    import glmhmmt.plots as plots

    # ── predictions & categorical performance ────────────────────────────────
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _frames = []
    for _subj in _selected:
        _p_pred = arrays_store[_subj]["p_pred"]        # (T, 3): pL, pC, pR
        _df_sub = (
            df_all
            .filter(pl.col("subject") == _subj)
            .sort("trial_idx")
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
    plots.plot_categorical_performance_all(_plot_df, f"glmhmm K={K}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
