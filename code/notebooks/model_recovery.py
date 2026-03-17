import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = root / "glmhmmt" / "src"
    if str(root) not in sys.path:
        sys.path.append(str(root))
    if str(src) not in sys.path:
        sys.path.append(str(src))

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns

    import paths
    from dynamax.utils.utils import find_permutation
    from glmhmmt.model import SoftmaxGLMHMM
    from glmhmmt.views import build_views
    from tasks import get_adapter

    sns.set_style("white")
    return (
        Path,
        SoftmaxGLMHMM,
        build_views,
        find_permutation,
        get_adapter,
        gridspec,
        jax,
        jnp,
        jr,
        mo,
        np,
        paths,
        pd,
        pl,
        plt,
        sns,
    )


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        (
            "# GLM-HMM Model Recovery\n\n"
            "This notebook now uses the task adapters and `SubjectFitView` pipeline.\n\n"
            "Workflow:\n"
            "1. Choose a task, subject, and regressors\n"
            "2. Reuse that subject's real covariates and session structure\n"
            "3. Simulate synthetic choices from user-defined ground-truth GLM-HMM parameters\n"
            "4. Fit the same model family back to the synthetic data\n"
            "5. Compare true vs fitted states, posteriors, weights, and transitions\n"
        )
    )
    return


@app.cell(hide_code=True)
def _(get_adapter, mo):
    ui_task = mo.ui.dropdown(
        options={"MCDR": "MCDR", "2AFC": "2AFC"},
        value="MCDR",
        label="Task",
    )
    ui_task
    return (ui_task,)


@app.cell
def _(get_adapter, paths, pl, ui_task):
    adapter = get_adapter(ui_task.value)
    df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df_all = adapter.subject_filter(df_all)
    return adapter, df_all


@app.cell(hide_code=True)
def _(adapter, df_all, mo):
    subject_options = df_all["subject"].unique().sort().to_list()
    emission_options = adapter.default_emission_cols()
    if hasattr(adapter, "sf_cols"):
        emission_options = emission_options + [
            c for c in adapter.sf_cols(df_all) if c not in emission_options
        ]

    ui_subject = mo.ui.dropdown(
        options=subject_options,
        value=subject_options[0],
        label="Subject",
    )
    ui_K = mo.ui.slider(start=2, stop=4, value=2, label="K (num states)")
    ui_tau = mo.ui.slider(start=1, stop=200, value=50, step=1, label="Action half-life")
    ui_seed = mo.ui.number(start=0, stop=9999, value=42, step=1, label="Random seed")
    ui_num_iters = mo.ui.slider(start=10, stop=300, value=50, step=10, label="EM iterations")
    ui_restarts = mo.ui.slider(start=1, stop=10, value=5, step=1, label="Restarts")
    ui_emission_cols = mo.ui.multiselect(
        options=emission_options,
        value=adapter.default_emission_cols(),
        label="Emission regressors (X)",
    )
    ui_transition_cols = mo.ui.multiselect(
        options=adapter.default_transition_cols(),
        value=adapter.default_transition_cols(),
        label="Transition regressors (U)",
    )
    ui_run_fit = mo.ui.run_button(label="Run recovery")

    mo.vstack(
        [
            mo.md("### 1. Data and model configuration"),
            mo.hstack([ui_subject, ui_K, ui_tau, ui_seed]),
            mo.hstack([ui_num_iters, ui_restarts]),
            mo.hstack([ui_emission_cols, ui_transition_cols]),
            ui_run_fit,
        ],
        align="start",
    )
    return (
        ui_K,
        ui_emission_cols,
        ui_num_iters,
        ui_restarts,
        ui_run_fit,
        ui_seed,
        ui_subject,
        ui_tau,
        ui_transition_cols,
    )


@app.cell(hide_code=True)
def _(mo, paths, ui_K, ui_task):
    _fit_root = paths.RESULTS / "fits" / ui_task.value / "glmhmm"
    _fit_files = sorted(_fit_root.rglob(f"*_K{ui_K.value}_glmhmm_arrays.npz"))
    _fit_options = {
        f"{_p.stem.split(f'_K{ui_K.value}')[0]}  [{_p.parent.name}]": str(_p)
        for _p in _fit_files
    }
    ui_fit_subject = mo.ui.dropdown(
        options=_fit_options,
        value=next(iter(_fit_options)) if _fit_options else None,
        label="Preset from existing fit",
    )
    ui_load_fit = mo.ui.switch(label="Use fit as parameter preset", value=bool(_fit_options))
    mo.hstack([ui_fit_subject, ui_load_fit])
    return ui_fit_subject, ui_load_fit


@app.cell
def _(
    adapter,
    df_all,
    np,
    pl,
    ui_emission_cols,
    ui_subject,
    ui_tau,
    ui_transition_cols,
):
    df_sub = df_all.filter(pl.col("subject") == ui_subject.value).sort(adapter.sort_col)
    y_real, X, U, names = adapter.load_subject(
        df_sub,
        tau=ui_tau.value,
        emission_cols=ui_emission_cols.value or None,
        transition_cols=ui_transition_cols.value or None,
    )
    session_ids = df_sub[adapter.session_col].to_numpy()
    _ids, _counts = np.unique(session_ids, return_counts=True)
    _keep = set(_ids[_counts >= 2])
    _mask = np.array([_s in _keep for _s in session_ids])

    y_real = y_real[_mask]
    X = X[_mask]
    U = U[_mask]
    session_ids = session_ids[_mask]
    df_sub = df_sub.filter(pl.Series(_mask))

    num_classes = adapter.num_classes
    contrast_labels = adapter.choice_labels[:-1]
    reference_label = adapter.choice_labels[-1]
    feat_names = list(names.get("X_cols", []))
    M = int(X.shape[1])
    T = int(y_real.shape[0])
    return M, T, U, X, contrast_labels, df_sub, feat_names, names, num_classes, reference_label, session_ids, y_real


@app.cell(hide_code=True)
def _(contrast_labels, mo, num_classes, reference_label, ui_task):
    mo.md(
        "\n".join(
            [
                "### Convention note: `SoftmaxGLMHMM` vs Dynamax",
                "",
                f"- Task: **{ui_task.value}**",
                f"- Number of classes: **{num_classes}**",
                f"- Our `SoftmaxGLMHMM` uses **`(K, C-1, M)`** emission weights and **the last class is the reference class**.",
                f"- In this task that means the learned explicit contrasts are **{', '.join(contrast_labels)}**, and **{reference_label}** is the implicit reference with logit `0`.",
                "- So the notebook simulation uses `logits = [eta..., 0]`, matching the model exactly.",
                "- Dynamax `CategoricalRegressionHMM` instead uses **full `(K, C, M)` softmax weights** with one explicit weight vector per class.",
                "- The two parameterizations are equivalent only after embedding our weights into Dynamax's full-softmax form.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(
    contrast_labels,
    feat_names,
    mo,
    np,
    num_classes,
    ui_fit_subject,
    ui_K,
    ui_load_fit,
):
    from wigglystuff import TangleSlider

    K_val = ui_K.value
    _contrast_count = num_classes - 1
    _preset_w = {}
    _preset_a = {}

    if ui_load_fit.value and ui_fit_subject.value:
        d = np.load(ui_fit_subject.value, allow_pickle=True)
        _W_fit = np.asarray(d["emission_weights"])
        _x_cols_fit = list(d["X_cols"]) if "X_cols" in d else feat_names
        if _W_fit.shape[0] == K_val and _W_fit.shape[1] == _contrast_count:
            for _k in range(K_val):
                for _ci in range(_contrast_count):
                    for _fi, _fname in enumerate(_x_cols_fit):
                        if _fname in feat_names:
                            _preset_w[(_k, _ci, _fname)] = float(
                                _W_fit[_k, _ci, _fi]
                            )
        if "transition_matrix" in d:
            _A_fit = np.asarray(d["transition_matrix"])
            if _A_fit.shape == (K_val, K_val):
                for _k in range(K_val):
                    for _j in range(K_val):
                        _preset_a[(_k, _j)] = float(_A_fit[_k, _j])

    w_sliders = {}
    for _k in range(K_val):
        for _ci, _contrast_label in enumerate(contrast_labels):
            for _fi, _fname in enumerate(feat_names):
                _key = f"W[{_k},{_ci},{_fname}]"
                _default = _preset_w.get(
                    (_k, _ci, _fname),
                    float(
                        np.random.default_rng(1000 + 100 * _k + 10 * _ci + _fi).uniform(-0.5, 0.5)
                    ),
                )
                w_sliders[_key] = mo.ui.anywidget(
                    TangleSlider(
                        amount=round(_default, 2),
                        min_value=-3.0,
                        max_value=3.0,
                        step=0.1,
                        digits=2,
                    )
                )

    a_sliders = {}
    for _k in range(K_val):
        for _j in range(K_val):
            _key = f"A[{_k}->{_j}]"
            _default = _preset_a.get(
                (_k, _j),
                0.9 if _k == _j else 0.1 / max(K_val - 1, 1),
            )
            a_sliders[_key] = mo.ui.anywidget(
                TangleSlider(
                    amount=round(float(_default), 2),
                    min_value=0.01,
                    max_value=0.99,
                    step=0.01,
                    digits=2,
                )
            )

    _col_labels = [f"State {_k} · {_contrast}" for _k in range(K_val) for _contrast in contrast_labels]
    _header_w = mo.hstack(
        [mo.md("**Feature**")] + [mo.md(f"**{_lbl}**") for _lbl in _col_labels],
        justify="start",
    )
    _rows_w = []
    for _fname in feat_names:
        _rows_w.append(
            mo.hstack(
                [mo.md(f"`{_fname}`")]
                + [
                    w_sliders[f"W[{_k},{_ci},{_fname}]"]
                    for _k in range(K_val)
                    for _ci in range(_contrast_count)
                ],
                justify="start",
            )
        )

    _header_a = mo.hstack(
        [mo.md("**from \\ to**")] + [mo.md(f"**→ {_j}**") for _j in range(K_val)],
        justify="start",
    )
    _rows_a = []
    for _k in range(K_val):
        _rows_a.append(
            mo.hstack(
                [mo.md(f"**from {_k}**")] + [a_sliders[f"A[{_k}->{_j}]"] for _j in range(K_val)],
                justify="start",
            )
        )

    mo.vstack(
        [
            mo.md(f"### 2. Ground-truth parameters  (K={K_val}, contrasts={_contrast_count})"),
            mo.md("**Emission weights**"),
            mo.vstack([_header_w] + _rows_w),
            mo.md("**Transition matrix**  (rows are normalized automatically)"),
            mo.vstack([_header_a] + _rows_a),
        ],
        align="start",
    )
    return K_val, a_sliders, w_sliders


@app.cell
def _(K_val, M, a_sliders, contrast_labels, feat_names, np, num_classes, w_sliders):
    _contrast_count = num_classes - 1
    W_true = np.zeros((K_val, _contrast_count, M), dtype=np.float32)
    for _k in range(K_val):
        for _ci in range(_contrast_count):
            for _fi, _fname in enumerate(feat_names):
                W_true[_k, _ci, _fi] = w_sliders[f"W[{_k},{_ci},{_fname}]"].value["amount"]

    A_raw = np.zeros((K_val, K_val), dtype=np.float32)
    for _k in range(K_val):
        for _j in range(K_val):
            A_raw[_k, _j] = a_sliders[f"A[{_k}->{_j}]"].value["amount"]
    A_true = A_raw / A_raw.sum(axis=1, keepdims=True)
    return A_true, W_true


@app.cell
def _(A_true, K_val, T, W_true, X, is_script_mode, jnp, jr, mo, np, session_ids, ui_run_fit, ui_seed):
    mo.stop(
        not (is_script_mode or ui_run_fit.value),
        mo.md("Adjust the parameters above and click **Run recovery**."),
    )

    rng = jr.PRNGKey(int(ui_seed.value))
    z_sim = np.zeros(T, dtype=np.int32)
    y_sim = np.zeros(T, dtype=np.int32)

    for _sess in list(dict.fromkeys(session_ids.tolist())):
        _idx = np.where(session_ids == _sess)[0]
        rng, key0 = jr.split(rng)
        z_sim[_idx[0]] = int(jr.categorical(key0, jnp.zeros(K_val)))

        for _local_t, _global_t in enumerate(_idx):
            if _local_t > 0:
                rng, key_z = jr.split(rng)
                _z_prev = z_sim[_idx[_local_t - 1]]
                z_sim[_global_t] = int(jr.categorical(key_z, jnp.log(jnp.asarray(A_true[_z_prev]))))
            _eta = W_true[z_sim[_global_t]] @ X[_global_t]
            logits = jnp.concatenate([jnp.asarray(_eta), jnp.zeros(1, dtype=jnp.float32)])
            rng, key_y = jr.split(rng)
            y_sim[_global_t] = int(jr.categorical(key_y, logits))

    y_sim = jnp.asarray(y_sim)
    return y_sim, z_sim


@app.cell
def _(
    K_val,
    M,
    SoftmaxGLMHMM,
    X,
    find_permutation,
    is_script_mode,
    jax,
    jnp,
    jr,
    np,
    num_classes,
    session_ids,
    ui_num_iters,
    ui_restarts,
    ui_seed,
    y_sim,
    z_sim,
):
    model = SoftmaxGLMHMM(
        num_states=K_val,
        num_classes=num_classes,
        emission_input_dim=M,
        transition_input_dim=0,
        m_step_num_iters=100,
        transition_matrix_stickiness=5.0,
    )

    best_lp = -np.inf
    best_params = None
    all_lps = []
    n_restarts = 1 if is_script_mode else int(ui_restarts.value)
    num_iters = 20 if is_script_mode else int(ui_num_iters.value)

    for _restart in range(n_restarts):
        _key = jr.PRNGKey(int(ui_seed.value) + _restart)
        params0, props = model.initialize(key=_key)
        fitted_params, _lps = model.fit_em_multisession(
            params=params0,
            props=props,
            emissions=y_sim,
            inputs=X,
            session_ids=session_ids,
            num_iters=num_iters,
            verbose=False,
        )
        all_lps.append(np.asarray(_lps))
        if float(_lps[-1]) > best_lp:
            best_lp = float(_lps[-1])
            best_params = fitted_params

    _W_fit = np.asarray(best_params.emissions.weights)
    _A_fit = np.asarray(best_params.transitions.transition_matrix)

    jit_viterbi = jax.jit(model.most_likely_states)
    vit_fit_raw = np.zeros(len(np.asarray(y_sim)), dtype=np.int32)
    _y_np = np.asarray(y_sim)
    for _sess in list(dict.fromkeys(session_ids.tolist())):
        _idx = np.where(session_ids == _sess)[0]
        vit_fit_raw[_idx] = np.asarray(
            jit_viterbi(best_params, jnp.asarray(_y_np[_idx]), jnp.asarray(X[_idx]))
        )

    perm = list(np.asarray(find_permutation(jnp.asarray(vit_fit_raw), jnp.asarray(z_sim))))
    W_fit_aligned = _W_fit[perm]
    A_fit_aligned = _A_fit[perm][:, perm]
    return A_fit_aligned, W_fit_aligned, all_lps, best_params, model, perm


@app.cell
def _(
    A_fit_aligned,
    A_true,
    K_val,
    W_fit_aligned,
    W_true,
    X,
    adapter,
    best_params,
    build_views,
    jax,
    jnp,
    jr,
    model,
    names,
    np,
    perm,
    session_ids,
    y_sim,
    z_sim,
):
    _true_params, _ = model.initialize(
        key=jr.PRNGKey(0),
        transition_matrix=jnp.asarray(A_true),
        emission_weights=jnp.asarray(W_true),
    )

    _jit_viterbi_batch = jax.jit(jax.vmap(model.most_likely_states, in_axes=(None, 0, 0)))

    def _infer_ms(params):
        _sessions = model._split_by_session(y_sim, X, session_ids)
        _e_pad, _i_pad, _lengths = model._pad_sessions(_sessions)
        _post = model._batched_smoother_jit(params, _e_pad, _i_pad)
        _vit_raw = np.asarray(_jit_viterbi_batch(params, _e_pad, _i_pad))
        _sm = np.asarray(_post.smoothed_probs)
        _fi = np.asarray(_post.filtered_probs)
        _sm_out = np.concatenate([_sm[_i, :_T_s] for _i, _T_s in enumerate(_lengths)], axis=0)
        _fi_out = np.concatenate([_fi[_i, :_T_s] for _i, _T_s in enumerate(_lengths)], axis=0)
        _vit_out = np.concatenate([_vit_raw[_i, :_T_s] for _i, _T_s in enumerate(_lengths)], axis=0)
        return _sm_out, _fi_out, _vit_out

    sm_true, fi_true, vit_true = _infer_ms(_true_params)
    sm_fit, fi_fit, vit_fit = _infer_ms(best_params)
    sm_fit_al = sm_fit[:, perm]
    fi_fit_al = fi_fit[:, perm]
    vit_fit_al = np.array([perm[int(s)] for s in vit_fit])

    recovery_arrays = {
        "true": {
            "smoothed_probs": sm_true,
            "emission_weights": W_true,
            "X": np.asarray(X),
            "y": np.asarray(y_sim),
            "X_cols": np.array(names.get("X_cols", []), dtype=object),
        },
        "fit": {
            "smoothed_probs": sm_fit_al,
            "emission_weights": W_fit_aligned,
            "X": np.asarray(X),
            "y": np.asarray(y_sim),
            "X_cols": np.array(names.get("X_cols", []), dtype=object),
        },
    }
    recovery_views = build_views(recovery_arrays, adapter, K_val, ["true", "fit"])
    return fi_fit_al, fi_true, recovery_views, sm_fit_al, sm_true, vit_fit_al, vit_true


@app.cell(hide_code=True)
def _(mo, recovery_views):
    _true_view = recovery_views["true"]
    _fit_view = recovery_views["fit"]
    _rows = ["| Model | State order | Labels |", "|---|---|---|"]
    for _name, _view in [("True", _true_view), ("Fitted", _fit_view)]:
        _labels = [_view.state_name_by_idx.get(_k, f"State {_k}") for _k in _view.state_idx_order]
        _rows.append(
            f"| {_name} | {_view.state_idx_order} | {_labels} |"
        )
    mo.md("### Recovery views\n" + "\n".join(_rows))
    return


@app.cell
def _(
    adapter,
    fi_fit_al,
    fi_true,
    gridspec,
    mo,
    np,
    plt,
    recovery_views,
    session_ids,
    sm_fit_al,
    sm_true,
    sns,
    vit_fit_al,
    vit_true,
    y_sim,
    z_sim,
):
    _true_view = recovery_views["true"]
    _order = _true_view.state_idx_order
    _labels = [_true_view.state_name_by_idx.get(_k, f"State {_k}") for _k in _order]
    _colors = [
        sns.color_palette("tab10", len(_order))[_true_view.state_rank_by_idx.get(int(_k), int(_k))]
        for _k in _order
    ]

    _uniq_s, _counts = np.unique(session_ids, return_counts=True)
    _selected_session = _uniq_s[np.argmax(_counts)]
    _mask = session_ids == _selected_session
    _t = np.arange(_mask.sum())
    _z = z_sim[_mask]
    _y = np.asarray(y_sim)[_mask]

    _fig, _axes = plt.subplots(
        5,
        2,
        figsize=(16, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 2, 2, 1]},
    )
    _panel_defs = [
        (
            "True params",
            sm_true[_mask][:, _order],
            fi_true[_mask][:, _order],
            np.array([_order.index(int(_s)) for _s in vit_true[_mask]]),
        ),
        (
            "Fitted params",
            sm_fit_al[_mask][:, _order],
            fi_fit_al[_mask][:, _order],
            np.array([_order.index(int(_s)) for _s in vit_fit_al[_mask]]),
        ),
    ]
    _z_ord = np.array([_order.index(int(_s)) for _s in _z])

    for _col, (_title, _sm, _fi, _vit) in enumerate(_panel_defs):
        _ax = _axes[0, _col]
        for _tt in _t:
            _ax.axvspan(_tt - 0.5, _tt + 0.5, color=_colors[_z_ord[_tt]], alpha=0.25)
        _ax.scatter(_t, _y, c=[_colors[_z_ord[_tt]] for _tt in _t], s=12, zorder=3)
        _ax.set_yticks(list(range(adapter.num_classes)))
        _ax.set_yticklabels(adapter.choice_labels)
        _ax.set_title(f"Observed choices  [{_title}]")
        sns.despine(ax=_ax)

        _ax = _axes[1, _col]
        for _pos, _label in enumerate(_labels):
            _ax.plot(_t, _fi[:, _pos], color=_colors[_pos], lw=1.2, label=_label)
        _ax.set_ylim(-0.05, 1.05)
        _ax.set_ylabel("p(z | y1:t)")
        _ax.set_title(f"Filtering  [{_title}]")
        if _col == 0:
            _ax.legend(fontsize=7, frameon=False)
        sns.despine(ax=_ax)

        _ax = _axes[2, _col]
        for _pos, _label in enumerate(_labels):
            _ax.plot(_t, _sm[:, _pos], color=_colors[_pos], lw=1.2, label=_label)
        _ax.set_ylim(-0.05, 1.05)
        _ax.set_ylabel("p(z | y1:T)")
        _ax.set_title(f"Smoothing  [{_title}]")
        sns.despine(ax=_ax)

        _ax = _axes[3, _col]
        _ax.step(_t, _z_ord, where="mid", lw=2, color="k", label="True z", alpha=0.7)
        _ax.step(_t, _vit, where="mid", lw=1.5, color="crimson", label="Viterbi", ls="--")
        _ax.set_yticks(list(range(len(_order))))
        _ax.set_yticklabels(_labels)
        _ax.set_ylabel("State")
        _ax.set_title(f"Viterbi MAP  (acc={float(np.mean(_vit == _z_ord)):.1%})  [{_title}]")
        if _col == 0:
            _ax.legend(fontsize=7, frameon=False)
        sns.despine(ax=_ax)

        _ax = _axes[4, _col]
        _ax.fill_between(_t, 0, (_vit != _z_ord).astype(float), color="crimson", alpha=0.5, step="mid")
        _ax.set_ylim(0, 1.5)
        _ax.set_yticks([0, 1])
        _ax.set_yticklabels(["ok", "err"])
        _ax.set_xlabel("Trial")
        _ax.set_title(f"Viterbi errors  [{_title}]")
        sns.despine(ax=_ax)

    _fig.suptitle(f"State inference on simulated data  (session {_selected_session})", fontsize=13)
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("### State inference: filtering, smoothing, and Viterbi"),
            _fig,
            mo.md(
                f"| | True params | Fitted params |\n"
                f"|---|---|---|\n"
                f"| Viterbi accuracy (all trials) | {float(np.mean(vit_true == z_sim)):.2%} | {float(np.mean(vit_fit_al == z_sim)):.2%} |"
            ),
        ],
        align="center",
    )
    return


@app.cell
def _(
    A_fit_aligned,
    A_true,
    K_val,
    W_fit_aligned,
    W_true,
    all_lps,
    contrast_labels,
    feat_names,
    gridspec,
    mo,
    pd,
    plt,
    sns,
):
    _palette = sns.color_palette("tab10", K_val)
    _fig = plt.figure(figsize=(16, 10))
    _gs = gridspec.GridSpec(3, max(2 * K_val, 4), figure=_fig, hspace=0.55, wspace=0.45)

    _ax_lc = _fig.add_subplot(_gs[0, :K_val])
    for _restart, _lps in enumerate(all_lps):
        _ax_lc.plot(_lps, alpha=0.7, label=f"restart {_restart}")
    _ax_lc.set_xlabel("EM iteration")
    _ax_lc.set_ylabel("Objective")
    _ax_lc.set_title("A. EM learning curves")
    if len(all_lps) > 1:
        _ax_lc.legend(fontsize=7, frameon=False)
    sns.despine(ax=_ax_lc)

    _ax_w = _fig.add_subplot(_gs[0, K_val:])
    _markers = ["o", "^", "s", "D"]
    for _k in range(K_val):
        for _ci, _contrast_label in enumerate(contrast_labels):
            _ax_w.scatter(
                W_true[_k, _ci],
                W_fit_aligned[_k, _ci],
                color=_palette[_k],
                marker=_markers[_ci % len(_markers)],
                alpha=0.7,
                s=55,
                label=f"state {_k} {_contrast_label}" if _k == 0 else None,
            )
    _lim = max(abs(W_true).max(), abs(W_fit_aligned).max()) * 1.1 + 0.2
    _ax_w.set_xlim(-_lim, _lim)
    _ax_w.set_ylim(-_lim, _lim)
    _ax_w.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    _ax_w.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    _ax_w.plot([-_lim, _lim], [-_lim, _lim], "k--", lw=0.8, alpha=0.4)
    _ax_w.set_xlabel("True weight")
    _ax_w.set_ylabel("Fitted weight")
    _ax_w.set_title("B. Emission weights: true vs fitted")
    sns.despine(ax=_ax_w)

    for _ki, (_mat, _title) in enumerate([(A_true, "C. True A"), (A_fit_aligned, "D. Fitted A")]):
        _ax_t = _fig.add_subplot(_gs[1, _ki * K_val : (_ki + 1) * K_val])
        sns.heatmap(
            _mat,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            ax=_ax_t,
            linewidths=0.5,
            linecolor="white",
            cbar=False,
        )
        _ax_t.set_xlabel("To state")
        _ax_t.set_ylabel("From state")
        _ax_t.set_title(_title)

    _hue_order = []
    _hue_palette = {}
    for _ci, _contrast_label in enumerate(contrast_labels):
        _hue_order.extend([f"True {_contrast_label}", f"Fit {_contrast_label}"])
    for _idx, _cond in enumerate(_hue_order):
        _hue_palette[_cond] = sns.color_palette("Paired", len(_hue_order))[_idx]

    for _k in range(K_val):
        _ax_bar = _fig.add_subplot(_gs[2, _k * 2 : _k * 2 + 2])
        _rows = []
        for _ci, _contrast_label in enumerate(contrast_labels):
            for _fi, _fname in enumerate(feat_names):
                _rows.append(
                    {"Feature": _fname, "Condition": f"True {_contrast_label}", "Weight": float(W_true[_k, _ci, _fi])}
                )
                _rows.append(
                    {"Feature": _fname, "Condition": f"Fit {_contrast_label}", "Weight": float(W_fit_aligned[_k, _ci, _fi])}
                )
        sns.barplot(
            data=pd.DataFrame(_rows),
            x="Feature",
            y="Weight",
            hue="Condition",
            hue_order=_hue_order,
            palette=_hue_palette,
            ax=_ax_bar,
        )
        _ax_bar.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        _ax_bar.tick_params(axis="x", rotation=40, labelsize=7)
        _ax_bar.set_xlabel("")
        _ax_bar.set_ylabel("Weight")
        _ax_bar.set_title(f"E. State {_k} contrasts")
        _ax_bar.legend(fontsize=6, frameon=False, ncol=2)
        sns.despine(ax=_ax_bar)

    _fig.suptitle("GLM-HMM model recovery", fontsize=14, y=1.01)
    plt.tight_layout()
    mo.vstack([mo.md("### Recovery results"), _fig], align="center")
    return


if __name__ == "__main__":
    app.run()
