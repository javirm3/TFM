import marimo

__generated_with = "0.21.0"
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
    from widgets import ModelManagerWidget
    from coefficient_editor_widget import CoefficientEditorWidget

    sns.set_style("white")
    return (
        CoefficientEditorWidget,
        ModelManagerWidget,
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
    mo.md("""
    # GLM-HMM Model Recovery

    This notebook now uses the task adapters and `SubjectFitView` pipeline.

    Workflow:
    1. Choose a task, subject, and regressors
    2. Reuse that subject's real covariates and session structure
    3. Simulate synthetic choices from user-defined ground-truth GLM-HMM parameters
    4. Fit the same model family back to the synthetic data
    5. Compare true vs fitted states, posteriors, weights, and transitions
    """)
    return


@app.cell(hide_code=True)
def _(ModelManagerWidget, mo):
    mm_widget = ModelManagerWidget(
        model_type="glmhmm",
        task="MCDR",
        K=2,
        tau=50,
    )
    ui_model_manager = mo.ui.anywidget(mm_widget)
    return (ui_model_manager,)


@app.cell
def _(get_adapter, paths, pl, ui_model_manager):
    task_name = ui_model_manager.value["task"]
    adapter = get_adapter(task_name)
    df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df_all = adapter.subject_filter(df_all)
    return adapter, df_all, task_name


@app.cell(hide_code=True)
def _(mo, ui_model_manager):
    class _V:
        def __init__(self, value):
            self.value = value

    _val = ui_model_manager.value
    _subjects = list(_val.get("subjects", []))
    _selected_subject = _subjects[0] if _subjects else None

    ui_subject = _V(_selected_subject)
    ui_K = _V(_val["K"])
    ui_tau = _V(_val["tau"])
    ui_emission_cols = _V(_val.get("emission_cols", []))
    ui_seed = mo.ui.number(start=0, stop=9999, value=42, step=1, label="Random seed")
    ui_num_iters = mo.ui.slider(start=10, stop=300, value=50, step=10, label="EM iterations")
    ui_restarts = mo.ui.slider(start=1, stop=10, value=5, step=1, label="Restarts")
    ui_run_recovery = mo.ui.run_button(
        label="Run recovery fit",
        kind="success",
    )

    mo.vstack(
        [
            mo.md("### 1. Data and model configuration"),
            ui_model_manager,
            mo.md(
                f"Using subject **{_selected_subject}** for the synthetic recovery dataset "
                "(the first selected subject in the model widget)."
                if _selected_subject is not None
                else "Select at least one subject in the model widget."
            ),
            mo.hstack([ui_seed, ui_num_iters, ui_restarts]),
            mo.md(
                "Use the model widget only to choose task / subject / regressors. "
                "Run the recovery from the button below after editing the ground-truth parameters."
            ),
            ui_run_recovery,
        ],
        align="start",
    )
    return (
        ui_K,
        ui_emission_cols,
        ui_num_iters,
        ui_restarts,
        ui_run_recovery,
        ui_seed,
        ui_subject,
        ui_tau,
    )


@app.cell(hide_code=True)
def _(mo, paths, task_name, ui_K, ui_model_manager):
    _fit_root = paths.RESULTS / "fits" / task_name / "glmhmm" / ui_model_manager.value["alias"]
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
def _(adapter, df_all, mo, np, pl, ui_emission_cols, ui_subject, ui_tau):
    mo.stop(ui_subject.value is None, mo.md("Select at least one subject in the model widget."))
    df_sub = df_all.filter(pl.col("subject") == ui_subject.value).sort(adapter.sort_col)
    y_real, X, U, names = adapter.load_subject(
        df_sub,
        tau=ui_tau.value,
        emission_cols=ui_emission_cols.value or None,
        transition_cols=adapter.default_transition_cols(),
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
    return (
        M,
        X,
        contrast_labels,
        feat_names,
        names,
        num_classes,
        reference_label,
        session_ids,
    )


@app.cell(hide_code=True)
def _(contrast_labels, mo, num_classes, reference_label, task_name):
    mo.md(
        "\n".join(
            [
                "### Convention note: `SoftmaxGLMHMM` vs Dynamax",
                "",
                f"- Task: **{task_name}**",
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
    CoefficientEditorWidget,
    contrast_labels,
    feat_names,
    mo,
    np,
    num_classes,
    reference_label,
    ui_K,
    ui_fit_subject,
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

    coef_editors = {}
    for _k in range(K_val):
        _state_weights = np.zeros((_contrast_count, len(feat_names)), dtype=float)
        for _ci in range(_contrast_count):
            for _fi, _fname in enumerate(feat_names):
                _state_weights[_ci, _fi] = _preset_w.get(
                    (_k, _ci, _fname),
                    float(
                        np.random.default_rng(1000 + 100 * _k + 10 * _ci + _fi).uniform(-0.5, 0.5)
                    ),
                )
        _subtitle = f"{reference_label} is the implicit reference class."
        coef_editors[f"state_{_k}"] = mo.ui.anywidget(
            CoefficientEditorWidget(
                title=f"State {_k} emission weights",
                subtitle=_subtitle,
                features=list(feat_names),
                channel_labels=list(contrast_labels),
                weights=_state_weights.tolist(),
                original_weights=_state_weights.tolist(),
                slider_min=-3.0,
                slider_max=3.0,
                slider_step=0.1,
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
            mo.vstack([coef_editors[f"state_{_k}"] for _k in range(K_val)], align="stretch"),
            mo.md("**Transition matrix**  (rows are normalized automatically)"),
            mo.vstack([_header_a] + _rows_a),
        ],
        align="start",
    )
    return K_val, a_sliders, coef_editors


@app.cell
def _(K_val, M, a_sliders, coef_editors, np, num_classes, ui_run_recovery):
    _ = ui_run_recovery.value
    _contrast_count = num_classes - 1
    W_true = np.zeros((K_val, _contrast_count, M), dtype=np.float32)
    for _k in range(K_val):
        W_true[_k] = np.asarray(coef_editors[f"state_{_k}"].value["weights"], dtype=np.float32)

    A_raw = np.zeros((K_val, K_val), dtype=np.float32)
    for _k in range(K_val):
        for _j in range(K_val):
            A_raw[_k, _j] = a_sliders[f"A[{_k}->{_j}]"].value["amount"]
    A_true = A_raw / A_raw.sum(axis=1, keepdims=True)
    return A_true, W_true


@app.cell
def _(mo):
    ui_sweep_enabled = mo.ui.switch(value=False, label="Run parameter sweep")
    ui_sweep_target = mo.ui.dropdown(
        options={
            "All parameters": "all",
            "Emission weights only": "weights",
            "Transition probabilities only": "transitions",
        },
        value="All parameters",
        label="Sweep target",
    )
    ui_sweep_half_range = mo.ui.number(
        start=0.05,
        stop=3.0,
        step=0.05,
        value=0.5,
        label="Half-range",
    )
    ui_sweep_points = mo.ui.slider(
        start=3,
        stop=9,
        step=2,
        value=5,
        label="Points per parameter",
    )
    mo.vstack(
        [
            mo.md("### 3. Optional parameter sweep"),
            mo.hstack(
                [
                    ui_sweep_enabled,
                    ui_sweep_target,
                    ui_sweep_half_range,
                    ui_sweep_points,
                ]
            ),
            mo.md(
                "When enabled, the notebook reruns recovery over a symmetric range around "
                "each current parameter while keeping all other parameters fixed."
            ),
        ],
        align="start",
    )
    return (
        ui_sweep_enabled,
        ui_sweep_half_range,
        ui_sweep_points,
        ui_sweep_target,
    )


@app.cell
def _(SoftmaxGLMHMM, find_permutation, jax, jnp, jr, np):
    def run_recovery_once(
        *,
        A_true,
        W_true,
        X,
        session_ids,
        K_val,
        M,
        num_classes,
        seed,
        num_iters,
        n_restarts,
    ):
        rng = jr.PRNGKey(int(seed))
        T = int(X.shape[0])
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
                    z_sim[_global_t] = int(
                        jr.categorical(key_z, jnp.log(jnp.asarray(A_true[_z_prev])))
                    )
                _eta = W_true[z_sim[_global_t]] @ X[_global_t]
                logits = jnp.concatenate(
                    [jnp.asarray(_eta), jnp.zeros(1, dtype=jnp.float32)]
                )
                rng, key_y = jr.split(rng)
                y_sim[_global_t] = int(jr.categorical(key_y, logits))

        y_sim = jnp.asarray(y_sim)

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
        for _restart in range(n_restarts):
            _key = jr.PRNGKey(int(seed) + _restart)
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

        perm = list(
            np.asarray(find_permutation(jnp.asarray(vit_fit_raw), jnp.asarray(z_sim)))
        )
        W_fit_aligned = _W_fit[perm]
        A_fit_aligned = _A_fit[perm][:, perm]

        _true_params, _ = model.initialize(
            key=jr.PRNGKey(0),
            transition_matrix=jnp.asarray(A_true),
            emission_weights=jnp.asarray(W_true),
        )
        _jit_viterbi_batch = jax.jit(
            jax.vmap(model.most_likely_states, in_axes=(None, 0, 0))
        )

        def _infer_ms(params):
            _sessions = model._split_by_session(y_sim, X, session_ids)
            _e_pad, _i_pad, _lengths = model._pad_sessions(_sessions)
            _post = model._batched_smoother_jit(params, _e_pad, _i_pad)
            _vit_raw = np.asarray(_jit_viterbi_batch(params, _e_pad, _i_pad))
            _sm = np.asarray(_post.smoothed_probs)
            _fi = np.asarray(_post.filtered_probs)
            _sm_out = np.concatenate(
                [_sm[_i, :_T_s] for _i, _T_s in enumerate(_lengths)], axis=0
            )
            _fi_out = np.concatenate(
                [_fi[_i, :_T_s] for _i, _T_s in enumerate(_lengths)], axis=0
            )
            _vit_out = np.concatenate(
                [_vit_raw[_i, :_T_s] for _i, _T_s in enumerate(_lengths)], axis=0
            )
            return _sm_out, _fi_out, _vit_out

        sm_true, fi_true, vit_true = _infer_ms(_true_params)
        sm_fit, fi_fit, vit_fit = _infer_ms(best_params)
        sm_fit_al = sm_fit[:, perm]
        fi_fit_al = fi_fit[:, perm]
        vit_fit_al = np.array([perm[int(s)] for s in vit_fit])

        return {
            "A_fit_aligned": A_fit_aligned,
            "W_fit_aligned": W_fit_aligned,
            "all_lps": all_lps,
            "best_params": best_params,
            "fi_fit_al": fi_fit_al,
            "fi_true": fi_true,
            "model": model,
            "perm": perm,
            "sm_fit_al": sm_fit_al,
            "sm_true": sm_true,
            "vit_fit_al": vit_fit_al,
            "vit_true": vit_true,
            "y_sim": y_sim,
            "z_sim": z_sim,
        }

    return (run_recovery_once,)


@app.cell
def _(
    A_true,
    K_val,
    M,
    W_true,
    X,
    is_script_mode,
    mo,
    num_classes,
    run_recovery_once,
    session_ids,
    ui_num_iters,
    ui_restarts,
    ui_run_recovery,
    ui_seed,
):
    mo.stop(
        not is_script_mode and not ui_run_recovery.value,
        mo.md(f"Adjust the parameters above and click {ui_run_recovery}."),
    )

    _result = run_recovery_once(
        A_true=A_true,
        W_true=W_true,
        X=X,
        session_ids=session_ids,
        K_val=K_val,
        M=M,
        num_classes=num_classes,
        seed=int(ui_seed.value),
        num_iters=20 if is_script_mode else int(ui_num_iters.value),
        n_restarts=1 if is_script_mode else int(ui_restarts.value),
    )
    A_fit_aligned = _result["A_fit_aligned"]
    W_fit_aligned = _result["W_fit_aligned"]
    all_lps = _result["all_lps"]
    best_params = _result["best_params"]
    fi_fit_al = _result["fi_fit_al"]
    fi_true = _result["fi_true"]
    model = _result["model"]
    perm = _result["perm"]
    sm_fit_al = _result["sm_fit_al"]
    sm_true = _result["sm_true"]
    vit_fit_al = _result["vit_fit_al"]
    vit_true = _result["vit_true"]
    y_sim = _result["y_sim"]
    z_sim = _result["z_sim"]
    return (
        A_fit_aligned,
        W_fit_aligned,
        all_lps,
        fi_fit_al,
        fi_true,
        sm_fit_al,
        sm_true,
        vit_fit_al,
        vit_true,
        y_sim,
        z_sim,
    )


@app.cell
def _(
    K_val,
    W_fit_aligned,
    W_true,
    X,
    adapter,
    build_views,
    names,
    np,
    sm_fit_al,
    sm_true,
    y_sim,
):
    recovery_views = build_views(
        {
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
        },
        adapter,
        K_val,
        ["true", "fit"],
    )
    return (recovery_views,)


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
    A_true,
    K_val,
    M,
    W_true,
    X,
    contrast_labels,
    feat_names,
    is_script_mode,
    mo,
    np,
    num_classes,
    pd,
    run_recovery_once,
    session_ids,
    ui_num_iters,
    ui_restarts,
    ui_seed,
    ui_sweep_enabled,
    ui_sweep_half_range,
    ui_sweep_points,
    ui_sweep_target,
):
    mo.stop(not ui_sweep_enabled.value)

    _rows = []
    _half = float(ui_sweep_half_range.value)
    _num_points = int(ui_sweep_points.value)
    _seed0 = int(ui_seed.value) + 100_000
    _num_iters = 20 if is_script_mode else int(ui_num_iters.value)
    _n_restarts = 1 if is_script_mode else int(ui_restarts.value)
    _targets = ui_sweep_target.value
    _run_idx = [0]

    def _record(_parameter, _kind, _value, _delta, _A_var, _W_var):
        _res = run_recovery_once(
            A_true=_A_var,
            W_true=_W_var,
            X=X,
            session_ids=session_ids,
            K_val=K_val,
            M=M,
            num_classes=num_classes,
            seed=_seed0 + _run_idx[0],
            num_iters=_num_iters,
            n_restarts=_n_restarts,
        )
        _run_idx[0] += 1
        _rows.append(
            {
                "parameter": _parameter,
                "kind": _kind,
                "value": float(_value),
                "delta": float(_delta),
                "viterbi_accuracy": float(
                    np.mean(_res["vit_fit_al"] == np.asarray(_res["z_sim"]))
                ),
                "weight_rmse": float(
                    np.sqrt(np.mean((_res["W_fit_aligned"] - _W_var) ** 2))
                ),
                "transition_rmse": float(
                    np.sqrt(np.mean((_res["A_fit_aligned"] - _A_var) ** 2))
                ),
                "final_lp": float(_res["all_lps"][-1][-1]),
            }
        )

    if _targets in ("all", "weights"):
        for _k in range(K_val):
            for _ci, _contrast in enumerate(contrast_labels):
                for _fi, _fname in enumerate(feat_names):
                    _base = float(W_true[_k, _ci, _fi])
                    for _value in np.linspace(_base - _half, _base + _half, _num_points):
                        _W_var = np.array(W_true, copy=True)
                        _W_var[_k, _ci, _fi] = float(_value)
                        _record(
                            f"W[{_k},{_contrast},{_fname}]",
                            "weight",
                            _value,
                            _value - _base,
                            np.array(A_true, copy=True),
                            _W_var,
                        )

    if _targets in ("all", "transitions"):
        for _k in range(K_val):
            for _j in range(K_val):
                _base = float(A_true[_k, _j])
                for _value in np.linspace(_base - _half, _base + _half, _num_points):
                    _A_var = np.array(A_true, copy=True)
                    _A_var[_k, _j] = float(np.clip(_value, 0.01, 0.99))
                    _A_var[_k] = _A_var[_k] / _A_var[_k].sum()
                    _record(
                        f"A[{_k}->{_j}]",
                        "transition",
                        _A_var[_k, _j],
                        _A_var[_k, _j] - _base,
                        _A_var,
                        np.array(W_true, copy=True),
                    )

    if not _rows:
        sweep_df = None
        sweep_summary = None

    sweep_df = pd.DataFrame(_rows)
    sweep_summary = (
        sweep_df.groupby(["parameter", "kind"], as_index=False)
        .agg(
            min_viterbi_accuracy=("viterbi_accuracy", "min"),
            mean_viterbi_accuracy=("viterbi_accuracy", "mean"),
            mean_weight_rmse=("weight_rmse", "mean"),
            mean_transition_rmse=("transition_rmse", "mean"),
        )
        .sort_values(["min_viterbi_accuracy", "mean_weight_rmse"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return sweep_df, sweep_summary


@app.cell(hide_code=True)
def _(mo, plt, sns, sweep_df, sweep_summary, ui_sweep_enabled):
    mo.stop(not ui_sweep_enabled.value, mo.md("Enable the parameter sweep to run sensitivity recovery."))
    mo.stop(sweep_df is None or sweep_summary is None, mo.md("No sweep results available."))

    _top_params = sweep_summary.head(12)["parameter"].tolist()
    _plot_df = sweep_df[sweep_df["parameter"].isin(_top_params)].copy()

    _fig, _axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(
        data=sweep_summary.head(20),
        y="parameter",
        x="min_viterbi_accuracy",
        hue="kind",
        dodge=False,
        ax=_axes[0],
    )
    _axes[0].set_title("Most sensitive parameters")
    _axes[0].set_xlabel("Worst-case Viterbi accuracy")
    _axes[0].set_ylabel("")
    _axes[0].set_xlim(0, 1)
    sns.despine(ax=_axes[0])

    sns.lineplot(
        data=_plot_df,
        x="delta",
        y="viterbi_accuracy",
        hue="parameter",
        style="kind",
        marker="o",
        ax=_axes[1],
    )
    _axes[1].set_title("Accuracy across sweep range")
    _axes[1].set_xlabel("Parameter perturbation from current value")
    _axes[1].set_ylabel("Viterbi accuracy")
    _axes[1].set_ylim(0, 1)
    _axes[1].legend(fontsize=7, frameon=False)
    sns.despine(ax=_axes[1])

    plt.tight_layout()
    mo.vstack(
        [
            mo.md("### Parameter sweep recovery"),
            _fig,
            sweep_summary.head(20),
        ],
        align="center",
    )
    return


@app.cell
def _(
    adapter,
    fi_fit_al,
    fi_true,
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
    np,
    pd,
    plt,
    recovery_views,
    sns,
):
    _true_view = recovery_views["true"]
    _fit_view = recovery_views["fit"]
    _state_labels = [
        _true_view.state_name_by_idx.get(_k, f"State {_k}")
        for _k in _true_view.state_idx_order
    ]

    def _idx_by_label(view, label, fallback_pos):
        for _idx in view.state_idx_order:
            if view.state_name_by_idx.get(_idx, f"State {_idx}") == label:
                return int(_idx)
        return int(view.state_idx_order[fallback_pos])

    _true_idx_order = np.array(
        [_idx_by_label(_true_view, _label, _pos) for _pos, _label in enumerate(_state_labels)],
        dtype=int,
    )
    _fit_idx_order = np.array(
        [_idx_by_label(_fit_view, _label, _pos) for _pos, _label in enumerate(_state_labels)],
        dtype=int,
    )

    _W_true_plot = W_true[_true_idx_order]
    _W_fit_plot = W_fit_aligned[_fit_idx_order]
    _A_true_plot = A_true[np.ix_(_true_idx_order, _true_idx_order)]
    _A_fit_plot = A_fit_aligned[np.ix_(_fit_idx_order, _fit_idx_order)]

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
                _W_true_plot[_k, _ci],
                _W_fit_plot[_k, _ci],
                color=_palette[_k],
                marker=_markers[_ci % len(_markers)],
                alpha=0.7,
                s=55,
                label=_state_labels[_k] if _ci == 0 else None,
            )
    _lim = max(abs(_W_true_plot).max(), abs(_W_fit_plot).max()) * 1.1 + 0.2
    _ax_w.set_xlim(-_lim, _lim)
    _ax_w.set_ylim(-_lim, _lim)
    _ax_w.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    _ax_w.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    _ax_w.plot([-_lim, _lim], [-_lim, _lim], "k--", lw=0.8, alpha=0.4)
    _ax_w.set_xlabel("True weight")
    _ax_w.set_ylabel("Fitted weight")
    _ax_w.set_title("B. Emission weights: true vs fitted")
    _ax_w.legend(fontsize=7, frameon=False)
    sns.despine(ax=_ax_w)

    for _ki, (_mat, _title) in enumerate([(_A_true_plot, "C. True A"), (_A_fit_plot, "D. Fitted A")]):
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
            xticklabels=_state_labels,
            yticklabels=_state_labels,
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
                    {"Feature": _fname, "Condition": f"True {_contrast_label}", "Weight": float(_W_true_plot[_k, _ci, _fi])}
                )
                _rows.append(
                    {"Feature": _fname, "Condition": f"Fit {_contrast_label}", "Weight": float(_W_fit_plot[_k, _ci, _fi])}
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
        _ax_bar.set_title(f"E. {_state_labels[_k]} contrasts")
        _ax_bar.legend(fontsize=6, frameon=False, ncol=2)
        sns.despine(ax=_ax_bar)

    _fig.suptitle("GLM-HMM model recovery", fontsize=14, y=1.01)
    plt.tight_layout()
    mo.vstack([mo.md("### Recovery results"), _fig], align="center")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
