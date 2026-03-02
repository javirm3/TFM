import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    import numpy as np
    import jax.numpy as jnp
    import jax.random as jr
    import polars as pl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    import paths
    from glmhmmt.model import SoftmaxGLMHMM
    from glmhmmt.features import build_sequence_from_df

    sns.set_style("white")
    return (
        SoftmaxGLMHMM,
        build_sequence_from_df,
        gridspec,
        jnp,
        jr,
        mo,
        np,
        paths,
        pl,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GLM-HMM Model Recovery

    **Goal**: verify that EM can recover known ground-truth parameters.

    1. Choose a subject & K → load its real trial covariates X
    2. Define true emission weights **W** and transition matrix **A**
    3. Simulate synthetic choices on top of the real trial structure
    4. Fit with EM
    5. Compare fitted ↔ true parameters
    """)
    return


@app.cell(hide_code=True)
def _(mo, paths, pl):
    from glmhmmt.features import _ALL_EMISSION_COLS, _ALL_TRANSITION_COLS

    df_all = pl.read_parquet(paths.DATA_PATH / "df_filtered.parquet")
    _subjects = df_all["subject"].unique().sort().to_list()

    ui_subject = mo.ui.dropdown(
        options=_subjects,
        value=_subjects[0],
        label="Subject",
    )
    ui_K = mo.ui.slider(start=2, stop=4, value=2, label="K (num states)")
    ui_tau = mo.ui.slider(
        start=1, stop=200, value=30, step=1, label="τ action half-life"
    )
    ui_seed = mo.ui.number(
        start=0, stop=9999, value=42, step=1, label="Random seed"
    )
    ui_num_iters = mo.ui.slider(
        start=10, stop=300, value=50, step=10, label="EM iterations"
    )
    ui_restarts = mo.ui.slider(
        start=1, stop=10, value=5, step=1, label="EM restarts"
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

    ui_run_fit = mo.ui.run_button(label="▶ Run Fit")

    mo.vstack(
        [
            mo.md("### 1 - Data & model configuration"),
            mo.hstack([ui_subject, ui_K, ui_tau, ui_seed]),
            mo.hstack([ui_num_iters, ui_restarts]),
            mo.hstack([ui_emission_cols, ui_transition_cols]),
            mo.hstack([ui_run_fit]),
        ],
        align="start",
    )
    return (
        df_all,
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
def _(
    build_sequence_from_df,
    df_all,
    np,
    pl,
    ui_emission_cols,
    ui_subject,
    ui_tau,
    ui_transition_cols,
):
    _df_sub = df_all.filter(pl.col("subject") == ui_subject.value).sort("trial_idx")
    y_real, X, _U, names, _AU = build_sequence_from_df(
        _df_sub,
        tau=ui_tau.value,
        emission_cols=ui_emission_cols.value or None,
        transition_cols=ui_transition_cols.value or None,
    )
    session_ids = _df_sub["session"].to_numpy()

    # drop sessions that are too short
    _ids, _cnts = np.unique(session_ids, return_counts=True)
    _keep = set(_ids[_cnts >= 2])
    _mask = np.array([s in _keep for s in session_ids])
    y_real, X, session_ids = y_real[_mask], X[_mask], session_ids[_mask]
    df_sub = _df_sub.filter(pl.Series(_mask))

    M = X.shape[1]  # number of emission features
    T = int(y_real.shape[0])
    return M, T, X, df_sub, session_ids


@app.cell(hide_code=True)
def _(mo, paths, ui_K):
    from glmhmmt.features import _ALL_EMISSION_COLS

    _fits_dir = paths.RESULTS / "fits" / "glmhmm_2"
    _K = ui_K.value
    _fit_files = sorted(_fits_dir.glob(f"*_K{_K}_glmhmm_arrays.npz"))
    _fit_options = {f.stem.split(f"_K{_K}")[0]: f for f in _fit_files}

    ui_fit_subject = mo.ui.dropdown(
        options={k: str(v) for k, v in _fit_options.items()},
        value=next(iter(_fit_options)) if _fit_options else None,
        label="Load preset from fit — subject",
    )
    ui_load_fit = mo.ui.switch(label="Use fitted params as preset", value=False)
    return ui_fit_subject, ui_load_fit


@app.cell(hide_code=True)
def _(M, mo, np, ui_K, ui_emission_cols, ui_fit_subject, ui_load_fit):
    from wigglystuff import TangleSlider

    K = ui_K.value
    feat_names = ui_emission_cols.value

    # ── preset defaults (from parameter recovery experiment) ─────────────────
    _W_preset = {
        # (state, class, feature) -> default weight
        (0, "L", "biasL"): 0.3,
        (0, "L", "biasR"): -0.2,
        (0, "L", "onsetL"): -0.4,
        (0, "L", "onsetC"): -0.3,
        (0, "L", "onsetR"): 0.0,
        (0, "L", "delay"): -0.5,
        (0, "L", "DR"): 0.1,
        (0, "L", "DL"): -0.2,
        (0, "L", "SL"): 0.5,
        (0, "L", "SC"): -0.3,
        (0, "L", "SR"): 0.0,
        (0, "L", "SLxdelay"): -0.5,
        (0, "L", "SCxdelay"): -0.4,
        (0, "L", "SRxdelay"): 0.0,
        (0, "L", "A_L"): 0.2,
        (0, "L", "A_C"): 0.2, 
        (0, "L", "A_R"): -0.3,
        (0, "R", "biasL"): 0.1,
        (0, "R", "biasR"): 0.5,
        (0, "R", "onsetL"): 0.0,
        (0, "R", "onsetC"): -0.4,
        (0, "R", "onsetR"): 0.3,
        (0, "R", "delay"): -0.4,
        (0, "R", "DR"): -0.3,
        (0, "R", "DL"): 0.0,
        (0, "R", "SL"): 0.0,
        (0, "R", "SC"): -0.4,
        (0, "R", "SR"): 0.4,
        (0, "R", "SLxdelay"): -0.2,
        (0, "R", "SCxdelay"): 0.3,
        (0, "R", "SRxdelay"): -0.4,
        (0, "R", "A_L"): 0.0,
        (0, "R", "A_C"): 0.2,
        (0, "R", "A_R"): 0.6,
        (1, "L", "biasL"): 0.6,
        (1, "L", "biasR"): -0.1,
        (1, "L", "onsetL"): 0.1,
        (1, "L", "onsetC"): 0.1,
        (1, "L", "onsetR"): 0.2,
        (1, "L", "delay"): -0.1,
        (1, "L", "DR"): -0.1,
        (1, "L", "DL"): 0.1,
        (1, "L", "SL"): 0.0,
        (1, "L", "SC"): 0.0,
        (1, "L", "SR"): 0.0,
        (1, "L", "SLxdelay"): -0.1,
        (1, "L", "SCxdelay"): -0.1,
        (1, "L", "SRxdelay"): 0.1,
        (1, "L", "A_L"): 1.2,
        (1, "L", "A_C"): 1.2,
        (1, "L", "A_R"): -0.9,
        (1, "R", "biasL"): 0.1,
        (1, "R", "biasR"): 0.9,
        (1, "R", "onsetL"): 0.2,
        (1, "R", "onsetC"): 0.1,
        (1, "R", "onsetR"): 0.1,
        (1, "R", "delay"): -0.2,
        (1, "R", "DR"): 0.2,
        (1, "R", "DL"): 0.2,
        (1, "R", "SL"): -0.1,
        (1, "R", "SC"): -0.1,
        (1, "R", "SR"): 0.2,
        (1, "R", "SLxdelay"): 0.1,
        (1, "R", "SCxdelay"): 0.0,
        (1, "R", "SRxdelay"): 0.1,
        (1, "R", "A_L"): -1.0,
        (1, "R", "A_C"): 1.2,
        (1, "R", "A_R"): 1.1,
    }

    w_preset_ext = {}
    a_preset_ext = {}
    if ui_load_fit.value and ui_fit_subject.value:
        _d = np.load(ui_fit_subject.value)
        _W = _d["emission_weights"]          # (K, 2, 16)
        _A = _d["transition_matrix"]         # (K, K)
        for _k in range(K):
            for _ci, _cl in enumerate(["L", "R"]):
                for _fi, _fn in enumerate(_ALL_EMISSION_COLS):
                    w_preset_ext[(_k, _cl, _fn)] = round(float(_W[_k, _ci, _fi]), 2)
            for _j in range(K):
                a_preset_ext[(_k, _j)] = round(float(_A[_k, _j]), 3)


    _W_preset = {k: v for k, v in _W_preset.items() if k[2] in feat_names}
    # override with values loaded from an existing fit (if any)
    _W_preset.update(w_preset_ext)
    _A_preset = {(0, 0): 0.75, (0, 1): 0.25, (1, 0): 0.30, (1, 1): 0.70}
    _A_preset.update(a_preset_ext)

    # ── emission weights: W[k, class, feat] ──────────────────────────────────
    w_sliders = {}
    for _k in range(K):
        for _c, _cl in enumerate(["L", "R"]):
            for _fi, _fn in enumerate(feat_names):
                _key = f"W[{_k},{_cl},{_fn}]"
                _default = _W_preset.get(
                    (_k, _cl, _fn),
                    round(
                        float(
                            np.random.default_rng(
                                42 + _k * 100 + _c * 50 + _fi
                            ).uniform(-0.5, 0.5)
                        ),
                        1,
                    ),
                )
                w_sliders[_key] = mo.ui.anywidget(
                    TangleSlider(
                        amount=_default,
                        min_value=-3.0,
                        max_value=3.0,
                        step=0.1,
                        digits=1,
                    )
                )

    # ── transition matrix: A[k, j] ────────────────────────────────────────────
    a_sliders = {}
    for _k in range(K):
        for _j in range(K):
            _key = f"A[{_k}->{_j}]"
            _default_a = _A_preset.get(
                (_k, _j), round(0.9 if _k == _j else 0.1 / max(K - 1, 1), 2)
            )
            a_sliders[_key] = mo.ui.anywidget(
                TangleSlider(
                    amount=_default_a,
                    min_value=0.01,
                    max_value=0.99,
                    step=0.01,
                    digits=2,
                )
            )

    # ── Emission weights grid ─────────────────────────────────────────────────
    _col_labels = [
        f"State {_k} · {_cl}" for _k in range(K) for _cl in ["L", "R"]
    ]
    _header_w = mo.hstack(
        [mo.md("**Feature**")] + [mo.md(f"**{_lbl}**") for _lbl in _col_labels],
        justify="start",
    )
    _rows_w = [
        mo.hstack(
            [mo.md(f"`{_fn}`")]
            + [w_sliders[f"W[{_k},{_cl},{_fn}]"] for _k in range(K) for _cl in ["L", "R"]],
            justify="start",
        )
        for _fn in feat_names
    ]
    _w_grid = mo.vstack([_header_w] + _rows_w)

    # ── Transition matrix grid ────────────────────────────────────────────────
    _header_a = mo.hstack(
        [mo.md("**from \\ to**")] + [mo.md(f"**→ {_j}**") for _j in range(K)],
        justify="start",
    )
    _rows_a = [
        mo.hstack(
            [mo.md(f"**from {_k}**")]
            + [a_sliders[f"A[{_k}->{_j}]"] for _j in range(K)],
            justify="start",
        )
        for _k in range(K)
    ]
    _a_grid = mo.vstack([_header_a] + _rows_a)

    mo.vstack(
        [
            mo.md(f"### 2 - True parameters  (K={K}, M={M} features)"),
            mo.hstack([ui_fit_subject, ui_load_fit]),
            mo.md(f"*{len(w_preset_ext) // (2 * K)} features loaded*") if w_preset_ext else mo.md("*No fit loaded — using manual presets*"),
            mo.md("**Emission weights** — drag a value to change it"),
            _w_grid,
            mo.md(
                "**Transition matrix** A[from → to]  *(rows normalised automatically)*"
            ),
            _a_grid,
        ],
        align="start",
    )
    return K, a_sliders, feat_names, w_sliders


@app.cell
def _(K, M, a_sliders, feat_names, jnp, np, w_sliders):
    # W_true: (K, 2, M) — emission weights array
    W_true = np.zeros((K, 2, M), dtype=np.float32)
    for _k in range(K):
        for _ci, _cl in enumerate(["L", "R"]):
            for _fi, _fn in enumerate(feat_names):
                W_true[_k, _ci, _fi] = w_sliders[f"W[{_k},{_cl},{_fn}]"].value[
                    "amount"
                ]

    # A_true: (K, K) – normalise each row to sum to 1
    _A_raw = np.zeros((K, K), dtype=np.float32)
    for _k in range(K):
        for _j in range(K):
            _A_raw[_k, _j] = a_sliders[f"A[{_k}->{_j}]"].value["amount"]
    A_true = _A_raw / _A_raw.sum(axis=1, keepdims=True)

    W_true_j = jnp.asarray(W_true)
    A_true_j = jnp.asarray(A_true)
    return A_true, W_true


@app.cell
def _(
    A_true,
    K,
    T,
    W_true,
    X,
    jnp,
    jr,
    mo,
    np,
    session_ids,
    ui_run_fit,
    ui_seed,
):
    """
    Simulate latent states z_t and choices y_t using the true GLM-HMM generative process.
    We reuse the real covariate matrix X so the trial structure is preserved.
    """

    mo.stop(
        not ui_run_fit.value,
        mo.md("⏸ Adjust parameters above, then click **▶ Run Fit**."),
    )

    rng = jr.PRNGKey(int(ui_seed.value))

    # --- simulate state sequence per session ---
    _unique_sess = list(dict.fromkeys(session_ids.tolist()))  # preserving order
    z_sim = np.zeros(T, dtype=np.int32)
    y_sim = np.zeros(T, dtype=np.int32)

    _idx = 0
    for _sess in _unique_sess:
        _mask_s = session_ids == _sess
        _inds = np.where(_mask_s)[0]
        _Ts = len(_inds)

        rng, k1, k2 = jr.split(rng, 3)

        # initial state: uniform
        _z0 = int(jr.categorical(k1, jnp.zeros(K)))
        _zs = np.zeros(_Ts, dtype=np.int32)
        _ys = np.zeros(_Ts, dtype=np.int32)
        _zs[0] = _z0

        # emission at t=0
        _x0 = X[_inds[0]]
        _eta0 = W_true[_z0] @ _x0  # (2,) = [eta_L, eta_R]
        _logits0 = jnp.array([_eta0[0], 0.0, _eta0[1]])
        rng, _k = jr.split(rng)
        _ys[0] = int(jr.categorical(_k, _logits0))

        for _t in range(1, _Ts):
            # transition
            rng, _k = jr.split(rng)
            _zs[_t] = int(
                jr.categorical(_k, jnp.log(jnp.array(A_true[_zs[_t - 1]])))
            )
            # emission
            _xt = X[_inds[_t]]
            _eta = W_true[_zs[_t]] @ _xt  # (2,)
            _logits = jnp.array([_eta[0], 0.0, _eta[1]])
            rng, _k = jr.split(rng)
            _ys[_t] = int(jr.categorical(_k, _logits))

        z_sim[_inds] = _zs
        y_sim[_inds] = _ys

    y_sim = jnp.asarray(y_sim)
    return y_sim, z_sim


@app.cell
def _(
    K,
    M,
    SoftmaxGLMHMM,
    X,
    jax,
    jnp,
    jr,
    np,
    session_ids,
    ui_num_iters,
    ui_restarts,
    ui_seed,
    y_sim,
    z_sim,
):
    from dynamax.utils.utils import find_permutation

    model = SoftmaxGLMHMM(
        num_states=K,
        num_classes=3,
        emission_input_dim=M,
        transition_input_dim=0,
        m_step_num_iters=100,
        transition_matrix_stickiness=5.0,
    )

    best_lp_em = -np.inf
    best_params_em = None
    all_lps = []

    for _r in range(int(ui_restarts.value)):
        _key = jr.PRNGKey(int(ui_seed.value) + _r)
        _params0, _props = model.initialize(key=_key)
        _fp, _lps = model.fit_em_multisession(
            params=_params0,
            props=_props,
            emissions=y_sim,
            inputs=X,
            session_ids=session_ids,
            num_iters=int(ui_num_iters.value),
            verbose=False,
        )
        all_lps.append(np.asarray(_lps))
        if float(_lps[-1]) > best_lp_em:
            best_lp_em = float(_lps[-1])
            best_params_em = _fp
    print("Done Fitting")
    W_fit = np.asarray(best_params_em.emissions.weights)  # (K, 2, M)
    A_fit = np.asarray(best_params_em.transitions.transition_matrix)  # (K, K)

    # Run Viterbi with fitted params to get fitted state sequence, then
    # use find_permutation(z_fit, z_true) to align fitted → true labels.
    _uniq_s = list(dict.fromkeys(session_ids.tolist()))
    _vit_fit_raw = np.zeros(len(np.asarray(y_sim)), dtype=np.int32)
    _y_np_fit = np.asarray(y_sim)
    _jit_viterbi_fit = jax.jit(model.most_likely_states)
    print("Starting viterbi")
    for _sess in _uniq_s:
        _msk_s = session_ids == _sess
        _inds_s = np.where(_msk_s)[0]
        _y_s = jnp.asarray(_y_np_fit[_inds_s])
        _x_s = jnp.asarray(X[_inds_s])
        _vit_fit_raw[_inds_s] = np.asarray(
            _jit_viterbi_fit(best_params_em, _y_s, _x_s)
        )

    perm = list(
        np.asarray(
            find_permutation(
                jnp.asarray(_vit_fit_raw),
                jnp.asarray(z_sim),
            )
        )
    )
    W_fit_aligned = W_fit[perm]
    A_fit_aligned = A_fit[perm][:, perm]
    return A_fit_aligned, W_fit_aligned, all_lps, best_params_em, model, perm


@app.cell
def _(A_true, K, M, W_true, X, jnp, jr, mo, np, session_ids, y_sim, z_sim):
    """
    Sanity check: run CategoricalRegressionHMM (base dynamax class) with the
    SAME true parameters to verify the emission convention and that plain
    dynamax inference works on this data.

    SoftmaxGLMHMM uses weights (K, C-1, M) with center as reference class 0:
        logits = [w_L·x,  0,  w_R·x]
    CategoricalRegressionHMM uses weights (K, C, M) — full softmax:
        logits = W[k] @ x + b[k]
    So we embed: W_cat[k,0,:] = W_true[k,0,:], W_cat[k,1,:] = 0, W_cat[k,2,:] = W_true[k,1,:]
    and biases = 0. The two models are then mathematically identical.
    """

    from dynamax.hidden_markov_model import CategoricalRegressionHMM
    import jax

    # ── Build equivalent (K, 3, M) weights ───────────────────────────────────
    _W_cat = np.zeros((K, 3, M), dtype=np.float32)
    _W_cat[:, 0, :] = W_true[:, 0, :]  # L logits
    _W_cat[:, 1, :] = 0.0  # C = reference
    _W_cat[:, 2, :] = W_true[:, 1, :]  # R logits
    _b_cat = np.zeros((K, 3), dtype=np.float32)

    _ref_hmm = CategoricalRegressionHMM(
        num_states=K,
        num_classes=3,
        input_dim=M,
        transition_matrix_stickiness=0.0,
        m_step_num_iters=50,
    )
    _ref_params, _ = _ref_hmm.initialize(
        key=jr.PRNGKey(0),
        transition_matrix=jnp.asarray(A_true),
        emission_weights=jnp.asarray(_W_cat),
        emission_biases=jnp.asarray(_b_cat),
    )

    # ── Per-session Viterbi + smoother ────────────────────────────────────────
    _jit_viterbi = jax.jit(_ref_hmm.most_likely_states)
    _jit_smoother = jax.jit(_ref_hmm.smoother)

    _uniq_sess = list(dict.fromkeys(session_ids.tolist()))
    _y_np = np.asarray(y_sim)
    _ref_vit = np.zeros(len(_y_np), dtype=np.int32)
    _ref_sm_list = []

    for _sess in _uniq_sess:
        _msk_s = session_ids == _sess
        _inds_s = np.where(_msk_s)[0]
        _y_s = jnp.asarray(_y_np[_inds_s])
        _x_s = jnp.asarray(X[_inds_s])
        _ref_vit[_inds_s] = np.asarray(_jit_viterbi(_ref_params, _y_s, _x_s))
        _post_s = _jit_smoother(_ref_params, _y_s, _x_s)
        _ref_sm_list.append(np.asarray(_post_s.smoothed_probs))

    _ref_sm = np.concatenate(_ref_sm_list, axis=0)  # (T, K)

    # ── Best-permutation accuracy via find_permutation ───────────────────────
    from dynamax.utils.utils import find_permutation as _find_perm

    _ref_perm = list(
        np.asarray(
            _find_perm(
                jnp.asarray(_ref_vit),
                jnp.asarray(z_sim),
            )
        )
    )
    _ref_vit_al = np.array([_ref_perm[int(s)] for s in _ref_vit])
    _ref_sm_al = _ref_sm[:, _ref_perm]
    _ref_acc = float(np.mean(_ref_vit_al == z_sim))

    # ── Emission log-likelihood check ─────────────────────────────────────────
    # If convention matches: mean ll >> -log(3) ≈ -1.099
    _eta = np.einsum("kcd,td->ktc", W_true, X)  # (K, T, 2)
    _logits_check = np.stack(
        [
            _eta[:, :, 0],
            np.zeros((K, len(_y_np))),
            _eta[:, :, 1],
        ],
        axis=2,
    )  # (K, T, 3)
    _log_p = _logits_check - np.log(
        np.sum(np.exp(_logits_check), axis=2, keepdims=True)
    )
    _mean_ll = float(
        np.mean([_log_p[z_sim[_t], _t, _y_np[_t]] for _t in range(len(_y_np))])
    )

    mo.vstack(
        [
            mo.md(
                "### Sanity check — `CategoricalRegressionHMM` with true params"
            ),
            mo.md(
                f"| Check | Value |\n"
                f"|---|---|\n"
                f"| Emission log-lik (true params) | {_mean_ll:.4f} |\n"
                f"| Chance log-lik (−log 3) | {-np.log(3):.4f} |\n"
                f"| Gap (> 0 → convention OK) | {_mean_ll + np.log(3):.4f} |\n"
                f"| **CategoricalRegressionHMM Viterbi acc** (best perm) | **{_ref_acc:.2%}** |\n"
            ),
            mo.md(
                "- If **gap > 0** and **CategoricalRegressionHMM accuracy is high**: emission convention is correct; any issue is inside `SoftmaxGLMHMM`  \n"
                "- If **gap ≈ 0**: convention mismatch between simulation and model  \n"
                "- If **both low**: states are simply hard to distinguish from choices alone (need more data or stronger weights)"
            ),
        ]
    )
    return (jax,)


@app.cell
def _(
    A_true,
    K,
    W_true,
    X,
    best_params_em,
    jax,
    jnp,
    jr,
    mo,
    model,
    np,
    paths,
    perm,
    plt,
    session_ids,
    sns,
    y_sim,
    z_sim,
):
    # Use model.initialize to build true params properly — avoids fragile _replace
    _true_params, _ = model.initialize(
        key=jr.PRNGKey(0),
        transition_matrix=jnp.asarray(A_true),
        emission_weights=jnp.asarray(W_true),
    )

    # Reuse the already-compiled batched smoother on the model object.
    # HMMPosterior contains BOTH smoothed_probs AND filtered_probs, so one
    # call is sufficient — no need to call model.filter separately.
    _jax_vmap_viterbi = jax.jit(
        jax.vmap(model.most_likely_states, in_axes=(None, 0, 0))
    )


    def _infer_ms(params, y, X_arr, sess_ids):
        sessions = model._split_by_session(y, X_arr, sess_ids)
        e_pad, i_pad, lengths = model._pad_sessions(sessions)
        post = model._batched_smoother_jit(params, e_pad, i_pad)
        vit_raw = np.asarray(_jax_vmap_viterbi(params, e_pad, i_pad))  # (S, T_max)
        sm = np.asarray(post.smoothed_probs)  # (S, T_max, K)
        fi = np.asarray(post.filtered_probs)  # (S, T_max, K)
        sm_out = np.concatenate(
            [sm[i, :T_s] for i, T_s in enumerate(lengths)], axis=0
        )
        fi_out = np.concatenate(
            [fi[i, :T_s] for i, T_s in enumerate(lengths)], axis=0
        )
        vit_out = np.concatenate(
            [vit_raw[i, :T_s] for i, T_s in enumerate(lengths)], axis=0
        )
        return sm_out, fi_out, vit_out


    _sm_true, _fi_true, _vit_true = _infer_ms(_true_params, y_sim, X, session_ids)
    _sm_fit, _fi_fit, _vit_fit = _infer_ms(best_params_em, y_sim, X, session_ids)

    # Align fitted results to true state labels
    _sm_fit_al = _sm_fit[:, perm]
    _fi_fit_al = _fi_fit[:, perm]
    _vit_fit_al = np.array([perm[int(s)] for s in _vit_fit])

    # expose for downstream cells
    sm_true = _sm_true
    sm_fit_al = _sm_fit_al
    vit_fit_al = _vit_fit_al

    # Pick the longest session for the per-session plots
    _uniq_s, _cnts = np.unique(session_ids, return_counts=True)
    _sel = _uniq_s[np.argmax(_cnts)]
    _msk = session_ids == _sel
    _t = np.arange(_msk.sum())
    _z = z_sim[_msk]
    _y = np.asarray(y_sim)[_msk]

    _palette = sns.color_palette("tab10", K)
    _sc = [_palette[k] for k in range(K)]

    _fig, _axes = plt.subplots(
        5,
        2,
        figsize=(16, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 2, 2, 1]},
    )
    for _col, (_lbl, _sm, _fi, _vit) in enumerate(
        [
            ("True params", _sm_true[_msk], _fi_true[_msk], _vit_true[_msk]),
            (
                "Fitted params",
                _sm_fit_al[_msk],
                _fi_fit_al[_msk],
                _vit_fit_al[_msk],
            ),
        ]
    ):
        # ── Row 0: observed choices coloured by true state ────────────────────
        _ax = _axes[0, _col]
        for _tt in _t:
            _ax.axvspan(_tt - 0.5, _tt + 0.5, color=_sc[_z[_tt]], alpha=0.25)
        _ax.scatter(_t, _y, c=[_sc[_z[_tt]] for _tt in _t], s=12, zorder=3)
        _ax.set_yticks([0, 1, 2])
        _ax.set_yticklabels(["L", "C", "R"])
        _ax.set_title(f"Observed choices  [{_lbl}]")
        sns.despine(ax=_ax)

        # ── Row 1: filtering distribution ─────────────────────────────────────
        _ax = _axes[1, _col]
        for _k in range(K):
            _ax.plot(
                _t, _fi[:, _k], color=_palette[_k], lw=1.2, label=f"state {_k}"
            )
        for _tt in _t:
            _ax.axvspan(_tt - 0.5, _tt + 0.5, color=_sc[_z[_tt]], alpha=0.07)
        _ax.set_ylim(-0.05, 1.05)
        _ax.set_ylabel("p(z|y₁:t)")
        _ax.set_title(f"Filtering  [{_lbl}]")
        if _col == 0:
            _ax.legend(fontsize=7, frameon=False)
        sns.despine(ax=_ax)

        # ── Row 2: smoothing distribution ─────────────────────────────────────
        _ax = _axes[2, _col]
        for _k in range(K):
            _ax.plot(
                _t, _sm[:, _k], color=_palette[_k], lw=1.2, label=f"state {_k}"
            )
        for _tt in _t:
            _ax.axvspan(_tt - 0.5, _tt + 0.5, color=_sc[_z[_tt]], alpha=0.07)
        _ax.set_ylim(-0.05, 1.05)
        _ax.set_ylabel("p(z|y₁:T)")
        _ax.set_title(f"Smoothing  [{_lbl}]")
        sns.despine(ax=_ax)

        # ── Row 3: Viterbi MAP states vs true ─────────────────────────────────
        _ax = _axes[3, _col]
        _ax.step(_t, _z, where="mid", lw=2, color="k", label="True z", alpha=0.7)
        _ax.step(
            _t,
            _vit,
            where="mid",
            lw=1.5,
            color="crimson",
            label="Viterbi",
            ls="--",
        )
        _ax.set_yticks(list(range(K)))
        _ax.set_ylabel("State")
        _acc = float(np.mean(_vit == _z))
        _ax.set_title(f"Viterbi MAP  (acc={_acc:.1%})  [{_lbl}]")
        if _col == 0:
            _ax.legend(fontsize=7, frameon=False)
        sns.despine(ax=_ax)

        # ── Row 4: error bars ─────────────────────────────────────────────────
        _ax = _axes[4, _col]
        _ax.fill_between(
            _t,
            0,
            (_vit != _z).astype(float),
            color="crimson",
            alpha=0.5,
            step="mid",
        )
        _ax.set_ylim(0, 1.5)
        _ax.set_yticks([0, 1])
        _ax.set_yticklabels(["✓", "✗"])
        _ax.set_xlabel("Trial")
        _ax.set_title(f"Viterbi errors  [{_lbl}]")
        sns.despine(ax=_ax)

    plt.suptitle(
        f"State inference on simulated data  (session {_sel})", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(f"{paths.PLOTS}state_inference.png", bbox_inches="tight", dpi=150)

    _vit_true_acc_all = float(np.mean(_vit_true == z_sim))
    _vit_fit_acc_all = float(np.mean(_vit_fit_al == z_sim))

    mo.vstack(
        [
            mo.md("### State inference: filtering - smoothing - Viterbi"),
            _fig,
            mo.md(
                f"| | True params | Fitted params |\n"
                f"|---|---|---|\n"
                f"| Viterbi accuracy (all trials) | {_vit_true_acc_all:.2%} | {_vit_fit_acc_all:.2%} |\n"
            ),
        ],
        align="center",
    )
    return sm_fit_al, sm_true, vit_fit_al


@app.cell
def _(
    A_fit_aligned,
    A_true,
    K,
    W_fit_aligned,
    W_true,
    all_lps,
    feat_names,
    gridspec,
    mo,
    paths,
    plt,
    sns,
    ui_restarts,
):
    _n_restarts = int(ui_restarts.value)
    _palette = sns.color_palette("tab10", K)

    # ── Figure layout ─────────────────────────────────────────────────────────
    _fig = plt.figure(figsize=(16, 10))
    _gs = gridspec.GridSpec(3, 2 * K, figure=_fig, hspace=0.55, wspace=0.45)

    # ── Panel A: EM learning curves ───────────────────────────────────────────
    _ax_lc = _fig.add_subplot(_gs[0, :K])
    for _r, _lps in enumerate(all_lps):
        _ax_lc.plot(
            _lps, alpha=0.7, label=f"restart {_r}" if _n_restarts > 1 else "EM"
        )
    _ax_lc.set_xlabel("EM iteration")
    _ax_lc.set_ylabel("Log probability")
    _ax_lc.set_title("A - EM learning curves")
    if _n_restarts > 1:
        _ax_lc.legend(fontsize=7, frameon=False)
    sns.despine(ax=_ax_lc)

    # ── Panel B: True vs fitted emission weights (scatter) ────────────────────
    _ax_w = _fig.add_subplot(_gs[0, K:])
    _class_lbls = ["L", "R"]
    for _k in range(K):
        for _ci in range(2):
            _wt = W_true[_k, _ci]  # (M,)
            _wf = W_fit_aligned[_k, _ci]  # (M,)
            _ax_w.scatter(
                _wt,
                _wf,
                color=_palette[_k],
                marker=["o", "^"][_ci],
                alpha=0.7,
                s=55,
                label=f"state {_k} {_class_lbls[_ci]}"
                if _k + _ci == 0
                else f"s{_k}{_class_lbls[_ci]}",
            )
    _lim = max(abs(W_true).max(), abs(W_fit_aligned).max()) * 1.1 + 0.2
    _ax_w.set_xlim(-_lim, _lim)
    _ax_w.set_ylim(-_lim, _lim)
    _ax_w.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    _ax_w.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    _ax_w.plot([-_lim, _lim], [-_lim, _lim], "k--", lw=0.8, alpha=0.4, label="y=x")
    _ax_w.set_xlabel("True weight")
    _ax_w.set_ylabel("Fitted weight")
    _ax_w.set_title("B - Emission weights: true vs fitted")
    _ax_w.legend(fontsize=6, frameon=False, ncol=2)
    sns.despine(ax=_ax_w)

    # ── Panel C & D: Transition matrices ──────────────────────────────────────
    _vmax = 1.0
    for _ki, (_mat, _title) in enumerate(
        [(A_true, "C - True A"), (A_fit_aligned, "D - Fitted A")]
    ):
        _ax_t = _fig.add_subplot(_gs[1, _ki * K : (_ki + 1) * K])
        sns.heatmap(
            _mat,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=_vmax,
            ax=_ax_t,
            linewidths=0.5,
            linecolor="white",
            cbar=False,
        )
        _ax_t.set_xlabel("To state")
        _ax_t.set_ylabel("From state")
        _ax_t.set_title(_title)

    # ── Panel E: Emission weight bar comparison per state ─────────────────────
    import pandas as pd
    _hue_pal = dict(zip(
        ["True L", "Fit L", "True R", "Fit R"],
        sns.color_palette("Paired", 4),
    ))
    for _k in range(K):
        _ax_bar = _fig.add_subplot(_gs[2, _k * 2 : _k * 2 + 2])
        _rows = []
        for _ci, _cl in enumerate(["L", "R"]):
            for _fn in feat_names:
                _fi = feat_names.index(_fn)
                _rows += [
                    {"Feature": _fn, "Condition": f"True {_cl}", "Weight": float(W_true[_k, _ci, _fi])},
                    {"Feature": _fn, "Condition": f"Fit {_cl}",  "Weight": float(W_fit_aligned[_k, _ci, _fi])},
                ]
        sns.barplot(
            data=pd.DataFrame(_rows),
            x="Feature", y="Weight",
            hue="Condition",
            hue_order=["True L", "Fit L", "True R", "Fit R"],
            palette=_hue_pal,
            ax=_ax_bar,
        )
        _ax_bar.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        _ax_bar.set_xticklabels(_ax_bar.get_xticklabels(), rotation=40, ha="right", fontsize=7)
        _ax_bar.set_xlabel("")
        _ax_bar.set_ylabel("Weight")
        _ax_bar.set_title(f"E - State {_k} emission weights")
        _ax_bar.legend(fontsize=6, frameon=False, ncol=2)
        sns.despine(ax=_ax_bar)

    _fig.suptitle("GLM-HMM Model Recovery", fontsize=14, y=1.01)
    plt.savefig(f"{paths.PLOTS}model_recovery.png", bbox_inches="tight", dpi=150)

    mo.vstack([mo.md("### Recovery results"), _fig], align="center")
    return


@app.cell
def _(A_fit_aligned, A_true, W_fit_aligned, W_true, mo, np):
    # ── Numerical summary ─────────────────────────────────────────────────────
    _w_rmse = float(np.sqrt(np.mean((W_true - W_fit_aligned) ** 2)))
    _a_rmse = float(np.sqrt(np.mean((A_true - A_fit_aligned) ** 2)))
    _w_corr = float(np.corrcoef(W_true.ravel(), W_fit_aligned.ravel())[0, 1])
    _a_corr = float(np.corrcoef(A_true.ravel(), A_fit_aligned.ravel())[0, 1])

    mo.vstack(
        [
            mo.md("### Numerical summary"),
            mo.md(
                f"| Metric | Value |\n"
                f"|---|---|\n"
                f"| Emission weight RMSE | {_w_rmse:.4f} |\n"
                f"| Emission weight correlation | {_w_corr:.4f} |\n"
                f"| Transition matrix RMSE | {_a_rmse:.4f} |\n"
                f"| Transition matrix correlation | {_a_corr:.4f} |\n"
            )
        ]
    )
    return


@app.cell
def _(
    K,
    W_fit_aligned,
    W_true,
    X,
    df_sub,
    mo,
    np,
    pl,
    sm_fit_al,
    sm_true,
    vit_fit_al,
    y_sim,
    z_sim,
):
    from glmhmmt.plots import (
        prepare_predictions_df,
        plot_categorical_performance_all,
        plot_categorical_performance_by_state,
    )

    _state_labels = {
        k: ("Engaged" if k == 0 else ("Disengaged" if K == 2 else f"Disengaged {k}"))
        for k in range(K)
    }

    def _emission_probs(W, sm, X_arr):
        T_ = X_arr.shape[0]
        probs = np.zeros((T_, 3), dtype=np.float64)
        for _k in range(W.shape[0]):
            _eta = X_arr @ W[_k].T          # (T, 2)
            _logits = np.stack(
                [_eta[:, 0], np.zeros(T_), _eta[:, 1]], axis=1
            )  # (T, 3)
            _log_z = np.log(np.sum(np.exp(_logits), axis=1, keepdims=True))
            probs += sm[:, _k : _k + 1] * np.exp(_logits - _log_z)
        return probs

    _X_np = np.asarray(X)
    _probs_true = _emission_probs(W_true,       sm_true,    _X_np)
    _probs_fit  = _emission_probs(W_fit_aligned, sm_fit_al, _X_np)

    _y_np  = np.asarray(y_sim)
    _stim  = df_sub["stimulus"].to_numpy()
    _perf  = (_y_np == _stim).astype(int)

    def _make_pred_df(probs):
        return df_sub.with_columns([
            pl.Series("response",    _y_np.astype(int)),
            pl.Series("performance", _perf.astype(int)),
            pl.Series("pL",  probs[:, 0].astype(float)),
            pl.Series("pC",  probs[:, 1].astype(float)),
            pl.Series("pR",  probs[:, 2].astype(float)),
        ])

    _df_true_prep = prepare_predictions_df(_make_pred_df(_probs_true))
    _df_fit_prep  = prepare_predictions_df(_make_pred_df(_probs_fit))

    _fig_all_true, _ = plot_categorical_performance_all(_df_true_prep, "True params")
    _fig_all_fit, _  = plot_categorical_performance_all(_df_fit_prep,  "Fitted params")

    _fig_state_true, _ = plot_categorical_performance_by_state( _df_true_prep, smoothed_probs=sm_true, state_labels=_state_labels, model_name="True params", state_assign=np.asarray(z_sim),)
    _fig_state_fit, _ = plot_categorical_performance_by_state(_df_fit_prep, smoothed_probs=sm_fit_al, state_labels=_state_labels, model_name="Fitted params", state_assign=np.asarray(vit_fit_al),)

    mo.vstack(
        [
            mo.md("### Categorical performance — pooled (simulated data)"),
            mo.hstack([_fig_all_true, _fig_all_fit]),
            mo.md("### Per-state categorical performance"),
            mo.md("**True params · true state assignment (z_sim)**"),
            _fig_state_true,
            mo.md("**Fitted params · Viterbi state assignment**"),
            _fig_state_fit,
        ],
        align="center",
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
