import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import sys, os
    import seaborn as sns
    import tomllib

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import polars as pl
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib.pyplot as plt

    import paths
    from glmhmmt.model import SoftmaxGLMHMM
    from glmhmmt.features import build_sequence_from_df

    with paths.CONFIG.open("rb") as f:
        cfg = tomllib.load(f)
    return (
        SoftmaxGLMHMM,
        build_sequence_from_df,
        jnp,
        jr,
        np,
        paths,
        pl,
        plt,
        sns,
    )


@app.cell
def _(mo):
    range_slider = mo.ui.range_slider(
        start=1, stop=10, step=1, value=[2, 2], full_width=True
    )
    range_slider
    return (range_slider,)


@app.cell
def _(mo, paths, range_slider):
    DATA_PATH = paths.DATA_PATH / "df_filtered.parquet"
    K_LIST = range_slider.value

    # EM settings
    NUM_ITERS = 50
    M_STEP_NUM_ITERS = 100
    STICKINESS = 10.0

    N_RESTARTS = 1
    BASE_SEED = 12345

    # Output
    OUT_LONG = paths.DATA_PATH / "model_comparison_long.parquet"

    mo.md(
        f"""
        **Config actual**
        - K_LIST: `{K_LIST}`
        - NUM_ITERS: `{NUM_ITERS}`
        - M_STEP_NUM_ITERS: `{M_STEP_NUM_ITERS}`
        - STICKINESS: `{STICKINESS}`
        - N_RESTARTS: `{N_RESTARTS}`
        - OUT_LONG: `{OUT_LONG.name}`
        """
    )
    return (
        BASE_SEED,
        DATA_PATH,
        K_LIST,
        M_STEP_NUM_ITERS,
        NUM_ITERS,
        N_RESTARTS,
        OUT_LONG,
        STICKINESS,
    )


@app.cell
def _(DATA_PATH, mo, pl):
    df = pl.read_parquet(DATA_PATH)
    subjects = df.select("subject").unique().sort("subject").to_series().to_list()
    mo.md(f"Subjects in the df: **{len(subjects)}** $\Rightarrow$ `{subjects}`")
    multiselect_subjects = mo.ui.multiselect(options=subjects)
    multiselect_models = mo.ui.multiselect(options=["glmhmm", "glmhmm-t"])
    return df, multiselect_models, multiselect_subjects, subjects


@app.cell
def _(mo, multiselect_models, multiselect_subjects):
    selected_subjects = multiselect_subjects.value
    selected_models = multiselect_models.value
    mo.hstack(
        [
            mo.vstack(
                [
                    multiselect_subjects,
                    mo.md(
                        f"**Chosen subjects ({len(multiselect_subjects.value)}):** {', '.join(f'`{s}`' for s in multiselect_subjects.value) or '_None_'}"
                    ),
                ]
            ),
            mo.vstack(
                [
                    multiselect_models,
                    mo.md(
                        f"**Chosen models ({len(multiselect_models.value)}):** {', '.join(f'`{m}`' for m in multiselect_models.value) or '_None_'}"
                    ),
                ]
            ),
        ]
    )
    return selected_models, selected_subjects


@app.cell(hide_code=True)
def _(np):
    def brier_multiclass(p, y, *, eps=1e-12, renorm=True):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y)

        p = np.clip(p, eps, 1.0)
        if renorm:
            s = p.sum(axis=1, keepdims=True)
            p = p / np.clip(s, eps, None)

        T, C = p.shape
        oh = np.zeros((T, C), dtype=float)
        oh[np.arange(T), y] = 1.0

        return np.mean(np.sum((p - oh) ** 2, axis=1))

    return (brier_multiclass,)


@app.cell(hide_code=True)
def _(SoftmaxGLMHMM, brier_multiclass, build_sequence_from_df, jnp, jr, np):
    def fit_one_subject_one_model(
        df_sub: "pl.DataFrame",
        K: int,
        model_kind: str,  # "glmhmm" o "glmhmm-t"
        num_iters: int,
        m_step_num_iters: int,
        stickiness: float,
        base_seed: int,
        n_restarts: int,
    ):
        """
        Returns the best fit (by ll) between n_restarts
        Saves total LL and LL/trial of the last iteration
        """
        y, X, U, names, _ = build_sequence_from_df(df_sub)

        T = int(y.shape[0])
        X = jnp.asarray(X)
        U = jnp.asarray(U)

        if model_kind == "glmhmm":
            emission_input_dim = int(X.shape[1])
            transition_input_dim = 0
            inputs = X
        elif model_kind == "glmhmm-t":
            emission_input_dim = int(X.shape[1])
            transition_input_dim = int(U.shape[1])
            inputs = jnp.concatenate([X[:, :], U], axis=1)
        else:
            raise ValueError("model_kind must be 'glmhmm' or 'glmhmm-t'")

        best = None

        for r in range(n_restarts):
            seed = base_seed + 10_000 * K + 100 * r
            key = jr.PRNGKey(seed)

            model = SoftmaxGLMHMM(
                num_states=K,
                num_classes=3,
                emission_input_dim=emission_input_dim,
                transition_input_dim=transition_input_dim,
                transition_matrix_stickiness=stickiness,
                m_step_num_iters=m_step_num_iters,
            )

            params, props = model.initialize(key=key)
            fitted_params, lps = model.fit_em(
                params=params,
                props=props,
                emissions=y,
                inputs=inputs,
                num_iters=num_iters,
            )

            lps_np = np.asarray(lps)
            ll_total_final = float(lps_np[-1])
            ll_per_trial_final = float(lps_np[-1] / T)
            p_pred = np.asarray(
                model.predict_choice_probs(
                    fitted_params, y, jnp.concatenate([X[:, :], U], axis=1)
                )
            )
            p_true = p_pred[np.arange(T), y]
            acc = (p_pred.argmax(axis=1) == y).mean()
            brier = brier_multiclass(p_pred, y)
            p = p_pred[np.arange(len(y)), y]
            nll_manual = -np.mean(np.log(p))
            print("Manual NLL:", nll_manual)
            print("fit_em final:", -lps[-1] / T)
            cand = {
                "fitted_params": fitted_params,
                "model": model,
                "p_true": p_true,
                "acc": acc,
                "brier": brier,
                "ll_total_final": ll_total_final,
                "ll_per_trial_final": ll_per_trial_final,
                "T": T,
                "restart": r,
                "seed": seed,
                "emission_input_dim": emission_input_dim,
                "transition_input_dim": transition_input_dim,
            }

            if (best is None) or (cand["ll_total_final"] > best["ll_total_final"]):
                best = cand

        return {
            "fitted_params": best["fitted_params"],
            "model": best["model"],
            "p_true": best["p_true"],
            "acc": best["acc"],
            "brier": best["brier"],
            "ll_total_final": best["ll_total_final"],
            "ll_per_trial_final": best["ll_per_trial_final"],
            "T": best["T"],
            "best_restart": best["restart"],
            "seed": best["seed"],
            "emission_input_dim": best["emission_input_dim"],
            "transition_input_dim": best["transition_input_dim"],
        }

    return (fit_one_subject_one_model,)


@app.cell
def _(
    BASE_SEED,
    K_LIST,
    M_STEP_NUM_ITERS,
    NUM_ITERS,
    N_RESTARTS,
    OUT_LONG,
    STICKINESS,
    df,
    fit_one_subject_one_model,
    mo,
    pl,
    selected_models,
    selected_subjects,
):
    jobs = [
        (subj, K, model_kind)
        for subj in selected_subjects
        for K in K_LIST
        for model_kind in selected_models
    ]

    rows = []
    fitted_params_store = {}  # (subj, K, model_kind) -> {"fitted_params", "model", "p_true"}

    for i, (subj, K, model_kind) in enumerate(
        mo.status.progress_bar(
            jobs,
            title="Fitting models",
            subtitle="",
            show_eta=True,
            show_rate=True,
        ),
        start=1,
    ):
        df_sub = df.filter(pl.col("subject") == subj).sort("trial_idx")

        out = fit_one_subject_one_model(
            df_sub=df_sub,
            K=K,
            model_kind=model_kind,
            num_iters=NUM_ITERS,
            m_step_num_iters=M_STEP_NUM_ITERS,
            stickiness=STICKINESS,
            base_seed=BASE_SEED + (hash(subj) % 1000),
            n_restarts=N_RESTARTS,
        )

        fitted_params_store[(subj, K, model_kind)] = {
            "fitted_params": out.pop("fitted_params"),
            "model": out.pop("model"),
            "p_true": out.pop("p_true"),
        }

        rows.append(
            {
                "subject": subj,
                "K": K,
                "model": model_kind,
                **out,
                "num_iters": NUM_ITERS,
                "m_step_num_iters": M_STEP_NUM_ITERS,
                "stickiness": STICKINESS,
                "n_restarts": N_RESTARTS,
            }
        )

    results_long = pl.DataFrame(rows)
    results_long.write_parquet(OUT_LONG)
    results_long
    return K, df_sub, fitted_params_store, model_kind, results_long


@app.cell
def _(pl, results_long):
    agg = (
        results_long.group_by(["K", "model"])
        .agg(
            [
                pl.len().alias("n_subjects"),
                pl.mean("ll_per_trial_final").alias("ll_per_trial_mean"),
                pl.std("ll_per_trial_final").alias("ll_per_trial_std"),
            ]
        )
        .with_columns(
            [
                (pl.col("ll_per_trial_std") / (pl.col("n_subjects") ** 0.5)).alias(
                    "ll_per_trial_sem"
                ),
            ]
        )
        .sort(["model", "K"])
    )
    return (agg,)


@app.cell
def _(fitted_params_store, jnp, np):
    fp = fitted_params_store[("A92", 2, "glmhmm-t")]
    bias = np.asarray(fp["fitted_params"].transitions.bias)  # (K, K)
    # softmax over rows gives the "average" transition matrix
    from jax.nn import softmax

    A_avg = np.asarray(softmax(jnp.array(bias), axis=-1))
    print(A_avg)
    return (bias,)


@app.cell
def _(
    K,
    build_sequence_from_df,
    df_sub,
    fitted_params_store,
    mo,
    np,
    plt,
    selected_models,
    selected_subjects,
    sns,
):
    import pandas as pd

    records = []
    _, _, _, names, _ = build_sequence_from_df(df_sub)
    for _subj in selected_subjects:
        key = (_subj, K, selected_models[0])
        if key not in fitted_params_store:
            continue
        _fp = fitted_params_store[key]["fitted_params"]
        # W shape: (K, n_classes-1, input_dim)  or  (K, n_classes, input_dim)
        W = np.asarray(_fp.emissions.weights)  # adjust attribute name if needed
        for k in range(W.shape[0]):
            for c in range(W.shape[1]):
                for _f, fname in enumerate(names["X_cols"]):
                    records.append(
                        {
                            "subject": _subj,
                            "state": f"s{k}",
                            "class": c,
                            "feature": fname,
                            "weight": float(W[k, c, _f]),
                        }
                    )

    df_w = pd.DataFrame(records)

    _fig, _axes = plt.subplots(1, W.shape[1], figsize=(8, 4), sharey=True)

    for c, _ax in enumerate(_axes):
        sns.lineplot(
            data=df_w[df_w["class"] == c],
            x="feature",
            y="weight",
            hue="state",
            ax=_ax,
            markers=True,
            marker="o",
            markersize=10,
            markeredgewidth=0,
            alpha=0.75,
        )
        _ax.get_legend().remove()

        # label each line at its last point
        for line, (state, group) in zip(_ax.get_lines(), df_w[df_w["class"] == c].groupby("state")):
            last_x = names["X_cols"][1]
            first_y = group[group["feature"] == last_x]["weight"].mean()
            _ax.text(
                1.05, (5 if state == 's0' else -5),
                f"{'State 0' if state == 's0' else 'State 1'}",
                color=line.get_color(),
                fontsize=10, fontweight="bold", va="center"
            )
        _ax.set_title(f"{'Left' if c == 0 else 'Right'}")
        _ax.set_xticklabels(labels = names["X_cols"], rotation=30, ha="right")
        _ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        _ax.set_xlabel("")
    _fig.tight_layout()
    sns.despine()
    _fig


    trans_records = []

    for _subj in selected_subjects:
        key = (_subj, K, "glmhmm-t")
        if key not in fitted_params_store:
            continue
        _fp = fitted_params_store[key]["fitted_params"]
        W = np.asarray(_fp.transitions.weights)  # (K, K, D)
        for k_from in range(W.shape[0]):
            for k_to in range(W.shape[1]):
                for _f, fname in enumerate(names["U_cols"]):
                    trans_records.append({
                        "subject": _subj,
                        "transition": f"s{k_from}→s{k_to}",
                        "feature": fname,
                        "weight": float(W[k_from, k_to, _f]),
                    })

    df_trans = pd.DataFrame(trans_records)
    transitions = sorted(df_trans["transition"].unique())

    _fig2, _axes = plt.subplots(1, len(transitions), figsize=(4 * len(transitions), 4), sharey=True)
    _axes = np.atleast_1d(_axes)

    for _ax, trans in zip(_axes, transitions):
        sns.lineplot(
            data=df_trans[df_trans["transition"] == trans],
            x="feature", y="weight",
            ax=_ax,
            markers=True,
            marker="o",
            markersize=10,
            alpha=0.75,
            markeredgewidth=0,
            color="steelblue",
        )
        _ax.set_title(trans)
        _ax.set_xticks(range(len(names["U_cols"])))
        _ax.set_xticklabels(names["U_cols"], rotation=30, ha="right")
        _ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        _ax.set_xlabel("")

    _fig.tight_layout()
    sns.despine()
    mo.vstack([_fig,_fig2])

    return df_w, names, pd


@app.cell
def _(K, df_w, names, plt, selected_models, sns):
    _fig, _ax = plt.subplots(figsize=(6, 4))

    # map class to color, state to linestyle
    _palette = {0: "steelblue", 1: "tomato"}  # 0=L, 1=R
    _linestyles = {f"s{k}": ls for k, ls in enumerate([(1,0), (4,2), (2,2,4,2)])}

    for (_state, _class), _group in df_w.groupby(["state", "class"]):
        _mean = _group.groupby("feature")["weight"].mean().reindex(names["X_cols"])
        _ax.plot(
            range(len(names["X_cols"])),
            _mean.values,
            color=_palette[_class],
            dashes=_linestyles[_state],
            marker="o",
            markersize=8,
            markeredgewidth=0,
            alpha=0.75,
            label=f"{_state} {'L' if _class == 0 else 'R'}",
        )

    _ax.set_xticks(range(len(names["X_cols"])))
    _ax.set_xticklabels(names["X_cols"], rotation=30, ha="right")
    _ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    _ax.set_ylabel("GLM weight")
    _ax.set_xlabel("")
    _ax.set_title(f"{selected_models[0]} K={K} — emission weights (ref = C)")
    _ax.legend(title="state / side", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    _fig.tight_layout()
    sns.despine()
    _fig
    return


@app.cell
def _(K, fitted_params_store, names, np, pd, plt, selected_subjects, sns):
    import copy

    trans_records_std = []

    for _subj in selected_subjects:
        _key = (_subj, K, "glmhmm-t")
        if _key not in fitted_params_store:
            continue
        _fp = fitted_params_store[_key]["fitted_params"]
        W_raw = np.asarray(_fp.transitions.weights)  # (K, K, D)
    
        # For each from-state, standardize the (K, D) destination weights
        # Average across from-states to get a single (K, D) summary
        W_avg_from = W_raw.mean(axis=0)  # (K, D) — averaged over from-states
    
        # append zero reference row and mean-center (paper's trick)
        W_aug = np.vstack([W_avg_from, np.zeros((1, W_avg_from.shape[1]))])  # (K+1, D)
        v1 = -np.mean(W_aug, axis=0)
        W_std = copy.deepcopy(W_aug)
        W_std[-1] = v1
        for _k in range(K):
            W_std[_k] = v1 + W_avg_from[_k]

        for _k in range(K):
            for _f, _fname in enumerate(names["U_cols"]):
                trans_records_std.append({
                    "subject": _subj,
                    "state": f"s{_k}",
                    "feature": _fname,
                    "weight": float(W_std[_k, _f]),
                })

    df_trans_std = pd.DataFrame(trans_records_std)

    _fig2, _ax2 = plt.subplots(figsize=(5, 4))

    sns.lineplot(
        data=df_trans_std,
        x="feature", y="weight",
        hue="state",
        ax=_ax2,
        markers=True,
        marker="o",
        markersize=10,
        alpha=0.75,
        markeredgewidth=0,
    )

    _ax2.get_legend().remove()
    for _line, (_state, _group) in zip(_ax2.get_lines(), df_trans_std.groupby("state")):
        _last_y = _group[_group["feature"] == names["U_cols"][-1]]["weight"].mean()
        _ax2.text(
            len(names["U_cols"]) - 1 + 0.05, _last_y,
            _state, color=_line.get_color(),
            fontsize=10, fontweight="bold", va="center"
        )

    _ax2.set_xticks(range(len(names["U_cols"])))
    _ax2.set_xticklabels(names["U_cols"], rotation=30, ha="right")
    _ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    _ax2.set_ylabel("Transition Weights")
    _ax2.set_xlabel("")
    _ax2.set_title(f"glmhmm-t K={K} — transition weights")
    _fig2.tight_layout()
    sns.despine()
    _fig2
    return


@app.cell
def _(
    K,
    bias,
    fitted_params_store,
    model_kind,
    np,
    plt,
    selected_models,
    selected_subjects,
    sns,
):
    _fig, _axes = plt.subplots(1, len(selected_models), figsize=(4 * len(selected_models), 4))
    _axes = np.atleast_1d(_axes)

    for _ax, _model_kind in zip(_axes, selected_models):
        # collect transition matrices across subjects
        A_list = []
        for _subj in selected_subjects:
            _key = (_subj, K, _model_kind)
            if _key not in fitted_params_store:
                continue
            _fp = fitted_params_store[_key]["fitted_params"]
            if model_kind == "glmhmm":
                A = np.asarray(_fp.transitions.transition_matrix)  # (K, K)
            else:
                # glmhmm-t: softmax over bias (input-independent component)
                _bias = np.asarray(_fp.transitions.bias)  # (K, K)
                A = np.exp(bias) / np.exp(bias).sum(axis=-1, keepdims=True)
            A_list.append(A)

        A_mean = np.mean(A_list, axis=0)  # (K, K)
        light_blue = sns.color_palette("grey", as_cmap=True)
        sns.heatmap(
            A_mean, ax=_ax, annot=True, fmt=".2f", vmin=0, vmax=1,square=True,
            cbar=False, 
            linewidths=0.5,
            cmap = light_blue,
            xticklabels=[f"s{k}" for k in range(K)],
            yticklabels=[f"s{k}" for k in range(K)],
        )
        _ax.set_title(f"{model_kind} — mean transition matrix (K={K})")
        _ax.set_xlabel("To state")
        _ax.set_ylabel("From state")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(K_LIST, agg, plt, sns):
    fig, ax = plt.subplots(figsize=(7, 4))

    for model in ["glmhmm", "glmhmm-t"]:
        dfm = agg.filter(agg["model"] == model).sort("K")
        K_plot = dfm["K"].to_numpy()
        mu = dfm["ll_per_trial_mean"].to_numpy()
        sem = dfm["ll_per_trial_sem"].to_numpy()
        ax.plot(K_plot, mu, marker="o", label=model)
        # ax.fill_between(K_plot, mu - sem, mu + sem, alpha=0.2)

    ax.set_xlabel("Number of states (K)")
    ax.set_ylabel("Log-likelihood/trial")
    ax.set_title("")
    ax.set_xticks(K_LIST)
    ax.legend(labels=["GLMHMM", "GLMHMM-t"])
    fig.tight_layout()
    sns.despine()
    fig
    return


@app.cell
def _(df, np, pl):
    CLASSES = ["L", "C", "R"]


    def entropy_nats(p):
        p = np.asarray(p, dtype=float)
        p = p[(p > 0) & np.isfinite(p)]
        return float(-(p * np.log(p)).sum()) if p.size else 0.0


    def H_conditional_pl(
        df_subj: pl.DataFrame,
        cond_cols=("stimd_c", "ttype_c", "x_c"),
        resp_col="r_c",
    ) -> float:
        if df_subj.height == 0:
            return np.nan

        cols = list(cond_cols) + [resp_col]
        d = (
            df_subj.select(cols)
            .with_columns(
                [pl.col(c).cast(pl.Utf8).str.strip_chars() for c in cols]
            )
            .filter(pl.col(resp_col).is_in(CLASSES))
        )
        if d.height == 0:
            return np.nan

        N = float(d.height)
        H = 0.0

        # partition_by devuelve una lista de dataframes (uno por grupo)
        for df_s in d.partition_by(list(cond_cols), as_dict=False):
            n_s = float(df_s.height)
            if n_s <= 0:
                continue

            vc = df_s[
                resp_col
            ].value_counts()  # DataFrame con 2 columnas: [resp_col, count]
            # detecta el nombre real de la columna count
            count_col = [c for c in vc.columns if c != resp_col][0]

            # convierte a dict: { "L": 123, "C": 456, ... }
            count_map = dict(zip(vc[resp_col].to_list(), vc[count_col].to_list()))

            q = np.array([count_map.get(c, 0) / n_s for c in CLASSES], dtype=float)
            H += (n_s / N) * entropy_nats(q)

        return float(H)


    subjects2 = df.select("subject").unique().to_series().to_list()

    H_full_map = {}
    H_bal_map = {}

    for s2 in subjects2:
        df_sub2 = df.filter(pl.col("subject") == s2)

        H_full_map[s2] = H_conditional_pl(
            df_sub2, cond_cols=("stimd_n", "ttype_n", "x_c"), resp_col="r_c"
        )

        df_bal = df
        H_bal_map[s2] = H_conditional_pl(
            df_bal, cond_cols=("stimd_n", "ttype_n", "x_c"), resp_col="r_c"
        )

    # sanity check rápido
    print(
        "H_full min/max:",
        np.nanmin(list(H_full_map.values())),
        np.nanmax(list(H_full_map.values())),
    )
    print(
        "H_bal  min/max:",
        np.nanmin(list(H_bal_map.values())),
        np.nanmax(list(H_bal_map.values())),
    )
    return (H_conditional_pl,)


@app.cell
def _(H_conditional_pl, OUT_LONG, df, np, pl):
    def _():
        results_long = pl.read_parquet(OUT_LONG)
        subjects = df.select("subject").unique().to_series().to_list()

        H_full_map = {}
        H_bal_map = {}

        for subj in subjects:
            df_sub = df.filter(pl.col("subject") == subj)

            H_full_map[subj] = H_conditional_pl(
                df_sub, cond_cols=("stimd_n", "ttype_n", "x_c"), resp_col="r_c"
            )

            H_bal_map[subj] = H_conditional_pl(
                df_sub, cond_cols=("stimd_n", "ttype_n", "x_c"), resp_col="r_c"
            )

        # sanity check rápido
        print(
            "H_full min/max:",
            np.nanmin(list(H_full_map.values())),
            np.nanmax(list(H_full_map.values())),
        )
        print(
            "H_bal  min/max:",
            np.nanmin(list(H_bal_map.values())),
            np.nanmax(list(H_bal_map.values())),
        )
        print("log(3) =", np.log(3))

        def rel_vs_ceiling_from_ll(ll_per_trial: float, H_cs: float) -> float:
            if not np.isfinite(ll_per_trial) or not np.isfinite(H_cs):
                return np.nan
            nll = -ll_per_trial
            denom = np.log(3.0) - H_cs
            if denom <= 0:
                return np.nan
            val = 1.0 - ((nll - H_cs) / denom)
            return val

        results_gof = results_long.with_columns(
            [
                pl.col("subject")
                .map_elements(
                    lambda s: H_full_map.get(s, np.nan), return_dtype=pl.Float64
                )
                .alias("H_cond"),
                (-pl.col("ll_per_trial_final")).alias("nll_per_trial_final"),
            ]
        ).with_columns(
            [
                pl.struct(["nll_per_trial_final", "H_cond"])
                .map_elements(
                    lambda r: rel_vs_ceiling_from_ll(
                        r["nll_per_trial_final"], r["H_cond"]
                    ),
                    return_dtype=pl.Float64,
                )
                .alias("gof_insample")
            ]
        )
        return results_gof


    results_gof = _()
    results_gof
    return (results_gof,)


@app.cell
def _(K_LIST, pl, plt, results_gof, sns):
    agg_gof = (
        results_gof.group_by(["model", "K"])
        .agg(
            [
                pl.mean("gof_insample").alias("gof_mean"),
                pl.std("gof_insample").alias("gof_std"),
                pl.len().alias("n"),
            ]
        )
        .with_columns((pl.col("gof_std") / pl.col("n").sqrt()).alias("gof_sem"))
        .sort(["model", "K"])
    )

    fig2, ax2 = plt.subplots(figsize=(7, 4))

    for model2 in ["glmhmm", "glmhmm-t"]:
        dfm2 = agg_gof.filter(pl.col("model") == model2).sort("K")
        K_plot2 = dfm2["K"].to_numpy()
        mu2 = dfm2["gof_mean"].to_numpy()
        sem2 = dfm2["gof_sem"].to_numpy()

        ax2.plot(K_plot2, mu2, marker="o", label=model2)

        if len(K_plot2) > 1:
            ax2.fill_between(
                K_plot2, mu2 - 1.96 * sem2, mu2 + 1.96 * sem2, alpha=0.2
            )
        else:
            ax2.errorbar(K_plot2, mu2, yerr=1.96 * sem2, fmt="o", capsize=4)

    ax2.set_xlabel("Número de estados (K)")
    ax2.set_ylabel("GOF (rel vs empirical ceiling)")
    ax2.set_title("GOF in-sample GLMHMM vs GLMHMM-t")
    ax2.set_xticks(K_LIST)
    ax2.set_ylim(0, 5)
    ax2.legend()
    sns.despine()
    fig2.tight_layout()
    fig2
    return


@app.cell
def _(mo, subjects):
    COL_L = "#E41A1C"
    COL_C = "#377EB8"
    COL_R = "#4DAF4A"


    ui_subject = mo.ui.dropdown(options=subjects, value="A92", label="Subject")
    ui_tau = mo.ui.slider(start=1, stop=300, step=1, value=50, label="tau")
    ui_show_A = mo.ui.checkbox(value=True, label="Show A_L/A_C/A_R")
    ui_show_ew = mo.ui.checkbox(value=False, label="Show EWMA")
    return COL_C, COL_L, COL_R, ui_show_A, ui_show_ew, ui_subject, ui_tau


app._unparsable_cell(
    r"""
    _df_sub = df.filter(pl.col("subject") == ui_subject.value).sort("trial_idx")
    #y2, _X, U2, names, A_pm = build_sequence_from_df(
        _df_sub, tau=int(ui_tau.value)
    )

    _U = np.asarray(U2)
    T = _U.shape[0]

    ui_range = mo.ui.range_slider(
        start=0, stop=max(T - 1, 1), value=(0, T - 1), label="Trial range"
    )
    """,
    name="_"
)


@app.cell
def _(
    COL_C,
    COL_L,
    COL_R,
    T,
    U2,
    mo,
    np,
    pl,
    plt,
    sns,
    ui_range,
    ui_show_A,
    ui_show_ew,
    ui_subject,
    ui_tau,
    y2,
):
    _U = U2
    _y = y2
    lo, hi = ui_range.value
    x = np.arange(T)[lo : hi + 1]

    AL = _U[:, 1][lo : hi + 1]
    AC = _U[:, 2][lo : hi + 1]
    AR = _U[:, 3][lo : hi + 1]

    y_np = np.asarray(_y).astype(int).reshape(-1)
    lam = np.exp(-1.0 / float(ui_tau.value))

    df_ew = pl.DataFrame(
        {
            "L": (y_np == 0).astype(int),
            "C": (y_np == 1).astype(int),
            "R": (y_np == 2).astype(int),
        }
    ).with_columns(
        [
            pl.col("L")
            .ewm_mean(half_life=ui_tau.value, adjust=False)
            .alias("EW_L"),
            pl.col("C")
            .ewm_mean(half_life=ui_tau.value, adjust=False)
            .alias("EW_C"),
            pl.col("R")
            .ewm_mean(half_life=ui_tau.value, adjust=False)
            .alias("EW_R"),
        ]
    )

    EW_L = df_ew["EW_L"].to_numpy()[lo : hi + 1]
    EW_C = df_ew["EW_C"].to_numpy()[lo : hi + 1]
    EW_R = df_ew["EW_R"].to_numpy()[lo : hi + 1]

    fig1 = plt.figure(figsize=(12, 4))

    if ui_show_A.value:
        plt.plot(x, AL, color=COL_L, label=f"$A_L$")
        plt.plot(x, AC, color=COL_C, label=f"$A_C$")
        plt.plot(x, AR, color=COL_R, label=f"$A_R$")

    if ui_show_ew.value:
        plt.plot(x, EW_L, "--", color=COL_L, label=f"$EW_L$")
        plt.plot(x, EW_C, "--", color=COL_C, label=f"$EW_C$")
        plt.plot(x, EW_R, "--", color=COL_R, label=f"$EW_R$")

    plt.xlabel("trial")
    plt.ylabel("value")
    plt.title(f"{ui_subject.value} | tau={int(ui_tau.value)}")
    plt.legend(ncol=2)
    plt.tight_layout()
    sns.despine()

    mo.vstack(
        [
            mo.hstack([ui_subject, ui_tau, ui_show_A, ui_show_ew]),
            ui_range,
            fig1,
        ]
    )
    return fig1, hi, lo, x


@app.cell
def _(A_pm, U2, hi, lo, mo, np, y2):
    _U = U2
    _y = y2
    _Apm = np.asarray(A_pm, dtype=np.float32)  # (T,2): col0=A_plus, col1=A_minus

    Aplus = _Apm[:, 0][lo : hi + 1]
    Aminus = _Apm[:, 1][lo : hi + 1]

    # UI toggles
    ui_show_Aplus = mo.ui.checkbox(value=True, label="show $A^{+}$")
    ui_show_Aminus = mo.ui.checkbox(value=True, label="show $A^{-}$")
    return Aminus, Aplus, ui_show_Aminus, ui_show_Aplus


@app.cell
def _(
    Aminus,
    Aplus,
    fig1,
    mo,
    plt,
    sns,
    ui_range,
    ui_show_Aminus,
    ui_show_Aplus,
    ui_show_ew,
    ui_subject,
    ui_tau,
    x,
):
    _fig = plt.figure(figsize=(12, 4))

    if ui_show_Aplus.value:
        plt.plot(x, Aplus, label=r"$A^{+}$")

    if ui_show_Aminus.value:
        plt.plot(x, Aminus, label=r"$A^{-}$")

    plt.xlabel("trial")
    plt.ylabel("value")
    plt.title(f"{ui_subject.value} | tau={int(ui_tau.value)}")
    plt.legend(ncol=2)
    plt.tight_layout()
    sns.despine()

    mo.vstack(
        [
            mo.hstack(
                [ui_subject, ui_tau, ui_show_Aplus, ui_show_Aminus, ui_show_ew]
            ),
            ui_range,
            _fig,
            fig1,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
