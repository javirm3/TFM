import marimo

__generated_with = "0.19.10"
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
def _(mo, paths):
    DATA_PATH = paths.DATA_PATH / "df_filtered.parquet"
    K_LIST = range(2, 6)

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
    return df, subjects


@app.cell
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


@app.cell
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
        y, X, U, names = build_sequence_from_df(df_sub)

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
            "ll_total_final": best["ll_total_final"],
            "ll_per_trial_final": best["ll_per_trial_final"],
            "p_true": p_true,
            "acc": acc,
            "brier": brier,
            "T": best["T"],
            "best_restart": best["restart"],
            "seed": best["seed"],
            "emission_input_dim": best["emission_input_dim"],
            "transition_input_dim": best["transition_input_dim"],
        }

    return (fit_one_subject_one_model,)


@app.cell(disabled=True)
def _(df, fit_one_subject_one_model, pl, subjects):
    for s in subjects:
        ou2t = fit_one_subject_one_model(
            df_sub=df.filter(pl.col("subject") == s),
            K=2,
            model_kind="glmhmm-t",
            num_iters=100,
            m_step_num_iters=100,
            stickiness=10,
            base_seed=12345678 + (hash(s) % 100000),
            n_restarts=1,
        )
        print(ou2t)
    return


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
    subjects,
):
    models = ["glmhmm", "glmhmm-t"]

    jobs = [
        (subj, K, model_kind)
        for subj in subjects
        for K in K_LIST
        for model_kind in models
    ]

    rows = []

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
    return (results_long,)


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
    ax.legend(labels = ["GLMHMM", "GLMHMM-t"])
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
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
