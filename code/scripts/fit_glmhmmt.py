from glmhmmt.features import build_sequence_from_df
from glmhmmt.model import SoftmaxGLMHMM
import paths
import numpy as np
import polars as pl
import jax.numpy as jnp
import jax.random as jr
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _valid_trial_mask(session_ids: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Return a boolean mask keeping only trials from sessions with >= min_length trials."""
    ids, counts = np.unique(session_ids, return_counts=True)
    keep = set(ids[counts >= min_length])
    return np.array([s in keep for s in session_ids])


    subject: str,
    K: int,
    num_iters: int = 50,
    n_restarts: int = 5,
    base_seed: int = 0,
    m_step_num_iters: int = 100,
    stickiness: float = 10.0,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
) -> dict:
    df = pl.read_parquet(paths.DATA_PATH / "df_filtered.parquet")
    df_sub = df.filter(pl.col("subject") == subject).sort("trial_idx")
    y, X, U, names, _ = build_sequence_from_df(
        df_sub,
        emission_cols=emission_cols,
        transition_cols=transition_cols,
    )
    session_ids = df_sub["session"].to_numpy()

    # Drop trials from sessions too short for EM (must match _split_by_session)
    mask = _valid_trial_mask(session_ids)
    y, X, U = y[mask], X[mask], U[mask]
    session_ids = session_ids[mask]
    inputs_all = jnp.concatenate([X, U], axis=1)

    model = SoftmaxGLMHMM(
        num_states=K,
        num_classes=3,
        emission_input_dim=X.shape[1],
        transition_input_dim=U.shape[1],
        m_step_num_iters=m_step_num_iters,
        transition_matrix_stickiness=stickiness,
    )

    best_lp, best_params = -np.inf, None
    for r in range(n_restarts):
        key = jr.PRNGKey(base_seed + r)
        params, props = model.initialize(key=key)
        fp, lps = model.fit_em_multisession(
            params=params, props=props,
            emissions=y, inputs=inputs_all,
            session_ids=session_ids,
            num_iters=num_iters,
            verbose=True,
        )
        if float(lps[-1]) > best_lp:
            best_lp = float(lps[-1])
            best_params = fp
            best_lps = np.asarray(lps)

    smoothed_probs = model.smoother_multisession(
        params=best_params, emissions=y, inputs=inputs_all, session_ids=session_ids)
    p_pred = model.predict_choice_probs_multisession(
        best_params, y, inputs_all, session_ids=session_ids)
    T = int(y.shape[0])

    return {
        "subject": subject,
        "K": K,
        "model": model,
        "fitted_params": best_params,
        "lps": best_lps,
        "smoothed_probs": smoothed_probs,
        "p_pred": p_pred,
        "T": T,
        "names": names,
        "y": np.asarray(y),
        "X": np.asarray(X),
        "U": np.asarray(U),
    }


def save_results(result: dict, out_dir: Path) -> None:
    subj = result["subject"]
    K = result["K"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{subj}_K{K}_glmhmmt"

    T = result["T"]
    p_pred = result["p_pred"]
    acc = float(np.mean(np.argmax(p_pred, axis=1) == result["y"]))
    ll_per_trial = float(result["lps"][-1]) / T
    n_params = (
        result["K"] * (result["K"] - 1) *
        (1 + result["U"].shape[1])  # transition
        # emission
        + result["K"] * 2 * result["X"].shape[1]
    )
    bic = -2 * float(result["lps"][-1]) + n_params * np.log(T)

    pl.DataFrame({
        "subject": [subj], "K": [K], "model_kind": ["glmhmm-t"],
        "ll_per_trial": [ll_per_trial], "bic": [bic], "acc": [acc],
    }).write_parquet(str(prefix) + "_metrics.parquet")

    np.savez(
        str(prefix) + "_arrays.npz",
        lps=result["lps"],
        p_pred=p_pred,
        smoothed_probs=result["smoothed_probs"],
        emission_weights=np.asarray(result["fitted_params"].emissions.weights),
        transition_bias=np.asarray(result["fitted_params"].transitions.bias),
        transition_weights=np.asarray(
            result["fitted_params"].transitions.weights),
        y=result["y"],
        X=result["X"],
        U=result["U"],
    )


def main(
    subjects: list[str] | None = None,
    K_list: list[int] = [2, 3],
    num_iters: int = 50,
    n_restarts: int = 5,
    base_seed: int = 0,
    out_dir: Path | None = None,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
):
    if out_dir is None:
        out_dir = paths.RESULTS_PATH / "glmhmmt"
    if subjects is None:
        df = pl.read_parquet(paths.DATA_PATH / "df_filtered.parquet")
        subjects = df["subject"].unique().sort().to_list()

    for subj in subjects:
        for K in K_list:
            print(f"Fitting glmhmm-t | subject={subj} K={K} ...")
            result = fit_subject(
                subj, K,
                num_iters=num_iters,
                n_restarts=n_restarts,
                base_seed=base_seed,
                emission_cols=emission_cols,
                transition_cols=transition_cols,
            )
            save_results(result, out_dir)
            print(f"  ✓ saved to {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--K", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--n_restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    main(
        subjects=args.subjects,
        K_list=args.K,
        num_iters=args.num_iters,
        n_restarts=args.n_restarts,
        base_seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
