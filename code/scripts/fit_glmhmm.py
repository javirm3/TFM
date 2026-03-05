# filepath: /Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/scripts/fit_glmhmm.py
import hashlib
import json
import numpy as np
import polars as pl
import jax.numpy as jnp
import jax.random as jr
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
from glmhmmt.model import SoftmaxGLMHMM
from tasks import get_adapter


def generate_model_id(task: str, K: int, tau: float, emission_cols: list | None = None) -> str:
    """Stable 8-char MD5 hash over (task, K, tau, sorted emission_cols)."""
    cols = sorted(emission_cols) if emission_cols else []
    config = {"task": task, "K": int(K), "tau": float(tau), "emission_cols": cols}
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def _valid_trial_mask(session_ids: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Return a boolean mask keeping only trials from sessions with >= min_length trials."""
    ids, counts = np.unique(session_ids, return_counts=True)
    keep = set(ids[counts >= min_length])
    return np.array([s in keep for s in session_ids])


def fit_subject(
    subject: str,
    K: int,
    num_iters: int = 50,
    n_restarts: int = 5,
    base_seed: int = 0,
    m_step_num_iters: int = 100,
    emission_cols: list[str] | None = None,
    stickiness: float = 10.0,
    tau: float = 50.0,
    task: str = "MCDR",
) -> dict:
    adapter = get_adapter(task)
    df = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df = adapter.subject_filter(df)
    df_sub = df.filter(pl.col("subject") == subject).sort(adapter.sort_col)
    y, X, _, names = adapter.load_subject(df_sub, tau=tau, emission_cols=emission_cols)
    session_ids = df_sub[adapter.session_col].to_numpy()
    num_classes = adapter.num_classes

    # Drop trials from sessions too short for EM (must match _split_by_session)
    mask = _valid_trial_mask(session_ids)
    y, X = y[mask], X[mask]
    session_ids = session_ids[mask]

    model = SoftmaxGLMHMM(
        num_states=K,
        num_classes=num_classes,  # from adapter
        emission_input_dim=X.shape[1],
        transition_input_dim=0,
        m_step_num_iters=m_step_num_iters,
        transition_matrix_stickiness=stickiness,
    )

    best_lp, best_params = -np.inf, None
    for r in range(n_restarts):
        key = jr.PRNGKey(base_seed + r)
        params, props = model.initialize(key=key)
        fp, lps = model.fit_em_multisession(
            params=params, props=props,
            emissions=y, inputs=X,
            session_ids=session_ids,
            num_iters=num_iters,
            verbose=True,
        )
        if float(lps[-1]) > best_lp:
            best_lp = float(lps[-1])
            best_params = fp
            best_lps = np.asarray(lps)

    smoothed_probs = model.smoother_multisession(params=best_params, emissions=y, inputs=X, session_ids=session_ids)
    p_pred = model.predict_choice_probs_multisession(best_params, y, X, session_ids=session_ids)
    T = int(y.shape[0])

    return {
        "subject": subject,
        "K": K,
        "num_classes": num_classes,
        "model": model,
        "fitted_params": best_params,
        "lps": best_lps,
        "smoothed_probs": smoothed_probs,
        "p_pred": p_pred,
        "T": T,
        "names": names,
        "y": np.asarray(y),
        "X": np.asarray(X),
    }


def save_results(result: dict, out_dir: Path) -> None:
    subj = result["subject"]
    K = result["K"]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{subj}_K{K}_glmhmm"

    # scalar metrics as parquet
    T = result["T"]
    p_pred = result["p_pred"]
    acc = float(np.mean(np.argmax(p_pred, axis=1) == result["y"]))
    ll_per_trial = float(result["lps"][-1]) / T
    num_classes = result["num_classes"]
    n_params = result["K"] * (result["K"] - 1) + \
        result["K"] * (num_classes - 1) * result["X"].shape[1]
    bic = -2 * float(result["lps"][-1]) + n_params * np.log(T)

    pl.DataFrame({
        "subject": [subj], "K": [K], "model_kind": ["glmhmm"],
        "ll_per_trial": [ll_per_trial], "bic": [bic], "acc": [acc],
    }).write_parquet(str(prefix) + "_metrics.parquet")

    # arrays as npz
    # For input-driven transitions there is no single transition_matrix field;
    # save the input-marginal (bias-only) softmax as a summary (K, K) matrix.
    import jax.nn as jnn
    _tp = result["fitted_params"].transitions
    if hasattr(_tp, "transition_matrix"):
        _A = np.asarray(_tp.transition_matrix)
    else:
        _A = np.asarray(jnn.softmax(_tp.bias, axis=-1))  # (K, K)

    np.savez(
        str(prefix) + "_arrays.npz",
        lps=result["lps"],
        p_pred=p_pred,
        smoothed_probs=result["smoothed_probs"],
        emission_weights=np.asarray(result["fitted_params"].emissions.weights),
        transition_matrix=_A,
        y=result["y"],
        X=result["X"],
        X_cols=np.array(result["names"].get("X_cols", []), dtype=object),
    )


def main(
    subjects: list[str] | None = None,
    K_list: list[int] = [2, 3],
    num_iters: int = 50,
    n_restarts: int = 5,
    base_seed: int = 0,
    out_dir: Path | None = None,
    emission_cols: list[str] | None = None,
    tau: float = 50.0,
    task: str = "MCDR",
):
    import json
    adapter = get_adapter(task)
    if out_dir is None:
        out_dir = paths.RESULTS / "fits" / task / "glmhmm"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as _f:
        json.dump({
            "task": task,
            "tau": tau,
            "emission_cols": emission_cols or adapter.default_emission_cols(),
            "K_list": K_list,
            "model_id": out_dir.name,
        }, _f, indent=4)
    if subjects is None:
        df = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
        df = adapter.subject_filter(df)
        subjects = df["subject"].unique().sort().to_list()

    for subj in subjects:
        for K in K_list:
            print(f"Fitting glmhmm | subject={subj} K={K} tau={tau} task={task} ...")
            result = fit_subject(subj, K, num_iters=num_iters,
                                 n_restarts=n_restarts, base_seed=base_seed,
                                 tau=tau, emission_cols=emission_cols,
                                 task=task)
            print("Fitted, waiting to save")
            save_results(result, out_dir)
            print(f"  ✓ saved to {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="List of subject IDs. If None, fits all subjects.")
    parser.add_argument("--K", nargs="+", type=int, default=[2, 3],
                        help="List of number of states to fit.")
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--n_restarts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Defaults to RESULTS_PATH/glmhmm.")
    parser.add_argument("--tau", type=float, default=50.0,
                        help="Half-life for exponential action traces.")
    parser.add_argument("--task", type=str, default="MCDR",
                        help="Task to fit: 'MCDR' or '2AFC'. Affects data loading and features.")
    args = parser.parse_args()
    main(
        subjects=args.subjects,
        K_list=args.K,
        num_iters=args.num_iters,
        n_restarts=args.n_restarts,
        base_seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        tau=args.tau,
        task=args.task,
    )