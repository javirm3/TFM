import json
import numpy as np
import polars as pl
import jax.numpy as jnp
import jax.random as jr
import sys
from pathlib import Path
from typing import Any, Callable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
from glmhmmt.model import SoftmaxGLMHMM, normalize_frozen_emissions, serialize_frozen_emissions
from scripts.fit_common import stable_model_id, valid_trial_mask
from tasks import get_adapter

ProgressCallback = Callable[[dict[str, Any]], None]


def generate_model_id(
    task: str,
    K: int,
    tau: float,
    emission_cols: list | None = None,
    transition_cols: list | None = None,
    frozen_emissions: dict | None = None,
) -> str:
    """Stable 8-char hash over the GLMHMMT-defining model configuration."""
    return stable_model_id(
        task=task,
        K=K,
        tau=tau,
        emission_cols=emission_cols,
        transition_cols=transition_cols,
        frozen_emissions=frozen_emissions,
    )


def fit_subject(
    subject: str,
    K: int,
    num_iters: int = 50,
    n_restarts: int = 5,
    base_seed: int = 0,
    m_step_num_iters: int = 100,
    stickiness: float = 10.0,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
    frozen_emissions: dict[int, dict[str, float]] | dict[str, dict[str, float]] | None = None,
    tau: float = 50.0,
    task: str = "MCDR",
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    adapter = get_adapter(task)
    df = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df = adapter.subject_filter(df)
    df_sub = df.filter(pl.col("subject") == subject).sort(adapter.sort_col)
    y, X, U, names = adapter.load_subject(
        df_sub, tau=tau, emission_cols=emission_cols, transition_cols=transition_cols
    )
    num_classes = adapter.num_classes
    session_ids = df_sub[adapter.session_col].to_numpy()
    frozen = normalize_frozen_emissions(frozen_emissions)

    # Drop trials from sessions too short for EM (must match _split_by_session)
    mask = valid_trial_mask(session_ids)
    y, X, U = y[mask], X[mask], U[mask]
    session_ids = session_ids[mask]
    inputs_all = jnp.concatenate([X, U], axis=1)

    model = SoftmaxGLMHMM(
        num_states=K,
        num_classes=num_classes,
        emission_input_dim=X.shape[1],
        transition_input_dim=U.shape[1],
        m_step_num_iters=m_step_num_iters,
        transition_matrix_stickiness=stickiness,
        frozen_emissions=frozen or None,
        emission_feature_names=names.get("X_cols", []),
    )

    best_lp, best_params = -np.inf, None
    for r in range(n_restarts):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "restart_start",
                    "subject": subject,
                    "K": K,
                    "restart_index": r + 1,
                    "restart_total": n_restarts,
                }
            )
        key = jr.PRNGKey(base_seed + r)
        params, props = model.initialize(key=key)
        fp, lps = model.fit_em_multisession(
            params=params, props=props,
            emissions=y, inputs=inputs_all,
            session_ids=session_ids,
            num_iters=num_iters,
            verbose=verbose,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "restart_complete",
                    "subject": subject,
                    "K": K,
                    "restart_index": r + 1,
                    "restart_total": n_restarts,
                    "log_prob": float(lps[-1]),
                }
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
        "U": np.asarray(U),
        "frozen_emissions": serialize_frozen_emissions(frozen),
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
    num_classes = result["num_classes"]
    n_params = (
        result["K"] * (result["K"] - 1) *
        (1 + result["U"].shape[1])  # transition
        # emission
        + result["K"] * (num_classes - 1) * result["X"].shape[1]
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
        names=result["names"],
        y=result["y"],
        X=result["X"],
        U=result["U"],
        X_cols=np.array(result["names"].get("X_cols", []), dtype=object),
        U_cols=np.array(result["names"].get("U_cols", []), dtype=object),
        frozen_emissions_json=np.array(json.dumps(result["frozen_emissions"], sort_keys=True)),
    )


def main(
    subjects: list[str] | None = None,
    K_list: list[int] = [2, 3],
    num_iters: int = 50,
    n_restarts: int = 1,
    base_seed: int = 0,
    out_dir: Path | None = None,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
    frozen_emissions: dict[int, dict[str, float]] | dict[str, dict[str, float]] | None = None,
    tau: float = 50.0,
    task: str = "MCDR",
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
):
    import json
    adapter = get_adapter(task)
    if out_dir is None:
        out_dir = paths.RESULTS / "fits" / task / "glmhmmt"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_spec = serialize_frozen_emissions(frozen_emissions)
    with open(out_dir / "config.json", "w") as _f:
        json.dump({
            "task": task,
            "tau": tau,
            "subjects": subjects,
            "emission_cols": emission_cols or adapter.default_emission_cols(),
            "transition_cols": transition_cols or adapter.default_transition_cols(),
            "frozen_emissions": frozen_spec,
            "K_list": K_list,
            "model_id": out_dir.name,
        }, _f, indent=4)
    if subjects is None:
        df = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
        df = adapter.subject_filter(df)
        subjects = df["subject"].unique().sort().to_list()

    for subj_idx, subj in enumerate(subjects, start=1):
        for k_idx, K in enumerate(K_list, start=1):
            if verbose:
                print(f"Fitting glmhmm-t | subject={subj} K={K} task={task} ...")
            def _progress(info: dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                progress_callback(
                    {
                        **info,
                        "subject_index": subj_idx,
                        "subject_total": len(subjects),
                        "k_index": k_idx,
                        "k_total": len(K_list),
                    }
                )

            result = fit_subject(
                subj, K,
                num_iters=num_iters,
                n_restarts=n_restarts,
                base_seed=base_seed,
                emission_cols=emission_cols,
                transition_cols=transition_cols,
                frozen_emissions=frozen_spec,
                tau=tau,
                task=task,
                verbose=verbose,
                progress_callback=_progress if progress_callback is not None else None,
            )
            save_results(result, out_dir)
            if verbose:
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
    parser.add_argument("--tau", type=float, default=50.0,
                        help="Half-life for exponential action traces.")
    parser.add_argument("--task", type=str, default="MCDR",
                        help="Task to fit: 'MCDR' or '2AFC'.")
    parser.add_argument(
        "--frozen_emissions",
        type=str,
        default=None,
        help='JSON object mapping state indices to {feature: fixed_value}, e.g. \'{"0":{"SL":0.0}}\'.',
    )
    args = parser.parse_args()
    frozen_emissions = json.loads(args.frozen_emissions) if args.frozen_emissions else None
    main(
        subjects=args.subjects,
        K_list=args.K,
        num_iters=args.num_iters,
        n_restarts=args.n_restarts,
        base_seed=args.seed,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        tau=args.tau,
        task=args.task,
        frozen_emissions=frozen_emissions,
    )
