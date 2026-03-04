import numpy as np
import polars as pl
import sys
import argparse
import os
import json
import hashlib
from pathlib import Path
from scipy.special import log_softmax, softmax
from scipy.optimize import minimize

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

try:
    from glmhmmt.features import build_sequence_from_df, build_sequence_from_df_2afc
except ImportError:
    # Fallback if running from scripts dir directly without package install
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from glmhmmt.features import build_sequence_from_df, build_sequence_from_df_2afc


def fit_subject(
    subject: str,
    emission_cols: list[str] | None = None,
    tau: float = 5.0,
    num_classes: int = 3,
    task: str = "MCDR",
    lapse: bool = False,
    lapse_max: float = 0.2,
) -> dict:
    """Fit a GLM (K=1) to a single subject."""
    
    # Force binary for 2AFC
    if task == "2AFC":
        num_classes = 2

    # 1. Load Data
    if task == "MCDR":
        df = pl.read_parquet(paths.DATA_PATH / "df_filtered.parquet")
        if "subject" not in df.columns: return None
        df_sub = df.filter(pl.col("subject") == subject).sort("trial_idx")
        if len(df_sub) == 0: return None
        
        # GLM-HMM features builder returns (y, X, U, names, AU)
        res = build_sequence_from_df(df_sub, tau=tau, emission_cols=emission_cols)
        if len(res) == 5:
             y, X, _, names, _ = res
        else:
             y, X, names = res

    else:  # 2AFC
        df = pl.read_parquet(paths.DATA_PATH / "alexis_combined.parquet")
        df_sub = df.filter(pl.col("subject") == subject)
        if len(df_sub) == 0: return None
        y, X, names = build_sequence_from_df_2afc(df_sub, emission_cols=emission_cols)

    # 2. Minimize Negative Log Likelihood
    T, M = X.shape
    y_np = np.asarray(y, dtype=int)
    X_np = np.asarray(X, dtype=float)

    # Use lapse model only for 2AFC
    fit_lapse = lapse and (num_classes == 2)

    if fit_lapse:
        # Parameters: [w (M,), gamma_L, gamma_R]
        # gamma_L = P(right | truly left), gamma_R = P(left | truly right)
        def neg_log_likelihood(w_flat):
            w      = w_flat[:M]
            gL     = w_flat[M]       # lapse → right when stimulus is left
            gR     = w_flat[M + 1]   # lapse → left  when stimulus is right
            p_right_base = 1.0 / (1.0 + np.exp(-(X_np @ w)))
            p_right = gL + (1.0 - gL - gR) * p_right_base
            p_right = np.clip(p_right, 1e-10, 1 - 1e-10)
            log_p_R = np.log(p_right)
            log_p_L = np.log(1.0 - p_right)
            return -np.sum(np.where(y_np == 1, log_p_R, log_p_L))

        n_params = M + 2
        bounds   = [(-np.inf, np.inf)] * M + [(0.0, lapse_max), (0.0, lapse_max)]
        x0       = np.zeros(n_params)
    else:
        def neg_log_likelihood(w_flat):
            W_pair = w_flat.reshape(num_classes - 1, M)
            if num_classes == 3:
                logits = np.stack([X_np @ W_pair[0], np.zeros(T), X_np @ W_pair[1]], axis=1)
            else:
                logits = np.stack([np.zeros(T), X_np @ W_pair[0]], axis=1)
            log_p = log_softmax(logits, axis=1)
            return -np.sum(log_p[np.arange(T), y_np])

        n_params = (num_classes - 1) * M
        bounds   = None
        x0       = np.zeros(n_params)

    if T > 0:
        res = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9}
        )
        success = res.success
        w_flat  = res.x
        nll     = res.fun
    else:
        success = False
        w_flat  = np.zeros(n_params)
        nll     = np.nan

    # Extract lapse rates
    if fit_lapse:
        lapse_rates = np.array([w_flat[M], w_flat[M + 1]])  # [gamma_L, gamma_R]
        w_flat_w    = w_flat[:M]
    else:
        lapse_rates = np.zeros(2)
        w_flat_w    = w_flat

    # Reconstruct W_full (C, M)
    W_pair = w_flat_w.reshape(num_classes - 1, M)
    if num_classes == 3:
        W_full = np.stack([W_pair[0], np.zeros(M), W_pair[1]], axis=0)  # [L, C=0, R]
    else:
        W_full = np.stack([np.zeros(M), W_pair[0]], axis=0)  # [L=0, R]

    # Predict (with lapses if fitted)
    if num_classes == 3:
        logits = np.stack([X_np @ W_pair[0], np.zeros(T), X_np @ W_pair[1]], axis=1)
        p_pred = softmax(logits, axis=1)
    else:
        p_right_base = 1.0 / (1.0 + np.exp(-(X_np @ W_pair[0])))
        if fit_lapse:
            gL, gR = lapse_rates
            p_right = gL + (1.0 - gL - gR) * p_right_base
        else:
            p_right = p_right_base
        p_pred = np.stack([1.0 - p_right, p_right], axis=1)

    return {
        "subject": subject,
        "W": W_full,              # (C, M)
        "p_pred": p_pred,         # (T, C)
        "lapse_rates": lapse_rates,  # [gamma_L, gamma_R]
        "nll": nll,
        "success": success,
        "y": y_np,
        "X": X_np,
        "names": names,
        "T": T
    }

def save_results(result: dict, out_dir: Path, tau: float):
    if result is None: return
    
    subj = result["subject"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as {subj}_glm_arrays.npz
    prefix = out_dir / f"{subj}_glm"
    
    # Prepare W for saving in glmhmm compatible format (K, C-1, M)
    # result["W"] is (C, M) including the reference 0.
    # We want to exclude the reference.
    # For 3 classes (L, C, R), ref is C (idx 1). We want [L, R] -> indices [0, 2].
    # For 2 classes (L, R), ref is R (idx 1). We want [L] -> index [0].
    
    W_full = result["W"]
    C, M = W_full.shape
    if C == 3:
        W_save = W_full[[0, 2]]  # (2, M) — L and R, skip C=ref
    else:
        W_save = W_full[[1]]     # (1, M) — R is active, L=ref(idx 0) is zeros
    
    W_save = W_save[None, ...] # (1, C-1, M)

    np.savez(
        str(prefix) + "_arrays.npz",
        emission_weights=W_save,
        p_pred=result["p_pred"],
        y=result["y"],
        X=result["X"],
        smoothed_probs=np.ones((result["T"], 1)),  # K=1, prob=1 everywhere
        X_cols=result["names"]["X_cols"] if "X_cols" in result["names"] else [],
        lapse_rates=result.get("lapse_rates", np.zeros(2)),
        success=result["success"],
    )

    # Metrics — count lapse params in k if non-zero
    _lapse = result.get("lapse_rates", np.zeros(2))
    _n_lapse_params = int(np.any(_lapse > 0)) * 2
    acc = float(np.mean(np.argmax(result["p_pred"], axis=1) == result["y"])) if result["T"] > 0 else 0.0
    k = (result["W"].shape[0] - 1) * result["W"].shape[1] + _n_lapse_params
    bic = k * np.log(result["T"]) + 2 * result["nll"] if result["T"] > 0 else np.nan
    
    pl.DataFrame({
        "subject": [subj],
        "model_kind": ["glm"],
        "tau": [tau],
        "nll": [result["nll"]],
        "bic": [bic],
        "acc": [acc],
        "k": [k],
        "n_trials": [result["T"]]
    }).write_parquet(str(prefix) + "_metrics.parquet")


def generate_model_id(task, tau, emission_cols, lapse: bool = False):
    cols = sorted(emission_cols) if emission_cols else []
    config = {
        "task": task,
        "tau": float(tau),
        "emission_cols": cols,
        "lapse": lapse,
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def main(
    subjects: list[str] | None = None,
    out_dir: Path | None = None,
    tau: float = 5.0,
    emission_cols: list[str] | None = None,
    num_classes: int = 3,
    task: str = "MCDR",
    model_alias: str | None = None,
    lapse: bool = False,
    lapse_max: float = 0.2,
):
    # Compute base output directory
    base_out_dir = paths.RESULTS / "fits" / task 

    # Generate Hash
    model_hash = generate_model_id(task, tau, emission_cols, lapse=lapse)
    out_dirs = [base_out_dir / model_hash]

    if model_alias:
        out_dirs.append(base_out_dir / model_alias)
        # If out_dir provided as argument, it overrides only if model_alias is not set?
        # But wait, out_dir was previously just `paths.RESULTS / "fits" / task / "glm_baseline"`.
        # So I will overwrite out_dir based on logic now.

    if out_dir is not None:
         # If user explicitly passed out_dir, use it instead (legacy support or rigorous override)
         out_dirs = [out_dir]
         if model_alias:
             # If alias is also provided, perhaps save to both?
             out_dirs.append(base_out_dir / model_alias)
    
    # Ensure directories exist
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        # Save config
        with open(d / "config.json", "w") as f:
             json.dump({
                 "task": task,
                 "tau": tau,
                 "emission_cols": emission_cols,
                 "num_classes": num_classes,
                 "lapse": lapse,
                 "lapse_max": lapse_max,
                 "model_id": d.name
             }, f, indent=4)
        
    print(f"Fitting GLM | Task={task} Tau={tau} Hash={model_hash} Alias={model_alias} N={len(subjects) if subjects else 'All'}")

    # Force binary for 2AFC
    if task == "2AFC":
        num_classes = 2

    if subjects is None:
        if task == "MCDR":
            df = pl.read_parquet(paths.DATA_PATH / "df_filtered.parquet")
            df = df.filter(pl.col("subject") != "A84")
        else:
            df = pl.read_parquet(paths.DATA_PATH / "alexis_combined.parquet")
        subjects = df["subject"].unique().sort().to_list()

    print(f"Fitting GLM | Task={task} Tau={tau} N={len(subjects)}")
    
    for subj in subjects:
        print(f"  Fitting {subj}...")
        try:
            res = fit_subject(
                subj,
                tau=tau,
                emission_cols=emission_cols,
                num_classes=num_classes,
                task=task,
                lapse=lapse,
                lapse_max=lapse_max,
            )
            for d in out_dirs:
                save_results(res, d, tau)
        except Exception as e:
            print(f"  Failed {subj}: {e}")
            import traceback
            traceback.print_exc()
    
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--tau", type=float, default=50.0) 
    parser.add_argument("--task", type=str, default="MCDR", choices=["MCDR", "2AFC"])
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--model_alias", type=str, default=None)
    parser.add_argument("--lapse", action="store_true", default=False,
                        help="Fit symmetric lapse rates γ_L, γ_R ∈ [0, lapse_max]")
    parser.add_argument("--lapse_max", type=float, default=0.2,
                        help="Upper bound for each lapse rate (default 0.20)")

    args = parser.parse_args()

    main(
        subjects=args.subjects,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        tau=args.tau,
        task=args.task,
        num_classes=args.num_classes,
        model_alias=args.model_alias,
        lapse=args.lapse,
        lapse_max=args.lapse_max,
    )
