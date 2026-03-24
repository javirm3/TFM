import hashlib
import json
from typing import Any, Callable

import jax.random as jr
import jax.numpy as jnp
import numpy as np
import polars as pl

from glmhmmt.model import serialize_frozen_emissions

FIT_ROW_ID_COL = "_fit_row_id"
CV_SESSION_ID_COL = "_cv_session_id"


def valid_trial_mask(session_ids: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Return a boolean mask keeping only trials from sessions with >= min_length trials."""
    ids, counts = np.unique(session_ids, return_counts=True)
    keep = set(ids[counts >= min_length])
    return np.array([session_id in keep for session_id in session_ids])


def apply_valid_trial_mask(session_ids: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mask all arrays using the standard minimum-session-length rule."""
    mask = valid_trial_mask(session_ids)
    masked = [np.asarray(arr)[mask] for arr in arrays]
    return (*masked, np.asarray(session_ids)[mask])


def stable_model_id(
    task: str,
    K: int,
    tau: float,
    emission_cols: list | None = None,
    transition_cols: list | None = None,
    frozen_emissions: dict | None = None,
    cv_mode: str = "none",
    cv_repeats: int = 0,
) -> str:
    """Stable 8-char MD5 hash over the fit-defining model configuration."""
    cv_mode = str(cv_mode)
    cv_repeats = int(cv_repeats) if cv_mode != "none" else 0
    config = {
        "task": task,
        "K": int(K),
        "tau": float(tau),
        "emission_cols": sorted(emission_cols) if emission_cols else [],
        "transition_cols": sorted(transition_cols) if transition_cols else [],
        "frozen_emissions": serialize_frozen_emissions(frozen_emissions),
        "cv_mode": cv_mode,
        "cv_repeats": cv_repeats,
    }
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def fit_best_restart(
    model,
    *,
    n_restarts: int,
    base_seed: int,
    fit_once: Callable[[Any, Any], tuple[Any, Any]],
    failure_message: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_payload: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, int]:
    """Run repeated seeded fits and keep the restart with the best final log-probability."""
    best_lp = -np.inf
    best_params = None
    best_lps = None
    best_restart = -1
    payload = progress_payload or {}

    for restart_idx in range(int(n_restarts)):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "restart_start",
                    "restart_index": restart_idx + 1,
                    "restart_total": int(n_restarts),
                    **payload,
                }
            )

        key = jr.PRNGKey(int(base_seed) + restart_idx)
        params, props = model.initialize(key=key)
        fitted_params, lps = fit_once(params, props)
        final_lp = float(np.asarray(lps)[-1])

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "restart_complete",
                    "restart_index": restart_idx + 1,
                    "restart_total": int(n_restarts),
                    "log_prob": final_lp,
                    **payload,
                }
            )

        if final_lp > best_lp:
            best_lp = final_lp
            best_params = fitted_params
            best_lps = np.asarray(lps)
            best_restart = restart_idx + 1

    if best_params is None or best_lps is None:
        raise ValueError(failure_message)
    return best_params, best_lps, int(best_restart)


def attach_feature_row_ids(feature_df: pl.DataFrame) -> pl.DataFrame:
    """Ensure a stable row-id column exists for later split bookkeeping."""
    if FIT_ROW_ID_COL in feature_df.columns:
        return feature_df
    return feature_df.with_row_index(FIT_ROW_ID_COL)


def _sort_columns(sort_col: str | list[str]) -> list[str]:
    return list(sort_col) if isinstance(sort_col, list) else [sort_col]


def _trial_order_column(sort_col: str | list[str]) -> str:
    cols = _sort_columns(sort_col)
    return cols[-1]


def build_balanced_holdout(
    feature_df: pl.DataFrame,
    labels: pl.Series | np.ndarray | list[Any],
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Return balanced train/test trial subsets matched across all label bins."""
    df = attach_feature_row_ids(feature_df)
    label_arr = labels.to_numpy() if isinstance(labels, pl.Series) else np.asarray(labels)
    if label_arr.shape[0] != df.height:
        raise ValueError(
            f"CV labels length {label_arr.shape[0]} does not match feature_df height {df.height}."
        )

    row_ids = df[FIT_ROW_ID_COL].to_numpy()
    by_label: dict[Any, list[int]] = {}
    for row_id, label in zip(row_ids, label_arr, strict=True):
        if label is None:
            continue
        if isinstance(label, (float, np.floating)) and not np.isfinite(float(label)):
            continue
        by_label.setdefault(label, []).append(int(row_id))

    if len(by_label) < 2:
        raise ValueError("Balanced CV requires at least two non-empty condition bins.")

    min_count = min(len(rows) for rows in by_label.values())
    n_per_split = min_count // 2
    if n_per_split < 1:
        raise ValueError(
            "Balanced CV requires at least two trials in every condition bin "
            f"after filtering, got min_count={min_count}."
        )

    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    test_ids: list[int] = []
    label_counts: dict[str, int] = {}
    for label in sorted(by_label):
        pool = np.asarray(by_label[label], dtype=int)
        picked = rng.choice(pool, size=2 * n_per_split, replace=False)
        rng.shuffle(picked)
        train_ids.extend(picked[:n_per_split].tolist())
        test_ids.extend(picked[n_per_split:].tolist())
        label_counts[str(label)] = int(n_per_split)

    train_df = df.filter(pl.col(FIT_ROW_ID_COL).is_in(train_ids))
    test_df = df.filter(pl.col(FIT_ROW_ID_COL).is_in(test_ids))
    return train_df, test_df, {
        "labels": [float(lbl) if isinstance(lbl, (int, float, np.integer, np.floating)) else str(lbl) for lbl in sorted(by_label)],
        "min_count": int(min_count),
        "n_per_label_per_split": int(n_per_split),
        "label_counts": label_counts,
    }


def resegment_subsampled_trials(
    feature_df: pl.DataFrame,
    session_col: str,
    sort_col: str | list[str],
    output_session_col: str = CV_SESSION_ID_COL,
) -> pl.DataFrame:
    """Create synthetic session ids so removed trials do not imply fake adjacency."""
    sort_cols = _sort_columns(sort_col)
    trial_col = _trial_order_column(sort_col)
    df_sorted = feature_df.sort(sort_cols)
    if df_sorted.height == 0:
        return df_sorted.with_columns(pl.Series(output_session_col, [], dtype=pl.Int32))

    sess = np.asarray(df_sorted[session_col].to_numpy())
    trial = np.asarray(df_sorted[trial_col].to_numpy())
    seg_ids = np.zeros(df_sorted.height, dtype=np.int32)
    current_seg = 0
    for idx in range(1, df_sorted.height):
        prev_trial = float(trial[idx - 1])
        curr_trial = float(trial[idx])
        is_gap = (
            sess[idx] != sess[idx - 1]
            or not np.isfinite(prev_trial)
            or not np.isfinite(curr_trial)
            or curr_trial != prev_trial + 1
        )
        if is_gap:
            current_seg += 1
        seg_ids[idx] = current_seg
    return df_sorted.with_columns(pl.Series(output_session_col, seg_ids))


def raw_loglik_multisession(model, params, emissions, inputs, session_ids) -> float:
    """Return the summed data log-likelihood without any parameter prior."""
    sessions = model._split_by_session(emissions, inputs, session_ids)
    e_pad, i_pad, _ = model._pad_sessions(sessions)
    _batch_stats, ll_batch = model._batched_e_step_jit(params, e_pad, i_pad)
    return float(jnp.sum(ll_batch))


def score_split(model, params, emissions, inputs, session_ids) -> dict[str, Any]:
    """Evaluate one fitted model on a train or test split."""
    T = int(np.asarray(emissions).shape[0])
    raw_ll = raw_loglik_multisession(model, params, emissions, inputs, session_ids)
    p_pred = np.asarray(
        model.predict_choice_probs_multisession(params, emissions, inputs, session_ids=session_ids)
    )
    acc = float(np.mean(np.argmax(p_pred, axis=1) == np.asarray(emissions))) if T else np.nan
    return {
        "raw_ll": raw_ll,
        "ll_per_trial": raw_ll / T if T else np.nan,
        "acc": acc,
        "T": T,
        "p_pred": p_pred,
    }
