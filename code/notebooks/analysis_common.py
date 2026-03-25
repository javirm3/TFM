from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl


def resolve_selected_model_id(
    current_hash: str,
    existing_model: str | None,
    alias: str | None,
) -> str:
    return existing_model or alias or current_hash


def _extract_saved_names(arrays: dict) -> dict[str, list[str]]:
    saved_names: dict[str, list[str]] = {}
    raw_names = arrays.get("names")
    if raw_names is None or getattr(raw_names, "shape", None) != ():
        return saved_names
    raw_dict = raw_names.item()
    if not isinstance(raw_dict, dict):
        return saved_names
    for key in ("X_cols", "U_cols"):
        values = raw_dict.get(key)
        if values is not None:
            saved_names[key] = [str(v) for v in values]
    return saved_names


def _first_matching_width(width: int, *candidates) -> list[str]:
    if width <= 0:
        return []

    longer: list[str] | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        cols = [str(v) for v in list(candidate)]
        if len(cols) == width:
            return cols
        if longer is None and len(cols) > width:
            longer = cols[:width]
    if longer is not None:
        return longer
    return [f"feature_{idx}" for idx in range(width)]


def _first_matching_u_width(width: int, *candidates) -> list[str]:
    if width <= 0:
        return []

    longer: list[str] | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        cols = [str(v) for v in list(candidate)]
        if len(cols) == width:
            return cols
        if longer is None and len(cols) > width:
            longer = cols[:width]
    if longer is not None:
        return longer
    return [f"transition_feature_{idx}" for idx in range(width)]


def load_fit_arrays(
    *,
    out_dir: Path,
    arrays_suffix: str,
    adapter,
    df_all: pl.DataFrame,
    subjects: list[str],
    emission_cols: list[str] | None,
    transition_cols: list[str] | None = None,
    k: int | None = None,
    postprocess_array: Callable[[dict], dict] | None = None,
) -> tuple[dict, dict[str, list[str]]]:
    df_for_names = df_all.filter(pl.col("subject").is_in(subjects)) if subjects else df_all.head(0)
    try:
        resolved_names = adapter.resolve_design_names(
            emission_cols=emission_cols,
            transition_cols=transition_cols,
            df=df_for_names,
        )
    except Exception:
        resolved_names = {"X_cols": [], "U_cols": []}

    files = list(sorted(out_dir.glob(f"*_{arrays_suffix}")))
    if k is not None:
        files += [f for f in sorted(out_dir.glob(f"*_K{k}_{arrays_suffix}")) if f not in files]

    arrays_store = {}
    for path in files:
        subject = path.name.removesuffix(f"_{arrays_suffix}")
        if k is not None:
            subject = subject.removesuffix(f"_K{k}")

        arrays = dict(np.load(path, allow_pickle=True))
        if postprocess_array is not None:
            arrays = postprocess_array(arrays)

        saved_names = _extract_saved_names(arrays)
        emission_width = int(np.asarray(arrays["emission_weights"]).shape[2]) if "emission_weights" in arrays else 0
        arrays["X_cols"] = _first_matching_width(
            emission_width,
            arrays.get("X_cols"),
            saved_names.get("X_cols"),
            resolved_names.get("X_cols"),
        )

        transition_width = 0
        if "transition_weights" in arrays:
            transition_width = int(np.asarray(arrays["transition_weights"]).shape[2])
        elif "U" in arrays:
            transition_width = int(np.asarray(arrays["U"]).shape[1])
        arrays["U_cols"] = _first_matching_u_width(
            transition_width,
            arrays.get("U_cols"),
            saved_names.get("U_cols"),
            resolved_names.get("U_cols"),
        )

        arrays_store[subject] = arrays

    names = {
        "X_cols": list(resolved_names.get("X_cols", [])),
        "U_cols": list(resolved_names.get("U_cols", [])),
    }
    return arrays_store, names


def select_subject_behavior_df(
    df_all: pl.DataFrame,
    *,
    subject,
    sort_col,
    session_col: str,
    min_session_length: int = 1,
) -> pl.DataFrame:
    df_sub = df_all.filter(pl.col("subject") == subject).sort(sort_col)
    if min_session_length > 1:
        df_sub = df_sub.filter(pl.col(session_col).count().over(session_col) >= min_session_length)
    return df_sub


def build_trial_and_weights_df(
    df_all: pl.DataFrame,
    *,
    views: dict,
    adapter,
    min_session_length: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    from glmhmmt.postprocess import build_emission_weights_df, build_trial_df

    trial_frames = []
    for subject, view in views.items():
        df_sub = select_subject_behavior_df(
            df_all,
            subject=subject,
            sort_col=adapter.sort_col,
            session_col=adapter.session_col,
            min_session_length=min_session_length,
        )
        if df_sub.height != view.T:
            print(f"⚠️  {subject}: row mismatch ({df_sub.height} vs {view.T}), skipping")
            continue
        trial_frames.append(build_trial_df(view, adapter, df_sub, adapter.behavioral_cols))

    trial_df = pl.concat(trial_frames) if trial_frames else pl.DataFrame()
    weights_df = build_emission_weights_df(views)
    return trial_df, weights_df
