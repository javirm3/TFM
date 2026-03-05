"""Task adapter for the MCDR (3-AFC rats) task."""
from __future__ import annotations

import types
from typing import List, Tuple, Dict, Any

import jax.numpy as jnp
import polars as pl

from tasks import TaskAdapter, _register
from glmhmmt.features import (
    build_sequence_from_df,
    _ALL_EMISSION_COLS,
    _ALL_TRANSITION_COLS,
)


@_register(["mcdr"])
class MCDRAdapter(TaskAdapter):
    """Adapter for the 3-AFC MCDR rat data."""

    num_classes: int = 3
    data_file: str   = "df_filtered.parquet"
    sort_col: str    = "trial_idx"
    session_col: str = "session"

    # ── data preparation ────────────────────────────────────────────────────

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("subject") != "A84")

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` via :func:`build_sequence_from_df`."""
        y, X, U, names, _ = build_sequence_from_df(
            df_sub,
            tau=tau,
            emission_cols=emission_cols,
            transition_cols=transition_cols,
        )
        return y, X, U, names

    # ── column defaults ─────────────────────────────────────────────────────

    def default_emission_cols(self) -> List[str]:
        return list(_ALL_EMISSION_COLS)

    def default_transition_cols(self) -> List[str]:
        return list(_ALL_TRANSITION_COLS)

    # ── plots ────────────────────────────────────────────────────────────────

    def get_plots(self) -> types.ModuleType:
        import glmhmmt.plots as plots
        return plots
    # ── state labelling ─────────────────────────────────────────────────────

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        """MCDR engagement scoring: S_coh = mean(W[k, L, SL], W[k, R, SR])."""
        import numpy as np

        def _scoh(W, feat_names):
            name2fi = {n: i for i, n in enumerate(feat_names)}
            scores = np.zeros(W.shape[0])
            n = 0
            if "SL" in name2fi:
                scores += W[:, 0, name2fi["SL"]]; n += 1
            if "SR" in name2fi:
                scores += W[:, 1, name2fi["SR"]]; n += 1
            return scores / max(1, n)

        base_feat = list(names.get("X_cols", []))
        state_labels: dict = {}
        state_order: dict  = {}
        for subj in subjects:
            W = arrays_store[subj].get("emission_weights") if subj in arrays_store else None
            if W is None:
                state_labels[subj] = {k: f"State {k+1}" for k in range(K)}
                state_order[subj]  = list(range(K))
                continue
            feat    = list(arrays_store[subj].get("X_cols", base_feat))
            scores  = _scoh(np.asarray(W), feat)
            ranking = list(np.argsort(scores)[::-1])
            labels: dict = {}
            dis = 1
            for rank, k in enumerate(ranking):
                if rank == 0:
                    labels[int(k)] = "Engaged"
                else:
                    labels[int(k)] = "Disengaged" if K == 2 else f"Disengaged {dis}"
                    dis += 1
            state_labels[subj] = labels
            state_order[subj]  = [int(k) for k in ranking]
        return state_labels, state_order