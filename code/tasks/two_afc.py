"""Task adapter for the 2AFC (Alexis human) task."""
from __future__ import annotations

import types
from typing import List, Tuple, Dict, Any

import numpy as np
import jax.numpy as jnp
import polars as pl

from tasks import TaskAdapter, _register
from glmhmmt.features import (
    build_sequence_from_df_2afc,
    _ALL_2AFC_EMISSION_COLS,
    _SF_COL_PREFIX,
)

# Default experiments to keep (avoids habituation / drug sessions)
_KEEP_EXPERIMENTS = ["2AFC_2", "2AFC_3", "2AFC_4", "2AFC_6"]


@_register(["two_afc", "2afc"])
class TwoAFCAdapter(TaskAdapter):
    """Adapter for the binary 2-AFC human data (Alexis)."""

    num_classes: int = 2
    data_file: str   = "alexis_combined.parquet"
    sort_col: str    = "Trial"
    session_col: str = "Session"

    # ── data preparation ────────────────────────────────────────────────────

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("Experiment").is_in(_KEEP_EXPERIMENTS))

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,  # ignored for 2AFC
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` with an empty U (no transition features).

        ``U`` has shape ``(T, 0)`` and ``names["U_cols"] = []``.
        This makes the caller uniform: ``transition_input_dim = U.shape[1] == 0``
        degrades glmhmmt to standard HMM transitions automatically.
        """
        y, X, names = build_sequence_from_df_2afc(
            df_sub,
            emission_cols=emission_cols,
        )
        T = y.shape[0]
        U = jnp.empty((T, 0), dtype=jnp.float32)
        names = {**names, "U_cols": []}
        return y, X, U, names

    # ── column defaults ─────────────────────────────────────────────────────

    def default_emission_cols(self) -> List[str]:
        # Exclude stim_strength (multi-column) by default; include sf_ cols at runtime
        return [c for c in _ALL_2AFC_EMISSION_COLS if c != "stim_strength"]

    def default_transition_cols(self) -> List[str]:
        return []

    def sf_cols(self, df: pl.DataFrame) -> List[str]:
        """Return any stimulus-frame (sf_*) columns present in *df*."""
        return [c for c in df.columns if c.startswith(_SF_COL_PREFIX)]

    # ── plots ────────────────────────────────────────────────────────────────

    def get_plots(self) -> types.ModuleType:
        import glmhmmt.plots_alexis as plots
        return plots
    # ── state labelling ─────────────────────────────────────────────────────

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        """2AFC engagement: state with highest |stim weight| is 'Engaged'."""
        import numpy as np
        _STIM_NAMES = {"stim_vals", "stim_d", "ild_norm", "ILD", "ild",
                       "stimulus", "net_ild", "stim_strength"}
        base_feat = list(names.get("X_cols", []))
        state_labels: dict = {}
        state_order: dict  = {}
        for subj in subjects:
            W = arrays_store[subj].get("emission_weights") if subj in arrays_store else None
            if W is None:
                state_labels[subj] = {k: f"State {k+1}" for k in range(K)}
                state_order[subj]  = list(range(K))
                continue
            feat     = list(arrays_store[subj].get("X_cols", base_feat))
            W        = np.asarray(W)  # (K, 1, M)
            stim_idx = next((i for i, n in enumerate(feat) if n in _STIM_NAMES), None)
            if stim_idx is not None:
                scores = np.abs(W[:, 0, stim_idx])
            else:
                scores = np.abs(W[:, 0, :]).mean(axis=1)  # fallback
            ranking = list(np.argsort(scores)[::-1])       # highest |stim w| = Engaged
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