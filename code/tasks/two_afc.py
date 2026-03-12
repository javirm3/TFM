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

    # ── state-scoring options ────────────────────────────────────────────────
    # For 2AFC the weight matrix is (K, 1, M) where W[k,0,:] = logit(Left)
    # weights (reference = Right).  The plot shows -W for intuition.
    # Modes:
    #   "neg"  – -W[k, 0, fi]  (more negative raw = more stimulus-following)
    #   "abs"  – |W[k, 0, fi]|  (unsigned magnitude)
    #   "pos"  – +W[k, 0, fi]  (raw positive = anti-stimulus tendency)
    # Score per state = mean over listed pairs.
    _SCORING_OPTIONS: dict = {
        "stim_vals (-w)": [("stim_vals", "neg")],
        "stim_vals (|w|)": [("stim_vals", "abs")],
        "at_choice (|w|)": [("at_choice", "abs")],
        "wsls (|w|)": [("wsls", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "stim_vals (-w)"

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

    # ── column mapping ───────────────────────────────────────────────────────

    @property
    def behavioral_cols(self) -> dict:
        """2AFC column mapping (canonical → actual)."""
        return {
            "trial_idx":   "Trial",
            "trial":       "Trial",
            "session":     "Session",
            "stimulus":    "Side",
            "response":    "response",
            "performance": "performance",
        }

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
        """2AFC engagement scoring.

        K=2: Engaged = argmax(-stim_vals raw), Disengaged = the other.
        K=3: Engaged = argmax(-stim_vals raw); the remaining two are split
             by bias weight: min(displayed bias) = "Biased L",
             max(displayed bias) = "Biased R".
        K>3: remaining states labelled "Disengaged 1", "Disengaged 2", ...
             ordered by descending -stim_vals score.
        """
        import numpy as np

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
            W       = np.asarray(W)   # (K, 1, M)
            name2fi = {n: i for i, n in enumerate(feat)}

            # displayed weight = -raw; argmax(-raw) = most stimulus-following
            stim_fi = name2fi.get("stim_vals")
            if stim_fi is not None:
                stim_scores = -W[:, 0, stim_fi]
            else:
                stim_scores = -W[:, 0, :].mean(axis=1)  # fallback

            engaged_k = int(np.argmax(stim_scores))
            others    = [k for k in range(K) if k != engaged_k]

            labels: dict = {engaged_k: "Engaged"}

            if K == 2:
                labels[others[0]] = "Disengaged"
                order = [engaged_k, others[0]]

            elif K == 3:
                bias_fi = name2fi.get("bias")
                if bias_fi is not None:
                    # displayed bias = -raw; lower displayed = more left-biased
                    bias_disp = -W[others, 0, bias_fi]
                    biased_l = others[int(np.argmin(bias_disp))]
                    biased_r = others[int(np.argmax(bias_disp))]
                else:
                    biased_l, biased_r = others[0], others[1]
                labels[biased_l] = "Biased L"
                labels[biased_r] = "Biased R"
                order = [engaged_k, biased_l, biased_r]

            else:
                # K>3: rank remaining by stim score descending
                others_sorted = sorted(others, key=lambda k: stim_scores[k], reverse=True)
                for dis, k in enumerate(others_sorted, start=1):
                    labels[k] = f"Disengaged {dis}"
                order = [engaged_k] + others_sorted

            state_labels[subj] = labels
            state_order[subj]  = order

        return state_labels, state_order