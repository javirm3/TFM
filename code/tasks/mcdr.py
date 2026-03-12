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

    # ── state-scoring options ────────────────────────────────────────────────
    # Each entry maps a label to a list of (feature_name, class_idx) pairs;
    # class_idx 0 = Left weight, 1 = Right weight.
    # The score for each state k is the mean of W[k, cls, feat_idx] over the
    # listed pairs.  States are then ranked highest-score → "Engaged".
    _SCORING_OPTIONS: dict = {
        "S_coh":     [("SL", 0), ("SR", 1)],
        "S1_coh":   [("stim1L", 0), ("stim1R", 1)],
        "S2_coh":   [("stim2L", 0), ("stim2R", 1)],
        "S3_coh":   [("stim3L", 0), ("stim3R", 1)],
        "S4_coh":   [("stim4L", 0), ("stim4R", 1)],
        "onset_coh": [("onsetL", 0), ("onsetR", 1)],
        "bias_coh":  [("biasL", 0), ("biasR", 1)],
    }
    scoring_key: str = "S_coh"

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

    # ── column mapping ───────────────────────────────────────────────────────

    @property
    def behavioral_cols(self) -> dict:
        """MCDR column mapping (canonical → actual)."""
        return {
            "trial_idx":   "trial_idx",
            "trial":       "trial",
            "session":     "session",
            "stimulus":    "stimulus",
            "response":    "response",
            "performance": "performance",
        }

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
        """MCDR engagement scoring driven by ``self.scoring_key``.

        The state with the highest mean coherent weight (as defined by the
        selected scoring regressor) is labelled "Engaged"; the rest are
        "Disengaged" (or "Disengaged 1", "Disengaged 2", … for K>2).
        """
        import numpy as np

        pairs = self._SCORING_OPTIONS.get(
            getattr(self, "scoring_key", "S_coh"),
            self._SCORING_OPTIONS["S_coh"],
        )

        def _scoh(W, feat_names):
            name2fi = {n: i for i, n in enumerate(feat_names)}
            scores = np.zeros(W.shape[0])
            n = 0
            for feat, cls in pairs:
                if feat in name2fi:
                    scores += W[:, cls, name2fi[feat]]
                    n += 1
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
            W_np    = np.asarray(W)
            feat    = list(arrays_store[subj].get("X_cols", base_feat))
            scores  = _scoh(W_np, feat)
            ranking = list(np.argsort(scores)[::-1])
            engaged_k = int(ranking[0])
            others    = [int(k) for k in ranking[1:]]
            labels: dict = {engaged_k: "Engaged"}

            if K == 2:
                labels[others[0]] = "Disengaged"
                order = [engaged_k] + others

            elif K == 4:
                name2fi = {n: i for i, n in enumerate(feat)}
                sl_fi   = name2fi.get("SL")
                sr_fi   = name2fi.get("SR")

                # Disengaged L: state most driven by SL (left-choice weight)
                if sl_fi is not None:
                    dis_l = others[int(np.argmax(W_np[others, 0, sl_fi]))]
                else:
                    dis_l = others[0]
                remaining = [k for k in others if k != dis_l]

                # Disengaged R: state most driven by SR (right-choice weight)
                if sr_fi is not None:
                    dis_r = remaining[int(np.argmax(W_np[remaining, 1, sr_fi]))]
                else:
                    dis_r = remaining[0]
                dis_c = [k for k in remaining if k != dis_r][0]

                labels[dis_l] = "Disengaged L"
                labels[dis_r] = "Disengaged R"
                labels[dis_c] = "Disengaged C"
                order = [engaged_k, dis_l, dis_r, dis_c]

            else:
                dis = 1
                for k in others:
                    labels[k] = f"Disengaged {dis}"
                    dis += 1
                order = [engaged_k] + others

            state_labels[subj] = labels
            state_order[subj]  = order
        return state_labels, state_order