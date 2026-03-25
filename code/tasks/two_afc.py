"""Task adapter for the 2AFC (Alexis human) task."""
from __future__ import annotations

import types
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import jax.numpy as jnp
import polars as pl

from tasks import TaskAdapter, _register

# Default experiments to keep (avoids habituation / drug sessions)
_KEEP_EXPERIMENTS = ["2AFC_2", "2AFC_3", "2AFC_4", "2AFC_6"]
_SF_COL_PREFIX = "sf_"
_STIM_ABS_COL_PREFIX = "stim_"
_ALL_2AFC_EMISSION_COLS: list[str] = [
    "bias",
    "stim_vals",
    "stim_strength",
    "at_choice",
    "at_error",
    "at_correct",
    "reward_trace",
    "prev_choice",
    "wsls",
    "prev_reward",
    "cumulative_reward",
    "prev_abs_stim",
]
_AVAILABLE_2AFC_EMISSION_COLS: list[str] = list(_ALL_2AFC_EMISSION_COLS)
_ALL_2AFC_TRANSITION_COLS: list[str] = [
    "at_choice",
    "at_correct",
    "at_error",
    "reward_trace",
    "prev_abs_stim",
    "prev_reward",
    "cumulative_reward",
]
_AVAILABLE_2AFC_TRANSITION_COLS: list[str] = list(_ALL_2AFC_TRANSITION_COLS)


def _sf_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_SF_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _stim_abs_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix(_STIM_ABS_COL_PREFIX)
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _stim_abs_cols(columns: list[str]) -> list[str]:
    return sorted(
        [
            col
            for col in columns
            if col.startswith(_STIM_ABS_COL_PREFIX)
            and col.removeprefix(_STIM_ABS_COL_PREFIX).isdigit()
        ],
        key=_stim_abs_sort_key,
    )


def _infer_stim_abs_cols_from_df(df: pl.DataFrame | pd.DataFrame) -> list[str]:
    columns = list(df.columns)
    existing = _stim_abs_cols(columns)
    if existing:
        return existing
    if "ILD" not in columns:
        return []
    ild_series = df["ILD"].drop_nulls() if isinstance(df, pl.DataFrame) else df["ILD"].dropna()
    stim_abs_levels = sorted({int(abs(v)) for v in ild_series.to_list()})
    return [f"{_STIM_ABS_COL_PREFIX}{stim_abs}" for stim_abs in stim_abs_levels]


@_register(["two_afc", "2afc"])
class TwoAFCAdapter(TaskAdapter):
    """Adapter for the binary 2-AFC human data (Alexis)."""

    task_key: str    = "2AFC"
    task_label: str  = "2AFC"
    num_classes: int = 2
    data_file: str   = "alexis_combined.parquet"
    # Session-local trial numbers must be sorted within session to match the
    # per-session concatenation order used during fitting.
    sort_col         = ["Session", "Trial"]
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

    def _build_feature_df(
        self,
        df_sub: pl.DataFrame,
        tau: float = 50.0,
        include_stim_strength: bool = False,
    ) -> pl.DataFrame:
        """Return the Alexis 2AFC feature dataframe owned by this adapter."""
        from scripts.alexis_functions import get_action_trace, make_frames_dm

        df_pd = df_sub.to_pandas() if hasattr(df_sub, "to_pandas") else df_sub.copy()
        df_pd = df_pd.sort_values(["Session", "Trial"]).reset_index(drop=True)
        if df_pd.empty:
            return pl.from_pandas(df_pd)

        stim_scale = float(df_pd["ILD"].abs().max() or 0.0)
        if stim_scale <= 0:
            stim_scale = 1.0

        stim_set = 6 if df_pd["Experiment"].iloc[0] == "2AFC_6" else 2
        stim_abs_levels = sorted(
            {
                int(abs(v))
                for v in df_pd["ILD"].dropna().astype(int).tolist()
            }
        )
        parts = []
        for _, df_session in df_pd.groupby("Session", sort=False):
            part = df_session.copy().reset_index(drop=True)
            part["bias"] = 1.0
            part["stim_vals"] = part["ILD"].astype(float) / stim_scale
            for stim_abs in stim_abs_levels:
                if stim_abs == 0:
                    stim_col = np.where(part["ILD"] == 0, 1.0, 0.0).astype(np.float32)
                else:
                    stim_col = np.select(
                        [part["ILD"] == stim_abs, part["ILD"] == -stim_abs],
                        [1.0, -1.0],
                        default=0.0,
                    ).astype(np.float32)
                part[f"{_STIM_ABS_COL_PREFIX}{stim_abs}"] = stim_col

            existing_sf_cols = [
                c for c in part.columns if str(c).startswith(_SF_COL_PREFIX)
            ]
            if include_stim_strength and not existing_sf_cols and "Filename" in part.columns:
                stim_strength, _ = make_frames_dm(part, stim_set=stim_set, residuals=True, zscore=False)
                stim_strength = stim_strength.reset_index(drop=True)
                max_val = float(np.nanmax(np.abs(stim_strength.to_numpy()))) if not stim_strength.empty else 0.0
                if max_val > 0:
                    stim_strength = stim_strength / max_val
                stim_strength.columns = [f"{_SF_COL_PREFIX}{col}" for col in stim_strength.columns]
                part = pd.concat([part.reset_index(drop=True), stim_strength], axis=1)

            at_choice, at_error, at_correct, reward_trace = get_action_trace(part)
            part["at_choice"] = np.asarray(at_choice, dtype=np.float32)
            part["at_error"] = np.asarray(at_error, dtype=np.float32)
            part["at_correct"] = np.asarray(at_correct, dtype=np.float32)
            part["reward_trace"] = np.asarray(reward_trace, dtype=np.float32)
            part["prev_choice"] = part["Choice"].shift(1).fillna(0).astype(np.float32)
            part["prev_reward"] = part["Hit"].shift(1).fillna(0).astype(np.float32)

            cumulative_reward = part["Hit"].cumsum().shift(1).fillna(0).astype(float)
            max_cumulative_reward = float(np.nanmax(cumulative_reward.to_numpy())) if len(cumulative_reward) else 0.0
            if max_cumulative_reward > 0:
                cumulative_reward = cumulative_reward / max_cumulative_reward
            part["cumulative_reward"] = cumulative_reward.astype(np.float32)
            part["prev_abs_stim"] = (part["ILD"].abs().shift(1).fillna(0) / stim_scale).astype(np.float32)
            part["wsls"] = part["Side"].shift(1).fillna(0).replace({0: -1, 1: 1}).astype(np.float32)
            parts.append(part)

        return pl.from_pandas(pd.concat(parts, ignore_index=True))

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        """Return the default 2AFC feature dataframe without frame regressors."""
        return self._build_feature_df(df_sub, tau=tau, include_stim_strength=False)

    def _resolved_emission_cols(
        self,
        feature_df: pl.DataFrame,
        emission_cols: List[str] | None,
    ) -> list[str]:
        requested = emission_cols if emission_cols is not None else self.default_emission_cols()
        resolved: list[str] = []
        dynamic_sf_cols = sorted(
            [c for c in feature_df.columns if c.startswith(_SF_COL_PREFIX)],
            key=_sf_sort_key,
        )
        for col in requested:
            if col == "stim_strength":
                if not dynamic_sf_cols:
                    raise ValueError(
                        "Requested emission col 'stim_strength', but no frame-level "
                        f"'{_SF_COL_PREFIX}*' columns are available for 2AFC."
                    )
                resolved.extend(dynamic_sf_cols)
            else:
                resolved.append(col)
        return resolved

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` for the 2AFC task."""
        requested_emission_cols = emission_cols if emission_cols is not None else self.default_emission_cols()
        include_stim_strength = "stim_strength" in requested_emission_cols
        feature_df = self._build_feature_df(
            df_sub,
            tau=tau,
            include_stim_strength=include_stim_strength,
        )
        return self.build_design_matrices(
            feature_df,
            emission_cols=emission_cols,
            transition_cols=transition_cols,
        )

    def build_design_matrices(
        self,
        feature_df,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` for the 2AFC task."""
        ecols = self._resolved_emission_cols(feature_df, emission_cols)
        ucols = transition_cols if transition_cols is not None else self.default_transition_cols()
        allowed_ecols = set(self.available_emission_cols()) | {
            c for c in feature_df.columns if c.startswith(_SF_COL_PREFIX)
        } | set(_stim_abs_cols(feature_df.columns))
        bad_e = [c for c in ecols if c not in allowed_ecols]
        bad_u = [c for c in ucols if c not in _AVAILABLE_2AFC_TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {_AVAILABLE_2AFC_TRANSITION_COLS}"
            )

        y = jnp.asarray(feature_df["Choice"].to_numpy().astype(np.int32))
        X = jnp.asarray(feature_df.select(ecols).to_numpy().astype(np.float32)) if ecols else jnp.empty((len(y), 0), dtype=jnp.float32)
        U = jnp.asarray(feature_df.select(ucols).to_numpy().astype(np.float32)) if ucols else jnp.empty((len(y), 0), dtype=jnp.float32)
        names = {
            "X_cols": list(ecols),
            "U_cols": list(ucols),
        }
        return y, X, U, names

    def cv_balance_labels(self, feature_df: pl.DataFrame):
        """Return signed-ILD balance labels for CV splits."""
        if "ILD" not in feature_df.columns:
            return None
        return feature_df["ILD"].cast(pl.Float64)

    # ── column defaults ─────────────────────────────────────────────────────

    def default_emission_cols(self) -> List[str]:
        # Exclude stim_strength (multi-column) by default; include sf_ cols at runtime
        return [c for c in _ALL_2AFC_EMISSION_COLS if c != "stim_strength"]

    def default_transition_cols(self) -> List[str]:
        return list(_ALL_2AFC_TRANSITION_COLS)

    def available_emission_cols(self) -> List[str]:
        return list(_AVAILABLE_2AFC_EMISSION_COLS)

    def available_transition_cols(self) -> List[str]:
        return list(_AVAILABLE_2AFC_TRANSITION_COLS)

    def sf_cols(self, df: pl.DataFrame) -> List[str]:
        """Return any stimulus-frame (sf_*) columns present in *df*."""
        return [c for c in df.columns if c.startswith(_SF_COL_PREFIX)]

    def stim_abs_cols(self, df: pl.DataFrame) -> List[str]:
        """Return signed one-hot columns for absolute ILD magnitudes."""
        return _infer_stim_abs_cols_from_df(df)

    def available_extra_emission_cols(self, df: pl.DataFrame) -> List[str]:
        return list(dict.fromkeys(self.sf_cols(df) + self.stim_abs_cols(df)))

    def default_extra_emission_cols(self, df: pl.DataFrame) -> List[str]:
        return self.sf_cols(df)

    @property
    def choice_labels(self) -> list[str]:
        return ["Left", "Right"]

    @property
    def probability_columns(self) -> list[str]:
        return ["pL", "pR"]

    def get_correct_class(self, df: pl.DataFrame) -> np.ndarray:
        stim = df["stimulus"].to_numpy().astype(float)
        unique = set(np.unique(stim[~np.isnan(stim)]).tolist())
        if unique.issubset({0.0, 1.0}):
            return stim.astype(int)
        if unique.issubset({-1.0, 1.0}):
            return np.where(stim > 0, 1, 0).astype(int)
        return np.where(stim > 0, 1, np.where(stim < 0, 0, -1)).astype(int)

    # ── column mapping ───────────────────────────────────────────────────────

    @property
    def behavioral_cols(self) -> dict:
        """2AFC column mapping (canonical → actual)."""
        return {
            "trial_idx":   "Trial",
            "trial":       "Trial",
            "session":     "Session",
            "stimulus":    "Side",
            "response":    "Choice",
            "performance": "Hit",
        }

    # ── plots ────────────────────────────────────────────────────────────────

    def get_plots(self) -> types.ModuleType:
        import tasks.plots.two_afc as plots
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
