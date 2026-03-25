"""Task adapter for the Nuo auditory 2AFC task."""
from __future__ import annotations

import types
from typing import Any, List, Tuple, Dict

import jax.numpy as jnp
import numpy as np
import polars as pl

from tasks import TaskAdapter, _register

_NUO_AUDITORY_EMISSION_COLS: list[str] = [
    "bias",
    "stim_vals",
    "at_choice",
    "at_error",
    "at_correct",
    "reward_trace",
    "prev_choice",
    "prev_reward",
    "cumulative_reward",
    "prev_abs_stim",
]

_NUO_AUDITORY_TRANSITION_COLS: list[str] = [
    "at_choice",
    "at_correct",
    "at_error",
    "reward_trace",
    "prev_abs_stim",
    "prev_reward",
    "cumulative_reward",
]


def _validated_half_life(tau: float) -> float:
    """Return a positive half-life for Polars EWMA features."""
    half_life = float(tau)
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError(f"Nuo auditory requires tau > 0, got {tau!r}")
    return half_life


@_register(["nuo_auditory", "auditory_2afc", "nuo_auditive"])
class NuoAuditoryAdapter(TaskAdapter):
    """Adapter for the binary Nuo auditory task."""

    task_key: str = "nuo_auditory"
    task_label: str = "Nuo auditory"
    num_classes: int = 2
    data_file: str = "auditory_2AFC.parquet"
    sort_col = ["session", "trial"]
    session_col: str = "session"
    psychometric_x_col: str = "total_evidence_strength"

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Drop miss trials and add the canonical binary-task columns.

        The adapter keeps a task-owned feature contract by adding the canonical
        columns it will later use to build its own design matrices.
        """

        side_expr = (
            pl.when(pl.col("correct_side").str.to_lowercase() == "left")
            .then(pl.lit(0))
            .when(pl.col("correct_side").str.to_lowercase() == "right")
            .then(pl.lit(1))
            .otherwise(pl.lit(None))
            .cast(pl.Int64)
        )
        response_expr = (
            pl.when(pl.col("last_choice").str.to_lowercase() == "left")
            .then(pl.lit(0))
            .when(pl.col("last_choice").str.to_lowercase() == "right")
            .then(pl.lit(1))
            .otherwise(pl.lit(None))
            .cast(pl.Int64)
        )
        trial_idx_expr = (
            pl.col("__index_level_0__").cast(pl.Int64)
            if "__index_level_0__" in df.columns
            else pl.int_range(0, pl.len(), eager=False).cast(pl.Int64)
        )

        return (
            df.filter(~pl.col("miss_trial"))
            .with_columns(
                [
                    trial_idx_expr.alias("trial_idx"),
                    side_expr.alias("stimulus"),
                    response_expr.alias("response"),
                    pl.col("correct").cast(pl.Int64).alias("performance"),
                ]
            )
        )

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        """Return the Nuo auditory trial dataframe with derived regressors."""
        df_sub = df_sub.sort(["session", "trial"])
        half_life = _validated_half_life(tau)
        stim_scale = float(df_sub.select(pl.col("total_evidence_strength").abs().max()).item() or 0.0)
        if stim_scale <= 0:
            stim_scale = 1.0

        choice_signed_expr = (
            pl.when(pl.col("response") == 1)
            .then(pl.lit(1.0))
            .when(pl.col("response") == 0)
            .then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float32)
        )

        df_sub = df_sub.with_columns(
            [
                (pl.col("total_evidence_strength").cast(pl.Float32) / pl.lit(stim_scale)).alias("stim_vals"),
                pl.lit(1.0).cast(pl.Float32).alias("bias"),
                choice_signed_expr.alias("_choice_signed"),
            ]
        )
        df_sub = df_sub.with_columns(
            [
                pl.col("_choice_signed").shift(1).fill_null(0.0).over("session").cast(pl.Float32).alias("_prev_choice_signed"),
                (pl.col("_choice_signed") * pl.col("performance")).shift(1).fill_null(0.0).over("session").cast(pl.Float32).alias("_prev_correct_signed"),
                (pl.col("_choice_signed") * (1.0 - pl.col("performance"))).shift(1).fill_null(0.0).over("session").cast(pl.Float32).alias("_prev_error_signed"),
                pl.col("response").shift(1).fill_null(0).over("session").cast(pl.Float32).alias("prev_choice"),
                pl.col("performance").shift(1).fill_null(0.0).over("session").cast(pl.Float32).alias("prev_reward"),
                pl.col("stim_vals").abs().shift(1).fill_null(0.0).over("session").cast(pl.Float32).alias("prev_abs_stim"),
                pl.col("performance").shift(1).fill_null(0.0).cum_sum().over("session").cast(pl.Float32).alias("_cumulative_reward_raw"),
            ]
        )
        df_sub = df_sub.with_columns(
            [
                pl.when(pl.col("_cumulative_reward_raw").max().over("session") > 0)
                .then(pl.col("_cumulative_reward_raw") / pl.col("_cumulative_reward_raw").max().over("session"))
                .otherwise(pl.lit(0.0))
                .cast(pl.Float32)
                .alias("cumulative_reward"),
                pl.col("_prev_choice_signed").ewm_mean(half_life=half_life, adjust=False).over("session").cast(pl.Float32).alias("at_choice"),
                pl.col("_prev_correct_signed").ewm_mean(half_life=half_life, adjust=False).over("session").cast(pl.Float32).alias("at_correct"),
                pl.col("_prev_error_signed").ewm_mean(half_life=half_life, adjust=False).over("session").cast(pl.Float32).alias("at_error"),
                pl.col("prev_reward").ewm_mean(half_life=half_life, adjust=False).over("session").cast(pl.Float32).alias("reward_trace"),
            ]
        )
        return df_sub

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` for one subject."""
        feature_df = self.build_feature_df(df_sub, tau=tau)
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
        """Return ``(y, X, U, names)`` for one subject."""
        ecols = emission_cols if emission_cols is not None else self.default_emission_cols()
        ucols = transition_cols if transition_cols is not None else self.default_transition_cols()
        bad_e = [c for c in ecols if c not in _NUO_AUDITORY_EMISSION_COLS]
        bad_u = [c for c in ucols if c not in _NUO_AUDITORY_TRANSITION_COLS]
        if bad_e:
            raise ValueError(
                f"Unknown emission_cols: {bad_e}. Available: {_NUO_AUDITORY_EMISSION_COLS}"
            )
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {_NUO_AUDITORY_TRANSITION_COLS}"
            )

        y = jnp.asarray(feature_df["response"].to_numpy().astype(np.int32))
        X = jnp.asarray(feature_df.select(ecols).to_numpy().astype(np.float32)) if ecols else jnp.empty((len(y), 0), dtype=jnp.float32)
        U = jnp.asarray(feature_df.select(ucols).to_numpy().astype(np.float32)) if ucols else jnp.empty((len(y), 0), dtype=jnp.float32)
        names = {"X_cols": list(ecols), "U_cols": list(ucols)}
        return y, X, U, names

    def default_emission_cols(self) -> List[str]:
        return list(_NUO_AUDITORY_EMISSION_COLS)

    def default_transition_cols(self) -> List[str]:
        return list(_NUO_AUDITORY_TRANSITION_COLS)

    def available_emission_cols(self) -> List[str]:
        return list(_NUO_AUDITORY_EMISSION_COLS)

    def available_transition_cols(self) -> List[str]:
        return list(_NUO_AUDITORY_TRANSITION_COLS)

    def cv_balance_labels(self, feature_df: pl.DataFrame):
        """Return signed evidence labels for balanced session-holdout CV."""
        if self.psychometric_x_col not in feature_df.columns:
            return None
        return feature_df[self.psychometric_x_col].cast(pl.Float64)

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

    @property
    def behavioral_cols(self) -> dict:
        return {
            "trial_idx": "trial_idx",
            "trial": "trial",
            "session": "session",
            "stimulus": "stimulus",
            "response": "response",
            "performance": "performance",
        }

    def get_plots(self) -> types.ModuleType:
        import tasks.plots.nuo_auditory as plots

        return plots

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        """Binary-task state labels using the task's native stimulus sign."""
        base_feat = list(names.get("X_cols", []))
        state_labels: dict = {}
        state_order: dict = {}

        for subj in subjects:
            W = arrays_store[subj].get("emission_weights") if subj in arrays_store else None
            if W is None:
                state_labels[subj] = {k: f"State {k+1}" for k in range(K)}
                state_order[subj] = list(range(K))
                continue

            feat = list(arrays_store[subj].get("X_cols", base_feat))
            W = np.asarray(W)
            name2fi = {n: i for i, n in enumerate(feat)}

            stim_fi = name2fi.get("stim_vals")
            if stim_fi is not None:
                stim_scores = W[:, 0, stim_fi]
            else:
                stim_scores = W[:, 0, :].mean(axis=1)

            engaged_k = int(np.argmax(stim_scores))
            others = [k for k in range(K) if k != engaged_k]
            labels: dict = {engaged_k: "Engaged"}

            if K == 2:
                labels[others[0]] = "Disengaged"
                order = [engaged_k, others[0]]
            elif K == 3:
                bias_fi = name2fi.get("bias")
                if bias_fi is not None:
                    bias_disp = W[others, 0, bias_fi]
                    biased_l = others[int(np.argmin(bias_disp))]
                    biased_r = others[int(np.argmax(bias_disp))]
                else:
                    biased_l, biased_r = others[0], others[1]
                labels[biased_l] = "Biased L"
                labels[biased_r] = "Biased R"
                order = [engaged_k, biased_l, biased_r]
            else:
                others_sorted = sorted(others, key=lambda k: stim_scores[k], reverse=True)
                for dis, k in enumerate(others_sorted, start=1):
                    labels[k] = f"Disengaged {dis}"
                order = [engaged_k] + others_sorted

            state_labels[subj] = labels
            state_order[subj] = order

        return state_labels, state_order
