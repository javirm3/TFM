"""Task adapter for binary accuracy modelling on the MCDR task."""
from __future__ import annotations

import types
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import polars as pl

from glmhmmt.tasks import TaskAdapter, _register

EMISSION_COLS: list[str] = [
    "bias",
    "onset",
    "delay",
    "stimulus_duration_abs",
    "stimulus_duration_cat",
    "stimulus_duration_abs_x_delay",
    "stimulus_duration_cat_x_delay",
    "stim1",
    "stim2",
    "stim3",
    "stim4",
    "speed1",
    "speed2",
    "speed3",
    "A_plus",
    "A_minus",
]

TRANSITION_COLS: list[str] = ["A_plus", "A_minus"]


@_register(["mcdr_accuracy", "mcdr-accuracy"])
class MCDRAccuracyAdapter(TaskAdapter):
    """Adapter for binary error/correct modelling on MCDR."""

    task_key: str = "MCDR-Accuracy"
    task_label: str = "MCDR Accuracy"
    num_classes: int = 2
    data_file: str = "df_filtered.parquet"
    sort_col: str = "trial_idx"
    session_col: str = "session"

    # Binary convention: stored weights are for the non-reference class.
    # Here class 0 = Correct, class 1 = Error (reference).
    # Positive raw weights increase P(correct).
    _SCORING_OPTIONS: dict = {
        "stimulus_duration_cat (+w)": [("stimulus_duration_cat", "pos")],
        "stimulus_duration_cat (|w|)": [("stimulus_duration_cat", "abs")],
        "stimulus_x_delay (|w|)": [("stimulus_x_delay", "abs")],
        "delay (|w|)": [("delay", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "stimulus_duration_cat (+w)"

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("subject") != "A84")

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        df_sub = df_sub.sort(self.sort_col)
        df_sub = df_sub.with_columns(
            [((pl.col("stimd_n") - pl.col("stimd_n").mean()) / pl.col("stimd_n").std()).alias("stimd_n_z")]
        )
        df_sub = df_sub.with_columns(
            [
                pl.col("response").cast(pl.Int32),
                pl.col("stimd_n_z").cast(pl.Int32).alias("stimulus_duration_abs"),
                pl.col("performance").cast(pl.Boolean).alias("correct_bool"),
                pl.lit(1.0).cast(pl.Float32).alias("bias"),
                pl.col("onset").cast(pl.Float32).alias("onset"),
                pl.col("delay_d").cast(pl.Float32).alias("delay"),
                pl.col("ttype_n").cast(pl.Float32).alias("stimulus_duration_cat"),
                (pl.col("ttype_n") * pl.col("delay_d")).cast(pl.Float32).alias("stimulus_duration_cat_x_delay"),
                (pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("stimulus_duration_abs_x_delay"),
                (
                    ((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0)) | (pl.col("offset") == 0)
                ).cast(pl.Float32).alias("stim1"),
                (
                    ((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1"))) | (pl.col("offset") == 0)
                ).cast(pl.Float32).alias("stim2"),
                (
                    ((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2"))) | (pl.col("offset") == 0)
                ).cast(pl.Float32).alias("stim3"),
                ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3"))).cast(pl.Float32).alias("stim4"),
                pl.col("performance").shift(1).fill_null(0).cast(pl.Float32).over(self.session_col).alias("previous_outcome"),
                (1 / (pl.col("timepoint_3") - pl.col("timepoint_4"))).cast(pl.Float32).alias("speed3"),
                (1 / (pl.col("timepoint_3") - pl.col("timepoint_2"))).cast(pl.Float32).alias("speed2"),
                (1 / (pl.col("timepoint_2") - pl.col("timepoint_1"))).cast(pl.Float32).alias("speed1"),
            ]
        )
        df_sub = df_sub.with_columns(
            [
                pl.col("previous_outcome").ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_plus"),
                (1.0 - pl.col("previous_outcome")).ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_minus"),
            ]
        )
        return (
            df_sub.with_columns(
                [((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).cast(pl.Float32).alias(c) for c in ["speed1", "speed2", "speed3"]]
            )
        )

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        feature_df = self.build_feature_df(df_sub, tau=tau)
        return self.build_design_matrices(feature_df, emission_cols=emission_cols, transition_cols=transition_cols)

    def build_design_matrices(
        self,
        feature_df,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        ecols = emission_cols if emission_cols is not None else list(EMISSION_COLS)
        ucols = transition_cols if transition_cols is not None else list(TRANSITION_COLS)
        bad_e = [c for c in ecols if c not in EMISSION_COLS]
        bad_u = [c for c in ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {EMISSION_COLS}")
        if bad_u:
            raise ValueError(f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}")

        # Encode trials so the explicit softmax row corresponds to "Correct":
        # y = 0 for correct, y = 1 for error/reference.
        y = jnp.asarray((1 - feature_df["performance"].to_numpy().astype(np.int32)))
        X = jnp.asarray(feature_df.select(ecols).to_numpy().astype(np.float32)) if ecols else jnp.empty((len(y), 0), dtype=jnp.float32)
        U = jnp.asarray(feature_df.select(ucols).to_numpy().astype(np.float32)) if ucols else jnp.empty((len(y), 0), dtype=jnp.float32)
        names = {"X_cols": list(ecols), "U_cols": list(ucols)}
        return y, X, U, names

    def default_emission_cols(self, df=None) -> List[str]:
        del df
        return list(EMISSION_COLS)

    def default_transition_cols(self) -> List[str]:
        return list(TRANSITION_COLS)

    def resolve_design_names(
        self,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
        df=None,
    ) -> Dict[str, List[str]]:
        ecols = list(emission_cols) if emission_cols is not None else list(EMISSION_COLS)
        ucols = list(transition_cols) if transition_cols is not None else list(TRANSITION_COLS)
        bad_e = [c for c in ecols if c not in EMISSION_COLS]
        bad_u = [c for c in ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {EMISSION_COLS}")
        if bad_u:
            raise ValueError(f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}")
        return {"X_cols": ecols, "U_cols": ucols}

    @property
    def choice_labels(self) -> list[str]:
        return ["Correct", "Error"]

    @property
    def probability_columns(self) -> list[str]:
        return ["p_correct", "p_error"]

    def get_correct_class(self, df: pl.DataFrame) -> np.ndarray:
        return np.zeros(df.height, dtype=int)

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
        from .plots import mcdr_accuracy as plots
        return plots

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        pairs = self._SCORING_OPTIONS.get(
            getattr(self, "scoring_key", "stimulus_duration_cat (+w)"),
            self._SCORING_OPTIONS["stimulus_duration_cat (+w)"],
        )

        def _score_states(weights: np.ndarray, feat_names: list[str]) -> np.ndarray:
            name2fi = {n: i for i, n in enumerate(feat_names)}
            scores = np.zeros(weights.shape[0], dtype=float)
            n_terms = 0
            for feat_name, mode in pairs:
                fi = name2fi.get(feat_name)
                if fi is None:
                    continue
                vals = weights[:, 0, fi].astype(float)
                if mode == "neg":
                    vals = -vals
                elif mode == "abs":
                    vals = np.abs(vals)
                elif mode == "pos":
                    vals = vals
                else:
                    raise ValueError(f"Unknown MCDR-Accuracy scoring mode {mode!r}.")
                scores += vals
                n_terms += 1
            if n_terms > 0:
                return scores / n_terms
            return weights[:, 0, :].mean(axis=1)

        base_feat = list(names.get("X_cols", []))
        state_labels: dict = {}
        state_order: dict = {}
        for subj in subjects:
            weights = arrays_store[subj].get("emission_weights") if subj in arrays_store else None
            if weights is None:
                state_labels[subj] = {k: f"State {k + 1}" for k in range(K)}
                state_order[subj] = list(range(K))
                continue

            feat_names = list(arrays_store[subj].get("X_cols", base_feat))
            weights_np = np.asarray(weights)
            state_scores = _score_states(weights_np, feat_names)
            engaged_k = int(np.argmax(state_scores))
            others = [k for k in range(K) if k != engaged_k]

            labels: dict[int, str] = {engaged_k: "Engaged"}
            if K == 2:
                labels[others[0]] = "Disengaged"
                order = [engaged_k, others[0]]
            else:
                others_sorted = sorted(others, key=lambda k: state_scores[k], reverse=True)
                for idx, k in enumerate(others_sorted, start=1):
                    labels[k] = f"Disengaged {idx}"
                order = [engaged_k] + others_sorted

            state_labels[subj] = labels
            state_order[subj] = order
        return state_labels, state_order
