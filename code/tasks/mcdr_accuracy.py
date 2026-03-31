"""Task adapter for binary accuracy modelling on the MCDR task."""
from __future__ import annotations

import types
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import polars as pl

from glmhmmt.tasks import TaskAdapter, _register

_ALL_EMISSION_COLS: list[str] = [
    "bias",
    "biasL", "biasC", "biasR", "onsetL", "onsetC", "onsetR", "delay",
    "SL", "SC", "SR",
    "SLxdelay", "SCxdelay", "SRxdelay",
    "SLxD", "SCxD", "SRxD",
    "D", "DL", "DC", "DR",
    "A_L", "A_C", "A_R",
    "speed1", "speed2", "speed3",
    "stim1L", "stim1C", "stim1R",
    "stim2L", "stim2C", "stim2R",
    "stim3L", "stim3C", "stim3R",
    "stim4L", "stim4C", "stim4R",
]

_ALL_TRANSITION_COLS: list[str] = ["A_plus", "A_minus", "A_L", "A_C", "A_R"]


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
    # Here class 0 = Error, class 1 = Correct (reference).
    # Negative raw weights increase P(correct).
    _SCORING_OPTIONS: dict = {
        "S_coh (-w)": [("SL", "neg"), ("SR", "neg")],
        "S1_coh (-w)": [("stim1L", "neg"), ("stim1R", "neg")],
        "S2_coh (-w)": [("stim2L", "neg"), ("stim2R", "neg")],
        "S3_coh (-w)": [("stim3L", "neg"), ("stim3R", "neg")],
        "S4_coh (-w)": [("stim4L", "neg"), ("stim4R", "neg")],
        "onset_coh (-w)": [("onsetL", "neg"), ("onsetR", "neg")],
        "bias_coh (|w|)": [("biasL", "abs"), ("biasR", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "S_coh (-w)"

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("subject") != "A84")

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        df_sub = df_sub.sort(self.sort_col)
        df_sub = df_sub.with_columns(
            [((pl.col("stimd_n") - pl.col("stimd_n").mean()) / pl.col("stimd_n").std()).alias("stimd_n_z")]
        )
        df_sub = df_sub.with_columns(
            [
                pl.col("response").cast(pl.Int32).alias("choice_response"),
                pl.col("stimulus").cast(pl.Int32).alias("choice_stimulus"),
                pl.col("performance").cast(pl.Int32).alias("accuracy_response"),
                pl.lit(1).cast(pl.Int32).alias("accuracy_stimulus"),
                pl.col("performance").cast(pl.Boolean).alias("correct_bool"),
                (pl.col("x_c") == "L").cast(pl.Float32).alias("biasL"),
                (pl.col("x_c") == "C").cast(pl.Float32).alias("biasC"),
                (pl.col("x_c") == "R").cast(pl.Float32).alias("biasR"),
                pl.lit(1.0).cast(pl.Float32).alias("bias"),
                pl.col("delay_d").cast(pl.Float32).alias("delay"),
                ((pl.col("x_c") == "L") * pl.col("onset")).cast(pl.Float32).alias("onsetL"),
                ((pl.col("x_c") == "C") * pl.col("onset")).cast(pl.Float32).alias("onsetC"),
                ((pl.col("x_c") == "R") * pl.col("onset")).cast(pl.Float32).alias("onsetR"),
                ((pl.col("x_c") == "L") * pl.col("stimd_n_z")).cast(pl.Float32).alias("SL"),
                ((pl.col("x_c") == "C") * pl.col("stimd_n_z")).cast(pl.Float32).alias("SC"),
                ((pl.col("x_c") == "R") * pl.col("stimd_n_z")).cast(pl.Float32).alias("SR"),
                ((pl.col("x_c") == "L") * pl.col("delay_d")).cast(pl.Float32).alias("DL"),
                ((pl.col("x_c") == "C") * pl.col("delay_d")).cast(pl.Float32).alias("DC"),
                ((pl.col("x_c") == "R") * pl.col("delay_d")).cast(pl.Float32).alias("DR"),
                pl.col("ttype_n").cast(pl.Float32).alias("D"),
                ((pl.col("x_c") == "L") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SLxD"),
                ((pl.col("x_c") == "C") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SCxD"),
                ((pl.col("x_c") == "R") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SRxD"),
                ((pl.col("x_c") == "L") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SLxdelay"),
                ((pl.col("x_c") == "C") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SCxdelay"),
                ((pl.col("x_c") == "R") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SRxdelay"),
                (
                    (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0)) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "L")
                ).cast(pl.Float32).alias("stim1L"),
                (
                    (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0)) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "C")
                ).cast(pl.Float32).alias("stim1C"),
                (
                    (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0)) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "R")
                ).cast(pl.Float32).alias("stim1R"),
                (
                    (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "L")
                ).cast(pl.Float32).alias("stim2L"),
                (
                    (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "C")
                ).cast(pl.Float32).alias("stim2C"),
                (
                    (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "R")
                ).cast(pl.Float32).alias("stim2R"),
                (
                    (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "L")
                ).cast(pl.Float32).alias("stim3L"),
                (
                    (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "C")
                ).cast(pl.Float32).alias("stim3C"),
                (
                    (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2"))) | (pl.col("offset") == 0))
                    & (pl.col("x_c") == "R")
                ).cast(pl.Float32).alias("stim3R"),
                ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "L")).cast(pl.Float32).alias("stim4L"),
                ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "C")).cast(pl.Float32).alias("stim4C"),
                ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "R")).cast(pl.Float32).alias("stim4R"),
                pl.col("performance").shift(1).fill_null(0).cast(pl.Float32).over(self.session_col).alias("previous_outcome"),
                pl.col("response").shift(1).fill_null(0.0).eq(0).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_L"),
                pl.col("response").shift(1).fill_null(0.0).eq(1).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_C"),
                pl.col("response").shift(1).fill_null(0.0).eq(2).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).over(self.session_col).alias("A_R"),
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
            .with_columns(
                [
                    pl.col("accuracy_response").alias("response"),
                    pl.col("accuracy_stimulus").alias("stimulus"),
                ]
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
        ecols = emission_cols if emission_cols is not None else list(_ALL_EMISSION_COLS)
        ucols = transition_cols if transition_cols is not None else list(_ALL_TRANSITION_COLS)
        bad_e = [c for c in ecols if c not in _ALL_EMISSION_COLS]
        bad_u = [c for c in ucols if c not in _ALL_TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {_ALL_EMISSION_COLS}")
        if bad_u:
            raise ValueError(f"Unknown transition_cols: {bad_u}. Available: {_ALL_TRANSITION_COLS}")

        y = jnp.asarray(feature_df["accuracy_response"].to_numpy().astype(np.int32))
        X = jnp.asarray(feature_df.select(ecols).to_numpy().astype(np.float32)) if ecols else jnp.empty((len(y), 0), dtype=jnp.float32)
        U = jnp.asarray(feature_df.select(ucols).to_numpy().astype(np.float32)) if ucols else jnp.empty((len(y), 0), dtype=jnp.float32)
        names = {"X_cols": list(ecols), "U_cols": list(ucols)}
        return y, X, U, names

    def default_emission_cols(self) -> List[str]:
        return list(_ALL_EMISSION_COLS)

    def default_transition_cols(self) -> List[str]:
        return list(_ALL_TRANSITION_COLS)

    def resolve_design_names(
        self,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
        df=None,
    ) -> Dict[str, List[str]]:
        ecols = list(emission_cols) if emission_cols is not None else list(_ALL_EMISSION_COLS)
        ucols = list(transition_cols) if transition_cols is not None else list(_ALL_TRANSITION_COLS)
        bad_e = [c for c in ecols if c not in _ALL_EMISSION_COLS]
        bad_u = [c for c in ucols if c not in _ALL_TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {_ALL_EMISSION_COLS}")
        if bad_u:
            raise ValueError(f"Unknown transition_cols: {bad_u}. Available: {_ALL_TRANSITION_COLS}")
        return {"X_cols": ecols, "U_cols": ucols}

    @property
    def choice_labels(self) -> list[str]:
        return ["Error", "Correct"]

    @property
    def probability_columns(self) -> list[str]:
        return ["p_error", "p_correct"]

    def get_correct_class(self, df: pl.DataFrame) -> np.ndarray:
        if "stimulus" in df.columns:
            vals = df["stimulus"].to_numpy().astype(int)
            if np.all(vals == 1):
                return vals
        return np.ones(df.height, dtype=int)

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
        import tasks.plots.mcdr_accuracy as plots
        return plots

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        pairs = self._SCORING_OPTIONS.get(
            getattr(self, "scoring_key", "S_coh (-w)"),
            self._SCORING_OPTIONS["S_coh (-w)"],
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
            return -weights[:, 0, :].mean(axis=1)

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
