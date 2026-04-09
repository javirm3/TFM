"""Task adapter for the Tiffany 2AFC delay task."""
from __future__ import annotations

import types
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import polars as pl
import pandas as pd

from glmhmmt.cli.alexis_functions import get_action_trace
from glmhmmt.tasks import TaskAdapter, _register

EMISSION_COLS: list[str] = [
    "bias",
    "stim",
    "delay",
    "stim_x_delay",
    "at_choice",
    "at_error",
    "at_correct",
    "reward_trace",
    "prev_choice",
    "wsls",
    "prev_reward",
    "cumulative_reward",
    "prev_abs_stim",
    "after_correct",
    "repeat",
    "repeat_choice_side",
]
TRANSITION_COLS: list[str] = [
    "at_choice",
    "at_correct",
    "at_error",
    "reward_trace",
    "prev_abs_stim",
    "prev_reward",
    "cumulative_reward",
    "delay",
]

EMISSION_REGRESSOR_LABELS: dict[str, str] = {
    "stim": r"$\mathrm{Stimulus}$",
    "delay": r"$\mathrm{Delay}$",
    "stim_x_delay": r"$\mathrm{Stimulus}\times\mathrm{Delay}$",
    "bias": r"$\mid\mathrm{bias}\mid$",
    "at_choice": r"$\mathrm{A}_t^{\mathrm{choice}}$",
    "at_error": r"$\mathrm{A}_t^{\mathrm{error}}$",
    "at_correct": r"$\mathrm{A}_t^{\mathrm{correct}}$",
    "reward_trace": r"$\mathrm{Reward}_{\mathrm{trace}}$",
    "prev_choice": r"$\mathrm{PrevChoice}$",
    "prev_reward": r"$\mathrm{PrevReward}$",
    "prev_abs_stim": r"$|\mathrm{PrevStim}|$",
    "cumulative_reward": r"$\mathrm{CumReward}$",
    "wsls": r"$\mathrm{WSLS}$",
    "after_correct": r"$\mathrm{AfterCorrect}$",
    "repeat": r"$\mathrm{Repeat}$",
    "repeat_choice_side": r"$\mathrm{RepeatSide}$",
    "WM": r"$\mathrm{WM}$",
    "RL": r"$\mathrm{RL}$",
}


def _stim_param_weight_map() -> dict[int, float]:
    """Compatibility shim for copied 2AFC plotting code."""
    return {0: 0.0, 1: 1.0}


def _choice_to_binary(series: pd.Series) -> np.ndarray:
    return series.astype(np.int32).to_numpy()


def _signed_to_binary(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    unique = set(vals.dropna().unique().tolist())
    if unique.issubset({0, 1, 0.0, 1.0}):
        return vals.astype(np.float32)
    if unique.issubset({-1, 1, -1.0, 1.0}):
        return (vals > 0).astype(np.float32)
    return vals.astype(np.float32)


@_register(["two_afc_delay", "2afc_delay", "2AFC_delay"])
class TwoAFCDelayAdapter(TaskAdapter):
    """Adapter for the Tiffany binary 2AFC task with trial delay."""

    task_key: str = "2AFC_delay"
    task_label: str = "2AFC delay"
    num_classes: int = 2
    data_file: str = "tiffany.parquet"
    sort_col = ["session", "trial"]
    session_col: str = "session"

    _SCORING_OPTIONS: dict = {
        "stim (w)": [("stim", "pos")],
        "stim (|w|)": [("stim", "abs")],
        "delay (|w|)": [("delay", "abs")],
        "stim_x_delay (|w|)": [("stim_x_delay", "abs")],
        "at_choice (|w|)": [("at_choice", "abs")],
        "wsls (|w|)": [("wsls", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "stim (w)"

    def subject_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def _build_feature_df(
        self,
        df_sub: pl.DataFrame,
        tau: float = 50.0,
    ) -> pl.DataFrame:
        del tau

        df_pd = df_sub.to_pandas() if hasattr(df_sub, "to_pandas") else df_sub.copy()
        df_pd = df_pd.sort_values(["session", "trial"]).reset_index(drop=True)
        if df_pd.empty:
            return pl.from_pandas(df_pd)

        parts: list[pd.DataFrame] = []
        for _, df_session in df_pd.groupby("session", sort=False):
            part = df_session.copy().reset_index(drop=True)
            part["bias"] = 1.0
            part["stim_signed"] = pd.to_numeric(part["stim"], errors="coerce").astype(np.float32)
            part["stim"] = part["stim_signed"].astype(np.float32)
            part["choice_signed"] = pd.to_numeric(part["choices"], errors="coerce").astype(np.float32)
            part["choice_bin"] = _signed_to_binary(part["choices"]).astype(np.float32)
            part["delay_raw"] = part["delays"].astype(np.float32)

            trace_input = pd.DataFrame(
                {
                    "Choice": _choice_to_binary(part["choice_bin"]),
                    "Hit": part["hit"].astype(float).to_numpy(),
                    "Punish": (1.0 - part["hit"].astype(float)).to_numpy(),
                }
            )
            at_choice, at_error, at_correct, reward_trace = get_action_trace(trace_input)
            part["at_choice"] = np.asarray(at_choice, dtype=np.float32)
            part["at_error"] = np.asarray(at_error, dtype=np.float32)
            part["at_correct"] = np.asarray(at_correct, dtype=np.float32)
            part["reward_trace"] = np.asarray(reward_trace, dtype=np.float32)

            prev_choice = part["choice_signed"].shift(1).fillna(0).astype(np.float32)
            prev_reward = part["hit"].shift(1).fillna(0).astype(np.float32)
            part["prev_choice"] = prev_choice
            part["prev_reward"] = prev_reward

            cumulative_reward = part["hit"].cumsum().shift(1).fillna(0).astype(float)
            max_cumulative_reward = float(np.nanmax(cumulative_reward.to_numpy())) if len(cumulative_reward) else 0.0
            if max_cumulative_reward > 0:
                cumulative_reward = cumulative_reward / max_cumulative_reward
            part["cumulative_reward"] = cumulative_reward.astype(np.float32)
            part["prev_abs_stim"] = part["stim"].abs().shift(1).fillna(0).astype(np.float32)
            prev_choice_signed = prev_choice.to_numpy().astype(np.float32)
            signed_prev_reward = np.where(prev_reward.to_numpy() > 0, 1.0, -1.0).astype(np.float32)
            part["wsls"] = (prev_choice_signed * signed_prev_reward).astype(np.float32)

            part["after_correct"] = part["after_correct"].fillna(0).astype(np.float32)
            part["repeat"] = part["repeat"].fillna(0).astype(np.float32)
            part["repeat_choice_side"] = part["repeat_choice_side"].fillna(0).astype(np.float32)
            part["WM"] = part["WM"].fillna(0).astype(np.float32)
            part["RL"] = part["RL"].fillna(0).astype(np.float32)
            part["ILD"] = part["stim"].astype(np.float32)
            parts.append(part)

        feature_df = pd.concat(parts, ignore_index=True)
        delay_raw = pd.to_numeric(feature_df["delay_raw"], errors="coerce").astype(np.float32)
        delay_mean = float(np.nanmean(delay_raw.to_numpy())) if len(delay_raw) else 0.0
        delay_std = float(np.nanstd(delay_raw.to_numpy())) if len(delay_raw) else 0.0
        if delay_std > 0:
            delay_z = ((delay_raw - delay_mean) / delay_std).astype(np.float32)
        else:
            delay_z = pd.Series(np.zeros(len(feature_df), dtype=np.float32), index=feature_df.index)
        feature_df["delay"] = delay_z
        feature_df["stim_x_delay"] = (feature_df["stim"].astype(np.float32) * feature_df["delay"].astype(np.float32))
        return pl.from_pandas(feature_df)

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        return self._build_feature_df(df_sub, tau=tau)

    def _resolved_emission_cols(
        self,
        feature_df: pl.DataFrame,
        emission_cols: List[str] | None,
    ) -> list[str]:
        requested = emission_cols if emission_cols is not None else self.default_emission_cols(feature_df)
        return list(requested)

    def load_subject(
        self,
        df_sub,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        feature_df = self._build_feature_df(df_sub, tau=tau)
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
        ecols = self._resolved_emission_cols(feature_df, emission_cols)
        ucols = transition_cols if transition_cols is not None else self.default_transition_cols()
        allowed_ecols = set(self.available_emission_cols(feature_df))
        bad_e = [c for c in ecols if c not in allowed_ecols]
        bad_u = [c for c in ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}"
            )

        y_np = feature_df["choice_bin"].to_numpy().astype(np.int32)
        y = jnp.asarray(y_np)
        X = (
            jnp.asarray(feature_df.select(ecols).to_numpy().astype(np.float32))
            if ecols
            else jnp.empty((len(y), 0), dtype=jnp.float32)
        )
        U = (
            jnp.asarray(feature_df.select(ucols).to_numpy().astype(np.float32))
            if ucols
            else jnp.empty((len(y), 0), dtype=jnp.float32)
        )
        names = {
            "X_cols": list(ecols),
            "U_cols": list(ucols),
        }
        return y, X, U, names

    def cv_balance_labels(self, feature_df: pl.DataFrame):
        if "stim" not in feature_df.columns:
            return None
        return feature_df["stim"].cast(pl.Float64)

    def default_emission_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        del df
        return list(EMISSION_COLS)

    def default_transition_cols(self) -> List[str]:
        return list(TRANSITION_COLS)

    def available_emission_cols(self, df: pl.DataFrame | None = None) -> List[str]:
        del df
        return list(EMISSION_COLS)

    def available_transition_cols(self) -> List[str]:
        return list(TRANSITION_COLS)

    def resolve_design_names(
        self,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
        df: pl.DataFrame | None = None,
    ) -> Dict[str, List[str]]:
        requested_ecols = list(emission_cols) if emission_cols is not None else self.default_emission_cols(df)
        requested_ucols = list(transition_cols) if transition_cols is not None else self.default_transition_cols()
        allowed_ecols = set(self.available_emission_cols(df))
        bad_e = [c for c in requested_ecols if c not in allowed_ecols]
        bad_u = [c for c in requested_ucols if c not in TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {TRANSITION_COLS}"
            )
        return {"X_cols": list(requested_ecols), "U_cols": list(requested_ucols)}

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
            "trial_idx": "trial",
            "trial": "trial",
            "session": "session",
            "stimulus": "stim",
            "response": "choices",
            "performance": "hit",
        }

    def get_plots(self) -> types.ModuleType:
        import tasks.plots.two_afc_delay as plots

        return plots

    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        pairs = self._SCORING_OPTIONS.get(
            getattr(self, "scoring_key", "stim (w)"),
            self._SCORING_OPTIONS["stim (w)"],
        )

        def _score_states(W_np: np.ndarray, feat_names: list[str], *, stim: str = "stim_vals") -> np.ndarray:
            name2fi_local = {n: i for i, n in enumerate(feat_names)}
            scores = np.zeros(W_np.shape[0], dtype=float)
            n_terms = 0
            for feat_name, mode in pairs:
                fi = name2fi_local.get(feat_name)
                if fi is None:
                    continue
                vals = W_np[:, 0, fi].astype(float)
                if mode == "neg":
                    vals = -vals
                elif mode == "abs":
                    vals = np.abs(vals)
                elif mode == "pos":
                    vals = vals
                else:
                    raise ValueError(f"Unknown 2AFC-delay scoring mode {mode!r}.")
                scores += vals
                n_terms += 1

            if n_terms > 0:
                return scores / n_terms

            stim_candidates = [stim]
            if stim != "stim_vals":
                stim_candidates.append("stim_vals")
            for stim_name in stim_candidates:
                stim_fi_local = name2fi_local.get(stim_name)
                if stim_fi_local is not None:
                    return -W_np[:, 0, stim_fi_local]
            return -W_np[:, 0, :].mean(axis=1)

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

            selected_stim = "stim"
            state_scores = _score_states(W, feat, stim=selected_stim)

            engaged_k = int(np.argmax(state_scores))
            others = [k for k in range(K) if k != engaged_k]

            if K == 2:
                labels = {engaged_k: "Engaged", others[0]: "Disengaged"}
                order = [engaged_k, others[0]]
            elif K == 3:
                bias_fi = name2fi.get("bias", None)
                if bias_fi is None:
                    bias_vals = np.zeros(len(others))
                else:
                    bias_vals = W[others, 0, bias_fi]
                left_k = others[int(np.argmin(bias_vals))]
                right_k = others[int(np.argmax(bias_vals))]
                labels = {
                    engaged_k: "Engaged",
                    left_k: "Biased L",
                    right_k: "Biased R",
                }
                order = [engaged_k, left_k, right_k]
            else:
                ranked_rest = sorted(others, key=lambda k: state_scores[k], reverse=True)
                labels = {engaged_k: "Engaged"}
                for idx, k in enumerate(ranked_rest, start=1):
                    labels[k] = f"Disengaged {idx}"
                order = [engaged_k, *ranked_rest]

            state_labels[subj] = labels
            state_order[subj] = order

        return state_labels, state_order
