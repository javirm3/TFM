"""Task adapter for the Tiffany 2AFC delay task."""
from __future__ import annotations

import types
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl

from glmhmmt.cli.alexis_functions import get_action_trace
from glmhmmt.tasks import TaskAdapter, _register

_STIM_ABS_COL_PREFIX = "stim_"
_STIM_PARAM_COL = "stim_param"

_ALL_2AFC_DELAY_EMISSION_COLS: list[str] = [
    "bias",
    "stim_vals",
    "stim_param",
    "delay",
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
    "WM",
    "RL",
]
_AVAILABLE_2AFC_DELAY_EMISSION_COLS: list[str] = list(_ALL_2AFC_DELAY_EMISSION_COLS)
_ALL_2AFC_DELAY_TRANSITION_COLS: list[str] = [
    "at_choice",
    "at_correct",
    "at_error",
    "reward_trace",
    "prev_abs_stim",
    "prev_reward",
    "cumulative_reward",
    "delay",
]
_AVAILABLE_2AFC_DELAY_TRANSITION_COLS: list[str] = list(_ALL_2AFC_DELAY_TRANSITION_COLS)

EMISSION_REGRESSOR_LABELS: dict[str, str] = {
    "stim_vals": r"$\mathrm{Stimulus}$",
    "stim_param": r"$\mathrm{Stimulus}_{\mathrm{param}}$",
    "delay": r"$\mathrm{Delay}$",
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
    stim_col = "stim" if "stim" in columns else "stimulus" if "stimulus" in columns else None
    if stim_col is None:
        return []
    stim_series = df[stim_col].drop_nulls() if isinstance(df, pl.DataFrame) else df[stim_col].dropna()
    stim_abs_levels = sorted({int(abs(v)) for v in stim_series.to_list()})
    return [f"{_STIM_ABS_COL_PREFIX}{stim_abs}" for stim_abs in stim_abs_levels]


def _stim_param_weight_map() -> dict[int, float]:
    """Return a simple sign-preserving one-hot stimulus map for Tiffany 2AFC."""
    return {0: 0.0, 1: 1.0}


def _build_stim_param(part: pd.DataFrame, stim_abs_levels: list[int]) -> np.ndarray:
    weight_map = _stim_param_weight_map()
    stim = part["stim"].astype(float).to_numpy()
    values = np.zeros(len(part), dtype=np.float32)
    for stim_abs in stim_abs_levels:
        if stim_abs == 0:
            continue
        mask = np.abs(stim) == stim_abs
        values[mask] = np.sign(stim[mask]) * float(weight_map.get(stim_abs, 0.0))
    return values


def _choice_to_binary(series: pd.Series) -> np.ndarray:
    return (series.astype(float).to_numpy() > 0).astype(np.int32)


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
        "stim_vals (-w)": [("stim_vals", "neg")],
        "stim_vals (|w|)": [("stim_vals", "abs")],
        "stim_param (-w)": [("stim_param", "neg")],
        "stim_param (|w|)": [("stim_param", "abs")],
        "delay (|w|)": [("delay", "abs")],
        "at_choice (|w|)": [("at_choice", "abs")],
        "wsls (|w|)": [("wsls", "abs")],
        "bias (|w|)": [("bias", "abs")],
    }
    scoring_key: str = "stim_vals (-w)"

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

        stim_scale = float(df_pd["stim"].abs().max() or 0.0)
        if stim_scale <= 0:
            stim_scale = 1.0

        stim_abs_levels = sorted({int(abs(v)) for v in df_pd["stim"].dropna().astype(int).tolist()})
        parts: list[pd.DataFrame] = []
        for _, df_session in df_pd.groupby("session", sort=False):
            part = df_session.copy().reset_index(drop=True)
            part["bias"] = 1.0
            part["stim_vals"] = (part["stim"].astype(float) / stim_scale).astype(np.float32)
            for stim_abs in stim_abs_levels:
                if stim_abs == 0:
                    stim_col = np.where(part["stim"] == 0, 1.0, 0.0).astype(np.float32)
                else:
                    stim_col = np.select(
                        [part["stim"] == stim_abs, part["stim"] == -stim_abs],
                        [1.0, -1.0],
                        default=0.0,
                    ).astype(np.float32)
                part[f"{_STIM_ABS_COL_PREFIX}{stim_abs}"] = stim_col
            part[_STIM_PARAM_COL] = _build_stim_param(part, stim_abs_levels)
            part["delay"] = part["delays"].astype(np.float32)

            trace_input = pd.DataFrame(
                {
                    "Choice": _choice_to_binary(part["choices"]),
                    "Hit": part["hit"].astype(float).to_numpy(),
                    "Punish": (1.0 - part["hit"].astype(float)).to_numpy(),
                }
            )
            at_choice, at_error, at_correct, reward_trace = get_action_trace(trace_input)
            part["at_choice"] = np.asarray(at_choice, dtype=np.float32)
            part["at_error"] = np.asarray(at_error, dtype=np.float32)
            part["at_correct"] = np.asarray(at_correct, dtype=np.float32)
            part["reward_trace"] = np.asarray(reward_trace, dtype=np.float32)

            prev_choice = part["choices"].shift(1).fillna(0).astype(np.float32)
            prev_reward = part["hit"].shift(1).fillna(0).astype(np.float32)
            part["prev_choice"] = prev_choice
            part["prev_reward"] = prev_reward

            cumulative_reward = part["hit"].cumsum().shift(1).fillna(0).astype(float)
            max_cumulative_reward = float(np.nanmax(cumulative_reward.to_numpy())) if len(cumulative_reward) else 0.0
            if max_cumulative_reward > 0:
                cumulative_reward = cumulative_reward / max_cumulative_reward
            part["cumulative_reward"] = cumulative_reward.astype(np.float32)
            part["prev_abs_stim"] = (part["stim"].abs().shift(1).fillna(0) / stim_scale).astype(np.float32)
            signed_prev_reward = np.where(prev_reward.to_numpy() > 0, 1.0, -1.0).astype(np.float32)
            part["wsls"] = (prev_choice.to_numpy() * signed_prev_reward).astype(np.float32)

            part["after_correct"] = part["after_correct"].fillna(0).astype(np.float32)
            part["repeat"] = part["repeat"].fillna(0).astype(np.float32)
            part["repeat_choice_side"] = part["repeat_choice_side"].fillna(0).astype(np.float32)
            part["WM"] = part["WM"].fillna(0).astype(np.float32)
            part["RL"] = part["RL"].fillna(0).astype(np.float32)

            # Keep a signed evidence axis compatible with the copied 2AFC plots.
            part["ILD"] = part["stim"].astype(np.float32)
            parts.append(part)

        return pl.from_pandas(pd.concat(parts, ignore_index=True))

    def build_feature_df(self, df_sub: pl.DataFrame, tau: float = 50.0) -> pl.DataFrame:
        return self._build_feature_df(df_sub, tau=tau)

    def _resolved_emission_cols(
        self,
        feature_df: pl.DataFrame,
        emission_cols: List[str] | None,
    ) -> list[str]:
        requested = emission_cols if emission_cols is not None else self.default_emission_cols()
        resolved: list[str] = []
        for col in requested:
            resolved.append(col)
        return resolved

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
        allowed_ecols = set(self.available_emission_cols()) | set(_stim_abs_cols(feature_df.columns))
        bad_e = [c for c in ecols if c not in allowed_ecols]
        bad_u = [c for c in ucols if c not in _AVAILABLE_2AFC_DELAY_TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {_AVAILABLE_2AFC_DELAY_TRANSITION_COLS}"
            )

        y_np = (feature_df["choices"].to_numpy().astype(np.float32) > 0).astype(np.int32)
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

    def default_emission_cols(self) -> List[str]:
        return [c for c in _ALL_2AFC_DELAY_EMISSION_COLS if c != _STIM_PARAM_COL]

    def default_transition_cols(self) -> List[str]:
        return list(_ALL_2AFC_DELAY_TRANSITION_COLS)

    def available_emission_cols(self) -> List[str]:
        return list(_AVAILABLE_2AFC_DELAY_EMISSION_COLS)

    def available_transition_cols(self) -> List[str]:
        return list(_AVAILABLE_2AFC_DELAY_TRANSITION_COLS)

    def resolve_design_names(
        self,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
        df: pl.DataFrame | None = None,
    ) -> Dict[str, List[str]]:
        requested_ecols = list(emission_cols) if emission_cols is not None else self.default_emission_cols()
        requested_ucols = list(transition_cols) if transition_cols is not None else self.default_transition_cols()

        extra_cols: list[str] = []
        if df is not None:
            extra_cols = self.available_extra_emission_cols(df)

        allowed_ecols = set(self.available_emission_cols()) | set(extra_cols)
        bad_e = [c for c in requested_ecols if c not in allowed_ecols]
        bad_u = [c for c in requested_ucols if c not in _AVAILABLE_2AFC_DELAY_TRANSITION_COLS]
        if bad_e:
            raise ValueError(f"Unknown emission_cols: {bad_e}. Available: {sorted(allowed_ecols)}")
        if bad_u:
            raise ValueError(
                f"Unknown transition_cols: {bad_u}. Available: {_AVAILABLE_2AFC_DELAY_TRANSITION_COLS}"
            )
        return {"X_cols": list(requested_ecols), "U_cols": list(requested_ucols)}

    def stim_abs_cols(self, df: pl.DataFrame) -> List[str]:
        return _infer_stim_abs_cols_from_df(df)

    def available_extra_emission_cols(self, df: pl.DataFrame) -> List[str]:
        return list(dict.fromkeys(self.stim_abs_cols(df)))

    def default_extra_emission_cols(self, df: pl.DataFrame) -> List[str]:
        return []

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
        """Provisional Tiffany column mapping; confirm if you want different fields."""
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
            getattr(self, "scoring_key", "stim_vals (-w)"),
            self._SCORING_OPTIONS["stim_vals (-w)"],
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

            selected_stim = "stim_param" if getattr(self, "scoring_key", "").startswith("stim_param") else "stim_vals"
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
