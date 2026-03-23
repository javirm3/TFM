import numpy as np
import polars as pl
import jax
import jax.numpy as jnp
from typing import Tuple, Dict


def zscore_cols(M: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    mean = jnp.mean(M, axis=0, keepdims=True)
    std  = jnp.std(M, axis=0, keepdims=True)
    std  = jnp.where(std < eps, 1.0, std)
    return (M - mean) / std

def action_trace(r_c: jnp.ndarray, tau: float) -> jnp.ndarray:
    """
    r_c: (T,1) encoded as (L,C,R)
    returns A: (T,3) with A_t^X = sum_{k>=1} r_{t-k}^X * exp(-k/tau)
    using A_t = lam*A_{t-1} + r_{t-1}
    """
    
    r_onehot = jax.nn.one_hot(r_c.squeeze(), 3).astype(jnp.float32)
    lam = jnp.exp(-1.0 / tau).astype(jnp.float32)
    r_prev = jnp.vstack([jnp.zeros((1, r_onehot.shape[1]), dtype=jnp.float32), r_onehot[:-1]])

    def step(prev, current):
        new = lam * prev + current
        return new, new

    _, A = jax.lax.scan(step, jnp.zeros((r_onehot.shape[1],), dtype=jnp.float32), r_prev)
    return A


def action_trace_plus_minus(
    x_c: jnp.ndarray,       # (T,) int {0,1,2}  (context side)
    outcome: jnp.ndarray,   # (T,) 0/1 (1=correct, 0=incorrect)
    tau: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build outcome traces split into positive and negative parts.
    Returns:
      A_plus:      (T,1) merged = sum over sides of A_pos_sides  (so it's >=0)
      A_minus:     (T,1) merged = sum over sides of A_neg_sides  (so it's >=0)
    Recurrence: A_t = lam*A_{t-1} + r_{t-1}
    """
    x_c = jnp.asarray(x_c).astype(jnp.int32).squeeze()
    outcome = jnp.asarray(outcome).astype(jnp.float32).squeeze()

    # one-hot context: (T,3)
    ctx_oh = jax.nn.one_hot(x_c, 3).astype(jnp.float32)

    # impulses (magnitude only): correct -> +1 on ctx side, incorrect -> +1 on ctx side
    r_pos = ctx_oh * outcome[:, None]          # (T,3), outcome=1 -> 1, else 0
    r_neg = ctx_oh * (1.0 - outcome)[:, None]  # (T,3), outcome=0 -> 1, else 0

    lam = jnp.exp(-1.0 / tau).astype(jnp.float32)

    def exp_trace(r_mat: jnp.ndarray) -> jnp.ndarray:
        r_prev = jnp.vstack([jnp.zeros((1, 3), dtype=jnp.float32), r_mat[:-1]])

        def step(prev, cur):
            new = lam * prev + cur
            return new, new

        _, A = jax.lax.scan(step, jnp.zeros((3,), dtype=jnp.float32), r_prev)
        return A  # (T,3)

    A_pos_sides = exp_trace(r_pos)
    A_neg_sides = exp_trace(r_neg)
    # merged across sides -> (T,)
    A_plus = jnp.sum(A_pos_sides, axis=1)   # >=0
    A_minus = jnp.sum(A_neg_sides, axis=1)  # >=0

    return A_plus[:, None], A_minus[:, None]

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


def build_sequence_from_df(
    df_sub: pl.DataFrame,
    tau: float = 50,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
    session_col: str = "session",
):
    """Build (y, X, U, names, AU) arrays from a subject DataFrame.

    Args:
        df_sub          : raw trial DataFrame for one (or more) subjects.
        tau             : half-life for exponential action traces.
        emission_cols   : subset of emission features to include in X. Defaults to all features in ``_ALL_EMISSION_COLS``.
        transition_cols : subset of transition features to include in U. Defaults to all features in ``_ALL_TRANSITION_COLS``.

    Returns:
        y       : (T,) int {0,1,2} actions
        X       : (T, n_emission_features) emission features
        U       : (T, n_transition_features) transition features
        names   : dict with keys "X_cols" and "U_cols" listing the column names of the features in X and U, respectively.
        AU      : (T, 2) action traces for positive and negative outcomes (A_plus, A_minus), which can be included as features if desired.

    Raises:
        ValueError if requested column names are not in the available sets.
    """
    _ecols = emission_cols if emission_cols is not None else _ALL_EMISSION_COLS
    _ucols = transition_cols if transition_cols is not None else _ALL_TRANSITION_COLS

    # validate requested column names
    _bad_e = [c for c in _ecols if c not in _ALL_EMISSION_COLS]
    _bad_u = [c for c in _ucols if c not in _ALL_TRANSITION_COLS]
    if _bad_e:
        raise ValueError(f"Unknown emission_cols: {_bad_e}. Available: {_ALL_EMISSION_COLS}")
    if _bad_u:
        raise ValueError(f"Unknown transition_cols: {_bad_u}. Available: {_ALL_TRANSITION_COLS}")

    df_sub = df_sub.sort("trial_idx")
    # z-score stimd_n so that SL/SC/SR carry normalised stimulus strength
    df_sub = df_sub.with_columns([
        ((pl.col("stimd_n") - pl.col("stimd_n").mean()) / pl.col("stimd_n").std()).alias("stimd_n_z"),
    ])
    df_sub = df_sub.with_columns([
        pl.col("response").cast(pl.Int32),

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
        ((pl.col("ttype_n"))).cast(pl.Float32).alias("D"),
        ((pl.col("x_c") == "L") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SLxD"),
        ((pl.col("x_c") == "C") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SCxD"),
        ((pl.col("x_c") == "R") * pl.col("stimd_n_z") * pl.col("ttype_n")).cast(pl.Float32).alias("SRxD"),

        
        ((pl.col("x_c") == "L") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SLxdelay"),
        ((pl.col("x_c") == "C") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SCxdelay"),
        ((pl.col("x_c") == "R") * pl.col("stimd_n_z") * pl.col("delay_d")).cast(pl.Float32).alias("SRxdelay"),

        # stim interval one-hot × side
        # stim_i = 1 when [onset, offset] overlaps interval i, matching onset_offset_from_codes logic
        # VG (onset=0, offset=0) is detected by offset==0 and gets stim1=stim2=stim3=1
        # stim4 is only triggered by SIL (offset > timepoint_3)
        (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0) | (pl.col("offset") == 0)) & (pl.col("x_c") == "L")).cast(pl.Float32).alias("stim1L"),
        (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0) | (pl.col("offset") == 0)) & (pl.col("x_c") == "C")).cast(pl.Float32).alias("stim1C"),
        (((pl.col("onset") < pl.col("timepoint_1")) & (pl.col("offset") > 0) | (pl.col("offset") == 0)) & (pl.col("x_c") == "R")).cast(pl.Float32).alias("stim1R"),
        (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1")) | (pl.col("offset") == 0)) & (pl.col("x_c") == "L")).cast(pl.Float32).alias("stim2L"),
        (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1")) | (pl.col("offset") == 0)) & (pl.col("x_c") == "C")).cast(pl.Float32).alias("stim2C"),
        (((pl.col("onset") < pl.col("timepoint_2")) & (pl.col("offset") > pl.col("timepoint_1")) | (pl.col("offset") == 0)) & (pl.col("x_c") == "R")).cast(pl.Float32).alias("stim2R"),
        (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2")) | (pl.col("offset") == 0)) & (pl.col("x_c") == "L")).cast(pl.Float32).alias("stim3L"),
        (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2")) | (pl.col("offset") == 0)) & (pl.col("x_c") == "C")).cast(pl.Float32).alias("stim3C"),
        (((pl.col("onset") < pl.col("timepoint_3")) & (pl.col("offset") > pl.col("timepoint_2")) | (pl.col("offset") == 0)) & (pl.col("x_c") == "R")).cast(pl.Float32).alias("stim3R"),
        ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "L")).cast(pl.Float32).alias("stim4L"),
        ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "C")).cast(pl.Float32).alias("stim4C"),
        ((pl.col("onset") < pl.col("timepoint_4")) & (pl.col("offset") > pl.col("timepoint_3")) & (pl.col("x_c") == "R")).cast(pl.Float32).alias("stim4R"),

        ((pl.col("x_c") == "L") * pl.col("delay_d") * pl.col("ttype_n")).cast(pl.Float32).alias("stim1"),



        pl.col("performance").shift(1).fill_null(0).cast(pl.Float32).over(session_col).alias("previous_outcome"),
        pl.col("response").shift(1).fill_null(0.0).eq(0).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).over(session_col).alias("A_L"),
        pl.col("response").shift(1).fill_null(0.0).eq(1).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).over(session_col).alias("A_C"),
        pl.col("response").shift(1).fill_null(0.0).eq(2).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).over(session_col).alias("A_R"),
        (1/((pl.col("timepoint_3")-pl.col("timepoint_4")))).cast(pl.Float32).alias("speed3"),
        (1/((pl.col("timepoint_3")-pl.col("timepoint_2")))).cast(pl.Float32).alias("speed2"),
        (1/((pl.col("timepoint_2")-pl.col("timepoint_1")))).cast(pl.Float32).alias("speed1"),
    ])
    df_sub = df_sub.with_columns([
        pl.col("previous_outcome").ewm_mean(half_life=tau, adjust=False).over(session_col).alias("A_plus"),
        (1.0 - pl.col("previous_outcome")).ewm_mean(half_life=tau, adjust=False).over(session_col).alias("A_minus"),
        (pl.col("A_L") * pl.col("delay_d")).cast(pl.Float32).alias("ALxdelay"),
        (pl.col("A_R") * pl.col("delay_d")).cast(pl.Float32).alias("ARxdelay"),
    ])
    # z-score speed features
    df_sub = df_sub.with_columns([
        ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).cast(pl.Float32).alias(c)
        for c in ["speed1", "speed2", "speed3"]
    ])

    y = df_sub["response"].to_numpy()

    X_base = df_sub.select(_ecols).to_numpy().astype(jnp.float32)
    X = jnp.asarray(X_base)
    U_base = df_sub.select(_ucols).to_numpy().astype(jnp.float32)
    U = jnp.asarray(U_base)

    A_plus  = jnp.asarray(df_sub["A_plus"].to_numpy())[:, None]
    A_minus = jnp.asarray(df_sub["A_minus"].to_numpy())[:, None]

    names = {
        "X_cols": list(_ecols),
        "U_cols": list(_ucols),
    }
    return jnp.asarray(y), jnp.asarray(X), jnp.asarray(U), names, jnp.concatenate([A_plus, A_minus], axis=1)


# ── 2AFC / binary-choice variant ──────────────────────────────────────────────

# Scalar covariates produced by parse_glmhmm (one column each).
# Multi-column covariates (net_ild, stim_strength, session_index) are
# excluded here; pass them explicitly via the covariates kwarg.
_ALL_2AFC_EMISSION_COLS: list[str] = [
    "bias",       # constant 1.0
    "stim_vals",  # ILD normalised to [-1, 1] per session
    "stim_strength",
    "at_choice",  # EWMA of signed choice history
    "at_error",   # EWMA of error-weighted signed choice
    "at_correct", # EWMA of correct-weighted signed choice
    "reward_trace",
    "prev_choice",# previous choice
    "wsls",       # win-stay-lose-switch
    "prev_reward",
    "cumulative_reward",
    "prev_abs_stim",
]
_AVAILABLE_2AFC_EMISSION_COLS: list[str] = list(_ALL_2AFC_EMISSION_COLS)
_ALL_2AFC_TRANSITION_COLS: list[str] = [
    "at_choice",  # EWMA of signed choice history
    "at_correct", # EWMA of correct-weighted signed choice
    "at_error",   # EWMA of error-weighted signed choice
    "reward_trace",  # EWMA of reward history (unsigned Hit)
    "prev_abs_stim",
    "prev_reward",
    "cumulative_reward",
]
_AVAILABLE_2AFC_TRANSITION_COLS: list[str] = list(_ALL_2AFC_TRANSITION_COLS)

# Frame-level stimulus columns (sf_0 … sf_N) are validated separately
_SF_COL_PREFIX = "sf_"

_PARSE_GLMHMM_ORDER = [
    "stim_vals", "stim_strength", "net_ild",
    "bias", "session_index",
    "at_choice", "at_error", "at_correct", "reward_trace",
    "prev_choice", "wsls",
    "prev_reward", "cumulative_reward", "prev_abs_stim",
]


def _ordered_2afc_covariates(covariates: list[str]) -> list[str]:
    ordered = [c for c in _PARSE_GLMHMM_ORDER if c in covariates]
    ordered += [c for c in covariates if c not in ordered]
    return ordered


def _ordered_parse_glmhmm_covariates(covariates: list[str]) -> list[str]:
    """Backward-compatible alias for the 2AFC covariate ordering helper."""
    return _ordered_2afc_covariates(covariates)


def _concat_glmhmm_sessions(
    inputs: list[np.ndarray],
    choices: list[np.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    X = jnp.asarray(np.vstack(inputs).astype(np.float32))
    y = jnp.asarray(np.concatenate([np.asarray(c).reshape(-1) for c in choices]).astype(np.int32))
    return y, X


def build_sequence_from_df_binary(
    df_sub: pl.DataFrame,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
    session_col: str = "session",
    trial_col: str = "trial",
    response_col: str = "response",
    action_choice_col: str | None = None,
    performance_col: str = "performance",
    stim_value_col: str = "ILD",
    include_action_trace: bool = True,
    tau_choice: float = 1.58,
    tau_error: float = 2.22,
    tau_correct: float = 0.95,
    tau_reward: float | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
    """Build ``(y, X, U, names)`` for a binary task directly from a Polars df.

    This is the dataframe-native alternative to the Alexis parser path. It
    assumes the input already contains task-normalised columns and computes the
    binary regressors with Polars expressions and session-local ``ewm_mean``.
    """
    _ecols = emission_cols if emission_cols is not None else _ALL_2AFC_EMISSION_COLS
    _ucols = transition_cols if transition_cols is not None else _ALL_2AFC_TRANSITION_COLS

    _bad_e = [c for c in _ecols if c not in _ALL_2AFC_EMISSION_COLS]
    _bad_u = [c for c in _ucols if c not in _ALL_2AFC_TRANSITION_COLS]
    if _bad_e:
        raise ValueError(f"Unknown emission_cols: {_bad_e}. Available: {_ALL_2AFC_EMISSION_COLS}")
    if _bad_u:
        raise ValueError(f"Unknown transition_cols: {_bad_u}. Available: {_ALL_2AFC_TRANSITION_COLS}")

    trace_cols = {"at_choice", "at_error", "at_correct", "reward_trace", "wsls"}
    requested_cols = set(_ecols) | set(_ucols)
    if not include_action_trace and requested_cols & trace_cols:
        raise ValueError(
            "Action-trace regressors were requested, but include_action_trace=False. "
            f"Requested trace cols: {sorted(requested_cols & trace_cols)}"
        )

    tau_reward = tau_correct if tau_reward is None else tau_reward
    action_choice_col = action_choice_col or response_col

    df_sub = df_sub.sort([session_col, trial_col])

    stim_scale = float(df_sub.select(pl.col(stim_value_col).abs().max()).item() or 0.0)
    if stim_scale <= 0:
        stim_scale = 1.0

    choice_signed_expr = (
        pl.when(pl.col("_action_choice") == 1)
        .then(pl.lit(1.0))
        .when(pl.col("_action_choice") == 0)
        .then(pl.lit(-1.0))
        .otherwise(pl.lit(0.0))
        .cast(pl.Float32)
    )

    df_sub = df_sub.with_columns(
        [
            pl.col(response_col).cast(pl.Int32).alias("_response"),
            pl.col(action_choice_col).cast(pl.Int32).alias("_action_choice"),
            pl.col(performance_col).cast(pl.Float32).alias("_reward"),
            (pl.col(stim_value_col).cast(pl.Float32) / pl.lit(stim_scale)).alias("stim_vals"),
            pl.lit(1.0).cast(pl.Float32).alias("bias"),
        ]
    )
    df_sub = df_sub.with_columns(
        [
            pl.col("stim_vals").abs().cast(pl.Float32).alias("stim_strength"),
            choice_signed_expr.alias("_choice_signed"),
            pl.col("_choice_signed").shift(1).fill_null(0.0).over(session_col).cast(pl.Float32).alias("_prev_choice_signed"),
            (pl.col("_choice_signed") * pl.col("_reward")).shift(1).fill_null(0.0).over(session_col).cast(pl.Float32).alias("_prev_correct_signed"),
            (pl.col("_choice_signed") * (1.0 - pl.col("_reward"))).shift(1).fill_null(0.0).over(session_col).cast(pl.Float32).alias("_prev_error_signed"),
            pl.col("_action_choice").shift(1).fill_null(0).over(session_col).cast(pl.Float32).alias("prev_choice"),
            pl.col("_reward").shift(1).fill_null(0.0).over(session_col).cast(pl.Float32).alias("prev_reward"),
            pl.col("stim_vals").abs().shift(1).fill_null(0.0).over(session_col).cast(pl.Float32).alias("prev_abs_stim"),
            pl.col("_reward").shift(1).fill_null(0.0).cum_sum().over(session_col).cast(pl.Float32).alias("_cumulative_reward_raw"),
        ]
    )
    df_sub = df_sub.with_columns(
        [
            pl.when(pl.col("_cumulative_reward_raw").max().over(session_col) > 0)
            .then(pl.col("_cumulative_reward_raw") / pl.col("_cumulative_reward_raw").max().over(session_col))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float32)
            .alias("cumulative_reward"),
        ]
    )

    if include_action_trace:
        df_sub = df_sub.with_columns(
            [
                pl.col("_prev_choice_signed").ewm_mean(half_life=tau_choice, adjust=False).over(session_col).cast(pl.Float32).alias("at_choice"),
                pl.col("_prev_correct_signed").ewm_mean(half_life=tau_correct, adjust=False).over(session_col).cast(pl.Float32).alias("at_correct"),
                pl.col("_prev_error_signed").ewm_mean(half_life=tau_error, adjust=False).over(session_col).cast(pl.Float32).alias("at_error"),
                pl.col("prev_reward").ewm_mean(half_life=tau_reward, adjust=False).over(session_col).cast(pl.Float32).alias("reward_trace"),
                pl.when(pl.col("prev_reward") > 0)
                .then(pl.col("_prev_choice_signed"))
                .otherwise(-pl.col("_prev_choice_signed"))
                .cast(pl.Float32)
                .alias("wsls"),
            ]
        )

    y = df_sub["_response"].to_numpy().astype(np.int32)
    X = jnp.asarray(df_sub.select(_ecols).to_numpy().astype(np.float32)) if _ecols else jnp.empty((len(y), 0), dtype=jnp.float32)
    U = jnp.asarray(df_sub.select(_ucols).to_numpy().astype(np.float32)) if _ucols else jnp.empty((len(y), 0), dtype=jnp.float32)
    names = {
        "X_cols": _ordered_2afc_covariates(list(_ecols)),
        "U_cols": _ordered_2afc_covariates(list(_ucols)),
    }
    return jnp.asarray(y), X, U, names




def build_sequence_from_df_2afc(
    df_sub,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
    clean_start: bool = True,
    drop_miss:   bool = True,
    filter_drug: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
    """Build ``(y, X, U, names)`` for a 2-class GLM-HMM.

    Delegates to :func:`alexis_functions.filter_behavior` for data cleaning
    and :func:`alexis_functions.parse_glmhmm` for feature construction.

    Args:
        df_sub:      DataFrame (Polars or pandas) for one subject.
        covariates:  Covariates passed to ``parse_glmhmm``. Defaults to all
                     scalar covariates in ``_ALL_2AFC_EMISSION_COLS``.
                     Multi-column options (``'stim_strength'``,
                     ``'net_ild'``, ``'session_index'``) can also be added.
        clean_start: Passed to ``filter_behavior``.
        drop_miss:   Passed to ``filter_behavior``.
        filter_drug: Passed to ``filter_behavior``.

    Returns:
        y     : ``(T,)`` int32 array of choices {0, 1}.
        X     : ``(T, M)`` float32 JAX array of emission features.
        U     : ``(T, D)`` float32 JAX array of transition features.
        names : dict with keys ``"X_cols"`` and ``"U_cols"``.
    """
    from scripts.alexis_functions import parse_glmhmm

    # ── convert Polars → pandas if needed ──────────────────────────────────
    if hasattr(df_sub, "to_pandas"):
        df_pd = df_sub.to_pandas()
    else:
        df_pd = df_sub.copy()

    # ── build features via parse_glmhmm ────────────────────────────────────
    if emission_cols is None:
        emission_cols = list(_ALL_2AFC_EMISSION_COLS)  # all scalar covariates
    if transition_cols is None:
        transition_cols = list(_ALL_2AFC_TRANSITION_COLS)

    y = None
    if emission_cols:
        x_inputs, x_choices = parse_glmhmm(df_pd, covariates=emission_cols)
        y, X = _concat_glmhmm_sessions(x_inputs, x_choices)
    else:
        X = None

    if transition_cols:
        u_inputs, u_choices = parse_glmhmm(df_pd, covariates=transition_cols)
        y_u, U = _concat_glmhmm_sessions(u_inputs, u_choices)
        if y is None:
            y = y_u
        elif y.shape != y_u.shape or not np.array_equal(np.asarray(y), np.asarray(y_u)):
            raise ValueError("Emission and transition parses produced inconsistent choice arrays.")
    else:
        if y is None:
            raise ValueError("At least one of emission_cols or transition_cols must be non-empty.")
        U = jnp.empty((int(y.shape[0]), 0), dtype=jnp.float32)

    if X is None:
        X = jnp.empty((int(y.shape[0]), 0), dtype=jnp.float32)

    names = {
        "X_cols": _ordered_2afc_covariates(emission_cols),
        "U_cols": _ordered_2afc_covariates(transition_cols),
    }
    return y, X, U, names
