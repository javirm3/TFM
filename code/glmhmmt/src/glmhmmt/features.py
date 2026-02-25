import numpy as np
import polars as pl
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, List

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
    "biasL", "biasR", "onsetL", "onsetC", "onsetR", "delay",
    "DR", "DL", "SL", "SC", "SR",
    "SLxdelay", "SCxdelay", "SRxdelay",
    "A_L", "A_R",
]
_ALL_TRANSITION_COLS: list[str] = ["A_plus", "A_minus", "A_L", "A_C", "A_R"]


def build_sequence_from_df(
    df_sub: pl.DataFrame,
    tau: float = 50,
    emission_cols: list[str] | None = None,
    transition_cols: list[str] | None = None,
):
    """Build (y, X, U, names, AU) arrays from a subject DataFrame.

    Parameters
    ----------
    df_sub          : raw trial DataFrame for one (or more) subjects.
    tau             : half-life for exponential action traces.
    emission_cols   : subset of emission features to include in X.
                      Defaults to all features in ``_ALL_EMISSION_COLS``.
    transition_cols : subset of transition features to include in U.
                      Defaults to all features in ``_ALL_TRANSITION_COLS``.
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
    df_sub = df_sub.with_columns([
        pl.col("response").cast(pl.Int32),

        (pl.col("x_c") == "L").cast(pl.Float32).alias("biasL"),
        (pl.col("x_c") == "R").cast(pl.Float32).alias("biasR"),
        
        pl.lit(1.0).cast(pl.Float32).alias("bias"),
        
        pl.col("delay_d").cast(pl.Float32).alias("delay"),
        ((pl.col("x_c") == "L") * pl.col("onset")).cast(pl.Float32).alias("onsetL"),
        ((pl.col("x_c") == "C") * pl.col("onset")).cast(pl.Float32).alias("onsetC"),
        ((pl.col("x_c") == "R") * pl.col("onset")).cast(pl.Float32).alias("onsetR"),

        ((pl.col("x_c") == "L") * pl.col("stim_d")).cast(pl.Float32).alias("SL"),
        ((pl.col("x_c") == "C") * pl.col("stim_d")).cast(pl.Float32).alias("SC"),
        ((pl.col("x_c") == "R") * pl.col("stim_d")).cast(pl.Float32).alias("SR"),

        ((pl.col("x_c") == "L") * pl.col("delay_d")).cast(pl.Float32).alias("DL"),
        ((pl.col("x_c") == "C") * pl.col("delay_d")).cast(pl.Float32).alias("DC"),
        ((pl.col("x_c") == "R") * pl.col("delay_d")).cast(pl.Float32).alias("DR"),
        
        ((pl.col("x_c") == "L") * pl.col("stim_d") * pl.col("delay_d")).cast(pl.Float32).alias("SLxdelay"),
        ((pl.col("x_c") == "C") * pl.col("stim_d") * pl.col("delay_d")).cast(pl.Float32).alias("SCxdelay"),
        ((pl.col("x_c") == "R") * pl.col("stim_d") * pl.col("delay_d")).cast(pl.Float32).alias("SRxdelay"),


        pl.col("performance").shift(1).fill_null(0).cast(pl.Float32).alias("previous_outcome"),
        pl.col("response").shift(1).fill_null(0.0).eq(0).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).alias("A_L"),
        pl.col("response").shift(1).fill_null(0.0).eq(1).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).alias("A_C"),
        pl.col("response").shift(1).fill_null(0.0).eq(2).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).alias("A_R"),
    ])
    df_sub = df_sub.with_columns([
        pl.col("previous_outcome").ewm_mean(half_life=tau, adjust=False).alias("A_plus"),
        (1.0 - pl.col("previous_outcome")).ewm_mean(half_life=tau, adjust=False).alias("A_minus"),
        (pl.col("A_L") * pl.col("delay_d")).cast(pl.Float32).alias("ALxdelay"),
        (pl.col("A_R") * pl.col("delay_d")).cast(pl.Float32).alias("ARxdelay"),
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
