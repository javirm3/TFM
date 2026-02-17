import numpy as np
import polars as pl
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, List

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


def build_sequence_from_df(df_sub: pl.DataFrame):
    df_sub = df_sub.sort("trial_idx")
    df_sub = df_sub.with_columns([
        pl.col("response").cast(pl.Int32),

        (pl.col("x_c") == "L").cast(pl.Float32).alias("biasL"),
        (pl.col("x_c") == "C").cast(pl.Float32).alias("biasC"),
        (pl.col("x_c") == "R").cast(pl.Float32).alias("biasR"),
        
        pl.lit(1.0).cast(pl.Float32).alias("bias"),
        
        pl.col("delay_d").cast(pl.Float32).alias("delay"),
        pl.col("stim_d").cast(pl.Float32).alias("stim"),

        ((pl.col("x_c") == "L") * pl.col("stim_d")).cast(pl.Float32).alias("SL"),
        ((pl.col("x_c") == "C") * pl.col("stim_d")).cast(pl.Float32).alias("SC"),
        ((pl.col("x_c") == "R") * pl.col("stim_d")).cast(pl.Float32).alias("SR"),

        pl.col("performance").shift(1).fill_null(0).cast(pl.Float32).alias("previous_outcome"),
    ])

    y = df_sub["response"].to_numpy()

    A = jnp.asarray(action_trace(jnp.asarray(y), tau=50)).astype(np.float32)  # (T,3)
    mean_A = jnp.mean(A, axis=0, keepdims=True)
    std_A = jnp.std(A, axis=0, keepdims=True)
    std_A = jnp.where(std_A < 1e-6, 1.0, std_A)  
    A = (A - mean_A) / std_A
    X = df_sub.select(["bias", "biasL", "biasC", "biasR", "delay", "SL", "SC", "SR", "previous_outcome"]).to_numpy().astype(jnp.float32)
    X = jnp.concatenate([X, A], axis=1)
    U = jnp.column_stack([jnp.ones(len(y), dtype=jnp.float32), A[:,0], A[:,1], A[:,2], jnp.asarray(df_sub.select("previous_outcome"))]).astype(jnp.float32)
    
    X_cols = ["bias","biasL","biasC","biasR","delay","SL","SC","SR","previous_outcome"]
    names = {
        "X_cols": X_cols + ["A_L","A_C","A_R"],
        "U_cols": ["1","A_L","A_C","A_R","previous_outcome"],
    }
    return jnp.asarray(y), jnp.asarray(X), jnp.asarray(U), names    
