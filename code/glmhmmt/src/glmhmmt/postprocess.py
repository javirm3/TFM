"""postprocess.py — Standard trial-level DataFrame builders.

All DataFrames use **Polars**.  These are the canonical intermediate
representations consumed by all plotting functions.

Key invariant
-------------
``p_state_k`` columns in :func:`build_trial_df` output come **directly** from
``view.smoothed_probs[:, k]`` (the HMM forward-backward posterior), **never**
from a re-application of the emission model.  This guarantees that posterior
plots show the correct, temporally-smooth HMM quantities.

``p_model_correct`` (MAP-state emission) and ``p_model_correct_marginal``
(marginal across states) are computed from the emission weights and are used
only for accuracy analyses — not for posterior visualisation.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from glmhmmt.views import SubjectFitView, _LABEL_RANK


# ── helpers ────────────────────────────────────────────────────────────────────

def _emission_probs(
    W: np.ndarray,        # (K, C-1, F)
    X: np.ndarray,        # (T, F)
    map_k: np.ndarray,    # (T,) int — MAP state per trial
    C: int,
) -> np.ndarray:
    """Return MAP-state emission probabilities: softmax(W[map_k[t]] @ X[t]).

    Returns
    -------
    p_map : (T, C)
    """
    W_map = W[map_k]                                   # (T, C-1, F)
    logits_ce = np.einsum("tcf,tf->tc", W_map, X)     # (T, C-1)
    logits_map = _insert_reference(logits_ce, C)       # (T, C)
    return _stable_softmax(logits_map)                 # (T, C)


def _insert_reference(logits_ce: np.ndarray, C: int) -> np.ndarray:
    """Insert the reference class (logit = 0) into a (*, C-1) array → (*, C).

    Convention
    ----------
    The model always appends the reference class last
    (SoftmaxGLMHMMEmissions: ``logits = [eta..., 0]``).

    * C == 2 (binary): logits_ce = [logit_L]           →  [logit_L, 0]
    * C == 3 (3-AFC):  logits_ce = [logit_L, logit_C]  →  [logit_L, logit_C, 0_R]
    """
    shape_prefix = logits_ce.shape[:-1]
    return np.concatenate(
        [logits_ce, np.zeros((*shape_prefix, 1))], axis=-1
    )


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax, input shape ``(N, C)``."""
    lse = logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits - lse)
    return exp / exp.sum(axis=-1, keepdims=True)


# ── public builders ────────────────────────────────────────────────────────────

def build_trial_df(
    view: SubjectFitView,
    df_behavioral: pl.DataFrame,
    behavioral_cols: dict[str, str],
) -> pl.DataFrame:
    """Build a trial-level Polars DataFrame for one subject.

    The caller must ensure that *df_behavioral* has been:

    * filtered to this subject only,
    * sorted by trial index (``sort_col``),
    * filtered to sessions with ≥ 2 trials (same mask as used during fitting).

    This guarantees ``df_behavioral.height == view.T`` and row ``i`` of the
    behavioral DataFrame aligns exactly with row ``i`` of ``view.smoothed_probs``.

    Parameters
    ----------
    view :
        :class:`~glmhmmt.views.SubjectFitView` for this subject.
    df_behavioral :
        Pre-filtered behavioral DataFrame (see constraints above).
    behavioral_cols :
        Mapping from canonical column name → actual column name in
        *df_behavioral*.  Required keys: ``"trial_idx"``, ``"session"``,
        ``"stimulus"``, ``"response"``, ``"performance"``.  All other columns
        in *df_behavioral* are preserved as-is (e.g. ``stimd_n``, ``ttype_n``).

    Returns
    -------
    pl.DataFrame with columns:

    * All columns from *df_behavioral* (standard ones renamed to canonical
      names).
    * ``subject``
    * ``p_state_0`` … ``p_state_{K-1}``  — HMM posterior (direct copy)
    * ``state_idx``, ``state_rank``, ``state_label``  — MAP assignment
    * ``pL`` [, ``pC``], ``pR``  — marginal class probabilities from p_pred
    * ``p_model_correct``  — MAP-state emission P(correct class)
    * ``p_model_correct_marginal``  — marginal P(correct class)
    * ``correct_bool``
    """
    T = view.T
    if df_behavioral.height != T:
        raise ValueError(
            f"Subject {view.subject!r}: df_behavioral has {df_behavioral.height} "
            f"rows but view.smoothed_probs has T={T}. "
            "Ensure the same session-length filter was applied to both."
        )

    # ── rename standard behavioral columns to canonical names ─────────────────
    rename_map: dict[str, str] = {}
    for canonical, actual in behavioral_cols.items():
        if actual in df_behavioral.columns and actual != canonical:
            rename_map[actual] = canonical
    if rename_map:
        df_out = df_behavioral.rename(rename_map)
    else:
        df_out = df_behavioral.clone()

    # ── p_state_k  — HMM posterior, direct copy (NEVER recomputed) ───────────
    posterior_series = [
        pl.Series(f"p_state_{k}", view.smoothed_probs[:, k].astype(np.float64))
        for k in range(view.K)
    ]

    # ── MAP state assignment ───────────────────────────────────────────────────
    map_k            = view.map_states()                              # (T,) int
    state_rank_arr   = np.array(
        [view.state_rank_by_idx.get(int(ki), ki) for ki in map_k], dtype=np.int32
    )
    state_label_arr  = np.array(
        [view.state_name_by_idx.get(int(ki), f"State {ki}") for ki in map_k]
    )

    # ── MAP-state emission probabilities ──────────────────────────────────────
    C = view.num_classes
    p_map = _emission_probs(view.emission_weights, view.X, map_k, C)  # (T, C)

    stim_col = behavioral_cols.get("stimulus", "stimulus")
    print(f"Stimulus column for subject {view.subject!r}: {stim_col!r}")
    if stim_col in df_out.columns:
        print(f"Using stimulus column {stim_col!r} for subject {view.subject!r}")
        stimulus_vals = df_out[stim_col].to_numpy().astype(int)
    else:
        # fallback: use the canonical name (after rename)
        stimulus_vals = df_out["stimulus"].to_numpy().astype(int)
    p_model_correct_map = p_map[np.arange(T), stimulus_vals]

    # ── marginal class probabilities (from view.p_pred if available) ──────────
    if view.p_pred is not None:
        p_marginal = np.asarray(view.p_pred)   # (T, C)
    else:
        # Recompute marginal via weighted sum: Σ_k γ(z_t=k) * p(y_t | z_t=k, x_t)
        W, X = view.emission_weights, view.X
        logits_all_ce = np.einsum("kcf,tf->tkc", W, X)       # (T, K, C-1)
        Tk = T * view.K
        logits_flat = _insert_reference(logits_all_ce.reshape(Tk, C - 1), C)
        p_per_k = _stable_softmax(logits_flat).reshape(T, view.K, C)  # (T, K, C)
        p_marginal = np.einsum("tk,tkc->tc", view.smoothed_probs, p_per_k)

    p_marginal_correct = p_marginal[np.arange(T), stimulus_vals]

    # ── assemble all new columns ───────────────────────────────────────────────
    new_cols = [
        pl.Series("subject",                   [view.subject] * T),
        *posterior_series,
        pl.Series("state_idx",                 map_k.astype(np.int32)),
        pl.Series("state_rank",                state_rank_arr),
        pl.Series("state_label",               state_label_arr),
        pl.Series("p_model_correct",           p_model_correct_map.astype(np.float64)),
        pl.Series("p_model_correct_marginal",  p_marginal_correct.astype(np.float64)),
    ]

    if C == 2:
        new_cols += [
            pl.Series("pL", p_marginal[:, 0].astype(np.float64)),
            pl.Series("pR", p_marginal[:, 1].astype(np.float64)),
        ]
    else:
        new_cols += [
            pl.Series("pL", p_marginal[:, 0].astype(np.float64)),
            pl.Series("pC", p_marginal[:, 1].astype(np.float64)),
            pl.Series("pR", p_marginal[:, 2].astype(np.float64)),
        ]

    # overwrite/add; drop pre-existing computed cols (pL/pC/pR, subject)
    _computed_names = {s.name for s in new_cols}
    df_out = df_out.select(
        [c for c in df_out.columns if c not in _computed_names]
    )
    df_out = df_out.with_columns(new_cols)

    # ── correct_bool from performance ─────────────────────────────────────────
    if "performance" in df_out.columns and "correct_bool" not in df_out.columns:
        df_out = df_out.with_columns(
            pl.col("performance").cast(pl.Boolean).alias("correct_bool")
        )

    return df_out


def build_emission_weights_df(views: dict[str, SubjectFitView]) -> pl.DataFrame:
    """Long-format Polars DataFrame of all subjects' emission weights.

    Columns
    -------
    subject, state_idx, state_label, state_rank, class_idx, feature, weight
    """
    records: list[dict] = []
    for subj, view in views.items():
        W = view.emission_weights   # (K, C-1, F)
        for k in range(view.K):
            lbl  = view.state_name_by_idx.get(k, f"State {k}")
            rank = view.state_rank_by_idx.get(k, k)
            for c in range(W.shape[1]):
                for fi, fname in enumerate(view.feat_names):
                    records.append({
                        "subject":     subj,
                        "state_idx":   k,
                        "state_label": lbl,
                        "state_rank":  rank,
                        "class_idx":   c,
                        "feature":     fname,
                        "weight":      float(W[k, c, fi]),
                    })
    if not records:
        return pl.DataFrame()
    return pl.DataFrame(records)


def build_posterior_df(views: dict[str, SubjectFitView]) -> pl.DataFrame:
    """Long-format Polars DataFrame of HMM posterior probabilities.

    The ``probability`` column contains the raw HMM posterior
    γ(z_t = k | y_{1:T}) — the **same** values as ``smoothed_probs``.

    Columns
    -------
    subject, trial_idx, state_idx, state_label, state_rank, probability
    """
    frames: list[pl.DataFrame] = []
    for subj, view in views.items():
        T = view.T
        for k in range(view.K):
            lbl  = view.state_name_by_idx.get(k, f"State {k}")
            rank = view.state_rank_by_idx.get(k, k)
            frames.append(pl.DataFrame({
                "subject":     [subj] * T,
                "trial_idx":   list(range(T)),
                "state_idx":   [k] * T,
                "state_label": [lbl] * T,
                "state_rank":  [rank] * T,
                "probability": view.smoothed_probs[:, k].tolist(),
            }))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)
