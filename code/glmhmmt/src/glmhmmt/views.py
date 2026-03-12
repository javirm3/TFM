"""views.py — SubjectFitView dataclass and pipeline-wide constants.

This module is the **single source of truth** for:

* :class:`SubjectFitView` — a task-agnostic, per-subject representation of a
  fitted GLM-HMM model.
* :data:`_LABEL_RANK` — canonical mapping from semantic state label to integer
  rank (Engaged = 0, Disengaged variants = 1, 2, …).
* :data:`_STATE_HEX` — palette of rank-indexed hex colours loaded from
  ``config.toml``; every plotting function uses this for consistent colouring.
* :func:`build_views` — factory that combines ``arrays_store`` with a
  :class:`~tasks.TaskAdapter` to produce a ``{subject: SubjectFitView}`` dict.

Design contract
---------------
State labelling is **not** performed here.  The labelling strategy (e.g.
engagement scoring) lives on the task adapter and is applied when building a
view via :func:`build_views`, so this module stays task-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import tomllib

import numpy as np

import paths

# ── palette & rank constants ──────────────────────────────────────────────────

with paths.CONFIG.open("rb") as _f:
    _cfg = tomllib.load(_f)

_STATE_HEX: list[str] = _cfg.get("palettes", {}).get("states_hex", [
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02",
])

_LABEL_RANK: dict[str, int] = {
    "Engaged": 0,
    "Disengaged": 1,
    "Biased L": 1,
    "Biased R": 2,
    "Disengaged L": 1,
    "Disengaged R": 2,
    "Disengaged C": 3,
    **{f"Disengaged {i}": i for i in range(1, 10)},
}


# ── dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class SubjectFitView:
    """Task-agnostic view of a fitted GLM-HMM model for one subject.

    All arrays are plain NumPy (never JAX) so they can be pickled, serialised,
    and used in plotting without any JAX dependency.

    Parameters
    ----------
    subject :
        Subject identifier string.
    K :
        Number of hidden states.
    smoothed_probs :
        HMM posterior  γ(z_t = k | y_{1:T}),  shape ``(T, K)``.
        **This is the canonical quantity for posterior plots.**  It is saved
        directly from ``model.smoother_multisession`` and loaded verbatim —
        never recomputed.
    emission_weights :
        Softmax regression weight tensor,  shape ``(K, C-1, F)``.
    X :
        Emission design matrix,  shape ``(T, F)``.
    y :
        Integer choices,  shape ``(T,)``.
    feat_names :
        Emission feature names, length ``F``.
    state_name_by_idx :
        Mapping ``{state_int_k: label_str}`` — e.g.
        ``{0: "Engaged", 1: "Disengaged"}``.
    state_idx_order :
        State indices sorted by engagement rank (Engaged first).
    state_rank_by_idx :
        Mapping ``{state_int_k: rank_int}`` where rank 0 = Engaged.

    Optional
    --------
    p_pred :
        Marginal model predictions shape ``(T, C)``.  Pre-computed during
        fitting; if absent, :func:`~glmhmmt.postprocess.build_trial_df` will
        recompute it.
    transition_weights :
        Input-driven transition weight tensor, shape ``(K, K, D)`` —
        GLM-HMM-t only.
    U :
        Transition design matrix, shape ``(T, D)`` — GLM-HMM-t only.
    U_cols :
        Transition feature names, length ``D``.
    """

    subject: str
    K: int
    smoothed_probs: np.ndarray         # (T, K)
    emission_weights: np.ndarray       # (K, C-1, F)
    X: np.ndarray                      # (T, F)
    y: np.ndarray                      # (T,)
    feat_names: list[str]
    state_name_by_idx: dict[int, str]
    state_idx_order: list[int]
    state_rank_by_idx: dict[int, int]

    # optional
    p_pred: Optional[np.ndarray] = None             # (T, C)
    transition_weights: Optional[np.ndarray] = None  # (K, K, D)
    U: Optional[np.ndarray] = None                  # (T, D)
    U_cols: list[str] = field(default_factory=list)

    # ── derived helpers ───────────────────────────────────────────────────────

    @property
    def T(self) -> int:
        """Number of trials."""
        return int(self.smoothed_probs.shape[0])

    @property
    def num_classes(self) -> int:
        """Number of choice classes (C)."""
        return self.emission_weights.shape[1] + 1  # C-1 rows + reference class

    def map_states(self) -> np.ndarray:
        """Return MAP state assignment ``(T,)`` = argmax(smoothed_probs, axis=1)."""
        return np.argmax(self.smoothed_probs, axis=1).astype(int)

    def engaged_k(self) -> int:
        """Index of the Engaged state (rank 0)."""
        for k, rank in self.state_rank_by_idx.items():
            if rank == 0:
                return k
        return self.state_idx_order[0]


# ── factory ───────────────────────────────────────────────────────────────────

def build_views(
    arrays_store: dict,
    adapter,
    K: int,
    subjects: list[str],
) -> dict[str, SubjectFitView]:
    """Build a ``{subject: SubjectFitView}`` dict from *arrays_store*.

    Parameters
    ----------
    arrays_store :
        Dict ``{subj: npz-dict}`` as loaded by the notebook I/O cell.
        Each entry must contain at least ``smoothed_probs``, ``emission_weights``,
        ``X``, ``y``, and ``X_cols``.
    adapter :
        Any :class:`~tasks.TaskAdapter` instance.  Used to apply the
        task-specific labelling strategy (``adapter.label_states``).
    K :
        Number of states.
    subjects :
        Subject IDs to include (those absent from *arrays_store* are silently
        skipped).

    Returns
    -------
    dict[str, SubjectFitView]
    """
    _selected = [s for s in subjects if s in arrays_store]
    if not _selected:
        return {}

    # State labelling is task-specific and performed here, outside the dataclass.
    # names dict is populated from the first available subject's X_cols.
    _names: dict = {}
    for s in _selected:
        _cols = arrays_store[s].get("X_cols")
        if _cols is not None:
            _names = {"X_cols": list(_cols)}
            break

    state_labels, state_order = adapter.label_states(
        arrays_store, _names, K, _selected
    )

    views: dict[str, SubjectFitView] = {}
    for subj in _selected:
        d = arrays_store[subj]
        feat_names = list(d.get("X_cols", []))
        slbls  = state_labels.get(subj, {k: f"State {k}" for k in range(K)})
        sorder = state_order.get(subj, list(range(K)))

        # rank map: ordered position in sorder → LABEL_RANK (semantic)
        srank: dict[int, int] = {}
        for k in range(K):
            lbl  = slbls.get(k, "")
            srank[k] = _LABEL_RANK.get(lbl, sorder.index(k) if k in sorder else k)

        views[subj] = SubjectFitView(
            subject=subj,
            K=K,
            smoothed_probs=np.asarray(d["smoothed_probs"]),
            emission_weights=np.asarray(d["emission_weights"]),
            X=np.asarray(d["X"]),
            y=np.asarray(d["y"]),
            feat_names=feat_names,
            state_name_by_idx={int(k): v for k, v in slbls.items()},
            state_idx_order=[int(k) for k in sorder],
            state_rank_by_idx={int(k): v for k, v in srank.items()},
            p_pred=np.asarray(d["p_pred"]) if "p_pred" in d else None,
            transition_weights=(
                np.asarray(d["transition_weights"])
                if "transition_weights" in d else None
            ),
            U=np.asarray(d["U"]) if "U" in d else None,
            U_cols=list(d.get("U_cols", [])),
        )

    return views
