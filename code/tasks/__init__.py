"""Task adapters for the GLM-HMM pipeline.

Each adapter encapsulates all task-specific knowledge so that fit scripts
and analysis notebooks can be written once and work for any task.

Usage
-----
    from tasks import get_adapter

    adapter = get_adapter("mcdr")   # or "two_afc" / "2AFC" / "MCDR"
    df      = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df      = adapter.subject_filter(df)
    y, X, U, names = adapter.load_subject(df_sub, tau=50.0)
    plots   = adapter.get_plots()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

import types


class TaskAdapter(ABC):
    """Abstract base for task-specific configuration & data loading."""

    # ── class-level attributes (override in subclass) ──────────────────────
    num_classes: int = NotImplemented   # 2 or 3
    data_file: str = NotImplemented     # filename under paths.DATA_PATH
    sort_col: str = NotImplemented      # trial ordering column
    session_col: str = NotImplemented   # session identifier column

    # ── data preparation ────────────────────────────────────────────────────

    @abstractmethod
    def subject_filter(self, df: Any) -> Any:
        """Apply task-specific subject/session filtering to the full DataFrame."""

    @abstractmethod
    def load_subject(
        self,
        df_sub: Any,
        tau: float = 50.0,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` for one subject's DataFrame slice.

        ``U`` must always be returned (use an empty array for tasks that lack
        transition features — shape ``(T, 0)``).
        ``names`` must contain ``"X_cols"`` and ``"U_cols"``.
        """

    # ── column defaults  ────────────────────────────────────────────────────

    @abstractmethod
    def default_emission_cols(self) -> List[str]:
        """Ordered list of emission regressor names for UI initialisation."""

    @abstractmethod
    def default_transition_cols(self) -> List[str]:
        """Ordered list of transition regressor names for UI initialisation."""

    # ── plot module ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_plots(self) -> types.ModuleType:
        """Return the task-specific plots module."""

    # ── column mapping  ──────────────────────────────────────────────────────

    @property
    @abstractmethod
    def behavioral_cols(self) -> Dict[str, str]:
        """Mapping from canonical column names to actual column names.

        Required canonical keys and their semantics:
            ``"trial_idx"``   — global, monotonically increasing trial index
            ``"trial"``       — within-session trial number (may equal trial_idx)
            ``"session"``     — session identifier
            ``"stimulus"``    — integer correct-class index (0/1/2 for L/C/R)
            ``"response"``    — integer chosen class
            ``"performance"`` — 0/1 trial outcome
        """

    # ── state labelling ──────────────────────────────────────────────────────

    @abstractmethod
    def label_states(
        self,
        arrays_store: dict,
        names: dict,
        K: int,
        subjects: list,
    ) -> tuple:
        """Return ``(state_labels, state_order)`` for all subjects.

        state_labels : {subj: {state_idx: label_str}}
        state_order  : {subj: [state_idx, ...]}  sorted by engagement rank desc
        """
        ...


# ── registry & factory ─────────────────────────────────────────────────────

_REGISTRY: dict[str, type[TaskAdapter]] = {}

def _register(keys: list[str]):
    """Class decorator that registers an adapter under one or more keys."""
    def decorator(cls: type[TaskAdapter]) -> type[TaskAdapter]:
        for k in keys:
            _REGISTRY[k.lower()] = cls
        return cls
    return decorator


def get_adapter(task: str) -> TaskAdapter:
    """Return an instantiated TaskAdapter for *task*.

    Accepted values (case-insensitive):
        ``"mcdr"``, ``"MCDR"``          → MCDRAdapter
        ``"two_afc"``, ``"2afc"``,
        ``"2AFC"``, ``"two_AFC"``        → TwoAFCAdapter
    """
    key = task.lower().replace("-", "_")
    cls = _REGISTRY.get(key)
    if cls is None:
        known = ", ".join(f'"{k}"' for k in _REGISTRY)
        raise ValueError(f"Unknown task {task!r}. Known tasks: {known}")
    return cls()


# Import adapters so they self-register via @_register.
# Done at the bottom to avoid circular imports.
from tasks import mcdr as _mcdr_mod       # noqa: E402, F401
from tasks import two_afc as _two_afc_mod # noqa: E402, F401
