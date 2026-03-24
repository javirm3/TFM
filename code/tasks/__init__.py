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
    task_key: str = NotImplemented      # canonical UI / CLI task name
    task_label: str = NotImplemented    # human-readable task label
    num_classes: int = NotImplemented   # 2 or 3
    data_file: str = NotImplemented     # filename under paths.DATA_PATH
    sort_col: str = NotImplemented      # trial ordering column
    session_col: str = NotImplemented   # session identifier column

    # ── data preparation ────────────────────────────────────────────────────

    @abstractmethod
    def subject_filter(self, df: Any) -> Any:
        """Apply task-specific subject/session filtering to the full DataFrame."""

    @abstractmethod
    def build_feature_df(self, df_sub: Any, tau: float = 50.0) -> Any:
        """Return a task-owned trial dataframe with design-matrix columns.

        This dataframe is the explicit contract between each task and the
        shared fitting code. It should contain the task's raw behavioral
        columns plus any derived emission / transition regressors needed to
        assemble ``X`` and ``U``.
        """

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

    @abstractmethod
    def build_design_matrices(
        self,
        feature_df: Any,
        emission_cols: List[str] | None = None,
        transition_cols: List[str] | None = None,
    ) -> Tuple[Any, Any, Any, Dict]:
        """Return ``(y, X, U, names)`` from a task-owned feature dataframe."""

    # ── column defaults  ────────────────────────────────────────────────────

    @abstractmethod
    def default_emission_cols(self) -> List[str]:
        """Ordered list of emission regressor names for UI initialisation."""

    @abstractmethod
    def default_transition_cols(self) -> List[str]:
        """Ordered list of transition regressor names for UI initialisation."""

    def available_emission_cols(self) -> List[str]:
        """Ordered list of selectable emission regressors."""
        return self.default_emission_cols()

    def available_transition_cols(self) -> List[str]:
        """Ordered list of selectable transition regressors."""
        return self.default_transition_cols()

    def sf_cols(self, df: Any) -> List[str]:
        """Optional dynamic stimulus-frame columns for binary tasks."""
        return []

    def cv_balance_labels(self, feature_df: Any):
        """Return per-trial labels used for CV balancing, or ``None`` if unsupported."""
        return None

    # ── plot module ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_plots(self) -> types.ModuleType:
        """Return the task-specific plots module."""

    @property
    def choice_labels(self) -> list[str]:
        """Ordered human-readable labels for choice classes."""
        return [f"Choice {idx}" for idx in range(self.num_classes)]

    @property
    def probability_columns(self) -> list[str]:
        """Trial-level probability column names aligned with choice_labels."""
        return [f"p_{idx}" for idx in range(self.num_classes)]

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

    def get_correct_class(self, df: pl.DataFrame) -> np.ndarray:
        """Return correct class index per trial as int array of shape (T,).

        Must return values in {0, ..., C-1}. Invalid/ambiguous trials may be -1.
        """
        raise NotImplementedError


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
        ``"nuo_auditory"``,
        ``"auditory_2afc"``              → NuoAuditoryAdapter
    """
    key = task.lower().replace("-", "_")
    cls = _REGISTRY.get(key)
    if cls is None:
        known = ", ".join(f'"{k}"' for k in _REGISTRY)
        raise ValueError(f"Unknown task {task!r}. Known tasks: {known}")
    return cls()


def get_task_options() -> list[dict[str, str]]:
    """Return unique task options for UI selectors.

    Each option is ``{"value": <task_key>, "label": <task_label>}``.
    """
    seen: dict[str, str] = {}
    for cls in dict.fromkeys(_REGISTRY.values()):
        task_key = getattr(cls, "task_key", None)
        task_label = getattr(cls, "task_label", None)
        if task_key and task_label and task_key not in seen:
            seen[task_key] = task_label
    return [{"value": key, "label": seen[key]} for key in seen]


# Import adapters so they self-register via @_register.
# Done at the bottom to avoid circular imports.
from tasks import mcdr as _mcdr_mod       # noqa: E402, F401
from tasks import two_afc as _two_afc_mod # noqa: E402, F401
from tasks import nuo_auditory as _nuo_auditory_mod  # noqa: E402, F401
