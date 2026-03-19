import hashlib
import json
import numpy as np

from glmhmmt.model import serialize_frozen_emissions


def valid_trial_mask(session_ids: np.ndarray, min_length: int = 2) -> np.ndarray:
    """Return a boolean mask keeping only trials from sessions with >= min_length trials."""
    ids, counts = np.unique(session_ids, return_counts=True)
    keep = set(ids[counts >= min_length])
    return np.array([session_id in keep for session_id in session_ids])


def stable_model_id(
    task: str,
    K: int,
    tau: float,
    emission_cols: list | None = None,
    transition_cols: list | None = None,
    frozen_emissions: dict | None = None,
) -> str:
    """Stable 8-char MD5 hash over the fit-defining model configuration."""
    config = {
        "task": task,
        "K": int(K),
        "tau": float(tau),
        "emission_cols": sorted(emission_cols) if emission_cols else [],
        "transition_cols": sorted(transition_cols) if transition_cols else [],
        "frozen_emissions": serialize_frozen_emissions(frozen_emissions),
    }
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
