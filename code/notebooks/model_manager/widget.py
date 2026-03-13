"""ModelManagerWidget — anywidget-backed UI for selecting and loading GLM-HMM fits.

Python layer handles all semantics: regressor grouping, named-model detection,
fitted-subject counting, and config-to-state application.  The JS layer
(widget.js) handles rendering and user events only.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import anywidget
import polars as pl
import traitlets

# Make sure sibling packages (paths, tasks) are importable from notebooks/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import paths
from tasks import get_adapter

# ── Regressor group registries ───────────────────────────────────────────────
# Shape: list of {"key": str, "label": str, "members": {"L"|"C"|"R"|"N": col}}
# JS renders: Label | L | N/C | R  columns.

_MCDR_EMISSION_GROUPS: list[dict] = [
    {"key": "bias",      "label": "bias",       "members": {"N": "bias"}},
    {"key": "bias_side", "label": "bias side",  "members": {"L": "biasL",    "C": "biasC",    "R": "biasR"}},
    {"key": "onset",     "label": "onset",      "members": {"L": "onsetL",   "C": "onsetC",   "R": "onsetR"}},
    {"key": "delay",     "label": "delay",      "members": {"N": "delay"}},
    {"key": "S",         "label": "S",          "members": {"L": "SL",       "C": "SC",       "R": "SR"}},
    {"key": "SxDelay",   "label": "S×delay",    "members": {"L": "SLxdelay", "C": "SCxdelay", "R": "SRxdelay"}},
    {"key": "D",         "label": "D (type)",   "members": {"N": "D"}},
    {"key": "D_side",    "label": "D side",     "members": {"L": "DL",       "C": "DC",       "R": "DR"}},
    {"key": "A",         "label": "A (action)", "members": {"L": "A_L",      "C": "A_C",      "R": "A_R"}},
    {"key": "stim1",     "label": "stim 1",     "members": {"L": "stim1L",   "C": "stim1C",   "R": "stim1R"}},
    {"key": "stim2",     "label": "stim 2",     "members": {"L": "stim2L",   "C": "stim2C",   "R": "stim2R"}},
    {"key": "stim3",     "label": "stim 3",     "members": {"L": "stim3L",   "C": "stim3C",   "R": "stim3R"}},
    {"key": "stim4",     "label": "stim 4",     "members": {"L": "stim4L",   "C": "stim4C",   "R": "stim4R"}},
    {"key": "speed1",    "label": "speed 1",    "members": {"N": "speed1"}},
    {"key": "speed2",    "label": "speed 2",    "members": {"N": "speed2"}},
    {"key": "speed3",    "label": "speed 3",    "members": {"N": "speed3"}},
]

_2AFC_EMISSION_GROUPS: list[dict] = [
    {"key": "bias",          "label": "bias",            "members": {"N": "bias"}},
    {"key": "stim_vals",     "label": "stim vals",       "members": {"N": "stim_vals"}},
    {"key": "stim_strength", "label": "stim strength",   "members": {"N": "stim_strength"}},
    {"key": "at_choice",     "label": "action (choice)", "members": {"N": "at_choice"}},
    {"key": "at_error",      "label": "action (error)",  "members": {"N": "at_error"}},
    {"key": "at_correct",    "label": "action (correct)","members": {"N": "at_correct"}},
    {"key": "prev_choice",   "label": "prev choice",     "members": {"N": "prev_choice"}},
    {"key": "wsls",          "label": "WSLS",            "members": {"N": "wsls"}},
]

_MCDR_TRANSITION_GROUPS: list[dict] = [
    {"key": "A_plus",  "label": "A+",         "members": {"N": "A_plus"}},
    {"key": "A_minus", "label": "A−",         "members": {"N": "A_minus"}},
    {"key": "A_trans", "label": "A (action)", "members": {"L": "A_L", "C": "A_C", "R": "A_R"}},
]

# ── Private helpers ───────────────────────────────────────────────────────────

_HASH_RE   = re.compile(r"^[0-9a-f]{8}$")
_ARRAYS_RE = re.compile(r"^(.+?)_(glm|glmhmm|glmhmmt)_arrays\.npz$")


def _build_regressor_groups(available_cols: list[str], registry: list[dict]) -> list[dict]:
    """Filter *registry* to columns present in *available_cols*.

    Columns not covered by any registry entry are appended as solo N-only groups
    so they're always accessible (e.g., dynamic sf_* columns in 2AFC).
    """
    available = set(available_cols)
    registered: set[str] = set()
    result: list[dict] = []

    for group in registry:
        filtered = {k: v for k, v in group["members"].items() if v in available}
        if filtered:
            result.append({**group, "members": filtered})
            registered.update(filtered.values())

    # Append any column not yet covered as a singleton group
    for col in available_cols:
        if col not in registered:
            result.append({"key": col, "label": col, "members": {"N": col}})

    return result


def _count_fitted_subjects(model_dir: Path) -> int:
    """Count unique subjects from *_arrays.npz files in *model_dir*.

    Expected filename format (new, without _K{N}): {subject}_{modelname}_arrays.npz
    Falls back gracefully to 0 for old _K{N}-format files.
    """
    subjects: set[str] = set()
    for f in model_dir.glob("*_arrays.npz"):
        m = _ARRAYS_RE.match(f.name)
        if m:
            subjects.add(m.group(1))
    return len(subjects)


def _is_displayable(cfg: dict) -> bool:
    """Return True if *cfg* represents a human-named model.

    A model is displayable when either:
    - it has a non-empty ``alias`` field, or
    - its ``model_id`` does not look like an 8-character hex hash.
    """
    alias = cfg.get("alias", "")
    if alias and alias.strip():
        return True
    model_id = cfg.get("model_id", "")
    return bool(model_id) and not _HASH_RE.match(model_id)


def _get_display_name(cfg: dict) -> str:
    """Return the user-facing display name for a model config."""
    alias = cfg.get("alias", "")
    if alias and alias.strip():
        return alias.strip()
    return cfg.get("model_id", "")


def _get_K_from_config(cfg: dict) -> Any:
    """Extract a concise K representation from a config dict."""
    if cfg.get("K"):
        return cfg["K"]
    if "K_list" in cfg and cfg["K_list"]:
        klist = cfg["K_list"]
        return klist[0] if len(klist) == 1 else f"{klist[0]}–{klist[-1]}"
    return "?"


# ── Widget ────────────────────────────────────────────────────────────────────

class ModelManagerWidget(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "widget.js"
    _css = Path(__file__).parent / "widget.css"

    # ── traitlets ─────────────────────────────────────────────────────────────
    ui_mode    = traitlets.Unicode("new").tag(sync=True)      # "new" | "load"
    model_type = traitlets.Unicode("glmhmm").tag(sync=True)  # "glm" | "glmhmm" | "glmhmmt"
    task       = traitlets.Unicode("MCDR").tag(sync=True)
    is_2afc    = traitlets.Bool(False).tag(sync=True)

    existing_models      = traitlets.List(traitlets.Unicode()).tag(sync=True)
    existing_models_info = traitlets.List(traitlets.Dict()).tag(sync=True)
    existing_model       = traitlets.Unicode("").tag(sync=True)

    alias = traitlets.Unicode("").tag(sync=True)

    subjects_list = traitlets.List(traitlets.Unicode()).tag(sync=True)
    subjects      = traitlets.List(traitlets.Unicode()).tag(sync=True)

    k_options = traitlets.List(traitlets.Int(), default_value=[2, 3, 4, 5, 6]).tag(sync=True)
    K         = traitlets.Int(2).tag(sync=True)

    tau       = traitlets.Int(50).tag(sync=True)

    lapse     = traitlets.Bool(False).tag(sync=True)
    lapse_max = traitlets.Float(0.2).tag(sync=True)

    emission_cols_options = traitlets.List(traitlets.Unicode()).tag(sync=True)
    emission_cols         = traitlets.List(traitlets.Unicode()).tag(sync=True)
    emission_groups       = traitlets.List(traitlets.Dict()).tag(sync=True)

    transition_cols_options = traitlets.List(traitlets.Unicode()).tag(sync=True)
    transition_cols         = traitlets.List(traitlets.Unicode()).tag(sync=True)
    transition_groups       = traitlets.List(traitlets.Dict()).tag(sync=True)

    run_fit_clicks   = traitlets.Int(0).tag(sync=True)
    save_alias_clicks = traitlets.Int(0).tag(sync=True)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_options()

    # ── observers ─────────────────────────────────────────────────────────────

    @traitlets.observe("task")
    def _on_task_change(self, change):
        # Different task → different column set; clear selections so defaults repopulate
        self.emission_cols = []
        self.transition_cols = []
        self._update_options()

    @traitlets.observe("model_type")
    def _on_model_type_change(self, change):
        # Same task, different model type; regressors are shared but refresh groups
        self._update_options()

    @traitlets.observe("existing_model")
    def _on_existing_model_change(self, change):
        selected = change["new"]
        if not selected:
            return
        if selected == "__default__":
            self._apply_default_state()
            return
        # Find the real folder whose display name matches *selected*
        fits_path = paths.RESULTS / "fits" / self.task / self.model_type
        if not fits_path.exists():
            return
        for d in fits_path.iterdir():
            if not (d.is_dir() and (d / "config.json").exists()):
                continue
            try:
                cfg = json.loads((d / "config.json").read_text())
            except Exception:
                continue
            if _is_displayable(cfg) and _get_display_name(cfg) == selected:
                self._apply_config_to_state(cfg)
                return

    # ── helpers ───────────────────────────────────────────────────────────────

    def _apply_config_to_state(self, cfg: dict) -> None:
        """Write a config dict's values into the widget traitlets."""
        if "emission_cols" in cfg:
            self.emission_cols = cfg["emission_cols"]
        if "transition_cols" in cfg:
            self.transition_cols = cfg["transition_cols"]
        if "subjects" in cfg:
            self.subjects = cfg["subjects"]
        if "tau" in cfg:
            self.tau = int(cfg["tau"])
        k = _get_K_from_config(cfg)
        if isinstance(k, int):
            self.K = k
        if "lapse" in cfg:
            self.lapse = bool(cfg["lapse"])
        if "lapse_max" in cfg:
            self.lapse_max = float(cfg["lapse_max"])
        self._refresh_groups()

    def _apply_default_state(self) -> None:
        """Reset widget traitlets to the adapter's default configuration."""
        try:
            adapter = get_adapter(self.task)
            df_all  = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
            df_all  = adapter.subject_filter(df_all)
            subjects = df_all["subject"].unique().to_list()
            self.subjects  = subjects
            self.K         = 2
            self.tau       = 50
            self.lapse     = False
            self.lapse_max = 0.2
            is_2afc = adapter.num_classes == 2
            ecols = (
                adapter.default_emission_cols() + adapter.sf_cols(df_all)
                if is_2afc else adapter.default_emission_cols()
            )
            self.emission_cols  = ecols[:10] if self.model_type == "glm" else ecols
            self.transition_cols = adapter.default_transition_cols()
            self._refresh_groups()
        except Exception as e:
            print(f"Error applying default state for task {self.task}: {e}")

    def _refresh_groups(self) -> None:
        """Rebuild emission_groups / transition_groups from current *_options traits."""
        if self.task.upper() == "2AFC":
            e_reg = _2AFC_EMISSION_GROUPS
            t_reg: list[dict] = []
        else:
            e_reg = _MCDR_EMISSION_GROUPS
            t_reg = _MCDR_TRANSITION_GROUPS
        self.emission_groups   = _build_regressor_groups(self.emission_cols_options, e_reg)
        self.transition_groups = _build_regressor_groups(self.transition_cols_options, t_reg)

    def _build_model_info_list(
        self, fits_path: Path, default_info: dict
    ) -> tuple[list[str], list[dict]]:
        """Scan *fits_path* for displayable named models.

        Returns ``(display_name_list, info_dict_list)`` sorted by name,
        always prepended with the synthetic Default row.
        """
        names: list[str]      = []
        info_list: list[dict] = []

        if fits_path.exists():
            for d in fits_path.iterdir():
                if not (d.is_dir() and (d / "config.json").exists()):
                    continue
                try:
                    cfg = json.loads((d / "config.json").read_text())
                except Exception:
                    continue
                if not _is_displayable(cfg):
                    continue
                display_name = _get_display_name(cfg)
                n_subjects   = _count_fitted_subjects(d)
                info_list.append({
                    "id":       display_name,
                    "name":     display_name,
                    # prefer file-based count; fall back to config list length
                    "subjects": n_subjects if n_subjects > 0 else len(cfg.get("subjects", [])),
                    "K":        _get_K_from_config(cfg),
                    "tau":      cfg.get("tau", ""),
                    "regressors": ", ".join(cfg.get("emission_cols", [])),
                })
                names.append(display_name)

        info_list.sort(key=lambda x: x["name"])
        names.sort()
        return (["__default__"] + names, [default_info] + info_list)

    def _update_options(self) -> None:
        """Refresh all dynamic options from the adapter and the fits directory."""
        # ── adapter-derived options ───────────────────────────────────────────
        default_info: dict = {
            "id": "__default__", "name": "Default",
            "subjects": "?", "K": 2, "tau": 50, "regressors": "",
        }
        try:
            adapter = get_adapter(self.task)
            self.is_2afc = adapter.num_classes == 2

            df_all   = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
            df_all   = adapter.subject_filter(df_all)
            subjects = df_all["subject"].unique().to_list()

            self.subjects_list = subjects
            if not self.subjects:
                self.subjects = subjects

            ecols = (
                adapter.default_emission_cols() + adapter.sf_cols(df_all)
                if self.is_2afc else adapter.default_emission_cols()
            )
            self.emission_cols_options = ecols
            if not self.emission_cols:
                self.emission_cols = ecols[:10] if self.model_type == "glm" else ecols

            tcols = adapter.default_transition_cols()
            self.transition_cols_options = tcols
            if not self.transition_cols:
                self.transition_cols = tcols

            default_ecols = ecols[:10] if self.model_type == "glm" else ecols
            default_info = {
                "id":         "__default__",
                "name":       "Default",
                "subjects":   len(subjects),
                "K":          2,
                "tau":        50,
                "regressors": ", ".join(default_ecols),
            }
        except Exception as e:
            print(f"Error loading adapter for task {self.task}: {e}")

        # ── saved-model list ──────────────────────────────────────────────────
        fits_path = paths.RESULTS / "fits" / self.task / self.model_type
        display_names, info_list = self._build_model_info_list(fits_path, default_info)
        self.existing_models      = display_names
        self.existing_models_info = info_list

        # Drop stale selection if the selected model no longer exists
        if self.existing_model and self.existing_model not in display_names:
            self.existing_model = ""

        self._refresh_groups()
