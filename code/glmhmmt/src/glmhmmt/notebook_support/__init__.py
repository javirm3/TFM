"""Shared helpers for repository marimo notebooks."""

from .anywidget_compat import wrap_anywidget
from .coefficient_editor_utils import (
    apply_state_tweak_to_trial_df,
    apply_state_tweak_to_view,
    build_editor_payload,
)
from .coefficient_editor_widget import CoefficientEditorWidget
from .figure_save_utils import make_plot_saver
from .model_manager import ModelManagerWidget, model_cfg

__all__ = [
    "wrap_anywidget",
    "apply_state_tweak_to_trial_df",
    "apply_state_tweak_to_view",
    "build_editor_payload",
    "CoefficientEditorWidget",
    "make_plot_saver",
    "ModelManagerWidget",
    "model_cfg",
]
