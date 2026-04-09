from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glmhmmt import model_plots
from glmhmmt.notebook_support import figure_save_utils
from glmhmmt.runtime import load_app_config
from glmhmmt.views import SubjectFitView

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import tasks.plots.mcdr as mcdr_plots
import tasks.plots.nuo_auditory as nuo_auditory_plots
import tasks.plots.two_afc as two_afc_plots


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _make_view(subject: str = "S1") -> SubjectFitView:
    smoothed_probs = np.asarray(
        [
            [0.92, 0.08],
            [0.85, 0.15],
            [0.18, 0.82],
            [0.12, 0.88],
            [0.86, 0.14],
            [0.89, 0.11],
        ],
        dtype=float,
    )
    emission_weights = np.asarray(
        [
            [[0.2, 0.8, -0.3]],
            [[-0.4, -0.2, 0.5]],
        ],
        dtype=float,
    )
    X = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    p_pred = np.asarray(
        [
            [0.65, 0.35],
            [0.35, 0.65],
            [0.70, 0.30],
            [0.30, 0.70],
            [0.68, 0.32],
            [0.32, 0.68],
        ],
        dtype=float,
    )
    return SubjectFitView(
        subject=subject,
        K=2,
        smoothed_probs=smoothed_probs,
        emission_weights=emission_weights,
        X=X,
        y=np.asarray([0, 1, 0, 1, 0, 1], dtype=int),
        feat_names=["bias", "A_L", "A_R"],
        state_name_by_idx={0: "Engaged", 1: "Disengaged"},
        state_idx_order=[0, 1],
        state_rank_by_idx={0: 0, 1: 1},
        predictive_state_probs=smoothed_probs.copy(),
        p_pred=p_pred,
        transition_matrix=np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=float),
        transition_bias=None,
        transition_weights=None,
        U=None,
        U_cols=[],
    )


def _make_views() -> dict[str, SubjectFitView]:
    return {"S1": _make_view("S1")}


def _make_transition_views() -> dict[str, SubjectFitView]:
    transition_weights = np.asarray(
        [
            [[0.6, -0.2, 0.1], [0.3, 0.1, -0.4]],
            [[-0.1, 0.4, 0.2], [0.2, -0.3, 0.5]],
        ],
        dtype=float,
    )
    base = _make_view("S1")
    return {
        "S1": SubjectFitView(
            subject=base.subject,
            K=base.K,
            smoothed_probs=base.smoothed_probs,
            emission_weights=base.emission_weights,
            X=base.X,
            y=base.y,
            feat_names=base.feat_names,
            state_name_by_idx=base.state_name_by_idx,
            state_idx_order=base.state_idx_order,
            state_rank_by_idx=base.state_rank_by_idx,
            predictive_state_probs=base.predictive_state_probs,
            p_pred=base.p_pred,
            transition_matrix=base.transition_matrix,
            transition_bias=base.transition_bias,
            transition_weights=transition_weights,
            U=np.ones((base.smoothed_probs.shape[0], transition_weights.shape[2]), dtype=float),
            U_cols=["bias", "prev_choice", "reward"],
        )
    }


def _make_trial_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject": ["S1"] * 6,
            "session": [1, 1, 1, 2, 2, 2],
            "trial_idx": [0, 1, 2, 0, 1, 2],
            "stimulus": [0, 1, 0, 1, 0, 1],
            "response": [0, 1, 1, 1, 0, 1],
            "performance": [1, 1, 0, 1, 1, 1],
            "correct_bool": [True, True, False, True, True, True],
            "pL": [0.65, 0.35, 0.70, 0.30, 0.68, 0.32],
            "pR": [0.35, 0.65, 0.30, 0.70, 0.32, 0.68],
        }
    )


def test_generic_plots_accept_canonical_trial_df():
    views = _make_views()
    trial_df = _make_trial_df()

    fig_acc, table = model_plots.plot_state_accuracy(views, trial_df)
    fig_traj = model_plots.plot_session_trajectories(views, trial_df)
    fig_deep = model_plots.plot_session_deepdive(views, trial_df, subj="S1", sess=1)

    assert fig_acc is not None
    assert fig_traj is not None
    assert fig_deep is not None
    assert not table.empty


def test_generic_plots_fail_fast_without_canonical_columns():
    views = _make_views()
    trial_df = _make_trial_df()
    alias_df = trial_df.rename({"session": "Session", "trial_idx": "Trial", "response": "Choice"})

    with pytest.raises(KeyError):
        model_plots.plot_session_trajectories(views, alias_df)
    with pytest.raises(KeyError):
        model_plots.plot_session_deepdive(views, alias_df, subj="S1", sess=1)
    with pytest.raises(KeyError):
        model_plots.plot_state_accuracy(views, trial_df.drop("correct_bool"))


def test_task_facades_render_through_model_plots():
    views = _make_views()
    trial_df = _make_trial_df()
    arrays_store = {"S1": {"transition_matrix": np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=float)}}
    state_labels = {"S1": {0: "Engaged", 1: "Disengaged"}}

    fig_two_afc = two_afc_plots.plot_emission_weights_by_subject(views=views, K=2)
    fig_nuo = nuo_auditory_plots.plot_emission_weights_by_subject(views=views, K=2)
    fig_mcdr = mcdr_plots.plot_session_trajectories(views, trial_df)
    fig_trans = two_afc_plots.plot_transition_matrix(arrays_store=arrays_store, state_labels=state_labels, K=2, subjects=["S1"])

    assert fig_two_afc is not None
    assert fig_nuo is not None
    assert fig_mcdr is not None
    assert fig_trans is not None


def test_transition_weight_plots_render_from_views():
    fig_line, fig_box = model_plots.plot_transition_weights(views=_make_transition_views())

    assert fig_line is not None
    assert fig_box is not None


def test_task_plot_sources_have_no_local_transition_matrix_or_plots_common_imports():
    for rel_path in (
        "tasks/plots/two_afc.py",
        "tasks/plots/two_afc_delay.py",
        "tasks/plots/nuo_auditory.py",
    ):
        source = (_CODE_ROOT / rel_path).read_text(encoding="utf-8")
        assert "def plot_transition_matrix(" not in source
        assert "def plot_transition_matrix_by_subject(" not in source
        assert "def plot_trans_mat(" not in source
        assert "def plot_trans_mat_boxplots(" not in source
        assert "def plot_occupancy(" not in source
        assert "def plot_occupancy_boxplot(" not in source
        assert "def norm_ll(" not in source
        assert "def plot_ll(" not in source
        assert "def plot_model_comparison(" not in source
        assert "def plot_model_comparison_diffs(" not in source
        assert "from glmhmmt.plots_common import" not in source
        assert "_resolve_plot_col" not in source


def test_save_figure_config_defaults_and_override(tmp_path):
    default_cfg = load_app_config()
    save_cfg = default_cfg["widgets"]["save_figure"]
    assert save_cfg["default_label"] == "Save"
    assert save_cfg["save_all_label"] == "Save all model plots"

    override_path = tmp_path / "config.toml"
    override_path.write_text(
        "\n".join(
            [
                "[widgets.save_figure]",
                'default_label = "Archive"',
                'radius = "12px"',
                "",
                "[plots]",
                'save_format = "svg"',
            ]
        ),
        encoding="utf-8",
    )

    merged = load_app_config(override_path)
    assert merged["widgets"]["save_figure"]["default_label"] == "Archive"
    assert merged["widgets"]["save_figure"]["radius"] == "12px"
    assert figure_save_utils.get_plot_save_format(override_path) == "svg"


def test_notebook_support_imports_do_not_eagerly_import_model():
    for name in [
        "glmhmmt.model",
        "glmhmmt.notebook_support.analysis_common",
        "glmhmmt.notebook_support.figure_save_utils",
        "notebooks.analysis_common",
        "notebooks.figure_save_utils",
    ]:
        sys.modules.pop(name, None)

    importlib.import_module("glmhmmt.notebook_support")
    assert "glmhmmt.model" not in sys.modules

    importlib.import_module("notebooks.analysis_common")
    assert "glmhmmt.model" not in sys.modules

    importlib.import_module("notebooks.figure_save_utils")
    assert "glmhmmt.model" not in sys.modules
