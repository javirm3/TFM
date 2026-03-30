import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import sys, os
    from pathlib import Path
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    # Path setup
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import paths
    from analysis_common import (
        build_trial_and_weights_df,
        load_fit_arrays,
        resolve_selected_model_id,
        select_subject_behavior_df,
    )
    from scripts.fit_glm import main as fit_main, generate_model_id
    from tasks import get_adapter
    from widgets import ModelManagerWidget
    from figure_save_utils import make_plot_saver
    from coefficient_editor_widget import CoefficientEditorWidget
    from coefficient_editor_utils import (
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
    )

    sns.set_style("white")
    return (
        CoefficientEditorWidget,
        ModelManagerWidget,
        apply_state_tweak_to_trial_df,
        apply_state_tweak_to_view,
        build_editor_payload,
        build_trial_and_weights_df,
        fit_main,
        generate_model_id,
        get_adapter,
        load_fit_arrays,
        make_plot_saver,
        mo,
        np,
        paths,
        pd,
        pl,
        plt,
        resolve_selected_model_id,
        select_subject_behavior_df,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We import from the ui widgets and the model adapters
    """)
    return


@app.cell
def _(get_adapter, paths, pl, ui_model_manager):
    task_name = ui_model_manager.value.get("task", "MCDR")
    adapter = get_adapter(task_name)
    df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df_all = adapter.subject_filter(df_all)
    plots = adapter.get_plots()
    return adapter, df_all, plots, task_name


@app.cell
def _(ModelManagerWidget, mo):
    mm_widget = ModelManagerWidget(
        model_type="glm",
        task="MCDR",
        tau=50,
        lapse=False,
        lapse_max=0.2,
    )
    ui_model_manager = mo.ui.anywidget(mm_widget)
    return (ui_model_manager,)


@app.cell
def _(adapter, ui_model_manager):
    class _V:
        def __init__(self, value):
            self.value = value

    _val = ui_model_manager.value
    is_2afc = adapter.num_classes == 2

    ui_existing = _V(None if _val.get("existing_model") in ("", "__default__") else _val.get("existing_model"))
    ui_alias = _V(_val.get("alias", ""))
    ui_subjects = _V(_val.get("subjects", []))
    ui_tau = _V(_val.get("tau", 5))
    ui_lapse = _V(_val.get("lapse", False))
    ui_lapse_max = _V(_val.get("lapse_max", 0.2))
    ui_emission_cols = _V(_val.get("emission_cols", []))
    return (
        is_2afc,
        ui_alias,
        ui_emission_cols,
        ui_existing,
        ui_lapse,
        ui_lapse_max,
        ui_subjects,
        ui_tau,
    )


@app.cell
def _(generate_model_id, task_name, ui_emission_cols, ui_lapse, ui_tau):
    current_hash = generate_model_id(task_name, ui_tau.value, ui_emission_cols.value, lapse=ui_lapse.value)
    return (current_hash,)


@app.cell
def _(
    current_hash,
    make_plot_saver,
    mo,
    paths,
    resolve_selected_model_id,
    task_name,
    ui_alias,
    ui_existing,
):
    selected_model_id = resolve_selected_model_id(
        current_hash,
        ui_existing.value,
        ui_alias.value,
    )
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name=task_name,
        model_id=selected_model_id,
    )
    return save_plot, selected_model_id


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Configuration
    """)
    return


@app.cell
def _(current_hash, mo, save_plot, ui_model_manager):
    mo.vstack([
        ui_model_manager,
        save_plot.save_all_widget(label="Save all model plots"),
        mo.md(f"**Current params hash:** `{current_hash}`"),
    ])
    return


@app.cell
def _(
    fit_main,
    mo,
    task_name,
    ui_alias,
    ui_emission_cols,
    ui_lapse,
    ui_lapse_max,
    ui_model_manager,
    ui_subjects,
    ui_tau,
):
    _clicks = ui_model_manager.value.get("run_fit_clicks", 0)
    mo.stop(_clicks == 0, mo.md("Configure parameters and press **Run GLM Fit**."))

    with mo.status.spinner(title=f"Fitting GLM for {len(ui_subjects.value)} subjects..."):
        fit_main(
            subjects=ui_subjects.value,
            out_dir=None,
            tau=ui_tau.value,
            emission_cols=ui_emission_cols.value,
            task=task_name,
            model_alias=ui_alias.value if ui_alias.value else None,
            lapse=ui_lapse.value,
            lapse_max=ui_lapse_max.value,
        )

    mo.md("✅ Fit complete. Plots updating...")
    return


@app.cell
def _(
    adapter,
    df_all,
    load_fit_arrays,
    mo,
    paths,
    selected_model_id,
    task_name,
    ui_emission_cols,
    ui_subjects,
):
    def _normalize_glm_arrays(arrays: dict) -> dict:
        # ── Backward-compatibility: old fit_glm.py saved W_R at index 0.
        # New convention stores W_L (negative stim weight) at index 0.
        _weights = arrays.get("emission_weights")
        if _weights is None:
            return arrays

        stim_idx = next(
            (idx for idx, col in enumerate(arrays.get("X_cols", [])) if col in {"stim_vals", "stim_d", "ild_norm"}),
            None,
        )
        if stim_idx is not None and float(_weights[0, 0, stim_idx]) > 0:
            arrays["emission_weights"] = -_weights
        return arrays


    OUT = paths.RESULTS / "fits" / task_name / "glm" / selected_model_id
    arrays_store, names = load_fit_arrays(
        out_dir=OUT,
        arrays_suffix="glm_arrays.npz",
        adapter=adapter,
        df_all=df_all,
        subjects=list(ui_subjects.value),
        emission_cols=list(ui_emission_cols.value),
        postprocess_array=_normalize_glm_arrays,
    )

    mo.md(f"Loaded {len(arrays_store)} subjects from `{selected_model_id}`")
    return arrays_store, names


@app.cell
def _(adapter, arrays_store, mo, ui_subjects):
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))
    from glmhmmt.views import build_views
    K = 1
    views = build_views(arrays_store, adapter, K, _selected)
    return K, build_views, views


@app.cell
def _(adapter, arrays_store, build_views):
    editor_views = build_views(arrays_store, adapter, 1, list(arrays_store.keys()))
    return (editor_views,)


@app.cell
def _(is_2afc, np, pd, plt, sns):
    import re

    def plot_sequence_feature_weights(weights_df) -> plt.Figure | None:
        """Plot only sequential stimulus features (s_i / sf_i) from the canonical weights df."""
        feature_pattern = re.compile(r"^(?:s|sf)_(\d+)$")
        if weights_df is None or getattr(weights_df, "is_empty", lambda: False)():
            return None

        df_plot = weights_df.to_pandas() if hasattr(weights_df, "to_pandas") else pd.DataFrame(weights_df)
        if df_plot.empty:
            return None

        df_plot["feature_name"] = df_plot["feature"].astype(str)
        df_plot["seq_idx"] = df_plot["feature_name"].str.extract(feature_pattern, expand=False)
        df_plot = df_plot[df_plot["seq_idx"].notna()].copy()
        if df_plot.empty:
            return None

        df_plot["seq_idx"] = df_plot["seq_idx"].astype(int)
        if is_2afc:
            # Binary fits store logit(Left); flip sign so the plot keeps the intuitive rightward convention.
            df_plot["weight"] = -df_plot["weight"]

        # Collapse across class_idx so each subject/state/feature contributes one value.
        df_plot = (
            df_plot.groupby(
                ["subject", "state_rank", "state_label", "seq_idx", "feature_name"],
                as_index=False,
            )["weight"]
            .mean()
        )

        state_order = (
            df_plot[["state_rank", "state_label"]]
            .drop_duplicates()
            .sort_values("state_rank")
        )
        n_states = max(1, len(state_order))
        fig, axes = plt.subplots(1, n_states, figsize=(4.8 * n_states, 3.8), sharey=True)
        axes = np.atleast_1d(axes)

        for ax, (_, state_row) in zip(axes, state_order.iterrows()):
            state_rank = int(state_row["state_rank"])
            state_label = str(state_row["state_label"])
            state_df = df_plot[df_plot["state_rank"] == state_rank].copy()
            state_df = state_df.sort_values(["subject", "seq_idx"])

            for _, subj_df in state_df.groupby("subject", sort=False):
                ax.plot(
                    subj_df["seq_idx"],
                    subj_df["weight"],
                    color="#bdbdbd",
                    alpha=0.35,
                    linewidth=1.0,
                )

            summary = (
                state_df.groupby(["seq_idx", "feature_name"], as_index=False)
                .agg(
                    mean=("weight", "mean"),
                    std=("weight", "std"),
                    count=("weight", "count"),
                )
            )
            summary["sem"] = np.where(
                summary["count"] > 1,
                summary["std"] / np.sqrt(summary["count"]),
                0.0,
            )
            summary = summary.sort_values("seq_idx")

            ax.plot(
                summary["seq_idx"],
                summary["mean"],
                color="#1f77b4",
                marker="o",
                linewidth=2.2,
            )
            if len(summary) > 1:
                ax.fill_between(
                    summary["seq_idx"],
                    summary["mean"] - summary["sem"],
                    summary["mean"] + summary["sem"],
                    color="#1f77b4",
                    alpha=0.15,
                )

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_title(state_label)
            ax.set_xlabel("Sequential stimulus features")
            ax.set_xticks(summary["seq_idx"])
            ax.set_xticklabels(summary["feature_name"], rotation=35, ha="right")
            sns.despine(ax=ax)

        axes[0].set_ylabel("Weight")
        fig.suptitle("s_i / sf_i coefficients", y=1.02)
        fig.tight_layout()
        return fig

    return (plot_sequence_feature_weights,)


@app.cell
def _(adapter, build_trial_and_weights_df, df_all, mo, views):
    trial_df, weights_df = build_trial_and_weights_df(
        df_all,
        views=views,
        adapter=adapter,
        min_session_length=1,
    )
    mo.stop(trial_df.height == 0, mo.md("No subjects with matching data lengths."))
    return trial_df, weights_df


@app.cell
def _(
    K,
    arrays_store,
    is_2afc,
    mo,
    names,
    paths,
    pl,
    plot_sequence_feature_weights,
    plots,
    ui_subjects,
    views,
    weights_df,
):
    # Plot Weights (Folded / Agonist)
    # GLM is essentially K=1.
    # State Labels Trivial

    if not arrays_store:
        mo.stop(True, mo.md("No results loaded."))
    _selected = [s for s in ui_subjects.value if s in arrays_store]
    _save_path = paths.RESULTS / "plots/GLMHMM/emissions_coefs.png"
    _views_sel = {s: views[s] for s in _selected}
    _weights_df_sel = weights_df.filter(pl.col("subject").is_in(_selected))
    _state_labels = {s: dict(views[s].state_name_by_idx) for s in _selected}

    if hasattr(plots, "plot_emission_weights_by_subject"):
        if is_2afc:
            _fig_by_subject = plots.plot_emission_weights_by_subject(
                views=_views_sel,
                K=K,
                save_path=_save_path,
            )
        else:
            _fig_by_subject = plots.plot_emission_weights_by_subject(
                arrays_store=arrays_store,
                state_labels=_state_labels,
                names=names,
                K=K,
                subjects=_selected,
                save_path=_save_path,
            )
    else:
        _fig_by_subject, _ = plots.plot_emission_weights(
            views=_views_sel,
            K=K,
            save_path=_save_path,
        )

    if hasattr(plots, "plot_emission_weights_summary"):
        _summary_figs = [plots.plot_emission_weights_summary(views=_views_sel, K=K)]
    elif is_2afc:
        _summary_figs = [plots.plot_emission_weights(views=_views_sel, K=K)[1]]
    else:
        _summary_figs = list(
            plots.plot_emission_weights(
                arrays_store=arrays_store,
                state_labels=_state_labels,
                names=names,
                K=K,
                subjects=_selected,
            )
        )

    _fig_seq = plot_sequence_feature_weights(_weights_df_sel)
    _items = [mo.md("### Emission weights"), mo.md("#### By subject"), _fig_by_subject]
    if _fig_seq is not None:
        _items.extend([mo.md("#### Sequential coefficients"), _fig_seq])
    else:
        _items.extend(
            [
                mo.md("#### Sequential coefficients"),
                mo.md("No `s_i` / `sf_i` regressors found in the current GLM fit."),
            ]
        )
    _items.extend([mo.md("#### Summary"), *_summary_figs])
    mo.vstack(_items)
    return


@app.cell
def _(is_2afc, mo, pl, plots, save_plot, trial_df, ui_subjects, views):
    _selected = [s for s in ui_subjects.value if s in views]
    mo.stop(not _selected, mo.md("No fitted arrays found — run the fit first."))

    _views_sel = {s: views[s] for s in _selected}
    _trial_df_sel = trial_df.filter(pl.col("subject").is_in(_selected))

    mo.stop(_trial_df_sel.height == 0, mo.md("No subjects with matching data lengths."))

    _plot_df_all = plots.prepare_predictions_df(_trial_df_sel)
    _perf_kwargs = {"views": _views_sel} if is_2afc else {}
    _fig_all, _ = plots.plot_categorical_performance_all(
        _plot_df_all,
        "glm",
        # background_style=ui_psychometric_background.value,
        **_perf_kwargs,
    )
    _side_plot_fn = getattr(plots, "plot_categorical_strat_by_side", None)
    if _side_plot_fn is None:
        _side_section = mo.md("This task does not expose a side-stratified categorical plot.")
    else:
        _side_fig, _ = _side_plot_fn(
            _plot_df_all,
            subject=_selected[0] if len(_selected) == 1 else None,
            model_name="glm",
        )
        _side_section = mo.vstack(
            [
                mo.md("### Categorical performance by stimulus side"),
                _side_fig,
            ],
            align="center",
        )
    mo.vstack(
        [
            mo.md("### Categorical plots for accuracy"),
            mo.hstack(
                [
                    mo.vstack([_fig_all], align="center"),
                    mo.vstack(
                        [
                            save_plot(_fig_all, "overall psychometric", stem="categorical_overall"),
                        ],
                        align="start",
                    ),
                ],
                justify="space-between",
                align="center",
                widths=[4, 1],
            ),
            _side_section,
        ],
        align="center",
    )
    return


@app.cell
def _(editor_views, mo):
    subjects = list(editor_views.keys())
    mo.stop(not subjects, mo.md("No fitted subjects available for coefficient editing."))
    ui_editor_subject = mo.ui.dropdown(
        options=subjects,
        value=subjects[0],
        label="Subject",
    )
    ui_editor_subject
    return (ui_editor_subject,)


@app.cell
def _(editor_views, mo, ui_editor_subject):
    _view = editor_views[ui_editor_subject.value]
    _state_options = [
        f"{_k} — {_view.state_name_by_idx.get(_k, f'State {_k}')}"
        for _k in _view.state_idx_order
    ]
    ui_editor_state = mo.ui.dropdown(
        options=_state_options,
        value=_state_options[0],
        label="State",
    )
    ui_editor_state
    return (ui_editor_state,)


@app.cell
def _(adapter, mo):
    if adapter.num_classes != 2:
        ui_editor_side = None
    else:
        _choices = [str(label) for label in adapter.choice_labels]
        ui_editor_side = mo.ui.dropdown(
            options=_choices,
            value=_choices[0],
            label="Side",
        )
        ui_editor_side
    return (ui_editor_side,)


@app.cell
def _(
    CoefficientEditorWidget,
    adapter,
    build_editor_payload,
    editor_views,
    mo,
    np,
    ui_editor_side,
    ui_editor_state,
    ui_editor_subject,
):
    _subj = ui_editor_subject.value
    _view = editor_views[_subj]
    coef_state_idx = int(ui_editor_state.value.split(" — ", 1)[0])
    coef_state_label = _view.state_name_by_idx.get(
        coef_state_idx, f"State {coef_state_idx}"
    )
    _stored_weights = np.asarray(_view.emission_weights[coef_state_idx], dtype=float)
    _choice_labels = [str(label) for label in adapter.choice_labels]
    _stored_class_indices = [0] if _view.num_classes == 2 else [0, 2]
    _reference_class_idx = 1 if _view.num_classes > 2 else (_view.num_classes - 1)
    if _view.num_classes == 2 and ui_editor_side is not None:
        _display_class_idx = _choice_labels.index(ui_editor_side.value)
        _display_reference_class_idx = next(
            idx for idx in range(_view.num_classes) if idx != _display_class_idx
        )
    else:
        _display_reference_class_idx = None
    _payload = build_editor_payload(
        _stored_weights,
        choice_labels=_choice_labels,
        stored_class_indices=_stored_class_indices,
        reference_class_idx=_reference_class_idx,
        display_reference_class_idx=_display_reference_class_idx,
    )

    coef_editor = mo.ui.anywidget(
        CoefficientEditorWidget(
            title="Coefficient editor",
            subtitle=_payload["subtitle"],
            features=list(_view.feat_names),
            channel_labels=_payload["channel_labels"],
            weights=_payload["weights"].tolist(),
            original_weights=_payload["weights"].tolist(),
            slider_min=-6.0,
            slider_max=6.0,
            slider_step=0.05,
        )
    )
    _controls = [ui_editor_subject, ui_editor_state]
    if ui_editor_side is not None:
        _controls.append(ui_editor_side)

    coef_editor_panel = mo.vstack(
        [
            mo.md("### Interactive coefficient editor"),
            mo.md(
                "The edited state is recomputed live and the categorical plots "
                "below use the updated probabilities."
            ),
            mo.hstack(_controls),
            coef_editor,
        ],
        align="center",
    )
    coef_editor_panel
    coef_editor_explicit_class_indices = _payload["explicit_class_indices"]
    coef_editor_reference_class_idx = _payload["reference_class_idx"]
    coef_editor_stored_class_indices = _payload["stored_class_indices"]
    coef_editor_stored_reference_class_idx = _payload["stored_reference_class_idx"]
    coef_editor
    return (
        coef_editor,
        coef_editor_explicit_class_indices,
        coef_editor_reference_class_idx,
        coef_editor_stored_class_indices,
        coef_editor_stored_reference_class_idx,
        coef_state_idx,
        coef_state_label,
    )


@app.cell
def _(
    adapter,
    df_all,
    editor_views,
    mo,
    select_subject_behavior_df,
    ui_editor_subject,
):
    _subj = ui_editor_subject.value
    _view = editor_views[_subj]
    from glmhmmt.postprocess import build_trial_df

    _df_sub = select_subject_behavior_df(
        df_all,
        subject=_subj,
        sort_col=adapter.sort_col,
        session_col=adapter.session_col,
        min_session_length=1,
    )
    mo.stop(_df_sub.height != _view.T, mo.md(f"Subject {_subj} does not match the loaded fit arrays."))
    editor_trial_df = build_trial_df(_view, adapter, _df_sub, adapter.behavioral_cols)
    editor_view = _view
    return editor_trial_df, editor_view


@app.cell
def _(
    adapter,
    apply_state_tweak_to_trial_df,
    apply_state_tweak_to_view,
    coef_editor,
    coef_editor_explicit_class_indices,
    coef_editor_reference_class_idx,
    coef_editor_stored_class_indices,
    coef_editor_stored_reference_class_idx,
    coef_state_idx,
    coef_state_label,
    editor_trial_df,
    editor_view,
    mo,
    np,
    plots,
    save_plot,
    ui_editor_subject,
):
    _subj = ui_editor_subject.value
    _view = editor_view
    _trial_df_sub = editor_trial_df
    _edited_weights = np.asarray(coef_editor.value["weights"], dtype=float)

    _trial_df_tweaked = apply_state_tweak_to_trial_df(
        _trial_df_sub,
        adapter=adapter,
        view=_view,
        state_idx=coef_state_idx,
        edited_weights=_edited_weights,
        original_weights=np.asarray(coef_editor.value["original_weights"], dtype=float),
        explicit_class_indices=list(coef_editor_explicit_class_indices),
        reference_class_idx=int(coef_editor_reference_class_idx),
    )
    _view_tweaked = apply_state_tweak_to_view(
        _view,
        state_idx=coef_state_idx,
        edited_weights=_edited_weights,
        explicit_class_indices=list(coef_editor_explicit_class_indices),
        reference_class_idx=int(coef_editor_reference_class_idx),
        stored_class_indices=list(coef_editor_stored_class_indices),
        stored_reference_class_idx=int(coef_editor_stored_reference_class_idx),
    )
    _plot_df_tweaked = plots.prepare_predictions_df(_trial_df_tweaked)

    _title = f"{_subj} — tweaked {coef_state_label}"
    _fig_all_tweaked, _ = plots.plot_categorical_performance_all(
        _plot_df_tweaked,
        _title,
        # background_style=ui_psychometric_background.value,
    )
    _side_plot_fn = getattr(plots, "plot_categorical_strat_by_side", None)
    if _side_plot_fn is None:
        _side_section = mo.md("This task does not expose a side-stratified categorical plot.")
    else:
        _fig_side_tweaked, _ = plots.plot_categorical_strat_by_side(
            _plot_df_tweaked,
            subject=_subj,
            model_name=f"{_subj}_tweaked_{coef_state_idx}",
        )
        _side_section = mo.vstack(
            [
                mo.md("### Tweaked categorical performance by stimulus side"),
                _fig_side_tweaked,
            ]
        )

    mo.vstack(
        [
            mo.md("### Tweaked categorical plots"),
            mo.vstack(
                [
                    mo.vstack([_fig_all_tweaked], align="center"),
                    mo.vstack(
                        [
                            # ui_psychometric_background,
                            save_plot(
                                _fig_all_tweaked,
                                "tweaked overall psychometric",
                                stem="tweaked_categorical_overall",
                            ),
                        ],
                        align="start",
                    ),
                ],
                justify="space-between",
                align="center",
            ),
            _side_section,
        ],
        align="center",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
