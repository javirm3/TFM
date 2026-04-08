import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from matplotlib.lines import Line2D
    from scipy.stats import ttest_ind, ttest_rel

    from glmhmmt.plots_common import custom_boxplot
    from glmhmmt.postprocess import build_trial_df
    from glmhmmt.runtime import get_runtime_paths
    from glmhmmt.tasks import get_adapter
    from glmhmmt.views import build_views
    from glmhmmt.notebook_support import make_plot_saver

    paths = get_runtime_paths()
    sns.set_style("ticks")
    return (
        Line2D,
        build_trial_df,
        build_views,
        custom_boxplot,
        get_adapter,
        json,
        make_plot_saver,
        mo,
        np,
        paths,
        pd,
        pl,
        plt,
        sns,
        ttest_ind,
        ttest_rel,
    )


@app.cell
def _(make_plot_saver, mo, paths):
    save_plot = make_plot_saver(
        mo,
        results_dir=paths.RESULTS,
        config_path=paths.CONFIG,
        task_name="2AFC",
        model_id="DrugComparison"
    )
    return (save_plot,)


@app.cell
def _(pd, ttest_ind, ttest_rel):
    import itertools

    def _sig_label(pval: float) -> str:
        if pval < 0.001:
            return "***"
        if pval < 0.01:
            return "**"
        if pval < 0.05:
            return "*"
        return "ns"

    def add_sig_bars(
        ax,
        df,
        *,
        x_col,
        y_col,
        hue_col,
        order,
        hue_order,
        pair_col=None,
        fallback_to_unpaired: bool = False,
    ):
        n_hue = max(1, len(hue_order))
        hue_width = 0.8 / n_hue
        y_range = df[y_col].max() - df[y_col].min()
        if pd.isna(y_range) or y_range == 0:
            y_range = 1.0

        for _m, _xval in enumerate(order):
            _sub = df[df[x_col] == _xval]
            if _sub.empty:
                continue

            current_y = _sub[y_col].max() + y_range * 0.05
            h = y_range * 0.02

            for _p1, _p2 in itertools.combinations(range(n_hue), 2):
                _g1 = hue_order[_p1]
                _g2 = hue_order[_p2]

                _s1 = _sub[_sub[hue_col] == _g1]
                _s2 = _sub[_sub[hue_col] == _g2]

                if pair_col is not None:
                    _v1 = _s1.set_index(pair_col)[y_col]
                    _v2 = _s2.set_index(pair_col)[y_col]
                    _common = _v1.index.intersection(_v2.index)
                    if len(_common) >= 2:
                        _, _pval = ttest_rel(_v1.loc[_common].values, _v2.loc[_common].values)
                    elif fallback_to_unpaired:
                        _v1 = _s1[y_col].dropna().values
                        _v2 = _s2[y_col].dropna().values
                        if min(len(_v1), len(_v2)) < 2:
                            continue
                        _, _pval = ttest_ind(_v1, _v2, equal_var=False)
                    else:
                        continue
                else:
                    _v1 = _s1[y_col].dropna().values
                    _v2 = _s2[y_col].dropna().values
                    if min(len(_v1), len(_v2)) < 2:
                        continue
                    _, _pval = ttest_ind(_v1, _v2, equal_var=False)

                _star = _sig_label(float(_pval))
                if _star == "ns":
                    continue

                _x1 = _m + (_p1 - (n_hue - 1) / 2) * hue_width
                _x2 = _m + (_p2 - (n_hue - 1) / 2) * hue_width

                ax.plot(
                    [_x1, _x1, _x2, _x2],
                    [current_y, current_y + h, current_y + h, current_y],
                    lw=1,
                    c="k",
                )
                ax.text((_x1 + _x2) / 2, current_y + h, _star, ha="center", va="bottom", color="k")
                current_y += y_range * 0.075

    return (add_sig_bars,)


@app.cell
def _(json, paths, pl):
    def model_aliases(task: str) -> list[str]:
        fit_root = paths.RESULTS / "fits" / task / "glmhmm"
        if not fit_root.exists():
            return []
        return sorted([_child.name for _child in fit_root.iterdir() if _child.is_dir()])

    def load_model_config(task: str, alias: str | None):
        if not alias:
            return {}
        cfg_path = paths.RESULTS / "fits" / task / "glmhmm" / alias / "config.json"
        if not cfg_path.exists():
            return {}
        return json.loads(cfg_path.read_text())

    def load_metrics_dir(task: str, alias: str | None):
        if not alias:
            return None
        fit_dir = paths.RESULTS / "fits" / task / "glmhmm" / alias
        if not fit_dir.exists():
            return None

        _files = sorted(fit_dir.glob("*_metrics.parquet"))
        if not _files:
            return None

        _frames = []
        for _path in _files:
            _frames.append(pl.read_parquet(_path))
        df = pl.concat(_frames, how="diagonal")

        if "nll" in df.columns and "ll_per_trial" not in df.columns:
            df = df.with_columns((-pl.col("nll") / pl.col("n_trials")).alias("ll_per_trial"))
        if "K" not in df.columns:
            df = df.with_columns(pl.lit(1, dtype=pl.Int64).alias("K"))
        else:
            df = df.with_columns(pl.col("K").cast(pl.Int64))
        return df

    def model_k_options(task: str, alias: str | None) -> list[int]:
        df = load_metrics_dir(task, alias)
        if df is None or df.is_empty():
            return []
        return sorted({int(_value) for _value in df["K"].drop_nulls().to_list()})

    def preferred_or_first(options: list[str], preferred: str) -> str | None:
        if preferred in options:
            return preferred
        return options[0] if options else None

    return (
        load_metrics_dir,
        load_model_config,
        model_aliases,
        model_k_options,
        preferred_or_first,
    )


@app.cell
def _(build_views, get_adapter, np, paths):
    def load_fit_bundle(task: str, alias: str, K: int, subjects: list[str], scoring_key: str | None):
        adapter = get_adapter(task)
        fit_dir = paths.RESULTS / "fits" / task / "glmhmm" / alias
        arrays_store = {}

        for _subject in subjects:
            _candidates = [
                fit_dir / f"{_subject}_K{K}_glmhmm_arrays.npz",
                fit_dir / f"{_subject}_glmhmm_arrays.npz",
            ]
            for _path in _candidates:
                if not _path.exists():
                    continue
                arrays_store[_subject] = dict(np.load(_path, allow_pickle=True))
                break

        if scoring_key is not None and hasattr(adapter, "scoring_key"):
            adapter.scoring_key = scoring_key
        views = build_views(arrays_store, adapter, K, list(arrays_store.keys())) if arrays_store else {}
        return adapter, views

    return (load_fit_bundle,)


@app.cell
def _(get_adapter, mo, model_aliases, preferred_or_first):
    normal_aliases = model_aliases("2AFC")
    drug_aliases = model_aliases("2AFC_DRUG")

    normal_adapter = get_adapter("2AFC")
    scoring_options = list(normal_adapter._SCORING_OPTIONS.keys()) if hasattr(normal_adapter, "_SCORING_OPTIONS") else []
    default_scoring = getattr(normal_adapter, "scoring_key", scoring_options[0] if scoring_options else None)
    if scoring_options and default_scoring not in scoring_options:
        default_scoring = scoring_options[0]

    ui_alias_normal = mo.ui.dropdown(
        options=normal_aliases,
        value=preferred_or_first(normal_aliases, "3covs_2states"),
        label="2AFC alias",
    )
    ui_alias_drug_all = mo.ui.dropdown(
        options=drug_aliases,
        value=preferred_or_first(drug_aliases, "3covs_2states"),
        label="2AFC_DRUG all",
    )
    ui_alias_drug_saline = mo.ui.dropdown(
        options=drug_aliases,
        value=preferred_or_first(drug_aliases, "3covs_2states_saline"),
        label="2AFC_DRUG saline",
    )
    ui_alias_drug_drug = mo.ui.dropdown(
        options=drug_aliases,
        value=preferred_or_first(drug_aliases, "3covs_2states_drug"),
        label="2AFC_DRUG drug",
    )
    ui_scoring_key = mo.ui.dropdown(
        options=scoring_options,
        value=default_scoring,
        label="State scoring regressor",
    )
    return (
        ui_alias_drug_all,
        ui_alias_drug_drug,
        ui_alias_drug_saline,
        ui_alias_normal,
        ui_scoring_key,
    )


@app.cell
def _(
    mo,
    model_k_options,
    ui_alias_drug_all,
    ui_alias_drug_drug,
    ui_alias_drug_saline,
    ui_alias_normal,
):
    k_sets = [
        set(model_k_options("2AFC", ui_alias_normal.value)),
        set(model_k_options("2AFC_DRUG", ui_alias_drug_all.value)),
        set(model_k_options("2AFC_DRUG", ui_alias_drug_saline.value)),
        set(model_k_options("2AFC_DRUG", ui_alias_drug_drug.value)),
    ]
    shared_k = sorted(set.intersection(*k_sets)) if all(k_sets) else []
    ui_shared_K = mo.ui.dropdown(
        options=shared_k,
        value=shared_k[0] if shared_k else None,
        label="Shared K",
    )
    return (ui_shared_K,)


@app.cell
def _(
    mo,
    ui_alias_drug_all,
    ui_alias_drug_drug,
    ui_alias_drug_saline,
    ui_alias_normal,
    ui_scoring_key,
    ui_shared_K,
):
    mo.vstack(
        [
            mo.md("## GLMHMM Drug Comparison"),
            mo.md(
                "Compare occupancy and fit quality across `2AFC` and `2AFC_DRUG` GLM-HMM fits. "
                "The overall section uses cohort-level grouped comparisons; the drug-only section reuses matched subjects across the three drug aliases."
            ),
            mo.hstack([ui_alias_normal, ui_shared_K]),
            mo.hstack([ui_alias_drug_all, ui_alias_drug_saline, ui_alias_drug_drug]),
            mo.hstack([ui_scoring_key]),
        ]
    )
    return


@app.cell
def _(
    get_adapter,
    load_fit_bundle,
    load_metrics_dir,
    load_model_config,
    mo,
    paths,
    pl,
    ui_alias_drug_all,
    ui_alias_drug_drug,
    ui_alias_drug_saline,
    ui_alias_normal,
    ui_scoring_key,
    ui_shared_K,
):
    mo.stop(
        not ui_alias_normal.value
        or not ui_alias_drug_all.value
        or not ui_alias_drug_saline.value
        or not ui_alias_drug_drug.value,
        mo.md("Select all four aliases above."),
    )
    mo.stop(ui_shared_K.value is None, mo.md("No shared `K` was found for the selected aliases."))

    _metric_schema = {
        "subject": pl.Utf8,
        "K": pl.Int64,
        "ll_per_trial": pl.Float64,
        "bic": pl.Float64,
        "acc": pl.Float64,
    }

    overall_k = int(ui_shared_K.value)

    normal_adapter_base = get_adapter("2AFC")
    drug_adapter_base = get_adapter("2AFC_DRUG")
    df_normal_all = normal_adapter_base.subject_filter(pl.read_parquet(paths.DATA_PATH / normal_adapter_base.data_file))
    df_drug_all = drug_adapter_base.subject_filter(pl.read_parquet(paths.DATA_PATH / drug_adapter_base.data_file))

    selection_defs = [
        ("2AFC", "2AFC", ui_alias_normal.value, df_normal_all),
        ("2AFC_DRUG", "2AFC_DRUG all", ui_alias_drug_all.value, df_drug_all),
        ("2AFC_DRUG", "2AFC_DRUG saline", ui_alias_drug_saline.value, df_drug_all),
        ("2AFC_DRUG", "2AFC_DRUG drug", ui_alias_drug_drug.value, df_drug_all),
    ]

    overall_specs = []
    for _task, _label, _alias, _df_all in selection_defs:
        _cfg = load_model_config(_task, _alias)
        _requested_subjects = [str(_subject) for _subject in _cfg.get("subjects", [])]
        _adapter, _views = load_fit_bundle(_task, _alias, overall_k, _requested_subjects, ui_scoring_key.value)
        _metrics = load_metrics_dir(_task, _alias)
        if _metrics is None:
            _metrics = pl.DataFrame(schema=_metric_schema)
        else:
            _metrics = _metrics.filter(
                (pl.col("K") == overall_k) & pl.col("subject").is_in(sorted(_views.keys()))
            ).select(["subject", "K", "ll_per_trial", "bic", "acc"])
        overall_specs.append(
            {
                "task": _task,
                "model_label": _label,
                "alias": _alias,
                "condition_filter": str(_cfg.get("condition_filter", "all")),
                "adapter": _adapter,
                "views": _views,
                "metrics": _metrics,
                "df_all": _df_all,
            }
        )

    drug_specs = overall_specs[1:]
    common_drug_subjects = []
    if drug_specs and all(_spec["views"] for _spec in drug_specs):
        common_drug_subjects = sorted(
            set.intersection(*(set(_spec["views"].keys()) for _spec in drug_specs))
        )

    overall_hue_order = [_spec["model_label"] for _spec in overall_specs]
    drug_hue_order = [_spec["model_label"] for _spec in drug_specs]

    notes = [
        f"- Shared `K`: **{overall_k}**.",
        f"- Semantic state alignment uses scoring key `{ui_scoring_key.value}`.",
    ]
    for _spec in overall_specs:
        notes.append(
            f"- `{_spec['model_label']}` uses `{_spec['alias']}` with **{len(_spec['views'])}** cached subjects."
        )
    notes.append(
        f"- Drug-only matched subset across all/saline/drug: **{len(common_drug_subjects)}** subjects."
    )
    overall_notes_md = "\n".join(notes)
    return (
        common_drug_subjects,
        drug_hue_order,
        drug_specs,
        overall_hue_order,
        overall_k,
        overall_notes_md,
        overall_specs,
    )


@app.cell
def _(mo, overall_notes_md):
    mo.md(overall_notes_md)
    return


@app.cell
def _(pl):
    def subject_behavior_df(
        df_all,
        *,
        subject,
        sort_col,
        session_col,
        task_name="2AFC",
        condition_filter="all",
    ):
        if str(task_name).upper() == "2AFC_DRUG":
            selected = str(condition_filter or "all").strip().lower()
            if selected in {"saline", "drug"}:
                if "Drug" not in df_all.columns:
                    raise ValueError("2AFC_DRUG requires a 'Drug' column for condition filtering.")
                target = 1 if selected == "drug" else 0
                df_all = (
                    df_all.with_columns(
                        pl.col("Drug").fill_null(0).cast(pl.Int64, strict=False).alias("__drug_filter")
                    )
                    .filter(pl.col("__drug_filter") == target)
                    .drop("__drug_filter")
                )
        df_sub = df_all.filter(pl.col("subject") == subject).sort(sort_col)
        if session_col in df_sub.columns:
            df_sub = df_sub.filter(pl.col(session_col).count().over(session_col) >= 2)
        return df_sub

    return (subject_behavior_df,)


@app.cell
def _(
    build_trial_df,
    common_drug_subjects,
    drug_specs,
    np,
    overall_specs,
    pl,
    subject_behavior_df,
):
    occupancy_schema = {
        "subject": pl.Utf8,
        "model_label": pl.Utf8,
        "model_alias": pl.Utf8,
        "state_label": pl.Utf8,
        "occupancy": pl.Float64,
    }
    _metric_schema = {
        "subject": pl.Utf8,
        "K": pl.Int64,
        "model_label": pl.Utf8,
        "model_alias": pl.Utf8,
        "ll_per_trial": pl.Float64,
        "bic": pl.Float64,
        "acc": pl.Float64,
    }
    accuracy_schema = {
        "subject": pl.Utf8,
        "model_label": pl.Utf8,
        "model_alias": pl.Utf8,
        "state_label": pl.Utf8,
        "accuracy": pl.Float64,
    }
    trial_schema = {
        "subject": pl.Utf8,
        "model_label": pl.Utf8,
        "model_alias": pl.Utf8,
        "state_label": pl.Utf8,
        "correct_bool": pl.Boolean,
    }

    def build_subject_occupancy(model_specs: list[dict], *, restrict_subjects: list[str] | None = None):
        records = []
        subject_whitelist = None if restrict_subjects is None else set(restrict_subjects)

        for _spec in model_specs:
            for _subject, _view in _spec["views"].items():
                if subject_whitelist is not None and _subject not in subject_whitelist:
                    continue

                _df_sub = subject_behavior_df(
                    _spec["df_all"],
                    subject=_subject,
                    sort_col=_spec["adapter"].sort_col,
                    session_col=_spec["adapter"].session_col,
                    task_name=_spec["adapter"].task_key,
                    condition_filter=_spec["condition_filter"],
                )
                if _spec["adapter"].session_col not in _df_sub.columns or _df_sub.height != _view.T:
                    continue

                _sessions = np.asarray(_df_sub[_spec["adapter"].session_col])
                _probs = np.asarray(_view.smoothed_probs, dtype=float)
                for _session in np.unique(_sessions):
                    _mask = _sessions == _session
                    for _state_idx in _view.state_idx_order:
                        records.append(
                            {
                                "subject": str(_subject),
                                "model_label": _spec["model_label"],
                                "model_alias": _spec["alias"],
                                "state_label": _view.state_name_by_idx.get(int(_state_idx), f"State {_state_idx}"),
                                "occupancy": float(np.mean(_probs[_mask, int(_state_idx)])),
                            }
                        )

        if not records:
            return pl.DataFrame(schema=occupancy_schema)

        return (
            pl.DataFrame(records)
            .group_by(["subject", "model_label", "model_alias", "state_label"])
            .agg(pl.mean("occupancy").alias("occupancy"))
            .sort(["state_label", "model_label", "subject"])
        )

    def build_metrics_df(model_specs: list[dict], *, restrict_subjects: list[str] | None = None):
        frames = []
        subject_whitelist = None if restrict_subjects is None else set(restrict_subjects)

        for _spec in model_specs:
            _metrics = _spec["metrics"]
            if _metrics.is_empty():
                continue
            if subject_whitelist is not None:
                _metrics = _metrics.filter(pl.col("subject").is_in(sorted(subject_whitelist)))
            if _metrics.is_empty():
                continue
            frames.append(
                _metrics.with_columns(
                    [
                        pl.lit(_spec["model_label"]).alias("model_label"),
                        pl.lit(_spec["alias"]).alias("model_alias"),
                    ]
                ).select(["subject", "K", "model_label", "model_alias", "ll_per_trial", "bic", "acc"])
            )

        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame(schema=_metric_schema)

    def build_trial_level_df(model_specs: list[dict], *, restrict_subjects: list[str] | None = None):
        frames = []
        subject_whitelist = None if restrict_subjects is None else set(restrict_subjects)

        for _spec in model_specs:
            for _subject, _view in _spec["views"].items():
                if subject_whitelist is not None and _subject not in subject_whitelist:
                    continue

                _df_sub = subject_behavior_df(
                    _spec["df_all"],
                    subject=_subject,
                    sort_col=_spec["adapter"].sort_col,
                    session_col=_spec["adapter"].session_col,
                    task_name=_spec["adapter"].task_key,
                    condition_filter=_spec["condition_filter"],
                )
                if _df_sub.height != _view.T:
                    continue

                _trial_df = build_trial_df(
                    _view,
                    _spec["adapter"],
                    _df_sub,
                    _spec["adapter"].behavioral_cols,
                )
                if "correct_bool" in _trial_df.columns:
                    _correct_expr = pl.col("correct_bool").cast(pl.Boolean, strict=False).alias("correct_bool")
                elif "performance" in _trial_df.columns:
                    _correct_expr = pl.col("performance").cast(pl.Boolean, strict=False).alias("correct_bool")
                else:
                    continue

                frames.append(
                    _trial_df.select(
                        [
                            pl.col("subject").cast(pl.Utf8).alias("subject"),
                            pl.col("state_label").cast(pl.Utf8).alias("state_label"),
                            _correct_expr,
                        ]
                    ).with_columns(
                        [
                            pl.lit(_spec["model_label"]).alias("model_label"),
                            pl.lit(_spec["alias"]).alias("model_alias"),
                        ]
                    )
                )

        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame(schema=trial_schema)

    def build_subject_accuracy(trial_df: pl.DataFrame):
        if trial_df.is_empty() or "state_label" not in trial_df.columns:
            return pl.DataFrame(schema=accuracy_schema)

        working_df = trial_df
        if "correct_bool" not in working_df.columns:
            if "performance" not in working_df.columns:
                return pl.DataFrame(schema=accuracy_schema)
            working_df = working_df.with_columns(
                pl.col("performance").cast(pl.Boolean).alias("correct_bool")
            )

        return (
            working_df
            .filter(pl.col("state_label").is_not_null() & pl.col("correct_bool").is_not_null())
            .group_by(["subject", "model_label", "model_alias", "state_label"])
            .agg((pl.col("correct_bool").cast(pl.Float64).mean() * 100.0).alias("accuracy"))
            .sort(["state_label", "model_label", "subject"])
        )

    overall_subject_occupancy = build_subject_occupancy(overall_specs)
    overall_metrics = build_metrics_df(overall_specs)
    overall_trial_df = build_trial_level_df(overall_specs)
    overall_subject_accuracy = build_subject_accuracy(overall_trial_df)
    drug_subject_occupancy = build_subject_occupancy(drug_specs, restrict_subjects=common_drug_subjects)
    drug_metrics = build_metrics_df(drug_specs, restrict_subjects=common_drug_subjects)
    drug_trial_df = build_trial_level_df(drug_specs, restrict_subjects=common_drug_subjects)
    drug_subject_accuracy = build_subject_accuracy(drug_trial_df)
    overall_num_classes = max((int(_spec["adapter"].num_classes) for _spec in overall_specs), default=2)
    drug_num_classes = max((int(_spec["adapter"].num_classes) for _spec in drug_specs), default=2)
    return (
        drug_metrics,
        drug_num_classes,
        drug_subject_accuracy,
        drug_subject_occupancy,
        overall_metrics,
        overall_num_classes,
        overall_subject_accuracy,
        overall_subject_occupancy,
    )


@app.cell
def _(Line2D, add_sig_bars, custom_boxplot, np, plt, sns):
    from matplotlib.colors import to_hex, to_rgb

    def _darken(color: str, factor: float = 0.75) -> str:
        rgb = np.array(to_rgb(color))
        return to_hex(np.clip(rgb * factor, 0, 1))

    def _palette_for(hue_order: list[str]):
        base = sns.color_palette("tab10", n_colors=max(1, len(hue_order)))
        palette = {_label: to_hex(base[_idx]) for _idx, _label in enumerate(hue_order)}
        strip_palette = {_label: _darken(_color, 0.7) for _label, _color in palette.items()}
        return palette, strip_palette

    def _empty_figure(title: str, *, ncols: int = 1):
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), constrained_layout=False)
        axes = np.atleast_1d(axes)
        for _ax in axes:
            _ax.text(0.5, 0.5, "No data", ha="center", va="center")
            _ax.set_axis_off()
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    def plot_grouped_state_boxplot(
        *,
        df_pd,
        hue_order: list[str],
        value_col: str,
        title: str,
        ylabel: str,
        pair_col: str | None = None,
        chance: float | None = None,
        ylim: tuple[float, float] | None = None,
    ):
        if df_pd.empty:
            return _empty_figure(title)

        state_order = list(dict.fromkeys(df_pd["state_label"].tolist()))
        palette, _ = _palette_for(hue_order)
        hue_width = 0.8 / max(1, len(hue_order))

        fig, ax = plt.subplots(
            figsize=(max(6.0, 1.9 * max(1, len(state_order))), 4.5),
            constrained_layout=False,
        )
        for _x_idx, _state in enumerate(state_order):
            _state_df = df_pd[df_pd["state_label"] == _state]
            _positions = [
                _x_idx + (_hue_idx - (len(hue_order) - 1) / 2) * hue_width
                for _hue_idx in range(len(hue_order))
            ]
            _grouped_values = [
                _state_df[_state_df["model_label"] == _label][value_col].dropna().to_numpy(dtype=float)
                for _label in hue_order
            ]
            _line_values = None
            if pair_col is not None and pair_col in _state_df.columns:
                _line_values = (
                    _state_df.pivot_table(
                        index=pair_col,
                        columns="model_label",
                        values=value_col,
                        aggfunc="first",
                    )
                    .reindex(columns=hue_order)
                    .to_numpy(dtype=float)
                )
            custom_boxplot(
                ax,
                _grouped_values,
                positions=_positions,
                widths=hue_width * 0.9,
                median_colors=[palette[_label] for _label in hue_order],
                line_values=_line_values,
                line_color="#B0B0B0",
                line_alpha=0.35,
                line_linewidth=1.0,
                showfliers=False,
                showcaps=False,
                zorder=1,
            )

        add_sig_bars(
            ax,
            df_pd,
            x_col="state_label",
            y_col=value_col,
            hue_col="model_label",
            order=state_order,
            hue_order=hue_order,
            pair_col=pair_col,
            fallback_to_unpaired=pair_col is not None,
        )

        if chance is not None:
            ax.axhline(chance, color="#7A7A7A", linestyle="--", linewidth=1.0, alpha=0.85)

        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(range(len(state_order)))
        ax.set_xticklabels(state_order, rotation=20, ha="right")
        ax.set_xlim(-0.5, len(state_order) - 0.5)
        if ylim is not None:
            ax.set_ylim(ylim)
        sns.despine(ax=ax)

        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=palette[_label], label=_label, markersize=6)
            for _label in hue_order
        ]
        fig.legend(
            handles,
            hue_order,
            title="Model",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, max(1, len(hue_order))),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        return fig

    def plot_grouped_metric_boxplots(
        *,
        df_pd,
        hue_order: list[str],
        title: str,
        pair_col: str | None = None,
    ):
        if df_pd.empty:
            return _empty_figure(title, ncols=2)

        palette, _ = _palette_for(hue_order)
        k_order = sorted(df_pd["K"].dropna().unique().tolist())
        hue_width = 0.8 / max(1, len(hue_order))

        fig, (ax_ll, ax_bic) = plt.subplots(1, 2, figsize=(9.0, 4.5), constrained_layout=False)

        def _draw(ax, y_col: str):
            for _x_idx, _k in enumerate(k_order):
                _k_df = df_pd[df_pd["K"] == _k]
                _positions = [
                    _x_idx + (_hue_idx - (len(hue_order) - 1) / 2) * hue_width
                    for _hue_idx in range(len(hue_order))
                ]
                _grouped_values = [
                    _k_df[_k_df["model_label"] == _label][y_col].dropna().to_numpy(dtype=float)
                    for _label in hue_order
                ]
                _line_values = None
                if pair_col is not None and pair_col in _k_df.columns:
                    _line_values = (
                        _k_df.pivot_table(
                            index=pair_col,
                            columns="model_label",
                            values=y_col,
                            aggfunc="first",
                        )
                        .reindex(columns=hue_order)
                        .to_numpy(dtype=float)
                    )
                custom_boxplot(
                    ax,
                    _grouped_values,
                    positions=_positions,
                    widths=hue_width * 0.9,
                    median_colors=[palette[_label] for _label in hue_order],
                    line_values=_line_values,
                    line_color="#B0B0B0",
                    line_alpha=0.35,
                    line_linewidth=1.0,
                    showfliers=False,
                    showcaps=False,
                    zorder=1,
                )

            add_sig_bars(
                ax,
                df_pd,
                x_col="K",
                y_col=y_col,
                hue_col="model_label",
                order=k_order,
                hue_order=hue_order,
                pair_col=pair_col,
                fallback_to_unpaired=pair_col is not None,
            )

        _draw(ax_ll, "ll_per_trial")
        _draw(ax_bic, "bic")

        ax_ll.set_ylabel("Log-likelihood / trial")
        ax_ll.set_title("LL / trial (higher = better)")
        ax_bic.set_ylabel("BIC")
        ax_bic.set_title("BIC (lower = better)")
        ax_bic.axhline(0, color="#B0B0B0", lw=0.9, ls="--", alpha=0.8)
        for _ax in (ax_ll, ax_bic):
            _ax.set_xlabel("K")
            _ax.set_xticks(range(len(k_order)))
            _ax.set_xticklabels(k_order)
            _ax.set_xlim(-0.5, len(k_order) - 0.5)
            sns.despine(ax=_ax)

        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=palette[_label], label=_label, markersize=6)
            for _label in hue_order
        ]
        fig.legend(
            handles,
            hue_order,
            title="Model",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, max(1, len(hue_order))),
            frameon=False,
        )
        fig.suptitle(title, y=1.02)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        return fig

    return plot_grouped_metric_boxplots, plot_grouped_state_boxplot


@app.cell
def _(
    mo,
    overall_hue_order,
    overall_k,
    overall_metrics,
    overall_num_classes,
    overall_subject_accuracy,
    overall_subject_occupancy,
    plot_grouped_metric_boxplots,
    plot_grouped_state_boxplot,
    save_plot,
):
    overall_occ_fig = plot_grouped_state_boxplot(
        df_pd=overall_subject_occupancy.to_pandas(),
        hue_order=overall_hue_order,
        value_col="occupancy",
        title="Overall occupancy by state",
        ylabel="Fractional occupancy",
        pair_col="subject",
        chance=1.0 / max(1, overall_k),
        ylim=(0, 1),
    )
    overall_acc_fig = plot_grouped_state_boxplot(
        df_pd=overall_subject_accuracy.to_pandas(),
        hue_order=overall_hue_order,
        value_col="accuracy",
        title="Overall accuracy by state",
        ylabel="Accuracy (%)",
        pair_col="subject",
        chance=100.0 / max(1, overall_num_classes),
        ylim=(0, 100),
    )
    overall_metric_fig = plot_grouped_metric_boxplots(
        df_pd=overall_metrics.to_pandas(),
        hue_order=overall_hue_order,
        title="Overall LL/BIC comparison",
        pair_col="subject",
    )
    mo.vstack(
        [
            mo.md(
                "### Overall comparison\n\n"
                "This section compares the four selected models as cohort-level groups. "
                "Because `2AFC` and `2AFC_DRUG` use different subject pools, significance bars here use unpaired tests."
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            overall_occ_fig,
                            save_plot(overall_occ_fig, "Save overall occupancy", stem="overall_occupancy"),
                        ],
                        align="center",
                    ),
                    mo.vstack(
                        [
                            overall_acc_fig,
                            save_plot(overall_acc_fig, "Save overall accuracy", stem="overall_state_accuracy"),
                        ],
                        align="center",
                    ),
                ]
            ),
            overall_metric_fig,
            save_plot(overall_metric_fig, "Save overall LL/BIC", stem="overall_ll_bic"),
        ]
    )
    return


@app.cell
def _(
    common_drug_subjects,
    drug_hue_order,
    drug_metrics,
    drug_num_classes,
    drug_subject_accuracy,
    drug_subject_occupancy,
    mo,
    plot_grouped_metric_boxplots,
    plot_grouped_state_boxplot,
    save_plot,
):
    if not common_drug_subjects:
        result = mo.md("### Drug-only matched comparison\n\nNo common cached subjects were found across the three drug aliases.")
    else:
        drug_occ_fig = plot_grouped_state_boxplot(
            df_pd=drug_subject_occupancy.to_pandas(),
            hue_order=drug_hue_order,
            value_col="occupancy",
            title="Drug-only occupancy by state",
            ylabel="Fractional occupancy",
            pair_col="subject",
            chance=None,
            ylim=(0, 1),
        )
        drug_acc_fig = plot_grouped_state_boxplot(
            df_pd=drug_subject_accuracy.to_pandas(),
            hue_order=drug_hue_order,
            value_col="accuracy",
            title="Drug-only accuracy by state",
            ylabel="Accuracy (%)",
            pair_col="subject",
            chance=100.0 / max(1, drug_num_classes),
            ylim=(0, 100),
        )
        drug_metric_fig = plot_grouped_metric_boxplots(
            df_pd=drug_metrics.to_pandas(),
            hue_order=drug_hue_order,
            title="Drug-only LL/BIC comparison",
            pair_col="subject",
        )
        result = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.vstack(
                            [
                                drug_occ_fig,
                                save_plot(drug_occ_fig, "Save drug-only occupancy", stem="drug_only_occupancy"),
                            ],
                            align="center",
                        ),
                        mo.vstack(
                            [
                                drug_acc_fig,
                                save_plot(drug_acc_fig, "Save drug-only accuracy", stem="drug_only_state_accuracy"),
                            ],
                            align="center",
                        ),
                    ]
                ),
                drug_metric_fig,
                save_plot(drug_metric_fig, "Save drug-only LL/BIC", stem="drug_only_ll_bic"),
            ]
        )
    result
    return


if __name__ == "__main__":
    app.run()
