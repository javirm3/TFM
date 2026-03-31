# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy",
#   "polars",
#   "matplotlib",
#   "seaborn",
#   "pandas",
#   "scipy",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn as sns
    from matplotlib.lines import Line2D
    from scipy.stats import ttest_rel
    from glmhmmt.runtime import get_runtime_paths

    paths = get_runtime_paths()
    from glmhmmt.postprocess import build_trial_df
    from glmhmmt.views import build_views, get_state_palette
    from glmhmmt.tasks import get_adapter, get_task_options

    sns.set_style("white")
    return (
        Line2D,
        build_trial_df,
        build_views,
        get_adapter,
        get_state_palette,
        get_task_options,
        mo,
        np,
        paths,
        pd,
        pl,
        plt,
        sns,
        ttest_rel,
    )


@app.cell
def _(get_task_options, mo):
    _task_options = get_task_options()
    _task_map = {opt["label"]: opt["value"] for opt in _task_options}
    _default_task = "2AFC" if "2AFC" in _task_map.values() else next(iter(_task_map.values()))
    ui_task = mo.ui.dropdown(
        options=_task_map,
        value=_default_task,
        label="Task",
    )
    return (ui_task,)


@app.cell
def _(paths, pl):
    def model_aliases(task: str) -> list[str]:
        fit_root = paths.RESULTS / "fits" / task / "glmhmm"
        if not fit_root.exists():
            return []
        return sorted([child.name for child in fit_root.iterdir() if child.is_dir()])

    def load_metrics_dir(task: str, alias: str | None):
        if not alias:
            return None
        fit_dir = paths.RESULTS / "fits" / task / "glmhmm" / alias
        if not fit_dir.exists():
            return None

        files = sorted(fit_dir.glob("*_metrics.parquet"))
        if not files:
            return None

        frames = []
        for _path in files:
            try:
                frames.append(pl.read_parquet(_path))
            except Exception:
                pass
        if not frames:
            return None

        df = pl.concat(frames, how="diagonal")
        if "nll" in df.columns and "ll_per_trial" not in df.columns:
            df = df.with_columns(
                (-pl.col("nll") / pl.col("n_trials")).alias("ll_per_trial")
            )
        if "K" not in df.columns:
            df = df.with_columns(pl.lit(1, dtype=pl.Int64).alias("K"))
        else:
            df = df.with_columns(pl.col("K").cast(pl.Int64))
        return df.with_columns(
            [
                pl.lit(alias).alias("model_alias"),
                pl.lit(f"GLMHMM ({alias})").alias("model_label"),
            ]
        )

    def model_k_options(task: str, alias: str | None) -> list[int]:
        df = load_metrics_dir(task, alias)
        if df is None or df.is_empty():
            return []
        return sorted({int(v) for v in df["K"].drop_nulls().to_list()})

    return load_metrics_dir, model_aliases, model_k_options


@app.cell
def _(get_adapter, mo, paths, pl, ui_task):
    adapter = get_adapter(ui_task.value)
    df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df_all = adapter.subject_filter(df_all)
    all_subjects = df_all["subject"].unique().sort().to_list()

    ui_subjects = mo.ui.multiselect(
        options=all_subjects,
        value=all_subjects,
        label="Subjects",
    )

    scoring_options = list(adapter._SCORING_OPTIONS.keys()) if hasattr(adapter, "_SCORING_OPTIONS") else ["default"]
    default_scoring = getattr(adapter, "scoring_key", scoring_options[0]) if scoring_options else None
    if scoring_options and default_scoring not in scoring_options:
        default_scoring = scoring_options[0]
    ui_scoring_key = mo.ui.dropdown(
        options=scoring_options,
        value=default_scoring,
        label="State scoring regressor",
    )
    return df_all, ui_scoring_key, ui_subjects


@app.cell
def _(mo, model_aliases, ui_task):
    aliases = model_aliases(ui_task.value)
    default_a = aliases[0] if aliases else None
    default_b = aliases[1] if len(aliases) > 1 else default_a
    ui_alias_a = mo.ui.dropdown(
        options=aliases,
        value=default_a,
        label="Model A",
    )
    ui_alias_b = mo.ui.dropdown(
        options=aliases,
        value=default_b,
        label="Model B",
    )
    return ui_alias_a, ui_alias_b


@app.cell
def _(mo, model_k_options, ui_alias_a, ui_alias_b, ui_task):
    k_options = sorted(
        set(model_k_options(ui_task.value, ui_alias_a.value))
        & set(model_k_options(ui_task.value, ui_alias_b.value))
    )
    ui_pairwise_K = mo.ui.dropdown(
        options=k_options,
        value=k_options[0] if k_options else None,
        label="Shared K",
    )
    return (ui_pairwise_K,)


@app.cell
def _(
    mo,
    ui_alias_a,
    ui_alias_b,
    ui_pairwise_K,
    ui_scoring_key,
    ui_subjects,
    ui_task,
):
    mo.vstack(
        [
            mo.md("## GLMHMM Pairwise Comparison"),
            mo.md(
                "Dedicated notebook for comparing two GLMHMM fits, especially useful for frozen-vs-unfrozen variants."
            ),
            mo.hstack([ui_task, ui_pairwise_K]),
            mo.hstack([ui_alias_a, ui_alias_b]),
            mo.hstack([ui_scoring_key]),
            mo.hstack([ui_subjects]),
        ]
    )
    return


@app.cell
def _(build_views, get_adapter, np, paths):
    def load_fit_bundle(task: str, alias: str, K: int, subjects: list[str], scoring_key: str | None):
        adapter = get_adapter(task)
        fit_dir = paths.RESULTS / "fits" / task / "glmhmm" / alias
        arrays_store = {}

        for _subject in subjects:
            candidates = [
                fit_dir / f"{_subject}_K{K}_glmhmm_arrays.npz",
                fit_dir / f"{_subject}_glmhmm_arrays.npz",
            ]
            for _path in candidates:
                if not _path.exists():
                    continue
                _data = dict(np.load(_path, allow_pickle=True))
                arrays_store[_subject] = _data
                break

        if scoring_key is not None and hasattr(adapter, "scoring_key"):
            adapter.scoring_key = scoring_key
        views = build_views(arrays_store, adapter, K, list(arrays_store.keys())) if arrays_store else {}
        return adapter, views

    return (load_fit_bundle,)


@app.cell
def _(
    load_fit_bundle,
    load_metrics_dir,
    mo,
    pl,
    ui_alias_a,
    ui_alias_b,
    ui_pairwise_K,
    ui_scoring_key,
    ui_subjects,
    ui_task,
):
    mo.stop(
        not ui_alias_a.value or not ui_alias_b.value,
        mo.md("Select two GLMHMM aliases above."),
    )
    mo.stop(
        ui_alias_a.value == ui_alias_b.value,
        mo.md("Choose two different GLMHMM aliases."),
    )
    mo.stop(
        ui_pairwise_K.value is None,
        mo.md("No shared K values were found for the selected aliases."),
    )

    pairwise_alias_a = ui_alias_a.value
    pairwise_alias_b = ui_alias_b.value
    pairwise_K = int(ui_pairwise_K.value)
    requested_subjects = list(ui_subjects.value)

    pairwise_adapter_a, pairwise_views_a = load_fit_bundle(
        ui_task.value,
        pairwise_alias_a,
        pairwise_K,
        requested_subjects,
        ui_scoring_key.value,
    )
    pairwise_adapter_b, pairwise_views_b = load_fit_bundle(
        ui_task.value,
        pairwise_alias_b,
        pairwise_K,
        requested_subjects,
        ui_scoring_key.value,
    )

    pairwise_common_subjects = [
        _subject
        for _subject in requested_subjects
        if _subject in pairwise_views_a and _subject in pairwise_views_b
    ]
    mo.stop(
        not pairwise_common_subjects,
        mo.md("No common cached subjects were found for these aliases and K."),
    )

    def _filtered_metrics(alias: str):
        df = load_metrics_dir(ui_task.value, alias)
        if df is None:
            return pl.DataFrame()
        return df.filter(
            pl.col("subject").is_in(pairwise_common_subjects)
            & (pl.col("K") == pairwise_K)
        )

    pairwise_metrics_a = _filtered_metrics(pairwise_alias_a)
    pairwise_metrics_b = _filtered_metrics(pairwise_alias_b)

    pairwise_notes_md = (
        f"- Comparing `{pairwise_alias_a}` vs `{pairwise_alias_b}` at `K={pairwise_K}`.\n"
        f"- Common cached subjects: **{len(pairwise_common_subjects)} / {len(requested_subjects)}**.\n"
        f"- State labels are aligned with scoring key `{ui_scoring_key.value}`.\n"
        "- Occupancy below uses posterior fractional occupancy averaged within session, then across sessions within subject.\n"
        "- Accuracy below uses MAP state labels from `build_trial_df`, averaged within subject."
    )
    return (
        pairwise_K,
        pairwise_adapter_a,
        pairwise_adapter_b,
        pairwise_alias_a,
        pairwise_alias_b,
        pairwise_common_subjects,
        pairwise_metrics_a,
        pairwise_metrics_b,
        pairwise_notes_md,
        pairwise_views_a,
        pairwise_views_b,
    )


@app.cell
def _(mo, pairwise_notes_md):
    mo.md(pairwise_notes_md)
    return


@app.cell
def _(
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_metrics_a,
    pairwise_metrics_b,
    pl,
):
    pairwise_metric_summary = (
        pl.concat(
            [
                pairwise_metrics_a.with_columns(pl.lit(pairwise_alias_a).alias("model_alias")),
                pairwise_metrics_b.with_columns(pl.lit(pairwise_alias_b).alias("model_alias")),
            ],
            how="diagonal",
        )
        .group_by("model_alias")
        .agg(
            [
                pl.len().alias("n_subjects"),
                pl.mean("ll_per_trial").alias("ll_mean"),
                pl.std("ll_per_trial").alias("ll_std"),
                pl.mean("bic").alias("bic_mean"),
                pl.std("bic").alias("bic_std"),
            ]
        )
        .sort("model_alias")
    )

    pairwise_metric_deltas = (
        pairwise_metrics_a.select(
            [
                "subject",
                pl.col("ll_per_trial").alias("ll_a"),
                pl.col("bic").alias("bic_a"),
            ]
        )
        .join(
            pairwise_metrics_b.select(
                [
                    "subject",
                    pl.col("ll_per_trial").alias("ll_b"),
                    pl.col("bic").alias("bic_b"),
                ]
            ),
            on="subject",
            how="inner",
        )
        .with_columns(
            [
                (pl.col("ll_b") - pl.col("ll_a")).alias("delta_ll_per_trial"),
                (pl.col("bic_b") - pl.col("bic_a")).alias("delta_bic"),
            ]
        )
        .sort("subject")
    )
    return (pairwise_metric_deltas,)


@app.cell
def _():
    # mo.vstack(
    #     [
    #         mo.md("### LL and BIC summary"),
    #         pairwise_metric_summary,
    #         pairwise_metric_deltas,
    #     ]
    # )
    return


@app.cell
def _(pl):
    def subject_behavior_df(df_all, *, subject, sort_col, session_col):
        df_sub = df_all.filter(pl.col("subject") == subject).sort(sort_col)
        if session_col in df_sub.columns:
            df_sub = df_sub.filter(
                pl.col(session_col).count().over(session_col) >= 2
            )
        return df_sub

    return (subject_behavior_df,)


@app.cell
def _(
    build_trial_df,
    df_all,
    pairwise_adapter_a,
    pairwise_adapter_b,
    pairwise_common_subjects,
    pairwise_views_a,
    pairwise_views_b,
    pl,
    subject_behavior_df,
):
    def _build_pairwise_trial_df(adapter, views):
        frames = []
        for _subject in pairwise_common_subjects:
            if _subject not in views:
                continue
            df_sub = subject_behavior_df(
                df_all,
                subject=_subject,
                sort_col=adapter.sort_col,
                session_col=adapter.session_col,
            )
            if df_sub.height != views[_subject].T:
                continue
            frames.append(
                build_trial_df(
                    views[_subject],
                    adapter,
                    df_sub,
                    adapter.behavioral_cols,
                )
            )
        return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()

    pairwise_trial_df_a = _build_pairwise_trial_df(pairwise_adapter_a, pairwise_views_a)
    pairwise_trial_df_b = _build_pairwise_trial_df(pairwise_adapter_b, pairwise_views_b)
    return pairwise_trial_df_a, pairwise_trial_df_b


@app.cell
def _(
    df_all,
    np,
    pairwise_adapter_a,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_common_subjects,
    pairwise_views_a,
    pairwise_views_b,
    pl,
    subject_behavior_df,
):
    def _session_occupancy_records(alias: str, views: dict):
        records = []
        for _subject in pairwise_common_subjects:
            if _subject not in views:
                continue
            _view = views[_subject]
            df_sub = subject_behavior_df(
                df_all,
                subject=_subject,
                sort_col=pairwise_adapter_a.sort_col,
                session_col=pairwise_adapter_a.session_col,
            )
            if df_sub.height != _view.T:
                continue
            sessions = np.asarray(df_sub[pairwise_adapter_a.session_col])
            probs = np.asarray(_view.smoothed_probs, dtype=float)
            for _session in np.unique(sessions):
                mask = sessions == _session
                for _state_idx in _view.state_idx_order:
                    records.append(
                        {
                            "subject": str(_subject),
                            "model_alias": alias,
                            "state_label": _view.state_name_by_idx.get(int(_state_idx), f"State {_state_idx}"),
                            "occupancy": float(np.mean(probs[mask, int(_state_idx)])),
                        }
                    )
        return records

    occupancy_records = _session_occupancy_records(pairwise_alias_a, pairwise_views_a)
    occupancy_records += _session_occupancy_records(pairwise_alias_b, pairwise_views_b)
    pairwise_subject_occupancy = (
        pl.DataFrame(occupancy_records)
        .group_by(["subject", "model_alias", "state_label"])
        .agg(pl.mean("occupancy").alias("occupancy"))
        .sort(["state_label", "model_alias", "subject"])
        if occupancy_records
        else pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "model_alias": pl.Utf8,
                "state_label": pl.Utf8,
                "occupancy": pl.Float64,
            }
        )
    )

    pairwise_occupancy_summary = (
        pairwise_subject_occupancy
        .group_by(["model_alias", "state_label"])
        .agg(
            [
                pl.len().alias("n_subjects"),
                pl.mean("occupancy").alias("occupancy_mean"),
                pl.std("occupancy").alias("occupancy_std"),
            ]
        )
        .with_columns(
            (pl.col("occupancy_std") / pl.col("n_subjects").sqrt()).alias("occupancy_sem")
        )
        .sort(["state_label", "model_alias"])
    )
    return (pairwise_subject_occupancy,)


@app.cell
def _(
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_trial_df_a,
    pairwise_trial_df_b,
    pl,
):
    schema = {
        "subject": pl.Utf8,
        "model_alias": pl.Utf8,
        "state_label": pl.Utf8,
        "accuracy": pl.Float64,
    }

    def _subject_accuracy(alias: str, df):
        if df.is_empty() or "state_label" not in df.columns:
            return pl.DataFrame(schema=schema)
        working_df = df
        if "correct_bool" not in working_df.columns:
            if "performance" not in working_df.columns:
                return pl.DataFrame(schema=schema)
            working_df = working_df.with_columns(
                pl.col("performance").cast(pl.Boolean).alias("correct_bool")
            )
        return (
            working_df
            .filter(pl.col("state_label").is_not_null() & pl.col("correct_bool").is_not_null())
            .group_by(["subject", "state_label"])
            .agg((pl.col("correct_bool").cast(pl.Float64).mean() * 100.0).alias("accuracy"))
            .with_columns(pl.lit(alias).alias("model_alias"))
            .select(["subject", "model_alias", "state_label", "accuracy"])
        )

    frames = []
    acc_a = _subject_accuracy(pairwise_alias_a, pairwise_trial_df_a)
    acc_b = _subject_accuracy(pairwise_alias_b, pairwise_trial_df_b)
    if not acc_a.is_empty():
        frames.append(acc_a)
    if not acc_b.is_empty():
        frames.append(acc_b)
    pairwise_subject_accuracy = pl.concat(frames, how="diagonal") if frames else pl.DataFrame(schema=schema)

    pairwise_accuracy_summary = (
        pairwise_subject_accuracy
        .group_by(["model_alias", "state_label"])
        .agg(
            [
                pl.len().alias("n_subjects"),
                pl.mean("accuracy").alias("accuracy_mean"),
                pl.std("accuracy").alias("accuracy_std"),
            ]
        )
        .with_columns(
            (pl.col("accuracy_std") / pl.col("n_subjects").sqrt()).alias("accuracy_sem")
        )
        .sort(["state_label", "model_alias"])
    )
    return (pairwise_subject_accuracy,)


@app.cell
def _(Line2D, np, plt, sns, ttest_rel):
    def _p_to_stars(pval: float) -> str:
        if not np.isfinite(pval):
            return ""
        if pval < 0.001:
            return "***"
        if pval < 0.01:
            return "**"
        if pval < 0.05:
            return "*"
        return ""

    def plot_paired_category_boxplot(
        *,
        df_pd,
        alias_a: str,
        alias_b: str,
        value_col: str,
        category_col: str,
        title: str,
        ylabel: str,
        chance: float | None = None,
        ylim: tuple[float, float] | None = None,
    ):
        palette = {
            alias_a: "#1B6CA8",
            alias_b: "#C76D3A",
        }
        state_order = list(dict.fromkeys(df_pd[category_col].tolist())) if not df_pd.empty else []
        models = [alias_a, alias_b]
        offsets = np.linspace(-0.18, 0.18, len(models))
        width = 0.26
        rng = np.random.default_rng(42)

        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=False)
        if df_pd.empty or not state_order:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            return fig

        if chance is not None:
            ax.axhline(chance, color="#7A7A7A", linestyle="--", linewidth=1.0, alpha=0.85)

        y_min = float(df_pd[value_col].min())
        y_max = float(df_pd[value_col].max())
        y_span = max(y_max - y_min, 1.0)
        line_pad = 0.08 * y_span

        for state_idx, state_label in enumerate(state_order):
            rows_a = df_pd[
                (df_pd[category_col] == state_label)
                & (df_pd["model_alias"] == alias_a)
            ][["subject", value_col]]
            rows_b = df_pd[
                (df_pd[category_col] == state_label)
                & (df_pd["model_alias"] == alias_b)
            ][["subject", value_col]]
            paired = rows_a.merge(rows_b, on="subject", how="inner", suffixes=("_a", "_b"))
            paired_jitter = rng.uniform(-0.035, 0.035, size=len(paired)) if len(paired) > 0 else np.array([])

            for model_idx, model_alias in enumerate(models):
                rows = df_pd[
                    (df_pd[category_col] == state_label)
                    & (df_pd["model_alias"] == model_alias)
                ]
                if rows.empty:
                    continue

                values = rows[value_col].to_numpy(dtype=float)
                pos = state_idx + offsets[model_idx]
                box = ax.boxplot(
                    values,
                    positions=[pos],
                    widths=width,
                    patch_artist=True,
                    showfliers=False,
                    zorder=1,
                )
                for patch in box["boxes"]:
                    patch.set(facecolor="white", edgecolor="#666666", linewidth=1.1)
                for elem in ("whiskers", "caps"):
                    for artist in box[elem]:
                        artist.set(color="#666666", linewidth=1.0)
                for median in box["medians"]:
                    median.set(color=palette[model_alias], linewidth=2.2)

                paired_col = f"{value_col}_a" if model_alias == alias_a else f"{value_col}_b"
                if not paired.empty:
                    paired_x = pos + paired_jitter
                    ax.scatter(
                        paired_x,
                        paired[paired_col].to_numpy(dtype=float),
                        color=palette[model_alias],
                        alpha=0.55,
                        s=24,
                        zorder=4,
                    )

                paired_subjects = set(paired["subject"].tolist()) if not paired.empty else set()
                unpaired = rows[~rows["subject"].isin(paired_subjects)]
                jitter = rng.uniform(-0.035, 0.035, size=len(unpaired))
                ax.scatter(
                    np.full(len(unpaired), pos) + jitter,
                    unpaired[value_col].to_numpy(dtype=float),
                    color=palette[model_alias],
                    alpha=0.45,
                    s=24,
                    zorder=3,
                )

            if not paired.empty:
                x_a = state_idx + offsets[0] + paired_jitter
                x_b = state_idx + offsets[1] + paired_jitter
                y_a = paired[f"{value_col}_a"].to_numpy(dtype=float)
                y_b = paired[f"{value_col}_b"].to_numpy(dtype=float)
                for xa, xb, ya, yb in zip(x_a, x_b, y_a, y_b, strict=False):
                    ax.plot([xa, xb], [ya, yb], color="#B0B0B0", linewidth=0.9, alpha=0.7, zorder=2)

                if len(paired) >= 2:
                    _, pval = ttest_rel(y_a, y_b, nan_policy="omit")
                    stars = _p_to_stars(float(pval))
                    if stars:
                        line_y = max(np.nanmax(y_a), np.nanmax(y_b)) + line_pad
                        ax.plot(
                            [state_idx + offsets[0], state_idx + offsets[1]],
                            [line_y, line_y],
                            color="black",
                            linewidth=1.0,
                            zorder=5,
                        )
                        ax.text(state_idx, line_y + 0.02 * y_span, stars, ha="center", va="bottom", fontsize=10)

        ax.set_xticks(range(len(state_order)))
        ax.set_xticklabels(state_order, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        if ylim is not None:
            ax.set_ylim(ylim[0], max(ylim[1], y_max + 2.5 * line_pad))
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=palette[model], label=model, markersize=6)
            for model in models
        ]
        fig.legend(
            handles,
            models,
            title="Model",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
        )
        sns.despine(ax=ax)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        return fig

    def plot_paired_metric_boxplot(
        *,
        df_pd,
        alias_a: str,
        alias_b: str,
        value_a_col: str,
        value_b_col: str,
        title: str,
        ylabel: str,
    ):
        palette = {
            alias_a: "#1B6CA8",
            alias_b: "#C76D3A",
        }
        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=False)
        if df_pd.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            return fig

        y_a = df_pd[value_a_col].to_numpy(dtype=float)
        y_b = df_pd[value_b_col].to_numpy(dtype=float)
        y_all = np.concatenate([y_a, y_b]) if len(df_pd) else np.array([0.0])
        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        y_span = max(y_max - y_min, 1.0)
        line_pad = 0.08 * y_span
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.04, 0.04, size=len(df_pd))

        for pos, model_alias, values in [(0, alias_a, y_a), (1, alias_b, y_b)]:
            box = ax.boxplot(
                values,
                positions=[pos],
                widths=0.34,
                patch_artist=True,
                showfliers=False,
                zorder=1,
            )
            for patch in box["boxes"]:
                patch.set(facecolor="white", edgecolor="#666666", linewidth=1.1)
            for elem in ("whiskers", "caps"):
                for artist in box[elem]:
                    artist.set(color="#666666", linewidth=1.0)
            for median in box["medians"]:
                median.set(color=palette[model_alias], linewidth=2.2)
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                color=palette[model_alias],
                alpha=0.45,
                s=24,
                zorder=3,
            )

        for xj, ya, yb in zip(jitter, y_a, y_b, strict=False):
            ax.plot([0 + xj, 1 + xj], [ya, yb], color="#B0B0B0", linewidth=0.9, alpha=0.7, zorder=2)

        if len(df_pd) >= 2:
            _, pval = ttest_rel(y_a, y_b, nan_policy="omit")
            stars = _p_to_stars(float(pval))
            if stars:
                line_y = max(np.nanmax(y_a), np.nanmax(y_b)) + line_pad
                ax.plot([0, 1], [line_y, line_y], color="black", linewidth=1.0, zorder=5)
                ax.text(0.5, line_y + 0.02 * y_span, stars, ha="center", va="bottom", fontsize=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([alias_a, alias_b], rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
        fig.tight_layout()
        return fig

    return plot_paired_category_boxplot, plot_paired_metric_boxplot


@app.cell
def _(
    pairwise_K,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_subject_occupancy,
    plot_paired_category_boxplot,
):
    occ_fig = plot_paired_category_boxplot(
        df_pd=pairwise_subject_occupancy.to_pandas(),
        alias_a=pairwise_alias_a,
        alias_b=pairwise_alias_b,
        value_col="occupancy",
        category_col="state_label",
        title="Mean session occupancy by state",
        ylabel="Fractional occupancy",
        chance=1.0 / max(1, pairwise_K),
        ylim=(0, 1),
    )
    return (occ_fig,)


@app.cell
def _(
    mo,
    occ_fig,
    pairwise_adapter_a,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_subject_accuracy,
    plot_paired_category_boxplot,
):
    mo.stop(pairwise_subject_accuracy.is_empty(), mo.md("### Accuracy\n\nNo state-wise accuracy data could be built."))
    acc_fig = plot_paired_category_boxplot(
        df_pd=pairwise_subject_accuracy.to_pandas(),
        alias_a=pairwise_alias_a,
        alias_b=pairwise_alias_b,
        value_col="accuracy",
        category_col="state_label",
        title="Mean accuracy by state",
        ylabel="Accuracy (%)",
        chance=100.0 / max(1, pairwise_adapter_a.num_classes),
        ylim=(0, 100),
    )
    mo.hstack([mo.vstack([mo.md("### Accuracy"), acc_fig], align = "center") , mo.vstack([mo.md("### Occupancy"), occ_fig], align = "center")])
    return


@app.cell
def _(
    mo,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_metric_deltas,
    plot_paired_metric_boxplot,
):
    ll_fig = plot_paired_metric_boxplot(
        df_pd=pairwise_metric_deltas.to_pandas(),
        alias_a=pairwise_alias_a,
        alias_b=pairwise_alias_b,
        value_a_col="ll_a",
        value_b_col="ll_b",
        title="Log-likelihood per trial",
        ylabel="LL / trial",
    )
    mo.vstack([mo.md("### LL comparison"), ll_fig])
    return


@app.cell
def _(
    mo,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_metric_deltas,
    plot_paired_metric_boxplot,
):
    bic_fig = plot_paired_metric_boxplot(
        df_pd=pairwise_metric_deltas.to_pandas(),
        alias_a=pairwise_alias_a,
        alias_b=pairwise_alias_b,
        value_a_col="bic_a",
        value_b_col="bic_b",
        title="BIC",
        ylabel="BIC",
    )
    mo.vstack([mo.md("### BIC comparison"), bic_fig])
    return


@app.cell
def _(get_state_palette, np, pd, plt, sns):
    def build_emission_records(alias: str, views: dict, *, num_classes: int):
        records = []
        state_order = []
        feature_order = []
        for _subject, _view in views.items():
            weights = np.asarray(_view.emission_weights, dtype=float)
            if weights.ndim == 2:
                weights = weights[:, None, :]
            if num_classes == 2:
                weights = -weights
            rank_order = [int(k) for k in _view.state_idx_order]
            weights = weights[rank_order].mean(axis=1)
            state_labels = [
                _view.state_name_by_idx.get(int(k), f"State {k}")
                for k in rank_order
            ]
            feature_names = list(_view.feat_names)
            for _label in state_labels:
                if _label not in state_order:
                    state_order.append(_label)
            for _feature in feature_names:
                if _feature not in feature_order:
                    feature_order.append(_feature)
            for _state_idx, _label in enumerate(state_labels):
                for _feature_idx, _feature in enumerate(feature_names):
                    records.append(
                        {
                            "subject": str(_subject),
                            "model_alias": alias,
                            "state_label": _label,
                            "feature": _feature,
                            "weight": float(weights[_state_idx, _feature_idx]),
                        }
                    )
        return records, state_order, feature_order

    def plot_emission_comparison_boxplots(
        *,
        records_a: list[dict],
        records_b: list[dict],
        alias_a: str,
        alias_b: str,
        state_order: list[str],
        feature_order: list[str],
    ):
        records = records_a + records_b
        df = pd.DataFrame(records)
        if df.empty:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        palette = get_state_palette(len(state_order) or None)
        state_palette = {
            state_label: palette[idx % len(palette)]
            for idx, state_label in enumerate(state_order)
        }
        y_min = float(df["weight"].min())
        y_max = float(df["weight"].max())
        y_pad = 0.08 * max(y_max - y_min, 1.0)

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(max(8.5, 1.2 * len(feature_order) * 2), 4.5),
            sharey=True,
            constrained_layout=False,
        )

        for ax, alias in zip(axes, [alias_a, alias_b], strict=False):
            sub = df[df["model_alias"] == alias]
            sns.boxplot(
                data=sub,
                x="feature",
                y="weight",
                hue="state_label",
                order=feature_order,
                hue_order=state_order,
                palette=state_palette,
                width=0.8,
                showfliers=False,
                boxprops={"alpha": 0.7},
                ax=ax,
            )
            sns.stripplot(
                data=sub,
                x="feature",
                y="weight",
                hue="state_label",
                order=feature_order,
                hue_order=state_order,
                palette=state_palette,
                dodge=True,
                alpha=0.35,
                size=3.0,
                ax=ax,
                legend=False,
            )
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_title(alias)
            ax.set_xlabel("")
            ax.set_ylabel("Weight")
            ax.set_xticklabels(feature_order, rotation=35, ha="right")
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            sns.despine(ax=ax)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles[: len(state_order)],
                labels[: len(state_order)],
                title="State",
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(4, max(1, len(state_order))),
                frameon=False,
            )
        fig.suptitle("Emission weights across subjects", y=1.02)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        return fig

    return build_emission_records, plot_emission_comparison_boxplots


@app.cell
def _(
    build_emission_records,
    mo,
    pairwise_adapter_a,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_views_a,
    pairwise_views_b,
    plot_emission_comparison_boxplots,
):
    records_a, state_order_a, feature_order_a = build_emission_records(
        pairwise_alias_a,
        pairwise_views_a,
        num_classes=pairwise_adapter_a.num_classes,
    )
    records_b, state_order_b, feature_order_b = build_emission_records(
        pairwise_alias_b,
        pairwise_views_b,
        num_classes=pairwise_adapter_a.num_classes,
    )
    state_order = list(dict.fromkeys(state_order_a + state_order_b))
    feature_order = list(dict.fromkeys(feature_order_a + feature_order_b))
    emission_fig = plot_emission_comparison_boxplots(
        records_a=records_a,
        records_b=records_b,
        alias_a=pairwise_alias_a,
        alias_b=pairwise_alias_b,
        state_order=state_order,
        feature_order=feature_order,
    )
    mo.vstack([mo.md("### Emission weights"), emission_fig])
    return


if __name__ == "__main__":
    app.run()
