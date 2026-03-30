# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy",
#   "polars",
#   "matplotlib",
#   "seaborn",
#   "pandas",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys, os
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        import paths
    except ImportError:
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        import paths

    # Add src to sys.path to prevent namespace shadowing by the code/glmhmmt directory
    sys.path.insert(1, str(paths.CODE_DIR / "glmhmmt" / "src"))

    from tasks import get_adapter, get_task_options
    from glmhmmt.postprocess import build_trial_df
    from glmhmmt.views import build_views

    sns.set_style("white")
    return (
        build_trial_df,
        build_views,
        get_adapter,
        get_task_options,
        mo,
        np,
        paths,
        pl,
        plt,
        sns,
    )


@app.cell
def _(get_task_options, mo):
    _task_options = get_task_options()
    ui_task = mo.ui.dropdown(
        options={opt["label"]: opt["value"] for opt in _task_options},
        value="MCDR",
        label="Task",
    )
    return (ui_task,)


@app.cell
def _(paths, pl):
    _MODEL_LABELS = {
        "glm": "GLM",
        "glmhmm": "GLMHMM",
        "glmhmmt": "GLMHMM-T",
    }

    def model_aliases(task: str, kind: str) -> list[str]:
        p = paths.RESULTS / "fits" / task / kind
        if not p.exists():
            return []
        return sorted([d.name for d in p.iterdir() if d.is_dir()])

    def load_metrics_dir(task_name: str, folder_name: str | None, expected_model_kind: str):
        if not folder_name:
            return None
        d = paths.RESULTS / "fits" / task_name / expected_model_kind / folder_name
        if not d.exists():
            return None
        files = list(d.glob("*_metrics.parquet"))
        if not files:
            return None

        frames = []
        for f in files:
            try:
                frames.append(pl.read_parquet(f))
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
        if "model_kind" not in df.columns:
            df = df.with_columns(pl.lit(expected_model_kind).alias("model_kind"))
        df = df.with_columns(
            [
                pl.lit(folder_name).alias("model_alias"),
                pl.lit(
                    f"{_MODEL_LABELS.get(expected_model_kind, expected_model_kind)} ({folder_name})"
                ).alias("model_label"),
            ]
        )
        keep = [
            "subject",
            "K",
            "model_kind",
            "model_alias",
            "model_label",
            "ll_per_trial",
            "bic",
            "acc",
        ]
        return df.select([c for c in keep if c in df.columns])

    def model_k_options(task: str, kind: str, alias: str | None) -> list[int]:
        df = load_metrics_dir(task, alias, kind)
        if df is None or df.is_empty():
            return []
        return sorted(
            {
                int(k)
                for k in df["K"].drop_nulls().to_list()
            }
        )

    return load_metrics_dir, model_aliases, model_k_options


@app.cell
def _(get_adapter, mo, model_aliases, ui_task):

    adapter = get_adapter(ui_task.value)

    ui_glm_dir = mo.ui.multiselect(
        options=model_aliases(ui_task.value, "glm"),
        value=[],
        label="GLM aliases",
    )
    ui_glmhmm_dir = mo.ui.multiselect(
        options=model_aliases(ui_task.value, "glmhmm"),
        value=[],
        label="GLMHMM aliases",
    )
    ui_glmhmmt_dir = mo.ui.multiselect(
        options=model_aliases(ui_task.value, "glmhmmt"),
        value=[],
        label="GLMHMM-T aliases",
    )

    mo.vstack([
        mo.md("### Model Comparison — Configuration"),
        mo.md(
            "Select one or more aliases for each model kind. "
            "Leave empty to skip that model."
        ),
        mo.hstack([ui_task]),
        mo.hstack([ui_glm_dir, ui_glmhmm_dir, ui_glmhmmt_dir]),
    ])
    return adapter, ui_glm_dir, ui_glmhmm_dir, ui_glmhmmt_dir


@app.cell
def _(adapter, mo, paths, pl):
    df_all = pl.read_parquet(paths.DATA_PATH / adapter.data_file)
    df_all = adapter.subject_filter(df_all)
    _all_subjects = df_all["subject"].unique().sort().to_list()

    ui_subjects = mo.ui.multiselect(
        options=_all_subjects,
        value=_all_subjects,
        label="Subjects",
    )
    ui_K_range = mo.ui.range_slider(
        start=1, stop=10, step=1, value=[1, 5],
        full_width=True, label="K range",
    )

    mo.vstack([
        mo.hstack([ui_subjects]),
        mo.hstack([mo.md("K range:"), ui_K_range]),
    ])
    return df_all, ui_K_range, ui_subjects


@app.cell
def _(
    load_metrics_dir,
    mo,
    pl,
    ui_glm_dir,
    ui_glmhmm_dir,
    ui_glmhmmt_dir,
    ui_task,
):
    _parts = []
    for _names, _kind in [
        (ui_glm_dir.value, "glm"),
        (ui_glmhmm_dir.value, "glmhmm"),
        (ui_glmhmmt_dir.value, "glmhmmt"),
    ]:
        for _name in _names:
            _p = load_metrics_dir(ui_task.value, _name, _kind)
            if _p is not None:
                _parts.append(_p)

    if _parts:
        results_long = pl.concat(_parts, how="diagonal")
    else:
        results_long = pl.DataFrame(
            schema={
                "subject": pl.Utf8, "K": pl.Int64, "model_kind": pl.Utf8,
                "model_alias": pl.Utf8, "model_label": pl.Utf8,
                "ll_per_trial": pl.Float64, "bic": pl.Float64, "acc": pl.Float64,
            }
        )

    mo.stop(
        results_long.is_empty(),
        mo.md("⚠️  No metrics loaded — select at least one fit folder above."),
    )
    mo.md(
        f"Loaded **{results_long.height}** fit rows from "
        f"**{len(_parts)}** model folder(s)."
    )
    return (results_long,)


@app.cell
def _(pl, results_long, ui_K_range, ui_subjects):
    K_min, K_max = ui_K_range.value
    results_filtered = results_long.filter(
        pl.col("subject").is_in(ui_subjects.value)
        & pl.col("K").is_between(K_min, K_max)
    )
    results_filtered
    return (results_filtered,)


@app.cell
def _(adapter, df_all, mo, pl):
    _enum_dtype = getattr(pl, "Enum", None)
    if getattr(adapter, "num_classes", None) == 3:
        _preferred = [
            "stimd_n",
            "stimd_c",
            "ttype_n",
            "ttype_c",
            "condition",
            "Condition",
            "Experiment",
            adapter.session_col,
        ]
        _default_candidates = ["stimd_n", "stimd_c", "ttype_n", "ttype_c"]
    else:
        _preferred = [
            "ILD",
            "ild",
            "stim_vals",
            "stim_d",
            "stim_strength",
            "condition",
            "Condition",
            "Experiment",
            adapter.session_col,
        ]
        _default_candidates = ["ILD", "ild", "stim_vals", "stim_d", "stim_strength"]
    _seen = set()
    _options = []
    for _col in _preferred:
        if _col in df_all.columns and _col not in _seen:
            _options.append(_col)
            _seen.add(_col)
    for _col, _dtype in df_all.schema.items():
        if _col in _seen or _col == "subject":
            continue
        if _dtype in tuple(
            _dt for _dt in (pl.Utf8, pl.Categorical, _enum_dtype, pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
            if _dt is not None
        ):
            _options.append(_col)
            _seen.add(_col)

    _default = next((_col for _col in _default_candidates if _col in _options), None)
    if _default is None:
        _default = "condition" if "condition" in _options else (_options[0] if _options else None)
    ui_ce_condition = mo.ui.dropdown(
        options=_options,
        value=_default,
        label="Cross-entropy grouping",
    )
    mo.hstack([ui_ce_condition])
    return (ui_ce_condition,)


@app.cell
def _(mo, results_filtered):
    _baseline_options = results_filtered["model_label"].unique().sort().to_list()
    _baseline_value = _baseline_options[0] if _baseline_options else None
    ui_bic_baseline = mo.ui.dropdown(
        options=_baseline_options,
        value=_baseline_value,
        label="BIC baseline model",
    )
    mo.hstack([ui_bic_baseline])
    return (ui_bic_baseline,)


@app.cell
def _(pl, results_filtered, ui_bic_baseline):
    if results_filtered.is_empty() or ui_bic_baseline.value is None:
        results_plot = results_filtered.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("bic_delta")
        )
    else:
        _baseline_bic = (
            results_filtered
            .filter(pl.col("model_label") == ui_bic_baseline.value)
            .group_by("subject")
            .agg(pl.first("bic").alias("bic_baseline"))
        )
        results_plot = (
            results_filtered
            .join(_baseline_bic, on="subject", how="left")
            .with_columns(((pl.col("bic") - pl.col("bic_baseline"))/pl.col("bic_baseline")).alias("bic_delta"))
        )
    results_plot
    return (results_plot,)


@app.cell
def _(np):
    def observed_choice_index(adapter, trial_df):
        _resp = np.asarray(trial_df["response"]).astype(object)
        _out = np.full(len(_resp), -1, dtype=int)

        if adapter.num_classes == 2:
            for _i, _val in enumerate(_resp):
                if _val is None:
                    continue
                try:
                    _f = float(_val)
                    if _f in (0.0, 1.0):
                        _out[_i] = int(_f)
                    elif _f in (-1.0, 1.0):
                        _out[_i] = 1 if _f > 0 else 0
                except (TypeError, ValueError):
                    _s = str(_val).strip().upper()
                    if _s in {"L", "LEFT"}:
                        _out[_i] = 0
                    elif _s in {"R", "RIGHT"}:
                        _out[_i] = 1
        else:
            for _i, _val in enumerate(_resp):
                if _val is None:
                    continue
                try:
                    _f = float(_val)
                    if _f in (0.0, 1.0, 2.0):
                        _out[_i] = int(_f)
                    elif _f in (1.0, 2.0, 3.0):
                        _out[_i] = int(_f) - 1
                except (TypeError, ValueError):
                    _s = str(_val).strip().upper()
                    if _s in {"L", "LEFT"}:
                        _out[_i] = 0
                    elif _s in {"C", "CENTER", "CENTRE"}:
                        _out[_i] = 1
                    elif _s in {"R", "RIGHT"}:
                        _out[_i] = 2
        return _out

    return (observed_choice_index,)


@app.cell
def _(pl, results_filtered):
    agg = (
        results_filtered.group_by(["model_kind", "model_alias", "model_label", "K"])
        .agg([
            pl.len().alias("n_subjects"),
            pl.mean("ll_per_trial").alias("ll_mean"),
            pl.std("ll_per_trial").alias("ll_std"),
            pl.mean("bic").alias("bic_mean"),
            pl.std("bic").alias("bic_std"),
            pl.mean("acc").alias("acc_mean"),
        ])
        .with_columns([
            (pl.col("ll_std")  / pl.col("n_subjects").sqrt()).alias("ll_sem"),
            (pl.col("bic_std") / pl.col("n_subjects").sqrt()).alias("bic_sem"),
        ])
        .sort(["model_kind", "model_alias", "K"])
    )
    agg
    return (agg,)


@app.cell
def _(agg):
    agg
    return


@app.cell
def _():
    import itertools
    import pandas as pd
    from scipy.stats import ttest_rel, ttest_ind

    def _sig_label(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    def add_sig_bars(ax, df, *, x_col, y_col, hue_col, order, hue_order, pair_col=None):
        n_hue = max(1, len(hue_order))
        hue_width = 0.8 / n_hue
        y_range = df[y_col].max() - df[y_col].min()
        if pd.isna(y_range) or y_range == 0:
            y_range = 1.0

        for m, xval in enumerate(order):
            sub = df[df[x_col] == xval]
            if sub.empty:
                continue

            current_y = sub[y_col].max() + y_range * 0.05
            h = y_range * 0.02

            for p1, p2 in itertools.combinations(range(n_hue), 2):
                g1 = hue_order[p1]
                g2 = hue_order[p2]

                s1 = sub[sub[hue_col] == g1]
                s2 = sub[sub[hue_col] == g2]

                if pair_col is not None:
                    v1 = s1.set_index(pair_col)[y_col]
                    v2 = s2.set_index(pair_col)[y_col]
                    common = v1.index.intersection(v2.index)
                    if len(common) < 2:
                        continue
                    _, pval = ttest_rel(v1.loc[common].values, v2.loc[common].values)
                else:
                    v1 = s1[y_col].dropna().values
                    v2 = s2[y_col].dropna().values
                    if min(len(v1), len(v2)) < 2:
                        continue
                    _, pval = ttest_ind(v1, v2, equal_var=False)

                star = _sig_label(pval)
                if star == "ns":
                    continue

                x1 = m + (p1 - (n_hue - 1) / 2) * hue_width
                x2 = m + (p2 - (n_hue - 1) / 2) * hue_width

                ax.plot([x1, x1, x2, x2], [current_y, current_y + h, current_y + h, current_y], lw=1, c="k")
                ax.text((x1 + x2) / 2, current_y + h, star, ha="center", va="bottom", color="k")
                current_y += y_range * 0.075


    return add_sig_bars, ttest_rel


@app.cell
def _(add_sig_bars, np, plt, results_plot, sns, ui_bic_baseline):
    from matplotlib.colors import to_rgb, to_hex

    _MODEL_STYLES = {
        "glm": {"marker": "s", "label": "GLM"},
        "glmhmm": {"marker": "o", "label": "GLMHMM"},
        "glmhmmt": {"marker": "^", "label": "GLMHMM-T"},
    }

    def darken(color, factor=0.75):
        rgb = np.array(to_rgb(color))
        return to_hex(np.clip(rgb * factor, 0, 1))

    raw = results_plot.to_pandas()

    _label_df = raw[["model_kind", "model_label"]].drop_duplicates()
    hue_order = _label_df["model_label"].tolist()
    _base_colors = sns.color_palette("tab20", n_colors=max(1, len(hue_order)))
    palette = {
        _label: to_hex(_base_colors[_i])
        for _i, _label in enumerate(hue_order)
    }
    strip_palette = {
        _label: darken(palette[_label], 0.70)
        for _label in hue_order
    }
    K_order = sorted(raw["K"].unique())

    fig_cmp, (ax_ll, ax_bic) = plt.subplots(1, 2, figsize=(8, 4.8), constrained_layout=False)

    for ax, ycol in [(ax_ll, "ll_per_trial"), (ax_bic, "bic_delta")]:
        sns.boxplot(
            data=raw,
            x="K",
            y=ycol,
            hue="model_label",
            order=K_order,
            hue_order=hue_order,
            palette=palette,
            width=0.8,
            showfliers=False,
            boxprops={"alpha": 0.45},
            ax=ax,
        )

        sns.stripplot(
            data=raw,
            x="K",
            y=ycol,
            hue="model_label",
            order=K_order,
            hue_order=hue_order,
            palette=strip_palette,
            dodge=True,
            jitter=0.18,
            alpha=0.85,
            size=4,
            ax=ax,
            legend=False,
        )

    add_sig_bars(
        ax_ll, raw,
        x_col="K", y_col="ll_per_trial", hue_col="model_label",
        order=K_order, hue_order=hue_order, pair_col="subject",
    )

    add_sig_bars(
        ax_bic, raw,
        x_col="K", y_col="bic_delta", hue_col="model_label",
        order=K_order, hue_order=hue_order, pair_col="subject",
    )

    ax_ll.set_ylabel("Log-likelihood / trial")
    ax_ll.set_title("LL / trial (higher = better)")

    ax_bic.axhline(0, color="grey", lw=0.9, linestyle="--", alpha=0.7)
    ax_bic.set_ylabel("ΔBIC vs baseline")
    ax_bic.set_title(f"ΔBIC vs {ui_bic_baseline.value} (lower = better)")

    handles, labels = ax_ll.get_legend_handles_labels()
    _legend_handles = []
    _legend_labels = []
    for _h, _l in zip(handles, labels):
        if _l in hue_order and _l not in _legend_labels:
            _legend_handles.append(_h)
            _legend_labels.append(_l)
    if ax_ll.get_legend() is not None:
        ax_ll.get_legend().remove()
    if ax_bic.get_legend() is not None:
        ax_bic.get_legend().remove()
    fig_cmp.legend(
        _legend_handles,
        _legend_labels,
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, max(1, len(_legend_labels))),
        frameon=False,
    )

    sns.despine(fig=fig_cmp)
    fig_cmp.tight_layout(rect=(0, 0.12, 1, 1))
    fig_cmp
    return


@app.cell
def _(plt, results_plot, sns, ui_bic_baseline):
    def _cov_group(label):
        _label = str(label).lower()
        if "3 cov" in _label:
            return "3 covs"
        if "2 cov" in _label:
            return "2 covs"
        return None

    _raw = results_plot.to_pandas()
    _cov_mean = (
        _raw.assign(cov_group=_raw["model_label"].map(_cov_group))
        .dropna(subset=["cov_group"])
        .groupby(["cov_group", "K"], as_index=False)[["ll_per_trial", "bic_delta"]]
        .mean()
        .sort_values(["cov_group", "K"])
    )

    _cov_order = [grp for grp in ["2 covs", "3 covs"] if grp in _cov_mean["cov_group"].unique()]
    _palette = {
        _group: _color
        for _group, _color in zip(_cov_order, sns.color_palette("tab10", n_colors=max(1, len(_cov_order))))
    }

    fig_cov_mean, (ax_cov_ll, ax_cov_bic) = plt.subplots(
        1, 2, figsize=(8, 4.2), constrained_layout=False
    )

    for _group in _cov_order:
        _group_df = _cov_mean[_cov_mean["cov_group"] == _group]
        ax_cov_ll.plot(
            _group_df["K"],
            _group_df["ll_per_trial"],
            marker="o",
            linewidth=2,
            color=_palette[_group],
            label=_group,
        )
        ax_cov_bic.plot(
            _group_df["K"],
            _group_df["bic_delta"],
            marker="o",
            linewidth=2,
            color=_palette[_group],
            label=_group,
        )

    ax_cov_ll.set_xlabel("Number of states K")
    ax_cov_ll.set_ylabel("Mean log-likelihood / trial")
    ax_cov_ll.set_title("Mean LL / trial by covariate count")

    ax_cov_bic.axhline(0, color="grey", lw=0.9, linestyle="--", alpha=0.7)
    ax_cov_bic.set_xlabel("Number of states K")
    ax_cov_bic.set_ylabel("Mean ΔBIC vs baseline")
    ax_cov_bic.set_title(f"Mean ΔBIC vs {ui_bic_baseline.value}")

    if _cov_order:
        fig_cov_mean.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=len(_cov_order),
            frameon=False,
            title="Grouped models",
        )

    sns.despine(fig=fig_cov_mean)
    fig_cov_mean.tight_layout(rect=(0, 0.12, 1, 1))
    fig_cov_mean
    return


@app.cell
def _(
    build_trial_df,
    df_all,
    load_fit_bundle,
    np,
    observed_choice_index,
    pl,
    results_filtered,
    ui_ce_condition,
    ui_subjects,
    ui_task,
):
    _cond_col = ui_ce_condition.value
    mo_delim = 1e-12

    if results_filtered.is_empty() or _cond_col is None:
        ce_by_subject_condition = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "condition": pl.Utf8,
                "model_kind": pl.Utf8,
                "model_alias": pl.Utf8,
                "model_label": pl.Utf8,
                "K": pl.Int64,
                "cross_entropy": pl.Float64,
                "n_trials": pl.Int64,
            }
        )
    else:
        _model_specs = (
            results_filtered
            .select(["model_kind", "model_alias", "model_label", "K"])
            .unique()
            .sort(["model_kind", "model_alias", "K"])
            .iter_rows(named=True)
        )
        _frames = []
        for _spec in _model_specs:
            _adapter_fit, _arrays_store, _names, _views = load_fit_bundle(
                ui_task.value,
                _spec["model_kind"],
                _spec["model_alias"],
                int(_spec["K"]),
                ui_subjects.value,
            )
            if not _views:
                continue

            _prob_cols = _adapter_fit.probability_columns
            _bcols = _adapter_fit.behavioral_cols
            _sort_col = _adapter_fit.sort_col
            _ses_col = _adapter_fit.session_col

            for _subj, _view in _views.items():
                _df_sub = (
                    df_all
                    .filter(pl.col("subject") == _subj)
                    .sort(_sort_col)
                    .filter(pl.col(_ses_col).count().over(_ses_col) >= 2)
                )
                if _df_sub.height != _view.T or _cond_col not in _df_sub.columns:
                    continue

                _trial_df = build_trial_df(_view, _adapter_fit, _df_sub, _bcols)
                _choice_idx = observed_choice_index(_adapter_fit, _trial_df)
                _probs = np.column_stack([np.asarray(_trial_df[_c], dtype=float) for _c in _prob_cols])
                _valid = (
                    (_choice_idx >= 0)
                    & (_choice_idx < _probs.shape[1])
                    & np.all(np.isfinite(_probs), axis=1)
                )
                if not np.any(_valid):
                    continue

                _picked = _probs[np.arange(len(_choice_idx)), np.clip(_choice_idx, 0, _probs.shape[1] - 1)]
                _ce = np.full(len(_choice_idx), np.nan, dtype=float)
                _ce[_valid] = -np.log(np.clip(_picked[_valid], mo_delim, 1.0))

                _ce_df = _trial_df.select(["subject", _cond_col]).with_columns([
                    pl.lit(_spec["model_kind"]).alias("model_kind"),
                    pl.lit(_spec["model_alias"]).alias("model_alias"),
                    pl.lit(_spec["model_label"]).alias("model_label"),
                    pl.lit(int(_spec["K"])).alias("K"),
                    pl.Series("cross_entropy", _ce),
                ])
                _ce_df = (
                    _ce_df
                    .filter(pl.col("cross_entropy").is_finite())
                    .with_columns(pl.col(_cond_col).cast(pl.Utf8).alias("condition"))
                    .drop(_cond_col)
                )
                if _ce_df.height > 0:
                    _frames.append(_ce_df)

        if _frames:
            ce_by_subject_condition = (
                pl.concat(_frames, how="diagonal")
                .group_by(["subject", "condition", "model_kind", "model_alias", "model_label", "K"])
                .agg([
                    pl.mean("cross_entropy").alias("cross_entropy"),
                    pl.len().alias("n_trials"),
                ])
                .sort(["K", "condition", "model_kind", "model_alias", "subject"])
            )
        else:
            ce_by_subject_condition = pl.DataFrame(
                schema={
                    "subject": pl.Utf8,
                    "condition": pl.Utf8,
                    "model_kind": pl.Utf8,
                    "model_alias": pl.Utf8,
                    "model_label": pl.Utf8,
                    "K": pl.Int64,
                    "cross_entropy": pl.Float64,
                    "n_trials": pl.Int64,
                }
            )

    ce_by_subject_condition
    return (ce_by_subject_condition,)


@app.cell
def _(ce_by_subject_condition, mo, plt, sns):
    mo.stop(ce_by_subject_condition.is_empty(), mo.md("No trial-level cross-entropy data could be built for the current selection."))

    _ce_raw = ce_by_subject_condition.to_pandas()
    _K_order = sorted(_ce_raw["K"].unique())
    _cond_order = sorted(_ce_raw["condition"].dropna().unique())
    _labels = _ce_raw["model_label"].drop_duplicates().tolist()
    _base_colors = sns.color_palette("tab20", n_colors=max(1, len(_labels)))
    _palette = {_label: _base_colors[_i] for _i, _label in enumerate(_labels)}

    _fig_ce, _axes = plt.subplots(
        len(_K_order),
        1,
        figsize=(max(7, 1.4 * len(_cond_order)), 3.8 * max(1, len(_K_order))),
        squeeze=False,
    )

    for _row, _K in enumerate(_K_order):
        _ax = _axes[_row, 0]
        _sub = _ce_raw[_ce_raw["K"] == _K]
        sns.boxplot(
            data=_sub,
            x="condition",
            y="cross_entropy",
            hue="model_label",
            order=_cond_order,
            hue_order=_labels,
            palette=_palette,
            width=0.8,
            showfliers=False,
            boxprops={"alpha": 0.45},
            ax=_ax,
        )
        sns.stripplot(
            data=_sub,
            x="condition",
            y="cross_entropy",
            hue="model_label",
            order=_cond_order,
            hue_order=_labels,
            palette=_palette,
            dodge=True,
            jitter=0.18,
            alpha=0.75,
            size=3.5,
            ax=_ax,
            legend=False,
        )
        _ax.set_title(f"Cross-entropy by condition (K={_K})")
        _ax.set_xlabel("Condition")
        _ax.set_ylabel("Cross-entropy")
        _ax.tick_params(axis="x", rotation=20)
        if _ax.get_legend() is not None:
            _ax.get_legend().remove()
        sns.despine(ax=_ax)

    _handles, _legend_labels = _axes[0, 0].get_legend_handles_labels()
    _handles_out = []
    _labels_out = []
    for _h, _l in zip(_handles, _legend_labels):
        if _l in _labels and _l not in _labels_out:
            _handles_out.append(_h)
            _labels_out.append(_l)
    if _handles_out:
        _fig_ce.legend(
            _handles_out,
            _labels_out,
            title="Model",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(3, max(1, len(_labels_out))),
            frameon=False,
        )
    _fig_ce.tight_layout(rect=(0, 0.08, 1, 1))
    _fig_ce
    return


@app.cell
def _(mo, pl, plt, results_filtered, sns):
    _pivot_df = (
        results_filtered
        .with_columns(
            (pl.col("model_label") + "_K" + pl.col("K").cast(pl.Utf8)).alias("model_K")
        )
        .pivot(index="subject", on="model_K", values="ll_per_trial")
        .to_pandas()
        .set_index("subject")
    )

    mo.stop(_pivot_df.empty, mo.md("No data to plot."))

    _fig_heat, _ax_h = plt.subplots(
        figsize=(max(6, _pivot_df.shape[1] * 0.9), max(4, _pivot_df.shape[0] * 0.4))
    )
    sns.heatmap(
        _pivot_df, ax=_ax_h, cmap="RdYlGn",
        annot=True, fmt=".3f", linewidths=0.3,
        cbar_kws={"label": "LL / trial"},
    )
    _ax_h.set_title("Log-likelihood per trial — subject × model/K")
    _ax_h.set_xlabel("")
    _fig_heat.tight_layout()
    _fig_heat
    return


@app.cell
def _(agg, plt, sns):
    _MODEL_STYLES = {
        "glm": {"marker": "s", "label": "GLM"},
        "glmhmm": {"marker": "o", "label": "GLMHMM"},
        "glmhmmt": {"marker": "^", "label": "GLMHMM-T"},
    }

    fig_acc, _ax_acc = plt.subplots(figsize=(6, 4))
    _labels = agg["model_label"].unique().to_list()
    _colors = sns.color_palette("tab20", n_colors=max(1, len(_labels)))
    _palette = {_label: _colors[_i] for _i, _label in enumerate(_labels)}
    for _label_tup, _group in agg.group_by("model_label"):
        _label = _label_tup[0]
        _g = _group.sort("K").to_pandas()
        _kind = _g["model_kind"].iloc[0]
        _st = _MODEL_STYLES.get(_kind, {"marker": "o", "label": _label})
        _ax_acc.plot(
            _g["K"], _g["acc_mean"],
            color=_palette[_label], marker=_st["marker"],
            label=_label, linewidth=1.5,
        )

    _ax_acc.set_xlabel("Number of states K")
    _ax_acc.set_ylabel("Accuracy (mean over subjects)")
    _ax_acc.set_title("Model accuracy vs K")
    _ax_acc.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(3, max(1, len(_labels))),
    )
    sns.despine(ax=_ax_acc)
    fig_acc.tight_layout(rect=(0, 0.08, 1, 1))
    fig_acc
    return


@app.cell
def _(build_views, get_adapter, np, paths):
    def load_fit_bundle(task_name, model_kind, alias, K, subjects, scoring_key=None):
        adapter = get_adapter(task_name)
        fit_dir = paths.RESULTS / "fits" / task_name / model_kind / alias
        suffix = {
            "glm": "glm",
            "glmhmm": "glmhmm",
            "glmhmmt": "glmhmmt",
            "glmhmm-t": "glmhmmt",
        }[model_kind]

        arrays_store = {}
        for _subj in subjects:
            candidates = []
            if model_kind == "glm":
                candidates.append(fit_dir / f"{_subj}_glm_arrays.npz")
            else:
                candidates.extend(
                    [
                        fit_dir / f"{_subj}_{suffix}_arrays.npz",
                        fit_dir / f"{_subj}_K{K}_{suffix}_arrays.npz",
                    ]
                )
            for _path in candidates:
                if not _path.exists():
                    continue
                _data = dict(np.load(_path, allow_pickle=True))
                _saved_names = {}
                if "names" in _data and getattr(_data["names"], "shape", None) == ():
                    _saved_names = _data["names"].item()
                if "X_cols" in _data:
                    _data["X_cols"] = list(_data["X_cols"])
                elif "X_cols" in _saved_names:
                    _data["X_cols"] = list(_saved_names["X_cols"])
                if "U_cols" in _data:
                    _data["U_cols"] = list(_data["U_cols"])
                elif "U_cols" in _saved_names:
                    _data["U_cols"] = list(_saved_names["U_cols"])
                arrays_store[_subj] = _data
                break

        if not arrays_store:
            return adapter, {}, {}, {}

        _first = next(iter(arrays_store.values()))
        names = {}
        if "X_cols" in _first:
            names["X_cols"] = list(_first["X_cols"])
        if "U_cols" in _first:
            names["U_cols"] = list(_first["U_cols"])

        if scoring_key is not None and hasattr(adapter, "scoring_key"):
            adapter.scoring_key = scoring_key
        views = build_views(arrays_store, adapter, K, list(arrays_store.keys()))
        return adapter, arrays_store, names, views

    return (load_fit_bundle,)


@app.cell
def _(mo, model_aliases, ui_task):
    _pair_aliases = model_aliases(ui_task.value, "glmhmm")
    _default_a = _pair_aliases[0] if _pair_aliases else None
    _default_b = _pair_aliases[1] if len(_pair_aliases) > 1 else _default_a
    ui_pairwise_alias_a = mo.ui.dropdown(
        options=_pair_aliases,
        value=_default_a,
        label="Model A",
    )
    ui_pairwise_alias_b = mo.ui.dropdown(
        options=_pair_aliases,
        value=_default_b,
        label="Model B",
    )
    return ui_pairwise_alias_a, ui_pairwise_alias_b


@app.cell
def _(mo, model_k_options, ui_pairwise_alias_a, ui_pairwise_alias_b, ui_task):
    _k_opts_a = set(model_k_options(ui_task.value, "glmhmm", ui_pairwise_alias_a.value))
    _k_opts_b = set(model_k_options(ui_task.value, "glmhmm", ui_pairwise_alias_b.value))
    _pair_k_options = sorted(_k_opts_a & _k_opts_b)
    ui_pairwise_K = mo.ui.dropdown(
        options=_pair_k_options,
        value=_pair_k_options[0] if _pair_k_options else None,
        label="Shared K",
    )
    mo.vstack(
        [
            mo.md("### Pairwise GLMHMM comparison"),
            mo.md(
                "Compare two GLMHMM aliases at the same `K`. "
                "This section focuses on paired metrics, semantic state alignment, "
                "transition matrices, emission weights, and occupancy."
            ),
            mo.hstack([ui_pairwise_alias_a, ui_pairwise_alias_b, ui_pairwise_K]),
            mo.md(
                "Uses the subjects selected above. Pick a smaller subset there if you want a tighter visual comparison."
            ),
        ]
    )
    return (ui_pairwise_K,)


@app.cell
def _(adapter, mo):
    _opts = list(adapter._SCORING_OPTIONS.keys()) if hasattr(adapter, "_SCORING_OPTIONS") else ["default"]
    _default_key = getattr(adapter, "scoring_key", _opts[0]) if _opts else None
    if _opts and _default_key not in _opts:
        _default_key = _opts[0]
    ui_pairwise_scoring_key = mo.ui.dropdown(
        options=_opts,
        value=_default_key,
        label="State scoring regressor",
    )
    mo.hstack([ui_pairwise_scoring_key])
    return (ui_pairwise_scoring_key,)


@app.cell
def _(
    load_fit_bundle,
    load_metrics_dir,
    mo,
    pl,
    ui_pairwise_K,
    ui_pairwise_alias_a,
    ui_pairwise_alias_b,
    ui_pairwise_scoring_key,
    ui_subjects,
    ui_task,
):
    mo.stop(
        not ui_pairwise_alias_a.value or not ui_pairwise_alias_b.value,
        mo.md("Select two GLMHMM aliases above."),
    )
    mo.stop(
        ui_pairwise_alias_a.value == ui_pairwise_alias_b.value,
        mo.md("Choose two different GLMHMM aliases for a pairwise comparison."),
    )
    mo.stop(
        ui_pairwise_K.value is None,
        mo.md("No shared `K` values were found for the selected aliases."),
    )

    pairwise_alias_a = ui_pairwise_alias_a.value
    pairwise_alias_b = ui_pairwise_alias_b.value
    pairwise_K = int(ui_pairwise_K.value)
    requested_subjects = list(ui_subjects.value)

    pairwise_adapter_a, pairwise_arrays_a, pairwise_names_a, pairwise_views_a = load_fit_bundle(
        ui_task.value,
        "glmhmm",
        pairwise_alias_a,
        pairwise_K,
        requested_subjects,
        scoring_key=ui_pairwise_scoring_key.value,
    )
    pairwise_adapter_b, pairwise_arrays_b, pairwise_names_b, pairwise_views_b = load_fit_bundle(
        ui_task.value,
        "glmhmm",
        pairwise_alias_b,
        pairwise_K,
        requested_subjects,
        scoring_key=ui_pairwise_scoring_key.value,
    )

    pairwise_common_subjects = [
        _subject
        for _subject in requested_subjects
        if _subject in pairwise_views_a and _subject in pairwise_views_b
    ]
    mo.stop(
        not pairwise_common_subjects,
        mo.md(
            "No common cached subjects were found for the selected aliases and `K`. "
            "Check the subject subset or the cached fits."
        ),
    )

    _metric_schema = {
        "subject": pl.Utf8,
        "K": pl.Int64,
        "model_kind": pl.Utf8,
        "model_alias": pl.Utf8,
        "model_label": pl.Utf8,
        "ll_per_trial": pl.Float64,
        "bic": pl.Float64,
        "acc": pl.Float64,
    }

    def _pair_metrics(alias: str):
        _df = load_metrics_dir(ui_task.value, alias, "glmhmm")
        if _df is None:
            return pl.DataFrame(schema=_metric_schema)
        return _df.filter(
            pl.col("subject").is_in(pairwise_common_subjects)
            & (pl.col("K") == pairwise_K)
        )

    pairwise_metrics_a = _pair_metrics(pairwise_alias_a)
    pairwise_metrics_b = _pair_metrics(pairwise_alias_b)
    pairwise_missing_a = [s for s in requested_subjects if s not in pairwise_views_a]
    pairwise_missing_b = [s for s in requested_subjects if s not in pairwise_views_b]
    return (
        pairwise_K,
        pairwise_adapter_a,
        pairwise_adapter_b,
        pairwise_alias_a,
        pairwise_alias_b,
        pairwise_arrays_a,
        pairwise_arrays_b,
        pairwise_common_subjects,
        pairwise_metrics_a,
        pairwise_metrics_b,
        pairwise_missing_a,
        pairwise_missing_b,
        pairwise_names_a,
        pairwise_names_b,
        pairwise_views_a,
        pairwise_views_b,
        requested_subjects,
    )


@app.cell
def _(
    mo,
    pairwise_K,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_common_subjects,
    pairwise_missing_a,
    pairwise_missing_b,
    requested_subjects,
    ui_pairwise_scoring_key,
):
    _notes = [
        f"- Comparing `{pairwise_alias_a}` vs `{pairwise_alias_b}` at `K={pairwise_K}`.",
        f"- Semantic state alignment uses scoring key `{ui_pairwise_scoring_key.value}`.",
        f"- Common cached subjects: **{len(pairwise_common_subjects)} / {len(requested_subjects)}**.",
    ]
    if pairwise_missing_a:
        _notes.append(
            f"- Missing in `{pairwise_alias_a}`: {', '.join(map(str, pairwise_missing_a[:8]))}"
            + (" ..." if len(pairwise_missing_a) > 8 else "")
        )
    if pairwise_missing_b:
        _notes.append(
            f"- Missing in `{pairwise_alias_b}`: {', '.join(map(str, pairwise_missing_b[:8]))}"
            + (" ..." if len(pairwise_missing_b) > 8 else "")
        )
    mo.md("\n".join(_notes))
    return


@app.cell
def _(
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_metrics_a,
    pairwise_metrics_b,
    pl,
):
    _frames = []
    if not pairwise_metrics_a.is_empty():
        _frames.append(
            pairwise_metrics_a.with_columns(pl.lit("A").alias("model_slot"))
        )
    if not pairwise_metrics_b.is_empty():
        _frames.append(
            pairwise_metrics_b.with_columns(pl.lit("B").alias("model_slot"))
        )

    if _frames:
        pairwise_metrics = pl.concat(_frames, how="diagonal")
    else:
        pairwise_metrics = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "K": pl.Int64,
                "model_kind": pl.Utf8,
                "model_alias": pl.Utf8,
                "model_label": pl.Utf8,
                "ll_per_trial": pl.Float64,
                "bic": pl.Float64,
                "acc": pl.Float64,
                "model_slot": pl.Utf8,
            }
        )

    pairwise_metric_summary = (
        pairwise_metrics
        .group_by(["model_slot", "model_alias"])
        .agg(
            [
                pl.len().alias("n_subjects"),
                pl.mean("ll_per_trial").alias("ll_mean"),
                pl.mean("bic").alias("bic_mean"),
                pl.mean("acc").alias("acc_mean"),
            ]
        )
        .sort("model_slot")
    )

    pairwise_metric_deltas = (
        pairwise_metrics_a.select(
            [
                "subject",
                pl.col("ll_per_trial").alias("ll_a"),
                pl.col("bic").alias("bic_a"),
                pl.col("acc").alias("acc_a"),
            ]
        )
        .join(
            pairwise_metrics_b.select(
                [
                    "subject",
                    pl.col("ll_per_trial").alias("ll_b"),
                    pl.col("bic").alias("bic_b"),
                    pl.col("acc").alias("acc_b"),
                ]
            ),
            on="subject",
            how="inner",
        )
        .with_columns(
            [
                pl.lit(pairwise_alias_a).alias("model_a"),
                pl.lit(pairwise_alias_b).alias("model_b"),
                (pl.col("ll_b") - pl.col("ll_a")).alias("delta_ll_per_trial"),
                (pl.col("bic_b") - pl.col("bic_a")).alias("delta_bic"),
                (pl.col("acc_b") - pl.col("acc_a")).alias("delta_acc"),
            ]
        )
        .sort("subject")
    )

    pairwise_metric_delta_summary = (
        pairwise_metric_deltas
        .select(["delta_ll_per_trial", "delta_bic", "delta_acc"])
        .mean()
        .with_columns(
            [
                pl.lit(pairwise_alias_b).alias("model_b"),
                pl.lit(pairwise_alias_a).alias("model_a"),
            ]
        )
        .select(["model_b", "model_a", "delta_ll_per_trial", "delta_bic", "delta_acc"])
    )
    return (
        pairwise_metric_delta_summary,
        pairwise_metric_deltas,
        pairwise_metric_summary,
        pairwise_metrics,
    )


@app.cell
def _(trial_df):
    trial_df
    return


@app.cell
def _(
    mo,
    pairwise_metric_delta_summary,
    pairwise_metric_deltas,
    pairwise_metric_summary,
    pairwise_metrics,
):
    mo.stop(pairwise_metrics.is_empty(), mo.md("No paired metrics were found for the selected aliases and `K`."))
    mo.vstack(
        [
            mo.md("#### Paired metrics"),
            pairwise_metric_summary,
            mo.md("Mean deltas are computed as **B - A**."),
            pairwise_metric_delta_summary,
            pairwise_metric_deltas,
        ]
    )
    return


@app.cell
def _(np, plt, sns):
    def _resolve_transition_matrix(arrays: dict) -> np.ndarray | None:
        if "transition_matrix" in arrays:
            return np.asarray(arrays["transition_matrix"], dtype=float)
        if "transition_bias" in arrays:
            _bias = np.asarray(arrays["transition_bias"], dtype=float)
            _exp = np.exp(_bias - _bias.max(axis=-1, keepdims=True))
            return _exp / _exp.sum(axis=-1, keepdims=True)
        return None

    def plot_pairwise_transition_matrices(
        *,
        arrays_a: dict,
        arrays_b: dict,
        views_a: dict,
        views_b: dict,
        subjects: list,
        alias_a: str,
        alias_b: str,
    ) -> plt.Figure:
        def _mean_transition(arrays_store: dict, views: dict):
            _mats = []
            _labels = None
            for _subject in subjects:
                if _subject not in views:
                    continue
                _mat = _resolve_transition_matrix(arrays_store.get(_subject, {}))
                if _mat is None:
                    continue
                _order = [int(k) for k in views[_subject].state_idx_order]
                _mats.append(_mat[np.ix_(_order, _order)])
                if _labels is None:
                    _labels = [
                        views[_subject].state_name_by_idx.get(int(k), f"State {k}")
                        for k in _order
                    ]
            if not _mats:
                return None, []
            return np.mean(_mats, axis=0), _labels or []

        _mat_a, _labels_a = _mean_transition(arrays_a, views_a)
        _mat_b, _labels_b = _mean_transition(arrays_b, views_b)
        if _mat_a is None or _mat_b is None:
            raise ValueError("No transition matrices were available for the common subject set.")

        _labels = _labels_a if _labels_a else _labels_b
        _vmax = max(float(np.nanmax(_mat_a)), float(np.nanmax(_mat_b)), 1e-12)
        _delta = _mat_b - _mat_a
        _dmax = max(float(np.nanmax(np.abs(_delta))), 1e-12)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=False)
        for _ax, _mat, _title in [
            (axes[0], _mat_a, alias_a),
            (axes[1], _mat_b, alias_b),
        ]:
            sns.heatmap(
                _mat,
                ax=_ax,
                cmap="Blues",
                vmin=0,
                vmax=_vmax,
                annot=True,
                fmt=".2f",
                square=True,
                cbar=False,
            )
            _ax.set_title(_title)
            _ax.set_xticklabels(_labels, rotation=25, ha="right")
            _ax.set_yticklabels(_labels, rotation=0)
            _ax.set_xlabel("To state")
            _ax.set_ylabel("From state")

        sns.heatmap(
            _delta,
            ax=axes[2],
            cmap="RdBu_r",
            center=0,
            vmin=-_dmax,
            vmax=_dmax,
            annot=True,
            fmt=".2f",
            square=True,
            cbar=False,
        )
        axes[2].set_title(f"{alias_b} - {alias_a}")
        axes[2].set_xticklabels(_labels, rotation=25, ha="right")
        axes[2].set_yticklabels(_labels, rotation=0)
        axes[2].set_xlabel("To state")
        axes[2].set_ylabel("From state")
        fig.suptitle(f"Mean transition matrices in semantic state order  (n={len(subjects)} subjects)")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    return (plot_pairwise_transition_matrices,)


@app.cell
def _(
    mo,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_arrays_a,
    pairwise_arrays_b,
    pairwise_common_subjects,
    pairwise_views_a,
    pairwise_views_b,
    plot_pairwise_transition_matrices,
):
    try:
        _fig_transition_pair = plot_pairwise_transition_matrices(
            arrays_a=pairwise_arrays_a,
            arrays_b=pairwise_arrays_b,
            views_a=pairwise_views_a,
            views_b=pairwise_views_b,
            subjects=pairwise_common_subjects,
            alias_a=pairwise_alias_a,
            alias_b=pairwise_alias_b,
        )
        mo.vstack([mo.md("#### Transition matrices"), _fig_transition_pair])
    except Exception as _e:
        mo.md(f"#### Transition matrices\n\nCould not render the pairwise transition comparison: `{_e}`")
    return


@app.cell
def _(
    mo,
    pairwise_K,
    pairwise_adapter_a,
    pairwise_adapter_b,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_arrays_a,
    pairwise_arrays_b,
    pairwise_common_subjects,
    pairwise_names_a,
    pairwise_names_b,
    pairwise_views_a,
    pairwise_views_b,
):
    def _emission_summary(_adapter, _views, _arrays_store, _names):
        _plots = _adapter.get_plots()
        if _adapter.num_classes == 2:
            return _plots.plot_emission_weights_summary(views=_views, K=pairwise_K)
        return _plots.plot_emission_weights_summary(
            arrays_store=_arrays_store,
            state_labels={s: v.state_name_by_idx for s, v in _views.items()},
            names=_names,
            K=pairwise_K,
            subjects=pairwise_common_subjects,
        )

    try:
        _fig_emission_a = _emission_summary(
            pairwise_adapter_a,
            pairwise_views_a,
            pairwise_arrays_a,
            pairwise_names_a,
        )
        _fig_emission_b = _emission_summary(
            pairwise_adapter_b,
            pairwise_views_b,
            pairwise_arrays_b,
            pairwise_names_b,
        )
        mo.vstack(
            [
                mo.md("#### Emission weights"),
                mo.hstack(
                    [
                        mo.vstack([mo.md(f"**A** — `{pairwise_alias_a}`"), _fig_emission_a]),
                        mo.vstack([mo.md(f"**B** — `{pairwise_alias_b}`"), _fig_emission_b]),
                    ],
                    widths="equal",
                ),
            ]
        )
    except Exception as _e:
        mo.md(f"#### Emission weights\n\nCould not render the pairwise emission summaries: `{_e}`")
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
def _():
    return


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
    def _pairwise_trial_df(adapter, views):
        _frames = []
        for _subject in pairwise_common_subjects:
            if _subject not in views:
                continue
            _df_sub = subject_behavior_df(
                df_all,
                subject=_subject,
                sort_col=adapter.sort_col,
                session_col=adapter.session_col,
            )
            if _df_sub.height != views[_subject].T:
                continue
            try:
                _frames.append(
                    build_trial_df(
                        views[_subject],
                        adapter,
                        _df_sub,
                        adapter.behavioral_cols,
                    )
                )
            except Exception:
                pass
        if not _frames:
            return pl.DataFrame()
        return pl.concat(_frames, how="diagonal")

    pairwise_trial_df_a = _pairwise_trial_df(pairwise_adapter_a, pairwise_views_a)
    pairwise_trial_df_b = _pairwise_trial_df(pairwise_adapter_b, pairwise_views_b)
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
    def _session_occupancy_records(*, alias: str, views: dict):
        _records = []
        for _subject in pairwise_common_subjects:
            if _subject not in views:
                continue
            _view = views[_subject]
            _df_sub = subject_behavior_df(
                df_all,
                subject=_subject,
                sort_col=pairwise_adapter_a.sort_col,
                session_col=pairwise_adapter_a.session_col,
            )
            if _df_sub.height != _view.T:
                continue
            if pairwise_adapter_a.session_col not in _df_sub.columns:
                continue
            _session_col = pairwise_adapter_a.session_col
            _sessions = np.asarray(_df_sub[_session_col])
            _probs = np.asarray(_view.smoothed_probs, dtype=float)
            for _session in np.unique(_sessions):
                _mask = _sessions == _session
                if not np.any(_mask):
                    continue
                for _state_idx in _view.state_idx_order:
                    _records.append(
                        {
                            "subject": str(_subject),
                            "session": str(_session),
                            "model_alias": alias,
                            "state_label": _view.state_name_by_idx.get(
                                int(_state_idx), f"State {_state_idx}"
                            ),
                            "occupancy": float(np.mean(_probs[_mask, int(_state_idx)])),
                        }
                    )
        return _records

    _records = _session_occupancy_records(alias=pairwise_alias_a, views=pairwise_views_a)
    _records += _session_occupancy_records(alias=pairwise_alias_b, views=pairwise_views_b)

    if _records:
        pairwise_session_occupancy = pl.DataFrame(_records)
    else:
        pairwise_session_occupancy = pl.DataFrame(
            schema={
                "subject": pl.Utf8,
                "session": pl.Utf8,
                "model_alias": pl.Utf8,
                "state_label": pl.Utf8,
                "occupancy": pl.Float64,
            }
        )

    pairwise_subject_occupancy = (
        pairwise_session_occupancy
        .group_by(["subject", "model_alias", "state_label"])
        .agg(pl.mean("occupancy").alias("occupancy"))
        .sort(["state_label", "model_alias", "subject"])
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
    return pairwise_session_occupancy, pairwise_subject_occupancy


@app.cell
def _(
    mo,
    np,
    pairwise_K,
    pairwise_adapter_a,
    pairwise_alias_a,
    pairwise_alias_b,
    pairwise_session_occupancy,
    pairwise_subject_occupancy,
    pairwise_trial_df_a,
    pairwise_trial_df_b,
    pl,
    plt,
    sns,
    ttest_rel,
):
    from matplotlib.lines import Line2D

    mo.stop(
        pairwise_session_occupancy.is_empty(),
        mo.md(
            "#### Session occupancy and mean accuracy by state\n\nSession-level occupancy could not be built for the current subject subset."
        ),
    )

    _acc_schema = {
        "subject": pl.Utf8,
        "model_alias": pl.Utf8,
        "state_label": pl.Utf8,
        "accuracy": pl.Float64,
        "n_trials": pl.Int64,
    }

    def _subject_accuracy(alias: str, df):
        if df.is_empty() or "state_label" not in df.columns:
            return pl.DataFrame(schema=_acc_schema)
        _df = df
        if "correct_bool" not in _df.columns:
            if "performance" not in _df.columns:
                return pl.DataFrame(schema=_acc_schema)
            _df = _df.with_columns(
                pl.col("performance").cast(pl.Boolean).alias("correct_bool")
            )
        return (
            _df
            .filter(
                pl.col("state_label").is_not_null()
                & pl.col("correct_bool").is_not_null()
            )
            .group_by(["subject", "state_label"])
            .agg(
                [
                    (pl.col("correct_bool").cast(pl.Float64).mean() * 100.0).alias("accuracy"),
                    pl.len().alias("n_trials"),
                ]
            )
            .with_columns(pl.lit(alias).alias("model_alias"))
            .select(["subject", "model_alias", "state_label", "accuracy", "n_trials"])
        )

    _acc_frames = []
    _acc_a = _subject_accuracy(pairwise_alias_a, pairwise_trial_df_a)
    _acc_b = _subject_accuracy(pairwise_alias_b, pairwise_trial_df_b)
    if not _acc_a.is_empty():
        _acc_frames.append(_acc_a)
    if not _acc_b.is_empty():
        _acc_frames.append(_acc_b)
    pairwise_subject_accuracy = (
        pl.concat(_acc_frames, how="diagonal")
        if _acc_frames
        else pl.DataFrame(schema=_acc_schema)
    )
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

    _palette = {
        pairwise_alias_a: "#1B6CA8",
        pairwise_alias_b: "#C76D3A",
    }
    _occ_pd = pairwise_subject_occupancy.to_pandas()
    _acc_pd = pairwise_subject_accuracy.to_pandas()
    _state_order = []
    for _label in list(_occ_pd.get("state_label", [])) + list(_acc_pd.get("state_label", [])):
        if _label not in _state_order:
            _state_order.append(_label)
    _models = [pairwise_alias_a, pairwise_alias_b]
    _offsets = np.linspace(-0.18, 0.18, len(_models))
    _width = 0.26
    _rng = np.random.default_rng(42)

    def _p_label(pval: float) -> str:
        if not np.isfinite(pval):
            return ""
        if pval < 0.001:
            return "***"
        if pval < 0.01:
            return "**"
        if pval < 0.05:
            return "*"
        return ""

    def _draw(ax, df_pd, value_col, title, ylabel, ylim=None, chance=None):
        if df_pd.empty or not _state_order:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            return

        if chance is not None:
            ax.axhline(
                chance,
                color="#7A7A7A",
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
            )

        _y_min = float(df_pd[value_col].min()) if not df_pd.empty else 0.0
        _y_max = float(df_pd[value_col].max()) if not df_pd.empty else 1.0
        _y_span = max(_y_max - _y_min, 1.0)
        _line_pad = 0.08 * _y_span

        for _state_idx, _state in enumerate(_state_order):
            _rows_a = df_pd[
                (df_pd["state_label"] == _state)
                & (df_pd["model_alias"] == pairwise_alias_a)
            ][["subject", value_col]]
            _rows_b = df_pd[
                (df_pd["state_label"] == _state)
                & (df_pd["model_alias"] == pairwise_alias_b)
            ][["subject", value_col]]
            _paired = _rows_a.merge(_rows_b, on="subject", how="inner", suffixes=("_a", "_b"))
            _paired_jitter = (
                _rng.uniform(-0.035, 0.035, size=len(_paired))
                if len(_paired) > 0
                else np.array([])
            )

            for _model_idx, _model in enumerate(_models):
                _rows = df_pd[
                    (df_pd["state_label"] == _state)
                    & (df_pd["model_alias"] == _model)
                ]
                if _rows.empty:
                    continue
                _values = _rows[value_col].to_numpy(dtype=float)
                _pos = _state_idx + _offsets[_model_idx]
                _box = ax.boxplot(
                    _values,
                    positions=[_pos],
                    widths=_width,
                    patch_artist=True,
                    showfliers=False,
                    zorder=1,
                )
                for _patch in _box["boxes"]:
                    _patch.set(facecolor="white", edgecolor="#666666", linewidth=1.1)
                for _elem in ("whiskers", "caps"):
                    for _artist in _box[_elem]:
                        _artist.set(color="#666666", linewidth=1.0)
                for _median in _box["medians"]:
                    _median.set(color=_palette[_model], linewidth=2.2)

                if _model == pairwise_alias_a:
                    _paired_value_col = f"{value_col}_a"
                else:
                    _paired_value_col = f"{value_col}_b"

                if not _paired.empty:
                    _paired_x = _pos + _paired_jitter
                    ax.scatter(
                        _paired_x,
                        _paired[_paired_value_col].to_numpy(dtype=float),
                        color=_palette[_model],
                        alpha=0.55,
                        s=26,
                        zorder=4,
                    )

                _paired_subjects = set(_paired["subject"].tolist()) if not _paired.empty else set()
                _unpaired = _rows[~_rows["subject"].isin(_paired_subjects)]
                _jitter = _rng.uniform(-0.035, 0.035, size=len(_unpaired))
                ax.scatter(
                    np.full(len(_unpaired), _pos) + _jitter,
                    _unpaired[value_col].to_numpy(dtype=float),
                    color=_palette[_model],
                    alpha=0.45,
                    s=26,
                    zorder=3,
                )

            if not _paired.empty:
                _x_a = _state_idx + _offsets[0] + _paired_jitter
                _x_b = _state_idx + _offsets[1] + _paired_jitter
                _y_a = _paired[f"{value_col}_a"].to_numpy(dtype=float)
                _y_b = _paired[f"{value_col}_b"].to_numpy(dtype=float)
                for _xa, _xb, _ya, _yb in zip(_x_a, _x_b, _y_a, _y_b, strict=False):
                    ax.plot(
                        [_xa, _xb],
                        [_ya, _yb],
                        color="#B0B0B0",
                        linewidth=0.9,
                        alpha=0.7,
                        zorder=2,
                    )

                if len(_paired) >= 2:
                    if np.allclose(_y_a, _y_b):
                        _pval = 1.0
                    else:
                        _, _pval = ttest_rel(_y_a, _y_b, nan_policy="omit")
                    _stars = _p_label(float(_pval))
                    if _stars:
                        _line_y = max(np.nanmax(_y_a), np.nanmax(_y_b)) + _line_pad
                        ax.plot(
                            [_state_idx + _offsets[0], _state_idx + _offsets[1]],
                            [_line_y, _line_y],
                            color="black",
                            linewidth=1.0,
                            zorder=5,
                        )
                        ax.text(
                            _state_idx,
                            _line_y + 0.02 * _y_span,
                            _stars,
                            ha="center",
                            va="bottom",
                            fontsize=10,
                        )

        ax.set_xticks(range(len(_state_order)))
        ax.set_xticklabels(_state_order, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            _upper = max(
                ylim[1],
                _y_max + 2.5 * _line_pad,
            )
            ax.set_ylim(ylim[0], _upper)
        sns.despine(ax=ax)

    fig_occ, ax_occ = plt.subplots(figsize=(4, 4), constrained_layout=False)
    _draw(
        ax_occ,
        _occ_pd,
        "occupancy",
        "Mean session occupancy by state",
        "Fractional occupancy",
        ylim=(0, 1),
        chance=1.0 / max(1, pairwise_K),
    )
    fig_acc2, ax_acc = plt.subplots(figsize=(4, 4), constrained_layout=False)
    _draw(
        ax_acc,
        _acc_pd,
        "accuracy",
        "Mean accuracy by state",
        "Accuracy (%)",
        ylim=(0, 100),
        chance=100.0 / max(1, pairwise_adapter_a.num_classes),
    )

    _handles = [
        Line2D([0], [0], marker="o", linestyle="", color=_palette[_model], label=_model, markersize=6)
        for _model in _models
    ]
    for _fig in (fig_occ, fig_acc2):
        _fig.legend(
            _handles,
            _models,
            title="Model",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
        )
        _fig.tight_layout(rect=(0, 0.08, 1, 1))
    mo.vstack([fig_occ, fig_acc2,],)
    return


@app.cell
def _(mo, model_aliases, ui_task):
    ui_viz_model = mo.ui.dropdown(
        options=["glm", "glmhmm", "glmhmmt"],
        value="glmhmm",
        label="Model kind",
    )
    ui_viz_alias = mo.ui.dropdown(
        options=model_aliases(ui_task.value, ui_viz_model.value),
        value=None,
        label="Model alias",
    )
    ui_viz_K = mo.ui.slider(start=1, stop=8, value=2, label="K (for GLMHMM/T)")

    mo.vstack([
        mo.md("### Emission weights from cached fits"),
        mo.hstack([ui_viz_model, ui_viz_alias, ui_viz_K]),
    ])
    return ui_viz_K, ui_viz_alias, ui_viz_model


@app.cell
def _(
    load_fit_bundle,
    mo,
    ui_subjects,
    ui_task,
    ui_viz_K,
    ui_viz_alias,
    ui_viz_model,
):
    mo.stop(
        not ui_viz_alias.value,
        mo.md("Select a model alias above to visualise weights."),
    )

    _kind = ui_viz_model.value
    _K = ui_viz_K.value
    _adapter_viz, _arrays_store, _names, _views = load_fit_bundle(
        ui_task.value,
        _kind,
        ui_viz_alias.value,
        _K,
        ui_subjects.value,
    )

    mo.stop(
        not _arrays_store,
        mo.md(
            f"No cached arrays were found for `{ui_viz_alias.value}` at K={_K}."
        ),
    )

    _plots = _adapter_viz.get_plots()

    try:
        if _adapter_viz.num_classes == 2:
            _fig_ag, _fig_cls = _plots.plot_emission_weights(
                views=_views,
                K=_K,
            )
        else:
            _fig_ag, _fig_cls = _plots.plot_emission_weights(
                arrays_store=_arrays_store,
                state_labels={s: v.state_name_by_idx for s, v in _views.items()},
                names=_names,
                K=_K,
                subjects=list(_views.keys()),
            )
        _viz_output = mo.vstack([
            mo.md(f"**{_kind}  K={_K}**  —  {ui_viz_alias.value}"),
            _fig_ag,
            _fig_cls,
        ])
    except Exception as _e:
        _viz_output = mo.md(f"⚠️  Could not render weight plot: `{_e}`")
    _viz_output
    return


@app.cell
def _(mo):
    refit_button = mo.ui.run_button(
        label="⚠️  Re-fit selected (overwrites cached metrics)"
    )
    mo.vstack([
        mo.md("---\n### Re-fit (optional)"),
        mo.md(
            "> Runs the fit scripts for the selected task / subjects / K range "
            "and overwrites `_metrics.parquet` files in the chosen folders.  \n"
            "> Reload the page afterward to see updated metrics."
        ),
        refit_button,
    ])
    return (refit_button,)


@app.cell
def _(
    mo,
    paths,
    refit_button,
    ui_K_range,
    ui_glmhmm_dir,
    ui_glmhmmt_dir,
    ui_subjects,
    ui_task,
):
    mo.stop(
        not refit_button.value,
        mo.md("Press the button above to trigger re-fitting."),
    )

    import sys as _sys, os as _os
    _sys.path.append(_os.path.join(_os.path.dirname(__file__), ".."))
    try:
        from scripts.fit_glmhmm  import main as _fit_glmhmm_main
        from scripts.fit_glmhmmt import main as _fit_glmhmmt_main
        _FITTING_AVAILABLE = True
    except ImportError:
        _FITTING_AVAILABLE = False

    _K_min, _K_max = ui_K_range.value
    _K_list = list(range(max(2, _K_min), _K_max + 1))

    if not _FITTING_AVAILABLE:
        mo.md("❌  Fitting scripts not available in this environment (likely WASM).")
        mo.stop(True)

    with mo.status.spinner(title="Re-fitting GLMHMM…"):
        if ui_glmhmm_dir.value:
            for _alias in ui_glmhmm_dir.value:
                _fit_glmhmm_main(
                    subjects=ui_subjects.value,
                    K_list=_K_list,
                    out_dir=paths.RESULTS / "fits" / ui_task.value / "glmhmm" / _alias,
                    task=ui_task.value,
                )

    with mo.status.spinner(title="Re-fitting GLMHMM-T…"):
        if ui_glmhmmt_dir.value:
            for _alias in ui_glmhmmt_dir.value:
                _fit_glmhmmt_main(
                    subjects=ui_subjects.value,
                    K_list=_K_list,
                    out_dir=paths.RESULTS / "fits" / ui_task.value / "glmhmmt" / _alias,
                    task=ui_task.value,
                )

    mo.md("✅  Re-fit complete. Reload the notebook to refresh cached metrics.")
    return


if __name__ == "__main__":
    app.run()
