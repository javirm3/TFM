# /// script
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
def _(get_adapter, mo, paths, ui_task):

    adapter = get_adapter(ui_task.value)

    def _model_aliases(task: str, kind: str) -> list:
        p = paths.RESULTS / "fits" / task / kind
        if not p.exists():
            return []
        return sorted([d.name for d in p.iterdir() if d.is_dir()])

    ui_glm_dir = mo.ui.multiselect(
        options=_model_aliases(ui_task.value, "glm"),
        value=[],
        label="GLM aliases",
    )
    ui_glmhmm_dir = mo.ui.multiselect(
        options=_model_aliases(ui_task.value, "glmhmm"),
        value=[],
        label="GLMHMM aliases",
    )
    ui_glmhmmt_dir = mo.ui.multiselect(
        options=_model_aliases(ui_task.value, "glmhmmt"),
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
def _(mo, paths, pl, ui_glm_dir, ui_glmhmm_dir, ui_glmhmmt_dir, ui_task):
    _MODEL_LABELS = {
        "glm": "GLM",
        "glmhmm": "GLMHMM",
        "glmhmmt": "GLMHMM-T",
    }

    def _load_dir(folder_name, expected_model_kind):
        """Scan a fit dir for *_metrics.parquet files and concat them."""
        if not folder_name:
            return None
        d = paths.RESULTS / "fits" / ui_task.value / expected_model_kind / folder_name
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
        # Normalise: glm writes nll+n_trials; glmhmm/t writes ll_per_trial
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
        df = df.with_columns([
            pl.lit(folder_name).alias("model_alias"),
            pl.lit(f"{_MODEL_LABELS.get(expected_model_kind, expected_model_kind)} ({folder_name})").alias("model_label"),
        ])
        keep = ["subject", "K", "model_kind", "model_alias", "model_label", "ll_per_trial", "bic", "acc"]
        return df.select([c for c in keep if c in df.columns])

    _parts = []
    for _names, _kind in [
        (ui_glm_dir.value, "glm"),
        (ui_glmhmm_dir.value, "glmhmm"),
        (ui_glmhmmt_dir.value, "glmhmmt"),
    ]:
        for _name in _names:
            _p = _load_dir(_name, _kind)
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


    return (add_sig_bars,)


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

    fig_acc, ax_acc = plt.subplots(figsize=(6, 4))
    _labels = agg["model_label"].unique().to_list()
    _colors = sns.color_palette("tab20", n_colors=max(1, len(_labels)))
    _palette = {_label: _colors[_i] for _i, _label in enumerate(_labels)}
    for _label_tup, _group in agg.group_by("model_label"):
        _label = _label_tup[0]
        _g = _group.sort("K").to_pandas()
        _kind = _g["model_kind"].iloc[0]
        _st = _MODEL_STYLES.get(_kind, {"marker": "o", "label": _label})
        ax_acc.plot(
            _g["K"], _g["acc_mean"],
            color=_palette[_label], marker=_st["marker"],
            label=_label, linewidth=1.5,
        )

    ax_acc.set_xlabel("Number of states K")
    ax_acc.set_ylabel("Accuracy (mean over subjects)")
    ax_acc.set_title("Model accuracy vs K")
    ax_acc.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(3, max(1, len(_labels))),
    )
    sns.despine(ax=ax_acc)
    fig_acc.tight_layout(rect=(0, 0.08, 1, 1))
    fig_acc
    return


@app.cell
def _(build_views, get_adapter, np, paths):
    def load_fit_bundle(task_name, model_kind, alias, K, subjects):
        adapter = get_adapter(task_name)
        fit_dir = paths.RESULTS / "fits" / task_name / model_kind / alias
        suffix = {"glm": "glm", "glmhmm": "glmhmm", "glmhmm-t": "glmhmmt"}[model_kind]

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

        views = build_views(arrays_store, adapter, K, list(arrays_store.keys()))
        return adapter, arrays_store, names, views

    return (load_fit_bundle,)


@app.cell
def _(mo, paths, ui_task):
    def _model_aliases_viz(task: str, kind: str) -> list:
        p = paths.RESULTS / "fits" / task / kind
        if not p.exists():
            return []
        return sorted([d.name for d in p.iterdir() if d.is_dir()])

    ui_viz_model = mo.ui.dropdown(
        options=["glm", "glmhmm", "glmhmmt"],
        value="glmhmm",
        label="Model kind",
    )
    ui_viz_alias = mo.ui.dropdown(
        options=_model_aliases_viz(ui_task.value, ui_viz_model.value),
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
