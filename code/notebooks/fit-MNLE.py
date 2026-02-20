import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    import paths
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sbi.utils import MultipleIndependent
    import paths
    import tomllib
    import pandas as pd
    with paths.CONFIG.open("rb") as f:
        cfg = tomllib.load(f)

    return cfg, np, paths, pd, plt, sns


@app.cell
def _(mo, paths, pd):
    fits_list = []

    files = sorted(p for p in paths.DATA_PATH.glob("fits_*")if p.is_file())
    for _file in files:
        _fit_df = pd.read_csv(_file)
        fits_list.append(_fit_df)
    fits_df = pd.concat(fits_list)
    dropdown_subj = mo.ui.dropdown(options=fits_df["subject"].unique(), value = "A86")
    return dropdown_subj, fits_df


@app.cell(hide_code=True)
def _(cfg, dropdown_subj, fits_df, mo, np, pd, plt, sns):
    import ast
    names = cfg["parameters"]["spatial_reduced2"]
    pretty_names = cfg["pretty_names"]["spatial_reduced2"]
    fit_df = fits_df[fits_df["subject"] == dropdown_subj.value]
    parsed = fit_df["x0"].apply(ast.literal_eval)
    flat = parsed.apply(lambda x: x[0])
    params_df = pd.DataFrame(flat.tolist(), columns=names)
    params_df["fval"] = fit_df["fval"]
    params_df["quality"] = np.where(params_df["fval"] < fit_df["fval"].mean(), "good", "bad")
    params_df["fit_id"] = np.arange(len(params_df))
    params_df = params_df.melt(
        id_vars=["fit_id", "fval", "quality"],
        var_name="parameter",
        value_name="value",
    )
    brewer = sns.color_palette("Set1", 2)
    palette = {
        "good": brewer[0],
        "bad": brewer[1]
    }

    fig1 = plt.figure(figsize=(4,4))
    sns.histplot(data=params_df, x = "fval",  hue="quality",  palette=palette)
    plt.axvline(x = params_df["fval"].mean(), linestyle = "--", color = "red")
    sns.despine()
    plt.tight_layout()


    fig2 = plt.figure(figsize=(8,4))
    sns.violinplot(
        data=params_df,
        x="parameter",
        y="value",
        hue="quality",
        palette = palette,
        alpha=0.75,
    )
    plt.xticks(ticks=np.arange(len(pretty_names)), labels=pretty_names)
    plt.axhline(0, xmin=-1, xmax=6, linestyle="--", color="black")
    sns.despine()
    plt.tight_layout()
    plt.ylim(-3, 6)


    fig3 = plt.figure(figsize=(8,4))

    sns.lineplot(
        data=params_df,
        x="parameter",
        y="value",
        hue="quality",
        units="fit_id",
        estimator=None,
        alpha=0.4,
        palette=palette,
    )

    plt.xticks(ticks=np.arange(len(pretty_names)), labels=pretty_names)
    plt.axhline(0, linestyle="--", color="black")
    sns.despine()
    plt.tight_layout()
    plt.ylim(-3, 5)


    mo.vstack(
        [
            mo.hstack([dropdown_subj]),
            mo.hstack([fig1, fig2]),
            fig3,
        ]
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
