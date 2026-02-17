import marimo

__generated_with = "0.19.10"
app = marimo.App()


@app.cell
def _():

    import marimo as mo
    import pandas as pd
    import numpy as np
    import pathlib
    import polars as pl
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # añade /code
    import paths
    import glmhmmt.plots as plots
    import matplotlib.pyplot as plt

    return mo, paths, pd, pl, plots, plt


@app.cell
def _(paths, pd, pl):
    df = pl.read_parquet(paths.DATA_PATH/"predictions.parquet")
    df2 = pd.read_parquet(paths.DATA_PATH/"df_filtered.parquet")
    return (df,)


@app.cell
def _(df, plots):
    plots_df = plots.prepare_predictions_df(df)
    plots_df
    return (plots_df,)


@app.cell
def _(plots, plots_df):
    plots.plot_categorical_performance_all(plots_df, "prueba")
    return


@app.cell
def _(df, plots, plots_df, plt):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    plots.plot_delay_or_stim_1d_on_ax(ax1, plots_df, subject=df["subject"].unique()[0], n_bins=7, which="delay")
    plots.plot_delay_or_stim_1d_on_ax(ax2, plots_df, subject=df["subject"].unique()[0], n_bins=7, which="stim")
    plt.show()
    return


@app.cell
def _(plots, plots_df):
    plots.plot_categorical_strat_by_side(plots_df, subject = "A89",model_name = "GLMHMM")
    return


@app.cell
def _(plots, plots_df):
    plots.plot_delay_binned_1d(plots_df, "GLMHMM", "A89")
    return


@app.cell
def _(mo, plots_df):
    mo.ui.table(plots_df)
    return


@app.cell
def _(paths):
    import tomllib
    with paths.CONFIG.open("rb") as f:
            cfg = tomllib.load(f)
    return


@app.cell
def _(pl, plots, plots_df, plt):
    fig2, ax = plt.subplots(1,1, figsize=(5,5))
    df_a = plots_df.filter(pl.col("ttype_c") == "VG")
    plots.plot_delay_or_stim_1d_on_ax(ax, df_a, subject=df_a["subject"].unique()[0], n_bins=7, which="stim")
    fig2.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
