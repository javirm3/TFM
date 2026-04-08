import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import polars as pl
    import pandas as pd
    from glmhmmt.runtime import get_runtime_paths

    paths = get_runtime_paths()
    return paths, pd, pl


@app.cell
def _(paths, pd, pl):
    df = pl.from_pandas(pd.read_csv(paths.DATA_PATH / "tiffany.csv",index_col=0))
    df.write_parquet(paths.DATA_PATH / "tiffany.parquet")
    df
    return


if __name__ == "__main__":
    app.run()
