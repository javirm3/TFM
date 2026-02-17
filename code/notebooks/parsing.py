import marimo

__generated_with = "0.19.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parsing filtered_df.csv
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import paths

    return paths, pd


@app.cell
def _(paths, pd):
    paths.show_paths()  # Verificar que las rutas se han configurado correctamente
    df = pd.read_csv(paths.DATA_PATH/"df_filtered.csv")
    return (df,)


@app.cell
def _(df):
    df_1 = df[['subject', 'trial', 'session', 'date', 'x_c', 'r_c', 'ttype_n', 'stimd_n', 'performance', 'timepoint_1', 'timepoint_2', 'timepoint_3', 'timepoint_4', 'onset', 'offset']]
    df_1
    return (df_1,)


@app.cell
def _(df_1):
    df_1['stim_d'] = df_1['offset'] - df_1['onset']
    df_1['delay_d'] = df_1['timepoint_4'] - df_1['offset']
    response_code = {'L': 0, 'C': 1, 'R': 2}
    df_1['response'] = df_1['r_c'].map(response_code)
    df_1['stimulus'] = df_1['x_c'].map(response_code)
    df_1.sort_values(['subject', 'session', 'trial', 'date'], inplace=True)
    df_1['trial_idx'] = df_1.groupby(['subject']).cumcount()
    return


@app.cell
def _(df_1, paths):
    df_1.to_parquet(paths.DATA_PATH / 'df_filtered.parquet', index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
