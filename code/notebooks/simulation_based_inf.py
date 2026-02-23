import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(pathlib):
    import numpy as np
    import os, sys
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from torch.distributions import Gamma, Normal

    from sbi.inference import MNLE
    from sbi.utils import MultipleIndependent
    import numba
    #numba.set_num_threads(64)

    import helpers 
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import paths
    CSV_PATH = f"{paths.DATA_PATH}/df_filtered.csv"
    df = pd.read_csv(CSV_PATH, sep =";")
    df = df[df['timepoint_4'] <= np.percentile(df['timepoint_4'], 95)]
    # theta_full = [sL, sC, sR, noise_amp, S_amp, dS, U_amp, U_base, U_on, U_ext_amp]
    TEMPLATE_THETA_FULL = np.array([ 0.0, 0.0, 0.0, 1.0,       0.0,   0.0, 0.0,   -1.0,   0.0,  0.0], dtype=float)
    FREE_TO_FULL = np.array([0, 2, 4, 5, 6], dtype=int)

    DT = np.float32(0.1 / 40.0)
    TH1 = np.float32(0.0)
    TH2 = np.float32(0.0)
    TH3 = np.float32(0.0)
    return np, torch


@app.cell
def _():
    return


@app.cell
def _(TOML):
    TOML
    return


@app.cell
def _(delayd, estimator, np, side, stimd, t1, t2, t3, t4, torch):
    import torch.nn.functional as F

    estimator.eval()

    # Construir tensor de condiciones: (N,7)
    cond_all = torch.tensor(
        np.stack([stimd, delayd, side, t1, t2, t3, t4], axis=1),
        dtype=torch.float32
    )

    # Subsample para tests
    N_test = 1500
    idx = torch.randperm(cond_all.shape[0])[:N_test]
    cond_test = cond_all[idx]

    print("Cond test shape:", cond_test.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    --- RECOVERY ---

    theta_true_free: [ 1.4686276 -0.4862646  0.5792301 -2.153729   1.7807858]

    theta_hat_free : [ 1.30210145 -0.27931015  0.47520909  0.24632346  1.65927665]

    abs error      : [0.16652612 0.20695444 0.10402104 2.40005243 0.12150915]

    NLL_hat        : -699405.375
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
