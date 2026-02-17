import marimo

__generated_with = "0.19.10"
app = marimo.App()


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
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Gamma, Normal
    from torch.utils.data import TensorDataset, DataLoader, random_split
    import seaborn as sns
    from sbi.utils import MultipleIndependent
    from glmhmmt.simulator_3WM_numba import get_choices_varying_numba

    import paths
    import tomllib

    with paths.CONFIG.open("rb") as f:
        cfg = tomllib.load(f)

    stim_map = cfg["encoding"]["stimd"]
    stim_map = {int(k): v for k, v in stim_map.items()}
    CSV_PATH = f"{paths.DATA_PATH}/df_filtered.parquet"
    df = pd.read_parquet(CSV_PATH)
    df = df[df["timepoint_4"] <= np.percentile(df["timepoint_4"], 95)]
    with paths.CONFIG.open("rb") as f:
        cfg = tomllib.load(f)

    stim_map = {int(k): v for k, v in cfg["encoding"]["stimd"].items()}
    ttype_map = {int(k): v for k, v in cfg["encoding"]["ttype"].items()}

    df["stimd_n"] = df["stimd_n"].astype(int)

    df["stimd_c"] = df["stimd_n"].map(stim_map)
    df["ttype_c"] = df["ttype_n"].map(ttype_map)
    TEMPLATE_THETA_FULL = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=float
    )
    FREE_TO_FULL = np.array([0, 2, 4, 5, 6], dtype=int)

    DT = np.float32(0.1 / 40.0)
    TH1 = np.float32(0.0)
    TH2 = np.float32(0.0)
    TH3 = np.float32(0.0)
    return (
        DT,
        DataLoader,
        F,
        FREE_TO_FULL,
        Gamma,
        MultipleIndependent,
        Normal,
        TEMPLATE_THETA_FULL,
        TH1,
        TH2,
        TH3,
        TensorDataset,
        cfg,
        df,
        get_choices_varying_numba,
        nn,
        np,
        os,
        pd,
        plt,
        random_split,
        sns,
        torch,
    )


@app.cell
def _(np):
    def validate_and_encode(
        df, stim_col="stimd_c", delay_col="ttype_c", side_col="x_c", resp_col="r_c"
    ):
        stim_map = {"VG": 0, "SS": 1, "SM": 2, "SL": 3, "SIL": 4}
        side_map = {"L": 0, "C": 1, "R": 2, "SIL": 3}
        resp_map = {"L": 0, "C": 1, "R": 2}
        delay_map = {"DS": 0, "DM": 1, "DL": 2}

        for col in (stim_col, delay_col, side_col, resp_col):
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip().str.upper()

        stim_series = df[stim_col]
        delay_series = df[delay_col]
        side_series = df[side_col]
        resp_series = df[resp_col]

        stimd = stim_series.map(stim_map).astype("Int64")
        side = side_series.map(side_map).astype("Int64")
        resp = resp_series.map(resp_map).astype("Int64")

        delayd = np.zeros(len(df), dtype=np.int64)
        mask_delay_needed = stim_series.isin(["SS", "SM"])
        delayd[mask_delay_needed.values] = (
            delay_series[mask_delay_needed]
            .map(delay_map)
            .astype("Int64")
            .to_numpy(dtype=np.int64)
        )

        tp_cols = ["timepoint_1", "timepoint_2", "timepoint_3", "timepoint_4"]
        if df[tp_cols].isna().any().any():
            raise ValueError("NaNs en timepoints.")
        if (df["timepoint_4"] <= 0).any():
            raise ValueError("Hay trials con timepoint_4 <= 0.")

        return (
            stimd.to_numpy(dtype=np.int8),
            delayd.astype(np.int8),
            side.to_numpy(dtype=np.int8),
            resp.to_numpy(dtype=np.int8),
            df["timepoint_1"].to_numpy(dtype=np.float32),
            df["timepoint_2"].to_numpy(dtype=np.float32),
            df["timepoint_3"].to_numpy(dtype=np.float32),
            df["timepoint_4"].to_numpy(dtype=np.float32),
            df,
        )

    return (validate_and_encode,)


@app.cell
def _(Gamma, MultipleIndependent, Normal, df, mo, torch, validate_and_encode):
    stimd, delayd, side, resp, t1, t2, t3, t4, df_clean = validate_and_encode(df)
    sPrior = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    stimulus_amplitude_prior = Gamma(torch.tensor([2.0]), torch.tensor([20.0 / 6]))
    stimulus_tail_duration_prior = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    urgency_amplitude_prior = Gamma(torch.tensor([4.0]), torch.tensor([2.0]))

    prior = MultipleIndependent(
        [
            sPrior,
            sPrior,
            stimulus_amplitude_prior,
            stimulus_tail_duration_prior,
            urgency_amplitude_prior,
        ],
        validate_args=False,
    )

    N_simul = df.shape[0]
    mo.md(f"Simularemos {N_simul} trials para el entrenamiento.")
    return N_simul, delayd, prior, side, stimd, t1, t2, t3, t4


@app.cell
def _(cfg, plt, prior, sns, torch):
    # sample priors
    with torch.no_grad():
        samp = prior.sample((50_000,)).reshape(50_000, -1).cpu().numpy()

    names = ["sL", "sR", "S_amp", "dS", "U_amp"]
    cols = [
        cfg["colors"]["parameters"][k]
        for k in ["sL", "sR", "S_amplitude", "S_d", "U_int_amplitude"]
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.ravel()

    for ax, i, name, col in zip(axes, range(len(names)), names, cols):
        sns.kdeplot(x=samp[:, i], ax=ax, fill=True, color=col, linewidth=1.5)
        ax.set_title(name)

    for ax in axes[len(names) :]:
        ax.axis("off")

    sns.despine(fig=fig)
    fig.tight_layout()
    plt.show()
    return cols, names, samp


@app.cell
def _(cols, names, np, pd, plt, samp, sns):
    _df = pd.DataFrame(samp, columns=names)
    _df_long = _df.melt(var_name="parameter", value_name="value")

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=_df_long,
        x="parameter",
        y="value",
        palette=dict(zip(names, cols)),
        hue="parameter",
        cut=0,
        gap=0.25,
        alpha = 0.75
    )
    pretty_names = ["$s_L$", "$s_R$", "$S_{amp}$", "$S_d$", "$U_{amp}$"]
    plt.xticks(ticks=np.arange(len(pretty_names)), labels=pretty_names)
    plt.axhline(0, xmin=-1, xmax=6, linestyle="--", color="black")
    sns.despine()
    plt.tight_layout()
    plt.ylim(-3, 5)
    plt.show()
    return


@app.cell
def _(
    DT,
    FREE_TO_FULL,
    N_simul,
    TEMPLATE_THETA_FULL,
    TH1,
    TH2,
    TH3,
    delayd,
    get_choices_varying_numba,
    np,
    prior,
    side,
    stimd,
    t1,
    t2,
    t3,
    t4,
    torch,
):
    param_values = prior.sample((N_simul,))  # (N, n_free)
    theta_free = param_values.numpy().astype(np.float32)

    # Expand free -> full (N,10)
    theta_full = np.tile(TEMPLATE_THETA_FULL[None, :], (N_simul, 1)).astype(
        np.float32
    )
    theta_full[:, FREE_TO_FULL] = theta_free

    # -------------------------
    # SIMULATE choices (Numba)
    # -------------------------
    choice = get_choices_varying_numba(
        stimd, delayd, side, t1, t2, t3, t4, theta_full, DT, TH1, TH2, TH3
    ).astype(np.int8)

    mask_ok = choice != -1
    n_miss = int((~mask_ok).sum())
    print(f"Miss trials: {n_miss} / {N_simul} ({n_miss / N_simul:.2%})")

    choice_ok = choice[mask_ok].astype(np.int64)  # labels 0/1/2
    theta_free_ok = theta_free[mask_ok].astype(np.float32)  # (N_ok, n_free)
    cond_ok = np.stack([stimd, delayd, side, t1, t2, t3, t4], axis=1).astype(
        np.float32
    )[mask_ok]  # (N_ok, 7)

    # -------------------------
    # Build MLP training data
    # x = [theta_free, cond]  -> logits -> CE(choice)
    # -------------------------
    X = np.concatenate([theta_free_ok, cond_ok], axis=1).astype(
        np.float32
    )  # (N_ok, n_free+7)
    y = choice_ok  # (N_ok,)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return X, X_t, y_t


@app.cell
def _(
    DataLoader,
    F,
    FREE_TO_FULL,
    TEMPLATE_THETA_FULL,
    TensorDataset,
    X,
    X_t,
    mo,
    nn,
    os,
    random_split,
    torch,
    y_t,
):
    class ChoiceMLP(nn.Module):
        def __init__(
            self,
            in_dim: int,
            hidden: int = 128,
            depth: int = 2,
            p_drop: float = 0.0,
        ):
            super().__init__()
            layers = []
            d = in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.ReLU()]
                if p_drop > 0:
                    layers += [nn.Dropout(p_drop)]
                d = hidden
            layers += [nn.Linear(d, 3)]  # logits for 3 classes
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    # Hyperparams
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HIDDEN = int(os.environ.get("MLP_HIDDEN", "128"))
    DEPTH = int(os.environ.get("MLP_DEPTH", "2"))
    DROPOUT = float(os.environ.get("MLP_DROPOUT", "0.0"))
    LR = float(os.environ.get("MLP_LR", "3e-4"))
    BATCH = int(os.environ.get("MLP_BATCH", "4096"))
    EPOCHS = int(os.environ.get("MLP_EPOCHS", "100"))
    VAL_FRAC = float(os.environ.get("MLP_VAL_FRAC", "0.1"))

    model = ChoiceMLP(
        in_dim=X.shape[1], hidden=HIDDEN, depth=DEPTH, p_drop=DROPOUT
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Dataset / split
    ds = TensorDataset(X_t, y_t)
    n_val = int(len(ds) * VAL_FRAC)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, drop_last=False, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False, drop_last=False, num_workers=0
    )


    def eval_loss_acc():
        model.eval()
        tot_loss = 0.0
        tot = 0
        corr = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, reduction="sum")
                tot_loss += float(loss.item())
                pred = logits.argmax(dim=1)
                corr += int((pred == yb).sum().item())
                tot += int(yb.numel())
        return tot_loss / max(1, tot), corr / max(1, tot)


    best_val = 1e9
    best_state = None

    for ep in mo.status.progress_bar(
        range(1, EPOCHS + 1),
        title="Loading",
        subtitle="Please wait",
        show_eta=True,
        show_rate=True,
    ):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * yb.numel()
            seen += int(yb.numel())

        train_loss = running / max(1, seen)
        val_loss, val_acc = eval_loss_acc()
        print(
            f"ep {ep:03d} | train CE {train_loss:.4f} | val CE {val_loss:.4f} | val acc {val_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save
    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": X.shape[1],
        "hidden": HIDDEN,
        "depth": DEPTH,
        "dropout": DROPOUT,
        "free_to_full": FREE_TO_FULL.tolist(),
        "template_theta_full": TEMPLATE_THETA_FULL.tolist(),
    }
    out_path = os.environ.get("MLP_OUT", "choice_mlp_surrogate.pt")
    torch.save(ckpt, out_path)
    print("Saved:", out_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
