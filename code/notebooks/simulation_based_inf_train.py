import marimo

__generated_with = "0.19.10"
app = marimo.App(
    width="medium",
    layout_file="layouts/simulation_based_inf_train.slides.json",
)


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
        paths,
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
def _():
    return


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
    return delayd, prior, side, stimd, t1, t2, t3, t4


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

    for ax, _i, name, col in zip(axes, range(len(names)), names, cols):
        sns.kdeplot(x=samp[:, _i], ax=ax, fill=True, color=col, linewidth=1.5)
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

    plt.figure(figsize=(4, 4), dpi=100)
    sns.violinplot(
        data=_df_long,
        x="parameter",
        y="value",
        palette=dict(zip(names, cols)),
        hue="parameter",
        cut=0,
        gap=0.25,
        alpha=0.75,
    )
    pretty_names = ["$s_L$", "$s_R$", "$S_{amp}$", "$S_d$", "$U_{amp}$"]
    plt.xticks(ticks=np.arange(len(pretty_names)), labels=pretty_names)
    plt.axhline(0, xmin=-1, xmax=6, linestyle="--", color="black")
    sns.despine()
    plt.tight_layout()
    plt.ylim(-3, 5)
    plt.show()
    return (pretty_names,)


@app.cell
def _(mo, paths, pd):
    fits_list = []

    files = sorted(p for p in paths.DATA_PATH.glob("fits_*")if p.is_file())
    for _file in files:
        _fit_df = pd.read_csv(_file)
        fits_list.append(_fit_df)
    fits_df = pd.concat(fits_list)
    dropdown_subj = mo.ui.dropdown(options=fits_df["subject"].unique())
    dropdown_subj
    return dropdown_subj, fits_df


@app.cell
def _(dropdown_subj, fits_df, names, np, pd, plt, pretty_names, sns):
    import ast

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

    plt.figure(figsize=(4,4))
    sns.histplot(data=params_df, x = "fval",  hue="quality",  palette=palette)
    plt.axvline(x = params_df["fval"].mean(), linestyle = "--", color = "red")
    sns.despine()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
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
    plt.show()


    plt.figure(figsize=(8,6))

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
    plt.show()
    return


@app.cell
def _(
    DT,
    FREE_TO_FULL,
    TEMPLATE_THETA_FULL,
    TH1,
    TH2,
    TH3,
    delayd,
    get_choices_varying_numba,
    np,
    os,
    prior,
    side,
    stimd,
    t1,
    t2,
    t3,
    t4,
    torch,
):
    # --- hyperparams ---
    K = int(os.environ.get("SIM_K", "30"))
    N_BASE = int(os.environ.get("SIM_N_BASE", "20000"))
    MIN_OK_FRAC = float(os.environ.get("SIM_MIN_OK_FRAC", "0.7"))  # descarta pares con demasiados -1

    # pool de condiciones reales
    cond_pool = np.stack([stimd, delayd, side, t1, t2, t3, t4], axis=1).astype(np.float32)
    N_pool = cond_pool.shape[0]

    # samplea N_BASE condiciones del pool real
    idx = np.random.randint(0, N_pool, size=N_BASE)
    cond_base = cond_pool[idx]  # (N_BASE, 7)

    # samplea N_BASE thetas
    theta_free_base = prior.sample((N_BASE,)).numpy().astype(np.float32)  # (N_BASE, n_free)

    # free -> full
    theta_full_base = np.tile(TEMPLATE_THETA_FULL[None, :], (N_BASE, 1)).astype(np.float32)
    theta_full_base[:, FREE_TO_FULL] = theta_free_base

    # repite K veces
    cond_rep = np.repeat(cond_base, K, axis=0).astype(np.float32)          # (N_BASE*K, 7)
    theta_full_rep = np.repeat(theta_full_base, K, axis=0).astype(np.float32)

    stim_rep  = cond_rep[:, 0].astype(np.float32)
    delay_rep = cond_rep[:, 1].astype(np.float32)
    side_rep  = cond_rep[:, 2].astype(np.float32)
    t1_rep    = cond_rep[:, 3].astype(np.float32)
    t2_rep    = cond_rep[:, 4].astype(np.float32)
    t3_rep    = cond_rep[:, 5].astype(np.float32)
    t4_rep    = cond_rep[:, 6].astype(np.float32)

    # simula choices
    choice_rep = get_choices_varying_numba(
        stim_rep, delay_rep, side_rep,
        t1_rep, t2_rep, t3_rep, t4_rep,
        theta_full_rep,
        DT, TH1, TH2, TH3
    ).astype(np.int16)

    # reshape por bloques (mantén estructura!)
    choice_rep = choice_rep.reshape(N_BASE, K)

    # construye targets probabilísticos P (N_BASE, 3)
    P = np.zeros((N_BASE, 3), dtype=np.float32)
    valid = np.ones((N_BASE,), dtype=np.bool_)

    n_miss_total = int((choice_rep == -1).sum())
    for i in range(N_BASE):
        ci = choice_rep[i]
        ok = ci != -1
        if ok.mean() < MIN_OK_FRAC:
            valid[i] = False
            continue
        counts = np.bincount(ci[ok].astype(np.int64), minlength=3).astype(np.float32)
        s = counts.sum()
        if s <= 0:
            valid[i] = False
            continue
        P[i] = counts / s

    print(f"Total miss: {n_miss_total} / {N_BASE*K} ({n_miss_total/(N_BASE*K):.2%})")
    print(f"Valid pairs: {int(valid.sum())} / {N_BASE} ({valid.mean():.2%})")

    # inputs: [theta_free, cond] sin repetir (base)
    X = np.concatenate([theta_free_base, cond_base], axis=1).astype(np.float32)
    X = X[valid]
    P = P[valid]

    X_t = torch.tensor(X, dtype=torch.float32)
    P_t = torch.tensor(P, dtype=torch.float32)
    return P_t, X_t, ok


@app.cell
def _(
    DataLoader,
    F,
    FREE_TO_FULL,
    P_t,
    TEMPLATE_THETA_FULL,
    TensorDataset,
    X_t,
    nn,
    os,
    random_split,
    torch,
):


    class ChoiceMLP(nn.Module):
        def __init__(self, in_dim, hidden=256, depth=3, p_drop=0.1):
            super().__init__()
            layers = [nn.LayerNorm(in_dim)]
            d = in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(p_drop)]
                d = hidden
            layers += [nn.Linear(d, 3)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)  # logits

    def soft_ce_mean(logits, p_target):
        # p_target: (B,3) probs
        logp = F.log_softmax(logits, dim=-1)
        return -(p_target * logp).sum(dim=-1).mean()

    @torch.no_grad()
    def expected_nll_sum(logits, p_target):
        # suma de E_{c~p_target}[-log p_hat(c)] en el batch
        logp = F.log_softmax(logits, dim=-1)
        return float((-(p_target * logp).sum(dim=-1)).sum().item())

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    HIDDEN   = int(os.environ.get("MLP_HIDDEN", "256"))
    DEPTH    = int(os.environ.get("MLP_DEPTH", "3"))
    DROPOUT  = float(os.environ.get("MLP_DROPOUT", "0.1"))
    LR       = float(os.environ.get("MLP_LR", "3e-4"))
    BATCH    = int(os.environ.get("MLP_BATCH", "1024"))
    EPOCHS   = int(os.environ.get("MLP_EPOCHS", "200"))
    VAL_FRAC = float(os.environ.get("MLP_VAL_FRAC", "0.1"))
    WD       = float(os.environ.get("MLP_WD", "1e-4"))

    # OJO: aquí asumo que ya tienes X_t y P_t (no y_t)
    model = ChoiceMLP(X_t.shape[1], hidden=HIDDEN, depth=DEPTH, p_drop=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    ds = TensorDataset(X_t, P_t)
    n_val = int(len(ds) * VAL_FRAC)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, drop_last=False, num_workers=0)

    @torch.no_grad()
    def eval_metrics():
        model.eval()
        tot = 0
        sum_nll = 0.0
        pseudo_corr = 0

        for xb, pb in val_loader:
            xb = xb.to(DEVICE)
            pb = pb.to(DEVICE)

            logits = model(xb)
            sum_nll += expected_nll_sum(logits, pb)
            tot += int(xb.shape[0])

            # “pseudo-acc” (opcional): compara argmax(p_target) vs argmax(pred)
            pred = logits.argmax(dim=1)
            targ = pb.argmax(dim=1)
            pseudo_corr += int((pred == targ).sum().item())

        return (sum_nll / max(1, tot)), (pseudo_corr / max(1, tot))

    best_val = 1e18
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        sum_train_nll = 0.0
        seen = 0

        for xb, pb in train_loader:
            xb = xb.to(DEVICE)
            pb = pb.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = soft_ce_mean(logits, pb)
            loss.backward()
            opt.step()

            # logging: NLL esperada (suma) para reporting
            sum_train_nll += expected_nll_sum(logits.detach(), pb)
            seen += int(xb.shape[0])

        train_nll = sum_train_nll / max(1, seen)
        val_nll, val_pacc = eval_metrics()

        print(f"ep {ep:03d} | train NLL {train_nll:.4f} | val NLL {val_nll:.4f} | pseudo-acc {val_pacc:.4f}")

        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": int(X_t.shape[1]),
        "hidden": HIDDEN,
        "depth": DEPTH,
        "dropout": DROPOUT,
        "free_to_full": FREE_TO_FULL.tolist(),
        "template_theta_full": TEMPLATE_THETA_FULL.tolist(),
        # opcional, pero útil para trazabilidad:
        "trained_with_soft_targets": True,
    }
    out_path = os.environ.get("MLP_OUT", "choice_mlp_surrogate.pt")
    torch.save(ckpt, out_path)
    print("Saved:", out_path)
    return


@app.cell
def _(DEVICE_EVAL, F, np, torch):
    @torch.no_grad()
    def mlp_choice_probs(mlp, theta_free: np.ndarray, cond_np: np.ndarray, chunk=200000):
        """
        theta_free: (n_free,)
        cond_np: (N,7)
        return probs: (N,3)
        """
        N = cond_np.shape[0]
        th = torch.tensor(theta_free, dtype=torch.float32, device=DEVICE_EVAL).view(1,-1).repeat(N, 1)
        cd = torch.tensor(cond_np, dtype=torch.float32, device=DEVICE_EVAL)
        x  = torch.cat([th, cd], dim=1)  # (N, n_free+7)

        probs = np.empty((N,3), dtype=np.float32)
        for s in range(0, N, chunk):
            e = min(N, s+chunk)
            logits = mlp(x[s:e])
            p = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
            probs[s:e] = p
        return probs

    @torch.no_grad()
    def mlp_nll_for_theta(mlp, theta_free_np, cond_np, choice_obs, chunk=200000):
        """
        NLL = sum_t -log p(choice_t | theta, cond_t)
        """
        N = cond_np.shape[0]
        th = torch.tensor(theta_free_np, dtype=torch.float32, device=DEVICE_EVAL).view(1,-1).repeat(N,1)
        cd = torch.tensor(cond_np, dtype=torch.float32, device=DEVICE_EVAL)
        x  = torch.cat([th, cd], dim=1)

        y = torch.tensor(choice_obs, dtype=torch.long, device=DEVICE_EVAL)
        nll = 0.0
        for s in range(0, N, chunk):
            e = min(N, s+chunk)
            logits = mlp(x[s:e])
            logp = F.log_softmax(logits, dim=1)
            idx = torch.arange(e-s, device=DEVICE_EVAL)
            nll += float((-logp[idx, y[s:e]]).sum().detach().cpu().numpy())
        return nll

    return (mlp_choice_probs,)


@app.cell
def _(np):
    def numba_probs_one_theta(
        theta_free,               # (n_free,)
        stimd, delayd, side, t1, t2, t3, t4,   # arrays reales (N,)
        expand_free_to_full,      # fn: (1,n_free)->(1,10) o tú lo adaptas
        get_choices_varying_numba,
        TEMPLATE_THETA_FULL, FREE_TO_FULL,
        DT, TH1, TH2, TH3,
        M=200,                    # replicates por trial
        chunk_trials=20000,       # para no petar RAM
        min_ok_frac=0.7,          # descarta trials con demasiados -1
    ):
        """
        Devuelve:
          probs_true: (N,3) float32
          miss: (N,) int32  número de misses en M replicates
        """
        N = stimd.shape[0]
        probs = np.zeros((N,3), dtype=np.float32)
        miss  = np.zeros((N,), dtype=np.int32)

        # theta_full vector (10,)
        theta_free = np.asarray(theta_free, dtype=np.float32)
        theta_full_1 = np.tile(TEMPLATE_THETA_FULL[None,:], (1,1)).astype(np.float32)
        theta_full_1[:, FREE_TO_FULL] = theta_free[None,:]
        theta_full_vec = theta_full_1[0]  # (10,)

        for s in range(0, N, chunk_trials):
            e = min(N, s + chunk_trials)
            B = e - s

            # repetimos trials M veces
            stim_rep  = np.repeat(stimd[s:e],  M).astype(np.float32)
            delay_rep = np.repeat(delayd[s:e], M).astype(np.float32)
            side_rep  = np.repeat(side[s:e],   M).astype(np.float32)
            t1_rep    = np.repeat(t1[s:e],     M).astype(np.float32)
            t2_rep    = np.repeat(t2[s:e],     M).astype(np.float32)
            t3_rep    = np.repeat(t3[s:e],     M).astype(np.float32)
            t4_rep    = np.repeat(t4[s:e],     M).astype(np.float32)

            theta_full_rep = np.tile(theta_full_vec[None,:], (B*M, 1)).astype(np.float32)

            ch = get_choices_varying_numba(
                stim_rep, delay_rep, side_rep,
                t1_rep, t2_rep, t3_rep, t4_rep,
                theta_full_rep,
                DT, TH1, TH2, TH3
            ).astype(np.int16)

            ch = ch.reshape(B, M)

            # agrega por trial
            for i in range(B):
                ci = ch[i]
                ok = (ci != -1)
                miss[s+i] = int((~ok).sum())
                if ok.mean() < min_ok_frac:
                    # demasiado miss -> lo dejamos probs=0
                    probs[s+i, :] = 0.0
                    continue
                cnt = np.bincount(ci[ok].astype(np.int64), minlength=3).astype(np.float32)
                probs[s+i, :] = cnt / max(1.0, cnt.sum())

        return probs, miss

    return (numba_probs_one_theta,)


@app.cell
def _(
    DT,
    FREE_TO_FULL,
    TEMPLATE_THETA_FULL,
    TH1,
    TH2,
    TH3,
    delayd,
    get_choices_varying_numba,
    mlp,
    mlp_choice_probs,
    mo,
    np,
    numba_probs_one_theta,
    ok,
    prior,
    side,
    stimd,
    t1,
    t2,
    t3,
    t4,
):
    from scipy.spatial import distance
    import numba
    numba.set_num_threads(8)
    def corr_flat(a, b):
        a = a.reshape(-1) - a.mean()
        b = b.reshape(-1) - b.mean()
        denom = (np.sqrt((a*a).mean()) * np.sqrt((b*b).mean()) + 1e-12)
        return float((a*b).mean() / denom)
    def build_cond_np(stimd, delayd, side, t1, t2, t3, t4) -> np.ndarray:
        # (N,7)
        return np.column_stack([
            stimd.astype(np.float32),
            delayd.astype(np.float32),
            side.astype(np.float32),
            t1.astype(np.float32),
            t2.astype(np.float32),
            t3.astype(np.float32),
            t4.astype(np.float32),
        ]).astype(np.float32, copy=False)

    def expand_free_to_full(theta_free_batch: np.ndarray) -> np.ndarray:
        K = theta_free_batch.shape[0]
        theta_full = np.tile(TEMPLATE_THETA_FULL[None, :], (K, 1)).astype(np.float32, copy=False)
        theta_full[:, FREE_TO_FULL] = theta_free_batch.astype(np.float32, copy=False)
        return theta_full

    # sample thetas para test
    N_THETAS = 5
    theta_free_batch = prior.sample((N_THETAS,)).numpy().astype(np.float32)

    rows = []

    for k in mo.status.progress_bar(
                range(N_THETAS),
                title="Fitting models",
                subtitle="",
                show_eta=True,
                show_rate=True,
            ):
        theta_free = theta_free_batch[k]

        # "ground truth" probs vía Numba MC (M replicates por trial)
        p_sim, miss = numba_probs_one_theta(
            theta_free,
            stimd, delayd, side, t1, t2, t3, t4,
            expand_free_to_full=None,  # no se usa en esta versión
            get_choices_varying_numba=get_choices_varying_numba,
            TEMPLATE_THETA_FULL=TEMPLATE_THETA_FULL,
            FREE_TO_FULL=FREE_TO_FULL,
            DT=DT, TH1=TH1, TH2=TH2, TH3=TH3,
            M=200,
            chunk_trials=1000,
            min_ok_frac=0.7
        )

        _ok = (miss == 0)  # si quieres estrictamente 0 misses; o usa miss < 0.3*M
        # surrogate probs
        cond_np = build_cond_np(stimd, delayd, side, t1, t2, t3, t4)
        p_hat = mlp_choice_probs(mlp, theta_free, cond_np, chunk=200000)

        if ok.sum() > 0:
            mse = float(np.mean((p_hat[_ok] - p_sim[_ok])**2))
            js  = float(np.mean(distance.jensenshannon(p_hat[_ok], p_sim[_ok])))
            cor = corr_flat(p_sim[_ok], p_hat[_ok])
        else:
            mse = np.nan; js = np.nan; cor = np.nan

        rows.append({
            "k": k,
            "mse_probs": mse,
            "js": js,
            "corr_probs": cor,
            "miss_rate": float(np.mean(miss > 0)),
        })

        print(f"[theta {k}] miss_rate={rows[-1]['miss_rate']:.3f} mse={mse:.4g} js={js:.4g} corr={cor:.4g}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
