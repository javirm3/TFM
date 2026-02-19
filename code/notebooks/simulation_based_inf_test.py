import marimo

__generated_with = "0.19.10"
app = marimo.App(width="full")


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
    return F, get_choices_varying_numba, nn, np, os, torch


@app.cell
def _(nn, os, torch):
    DEVICE_EVAL = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.environ.get("MLP_OUT", "choice_mlp_surrogate.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE_EVAL)

    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}

    in_dim  = int(ckpt.get("in_dim", 12))
    hidden  = int(ckpt.get("hidden", 256))
    depth   = int(ckpt.get("depth", 3))
    dropout = float(ckpt.get("dropout", 0.1))


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
            return self.net(x)

    mlp = ChoiceMLP(in_dim=in_dim, hidden=hidden, depth=depth, p_drop=dropout).to(DEVICE_EVAL)
    mlp.load_state_dict(state, strict=True)
    mlp.eval()

    print("Loaded surrogate:", {"path": ckpt_path, "in_dim": in_dim, "hidden": hidden, "depth": depth, "dropout": dropout, "device": str(DEVICE_EVAL)})
    return DEVICE_EVAL, mlp


@app.cell
def _():
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
    build_cond_np,
    delayd,
    get_choices_varying_numba,
    mlp,
    mlp_choice_probs,
    np,
    numba_probs_one_theta,
    prior,
    side,
    stimd,
    t1,
    t2,
    t3,
    t4,
):
    from scipy.spatial import distance

    def corr_flat(a, b):
        a = a.reshape(-1) - a.mean()
        b = b.reshape(-1) - b.mean()
        denom = (np.sqrt((a*a).mean()) * np.sqrt((b*b).mean()) + 1e-12)
        return float((a*b).mean() / denom)

    # sample thetas para test
    N_THETAS = 5
    theta_free_batch = prior.sample((N_THETAS,)).numpy().astype(np.float32)

    rows = []
    for k in range(N_THETAS):
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
            chunk_trials=10000,
            min_ok_frac=0.7
        )

        ok = (miss == 0)  # si quieres estrictamente 0 misses; o usa miss < 0.3*M
        # surrogate probs
        cond_np = build_cond_np(stimd, delayd, side, t1, t2, t3, t4)
        p_hat = mlp_choice_probs(mlp, theta_free, cond_np, chunk=200000)

        if ok.sum() > 0:
            mse = float(np.mean((p_hat[ok] - p_sim[ok])**2))
            js  = float(np.mean(distance.jensenshannon(p_hat[ok], p_sim[ok])))
            cor = corr_flat(p_sim[ok], p_hat[ok])
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
