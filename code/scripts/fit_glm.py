import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
import jax.numpy as jnp
import seaborn as sns
from scipy.special import softmax, log_softmax
from scipy.optimize import minimize
# from glmhmmt.features import build_sequence_from_df
import glmhmmt.plots as plots

sns.set_style("white")

# ── Data ──────────────────────────────────────────────────────────────────────

def build_sequence_from_df(df_sub: pl.DataFrame, tau = 50):
    df_sub = df_sub.sort("trial_idx")
    df_sub = df_sub.with_columns([
        pl.col("response").cast(pl.Int32),

        (pl.col("x_c") == "L").cast(pl.Float32).alias("biasL"),
        (pl.col("x_c") == "R").cast(pl.Float32).alias("biasR"),
        
        pl.lit(1.0).cast(pl.Float32).alias("bias"),
        
        pl.col("delay_d").cast(pl.Float32).alias("delay"),
        ((pl.col("x_c") == "L") * pl.col("onset")).cast(pl.Float32).alias("onsetL"),
        ((pl.col("x_c") == "C") * pl.col("onset")).cast(pl.Float32).alias("onsetC"),
        ((pl.col("x_c") == "R") * pl.col("onset")).cast(pl.Float32).alias("onsetR"),

        ((pl.col("x_c") == "L") * pl.col("stimd_n")).cast(pl.Float32).alias("SL"),
        ((pl.col("x_c") == "C") * pl.col("stimd_n")).cast(pl.Float32).alias("SC"),
        ((pl.col("x_c") == "R") * pl.col("stimd_n")).cast(pl.Float32).alias("SR"),

        ((pl.col("x_c") == "L") * pl.col("delay_d")).cast(pl.Float32).alias("DL"),
        ((pl.col("x_c") == "C") * pl.col("delay_d")).cast(pl.Float32).alias("DC"),
        ((pl.col("x_c") == "R") * pl.col("delay_d")).cast(pl.Float32).alias("DR"),
        
        ((pl.col("x_c") == "L") * pl.col("stimd_n") * pl.col("delay_d")).cast(pl.Float32).alias("SLxdelay"),
        ((pl.col("x_c") == "C") * pl.col("stimd_n") * pl.col("delay_d")).cast(pl.Float32).alias("SCxdelay"),
        ((pl.col("x_c") == "R") * pl.col("stimd_n") * pl.col("delay_d")).cast(pl.Float32).alias("SRxdelay"),


        pl.col("performance").shift(1).fill_null(0).cast(pl.Float32).alias("previous_outcome"),
        pl.col("response").shift(1).fill_null(0.0).eq(0).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).alias("A_L"),
        pl.col("response").shift(1).fill_null(0.0).eq(1).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).alias("A_C"),
        pl.col("response").shift(1).fill_null(0.0).eq(2).cast(pl.Float32).ewm_mean(half_life=tau, adjust=False).alias("A_R"),
    ])
    df_sub = df_sub.with_columns([
        pl.col("previous_outcome").shift(1).fill_null(0.0).ewm_mean(half_life=tau, adjust=False).alias("A_plus"),
        (1.0 - pl.col("previous_outcome")).shift(1).fill_null(0.0).ewm_mean(half_life=tau, adjust=False).alias("A_minus"),
        (pl.col("A_L") * pl.col("delay_d")).cast(pl.Float32).alias("ALxdelay"),
        (pl.col("A_R") * pl.col("delay_d")).cast(pl.Float32).alias("ARxdelay"),
    ])

    y = df_sub["response"].to_numpy()
    
    X_base = df_sub.select(["biasL", "biasR", "delay", "onsetL", "onsetC", "onsetR", "SL", "SC", "SR", "SLxdelay", "SCxdelay", "SRxdelay", "A_L", "A_C", "A_R"])
    X = jnp.asarray(X_base.to_numpy().astype(jnp.float32))
    U_base = df_sub.select(["A_plus", "A_minus"]).to_numpy().astype(jnp.float32)
    U = jnp.asarray(U_base)

    A_plus = jnp.asarray(df_sub["A_plus"].to_numpy())[:, None]
    A_minus = jnp.asarray(df_sub["A_minus"].to_numpy())[:, None]

    names = {
        "X_cols": X_base.columns,
        "U_cols": ["A_plus", "A_minus"],
    }
    return jnp.asarray(y), jnp.asarray(X), jnp.asarray(U), names, jnp.concatenate([A_plus, A_minus], axis=1)

df = pl.read_parquet(paths.DATA_PATH / "df_filtered.parquet")
y, X, U, names, _ = build_sequence_from_df(df.filter(pl.col("subject") == "A92"))

y_np = np.asarray(y)        # (T,)  int {0,1,2}
X_np = np.asarray(X)        # (T, M)
T, M = X_np.shape
print(f"T: {T}  input_dim: {M}")
print("Features:", names["X_cols"])

# ── Fit multinomial softmax GLM with C (class 1) as base category ─────────────
# Directly fixes C's logit to 0 — no remapping needed.
# logits = [W[0]@x,  0.0 (base=C),  W[1]@x]
# W shape (2, M): row 0 = L-vs-C, row 1 = R-vs-C  (matches model.py convention)

def neg_log_likelihood(w_flat):
    W_ = w_flat.reshape(2, M)
    eta_L = X_np @ W_[0]                                     # (T,)
    eta_R = X_np @ W_[1]                                     # (T,)
    logits = np.stack([eta_L, np.zeros(T), eta_R], axis=1)  # (T, 3): [L, C, R]
    log_p = log_softmax(logits, axis=1)                      # (T, 3)
    return -np.sum(log_p[np.arange(T), y_np])

result = minimize(neg_log_likelihood, np.zeros(2 * M), method="L-BFGS-B",
                  options={"maxiter": 2000, "ftol": 1e-12})
print("Optimisation success:", result.success, "|", result.message)

W = result.x.reshape(2, M)
C = 3   # total number of classes

# ── Log-likelihood per trial ──────────────────────────────────────────────────
eta_L = X_np @ W[0]
eta_R = X_np @ W[1]
logits_all = np.stack([eta_L, np.zeros(T), eta_R], axis=1)  # (T, 3): [L, C, R]
p_pred = softmax(logits_all, axis=1)                         # (T, 3): [L, C, R]
ll_per_trial = np.mean(np.log(p_pred[np.arange(T), y_np] + 1e-12))
print(f"LL per trial: {ll_per_trial:.4f}")

n_params = 2 * M    # only L and R weight vectors; C is the base (no params)
bic = -2 * ll_per_trial * T + n_params * np.log(T)
print(f"BIC: {bic:.1f}")

# ── Fold into congruent / incongruent using softmax (same as model.py) ────────
# For each L/R/C triple, average all three congruent/incongruent contributions:
#   cong   = (P(L)|fL + P(R)|fR + P(C)|fC) / 3 − BASE
#   incong = (P(R)|fL + P(L)|fR + (P(L)+P(R))/2|fC) / 3 − BASE
# bias has no C counterpart so averages only 2 terms.
feat = names["X_cols"]
idx = {f: i for i, f in enumerate(feat)}

# (fL, fR, fC or None, label)
lr_triples = [("biasL",    "biasR",    None,        "bias"),
              ("onsetL",   "onsetR",   "onsetC",    "onset"),
              ("SL",       "SR",       "SC",        "S"),
              ("SLxdelay", "SRxdelay", "SCxdelay",  "Sxdelay")]

neutral = ["delay"]   # no side — average P(L) and P(R) symmetrically

BASE = 1 / 3

folded_labels = []
P_cong   = []
P_incong = []

for fL, fR, fC, lbl in lr_triples:
    iL, iR = idx[fL], idx[fR]
    p_fL = softmax([W[0, iL], 0.0, W[1, iL]])   # logits = [W[0,fL], 0, W[1,fL]]
    p_fR = softmax([W[0, iR], 0.0, W[1, iR]])

    cong_vals  = [p_fL[0] - BASE, p_fR[2] - BASE]   # P(L)|fL, P(R)|fR
    incong_vals = [p_fL[2] - BASE, p_fR[0] - BASE]  # P(R)|fL, P(L)|fR

    if fC is not None and fC in idx:
        iC = idx[fC]
        p_fC = softmax([W[0, iC], 0.0, W[1, iC]])
        cong_vals.append(p_fC[1] - BASE)              # P(C)|fC  — congruent
        incong_vals.append((p_fC[0] + p_fC[2]) / 2 - BASE)  # lateral|fC — incong

    P_cong.append(float(np.mean(cong_vals)))
    P_incong.append(float(np.mean(incong_vals)))
    folded_labels.append(lbl)

for fn in neutral:
    i = idx[fn]
    p_fn = softmax([W[0, i], 0.0, W[1, i]])
    p_avg = float((p_fn[0] + p_fn[2]) / 2 - BASE)   # symmetric — same for both
    P_cong.append(p_avg)
    P_incong.append(p_avg)
    folded_labels.append(fn)

P_cong   = np.array(P_cong)
P_incong = np.array(P_incong)
n_folded = len(folded_labels)

print("\nFolded congruency (ΔP from 1/3 baseline):")
for lbl, pc, pi in zip(folded_labels, P_cong, P_incong):
    print(f"  {lbl:12s}  cong={pc:+.3f}  incong={pi:+.3f}")

# ── Plot folded weights ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 0.5 * n_folded + 1.5))
x = np.arange(n_folded)
w = 0.35
ax.barh(x + w/2, P_cong,   w, label="congruent",   color="steelblue")
ax.barh(x - w/2, P_incong, w, label="incongruent", color="tomato")
ax.set_yticks(x)
ax.set_yticklabels(folded_labels)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("ΔP (from 1/3 baseline)")
ax.set_title("GLM — congruent vs incongruent (probability scale)")
ax.legend()
plt.tight_layout()
plt.show()

# ── Plot weights ──────────────────────────────────────────────────────────────
# W shape (2, M): row 0 = L-vs-C, row 1 = R-vs-C  (matches model.py convention)
fig, ax = plt.subplots(figsize=(max(6, 0.8 * M), 3))
vmax = np.max(np.abs(W))
im = ax.imshow(W, aspect="auto", interpolation="none",
               cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_title("GLM weights (softmax, base = C)")
ax.set_xlabel("input feature")
ax.set_ylabel("contrast vs C")
ax.set_xticks(np.arange(M))
ax.set_xticklabels(names["X_cols"], rotation=45, ha="right")
ax.set_yticks([0, 1])
ax.set_yticklabels(["L vs C", "R vs C"])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# ── Bar plot per feature ──────────────────────────────────────────────────────
contrast_names = ["L vs C", "R vs C"]
fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
for c in range(2):
    axs[c].barh(np.arange(M), W[c], color=["steelblue" if v >= 0 else "tomato" for v in W[c]])
    axs[c].set_title(contrast_names[c])
    axs[c].set_yticks(np.arange(M))
    axs[c].set_yticklabels(names["X_cols"])
    axs[c].axvline(0, color="k", lw=0.8)
    axs[c].set_xlabel("weight")
plt.suptitle("GLM weights per contrast (base = C)", y=1.01)
plt.tight_layout()
plt.show()

# ── Save predictions ──────────────────────────────────────────────────────────
pL, pC, pR = p_pred[:, 0], p_pred[:, 1], p_pred[:, 2]
df_sub = df.filter(pl.col("subject") == "A92").sort("trial_idx")
df_sub = df_sub.with_columns([
    pl.Series("pL", pL),
    pl.Series("pC", pC),
    pl.Series("pR", pR),
    pl.Series("pred_choice", np.argmax(p_pred, axis=1)),
])
df_sub.write_parquet(paths.DATA_PATH / "predictions_glm.parquet")
print("Saved:", paths.DATA_PATH / "predictions_glm.parquet")

plot_df = plots.prepare_predictions_df(df_sub)
plots.plot_categorical_performance_all(plot_df, "glm")
plots.plot_categorical_strat_by_side(plot_df, subject = "A92", model_name = "GLM")

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
plots.plot_delay_or_stim_1d_on_ax(ax1, plot_df, subject="A92", n_bins=7, which="delay")
plots.plot_delay_or_stim_1d_on_ax(ax2, plot_df, subject="A92", n_bins=7, which="stim")
plt.show()