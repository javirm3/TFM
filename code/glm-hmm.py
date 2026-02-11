import numpy as np
import pandas as pd
import paths
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array
import matplotlib.pyplot as plt
import optax
import seaborn as sns
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

from typing import NamedTuple, Optional, Tuple, Union

from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.types import IntScalar, Scalar
from dynamax.utils.plotting import gradient_cmap



sns.set_style("white")

color_names = [
    "windows blue",
    "red",
    "amber"
]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

def build_sequence_from_df(df_sub):
    # ordena por tiempo si tienes una columna trial_idx o timestamp
    df_sub = df_sub.sort_values("trial_idx")

    # y en {0,1,2}
    y = df_sub["response"].to_numpy().astype(int)

    # diseño: intercept + delay + stimulus + delayxstim
    delay = df_sub["delay_d"].to_numpy().astype(float)
    stim  = df_sub["stim_d"].to_numpy().astype(float)
    SL = df_sub[df_sub["x_c"] == 'L']["stim_d"].to_numpy().astype(float)
    SL = (df_sub["x_c"] == 'L').astype(float) * df_sub["stim_d"].to_numpy().astype(float)
    SC = (df_sub["x_c"] == 'C').astype(float) * df_sub["stim_d"].to_numpy().astype(float)
    SR = (df_sub["x_c"] == 'R').astype(float) * df_sub["stim_d"].to_numpy().astype(float)
   
    delayxstim = delay * stim
    previous_outcome = df_sub["performance"].shift(1).fillna(0).to_numpy().astype(float)
    print(df_sub["delay_d"].describe())
    print(df_sub["stim_d"].describe())

    X = np.column_stack([np.ones(len(df_sub)), delay, SL, SC, SR, previous_outcome]).astype(np.float32)

    return jnp.asarray(y), jnp.asarray(X)


class ParamsSoftmaxGLMHMMEmissions(NamedTuple):
    # weights: (K, C, M)
    weights: Union[Float[Array, "num_states num_classes input_dim"], ParameterProperties]


class SoftmaxGLMHMMEmissions(HMMEmissions):
    def __init__(self,
                 num_states: int,
                 num_classes: int,
                 input_dim: int,
                 weight_scale: Scalar = 1.0,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 100):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.weight_scale = weight_scale

    @property
    def emission_shape(self):
        # y_t es un entero (clase) -> shape ()
        return ()

    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(self, key, method="random", emission_weights=None):
        if emission_weights is None:
            key, subkey = jr.split(key)
            W = self.weight_scale * jr.normal(subkey, (self.num_states, self.num_classes, self.input_dim))
        else:
            W = emission_weights

        params = ParamsSoftmaxGLMHMMEmissions(weights=W)
        props = ParamsSoftmaxGLMHMMEmissions(
            weights=ParameterProperties()  # puedes añadir regularización/prior aquí si quieres
        )
        return params, props

    def log_prior(self, params):
        return 0

    def distribution(self, params, state, inputs):
        # logits: (C,)  = (C,M) @ (M,)
        logits = params.weights[state] @ inputs
        return tfd.Categorical(logits=logits)
    

class ParamsSoftmaxGLMHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsSoftmaxGLMHMMEmissions


class SoftmaxGLMHMM(HMM):
    def __init__(self,
                 num_states: int,
                 num_classes: int,
                 input_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]] = 1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]] = 1.1,
                 transition_matrix_stickiness: Scalar = 0.0,
                 weight_scale: Scalar = 1.0,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 100):

        self.inputs_dim = input_dim
        self.num_states = num_states
        self.num_classes = num_classes

        initial_component = StandardHMMInitialState(
            num_states, initial_probs_concentration=initial_probs_concentration)

        transition_component = StandardHMMTransitions(
            num_states,
            concentration=transition_matrix_concentration,
            stickiness=transition_matrix_stickiness)

        emission_component = SoftmaxGLMHMMEmissions(
            num_states=num_states,
            num_classes=num_classes,
            input_dim=input_dim,
            weight_scale=weight_scale,
            m_step_optimizer=m_step_optimizer,
            m_step_num_iters=m_step_num_iters)

        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self) -> Tuple[int, ...]:
        return (self.inputs_dim,)

    def initialize(self,
                   key: Array = jr.PRNGKey(0),
                   method: str = "prior",
                   initial_probs: Optional[Float[Array, " num_states"]] = None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
                   emission_weights: Optional[Float[Array, "num_states num_classes input_dim"]] = None
                   ) -> Tuple[HMMParameterSet, HMMPropertySet]:

        key1, key2, key3 = jr.split(key, 3)
        params, props = dict(), dict()

        params["initial"], props["initial"] = self.initial_component.initialize(
            key1, method=method, initial_probs=initial_probs)

        params["transitions"], props["transitions"] = self.transition_component.initialize(
            key2, method=method, transition_matrix=transition_matrix)

        params["emissions"], props["emissions"] = self.emission_component.initialize(
            key3, method=method, emission_weights=emission_weights)

        return ParamsSoftmaxGLMHMM(**params), ParamsSoftmaxGLMHMM(**props)

num_states= 2         # nº estados
emmision_dim = 3          # 3 choices
input_dim = 6          # intercept + delay + stim_L + stim_C + stim_R + previous_outcome

model = SoftmaxGLMHMM(num_states=num_states, num_classes=emmision_dim, input_dim=input_dim, m_step_num_iters=100, transition_matrix_stickiness=10.0)

key = jr.PRNGKey(12345)
params, props = model.initialize(key=key)

df = pd.read_parquet(paths.DATA_PATH/"df_filtered.parquet")
y, X = build_sequence_from_df(df[df["subject"] == "A92"])

fitted_params, lps = model.fit_em(params=params, props=props, emissions=y, inputs=X, num_iters=50)

posterior = model.smoother(params=fitted_params, emissions=y, inputs=X)


import numpy as np
import matplotlib.pyplot as plt

T = int(y.shape[0])

lps_np = np.asarray(lps)

plt.figure(figsize=(6,3))
plt.plot(lps_np / T, "-o", ms=3)
plt.xlabel("EM Iteration")
plt.ylabel("Avg log prob per trial")
plt.title("EM log-likelihood")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

W = np.asarray(fitted_params.emissions.weights)  # (K, C, M)
K, C, M = W.shape

vmax = np.max(np.abs(W))  # escala simétrica alrededor de 0

fig, axs = plt.subplots(1, C, figsize=(3.5*C, 3), sharey=True)
if C == 1:
    axs = [axs]

for c in range(C):
    im = axs[c].imshow(W[:, c, :], aspect="auto", interpolation="none",
                       cmap="Greys", vmin=-vmax, vmax=vmax)
    axs[c].set_title(f"Emission weights (class {c})")
    axs[c].set_xlabel("input dim")
    axs[c].set_xticks(np.arange(M))
    axs[c].set_xticklabels([f"x{j}" for j in range(M)])
    if c == 0:
        axs[c].set_ylabel("state")
        axs[c].set_yticks(np.arange(K))
        axs[c].set_yticklabels(np.arange(K) + 1)
    plt.colorbar(im, ax=axs[c], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

posterior = model.smoother(params=fitted_params, emissions=y, inputs=X)
gamma = np.asarray(posterior.smoothed_probs)  # (T, K)

# Elige una ventana para visualizar
plot_slice = (0, min(1000, T))   # ajusta

fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

# 1) Heatmap de probabilidades posteriores
im = axs[0].imshow(gamma[plot_slice[0]:plot_slice[1]].T,
                   aspect="auto", interpolation="none",
                   cmap="Greys", vmin=0, vmax=1)
axs[0].set_ylabel("state k")
axs[0].set_yticks(np.arange(K))
axs[0].set_yticklabels(np.arange(K) + 1)
axs[0].set_title(r"Posterior probs $p(z_t=k \mid y_{1:T}, x_{1:T})$")
plt.colorbar(im, ax=axs[0], fraction=0.02, pad=0.02)

# 2) Estado MAP (argmax) como “barra”
z_map = np.argmax(gamma, axis=1)
axs[1].imshow(z_map[None, plot_slice[0]:plot_slice[1]],
              aspect="auto", interpolation="none",
              cmap=cmap, vmin=0, vmax=len(colors)-1)
axs[1].set_yticks([])
axs[1].set_xlabel("time (trial)")
axs[1].set_title("Most likely state (MAP)")

plt.tight_layout()
plt.show()

import numpy as np
A = np.asarray(fitted_params.transitions.transition_matrix)
print("Transition matrix A:")
print(np.round(A, 3))

import numpy as np
import jax.numpy as jnp

# 1) posterior sobre estados (smoothed)
posterior = model.smoother(params=fitted_params, emissions=y, inputs=X)
gamma = np.asarray(posterior.smoothed_probs)   # (T, K)

# 2) logits por estado y trial: logits[t,k,c] = W[k,c,:] @ X[t,:]
W = np.asarray(fitted_params.emissions.weights)  # (K, C, M)
logits = np.einsum("kcm,tm->tkc", W, np.asarray(X))  # (T, K, C)

# 3) softmax en C para obtener p(y|z=k,x)
logits = logits - logits.max(axis=2, keepdims=True)     # estabilidad numérica
p_y_given_z = np.exp(logits)
p_y_given_z = p_y_given_z / p_y_given_z.sum(axis=2, keepdims=True)  # (T,K,C)

# 4) mezcla por gamma: p_pred[t,c] = sum_k gamma[t,k] * p_y_given_z[t,k,c]
p_pred = np.einsum("tk,tkc->tc", gamma, p_y_given_z)    # (T, C)
filt = model.filter(params=fitted_params, emissions=y, inputs=X)
alpha = np.asarray(filt.filtered_probs)   # (T, K)  p(z_t | y_{1:t})

p_pred = np.einsum("tk,tkc->tc", alpha, p_y_given_z)
# p_pred[:,0]=pL, p_pred[:,1]=pC, p_pred[:,2]=pR (según tu codificación)
pL, pC, pR = p_pred[:,0], p_pred[:,1], p_pred[:,2]

df_sub = df[df["subject"] == "A92"].sort_values("trial_idx").copy()

df_sub["pL"] = pL
df_sub["pC"] = pC
df_sub["pR"] = pR

# (opcional) predicción de clase MAP
df_sub["pred_choice"] = np.argmax(p_pred, axis=1)

df_sub.to_parquet(paths.DATA_PATH / "predictions.parquet", index=False)
print("Saved:", paths.DATA_PATH / "predictions.parquet")