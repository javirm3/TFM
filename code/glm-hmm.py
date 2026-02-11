import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
import paths
import jax
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
from dynamax.hidden_markov_model.models.abstractions import HMMTransitions
from dynamax.hidden_markov_model import CategoricalRegressionHMM
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

def action_trace(r_c: jnp.ndarray, tau: float) -> jnp.ndarray:
    """
    r_c: (T,1) encoded as (L,C,R)
    returns A: (T,3) with A_t^X = sum_{k>=1} r_{t-k}^X * exp(-k/tau)
    using A_t = lam*A_{t-1} + r_{t-1}
    """
    
    r_onehot = jax.nn.one_hot(r_c.squeeze(), 3).astype(jnp.float32)
    lam = jnp.exp(-1.0 / tau).astype(jnp.float32)
    r_prev = jnp.vstack([jnp.zeros((1, r_onehot.shape[1]), dtype=jnp.float32), r_onehot[:-1]])

    def step(prev, current):
        new = lam * prev + current
        return new, new

    _, A = jax.lax.scan(step, jnp.zeros((r_onehot.shape[1],), dtype=jnp.float32), r_prev)
    return A


def build_sequence_from_df(df_sub: pl.DataFrame):
    df_sub = df_sub.sort("trial_idx")

    df_sub = df_sub.with_columns([
        pl.col("response").cast(pl.Int32),

        (pl.col("x_c") == "L").cast(pl.Float32).alias("biasL"),
        (pl.col("x_c") == "C").cast(pl.Float32).alias("biasC"),
        (pl.col("x_c") == "R").cast(pl.Float32).alias("biasR"),
        
        pl.lit(1.0).cast(pl.Float32).alias("bias"),
        
        pl.col("delay_d").cast(pl.Float32).alias("delay"),
        pl.col("stim_d").cast(pl.Float32).alias("stim"),

        ((pl.col("x_c") == "L") * pl.col("stim_d")).cast(pl.Float32).alias("SL"),
        ((pl.col("x_c") == "C") * pl.col("stim_d")).cast(pl.Float32).alias("SC"),
        ((pl.col("x_c") == "R") * pl.col("stim_d")).cast(pl.Float32).alias("SR"),

        pl.col("performance").shift(1).fill_null(0).cast(pl.Float32).alias("previous_outcome"),
    ])

    y = df_sub["response"].to_numpy()

    X = df_sub.select(["bias", "biasC", "biasR", "delay", "SL", "SC", "SR", "previous_outcome"]).to_numpy().astype(np.float32)

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
    
    def predict_choice_probs(self, params, emissions, inputs):
        """
        Devuelve p(y_t=c | y_{1:t-1}, x_{1:t})
        (one-step-ahead predictive distribution).
        """

        filt = self.filter(params=params, emissions=emissions, inputs=inputs)
        alpha = filt.filtered_probs

        T, K = alpha.shape
        A = self.transition_component._compute_transition_matrices(params.transitions, inputs)

        if A.ndim == 2:  # homogeneous transitions
            A = jnp.broadcast_to(A[None, :, :], (T-1, K, K))

        def emission_probs_t(t):
            return jax.vmap(lambda k: self.emission_component.distribution(params.emissions, k, inputs[t]).probs_parameter())(jnp.arange(K))

        p_y_given_z = jax.vmap(emission_probs_t)(jnp.arange(T))  # (T,K,C)
        pi = params.initial.probs  # (K,)
        pred_z = jnp.vstack([pi, jnp.einsum("tk,tkj->tj", alpha[:-1], A)])

        return jnp.einsum("tk,tkc->tc", pred_z, p_y_given_z)

class ParamsInputDrivenTransitions(NamedTuple):
    bias: Union[Float[Array, "num_states num_states"], ParameterProperties]
    weights: Union[Float[Array, "num_states num_states input_dim"], ParameterProperties]

class InputDrivenSoftmaxTransitions(HMMTransitions):
    """
    Transiciones no-homogéneas:
      p(z_t=j | z_{t-1}=i, u_t) = softmax_j( bias[i,j] + <weights[i,j,:], u_t> )
    Nota: la m_step base usa inputs[1:], así que u_t se alinea con transición (t-1)->t.
    """

    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 weight_scale: Scalar = 0.01,
                 m_step_optimizer=None,
                 m_step_num_iters: int = 50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.input_dim = input_dim
        self.weight_scale = weight_scale

    def initialize(self, key: Optional[Array] = None, method: str = "prior", bias=None, weights=None):
        if key is None:
            key = jr.PRNGKey(0)
        key1, key2 = jr.split(key, 2)
        K, D = self.num_states, self.input_dim

        if bias is None:
            b = jnp.zeros((K, K), dtype=jnp.float32)
        else:
            b = jnp.asarray(bias, dtype=jnp.float32)

        if weights is None:
            W = self.weight_scale * jr.normal(key2, (K, K, D), dtype=jnp.float32)
        else:
            W = jnp.asarray(weights, dtype=jnp.float32)

        params = ParamsInputDrivenTransitions(bias=b, weights=W)
        props  = ParamsInputDrivenTransitions(
            bias=ParameterProperties(),
            weights=ParameterProperties()
        )
        return params, props

    def log_prior(self, params: ParamsInputDrivenTransitions) -> Scalar:
        return 0.0

    def distribution(self,
                     params: ParamsInputDrivenTransitions,
                     state: IntScalar,
                     inputs: Optional[Float[Array, " input_dim"]] = None
                     ) -> tfd.Distribution:
        # inputs: (D,)
        # logits: (K,) para el siguiente estado
        u = inputs
        i = jnp.asarray(state, dtype=jnp.int32)
        logits = params.bias[i] + jnp.einsum("kd,d->k", params.weights[i], u)
        return tfd.Categorical(logits=logits)
    

num_states= 2         # nº estados
emmision_dim = 3          # 3 choices
input_dim = 7          # intercept + delay + stim_L + stim_C + stim_R + previous_outcome

model = SoftmaxGLMHMM(num_states=num_states, num_classes=emmision_dim, input_dim=input_dim, m_step_num_iters=100, transition_matrix_stickiness=10.0)

model2 = CategoricalRegressionHMM(num_states=num_states, num_classes=emmision_dim, input_dim=input_dim-1, transition_matrix_stickiness=10.0, m_step_optimizer=optax.adam(1e-3), m_step_num_iters=500,)

model3 = SoftmaxGLMHMM(num_states=num_states, num_classes=emmision_dim, input_dim=input_dim-1, m_step_num_iters=100, transition_matrix_stickiness=10.0)

key = jr.PRNGKey(12345)
params, props = model.initialize(key=key)
params2, props2 = model2.initialize(key=key)
params3, props3 = model3.initialize(key=key)

df = pl.read_parquet(paths.DATA_PATH/"df_filtered.parquet")
y, X = build_sequence_from_df(df.filter(pl.col("subject") == "A92"))

At = action_trace(y, tau=50.0)

fitted_params, lps = model.fit_em(params=params, props=props, emissions=y, inputs=X[:,1:], num_iters=50)
fitted_params2, lps2 = model2.fit_em(params=params2, props=props2, emissions=y, inputs=jnp.concatenate([X[:, :1], X[:, 3:]], axis=1), num_iters=50)
fitted_params3, lps3 = model3.fit_em(params=params3, props=props3, emissions=y, inputs=jnp.concatenate([X[:, :1], X[:, 3:]], axis=1), num_iters=50)


posterior = model.smoother(params=fitted_params, emissions=y, inputs=X[:,1:])
posterior2 = model2.smoother(params=fitted_params2, emissions=y, inputs=jnp.concatenate([X[:, :1], X[:, 3:]], axis=1))
posterior3 = model3.smoother(params=fitted_params3, emissions=y, inputs=jnp.concatenate([X[:, :1], X[:, 3:]], axis=1))

T = int(y.shape[0])

lps_np = np.asarray(lps)
lps2_np = np.asarray(lps2)

plt.figure(figsize=(6,3))
plt.plot(lps_np / T, "-o", ms=3)
plt.plot(lps2_np / T, "-o", ms=3)
plt.plot(np.asarray(lps3) / T, "-o", ms=3)
plt.legend(["SoftmaxGLMHMM", "CategoricalRegressionHMM", "SoftmaxGLMHMM (no bias)"])
plt.xlabel("EM Iteration")
plt.ylabel("Avg log prob per trial")
plt.title("EM log-likelihood")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


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

gamma = np.asarray(posterior.smoothed_probs)  # (T, K)

# Elige una ventana para visualizar
plot_slice = (0, min(10000, T))   # ajusta

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

A = np.asarray(fitted_params.transitions.transition_matrix)
print("Transition matrix A:")
print(np.round(A, 3))

gamma = np.asarray(posterior.smoothed_probs)   # (T, K)

# 2) logits por estado y trial: logits[t,k,c] = W[k,c,:] @ X[t,:]
W = np.asarray(fitted_params.emissions.weights)  # (K, C, M)
logits = np.einsum("kcm,tm->tkc", W, X[:,1:])  # (T, K, C)

# 3) softmax en C para obtener p(y|z=k,x)
logits = logits - logits.max(axis=2, keepdims=True)     # estabilidad numérica
p_y_given_z = np.exp(logits)
p_y_given_z = p_y_given_z / p_y_given_z.sum(axis=2, keepdims=True)  # (T,K,C)

# 4) mezcla por gamma: p_pred[t,c] = sum_k gamma[t,k] * p_y_given_z[t,k,c]
p_pred = np.einsum("tk,tkc->tc", gamma, p_y_given_z)    # (T, C)
filt = model.filter(params=fitted_params, emissions=y, inputs=X[:,1:]
alpha = np.asarray(filt.filtered_probs)   # (T, K)  p(z_t | y_{1:t})

p_pred = np.einsum("tk,tkc->tc", alpha, p_y_given_z)

# p_pred = np.asarray(model3.predict_choice_probs(fitted_params3, y, jnp.concatenate([X[:, :1], X[:, 3:]], axis=1)))
# p_pred = np.asarray(model.predict_choice_probs(fitted_params, y, X[:,1:]))

pL, pC, pR = p_pred[:,0], p_pred[:,1], p_pred[:,2]

df_sub = df.filter(pl.col("subject") == "A92").sort("trial_idx")

df_sub = df_sub.with_columns([pl.Series("pL", pL), pl.Series("pC", pC), pl.Series("pR", pR), pl.Series("pred_choice", np.argmax(p_pred, axis=1))])

df_sub.write_parquet(paths.DATA_PATH / "predictions.parquet")
print("Saved:", paths.DATA_PATH / "predictions.parquet")