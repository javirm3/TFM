import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Float, Array
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import NamedTuple, Optional, Tuple, Union

from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.hidden_markov_model.models.abstractions import HMMTransitions
from dynamax.types import IntScalar, Scalar


class ParamsSoftmaxGLMHMMEmissions(NamedTuple):
    # weights: (K, C, M)
    weights: Union[Float[Array, "num_states num_classes input_dim"], ParameterProperties]


class SoftmaxGLMHMMEmissions(HMMEmissions):
    def __init__(self,
                 num_states: int,
                 num_classes: int,
                 input_dim: int,
                 emission_input_dim: int,
                 weight_scale: Scalar = 1.0,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 100):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.emission_input_dim = emission_input_dim
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
        x = inputs[:self.emission_input_dim]  # (M,)
        # logits: (C,)  = (C,M) @ (M,)
        logits = params.weights[state] @ x
        return tfd.Categorical(logits=logits)
    
class ParamsInputDrivenTransitions(NamedTuple):
    bias: Union[Float[Array, "num_states num_states"], ParameterProperties]
    weights: Union[Float[Array, "num_states num_states input_dim"], ParameterProperties]

class InputDrivenSoftmaxTransitions(HMMTransitions):
    def __init__(self, num_states: int, emission_input_dim: int, transition_input_dim: int,
                 weight_scale=0.01, m_step_optimizer=optax.adam(1e-2), m_step_num_iters=50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.emission_input_dim = emission_input_dim
        self.transition_input_dim = transition_input_dim
        self.weight_scale = weight_scale
   
    def initialize(self, key: Optional[Array] = None, method: str = "prior", bias=None, weights=None):
        if key is None:
            key = jr.PRNGKey(0)
        key1, key2 = jr.split(key, 2)
        K, D = self.num_states, self.transition_input_dim

        b = jnp.zeros((K, K), dtype=jnp.float32) if bias is None else jnp.asarray(bias, jnp.float32)
        W = self.weight_scale * jr.normal(key2, (K, K, D), dtype=jnp.float32) if weights is None else jnp.asarray(weights, jnp.float32)

        params = ParamsInputDrivenTransitions(bias=b, weights=W)
        props  = ParamsInputDrivenTransitions(bias=ParameterProperties(), weights=ParameterProperties())
        return params, props

    def log_prior(self, params: ParamsInputDrivenTransitions) -> Scalar:
        return 0.0

    def distribution(self, params, state, inputs=None):
        u = inputs[self.emission_input_dim:self.emission_input_dim + self.transition_input_dim]
        i = jnp.asarray(state, jnp.int32)
        logits = params.bias[i] + jnp.einsum("kd,d->k", params.weights[i], u)
        return tfd.Categorical(logits=logits)
    def _compute_transition_matrices(self, params, inputs):
        """
        Return A[t,i,j] = P(z_{t+1}=j | z_t=i, inputs_t) for t=0..T-2
        Shape: (T-1, K, K)
        """
        K = self.num_states
        T = inputs.shape[0]

        # u_t used for transition t -> t+1, so only t=0..T-2
        u = inputs[:, self.emission_input_dim:self.emission_input_dim + self.transition_input_dim]  # (T-1, D)

        # logits[t, i, j] = bias[i,j] + sum_d W[i,j,d] * u[t,d]
        logits = params.bias[None, :, :] + jnp.einsum("ijd,td->tij", params.weights, u)  # (T-1,K,K)

        return jax.nn.softmax(logits, axis=-1)
    def m_step(self, params, props, batch_stats, m_step_state):
        # batch_stats = (xi, inputs_tr) con batch dimension
        xi_b, inputs_b = batch_stats  # shapes: (B,T-1,K,K), (B,T-1,Dall)

        # agrega batch
        xi = xi_b.sum(axis=0)         # (T-1,K,K)

        # si B>1, podrías concatenar/sumar; lo correcto para inputs es:
        # aquí como inputs es determinista por secuencia, normalmente B=1 en tu uso.
        inputs_tr = inputs_b[0]       # (T-1,Dall)

        # u_t para transiciones
        u = inputs_tr[:, self.emission_input_dim:self.emission_input_dim + self.transition_input_dim]  # (T-1,D)

        b0 = params.bias
        W0 = params.weights

        # hiperparámetros para estabilidad
        l2 = 1e-3
        clip_val = 20.0

        def loss_fn(b, W):
            logits = b[None, :, :] + jnp.einsum("ijd,td->tij", W, u)  # (T-1,K,K)
            logp = jax.nn.log_softmax(logits, axis=-1)                # (T-1,K,K)
            nll = -(xi * logp).sum()
            reg = l2 * (jnp.sum(b*b) + jnp.sum(W*W))
            return nll + reg

        opt = self.m_step_optimizer
        opt_state = opt.init((b0, W0))

        def step(carry, _):
            b, W, opt_state = carry
            (loss, grads) = jax.value_and_grad(loss_fn, argnums=(0,1))(b, W)
            updates, opt_state = opt.update(grads, opt_state, (b, W))
            b, W = optax.apply_updates((b, W), updates)
            b = jnp.clip(b, -clip_val, clip_val)
            W = jnp.clip(W, -clip_val, clip_val)
            return (b, W, opt_state), loss

        (b, W, _), _ = jax.lax.scan(step, (b0, W0, opt_state), None, length=self.m_step_num_iters)

        new_params = ParamsInputDrivenTransitions(bias=b, weights=W)
        return new_params, m_step_state
    
class ParamsSoftmaxGLMHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: Union[ParamsStandardHMMTransitions, ParamsInputDrivenTransitions]
    emissions: ParamsSoftmaxGLMHMMEmissions


class SoftmaxGLMHMM(HMM):
    def __init__(self,
                 num_states: int,
                 num_classes: int,
                 emission_input_dim: int,
                 transition_input_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]] = 1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]] = 1.1,
                 transition_matrix_stickiness: Scalar = 0.0,
                 weight_scale: Scalar = 1.0,
                 m_step_optimizer: optax.GradientTransformation = optax.adam(1e-2),
                 m_step_num_iters: int = 100):

        self.inputs_dim = emission_input_dim + transition_input_dim
        self.num_states = num_states
        self.num_classes = num_classes
        self.emission_input_dim = emission_input_dim
        self.transition_input_dim = transition_input_dim

        initial_component = StandardHMMInitialState( num_states, initial_probs_concentration=initial_probs_concentration)

        if transition_input_dim > 0:
            transition_component = InputDrivenSoftmaxTransitions(
                num_states=num_states, emission_input_dim=emission_input_dim, transition_input_dim=transition_input_dim, weight_scale=weight_scale, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters
            )
        else:
            transition_component = StandardHMMTransitions(
                num_states,
                concentration=transition_matrix_concentration,
                stickiness=transition_matrix_stickiness)

        emission_component = SoftmaxGLMHMMEmissions(
            num_states=num_states,
            num_classes=num_classes,
            input_dim=self.emission_input_dim,
            emission_input_dim = emission_input_dim,
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

        if self.transition_input_dim > 0:
            params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method)
        else:
            params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)

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
        A = self.transition_component._compute_transition_matrices(params.transitions, inputs[:-1]) # (T-1, K, K)

        if A.ndim == 2:  # homogeneous transitions
            A = jnp.broadcast_to(A[None, :, :], (T-1, K, K))

        def emission_probs_t(t):
            return jax.vmap(lambda k: self.emission_component.distribution(params.emissions, k, inputs[t]).probs_parameter())(jnp.arange(K))

        p_y_given_z = jax.vmap(emission_probs_t)(jnp.arange(T))  # (T,K,C)
        pi = params.initial.probs  # (K,)
        pred_z = jnp.vstack([pi, jnp.einsum("tk,tkj->tj", alpha[:-1], A)])

        return jnp.einsum("tk,tkc->tc", pred_z, p_y_given_z)

