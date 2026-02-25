import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jaxtyping import Float, Array
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import NamedTuple, Optional, Tuple, Union

from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.hidden_markov_model.models.abstractions import HMMTransitions
from dynamax.hidden_markov_model.inference import hmm_two_filter_smoother
from dynamax.types import IntScalar, Scalar


class ParamsSoftmaxGLMHMMEmissions(NamedTuple):
    # weights: (K, C-1, M)
    weights: Union[Float[Array, "num_states num_classes_minus1 input_dim"], ParameterProperties]

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
            W = self.weight_scale * jr.normal(subkey, (self.num_states, self.num_classes - 1, self.input_dim))
        else:
            W = emission_weights

        params = ParamsSoftmaxGLMHMMEmissions(weights=W)
        props = ParamsSoftmaxGLMHMMEmissions(weights=ParameterProperties())
        return params, props

    def log_prior(self, params):
        l2 = 1e-4
        return -l2 * jnp.sum(params.weights ** 2)

    def distribution(self, params, state, inputs):
        x = inputs[:self.emission_input_dim]              # (M,)
        eta = params.weights[state] @ x                   # (2,) -> [eta_L, eta_R]

        # logits full: [L, C(base=0), R]
        logits = jnp.array([eta[0], 0.0, eta[1]], dtype=jnp.float32)
        return tfd.Categorical(logits=logits)

    def _compute_conditional_logliks(self, params, emissions, inputs=None):
        """Like the base class but treats ``emission == num_classes`` as a padding sentinel.

        Padded timesteps contribute log-likelihood 0 (uniform emission) so they
        do not update the forward-backward probabilities.  This is the approach
        recommended in dynamax/issues/99: replace the sentinel with a valid label
        first (to avoid distribution errors), compute lls, then zero-mask.

        Valid emissions are 0 … num_classes-1.  The sentinel ``num_classes``
        (= 3 by default) can never appear in real data and does NOT collide with
        the miss/no-response code of -1.
        """
        mask = (emissions == self.num_classes)             # (T,)  True = padding
        safe_emissions = jnp.where(mask, 0, emissions)    # sentinel → 0, safe for Categorical
        f = lambda emission, inpt: jax.vmap(
            lambda state: self.distribution(params, state, inpt).log_prob(emission)
        )(jnp.arange(self.num_states))
        lls = jax.vmap(f)(safe_emissions, inputs)          # (T, K)
        return jnp.where(mask[:, None], 0.0, lls)

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        """Replace the padding sentinel (``num_classes``) with 0 before handing
        emissions to the M-step gradient.  Padded positions are already zeroed
        in ``smoothed_probs`` by ``SoftmaxGLMHMM.e_step``, so they contribute
        nothing to the gradient.

        Real miss-choice emissions (-1) are left untouched here; they are valid
        observed data and must be handled by the model distribution if needed.
        """
        safe_emissions = jnp.where(emissions == self.num_classes, 0, emissions)
        return posterior.smoothed_probs, safe_emissions, inputs
    
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

    def log_prior(self, params):
        l2_bias = 1e-3
        l2_w    = 1e-3
        return -(l2_bias * jnp.sum(params.bias**2) + l2_w * jnp.sum(params.weights**2))

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

        # u_t used for transition t -> t+1, so only t=0..T-2
        u = inputs[:-1, self.emission_input_dim:self.emission_input_dim + self.transition_input_dim]  # (T-1, D)
        # logits[t, i, j] = bias[i,j] + sum_d W[i,j,d] * u[t,d]
        logits = params.bias[None, :, :] + jnp.einsum("ijd,td->tij", params.weights, u)  # (T-1,K,K)
        return jax.nn.softmax(logits, axis=-1)
        
    def collect_suff_stats(self, params, posterior, inputs=None):
        # posterior.trans_probs: (T-1, K, K) o con batch fuera
        # inputs: (T, Dall)
        # usamos inputs_t para transición t->t+1: t=0..T-2
        xi = posterior.trans_probs
        u_inputs = inputs[:-1]  # (T-1, Dall)
        return (xi, u_inputs)

    def m_step(self, params, props, batch_stats, m_step_state):
        xi_b, inputs_b = batch_stats  # xi_b: (B,T-1,K,K), inputs_b: (B,T-1,Dall)

        B, Tm1, K, _ = xi_b.shape
        Dall = inputs_b.shape[-1]

        xi = xi_b.reshape((B * Tm1, K, K))                 # (N,K,K)
        inputs_tr = inputs_b.reshape((B * Tm1, Dall))      # (N,Dall)

        u = inputs_tr[:, self.emission_input_dim:self.emission_input_dim + self.transition_input_dim]  # (N,D)

        b0, W0 = params.bias, params.weights
        l2 = 1e-3
        clip_val = 20.0

        def loss_fn(b, W):
            # logits[n,i,j] = b[i,j] + sum_d W[i,j,d] u[n,d]
            logits = b[None, :, :] + jnp.einsum("ijd,nd->nij", W, u)        # (N,K,K)
            logp   = jax.nn.log_softmax(logits, axis=-1)                   # (N,K,K)
            nll    = -(xi * logp).sum()
            reg    = l2 * (jnp.sum(b*b) + jnp.sum(W*W))
            return nll + reg

        opt = self.m_step_optimizer
        opt_state = opt.init((b0, W0))

        def step(carry, _):
            b, W, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn, argnums=(0,1))(b, W)
            updates, opt_state = opt.update(grads, opt_state, (b, W))
            b, W = optax.apply_updates((b, W), updates)
            b = jnp.clip(b, -clip_val, clip_val)
            W = jnp.clip(W, -clip_val, clip_val)
            return (b, W, opt_state), loss

        (b, W, _), _ = jax.lax.scan(step, (b0, W0, opt_state), None, length=self.m_step_num_iters)
        return ParamsInputDrivenTransitions(bias=b, weights=W), m_step_state
    
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

        # JIT-compile once at construction time so every call reuses the same
        # compiled function object and no retracing occurs across subjects.
        self._e_step_jit   = jit(self.e_step)
        self._m_step_jit   = jit(self.m_step)
        self._smoother_jit = jit(self.smoother)
        self._predict_jit  = jit(self.predict_choice_probs)

        # Batched (vmap) versions used by the multisession helpers.
        # JAX traces lazily on first call; subsequent calls with the same
        # (S, T_max) shape reuse the compiled XLA executable.
        self._batched_e_step_jit   = jit(jax.vmap(self.e_step,               in_axes=(None, 0, 0)))
        self._batched_smoother_jit = jit(jax.vmap(self.smoother,             in_axes=(None, 0, 0)))
        self._batched_predict_jit  = jit(jax.vmap(self.predict_choice_probs, in_axes=(None, 0, 0)))

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
        A = self.transition_component._compute_transition_matrices(params.transitions, inputs) # (T-1, K, K)

        if A.ndim == 2:  # homogeneous transitions
            A = jnp.broadcast_to(A[None, :, :], (T-1, K, K))

        def emission_probs_t(t):
            return jax.vmap(lambda k: self.emission_component.distribution(params.emissions, k, inputs[t]).probs_parameter())(jnp.arange(K))

        p_y_given_z = jax.vmap(emission_probs_t)(jnp.arange(T))  # (T,K,C)
        pi = params.initial.probs  # (K,)
        pred_z = jnp.vstack([pi, jnp.einsum("tk,tkj->tj", alpha[:-1], A)])

        return jnp.einsum("tk,tkc->tc", pred_z, p_y_given_z)

    def e_step(self, params, emissions, inputs=None):
        """E-step with padding-mask support.

        Timesteps where ``emissions[t] == -1`` are treated as padding:

        * ``_compute_conditional_logliks`` (overridden in the emission component)
          sets their log-likelihoods to 0, so the forward-backward pass sees
          a uniform emission at those positions.
        * The posterior sufficient statistics (``smoothed_probs`` and
          ``trans_probs``) at padded positions are zeroed here, so the M-step
          only accumulates real-data information.

        When there is no padding (all ``emissions < num_classes``) the result is
        identical to the parent-class ``e_step``.
        """
        valid = (emissions != self.num_classes)           # (T,)  False = padding
        args  = self._inference_args(params, emissions, inputs)
        posterior = hmm_two_filter_smoother(*args)

        # Zero smoothed marginals at padded steps
        smoothed_probs = jnp.where(valid[:, None], posterior.smoothed_probs, 0.0)

        # Zero pairwise marginals — shape can be (K,K) for time-homogeneous
        # transitions (already summed) or (T-1,K,K) for input-driven ones.
        if posterior.trans_probs.ndim == 2:               # (K, K) — pre-summed
            trans_probs = posterior.trans_probs           # cannot mask per-step
        else:                                             # (T-1, K, K)
            valid_trans = (valid[:-1] & valid[1:])[:, None, None]
            trans_probs = jnp.where(valid_trans, posterior.trans_probs, 0.0)

        masked_post = posterior._replace(
            smoothed_probs=smoothed_probs, trans_probs=trans_probs
        )

        initial_stats    = self.initial_component.collect_suff_stats(params.initial, masked_post, inputs)
        transition_stats = self.transition_component.collect_suff_stats(params.transitions, masked_post, inputs)
        emission_stats   = self.emission_component.collect_suff_stats(params.emissions, masked_post, emissions, inputs)
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik

    # ------------------------------------------------------------------
    # Multi-session helpers
    # ------------------------------------------------------------------

    def _split_by_session(self, emissions, inputs, session_ids, min_length: int = 2):
        """Split (emissions, inputs) into per-session lists, preserving original order.

        Sessions shorter than ``min_length`` are dropped: they cannot produce
        at least one transition step and would cause index errors in the forward
        pass.
        """
        session_ids_np = np.asarray(session_ids)
        _, first_idx    = np.unique(session_ids_np, return_index=True)
        unique_sessions = session_ids_np[np.sort(first_idx)]
        emissions_np = np.asarray(emissions)
        inputs_np    = np.asarray(inputs)
        sessions = []
        for s in unique_sessions:
            mask = session_ids_np == s
            if mask.sum() < min_length:
                continue
            sessions.append((
                jnp.array(emissions_np[mask]),
                jnp.array(inputs_np[mask]),
            ))
        return sessions

    def _pad_sessions(self, sessions):
        """Pad a list of (emissions, inputs) tuples to a common length ``T_max``.

        * Emissions are padded with **``num_classes``** (e.g. ``3`` for a
          3-class model), the sentinel recognised by
          ``_compute_conditional_logliks`` and ``e_step``.  This value can
          never appear in real data (valid codes are ``0 … num_classes-1``)
          and does **not** collide with the miss/no-response code ``-1``.
        * Inputs are padded with **0**.

        Returns
        -------
        e_pad    : jnp.ndarray  shape (S, T_max)       int32
        i_pad    : jnp.ndarray  shape (S, T_max, D)    float32
        lengths  : list[int]    true length of each session
        """
        lengths = [int(e.shape[0]) for e, _ in sessions]
        T_max   = max(lengths)
        S       = len(sessions)
        D       = int(sessions[0][1].shape[-1])

        e_pad = np.full((S, T_max), self.num_classes, dtype=np.int32)
        i_pad = np.zeros((S, T_max, D), dtype=np.float32)
        for idx, (e_s, i_s) in enumerate(sessions):
            T_s = lengths[idx]
            e_pad[idx, :T_s] = np.asarray(e_s)
            i_pad[idx, :T_s] = np.asarray(i_s)
        return jnp.array(e_pad), jnp.array(i_pad), lengths

    def fit_em_multisession(self, params, props, emissions, inputs, session_ids,
                             num_iters=50, verbose=True):
        """Fit the model via EM treating each session as an independent sequence.

        Each session resets to ``pi0`` so there is no probability leak across
        session boundaries.  Sessions are padded to the same length with the
        sentinel ``num_classes`` (e.g. ``3``) and processed in a single ``vmap``
        call per iteration, which is compiled once by XLA and reused across EM steps.

        See dynamax/issues/99 for the masking strategy.
        """
        from tqdm.auto import trange

        sessions             = self._split_by_session(emissions, inputs, session_ids)
        e_pad, i_pad, _      = self._pad_sessions(sessions)
        m_step_state         = self.initialize_m_step_state(params, props)

        log_probs = []
        pbar = trange(num_iters, desc="EM") if verbose else range(num_iters)
        for _ in pbar:
            # Single vmapped E-step over all padded sessions (one XLA call)
            batch_stats, ll_batch = self._batched_e_step_jit(params, e_pad, i_pad)
            total_ll = float(jnp.sum(ll_batch))

            lp = self.log_prior(params) + total_ll
            log_probs.append(lp)
            if verbose:
                pbar.set_postfix({"log prob": f"{lp:.1f}"})

            # M-step: batch_stats has a leading session axis added by vmap;
            # each component's m_step sums over that axis internally.
            params, m_step_state = self._m_step_jit(params, props, batch_stats, m_step_state)

        return params, jnp.array(log_probs)

    def smoother_multisession(self, params, emissions, inputs, session_ids):
        """Run the smoother independently per session and concatenate results.

        Returns
        -------
        smoothed_probs : np.ndarray  shape (T_total, K)
        """
        sessions             = self._split_by_session(emissions, inputs, session_ids)
        e_pad, i_pad, lengths = self._pad_sessions(sessions)
        posterior_batch      = self._batched_smoother_jit(params, e_pad, i_pad)
        smoothed = np.asarray(posterior_batch.smoothed_probs)  # (S, T_max, K)
        return np.concatenate(
            [smoothed[i, :T_s] for i, T_s in enumerate(lengths)], axis=0
        )

    def predict_choice_probs_multisession(self, params, emissions, inputs, session_ids):
        """Compute one-step-ahead predictive choice probabilities per session.

        Returns
        -------
        choice_probs : np.ndarray  shape (T_total, C)
        """
        sessions              = self._split_by_session(emissions, inputs, session_ids)
        e_pad, i_pad, lengths = self._pad_sessions(sessions)
        probs_batch = np.asarray(self._batched_predict_jit(params, e_pad, i_pad))  # (S, T_max, C)
        return np.concatenate(
            [probs_batch[i, :T_s] for i, T_s in enumerate(lengths)], axis=0
        )

