import enum
import functools
from typing import Any, Callable

import jax
import jax.flatten_util
import jax.numpy as jnp

from jaxsim import typing as jtp
from jaxsim.physics.model.physics_model import PhysicsModel
from jaxsim.simulation.integrators import Time, TimeHorizon, TimeStep
from jaxsim.simulation.ode_data import ODEState
from jaxsim.sixd import so3

RTOL_DEFAULT = 1.4e-8
ATOL_DEFAULT = 1.4e-8

SAFETY_DEFAULT = 0.9
BETA_MIN_DEFAULT = 1.0 / 10
BETA_MAX_DEFAULT = 2.5
MAX_STEP_REJECTIONS_DEFAULT = 5

# Contrarily to the fixed-step integrators that operate on generic PyTrees,
# these variable-step integrators operate only on arrays (that could be the
# flatted PyTree).
State = jtp.Vector
StateNext = State
StateDerivative = jtp.Vector
StateDerivativeCallable = Callable[
    [State, Time], tuple[StateDerivative, dict[str, Any]]
]

# TODO: rename
SystemDynamics = StateDerivativeCallable
SystemDynamicsFlat = Callable[
    [jtp.VectorLike, Time], tuple[StateDerivative, dict[str, Any]]
]


class AdaptiveIntegratorType(enum.IntEnum):
    HeunEuler = enum.auto()
    BogackiShampine = enum.auto()
    SymplecticYoshida23 = enum.auto()
    SymplecticYoshida34 = enum.auto()


class VariableStepIntegratorFactory:
    @staticmethod
    def get(integrator_type: AdaptiveIntegratorType) -> tuple[Callable, int, int, bool]:
        """"""

        match integrator_type:
            case AdaptiveIntegratorType.HeunEuler:
                p = int(2)
                p̂ = int(p - 1)
                return heun_euler, p, p̂, True

            case AdaptiveIntegratorType.BogackiShampine:
                p = int(3)
                p̂ = int(p - 1)
                return bogacki_shampine, p, p̂, True

            case AdaptiveIntegratorType.SymplecticYoshida23:
                p = int(3)
                p̂ = int(2)
                symplectic_yoshida = functools.partial(
                    wrap_yoshida_embedded, symplectic_yoshida_xy=symplectic_yoshida_23
                )
                return symplectic_yoshida, p, p̂, False

            case AdaptiveIntegratorType.SymplecticYoshida34:
                p = int(4)
                p̂ = int(3)
                symplectic_yoshida = functools.partial(
                    wrap_yoshida_embedded, symplectic_yoshida_xy=symplectic_yoshida_34
                )
                return symplectic_yoshida, p, p̂, False

            case _:
                raise ValueError(integrator_type)


# =================
# Utility functions
# =================


# State = TypeVar("State")
# StateDerivative = State
# SystemDynamics = Callable[[State, Time], tuple[StateDerivative, dict[str, Any]]]
#
#
# from typing import get_args
#
#
# def f(x: jax.Array, t: Time, test: bool) -> jax.Array:
#     return x
#
#
# def wrap(fn: SystemDynamics) -> SystemDynamics:
#     return fn
#
#
# o, _ = wrap(f)(jnp.zeros(3), 1.0)


def flatten_system_dynamics(
    original_system_dynamics: SystemDynamics,
    original_state: State,
) -> SystemDynamicsFlat:
    """"""

    _, unflatten_fn = jax.flatten_util.ravel_pytree(pytree=original_state)

    def flat_system_dynamics(
        x: jtp.VectorLike, t: Time
    ) -> tuple[StateDerivative, dict[str, Any]]:

        ẋ, aux_dict = original_system_dynamics(unflatten_fn(x), t)
        return jax.flatten_util.ravel_pytree(pytree=ẋ)[0], aux_dict

    return flat_system_dynamics


@functools.partial(jax.jit, static_argnames=["f"])
def estimate_step_size(
    x0: State,
    t0: Time,
    f: StateDerivativeCallable,
    order: jtp.IntLike,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
) -> tuple[jtp.Float, StateDerivative]:
    """
    Compute the initial step size to warm-start an adaptive integrator.

    Args:
        x0: The initial state.
        t0: The initial time.
        f: The state derivative function $f(x, t)$.
        order: The order $p$ of an integrator with truncation error $\mathcal{O}(\Delta t^{p+1})$.
        rtol: The relative tolerance to scale the state.
        atol: The absolute tolerance to scale the state.

    Returns:
        A tuple containing the computed initial step size
        and the state derivative $\dot{x} = f(x_0, t_0)$.

    Note:
        Refer to the following reference for the implementation details:

        Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
        E. Hairer, S. P. Norsett G. Wanner.
    """

    # Compute the state derivative at the initial state.
    ẋ0 = f(x0, t0)[0]

    # Scale the initial state and its derivative.
    scale0 = atol + jnp.abs(x0) * rtol
    scale1 = atol + jnp.abs(ẋ0) * rtol
    d0 = jnp.linalg.norm(jnp.abs(x0) / scale0, ord=jnp.inf)  # noqa
    d1 = jnp.linalg.norm(jnp.abs(ẋ0) / scale1, ord=jnp.inf)  # noqa

    # Compute the first guess of the initial step size.
    h0 = jnp.where(jnp.minimum(d0, d1) <= 1e-5, 1e-6, 0.01 * d0 / d1)

    # Compute the next state and its derivative.
    x1 = x0 + h0 * ẋ0
    ẋ1 = f(x1, t0 + h0)[0]

    # Scale the difference of the state derivatives.
    scale2 = atol + jnp.maximum(jnp.abs(ẋ1), jnp.abs(ẋ0)) * rtol
    d2 = jnp.linalg.norm(jnp.abs((ẋ1 - ẋ0) / scale2), ord=jnp.inf) / h0  # noqa

    # Compute the second guess of the initial step size.
    h1 = jnp.where(
        jnp.maximum(d1, d2) <= 1e-15,
        jnp.maximum(1e-6, h0 * 1e-3),
        (0.01 / jnp.maximum(d1, d2)) ** (1.0 / (order + 1.0)),
    )

    # Propose the final guess of the initial step size.
    # Also return the state derivative computed at the initial state since
    # it is likely a quantity that needs to be computed again later.
    return jnp.array(jnp.minimum(100.0 * h0, h1), dtype=float), ẋ0


# TODO: decimal -> max_precision?
# def round_to_decimal(x: jax.Array, decimal: float = 1e-8) -> jax.Array:
#     """"""
#     return jnp.array(x / decimal, dtype=int).astype(dtype=float) * decimal


def scale_array(
    x1: State,
    x2: State | StateNext | None = None,
    rtol: jax.typing.ArrayLike = RTOL_DEFAULT,
    atol: jax.typing.ArrayLike = ATOL_DEFAULT,
) -> jax.Array:
    """
    Compute the component-wise state scale to use for the error estimate of
    the local integration error.

    Args:
        x1: The first state, usually $x(t_0)$.
        x2: The optional second state, usually $x(t_f)$.
        rtol: The relative tolerance to scale the state.
        atol: The absolute tolerance to scale the state.

    Returns:
        The component-wise state scale to use for the error estimate of
        the local integration error.
    """

    # Use a zeroed second state if not provided.
    x2 = x2 if x2 is not None else jnp.zeros_like(x1)

    # Return: atol + max(|x1|, |x2|) * rtol.
    return (
        atol
        + jnp.vstack(
            [
                jnp.abs(jnp.atleast_1d(x1.squeeze())),
                jnp.abs(jnp.atleast_1d(x2.squeeze())),
            ]
        ).max(axis=0)
        * rtol
    )


def error_local(
    x0: State,
    xf: StateNext,
    error_estimate: State | None = None,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
    norm_ord: jtp.IntLike | jtp.FloatLike = jnp.inf,
) -> jtp.Float:
    """
    Compute the local integration error.

    Args:
        x0: The initial state $x(t_0)$.
        xf: The final state $x(t_f)$.
        error_estimate: The optional error estimate. In not given, it is computed as the
            absolute value of the difference between the final and initial states.
        rtol: The relative tolerance to scale the state.
        atol: The absolute tolerance to scale the state.
        norm_ord: The norm to use to compute the error. Default is the infinity norm.

    Returns:
        The local integration error.
    """

    # First compute the component-wise scale using the initial and final states.
    sc = scale_array(x1=x0, x2=xf, rtol=rtol, atol=atol)

    # Compute the error estimate if not given.
    error_estimate = error_estimate if error_estimate is not None else jnp.abs(xf - x0)

    # Then, compute the local error by properly scaling the given error estimate and apply
    # the desired norm (default is infinity norm, that is the maximum absolute value).
    return jnp.linalg.norm(error_estimate / sc, ord=norm_ord)


@functools.partial(jax.jit, static_argnames=["f"])
def runge_kutta_from_butcher_tableau(
    x0: State,
    t0: Time,
    dt: TimeStep,
    f: StateDerivativeCallable,
    *,
    c: jax.Array,
    b: jax.Array,
    A: jax.Array,
    dxdt0: StateDerivative | None = None,
) -> tuple[State, StateDerivative, State, dict[str, Any]]:
    """
    Advance a state vector by integrating a system dynamics with a Runge-Kutta integrator.

    Args:
        x0: The initial state.
        t0: The initial time.
        dt: The integration time step.
        f: The state derivative function :math:`f(x, t)`.
        c: The :math:`\mathbf{c}` parameter of the Butcher tableau.
        b: The :math:`\mathbf{b}` parameter of the Butcher tableau.
        A: The :math:`\mathbf{A}` parameter of the Butcher tableau.
        dxdt0: The optional pre-computed state derivative at the
            initial :math:`(x_0, t_0)`, useful for FSAL schemes.

    Returns:
        A tuple containing the next state, the intermediate state derivatives
        :math:`\mathbf{k}_i`, the component-wise error estimate, and the auxiliary
        dictionary returned by `f`.

    Note:
        If `b.T` has multiple rows (used e.g. in embedded Runge-Kutta methods), the first
        returned argument is a 2D array having as many rows as `b.T`. Each i-th row
        corresponds to the solution computed with coefficients of the i-th row of `b.T`.
    """

    # Adjust sizes of Butcher tableau arrays.
    c = jnp.atleast_1d(c.squeeze())
    b = jnp.atleast_2d(b.squeeze())
    A = jnp.atleast_2d(A.squeeze())

    # assert A.shape == (c.size, b.T.shape[1])
    # h = tf - t0
    # assert h > 0

    # Use a symbol for the time step.
    Δt = dt

    # Initialize the carry of the for loop with the stacked kᵢ vectors.
    carry0 = jnp.zeros(shape=(c.size, x0.size), dtype=float)

    # Allow FSAL (first-same-as-last) property by passing ẋ0 = f(x0, t0) from
    # the previous iteration.
    get_ẋ0 = lambda: dxdt0 if dxdt0 is not None else f(x0, t0)[0]

    # We use a `jax.lax.scan` to have only a single instance of the compiled `f` function.
    # Otherwise, if we compute e.g. for RK4 sequentially, the jit-compiled code
    # would include 4 repetitions of the `f` logic, making everything extremely slow.
    def scan_body(carry: jax.Array, i: int | jax.Array) -> tuple[Any, None]:
        """"""

        # Unpack the carry
        k = carry

        def compute_ki():
            xi = x0 + Δt * jnp.dot(A[i, :], k)
            ti = t0 + c[i] * Δt
            return f(xi, ti)[0]

        # This selector enables FSAL property in the first iteration (i=0).
        ki = jax.lax.select(
            pred=(i == 0),
            on_true=get_ẋ0(),
            on_false=compute_ki(),
        )

        k = k.at[i].set(ki)
        return k, None

    # Compute the state derivatives k
    k, _ = jax.lax.scan(
        f=scan_body,
        init=carry0,
        xs=jnp.arange(c.size),
    )

    # Compute the output state.
    # Note that z contains as many new states as the rows of `b.T`.
    z = x0 + Δt * jnp.dot(b.T, k)

    # Compute the error estimate if `b.T` has multiple rows, otherwise return 0.
    error_estimate = jax.lax.select(
        pred=b.T.shape[0] == 1,
        on_true=jnp.zeros_like(x0, dtype=float),
        on_false=dt * jnp.dot(b.T[-1] - b.T[0], k),
    )

    # TODO: populate the auxiliary dictionary
    return z, k, error_estimate, dict()


PositionSymplectic = jax.Array
PositionSymplecticNext = PositionSymplectic
VelocitySymplectic = jax.Array
VelocitySymplecticNext = VelocitySymplectic
AccelerationSymplectic = jax.Array
AccelerationSymplecticCallable = Callable[
    [PositionSymplectic, VelocitySymplectic, Time], AccelerationSymplectic
]
VelocitySymplecticCallable = Callable[
    [PositionSymplectic, VelocitySymplectic, Time], VelocitySymplectic
]

# v̇ = f(x, v, t)
# ẋ = g(x, v, t)
#
# x_kk = x_k + g(x_k, v_k, t_k) * Δtx
# v̂_kk = v_k + f(x_k, v_k, t_k) * Δtx  # TODO
# v_kk = v_k + f(x_kk, v̂_kk, t_kk) * Δtv

# a = (v̇_WB, ω̇_WB, s̈, ṁ)
# v = (v_WB, ω_WB, ṡ, m)
# x = (W_p_B, W_Q_B, s)


def symplectic_from_yoshida_parameters(
    x0: PositionSymplectic,
    v0: VelocitySymplectic,
    t0: Time,
    dt: TimeStep,
    f: AccelerationSymplecticCallable,
    g: VelocitySymplecticCallable,
    *,
    c: jax.Array,
    d: jax.Array,
    tf_next_position: Callable[
        [
            PositionSymplectic,
            VelocitySymplectic,
            PositionSymplecticNext,
            Time,
            TimeStep,
        ],
        PositionSymplecticNext,
    ] = lambda x0, v0, xf, t0, dt: xf,
    tf_next_velocity: Callable[
        [
            PositionSymplectic,
            VelocitySymplectic,
            VelocitySymplecticNext,
            Time,
            TimeStep,
        ],
        VelocitySymplecticNext,
    ] = lambda x0, v0, vf, t0, dt: vf,
    # TODO FSAL with f0? if c0=0?
) -> tuple[tuple[jax.Array, jax.Array], dict[str, Any]]:
    """"""

    # Adjust sizes of Yoshida parameters.
    c = jnp.atleast_1d(c.squeeze())
    d = jnp.atleast_1d(d.squeeze())

    x0 = jnp.atleast_1d(x0.squeeze())
    v0 = jnp.atleast_1d(v0.squeeze())

    Carry = tuple[jax.Array, jax.Array]
    carry0: Carry = (x0, v0)

    # Initialize the carry of the for loop with the stacked kᵢ vectors.
    # carry0 = jnp.zeros(shape=(c.size, x0.size), dtype=float)

    # Allow FSAL (first-same-as-last) property by passing f0 = f(x0, t0) from
    # the previous iteration.
    # get_ẋ0 = lambda: f0 if f0 is not None else f(x0, t0)[0]

    # We use a `jax.lax.scan` to have only a single instance of the compiled `f` function.
    # Otherwise, if we compute e.g. for RK4 sequentially, the jit-compiled code
    # would include 4 repetitions of the `f` logic, making everything extremely slow.
    def scan_body(carry: Carry, i: int | jax.Array) -> tuple[Carry, None]:
        """"""

        x_i0, v_i0 = carry

        Δtx = c[i] * dt
        x_i = x_i0 + Δtx * g(x_i0, v_i0, t0)
        x_i = tf_next_position(x_i0, v_i0, x_i, t0, Δtx)

        # TODO: Δtx or Δtv??
        # v̂_ii = v_i + Δtv * f(x_ii, v_i, t0 + Δtx)  # what's the best predictor?

        Δtv = d[i] * dt
        v̂_i = v_i0 + Δtx * f(x_i0, v_i0, t0)  # TODO: this can be cached?
        v̂_i = tf_next_velocity(x_i0, v_i0, v̂_i, t0, Δtx)

        v_i = v_i0 + Δtv * f(x_i, v̂_i, t0 + Δtx)
        v_i = tf_next_velocity(x_i0, v_i0, v_i, t0, Δtv)

        return (x_i, v_i), None

    # Compute the state derivatives k
    (xf, vf), _ = jax.lax.scan(
        f=scan_body,
        init=carry0,
        xs=jnp.arange(c.size),
    )

    # TODO: which aux dict from f?
    return (xf, vf), dict()

    # # Compute the output state and the error estimate.
    # # Note that z contains as many new states as the rows of `b.T`.
    # z = x0 + h * jnp.dot(b.T, k)
    # error_estimate = dt * jnp.dot(b.T[-1] - b.T[0], k)
    #
    # return z, k, error_estimate, dict()


@functools.partial(jax.jit, static_argnames=["f"])
# TODO: f takes flat but x0 is ODEState.
#   -> maybe use generic types?
def odeint_symplectic_fixed_one_step(
    f: StateDerivativeCallable,
    x0: ODEState,
    t0: Time,
    tf: Time,  # TODO tf or dt?
    *,
    # physics_model: PhysicsModel,
    c: jtp.VectorLike = jnp.array([0, 1], dtype=float),
    d: jtp.VectorLike = jnp.array([1 / 2, 1 / 2], dtype=float),
) -> tuple[ODEState, dict[str, Any]]:
    """"""

    # dofs = physics_model.dofs()
    dofs = x0.physics_model.joint_positions.size

    # Define the functions to flatten and unflatten ODEState objects.
    flatten_fn = lambda ode_state: x0.flatten_fn()(ode_state)
    unflatten_fn = lambda x: x0.unflatten_fn()(x)

    v0_symplectic = jnp.hstack(
        [
            x0.physics_model.base_linear_velocity,
            x0.physics_model.base_angular_velocity,
            x0.physics_model.joint_velocities,
            jnp.hstack(x0.soft_contacts.tangential_deformation.T),
        ]
    )

    x0_symplectic = jnp.hstack(
        [
            x0.physics_model.base_position,
            x0.physics_model.base_quaternion,
            x0.physics_model.joint_positions,
            # TODO: adjust m/m_dot with (q_old, v_new)?
            # jnp.hstack(x0.soft_contacts.tangential_deformation.T),
        ]
    )

    from jaxsim.simulation.ode_data import PhysicsModelState, SoftContactsState

    def xv_to_ode_state(x: PositionSymplectic, v: VelocitySymplectic) -> ODEState:
        return ODEState.build(
            physics_model_state=PhysicsModelState.build(
                base_position=x[:3],
                base_quaternion=x[3:7],
                joint_positions=x[7 : 7 + dofs],
                base_linear_velocity=v[:3],
                base_angular_velocity=v[3:6],
                joint_velocities=v[6 : 6 + dofs],
            ),
            soft_contacts_state=SoftContactsState.build(
                tangential_deformation=jnp.reshape(v[6 + dofs :], (-1, 3)).T
            ),
        )

    # TODO try here only ABA with contact forces at t0 and no m
    #    Then calculate m_dot(q0, nu_new) and integrate m
    def f_symplectic(
        x: PositionSymplectic, v: VelocitySymplectic, t: Time
    ) -> AccelerationSymplectic:
        """"""

        ode_state = xv_to_ode_state(x=x, v=v)
        ode_state_dot = unflatten_fn(f(flatten_fn(ode_state), t)[0])

        return jnp.hstack(
            [
                ode_state_dot.physics_model.base_linear_velocity,
                ode_state_dot.physics_model.base_angular_velocity,
                ode_state_dot.physics_model.joint_velocities,
                # ode_state_dot.soft_contacts.tangential_deformation,
                jnp.hstack(ode_state_dot.soft_contacts.tangential_deformation.T),
            ]
        )

    def g_symplectic(
        x: PositionSymplectic, v: VelocitySymplectic, t: Time
    ) -> VelocitySymplectic:
        """"""

        from jaxsim.math.quaternion import Quaternion
        from jaxsim.sixd import se3

        ode_state = xv_to_ode_state(x=x, v=v)

        W_p_B = ode_state.physics_model.base_position
        W_Q_B = ode_state.physics_model.base_quaternion

        W_vl_WB = ode_state.physics_model.base_linear_velocity
        W_ω_WB = ode_state.physics_model.base_angular_velocity
        W_v_WB = jnp.hstack([W_vl_WB, W_ω_WB])

        W_H_BW = jnp.vstack(
            [
                jnp.block([jnp.eye(3), jnp.vstack(W_p_B)]),
                jnp.array([0, 0, 0, 1]),
            ]
        )

        BW_Xv_W = se3.SE3.from_matrix(W_H_BW).inverse().adjoint()
        BW_vl_WB = (BW_Xv_W @ W_v_WB)[0:3]

        return jnp.hstack(
            [
                BW_vl_WB,
                Quaternion.derivative(
                    quaternion=W_Q_B, omega=W_ω_WB, omega_in_body_fixed=False
                ).squeeze(),
                ode_state.physics_model.joint_velocities,
            ]
        )

    def tf_next_position(
        x0: PositionSymplectic,
        v0: VelocitySymplectic,
        xf: PositionSymplecticNext,
        t0: Time,
        dt: TimeStep,
    ) -> PositionSymplecticNext:
        """"""

        ode_state_t0 = xv_to_ode_state(x=x0, v=v0)

        # Indices to convert quaternions between serializations
        to_xyzw = jnp.array([1, 2, 3, 0])
        to_wxyz = jnp.array([3, 0, 1, 2])

        # Get the initial quaternion and the implicitly integrated angular velocity
        # TODO: instead of explicit, to trapezoidal with estimate with FE?
        W_ω_WB_t0 = ode_state_t0.physics_model.base_angular_velocity
        W_Q_B_t0 = so3.SO3.from_quaternion_xyzw(
            ode_state_t0.physics_model.base_quaternion[to_xyzw]
        )

        # Integrate the quaternion on its manifold using the implicit (TODO) angular velocity,
        # transformed in body-fixed representation since jaxlie uses this convention
        B_R_W = W_Q_B_t0.inverse().as_matrix()
        W_Q_B_tf = W_Q_B_t0 @ so3.SO3.exp(tangent=dt * B_R_W @ W_ω_WB_t0)
        # W_Q_B_tf = W_Q_B_t0 @ so3.SO3.exp(tangent=dt * B_R_W @ W_ω_WB_tf)

        # Store the quaternion in the final state
        xf = xf.at[3:7].set(W_Q_B_tf.as_quaternion_xyzw()[to_wxyz])
        return xf

    (xf, vf), aux_dict = symplectic_from_yoshida_parameters(
        x0=x0_symplectic,
        v0=v0_symplectic,
        t0=t0,
        dt=tf - t0,
        f=f_symplectic,
        g=g_symplectic,
        # c=jnp.array([0, 1], dtype=float),
        # d=jnp.array([1 / 2, 1 / 2], dtype=float),
        c=c,
        d=d,
        tf_next_position=tf_next_position,
        # tf_next_velocity=None,
    )

    return xv_to_ode_state(x=xf, v=vf), aux_dict


@functools.partial(jax.jit, static_argnames=["f"])
def wrap_odeint_symplectic_fixed_one_step(
    f: StateDerivativeCallable,
    x0: State,
    t0: Time,
    tf: Time,
    *,
    physics_model: PhysicsModel,
    c: jtp.VectorLike,
    d: jtp.VectorLike,
) -> tuple[State, dict[str, Any]]:
    """"""

    ode_state_t0 = ODEState.deserialize(data=x0, physics_model=physics_model)

    ode_state_tf, aux_dict = odeint_symplectic_fixed_one_step(
        f=f,
        x0=ode_state_t0,
        t0=t0,
        tf=tf,
        # physics_model=physics_model,
        c=c,
        d=d,
    )

    return ode_state_tf.flatten(), aux_dict


@functools.partial(jax.jit, static_argnames=["f"])
def symplectic_yoshida_23(
    x0: ODEState,
    t0: Time,
    dt: TimeStep,
    f: StateDerivativeCallable,
    *,
    physics_model: PhysicsModel,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    aux_dict: dict[str, Any] | None = None,
) -> tuple[ODEState, jtp.Float, dict[str, Any]]:
    """"""

    cc = jnp.vstack(
        [
            jnp.array([0, 1, 0], dtype=float),
            jnp.array([1, -2 / 3, 2 / 3], dtype=float),
        ]
    )

    dd = jnp.vstack(
        [
            jnp.array([1 / 2, 1 / 2, 0], dtype=float),
            jnp.array([-1 / 24, 3 / 4, 7 / 24], dtype=float),
        ]
    )

    symplectic_closed = lambda c, d: odeint_symplectic_fixed_one_step(
        f=f,
        x0=x0,
        t0=t0,
        tf=t0 + dt,
        # physics_model=physics_model,
    )

    Xf, aux_dict = jax.vmap(symplectic_closed)(cc, dd)

    xf_high = jax.tree_map(lambda l: l[1], Xf).flatten()

    xf_high_flat = xf_high.flatten()
    xf_low_flat = jax.tree_map(lambda l: l[0], Xf).flatten()

    error = error_local(
        x0=x0.flatten(),
        xf=xf_high_flat,
        error_estimate=jnp.abs(xf_high_flat - xf_low_flat),
        rtol=rtol,
        atol=atol,
    )

    return xf_high, error, jax.tree_map(lambda l: l[1], aux_dict)


@functools.partial(jax.jit, static_argnames=["f"])
def symplectic_yoshida_34(
    x0: ODEState,
    t0: Time,
    dt: TimeStep,
    f: StateDerivativeCallable,
    *,
    # physics_model: PhysicsModel,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    aux_dict: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[ODEState, jtp.Float, dict[str, Any]]:
    """"""

    cub_root_two = jnp.power(2, 1 / 3)
    w0 = -cub_root_two / (2 - cub_root_two)
    w1 = 1 / (2 - cub_root_two)
    c1 = c4 = 0.5 * w1
    c2 = c3 = 0.5 * (w0 + w1)
    d1 = d3 = w1
    d2 = w0
    d4 = 0

    cc = jnp.vstack(
        [
            jnp.array([1, -2 / 3, 2 / 3, 0], dtype=float),
            jnp.array([c1, c2, c3, c4], dtype=float),
        ]
    )

    dd = jnp.vstack(
        [
            jnp.array([-1 / 24, 3 / 4, 7 / 24, 0], dtype=float),
            jnp.array([d1, d2, d3, d4], dtype=float),
        ]
    )

    symplectic_closed = lambda c, d: odeint_symplectic_fixed_one_step(
        f=f,
        x0=x0,
        t0=t0,
        tf=t0 + dt,
        # physics_model=physics_model,
    )

    Xf, aux_dict = jax.vmap(symplectic_closed)(cc, dd)

    x_high = jax.tree_map(lambda l: l[1], Xf).flatten()

    x_high_flat = x_high.flatten()
    x_low_flat = jax.tree_map(lambda l: l[0], Xf).flatten()

    error = error_local(
        x0=x0.flatten(),
        xf=x_high_flat,
        error_estimate=jnp.abs(x_high_flat - x_low_flat),
        rtol=rtol,
        atol=atol,
    )

    return x_high, error, jax.tree_map(lambda l: l[1], aux_dict)


@functools.partial(jax.jit, static_argnames=["f", "symplectic_yoshida_xy"])
def wrap_yoshida_embedded(
    x0: State,
    t0: Time,
    dt: TimeStep,
    f: StateDerivativeCallable,
    symplectic_yoshida_xy: Callable,
    *,
    physics_model: PhysicsModel,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    aux_dict: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[State, jtp.Float, dict[str, Any]]:
    """"""

    ode_state_t0 = ODEState.deserialize(data=x0, physics_model=physics_model)

    ode_state_tf, error, aux_dict = symplectic_yoshida_xy(
        x0=ode_state_t0,
        t0=t0,
        dt=dt,
        f=f,
        # physics_model=physics_model,
        rtol=rtol,
        atol=atol,
        aux_dict=aux_dict,
        **kwargs,
    )

    return ode_state_tf.flatten(), error, aux_dict


# ================================
# Embedded Runge-Kutta integrators
# ================================


@functools.partial(jax.jit, static_argnames=["f", "tf_next_state"])
def heun_euler(
    x0: State,
    t0: Time,
    dt: TimeStep,
    f: StateDerivativeCallable,
    *,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    aux_dict: dict[str, Any] | None = None,
    tf_next_state: Callable[
        [State, StateNext, Time, TimeStep], StateNext
    ] = lambda x0, xf, t0, dt: xf,
    **kwargs,
) -> tuple[State, jtp.Float, dict[str, Any]]:
    """"""

    # b parameter of Butcher tableau.
    b = jnp.array(
        [
            [1 / 2, 1 / 2],
            [1, 0],
        ]
    ).T

    # c parameter of Butcher tableau.
    c = jnp.array([0, 1])

    # A parameter of Butcher tableau.
    A = jnp.array(
        [
            [0, 0],
            [1, 0],
        ]
    )

    # Integrate the state with the resulting integrator.
    (
        (xf_higher, xf_lower),
        (_, k2),
        error_estimate,
        aux_dict,
    ) = runge_kutta_from_butcher_tableau(
        x0=x0,
        t0=t0,
        dt=dt,
        f=f,
        c=c,
        b=b,
        A=A,
        f0=aux_dict.get("f0", None) if aux_dict is not None else None,
    )

    # Take the higher-order solution as the next state, and optionally apply
    # the user-defined transformation.
    x_next = tf_next_state(x0, xf_higher, t0, dt)

    # Define the order of the integrators used to compute each row of z
    # (corresponding to the rows of b.T).
    # z_order = jnp.array([2, 1], dtype=int)

    # Calculate the local integration error.
    error = error_local(
        x0=x0, xf=x_next, error_estimate=error_estimate, rtol=rtol, atol=atol
    )

    # Enable FSAL (first-same-as-last) property by returning k2.
    aux_dict = dict(f0=k2)

    # TODO: is z_order necessary? Would be much more simple if we switch to classes instead of fn!
    return x_next, error, aux_dict
    # return x_next, z_order, error, aux_dict


@functools.partial(jax.jit, static_argnames=["f", "tf_next_state"])
def bogacki_shampine(
    x0: State,
    t0: Time,
    dt: TimeStep,
    f: StateDerivativeCallable,
    *,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    aux_dict: dict[str, Any] | None = None,
    tf_next_state: Callable[
        [State, StateNext, Time, TimeStep], StateNext
    ] = lambda x0, xf, t0, dt: xf,
    **kwargs,
    # ) -> tuple[State, StateDerivative, jtp.Float, dict[str, Any]]:
) -> tuple[State, jtp.Float, dict[str, Any]]:
    """"""

    # b parameter of Butcher tableau.
    b = jnp.array(
        [
            [2 / 9, 1 / 3, 4 / 9, 0],
            [7 / 24, 1 / 4, 1 / 3, 1 / 8],
        ]
    ).T

    # c parameter of Butcher tableau.
    c = jnp.array([0, 1 / 2, 3 / 4, 1])

    # A parameter of Butcher tableau.
    A = jnp.array(
        [
            [0, 0, 0, 0],
            [1 / 2, 0, 0, 0],
            [0, 3 / 4, 0, 0],
            [2 / 9, 1 / 3, 4 / 9, 0],
        ]
    )

    # Integrate the state with the resulting integrator.
    (
        (xf_higher, xf_lower),
        (_, _, _, k4),
        error_estimate,
        aux_dict,
    ) = runge_kutta_from_butcher_tableau(
        x0=x0,
        t0=t0,
        dt=dt,
        f=f,
        c=c,
        b=b,
        A=A,
        dxdt0=aux_dict.get("f0", None) if aux_dict is not None else None,
    )

    # Take the higher-order solution as the next state, and optionally apply the
    # user-defined transformation.
    x_next = tf_next_state(x0, xf_higher, t0, dt)

    # Define the order of the integrators used to compute each row of z
    # (corresponding to the rows of b.T).
    # z_order = jnp.array([3, 2], dtype=int)

    # Calculate the local integration error.
    error = error_local(
        x0=x0, xf=x_next, error_estimate=error_estimate, rtol=rtol, atol=atol
    )

    # Enable FSAL (first-same-as-last) property by returning k4.
    aux_dict = dict(f0=k4)

    # return x_next, z_order, error, aux_dict
    return x_next, error, aux_dict


# ==========================================
# Variable-step RK integrators (single step)
# ==========================================


# TODO: create a ODESolution dataclass to simplify extraction of data / debug buffers


@functools.partial(
    jax.jit,
    static_argnames=["f", "integrator_type", "debug_buffers_size", "tf_next_state"],
)
def odeint_embedded_rk_one_step(
    f: StateDerivativeCallable,
    x0: State,
    t0: Time,
    tf: Time,
    *,
    integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.BogackiShampine,
    # integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.HeunEuler,
    dt0: jtp.FloatLike | None = None,
    # dt0: jtp.FloatLike = 0.0,
    dt_min: jtp.FloatLike = -jnp.inf,
    dt_max: jtp.FloatLike = jnp.inf,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
    safety: jtp.FloatLike = SAFETY_DEFAULT,
    beta_min: jtp.FloatLike = BETA_MIN_DEFAULT,
    beta_max: jtp.FloatLike = BETA_MAX_DEFAULT,
    debug_buffers_size: int | None = None,
    max_step_rejections: jtp.IntLike = MAX_STEP_REJECTIONS_DEFAULT,
    tf_next_state: Callable[
        [State, StateNext, Time, TimeStep], StateNext
    ] = lambda x0, xf, t0, dt: xf,
    **kwargs,
) -> tuple[State, dict[str, Any]]:
    """"""

    # Get the integrator and its order
    rk_method, p, p̂, fsal = VariableStepIntegratorFactory.get(
        integrator_type=integrator_type
    )
    q = jnp.minimum(p, p̂)

    # Close the integrator over its optional arguments
    rk_method_closed = lambda x0, t0, Δt, aux_dict: rk_method(
        x0=x0,
        t0=t0,
        dt=Δt,
        f=f,
        rtol=rtol,
        atol=atol,
        aux_dict=aux_dict,
        tf_next_state=tf_next_state,  # TODO: disable for yoshida?
        **kwargs,
    )

    # Compute the initial step size considering the order of the integrator,
    # and clip it to the given bounds, if necessary.
    # The function also returns the evaluation of the state derivative at the
    # initial state, saving a call to the f function.
    Δt0, ẋ0 = jax.lax.cond(
        # TODO: maybe None is not working here, only 0.0
        pred=jnp.where(dt0 is None, 0.0, dt0) == 0.0,
        true_fun=lambda _: estimate_step_size(
            x0=x0, t0=t0, f=f, order=p, atol=atol, rtol=rtol
        ),
        false_fun=lambda _: (dt0, f(x0, t0)[0]),
        operand=None,
    )

    # Clip the initial step size to the given bounds, if necessary.
    Δt0 = jnp.clip(
        a=Δt0,
        a_min=jnp.minimum(dt_min, tf - t0),
        a_max=jnp.minimum(dt_max, tf - t0),
    )

    # Round dt0 such that the halved value is not less then nanoseconds.
    # TODO: remove this for dt computed from error
    # dt0 = round_to_decimal(x=dt0, decimal=1e-8)

    # Initialize the size of the debug buffers.
    debug_buffers_size = debug_buffers_size if debug_buffers_size is not None else 0

    # Allocate the debug buffers.
    debug_buffers = (
        dict(
            idx=jnp.array(0, dtype=int),
            x_steps=-jnp.inf
            * jnp.ones(shape=(debug_buffers_size, x0.size), dtype=float),
            t_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
            dt_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
        )
        if debug_buffers_size > 0
        else dict()
    )

    # Initialize the debug buffers with the initial state and time.
    if debug_buffers_size > 0:
        debug_buffers["x_steps"] = debug_buffers["x_steps"].at[0].set(x0)
        debug_buffers["t_steps"] = debug_buffers["t_steps"].at[0].set(t0)

    # =========================================================
    # While loop to reach tf from t0 using an adaptive timestep
    # =========================================================

    # Initialize the carry of the while loop.
    Carry = tuple
    carry0: Carry = (
        Δt0,
        x0,
        t0,
        # TODO: this dict should be the aux_dict/carry of the integrator!
        # What if it has other elements?? PyTree structure would fail.
        dict(f0=ẋ0) if fsal else dict(),
        jnp.array(0, dtype=int),
        False,
        debug_buffers,
    )

    def cond_outer(carry: Carry) -> jtp.Bool:
        _, _, _, _, _, break_loop, _ = carry
        return jnp.logical_not(break_loop)

    # Each loop is an integration step with variable Δt.
    # Depending on the integration error, the step could be discarded and the while body
    # ran again from the same (x0, t0) but with a smaller Δt.
    # We run these loops until the final time tf is reached.
    def body_outer(carry: Carry) -> Carry:
        """While loop body."""

        # Unpack the carry.
        Δt0, x0, t0, carry_integrator, discarded_steps, _, debug_buffers = carry

        # Let's take care of the final (variable) step.
        # We want the final Δt to let us reach tf exactly.
        # Then we can exit the while loop.
        Δt0 = jnp.where(t0 + Δt0 < tf, Δt0, tf - t0)
        break_loop = jnp.where(t0 + Δt0 < tf, False, True)

        # Calculate the next initial state and the corresponding integration error.
        # We enable FSAL (first-same-as-last) through the aux_dict (carry_integrator).
        # x0_next, _, error, carry_integrator_next = rk_method_closed(
        x0_next, error, carry_integrator_next = rk_method_closed(
            x0, t0, Δt0, carry_integrator
        )

        # Shrink the Δt every time by the safety factor.
        # The β parameters define the bounds of the timestep update factor.
        s = jnp.clip(safety, a_min=0.0, a_max=1.0)
        β_min = jnp.maximum(0.0, beta_min)
        β_max = jnp.maximum(β_min, beta_max)

        # Compute the next Δt from the desired integration error.
        # This new time step is accepted if error <= 1.0, otherwise it is rejected.
        Δt_next = Δt0 * jnp.clip(
            a=s * jnp.power(1 / error, 1 / (q + 1)),
            a_min=β_min,
            a_max=β_max,
        )

        def accept_step(debug_buffers: dict[str, Any]):
            if debug_buffers_size > 0:
                idx = debug_buffers["idx"]
                x_steps = debug_buffers["x_steps"]
                t_steps = debug_buffers["t_steps"]
                dt_steps = debug_buffers["dt_steps"]
                #
                idx = jnp.minimum(idx + 1, len(t_steps) - 1)
                x_steps = x_steps.at[idx].set(x0_next)
                t_steps = t_steps.at[idx].set(t0 + Δt0)
                dt_steps = dt_steps.at[idx - 1].set(Δt0)
                #
                debug_buffers = dict(
                    idx=idx, x_steps=x_steps, t_steps=t_steps, dt_steps=dt_steps
                )

            # TODO: round_to_decimal(dt_next)? Break if dt_next < dt_min?
            return (
                x0_next,
                t0 + Δt0,
                jnp.clip(Δt_next, dt_min, dt_max),
                carry_integrator_next,
                jnp.array(0, dtype=int),
                debug_buffers,
            )

        def reject_step(debug_buffers):
            return (
                x0,
                t0,
                jnp.clip(Δt_next, dt_min, dt_max),
                carry_integrator,
                discarded_steps + 1,
                debug_buffers,
            )

        (
            x0_next,
            t0_next,
            Δt_next,
            carry_integrator,
            discarded_steps,
            debug_buffers,
        ) = jax.lax.cond(
            pred=jnp.logical_or(
                jnp.logical_or(error <= 1.0, Δt_next < dt_min),
                discarded_steps >= max_step_rejections,
            ),
            true_fun=accept_step,
            true_operand=debug_buffers,
            false_fun=reject_step,
            false_operand=debug_buffers,
        )

        # Adjust dt such that is not smaller than the minimum allowed
        # dt_next = jnp.where(dt_next < dt_min, dt_min, dt_next)

        # t0_next = t0 + dt
        # x0_next = x0_higher
        #
        # if debug_buffers_size > 0:
        #     idx = jnp.minimum(idx + 1, len(t_steps) - 1)
        #     x_steps = x_steps.at[idx].set(x0_next)
        #     t_steps = t_steps.at[idx].set(t0_next)
        #     dt_steps = dt_steps.at[idx - 1].set(dt)
        #     # dt_steps = dt_steps.at[idx].set(dt_next)

        # Even if we thought that this while loop was the last one, maybe the step was
        # discarded and the Δt shrank
        break_loop = jnp.where(t0_next + Δt_next < tf, False, break_loop)

        # If this is the last while loop, we want to make sure that the returned Δt
        # is not the one that got shrank for reaching tf, but the last one computed
        # from the desired integration error to properly warm-start the next call.
        Δt_next = jnp.where(break_loop, Δt0, Δt_next)

        return (
            Δt_next,
            x0_next,
            t0_next,
            carry_integrator,
            discarded_steps,
            break_loop,
            debug_buffers,
        )

    Δt_final, x0_final, t0_final, _, _, _, debug_buffers = jax.lax.while_loop(
        cond_fun=cond_outer,
        body_fun=body_outer,
        init_val=carry0,
    )

    xf = x0_final
    Δt = Δt_final
    tf = t0_final + Δt_final

    print("returning:odeint_embedded_rk_one_step")
    return xf, dict(dt=Δt) | debug_buffers


@functools.partial(
    jax.jit,
    static_argnames=[
        "f",
        "integrator_type",
        "semi_implicit_quaternion_integration",
        "debug_buffers_size",
    ],
)
# TODO: x0 is ODEState but f takes a flat state!
def odeint_embedded_rk_manifold_one_step(
    f: StateDerivativeCallable,
    x0: ODEState,
    t0: Time,
    tf: Time,
    *,
    integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.BogackiShampine,
    physics_model: PhysicsModel,
    # dt0: jtp.FloatLike | None = None, # TODO: with None is not working
    dt0: jtp.FloatLike = 0.0,
    dt_min: jtp.FloatLike = -jnp.inf,
    dt_max: jtp.FloatLike = jnp.inf,
    rtol: jtp.FloatLike = RTOL_DEFAULT,
    atol: jtp.FloatLike = ATOL_DEFAULT,
    safety: jtp.FloatLike = SAFETY_DEFAULT,
    beta_min: jtp.FloatLike = BETA_MIN_DEFAULT,
    beta_max: jtp.FloatLike = BETA_MAX_DEFAULT,
    semi_implicit_quaternion_integration: jtp.BoolLike = True,
    debug_buffers_size: int | None = None,
    max_step_rejections: jtp.IntLike = MAX_STEP_REJECTIONS_DEFAULT,
) -> tuple[ODEState, dict[str, Any]]:
    """"""
    # This integrator works only with x \in ODEState

    def tf_next_state(x0: State, xf: State, t0: Time, dt: TimeStep) -> State:
        """
        Replace the quaternion in the final state with the one implicitly integrated
        on the SO(3) manifold.
        """

        # Convert the flat state to an ODEState pytree
        x0_ode_state = ODEState.deserialize(data=x0, physics_model=physics_model)
        xf_ode_state = ODEState.deserialize(data=xf, physics_model=physics_model)

        # Indices to convert quaternions between serializations
        to_xyzw = jnp.array([1, 2, 3, 0])
        to_wxyz = jnp.array([3, 0, 1, 2])

        # Get the initial quaternion and the inertial-fixed angular velocity
        W_ω_WB_t0 = x0_ode_state.physics_model.base_angular_velocity
        W_ω_WB_tf = xf_ode_state.physics_model.base_angular_velocity
        W_Q_B_t0 = so3.SO3.from_quaternion_xyzw(
            x0_ode_state.physics_model.base_quaternion[to_xyzw]
        )

        # Integrate implicitly the quaternion on its manifold using the angular velocity
        # transformed in body-fixed representation since jaxlie uses this convention
        B_R_W = W_Q_B_t0.inverse().as_matrix()
        W_ω_WB = W_ω_WB_tf if semi_implicit_quaternion_integration else W_ω_WB_t0
        W_Q_B_tf = W_Q_B_t0 @ so3.SO3.exp(tangent=dt * B_R_W @ W_ω_WB)

        # Store the quaternion in the final state
        xf_ode_state_manifold = xf_ode_state.replace(
            physics_model=xf_ode_state.physics_model.replace(
                base_quaternion=W_Q_B_tf.as_quaternion_xyzw()[to_wxyz]
            )
        )

        return xf_ode_state_manifold.flatten()

    # Flatten the ODEState. We use the unflatten_fn to convert the flat state back.
    x0_flat, unflatten_fn = jax.flatten_util.ravel_pytree(x0)

    # Integrate the flat ODEState with the embedded Runge-Kutta integrator.
    xf_flat, aux_dict = odeint_embedded_rk_one_step(
        f=f,
        x0=x0_flat,
        t0=t0,
        tf=tf,
        integrator_type=integrator_type,
        dt0=dt0,
        dt_min=dt_min,
        dt_max=dt_max,
        rtol=rtol,
        atol=atol,
        safety=safety,
        beta_min=beta_min,
        beta_max=beta_max,
        debug_buffers_size=debug_buffers_size,
        max_step_rejections=max_step_rejections,
        tf_next_state=tf_next_state,
    )

    # Convert the flat state back to ODEState.
    # Note that the aux_dict might contain flattened data that is not post-processed.
    xf = unflatten_fn(xf_flat)

    print("returning:odeint_embedded_rk_manifold_one_step")
    return xf, aux_dict


# =============================================
# Other variable-step integrators (single step)
# =============================================


# TODO: finalize this since it's still very drafty

# Desiderata:
#
# - Accurate for stiff systems.
# - Adaptive time step (embedded-like?).
# - Integrate the quaternion part of x on its manifold.
# - Preferably, with the integration error at least O(dt^2).
# - Preferably implicit or semi-implicit.


def odeint_adaptive_leapfrog_manifold_one_step(
    f: StateDerivativeCallable,
    x0: ODEState,
    t0: Time,
    tf: Time,
    *,
    physics_model: PhysicsModel,
    dt0: jax.Array | float | None = None,
    dt_min: float = -jnp.inf,
    dt_max: float = jnp.inf,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    debug_buffers_size: int | None = None,
    max_step_rejections: int = 5,
) -> tuple[ODEState, dict[str, Any]]:
    """"""

    # Work on flattened data
    x0_ode_state = x0
    x0 = x0_ode_state.flatten()

    # Define the order
    q = 2
    p = 2  # TODO

    # Compute the initial step size considering the order of the integrator,
    # and clip it to the given bounds, if necessary.
    dt0 = jnp.clip(
        a=(
            jnp.array(dt0)
            if dt0 is not None
            else estimate_step_size(x0=x0, t0=t0, f=f, order=p, atol=atol, rtol=rtol)
        ),
        a_min=jnp.minimum(dt_min, tf - t0),
        a_max=jnp.minimum(dt_max, tf - t0),
    )

    # Round dt0 such that the halved value is not less then nanoseconds.
    # TODO: remove this for dt computed from error
    # dt0 = round_to_decimal(x=dt0, decimal=1e-8)

    # Initialize the size of the debug buffers
    debug_buffers_size = debug_buffers_size if debug_buffers_size is not None else 0

    debug_buffers = (
        dict(
            idx=jnp.array(0, dtype=int),
            x_steps=-jnp.inf
            * jnp.ones(shape=(debug_buffers_size, x0.size), dtype=float),
            t_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
            dt_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
        )
        if debug_buffers_size > 0
        else dict()
    )

    if debug_buffers_size > 0:
        debug_buffers["x_steps"] = debug_buffers["x_steps"].at[0].set(x0)
        debug_buffers["t_steps"] = debug_buffers["t_steps"].at[0].set(t0)

    Carry = tuple
    carry0: Carry = (
        dt0,
        x0,
        t0,
        dict(f0=f(x0, t0)[0]),
        jnp.array(0, dtype=int),
        False,
        debug_buffers,
    )

    def cond_outer(carry: Carry) -> bool | jax.Array:
        _, _, _, _, _, break_loop, _ = carry
        return jnp.logical_not(break_loop)

    # Each loop is an integration step with variable dt. Depending on the integration error,
    # the step could be discarded and start again from the same (x0, t0) with a smaller dt.
    # We run these loops until the final time tf is reached.
    def body_outer(carry: Carry) -> Carry:
        """"""

        # Unpack the carry
        dt0, x0, t0, carry_integrator, discarded_steps, _, debug_buffers = carry

        # Let's take care of the final (variable) step.
        # We want the final dt to let us reach tf exactly.
        # Then we can exit the while loop.
        dt0 = jnp.where(t0 + dt0 < tf, dt0, tf - t0)
        break_loop = jnp.where(t0 + dt0 < tf, False, True)

        # Calculate the next state and the corresponding integration error
        x0_next, _, error, carry_integrator_next = rk_method_closed(
            x0=x0, t0=t0, dt=dt0, aux_dict=carry_integrator
        )

        # Shrink the dt every time by the safety factor.
        # We allow reducing the step size by a factor of 10, and increasing it
        # by a factor of 2.5.
        safety = 0.9
        β_min, β_max = (1.0 / 10, 2.5)

        # Compute the next dt from the desired integration error.
        # The step is accepted if err <= 1.0, otherwise it is rejected.
        dt_next = dt0 * jnp.minimum(
            β_max, jnp.maximum(β_min, safety * jnp.power(1 / error, 1 / (q + 1)))
        )

        def accept_step(debug_buffers: dict[str, Any]):
            if debug_buffers_size > 0:
                idx = debug_buffers["idx"]
                x_steps = debug_buffers["x_steps"]
                t_steps = debug_buffers["t_steps"]
                dt_steps = debug_buffers["dt_steps"]
                #
                idx = jnp.minimum(idx + 1, len(t_steps) - 1)
                x_steps = x_steps.at[idx].set(x0_next)
                t_steps = t_steps.at[idx].set(t0 + dt0)
                dt_steps = dt_steps.at[idx - 1].set(dt0)
                #
                debug_buffers = dict(
                    idx=idx, x_steps=x_steps, t_steps=t_steps, dt_steps=dt_steps
                )

            # TODO: round_to_decimal(dt_next)? Break if dt_next < dt_min?
            return (
                x0_next,
                t0 + dt0,
                jnp.clip(dt_next, dt_min, dt_max),
                carry_integrator_next,
                jnp.array(0, dtype=int),
                debug_buffers,
            )

        def reject_step(debug_buffers):
            return (
                x0,
                t0,
                dt_next,
                carry_integrator,
                discarded_steps + 1,
                debug_buffers,
            )

        (
            x0_next,
            t0_next,
            dt_next,
            carry_integrator,
            discarded_steps,
            debug_buffers,
        ) = jax.lax.cond(
            pred=jnp.logical_or(
                jnp.logical_or(error <= 1.0, dt_next < dt_min),
                discarded_steps >= max_step_rejections,
            ),
            true_fun=accept_step,
            true_operand=debug_buffers,
            false_fun=reject_step,
            false_operand=debug_buffers,
        )

        # Even if we thought that this while loop was the last one, maybe the step was
        # discarded and the dt shrank
        break_loop = jnp.where(t0_next + dt_next < tf, False, break_loop)

        # If this is the last while loop, we want to make sure that the returned dt
        # is not the one that got shrank for reaching tf, but the last one computed
        # from the desired integration error
        dt_next = jnp.where(break_loop, dt0, dt_next)

        return (
            dt_next,
            x0_next,
            t0_next,
            carry_integrator,
            discarded_steps,
            break_loop,
            debug_buffers,
        )

    # ==================================
    # While loop to reach the final time
    # ==================================

    dt_final, x0_final, t0_final, _, _, _, debug_buffers = jax.lax.while_loop(
        cond_fun=cond_outer,
        body_fun=body_outer,
        init_val=carry0,
    )

    xf = x0_final
    dt = dt_final
    tf = t0_final + dt_final

    print("returning:odeint_adaptive_leapfrog_manifold_one_step")
    return xf, dict(dt=dt) | debug_buffers


# ===============================
# Integration over a time horizon
# ===============================


# TODO: to be compatible with the other integrators, we should not expose f


@functools.partial(
    jax.jit,
    static_argnames=["odeint_adaptive_one_step", "debug_buffers_size_per_step"],
)
def _ode_integration_adaptive_template(
    x0: State,
    t: TimeHorizon,
    *,
    odeint_adaptive_one_step: Callable[
        [State, Time, Time, TimeStep], tuple[StateNext, dict[str, Any]]
    ],
    # integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.BogackiShampine,
    dt0: jax.Array | float | None = None,
    debug_buffers_size_per_step: int | None = None,
) -> tuple[StateNext, dict[str, Any]]:
    """"""

    # TODO: think a way to extract this data in a smarter way
    # rk_method, q = (heun_euler, jnp.minimum(1, 2))
    # rk_method, q = (bogacki_shampine, jnp.minimum(2, 3))

    # Extract initial and final times
    # t0, tf = (t[0], t[-1])

    # Compute the initial step size considering the order of the integrator,
    # and clip it to the given bounds, if necessary.
    # TODO: update order
    # dt0 = jnp.clip(
    #     a=(
    #         jnp.array(dt0)
    #         if dt0 is not None
    #         else estimate_step_size(x0=x0, t0=t0, f=f, order=q, atol=atol, rtol=rtol)
    #     ),
    #     a_min=jnp.minimum(dt_min, tf - t0),
    #     a_max=jnp.minimum(dt_max, tf - t0),
    # )

    # Select the target one-step integrator.
    # odeint_embedded_closed = lambda x0, t0, tf, dt0: odeint_embedded_rk_one_step(
    #     f=f,
    #     x0=x0,
    #     t0=t0,
    #     tf=tf,
    #     integrator_type=integrator_type,
    #     dt0=dt0,
    #     dt_min=dt_min,
    #     dt_max=dt_max,
    #     rtol=rtol,
    #     atol=atol,
    #     safety=safety,
    #     beta_min=beta_min,
    #     beta_max=beta_max,
    #     debug_buffers_size=debug_buffers_size_per_step,
    #     max_step_rejections=max_step_rejections,
    #     tf_next_state=tf_next_state,
    # )

    # odeint_embedded_closed = odeint_one_step

    # Adjust some of the input arguments.
    x0 = jnp.array(x0, dtype=float)
    dt0 = (
        jnp.array(dt0, dtype=float) if dt0 is not None else jnp.array(0.0, dtype=float)
    )

    debug_buffers_size = (
        debug_buffers_size_per_step * len(t)
        if debug_buffers_size_per_step is not None
        else 0
    )

    debug_buffers = (
        dict(
            idx=jnp.array(0, dtype=int),
            x_steps=-jnp.inf
            * jnp.ones(shape=(debug_buffers_size, x0.size), dtype=float),
            t_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
            dt_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
        )
        if debug_buffers_size > 0
        else dict()
    )

    if debug_buffers_size > 0:
        debug_buffers["x_steps"] = debug_buffers["x_steps"].at[0].set(x0)
        debug_buffers["t_steps"] = debug_buffers["t_steps"].at[0].set(t[0])

    # =================================================
    # For loop to integrate on the horizon defined by t
    # =================================================

    Carry = tuple
    carry0 = (x0, dt0, debug_buffers)

    def body(carry: Carry, i: float | jax.Array) -> tuple[Carry, jax.Array]:
        """For loop body."""

        # Unpack the carry.
        x0, dt0, debug_buffers = carry

        # Calculate the final state (the integrator can take an arbitrary number of steps)
        # and the auxiliary data (e.g. the last dt and the debug buffers).
        xf, dict_aux = odeint_adaptive_one_step(x0, t[i], t[i + 1], dt0)

        if debug_buffers_size > 0:
            # Get the horizon data
            idx = debug_buffers["idx"]
            x_steps = debug_buffers["x_steps"]
            t_steps = debug_buffers["t_steps"]
            dt_steps = debug_buffers["dt_steps"]

            # Get the single-step data
            x_odeint = dict_aux["x_steps"]
            t_odeint = dict_aux["t_steps"]
            dt_odeint = dict_aux["dt_steps"]

            # Merge the buffers
            x_steps = jax.lax.dynamic_update_slice(x_steps, x_odeint, (idx, 0))
            t_steps = jax.lax.dynamic_update_slice(t_steps, t_odeint, (idx,))
            dt_steps = jax.lax.dynamic_update_slice(dt_steps, dt_odeint, (idx,))

            # Advance the index
            idx_odeint = dict_aux["idx"]
            idx += idx_odeint

            debug_buffers = dict(
                idx=idx, x_steps=x_steps, t_steps=t_steps, dt_steps=dt_steps
            )

        return (xf, dict_aux["dt"], debug_buffers), xf

    (_, dt_final, debug_buffers), X = jax.lax.scan(
        f=body,
        init=carry0,
        xs=jnp.arange(start=0, stop=len(t) - 1, dtype=int),
    )

    print("returning:ode_integration_embedded_rk")
    return (
        jnp.vstack([jnp.atleast_1d(x0.squeeze()), jnp.atleast_2d(X.squeeze())]),
        debug_buffers | dict(dt=dt_final),
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "f",
        "integrator_type",
        "debug_buffers_size_per_step",
        "tf_next_state",
    ],
)
def ode_integration_embedded_rk(
    x0: State,
    t: TimeHorizon,
    *,
    f: StateDerivativeCallable,
    integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.BogackiShampine,
    dt0: jax.Array | float | None = None,
    dt_min: float = -jnp.inf,
    dt_max: float = jnp.inf,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    safety: jtp.FloatLike = SAFETY_DEFAULT,
    beta_min: jtp.FloatLike = BETA_MIN_DEFAULT,
    beta_max: jtp.FloatLike = BETA_MAX_DEFAULT,
    debug_buffers_size_per_step: int | None = None,
    max_step_rejections: int = MAX_STEP_REJECTIONS_DEFAULT,
    tf_next_state: Callable[
        [State, StateNext, Time, TimeStep], StateNext
    ] = lambda x0, xf, t0, dt: xf,
) -> tuple[StateNext, dict[str, Any]]:
    """"""

    # Select the target one-step integrator.
    odeint_adaptive_one_step = lambda x0, t0, tf, dt0: odeint_embedded_rk_one_step(
        f=f,
        x0=x0,
        t0=t0,
        tf=tf,
        integrator_type=integrator_type,
        dt0=dt0,
        dt_min=dt_min,
        dt_max=dt_max,
        rtol=rtol,
        atol=atol,
        safety=safety,
        beta_min=beta_min,
        beta_max=beta_max,
        debug_buffers_size=debug_buffers_size_per_step,
        max_step_rejections=max_step_rejections,
        tf_next_state=tf_next_state,
    )

    # Integrate the state with an adaptive timestep over the horizon defined by `t`.
    return _ode_integration_adaptive_template(
        x0=x0,
        t=t,
        odeint_adaptive_one_step=odeint_adaptive_one_step,
        dt0=dt0,
        debug_buffers_size_per_step=debug_buffers_size_per_step,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "f",
        "integrator_type",
        "semi_implicit_quaternion_integration",
        "debug_buffers_size_per_step",
    ],
)
# TODO: f takes flat but x0 is ODEState
def ode_integration_embedded_rk_manifold(
    x0: ODEState,
    t: TimeHorizon,
    *,
    f: StateDerivativeCallable,
    integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.BogackiShampine,
    # integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.HeunEuler,
    physics_model: PhysicsModel,
    dt0: jax.Array | float | None = None,
    dt_min: float = -jnp.inf,
    dt_max: float = jnp.inf,
    rtol: float = RTOL_DEFAULT,
    atol: float = ATOL_DEFAULT,
    safety: jtp.FloatLike = SAFETY_DEFAULT,
    beta_min: jtp.FloatLike = BETA_MIN_DEFAULT,
    beta_max: jtp.FloatLike = BETA_MAX_DEFAULT,
    semi_implicit_quaternion_integration: jtp.BoolLike = True,
    debug_buffers_size_per_step: int | None = None,
    max_step_rejections: int = MAX_STEP_REJECTIONS_DEFAULT,
) -> tuple[ODEState, dict[str, Any]]:
    """"""

    # Define the functions to flatten and unflatten ODEState objects.
    flatten_fn = lambda ode_state: x0.flatten_fn()(ode_state)
    unflatten_fn = lambda x: x0.unflatten_fn()(x)

    # Select the target one-step integrator.
    def odeint_adaptive_one_step(x0, t0, tf, dt0):
        out = odeint_embedded_rk_manifold_one_step(
            f=f,
            x0=unflatten_fn(x0),
            t0=t0,
            tf=tf,
            integrator_type=integrator_type,
            physics_model=physics_model,
            dt0=dt0,
            dt_min=dt_min,
            dt_max=dt_max,
            rtol=rtol,
            atol=atol,
            safety=safety,
            beta_min=beta_min,
            beta_max=beta_max,
            semi_implicit_quaternion_integration=semi_implicit_quaternion_integration,
            debug_buffers_size=debug_buffers_size_per_step,
            max_step_rejections=max_step_rejections,
        )
        return flatten_fn(out[0]), out[1]

    # Flatten the ODEState. The adaptive template operates on flat data.
    # x0_flat, unflatten_fn = jax.flatten_util.ravel_pytree(x0)

    # Integrate the state with an adaptive timestep over the horizon defined by `t`.
    X_flat, dict_aux = _ode_integration_adaptive_template(
        # x0=x0_flat,
        x0=flatten_fn(x0),
        t=t,
        odeint_adaptive_one_step=odeint_adaptive_one_step,
        dt0=dt0,
        debug_buffers_size_per_step=debug_buffers_size_per_step,
    )

    # Unflatten the integrated flat data.
    X = jax.vmap(unflatten_fn)(X_flat)

    # Unflatten the optional debug data included in dict_aux.
    dict_aux_unflattened = dict_aux | (
        dict()
        if "x_steps" not in dict_aux
        else dict(x_steps=jax.vmap(unflatten_fn)(dict_aux["x_steps"]))
    )

    # Return the unflattened output.
    return X, dict_aux_unflattened


# @functools.partial(
#     jax.jit,
#     static_argnames=[
#         "f",
#         "integrator_type",
#         "debug_buffers_size_per_step",
#     ],
# )
# def ode_integration_embedded_rk_manifold_old(
#     x0: State,
#     t: TimeHorizon,
#     *,
#     f: Callable,
#     physics_model: PhysicsModel,
#     integrator_type: AdaptiveIntegratorType = AdaptiveIntegratorType.BogackiShampine,
#     dt0: jax.Array | float | None = None,
#     dt_min: float = -jnp.inf,
#     dt_max: float = jnp.inf,
#     rtol: float = RTOL_DEFAULT,
#     atol: float = ATOL_DEFAULT,
#     safety: jtp.FloatLike = SAFETY_DEFAULT,
#     beta_min: jtp.FloatLike = BETA_MIN_DEFAULT,
#     beta_max: jtp.FloatLike = BETA_MAX_DEFAULT,
#     debug_buffers_size_per_step: int | None = None,
#     max_step_rejections: int = 5,
# ) -> tuple[State, dict[str, Any]]:
#     """"""
#
#     # Select the target one-step integrator.
#     odeint_embedded_closed = (
#         lambda x0, t0, tf, dt0: odeint_embedded_rk_manifold_one_step(
#             f=f,
#             x0=x0,
#             t0=t0,
#             tf=tf,
#             integrator_type=integrator_type,
#             dt0=dt0,
#             dt_min=dt_min,
#             dt_max=dt_max,
#             rtol=rtol,
#             atol=atol,
#             safety=safety,
#             beta_min=beta_min,
#             beta_max=beta_max,
#             debug_buffers_size=debug_buffers_size_per_step,
#             max_step_rejections=max_step_rejections,
#         )
#     )
#
#     # TODO: think a way to extract this data in a smarter way
#     # rk_method, q = (heun_euler, jnp.minimum(1, 2))
#     # rk_method, q = (bogacki_shampine, jnp.minimum(2, 3))
#
#     # Extract initial and final times
#     # t0, tf = (t[0], t[-1])
#
#     # Compute the initial step size considering the order of the integrator,
#     # and clip it to the given bounds, if necessary.
#     # TODO: update order
#     # dt0 = jnp.clip(
#     #     a=(
#     #         jnp.array(dt0)
#     #         if dt0 is not None
#     #         else estimate_step_size(x0=x0, t0=t0, f=f, order=q, atol=atol, rtol=rtol)
#     #     ),
#     #     a_min=jnp.minimum(dt_min, tf - t0),
#     #     a_max=jnp.minimum(dt_max, tf - t0),
#     # )
#
#     # print(f"dt0={dt0}")
#
#     # Adjust some of the input arguments.
#     x0 = jnp.array(x0, dtype=float)
#     dt0 = (
#         jnp.array(dt0, dtype=float) if dt0 is not None else jnp.array(0.0, dtype=float)
#     )
#
#     debug_buffers_size = (
#         debug_buffers_size_per_step * len(t)
#         if debug_buffers_size_per_step is not None
#         else 0
#     )
#
#     # debug_buffers = dict(
#     #     idx=jnp.array(0, dtype=int),
#     #     x_steps=-jnp.inf * jnp.ones(shape=(debug_buffers_size, x0.size), dtype=float)
#     #     if debug_buffers_size > 0
#     #     else None,
#     #     t_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float)
#     #     if debug_buffers_size > 0
#     #     else None,
#     #     dt_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float)
#     #     if debug_buffers_size > 0
#     #     else None,
#     # )
#
#     debug_buffers = (
#         dict(
#             idx=jnp.array(0, dtype=int),
#             x_steps=-jnp.inf
#             * jnp.ones(shape=(debug_buffers_size, x0.size), dtype=float),
#             t_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
#             dt_steps=jnp.array([-jnp.inf] * debug_buffers_size, dtype=float),
#         )
#         if debug_buffers_size > 0
#         else dict()
#     )
#
#     if debug_buffers_size > 0:
#         debug_buffers["x_steps"] = debug_buffers["x_steps"].at[0].set(x0)
#         debug_buffers["t_steps"] = debug_buffers["t_steps"].at[0].set(t0)
#
#     # =================================================
#     # For loop to integrate on the horizon defined by t
#     # =================================================
#
#     Carry = tuple
#     carry0 = (x0, dt0, debug_buffers)
#
#     def body(carry: Carry, i: float | jax.Array) -> tuple[Carry, jax.Array]:
#         """For loop body."""
#
#         x0, dt0, debug_buffers = carry
#
#         xf, dict_aux = odeint_embedded_rk_manifold_one_step(
#             f=f,
#             x0=x0,
#             t0=t[i],
#             tf=t[i + 1],
#             physics_model=physics_model,
#             dt0=dt0,
#             dt_min=dt_min,
#             dt_max=dt_max,
#             rtol=rtol,
#             atol=atol,
#             debug_buffers_size=debug_buffers_size_per_step,
#             max_step_rejections=max_step_rejections,
#             # tf_next_state=tf_next_state,
#         )
#
#         if debug_buffers_size > 0:
#             idx = debug_buffers["idx"]
#             x_steps = debug_buffers["x_steps"]
#             t_steps = debug_buffers["t_steps"]
#             dt_steps = debug_buffers["dt_steps"]
#             #
#             x_steps = jax.lax.dynamic_update_slice(
#                 x_steps, dict_aux["x_steps"], (idx, 0)
#             )
#             t_steps = jax.lax.dynamic_update_slice(t_steps, dict_aux["t_steps"], (idx,))
#             dt_steps = jax.lax.dynamic_update_slice(
#                 dt_steps, dict_aux["dt_steps"], (idx,)
#             )
#             #
#             idx_in = dict_aux["idx"]
#             idx += idx_in
#             #
#             debug_buffers = dict(
#                 idx=idx, x_steps=x_steps, t_steps=t_steps, dt_steps=dt_steps
#             )
#
#         return (xf, dict_aux["dt"], debug_buffers), xf
#
#     (_, _, debug_buffers), X = jax.lax.scan(
#         f=body,
#         init=carry0,
#         xs=jnp.arange(start=0, stop=len(t) - 1, dtype=int),
#     )
#
#     print("returning:ode_integration_embedded_rk_manifold")
#     return (
#         jnp.vstack([jnp.atleast_1d(x0.squeeze()), jnp.atleast_2d(X.squeeze())]),
#         debug_buffers,
#     )
