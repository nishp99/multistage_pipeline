"""Six-state evolution in the basis P, C, C² with only degree three closed."""

import jax.numpy as jnp
from jax import lax


def signal_noise_parameters(state):
    J0, J1, J2, Q0, Q1, Q2 = state
    return J1, J0 - J1, Q1, Q0 - Q1


def second_order_parameters(state):
    """Exact (Jss, Jsn, Jnn, Qss, Qsn, Qnn); no closure at degree two."""
    J0, J1, J2, Q0, Q1, Q2 = state
    return J2, J1 - J2, J0 - 2 * J1 + J2, Q2, Q1 - Q2, Q0 - 2 * Q1 + Q2


def _closed_third_order(state, closure):
    return closure @ state[:3], closure @ state[3:]


# Field expectations and clean/noisy classification probabilities.
def Expectation_X_signY(Cov_XY, Var_Y):
    return jnp.sqrt(2 / jnp.pi) * Cov_XY / jnp.sqrt(Var_Y)


def Prob_correct_rl(J0, Q0, Q1, S, delta):
    overlap = J0 / jnp.sqrt(S * (delta * Q0 + (1 - delta) * Q1))
    overlap = jnp.clip(overlap, -1 + 1e-9, 1 - 1e-9)
    return 0.5 + jnp.arcsin(overlap) / jnp.pi


def Prob_correct_sft(J0, Q0, Q1, S, delta, epsilon):
    return (1 - 2 * epsilon) * Prob_correct_rl(J0, Q0, Q1, S, delta) + epsilon


def probability_gradient(state, S, delta):
    """Clean accuracy gradient in spectral order (J0,J1,J2,Q0,Q1,Q2)."""
    J0, _, _, Q0, Q1, _ = state
    qhat = delta * Q0 + (1 - delta) * Q1
    r = jnp.clip(J0 / jnp.sqrt(S * qhat), -1 + 1e-9, 1 - 1e-9)
    pJ = 1 / (jnp.pi * jnp.sqrt(S * qhat) * jnp.sqrt(1 - r**2))
    pQ = -J0 * pJ / (2 * qhat)
    return jnp.array([pJ, 0.0, 0.0, delta * pQ, (1 - delta) * pQ, 0.0])


def reprojection_opt_lr_coeffs(state, S, delta, A_J, A_Q, B_Q):
    """Leading-in-dt accuracy coefficients (b,a), unchanged by rescaling.

    Accuracy is scale invariant, so the radial correction has zero contraction
    with its gradient. These coefficients apply with or without the constraint.
    """
    gradient = probability_gradient(state, S, delta)
    b = gradient[:3] @ A_J + gradient[3:] @ A_Q
    a = gradient[3:] @ B_Q
    return b, a


def _optimal_lr(a, b):
    """Original heuristic, deliberately NOT the TeX bounded maximizer."""
    opt_lr = jnp.where(
        (a < 0) & (b > 0),
        -b / (2 * a),
        jnp.where(
            (a > 0) & (b < 0),
            jnp.maximum(-b / a, 0.0) + 1e-6,
            jnp.where(b > 0, 1.0, 1e-6),
        ),
    )
    return jnp.minimum(opt_lr, 1.0)


def _exact_sphere_reproject(state, radius):
    s2 = radius / jnp.maximum(state[3], 1e-300)
    return jnp.concatenate((state[:3] * jnp.sqrt(s2), state[3:] * s2))


# Linear drift and quadratic diffusion, vectorized over degrees k=0,1,2.
def _field_expectations(state, S, delta, teacher_moments, closure):
    J, Q = state[:3], state[3:]
    J3, Q3 = _closed_third_order(state, closure)
    J_next = jnp.concatenate((J[1:], jnp.atleast_1d(J3)))
    Q_next = jnp.concatenate((Q[1:], jnp.atleast_1d(Q3)))
    qhat = delta * Q[0] + (1 - delta) * Q[1]
    nu_sign_lambda = Expectation_X_signY(delta * J + (1 - delta) * J_next, qhat)
    lambda_sign_lambda = Expectation_X_signY(delta * Q + (1 - delta) * Q_next, qhat)
    nu_sign_nu = Expectation_X_signY(teacher_moments, S)
    lambda_sign_nu = Expectation_X_signY(J, S)
    return nu_sign_lambda, lambda_sign_lambda, nu_sign_nu, lambda_sign_nu


def diffusion_constants(moments, delta):
    return delta * moments[:3] + (1 - delta) * moments[1:4]


def sft_coefficients(state, S, delta, T, epsilon, teacher_moments, closure, H):
    nu_l, lambda_l, nu_n, lambda_n = _field_expectations(
        state, S, delta, teacher_moments, closure
    )
    P = Prob_correct_sft(state[0], state[3], state[4], S, delta, epsilon)
    A_J = -(nu_l - (1 - 2 * epsilon) * nu_n) / 2
    A_Q = -(lambda_l - (1 - 2 * epsilon) * lambda_n)
    B_Q = H * (1 - P) / T
    return A_J, A_Q, B_Q


def rl_coefficients(state, S, delta, T, teacher_moments, closure, H):
    nu_l, lambda_l, nu_n, lambda_n = _field_expectations(
        state, S, delta, teacher_moments, closure
    )
    P = Prob_correct_rl(state[0], state[3], state[4], S, delta)
    A_J = (nu_l + nu_n) * P ** (T - 1) / 2
    A_Q = (lambda_l + lambda_n) * P ** (T - 1)
    B_Q = H * P**T / T
    return A_J, A_Q, B_Q


def _euler_step(
    state,
    dt,
    delta,
    lr,
    radius,
    S,
    coefficients,
    use_optimal_lr,
    use_spherical_constraint_projection,
):
    A_J, A_Q, B_Q = coefficients

    # Avoid evaluating the accuracy gradient when a fixed LR is requested.
    def optimal(_):
        b, a = reprojection_opt_lr_coeffs(state, S, delta, A_J, A_Q, B_Q)
        return _optimal_lr(a, b)

    eta = lax.cond(use_optimal_lr, optimal, lambda _: jnp.asarray(lr), None)
    increments = jnp.concatenate((eta * A_J, eta * A_Q + eta**2 * B_Q))
    state = state + dt * increments
    state = lax.cond(
        use_spherical_constraint_projection,
        lambda z: _exact_sphere_reproject(z, radius),
        lambda z: z,
        state,
    )
    return state, eta
