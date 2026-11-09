"""Four-state Hilbert–Schmidt closure in signal/noise coordinates."""

from jax import lax
import jax.numpy as jnp
from jax.scipy.special import betainc


def _closed_second_order(a, b, j1, j2, q1, q2):
    j11, q11 = a * j1 + b * j2, a * q1 + b * q2
    j12, q12 = j1 - j11, q1 - q11
    return j11, j12, j2 - j12, q11, q12, q2 - q12


def Prob_correct_sft(J_1, J_2, Q_1, Q_2, S, delta, epsilon):
    overlap = (J_1 + J_2) / jnp.sqrt((S) * (Q_1 + delta * Q_2))
    overlap = jnp.clip(
        overlap, -1 + 1e-9, 1 - 1e-9
    )  # avoid numerical issues with arcsin
    P = (1 / 2 + 1 / jnp.pi * jnp.arcsin(overlap)) * (1 - 2 * epsilon) + epsilon
    return P


def Prob_correct_rl(J_1, J_2, Q_1, Q_2, S, delta, T, agreement_threshold):
    """Per-decision accuracy and episode-gate probabilities for n-or-more RL.

    Let p be the probability that one student decision agrees with the teacher,
    and K ~ Binomial(T, p).  This returns

      p,
      Pr[K >= n],
      Pr[Binomial(T-1,p) >= n-1],  # gate probability if the tagged decision is correct
      Pr[Binomial(T-1,p) >= n],    # gate probability if the tagged decision is wrong

    where n = agreement_threshold.  The regularized incomplete beta function
    evaluates the binomial tails without constructing an array of length T, so T
    may remain vmapped while n is a single non-vmapped scalar.
    """
    overlap = (J_1 + J_2) / jnp.sqrt(S * (Q_1 + delta * Q_2))
    overlap = jnp.clip(overlap, -1 + 1e-9, 1 - 1e-9)
    p = 1 / 2 + 1 / jnp.pi * jnp.arcsin(overlap)

    dtype = p.dtype
    T_f = jnp.asarray(T, dtype=dtype)
    n_f = jnp.asarray(agreement_threshold, dtype=dtype)

    # Pr[Binomial(T,p) >= n] = I_p(n, T-n+1).
    prob_reward = betainc(
        jnp.maximum(n_f, 1.0),
        jnp.maximum(T_f - n_f + 1.0, 1.0),
        p,
    )

    # Conditional gate probabilities for a tagged decision.  Clipping the beta
    # parameters keeps both branches numerically valid under JAX's select; the
    # exact boundary values are then restored with where.
    prob_reward_if_correct_raw = betainc(
        jnp.maximum(n_f - 1.0, 1.0),
        jnp.maximum(T_f - n_f + 1.0, 1.0),
        p,
    )
    prob_reward_if_correct = jnp.where(
        n_f <= 1.0, jnp.ones_like(p), prob_reward_if_correct_raw
    )

    prob_reward_if_incorrect_raw = betainc(
        jnp.maximum(n_f, 1.0),
        jnp.maximum(T_f - n_f, 1.0),
        p,
    )
    prob_reward_if_incorrect = jnp.where(
        n_f >= T_f, jnp.zeros_like(p), prob_reward_if_incorrect_raw
    )

    return p, prob_reward, prob_reward_if_correct, prob_reward_if_incorrect


def DP_DJ(J_1, J_2, Q_1, Q_2, S, delta):
    return 1 / (
        jnp.pi
        * jnp.sqrt(S * (Q_1 + delta * Q_2))
        * jnp.sqrt(1 - ((J_1 + J_2) ** 2) / (S * (Q_1 + delta * Q_2)))
    )


def DP_DQ1(J_1, J_2, Q_1, Q_2, S, delta):
    return -(J_1 + J_2) / (
        2
        * jnp.pi
        * jnp.sqrt(S)
        * (Q_1 + delta * Q_2) ** (3 / 2)
        * jnp.sqrt(1 - ((J_1 + J_2) ** 2) / (S * (Q_1 + delta * Q_2)))
    )


def DP_DQ2(J_1, J_2, Q_1, Q_2, S, delta):
    return (
        -delta
        * (J_1 + J_2)
        / (
            2
            * jnp.pi
            * jnp.sqrt(S)
            * (Q_1 + delta * Q_2) ** (3 / 2)
            * jnp.sqrt(1 - ((J_1 + J_2) ** 2) / (S * (Q_1 + delta * Q_2)))
        )
    )


def dP_eta_coefficient_unconstrained(dP_dJ, dP_dQ1, dP_dQ2, A_J1, A_J2, A_Q1, A_Q2):
    return dP_dJ * (A_J1 + A_J2) + dP_dQ1 * A_Q1 + dP_dQ2 * A_Q2


def dP_eta_squared_coefficient_unconstrained(dP_dQ1, dP_dQ2, B_Q1, B_Q2):
    return dP_dQ1 * B_Q1 + dP_dQ2 * B_Q2


def _optimal_lr(a, b):
    """Locally optimal learning rate from the (a*eta^2 + b*eta) accuracy model.

    Identical decision logic to the original update functions.
    """
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


def _exact_sphere_reproject(J_1, J_2, Q_1, Q_2, radius):
    """Project (J, Q) back onto the constraint surface Q_1 + Q_2 = radius.

    Geometric meaning: rescale the student weight vector w -> s w with
    s = sqrt(radius / (Q_1 + Q_2)).  Q ~ w^2 scales by s^2, J ~ w scales by s.

    Overlap (J_1+J_2)/sqrt(S (Q_1+delta Q_2)) is EXACTLY invariant under this map
    for any scale s (the s in the numerator cancels sqrt(s^2) in the denominator),
    so Prob_correct -- which depends only on the overlap -- is unchanged: the
    learning physics is preserved while Q_1+Q_2 is pinned to radius.

    The ``jnp.maximum`` guards the rescale if the post-increment norm ever
    underflows to ~0.  On-trajectory this never triggers (Q_1+Q_2 stays at radius
    and each increment is O(dt)), but it keeps the map total.
    """
    total = jnp.maximum(Q_1 + Q_2, 1e-300)
    s2 = radius / total
    s = jnp.sqrt(s2)
    return J_1 * s, J_2 * s, Q_1 * s2, Q_2 * s2


def reprojection_opt_lr_coeffs(
    dP_dJ, dP_dQ1, dP_dQ2, A_J1, A_J2, A_Q1, A_Q2, B_Q1, B_Q2
):
    """Linear and quadratic accuracy coefficients. Spherical rescaling preserves overlap."""
    b = dP_eta_coefficient_unconstrained(dP_dJ, dP_dQ1, dP_dQ2, A_J1, A_J2, A_Q1, A_Q2)
    a = dP_eta_squared_coefficient_unconstrained(dP_dQ1, dP_dQ2, B_Q1, B_Q2)
    return b, a


def compact_coefficients(
    state,
    delta,
    T,
    S,
    teacher_moments,
    omega,
    traces,
    agreement_threshold,
    epsilon=0.0,
    phase="rl",
    second_order=None,
):
    """Return the eta-linear J/Q and eta-squared Q coefficients.

    ``second_order`` supplies measured J11,J12,J22,Q11,Q12,Q22 only for
    same-state diagnostics. Production dynamics always use the compact closure.
    """
    J1, J2, Q1, Q2 = state
    if second_order is None:
        second_order = _closed_second_order(omega[0], omega[1], *state)
    J11, J12, J22, Q11, Q12, Q22 = second_order
    variance = Q1 + delta * Q2
    j_lambda = (
        jnp.sqrt(2 / jnp.pi)
        * jnp.array([J11 + delta * J12, J12 + delta * J22])
        / jnp.sqrt(variance)
    )
    j_nu = jnp.sqrt(2 / jnp.pi) * jnp.asarray(teacher_moments) / jnp.sqrt(S)
    q_lambda = (
        jnp.sqrt(2 / jnp.pi)
        * jnp.array([Q11 + delta * Q12, Q12 + delta * Q22])
        / jnp.sqrt(variance)
    )
    q_nu = jnp.sqrt(2 / jnp.pi) * jnp.array([J1, J2]) / jnp.sqrt(S)
    trace = jnp.array([traces[0] + delta * traces[1], traces[1] + delta * traces[2]])
    if phase == "sft":
        p = Prob_correct_sft(*state, S, delta, epsilon)
        return (
            (j_nu * (1 - 2 * epsilon) - j_lambda) / 2,
            q_nu * (1 - 2 * epsilon) - q_lambda,
            trace * (1 - p) / T,
        )
    if phase != "rl":
        raise ValueError(f"Unknown phase {phase}")

    def all_correct(_):
        p = Prob_correct_sft(*state, S, delta, 0.0)
        return p**T, p ** (T - 1), jnp.zeros_like(p)

    reward, a, b = lax.cond(
        agreement_threshold == T,
        all_correct,
        lambda _: Prob_correct_rl(*state, S, delta, T, agreement_threshold)[1:],
        None,
    )
    return (
        (a + b) * j_lambda / 2 + (a - b) * j_nu / 2,
        (a + b) * q_lambda + (a - b) * q_nu,
        trace * reward / T,
    )


def compact_step(
    state,
    dt,
    delta,
    T,
    radius,
    S,
    teacher_moments,
    omega,
    traces,
    lr,
    use_optimal_lr,
    spherical,
    agreement_threshold,
    epsilon=0.0,
    phase="rl",
):
    """Canonical compact Euler step, shared by production and validation."""
    AJ, AQ, BQ = compact_coefficients(
        state,
        delta,
        T,
        S,
        teacher_moments,
        omega,
        traces,
        agreement_threshold,
        epsilon,
        phase,
    )
    dJ = DP_DJ(*state, S, delta)
    dQ1 = DP_DQ1(*state, S, delta)
    dQ2 = DP_DQ2(*state, S, delta)
    b, a = reprojection_opt_lr_coeffs(dJ, dQ1, dQ2, *AJ, *AQ, *BQ)
    eta = jnp.where(use_optimal_lr, _optimal_lr(a, b), lr)
    next_state = jnp.asarray(state) + dt * jnp.concatenate(
        (eta * AJ, eta * AQ + eta**2 * BQ)
    )
    constrained = jnp.asarray(_exact_sphere_reproject(*next_state, radius))
    next_state = jnp.where(spherical, constrained, next_state)
    return (*next_state, eta)
