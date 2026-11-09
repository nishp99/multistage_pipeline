"""Random matrix integrals for the pretrained projector. Internal rho is the noise fraction."""

from __future__ import annotations

import math
import time
from functools import lru_cache
from itertools import combinations, product
from typing import Dict

import numpy as np
import jax.numpy as jnp
from jax import jit, lax, vmap

# Compatibility shim for older JAX versions (used in find_window)
_trapezoid = getattr(jnp, "trapezoid", getattr(jnp, "trapz", None))

# ---------------------------------------------------------------------------
# Gauss-Legendre helper
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _legendre_rule(n):
    """Identical GL rule via a tridiagonal solver, cached across JIT traces."""
    from scipy.special import roots_legendre

    nodes, weights = roots_legendre(n)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _gl_arrays(n: int, a: float, b: float):
    """
    GL nodes and weights on [a, b] as JAX float64 arrays.
    Computed once with numpy (outside JIT) and returned as static JAX arrays.
    """
    nodes_np, weights_np = _legendre_rule(n)
    # Map [-1, 1] -> [a, b]
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    nodes = jnp.array(scale * nodes_np + shift, dtype=jnp.float64)
    weights = jnp.array(scale * weights_np, dtype=jnp.float64)
    return nodes, weights


# Sign combinations for the 3-point term: (s1_idx, s2_idx, s3_idx, parity)
_SIGN_COMBOS = []
for _s1 in range(2):
    for _s2 in range(2):
        for _s3 in range(2):
            _SIGN_COMBOS.append((_s1, _s2, _s3, (-1) ** (_s1 + _s2 + _s3)))


# ---------------------------------------------------------------------------
# One-point resolvent (unchanged from v1)
# ---------------------------------------------------------------------------


def _cubic_coeffs(z, gamma, rho, Delta):
    return jnp.array(
        [
            Delta,
            -(z * (1.0 + Delta) + Delta * (1.0 - gamma)),
            z * (z + 1.0 + Delta - gamma * ((1.0 - rho) + rho * Delta)),
            -(z * z),
        ],
        dtype=jnp.complex128,
    )


def _roots_cubic(coeffs):
    c = coeffs / coeffs[0]
    companion = jnp.array(
        [
            [0, 0, -c[3]],
            [1, 0, -c[2]],
            [0, 1, -c[1]],
        ],
        dtype=jnp.complex128,
    )
    return jnp.linalg.eigvals(companion)


def _choose_root_single(z, gamma, rho, Delta):
    coeffs = _cubic_coeffs(z, gamma, rho, Delta)
    roots = _roots_cubic(coeffs)
    sign = jnp.where(z.imag >= 0.0, 1.0, -1.0)
    g1 = 1.0 / (roots - z)
    g2 = 1.0 / (Delta * roots - z)
    m = (1.0 - rho) * g1 + rho * g2
    scores = sign * m.imag
    return roots[jnp.argmax(scores)]


def _track_boundary_branch(xs, gamma, rho, Delta, eta=1e-10):
    n = xs.shape[0]
    zs = xs.astype(jnp.complex128) + 1j * eta
    xis = jnp.zeros(n, dtype=jnp.complex128)

    z0 = zs[n - 1]
    roots0 = _roots_cubic(_cubic_coeffs(z0, gamma, rho, Delta))
    target_m = -1.0 / z0
    g1_init = 1.0 / (roots0 - z0)
    g2_init = 1.0 / (Delta * roots0 - z0)
    m_init = (1.0 - rho) * g1_init + rho * g2_init
    err = jnp.abs(m_init - target_m)
    xi_last = roots0[jnp.argmin(err)]
    xis = xis.at[n - 1].set(xi_last)

    def body(k_fwd, carry):
        xis, prev = carry
        k = n - 2 - k_fwd
        z = zs[k]
        roots = _roots_cubic(_cubic_coeffs(z, gamma, rho, Delta))

        g1 = 1.0 / (roots - z)
        g2 = 1.0 / (Delta * roots - z)
        m = (1.0 - rho) * g1 + rho * g2
        ok = m.imag > 0.0

        dist = jnp.abs(roots - prev)
        good_dist = jnp.where(ok, dist, jnp.inf)
        any_good = jnp.any(ok)
        chosen_dist = jnp.where(any_good, good_dist, dist)
        best_xi = roots[jnp.argmin(chosen_dist)]

        xis = xis.at[k].set(best_xi)
        return xis, best_xi

    xis, _ = lax.fori_loop(0, n - 1, body, (xis, xi_last))
    return xis


def _one_point_transforms(z, gamma, rho, Delta):
    xi = _choose_root_single(z, gamma, rho, Delta)
    g1 = 1.0 / (xi - z)
    g2 = 1.0 / (Delta * xi - z)
    m = (1.0 - rho) * g1 + rho * g2
    return xi, g1, g2, m


def one_point_densities(x, gamma, rho, Delta, eta=1e-10):
    z = x + 1j * eta
    _, g1, _, m = _one_point_transforms(z, gamma, rho, Delta)
    rho_global = jnp.maximum(0.0, m.imag / math.pi)
    rho_M = jnp.maximum(0.0, g1.imag / math.pi)
    return rho_global, rho_M


# ---------------------------------------------------------------------------
# Bisection (unchanged)
# ---------------------------------------------------------------------------


def _bisect(f, a, b, n_iter=60):
    f_a = f(a)

    def body(_, carry):
        lo, f_lo, hi = carry
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        go_right = f_mid * f_lo > 0
        lo = jnp.where(go_right, mid, lo)
        f_lo = jnp.where(go_right, f_mid, f_lo)
        hi = jnp.where(go_right, hi, mid)
        return lo, f_lo, hi

    lo, _, hi = lax.fori_loop(0, n_iter, body, (a, f_a, b))
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# find_window — smooth spectral-window cut (see module docstring, fix #1)
# ---------------------------------------------------------------------------

# Density threshold used to detect the signal-bump lower edge.  Must sit above
# the spectral-gap floor (~1e-9 set by eta=1e-10) and well below the bump peak
# (~O(0.1..10)); 1e-6 satisfies both with ~1000x margin.
_GAP_THR = 1e-6

# Relative tolerance for the SMOOTH gap criterion (v6 fix; see find_window).
# A true spectral gap exists iff the signal bump alone carries ~ target mass
# (1-rho), i.e. the mass above the smooth signal-bump lower edge is within this
# relative tolerance of the target.  Measured separation (rho=0.8, Delta=0.1):
# the gap regime has mass_above_signal = target to ~1e-4, while the merged
# (no-gap, small-alpha) regime overshoots by >=50% of target.  0.1 sits ~1000x
# above the gap residual and well below the no-gap overshoot.
_GAP_MASS_RTOL = 0.1


def find_window(alpha, rho, Delta, scan_points=2000):
    gamma = 1.0 / alpha
    target = 1.0 - rho

    scan_max = (1.0 + jnp.sqrt(gamma)) ** 2 * jnp.maximum(1.0, Delta) + 2.0
    scan_xs = jnp.linspace(1e-8, scan_max, scan_points)

    def _rho_at_x(x):
        z = x + 1j * 1e-10
        _, g1, _, m = _one_point_transforms(z, gamma, rho, Delta)
        return jnp.maximum(0.0, m.imag / math.pi)

    def _rho_M_at_x(x):
        z = x + 1j * 1e-10
        _, g1, _, m = _one_point_transforms(z, gamma, rho, Delta)
        return jnp.maximum(0.0, g1.imag / math.pi)

    scan_rho = vmap(_rho_at_x)(scan_xs)

    active = scan_rho > 1e-8
    idx_max = len(scan_xs) - 1 - jnp.argmax(active[::-1])
    lambda_min = 1e-10
    # lambda_max is padded above the spectral edge (used for bisection safety)
    lambda_max = scan_xs[idx_max] + 0.05

    # With strictly positive population eigenvalues, rank fixes the continuous
    # mass exactly. Integrating from ~0 spuriously includes the regularized
    # zero atom when alpha<1 and can report a mass much larger than one.
    mass_cont = jnp.minimum(alpha, 1.0)

    # --- Mass-based cut (used as fallback when there is no spectral gap) --- #
    def tail_mass_minus_target(cut):
        tail_xs = jnp.linspace(cut, lambda_max, 500)
        tail_rho = vmap(_rho_at_x)(tail_xs)
        return _trapezoid(tail_rho, tail_xs) - target

    lambda_cut_mass = _bisect(tail_mass_minus_target, lambda_min, lambda_max, n_iter=60)

    # Bisect for the true upper spectral edge (where density drops to zero).
    # This is used as the right endpoint for GL integration so that the sqrt
    # singularity at the spectral edge falls exactly at the boundary.
    def rho_minus_threshold(x):
        return _rho_at_x(x) - 1e-8

    lambda_edge = _bisect(
        rho_minus_threshold, scan_xs[idx_max - 1], lambda_max, n_iter=60
    )

    # --- Smooth cut: signal-bump lower edge (fix #1) ---------------------- #
    # The selected eigenvectors are the top (1-rho)D PCA directions = the top
    # contiguous high-density block ("signal bump").  Its lower edge is a smooth
    # function of alpha.  Find the highest scan node below idx_max whose density
    # is <= _GAP_THR; in a true gap this is the top of the gap, and the bump foot
    # lies in the next cell, so bisecting the density crossing there gives the
    # smooth signal-bump lower edge.
    above_thr = scan_rho > _GAP_THR
    idx = jnp.arange(scan_points)
    inactive_below = (~above_thr) & (idx < idx_max)
    k_false = jnp.max(jnp.where(inactive_below, idx, -1))
    k_safe = jnp.maximum(k_false, 0)
    a0 = scan_xs[k_safe]
    b0 = scan_xs[k_safe + 1]

    def rho_minus_gapthr(x):
        return _rho_at_x(x) - _GAP_THR

    lambda_signal_lower = _bisect(rho_minus_gapthr, a0, b0, n_iter=60)

    # Use the smooth signal-edge cut only when a TRUE spectral gap exists.
    #
    # v6 fix: the gap test must be SMOOTH in alpha.  v5 tested the density at the
    # mass-based cut (rho_at_mass_cut < _GAP_THR), but lambda_cut_mass is itself
    # DEGENERATE inside the gap (any cut there gives tail mass = target), so the
    # bisection lands at a random point in the gap and rho_at_mass_cut flickers
    # across _GAP_THR -> has_gap flips False at scattered high-alpha points ->
    # lambda_cut jumps to the (low, jittery) mass cut -> u_max=sqrt(lam_edge-
    # lam_cut) redistributes every GL node -> ~1e-7 jitter in t_1/T_11 ->
    # amplified ~1e4x by the catastrophic-cancellation Delta_G=T_11*t-t_1^2
    # (~1e-6 at high alpha) -> zigzag in every Omega coefficient.
    #
    # Robust criterion: a true gap exists iff the signal bump ALONE carries ~the
    # target mass (1-rho), i.e. the mass above the SMOOTH signal-bump lower edge
    # equals the target to within _GAP_MASS_RTOL.  When the bumps are merged
    # (small alpha) the signal-edge sits at the bulk foot and the mass above it
    # overshoots the target by >=50%, so the well-defined mass cut is used (this
    # reproduces v2 at small alpha).  mass_above_signal is smooth (lambda_signal_
    # lower is smooth), so the gap flag no longer flickers.
    mass_above_signal = tail_mass_minus_target(lambda_signal_lower) + target
    # v7: mass alone cannot distinguish a true INTERNAL gap from the empty
    # interval below a connected positive bulk. When alpha is just above the
    # PCA rank fraction, that whole bulk has mass alpha, within the old 10%
    # tolerance of the target. Require an actual lower spectral bulk as well.
    # Exclude the first scan node: for alpha<1 it resolves the zero atom,
    # not a lower positive bulk.
    has_lower_bulk = jnp.any(above_thr & (idx > 0) & (idx < k_false))
    has_gap = (
        has_lower_bulk
        & (k_false >= 0)
        & (mass_above_signal < target * (1.0 + _GAP_MASS_RTOL))
    )
    lambda_cut = jnp.where(has_gap, lambda_signal_lower, lambda_cut_mass)

    # overlap_mass here is only used by v1; v2+ recompute it from the GL grid
    overlap_xs = jnp.linspace(lambda_cut, lambda_max, 500)
    overlap_rho_M = vmap(_rho_M_at_x)(overlap_xs)
    overlap_mass = _trapezoid(overlap_rho_M, overlap_xs)

    return {
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "lambda_edge": lambda_edge,
        "lambda_cut": lambda_cut,
        "lambda_cut_mass": lambda_cut_mass,
        "has_gap": has_gap,
        "mass_continuous": mass_cont,
        "overlap_mass": overlap_mass,
    }


# ---------------------------------------------------------------------------
# Boundary data
# ---------------------------------------------------------------------------


def _boundary_data(xs, gamma, rho, Delta, eta_boundary=1e-10):
    # xs must be sorted ascending for branch tracking
    # Use vmap(_choose_root_single) so every node is evaluated independently.
    # This avoids the branch-tracking requirement that xs be densely and
    # uniformly spaced — essential when xs comes from a trig-substituted GL
    # grid whose spacing is non-uniform.
    zs_plus = xs.astype(jnp.complex128) + 1j * eta_boundary

    def _eval_upper(z):
        xi = _choose_root_single(z, gamma, rho, Delta)
        g1 = 1.0 / (xi - z)
        g2 = 1.0 / (Delta * xi - z)
        return xi, g1, g2

    xi_plus, g1_plus, g2_plus = vmap(_eval_upper)(zs_plus)

    xi_minus = jnp.conjugate(xi_plus)
    g1_minus = jnp.conjugate(g1_plus)
    g2_minus = jnp.conjugate(g2_plus)

    m_plus = (1.0 - rho) * g1_plus + rho * g2_plus

    return {
        "xi_plus": xi_plus,
        "xi_minus": xi_minus,
        "g1_plus": g1_plus,
        "g2_plus": g2_plus,
        "g1_minus": g1_minus,
        "g2_minus": g2_minus,
        "rho_global": m_plus.imag / math.pi,
        "rho_M": g1_plus.imag / math.pi,
    }


def _precompute_sign_arrays(xs, boundary, eta_boundary):
    xi = jnp.stack([boundary["xi_plus"], boundary["xi_minus"]])
    g1 = jnp.stack([boundary["g1_plus"], boundary["g1_minus"]])
    g2 = jnp.stack([boundary["g2_plus"], boundary["g2_minus"]])
    zeta_plus = jnp.sqrt(xs + 1j * eta_boundary)
    zeta_minus = jnp.sqrt(xs - 1j * eta_boundary)
    zeta = jnp.stack([zeta_plus, zeta_minus])
    a1 = zeta * g1
    a2 = zeta * g2
    b = -xi / zeta
    return {"xi": xi, "g1": g1, "g2": g2, "zeta": zeta, "a1": a1, "a2": a2, "b": b}


def _precompute_pair_dressings(sign_arrays, gamma, rho, Delta):
    a1 = sign_arrays["a1"]
    a2 = sign_arrays["a2"]
    b = sign_arrays["b"]

    alpha1 = jnp.einsum("si,tj->stij", a1, a1)
    alpha2 = jnp.einsum("si,tj->stij", a2, a2)
    beta = jnp.einsum("si,tj->stij", b, b)

    den = 1.0 - gamma * beta * ((1.0 - rho) * alpha1 + rho * (Delta**2) * alpha2)

    V3 = gamma * beta * (1.0 - rho) / den
    V1 = 1.0 + alpha1 * V3
    V2 = Delta * alpha2 * V3

    return {
        "V1": V1,
        "V2": V2,
        "V3": V3,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "beta": beta,
        "den": den,
    }


# ---------------------------------------------------------------------------
# Two-point kernels  (GL-integrated)
# ---------------------------------------------------------------------------


def _two_point_boundary_kernels(xs, gamma, rho, Delta, boundary):
    xi_p = boundary["xi_plus"]
    xi_m = boundary["xi_minus"]
    g1_p = boundary["g1_plus"]
    g2_p = boundary["g2_plus"]
    g1_m = boundary["g1_minus"]
    g2_m = boundary["g2_minus"]
    rho_global = boundary["rho_global"]
    rho_M = boundary["rho_M"]

    diag_density = (1.0 - rho) ** 2 * (rho_M**2) / rho_global

    xi1 = xi_p[:, None]
    g11_p = g1_p[:, None]
    g21_p = g2_p[:, None]

    xi2_pp = xi_p[None, :]
    g12_pp = g1_p[None, :]
    g22_pp = g2_p[None, :]

    xi2_pm = xi_m[None, :]
    g12_pm = g1_m[None, :]
    g22_pm = g2_m[None, :]

    kappa_pp = (
        gamma
        * xi1
        * xi2_pp
        * ((1.0 - rho) * g11_p * g12_pp + rho * (Delta**2) * g21_p * g22_pp)
    )
    G_pp = (1.0 - rho) * (
        g11_p * g12_pp
        + gamma
        * (1.0 - rho)
        * xi1
        * xi2_pp
        * (g11_p**2)
        * (g12_pp**2)
        / (1.0 - kappa_pp)
    )

    kappa_pm = (
        gamma
        * xi1
        * xi2_pm
        * ((1.0 - rho) * g11_p * g12_pm + rho * (Delta**2) * g21_p * g22_pm)
    )
    G_pm = (1.0 - rho) * (
        g11_p * g12_pm
        + gamma
        * (1.0 - rho)
        * xi1
        * xi2_pm
        * (g11_p**2)
        * (g12_pm**2)
        / (1.0 - kappa_pm)
    )

    k_reg = -(jnp.real(G_pp) - jnp.real(G_pm)) / (2.0 * math.pi**2)

    # Fill diagonal singularity from neighbors
    n = xs.shape[0]
    idx = jnp.arange(n)

    def _fill_one(i):
        v1 = k_reg[i, i - 1]
        v2 = k_reg[i, i + 1]
        v3 = k_reg[i, i - 2]
        v4 = k_reg[i, i + 2]
        w1 = jnp.where(i > 0, 1.0, 0.0)
        w2 = jnp.where(i + 1 < n, 1.0, 0.0)
        w3 = jnp.where(i > 1, 1.0, 0.0)
        w4 = jnp.where(i + 2 < n, 1.0, 0.0)
        total_w = jnp.maximum(w1 + w2 + w3 + w4, 1.0)
        return (v1 * w1 + v2 * w2 + v3 * w3 + v4 * w4) / total_w

    diag_vals = vmap(_fill_one)(idx)
    k_reg = k_reg.at[idx, idx].set(diag_vals)

    return diag_density, k_reg


# ---------------------------------------------------------------------------
# Three-point distinct term  (GL weights passed in)
# ---------------------------------------------------------------------------


def _three_point_distinct_term(
    xs, weights3, sign_arrays, pair_data, gamma, rho, Delta, tube_cells=1
):
    """
    Same as v1 but uses GL weights instead of trapezoid weights * dx.
    The contribution becomes:  sum_j w3[j] * sum_{i,k} w3[i]*w3[k] * mask * K(i,j,k)
    """
    n = xs.shape[0]
    idx = jnp.arange(n)
    ik_dist = jnp.abs(idx[:, None] - idx[None, :])
    # Outer product of GL weights for i,k axes
    ww = weights3[:, None] * weights3[None, :]
    pref = 1.0 / ((2.0 * jnp.pi * 1j) ** 3)

    sa_a1 = sign_arrays["a1"]
    sa_a2 = sign_arrays["a2"]
    sa_b = sign_arrays["b"]
    sa_zeta = sign_arrays["zeta"]
    pd_V1 = pair_data["V1"]
    pd_V2 = pair_data["V2"]
    pd_V3 = pair_data["V3"]
    pd_alpha1 = pair_data["alpha1"]
    pd_beta = pair_data["beta"]
    pd_den = pair_data["den"]

    def body_j(j, total):
        comb = jnp.zeros((n, n), dtype=jnp.complex128)

        for s1, s2, s3, parity in _SIGN_COMBOS:
            A1_i = sa_a1[s1, :, None]
            A2_i = sa_a2[s1, :, None]
            B_i = sa_b[s1, :, None]

            A1_j = sa_a1[s2, j]
            A2_j = sa_a2[s2, j]
            B_j = sa_b[s2, j]

            A1_k = sa_a1[s3, None, :]
            A2_k = sa_a2[s3, None, :]
            B_k = sa_b[s3, None, :]

            zeta_prod = sa_zeta[s1, :, None] * sa_zeta[s2, j] * sa_zeta[s3, None, :]

            V1_12 = pd_V1[s1, s2, :, j][:, None]
            V2_12 = pd_V2[s1, s2, :, j][:, None]
            V3_12 = pd_V3[s1, s2, :, j][:, None]

            V1_23 = pd_V1[s2, s3, j, :][None, :]
            V2_23 = pd_V2[s2, s3, j, :][None, :]
            V3_23 = pd_V3[s2, s3, j, :][None, :]

            s1_src = A1_i * V1_12 * A1_j * V1_23 * A1_k
            s2_src = A2_i * V2_12 * A2_j * V2_23 * A2_k
            s3_src = B_i * V3_12 * B_j * V3_23 * B_k

            alpha1_13 = pd_alpha1[s1, s3]
            beta_13 = pd_beta[s1, s3]
            den_13 = pd_den[s1, s3]

            y3 = (
                s3_src + gamma * beta_13 * ((1.0 - rho) * s1_src + rho * Delta * s2_src)
            ) / den_13
            y1 = s1_src + alpha1_13 * y3

            G123 = (1.0 - rho) * y1 / zeta_prod
            comb = comb + parity * G123

        K = jnp.real(pref * comb)

        close_j = jnp.abs(idx - j) <= tube_cells
        mask = (~close_j[:, None]) & (~close_j[None, :]) & (ik_dist > tube_cells)

        contrib = jnp.sum(jnp.where(mask, K * ww, 0.0))
        return total + weights3[j] * contrib

    total = lax.fori_loop(0, n, body_j, 0.0)
    return total


# ---------------------------------------------------------------------------
# Utility: overlap structures
# ---------------------------------------------------------------------------


def block_coefficients_from_moments(m1_over_K, m2_over_K, rho) -> Dict[str, float]:
    q_signal = m1_over_K
    q_noise = (1.0 - rho) * (1.0 - m1_over_K) / rho
    qmq_signal = m2_over_K
    qmq_noise = (1.0 - rho) * (m1_over_K - m2_over_K) / rho

    return {
        "q_signal": q_signal,
        "q_noise": q_noise,
        "qmq_signal": qmq_signal,
        "qmq_noise": qmq_noise,
    }


def vector_vector_overlaps_from_moments(
    signal_overlap_over_D, noise_overlap_over_D, tr_qm_over_K, tr_qmqm_over_K, rho
):
    coeffs = block_coefficients_from_moments(tr_qm_over_K, tr_qmqm_over_K, rho)

    u_Q_v_over_D = (
        signal_overlap_over_D * coeffs["q_signal"]
        + noise_overlap_over_D * coeffs["q_noise"]
    )
    u_QMQ_v_over_D = (
        signal_overlap_over_D * coeffs["qmq_signal"]
        + noise_overlap_over_D * coeffs["qmq_noise"]
    )

    return u_Q_v_over_D, u_QMQ_v_over_D


# ---------------------------------------------------------------------------
# compute_traces  — main entry point
# ---------------------------------------------------------------------------


def lower_order_traces(
    alpha,
    rho,
    Delta,
    n_grid1d=5000,
    n_grid2=400,
    n_grid3=150,
    n_grid3_check=None,
    eta_boundary=1e-10,
    tube_cells=1,
    distinct_convergence_rtol=5e-4,
    distinct_convergence_width=2.0,
    compute_third=True,
):
    """
    Return the first three ambient-dimension-normalized trace moments.

    All integrals run on the true spectral window [lambda_cut, lambda_edge],
    with lambda_cut the smooth signal-bump lower edge:

    1D integrals (tr_qm, tr_qmqm_diag, tr_qmqmqm_diag) and the 3D distinct term
    use the substitution u = sqrt(lambda_edge - x), which removes the sqrt
    singularity at the spectral edge → exponential GL convergence.

    2D integrals (k2_reg kernel) use a linear GL grid; the integrand
    w_fn(x)*k2_reg(x,y) is smooth (the sqrt edge cancels in w_fn) so it
    converges fast on a linear grid.

    Default grids:
      n_grid1d = 5000  — GL for 1D integrals (max On11 jump ~0.04 at alpha>3)
      n_grid2  = 400   — GL for 2D k2_reg kernel (converges fast)
      n_grid3  = 150   — sqrt-substituted GL for the 3D distinct term
    Increase n_grid1d to 10000 for ~3x smaller errors (~0.5s per call).
    """
    gamma = 1.0 / alpha
    window = find_window(alpha, rho, Delta)

    lam_cut = window["lambda_cut"]
    lam_max = window["lambda_max"]
    lam_edge = window["lambda_edge"]  # true upper spectral edge (density → 0 here)

    # ------------------------------------------------------------------ #
    # Helper: linear GL grid on [lo, hi].
    # ------------------------------------------------------------------ #
    def _make_gl_grid(n, lo, hi):
        ref_nodes, ref_wts = _legendre_rule(n)
        ref_nodes = jnp.array(ref_nodes, dtype=jnp.float64)
        ref_wts = jnp.array(ref_wts, dtype=jnp.float64)
        scale = (hi - lo) / 2.0
        shift = (hi + lo) / 2.0
        return scale * ref_nodes + shift, scale * ref_wts

    # ------------------------------------------------------------------ #
    # Substitution u = sqrt(lam_edge - x) on [lam_cut, lam_edge].
    # rho_M ~ C*sqrt(lam_edge - x) at the right endpoint (spectral edge),
    # so rho_M(lam_edge - u^2) ~ C*u → smooth after multiplication by 2u dx.
    # The integrand in u-space has NO singularity → exponential GL convergence.
    # ------------------------------------------------------------------ #
    def _make_gl_grid_1d(n):
        ref_nodes, ref_wts = _legendre_rule(n)
        ref_nodes = jnp.array(ref_nodes, dtype=jnp.float64)
        ref_wts = jnp.array(ref_wts, dtype=jnp.float64)
        u_max = jnp.sqrt(lam_edge - lam_cut)
        u_half = u_max / 2.0
        u_nodes = u_half * (1.0 + ref_nodes)  # u ∈ [0, u_max]
        u_wts = u_half * ref_wts
        xs = lam_edge - u_nodes**2  # x = lam_edge - u²
        ws = 2.0 * u_nodes * u_wts  # Jacobian × GL weight
        return xs, ws

    # 1D and 3D integrals use the sqrt-substituted grid on [lam_cut, lam_edge]
    # (exponential GL convergence, no node-redistribution noise from lam_max).
    # 2D uses a linear grid on [lam_cut, lam_edge]; its integrand w_fn*k2_reg is
    # smooth (the sqrt edge cancels in w_fn) so a linear grid converges fast.
    if n_grid3_check is None:
        n_grid3_check = max(40, n_grid3 // 2)

    xs1d, w1d = _make_gl_grid_1d(n_grid1d)
    xs2d, w2d = _make_gl_grid(n_grid2, lam_cut, lam_edge)
    xs3d, w3d = _make_gl_grid_1d(n_grid3)
    xs3d_check, w3d_check = _make_gl_grid_1d(n_grid3_check)  # coarse convergence check

    # ------------------------------------------------------------------ #
    # Fine-grid boundary data for all 1D integrals
    # ------------------------------------------------------------------ #
    bnd1d = _boundary_data(xs1d, gamma, rho, Delta, eta_boundary=eta_boundary)

    rho_global_1d = bnd1d["rho_global"]
    rho_M_1d = bnd1d["rho_M"]
    # avoid division by zero at spectral-density zeros
    rho_global_1d_safe = jnp.where(rho_global_1d > 0, rho_global_1d, 1.0)
    w_fn_1d = (1.0 - rho) * rho_M_1d / rho_global_1d_safe

    tr_qm_over_K = jnp.sum(w1d * rho_M_1d)
    tr_qm_over_D = (1.0 - rho) * tr_qm_over_K

    diag_density_1d = (
        (1.0 - rho) ** 2
        * (rho_M_1d**2)
        / jnp.where(rho_global_1d > 0, rho_global_1d, 1.0)
    )
    tr_qmqm_diag_over_D = jnp.sum(w1d * diag_density_1d)

    tr_qmqmqm_diag_over_D = jnp.sum(w1d * rho_global_1d * (w_fn_1d**3))

    # ------------------------------------------------------------------ #
    # Coarse-grid 2D kernel (k2_reg) and associated integrals
    # k2_reg is smooth (diagonal delta removed analytically) so the coarse
    # GL grid gives exponentially small errors (~1e-8 at n=400).
    # ------------------------------------------------------------------ #
    bnd2d = _boundary_data(xs2d, gamma, rho, Delta, eta_boundary=eta_boundary)
    _, k2_reg = _two_point_boundary_kernels(xs2d, gamma, rho, Delta, bnd2d)

    rho_global_2d = bnd2d["rho_global"]
    rho_M_2d = bnd2d["rho_M"]
    rho_global_2d_safe = jnp.where(rho_global_2d > 0, rho_global_2d, 1.0)
    w_fn_2d = (1.0 - rho) * rho_M_2d / rho_global_2d_safe

    tr_qmqm_off_over_D = jnp.einsum("i,ij,j->", w2d, k2_reg, w2d)
    tr_qmqmqm_pair_over_D = 3.0 * jnp.einsum("i,i,ij,j->", w2d, w_fn_2d, k2_reg, w2d)

    tr_qmqm_over_D = tr_qmqm_diag_over_D + tr_qmqm_off_over_D
    tr_qmqm_over_K = tr_qmqm_over_D / (1.0 - rho)

    # ------------------------------------------------------------------ #
    # Three-point distinct term on the sqrt-substituted GL grid, with a
    # convergence gate ported from trace_expectations_new (the MC-verified
    # small-alpha behaviour).  The raw distinct integral is unreliable at small
    # alpha (large near-pole cancellations on the tube-masked grid -> a spurious
    # ~ -0.025 contribution at alpha~0.25 that even drives tau_111 negative,
    # though tau_111 = sum cos^6(theta_i) >= 0).  Evaluating it on the main grid
    # AND a coarser check grid and down-weighting by a smoothstep of their
    # disagreement gates this garbage toward zero at small alpha (-> tau ~
    # diag+pair, matching Monte-Carlo), while at high alpha the tiny smooth
    # distinct converges so distinct_weight ~ 1 (saturated, flat) and the
    # find_window high-alpha fix / Omega smoothness are untouched.
    # ------------------------------------------------------------------ #
    def _distinct_for_grid(xs, ws):
        bnd3d = _boundary_data(xs, gamma, rho, Delta, eta_boundary=eta_boundary)
        sign_arrays = _precompute_sign_arrays(xs, bnd3d, eta_boundary)
        pair_data = _precompute_pair_dressings(sign_arrays, gamma, rho, Delta)
        return _three_point_distinct_term(
            xs,
            ws,
            sign_arrays,
            pair_data,
            gamma,
            rho,
            Delta,
            tube_cells=tube_cells,
        )

    if compute_third:
        tr_qmqmqm_distinct_raw_over_D = _distinct_for_grid(xs3d, w3d)
        tr_qmqmqm_distinct_check_over_D = _distinct_for_grid(xs3d_check, w3d_check)

        tau_without_distinct = tr_qmqmqm_diag_over_D + tr_qmqmqm_pair_over_D
        distinct_error = jnp.abs(
            tr_qmqmqm_distinct_raw_over_D - tr_qmqmqm_distinct_check_over_D
        )
        distinct_scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(tau_without_distinct), jnp.abs(tr_qmqmqm_distinct_raw_over_D)
            ),
            1e-12,
        )
        tolerance = distinct_convergence_rtol * distinct_scale
        width = jnp.maximum(jnp.asarray(distinct_convergence_width), 1.0)
        reg_x = (width * tolerance - distinct_error) / (
            (width - 1.0) * tolerance + 1e-30
        )
        reg_x = jnp.clip(reg_x, 0.0, 1.0)
        distinct_weight = reg_x * reg_x * (3.0 - 2.0 * reg_x)
        tr_qmqmqm_distinct_over_D = distinct_weight * tr_qmqmqm_distinct_raw_over_D

        tr_qmqmqm_over_D = (
            tr_qmqmqm_diag_over_D + tr_qmqmqm_pair_over_D + tr_qmqmqm_distinct_over_D
        )
    else:
        # The recursive path needs only m1/m2 and the first two moments.
        tr_qmqmqm_over_D = jnp.nan
    return tr_qm_over_D, tr_qmqm_over_D, tr_qmqmqm_over_D


lower_order_traces = jit(
    lower_order_traces,
    static_argnames=(
        "n_grid1d",
        "n_grid2",
        "n_grid3",
        "n_grid3_check",
        "tube_cells",
        "compute_third",
    ),
)


# ---------------------------------------------------------------------------
# Branch-free chains. Last axes are spectral position and block (s,n,0).
# ---------------------------------------------------------------------------
def _validate_parameters(alpha, rho, Delta):
    if not (
        np.isfinite(alpha)
        and 0 < rho < 1
        and 0 < Delta < 1
        and alpha >= 1 - rho - 1e-14
    ):
        raise ValueError("Require 0<rho<1, 0<Delta<1 and alpha>=1-rho.")


def _pair_factors(qi, qj, gamma, rho, Delta):
    ss = qi[..., 0] * qj[..., 0]
    nn = qi[..., 1] * qj[..., 1]
    xx = gamma * qi[..., 2] * qj[..., 2]
    den = 1 - xx * ((1 - rho) * ss + rho * Delta**2 * nn)
    return ss, nn, xx, den


def _dress_factors(factors, source, rho, Delta):
    ss, nn, xx, den = factors
    u0 = (
        source[..., 2]
        + xx * ((1 - rho) * source[..., 0] + rho * Delta * source[..., 1])
    ) / den
    return jnp.stack(
        (source[..., 0] + ss * u0, source[..., 1] + Delta * nn * u0, u0), axis=-1
    )


def pair_dress(qi, qj, source, gamma, rho, Delta):
    """Apply the inverse pair stability operator; q=(gs,gn,-xi)."""
    return _dress_factors(_pair_factors(qi, qj, gamma, rho, Delta), source, rho, Delta)


def _chain_prefixes(q, gamma, rho, Delta):
    """All prefix chains in one pass, reusing contiguous subchains."""
    n = q.shape[-2]
    chains = {}
    for span in range(1, n):
        for i in range(n - span):
            j = i + span
            factors = _pair_factors(q[..., i, :], q[..., j, :], gamma, rho, Delta)
            if span == 1:
                ss = factors[0]
                source = jnp.stack((ss, jnp.zeros_like(ss), jnp.zeros_like(ss)), -1)
            else:
                source = jnp.zeros_like(q[..., 0, :])
                for count in range(1, span):
                    for cuts in combinations(range(i + 1, j), count):
                        points = (i,) + cuts + (j,)
                        term = chains[points[0], points[1]]
                        for a, b in zip(points[1:-1], points[2:]):
                            term = term * chains[a, b]
                        for cut in cuts:
                            term = term / q[..., cut, :]
                        source = source + (-1) ** (count + 1) * term
            chains[i, j] = _dress_factors(factors, source, rho, Delta)
    return jnp.stack(
        [(1 - rho) * q[..., 0, 0]]
        + [(1 - rho) * chains[0, j][..., 0] for j in range(1, n)],
        -1,
    )


@jit
def _q_from_z(z, gamma, rho, Delta):
    shape = z.shape
    xi = vmap(_choose_root_single, in_axes=(0, None, None, None))(
        z.reshape(-1), gamma, rho, Delta
    ).reshape(shape)
    return jnp.stack((1 / (xi - z), 1 / (Delta * xi - z), -xi), -1)


@jit
def resolvent_chain(zs, alpha, rho, Delta):
    """Normalized signal-inserted resolvent chain; zs has last axis length k (1 <= k <= 5)."""
    zs = jnp.asarray(zs, dtype=jnp.complex128)
    if zs.ndim < 1 or not 1 <= zs.shape[-1] <= 5:
        raise ValueError("The last zs axis must contain 1 to 5 spectral parameters.")
    q = _q_from_z(jnp.asarray(zs, dtype=jnp.complex128), 1 / alpha, rho, Delta)
    return _chain_prefixes(q, 1 / alpha, rho, Delta)[..., -1]


@jit
def _discontinuity_weighted(qplus, jacobian, gamma, rho, Delta):
    k = qplus.shape[-2]
    # Fix the first sign positive and add its conjugate partner analytically.
    signs = jnp.asarray([(1,) + s for s in product((1, -1), repeat=k - 1)])
    q = jnp.where(
        signs[None, ..., None] > 0, qplus[:, None, ...], jnp.conj(qplus[:, None, ...])
    )
    chains = _chain_prefixes(q, gamma, rho, Delta)
    signed_weights = jnp.where(
        signs[None, ...] > 0, jacobian[:, None, :], jnp.conj(jacobian[:, None, :])
    )
    chains = chains * jnp.cumprod(signed_weights, axis=-1)
    parity = jnp.cumprod(signs, axis=-1)
    powers = jnp.arange(1, k + 1)
    # Short prefixes repeat across unused signs; their mean removes duplication.
    return (
        jnp.mean(jnp.real(chains * parity / (2j * jnp.pi) ** powers), axis=1)
        * 2**powers
    )


@jit
def _discontinuity_from_q(qplus, gamma, rho, Delta):
    return _discontinuity_weighted(qplus, jnp.ones(qplus.shape[:-1]), gamma, rho, Delta)


def _contour(u, lo, hi, eta):
    # Cubic endpoint grading resolves the only near-singular regions. Deforming
    # the straight upper lip to this semicircle is exact for every eta>0.
    a, b = u**3, (1 - u) ** 3
    t = a / (a + b)
    dt = 3 * u**2 * (1 - u) ** 2 / (a + b) ** 2
    angle = np.pi * t
    radius = (hi - lo) / 2
    z = (lo + hi) / 2 - radius * np.cos(angle) + 1j * (eta + radius * np.sin(angle))
    dz = radius * np.pi * (np.sin(angle) + 1j * np.cos(angle)) * dt
    return z, dz


def discontinuity_integrand(xs, eta, alpha, rho, Delta):
    """Complete finite-eta integrand, returning orders 1..xs.shape[-1]."""
    xs = jnp.atleast_2d(xs)
    q = _q_from_z(xs + 1j * eta, 1 / alpha, rho, Delta)
    return _discontinuity_from_q(q, 1 / alpha, rho, Delta)


def _q_interpolator(lo, hi, eta, alpha, rho, Delta):
    """Interpolate xi, then Newton-polish at each actual integration point.

    Only one-dimensional roots are solved. The polishing residual is checked;
    interpolation is an initializer, not an approximation to the final kernel.
    """
    from scipy.interpolate import CubicSpline

    count = min(65537, max(257, int(np.ceil(16 * (hi - lo) / eta)) + 1))
    xs = np.linspace(lo, hi, count)
    q = np.asarray(_q_from_z(jnp.asarray(xs + 1j * eta), 1 / alpha, rho, Delta))
    return CubicSpline(xs, -q[:, 2])


@jit
def _polish_q(z, xi, gamma, rho, Delta):
    def step(_, xi):
        gs, gn = 1 / (xi - z), 1 / (Delta * xi - z)
        f = 1 / xi - 1 - gamma * ((1 - rho) * gs + rho * Delta * gn)
        df = -1 / xi**2 + gamma * ((1 - rho) * gs**2 + rho * Delta**2 * gn**2)
        return xi - f / df

    xi = lax.fori_loop(0, 4, step, xi)
    gs, gn = 1 / (xi - z), 1 / (Delta * xi - z)
    residual = jnp.max(
        jnp.abs(1 / xi - 1 - gamma * ((1 - rho) * gs + rho * Delta * gn))
        / (1 + jnp.abs(1 / xi))
    )
    physical = jnp.all((gs.imag > 0) & (gn.imag > 0))
    return jnp.stack((gs, gn, -xi), -1), residual, physical


def integrate_moments_qmc(
    alpha,
    rho,
    Delta,
    *,
    eta,
    power=16,
    scrambles=8,
    seed=0,
    batch_size=4096,
    max_order=5,
    window=None,
    method="contour",
):
    """Fixed-eta Sobol integral. Uncertainty is across independent scrambles.

    method='contour' deforms the upper straight segment into a graded upper
    semicircle with the same endpoints, and the lower segment by conjugation.
    Analyticity makes this exactly the same finite-eta integral, while moving
    the interior away from mixed-sign poles. method='real' is a reference.

    Prefix projections of the same Sobol sequence estimate every lower order.
    Returns raw replicate estimates so nested-power differences can be paired.
    """
    from scipy.stats import qmc

    _validate_parameters(alpha, rho, Delta)
    if not (
        eta > 0
        and np.isfinite(eta)
        and 1 <= max_order <= 5
        and power >= 0
        and scrambles >= 2
        and batch_size > 0
    ):
        raise ValueError("Invalid integration controls.")
    started = time.perf_counter()
    window = find_window(alpha, rho, Delta) if window is None else window
    lo, hi = float(window["lambda_cut"]), float(window["lambda_edge"])
    width = hi - lo
    if method == "real":
        interpolate = _q_interpolator(lo, hi, eta, alpha, rho, Delta)
    elif method == "contour":
        from scipy.interpolate import CubicSpline

        grid = np.linspace(0, 1, 4097)
        zs, _ = _contour(grid, lo, hi, eta)
        qs = np.asarray(_q_from_z(jnp.asarray(zs), 1 / alpha, rho, Delta))
        interpolate = CubicSpline(grid, -qs[:, 2])
    else:
        raise ValueError("method must be 'real' or 'contour'.")
    estimates = []
    max_residual = 0.0
    size = min(batch_size, 2**power)
    for scramble in range(scrambles):
        sampler = qmc.Sobol(max_order, scramble=True, seed=seed + scramble)
        total = np.zeros(max_order)
        for start in range(0, 2**power, size):
            count = min(size, 2**power - start)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                us = sampler.random(count)
            if count < size:
                us = np.pad(us, ((0, size - count), (0, 0)), mode="edge")
            if method == "real":
                xs = lo + width * us
                zs, jac = xs + 1j * eta, np.full_like(xs, width)
                initial = interpolate(xs)
            else:
                zs, jac = _contour(us, lo, hi, eta)
                initial = interpolate(us)
            q, residual, physical = _polish_q(
                jnp.asarray(zs), jnp.asarray(initial), 1 / alpha, rho, Delta
            )
            residual = float(residual)
            max_residual = max(max_residual, residual)
            if not bool(physical) or not np.isfinite(residual) or residual > 1e-10:
                raise ArithmeticError("Physical one-body branch failed residual check.")
            values = np.asarray(
                _discontinuity_weighted(q, jnp.asarray(jac), 1 / alpha, rho, Delta)
            )
            total += values[:count].sum(axis=0)
        estimates.append(total / 2**power)
    estimates = np.asarray(estimates)
    return dict(
        eta=float(eta),
        method=method,
        power=power,
        scrambles=scrambles,
        samples_per_scramble=2**power,
        estimates=estimates,
        covariance_mean=np.atleast_2d(np.cov(estimates, rowvar=False, ddof=1))
        / scrambles,
        moments=estimates.mean(0),
        sem=estimates.std(0, ddof=1) / np.sqrt(scrambles),
        max_root_residual=max_residual,
        seconds=time.perf_counter() - started,
    )


def integrate_moments_gl(
    alpha, rho, Delta, *, eta, nodes=40, max_order=3, batch_size=4096, window=None
):
    """Independent tensor Gauss-Legendre reference, streamed rather than stored."""
    _validate_parameters(alpha, rho, Delta)
    if not (eta > 0 and nodes >= 2 and 1 <= max_order <= 5):
        raise ValueError("Invalid GL controls.")
    window = find_window(alpha, rho, Delta) if window is None else window
    lo, hi = float(window["lambda_cut"]), float(window["lambda_edge"])
    x, w = _legendre_rule(nodes)
    x, w = lo + (x + 1) * (hi - lo) / 2, w * (hi - lo) / 2
    qnodes = _q_from_z(jnp.asarray(x + 1j * eta), 1 / alpha, rho, Delta)
    result = []
    for order in range(1, max_order + 1):
        total = 0.0
        for start in range(0, nodes**order, batch_size):
            flat = np.arange(start, min(start + batch_size, nodes**order))
            idx = np.stack(np.unravel_index(flat, (nodes,) * order), -1)
            values = np.asarray(
                _discontinuity_from_q(qnodes[idx], 1 / alpha, rho, Delta)
            )[:, -1]
            total += np.sum(values * np.prod(w[idx], axis=-1))
        result.append(total)
    return np.asarray(result)


def compute_moments_diagnostic(
    alpha,
    rho,
    Delta,
    *,
    eta_fractions=(0.001, 0.0003, 0.0001, 0.00003),
    powers=(14, 16, 18),
    scrambles=8,
    seed=0,
    batch_size=4096,
    rtol=0.01,
    atol=1e-4,
    max_order=5,
    method="contour",
):
    """Refine at fixed eta before comparing successive eta values.

    Eta is scaled by spectral-window width. A result is converged only if both
    three-SEM replicate uncertainty and two consecutive eta differences meet
    tolerance. No clipping, heuristic gating, or silent fallback is applied.
    """
    _validate_parameters(alpha, rho, Delta)
    if (
        len(eta_fractions) < 3
        or any(e <= 0 for e in eta_fractions)
        or any(a <= b for a, b in zip(eta_fractions, eta_fractions[1:]))
        or len(powers) < 2
        or any(a >= b for a, b in zip(powers, powers[1:]))
        or rtol < 0
        or atol <= 0
    ):
        raise ValueError("Use decreasing positive etas and increasing sample powers.")
    window = find_window(alpha, rho, Delta)
    width = float(window["lambda_edge"] - window["lambda_cut"])
    history, eta_results = [], []
    for fraction in eta_fractions:
        previous = None
        for power in powers:
            row = integrate_moments_qmc(
                alpha,
                rho,
                Delta,
                eta=fraction * width,
                power=power,
                scrambles=scrambles,
                seed=seed,
                batch_size=batch_size,
                max_order=max_order,
                window=window,
                method=method,
            )
            tol = np.maximum(atol, rtol * np.abs(row["moments"]))
            row["sample_converged"] = bool(
                previous is not None
                and np.all(3 * row["sem"] <= tol)
                and np.all(np.abs(row["moments"] - previous["moments"]) <= tol)
            )
            history.append(row)
            previous = row
            if row["sample_converged"]:
                break
        eta_results.append(row)
    last = eta_results[-1]
    tol = np.maximum(atol, rtol * np.abs(last["moments"]))
    eta_error = np.max(
        np.abs(np.diff(np.stack([r["moments"] for r in eta_results[-3:]]), axis=0)),
        axis=0,
    )
    converged = all(r["sample_converged"] for r in eta_results[-3:]) and np.all(
        eta_error <= tol
    )
    return dict(
        moments=last["moments"],
        sem=last["sem"],
        eta_error=eta_error,
        rtol=rtol,
        atol=atol,
        method=method,
        selected_eta=last["eta"],
        covariance_mean=last["covariance_mean"],
        eta_difference=last["moments"] - eta_results[-2]["moments"],
        covariance_total=last["covariance_mean"]
        + np.outer(
            last["moments"] - eta_results[-2]["moments"],
            last["moments"] - eta_results[-2]["moments"],
        ),
        converged=bool(converged),
        window={k: bool(v) if k == "has_gap" else float(v) for k, v in window.items()},
        history=history,
        seconds=sum(r["seconds"] for r in history),
    )


class TraceConvergenceError(ArithmeticError):
    """Nonconverged result; diagnostic contains the completed run for inspection."""

    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def five_moments(
    alpha, signal_fraction=0.2, delta=0.1, *, integration_options=None, **grids
):
    """RMT predictions for E01, using dedicated quadrature for orders one and two."""
    rho = 1 - signal_fraction
    _validate_parameters(alpha, rho, delta)
    first, second, _ = lower_order_traces(
        alpha, rho, delta, compute_third=False, **grids
    )
    diagnostic = compute_moments_diagnostic(
        alpha, rho, delta, **(integration_options or {})
    )
    if not diagnostic["converged"]:
        raise TraceConvergenceError(
            "Higher trace moments did not converge.", diagnostic
        )
    return np.r_[first, second, diagnostic["moments"][2:5]]
