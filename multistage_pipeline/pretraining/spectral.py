"""Host-side, uncertainty-aware geometry for the six-state hierarchy.

All moments are normalized by ambient D. No dynamical clipping or statistical
rank reduction is performed here. Exact numerical rank deficiency is supported.
"""

from functools import lru_cache
import numpy as np
from scipy.linalg import eigh


class GeometryError(ArithmeticError):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def matrices(m):
    indices = np.add.outer(np.arange(3), np.arange(3))
    return m[indices], m[indices + 1]


def geometry_diagnostic(m):
    G, B = matrices(np.asarray(m, dtype=float))
    result = dict(
        gram_eigenvalues=np.linalg.eigvalsh(G),
        lower_eigenvalues=np.linalg.eigvalsh(B),
        upper_eigenvalues=np.linalg.eigvalsh(G - B),
        condition=float(np.linalg.cond(G)),
    )
    try:
        result["nodes"] = eigh(B, G, eigvals_only=True)
    except np.linalg.LinAlgError:
        result["nodes"] = None
    return result


def symmetric_closure(m, *, rank_rtol=1e-13, support_atol=1e-10):
    """Positive Gaussian quadrature and x^3 remainder on the Gram support."""
    m = np.asarray(m, dtype=float)
    if m.shape != (6,) or not np.all(np.isfinite(m)) or m[0] < 0:
        raise GeometryError("Expected six finite moments and nonnegative mass.", {})
    if m[0] == 0:
        if np.any(m != 0):
            raise GeometryError("Zero mass with nonzero moments.", {})
        return dict(
            moments=m,
            nodes=np.empty(0),
            weights=np.empty(0),
            closure=np.zeros(3),
            rank=0,
            reconstruction_error=0.0,
        )
    G, B = matrices(m)
    d, V = eigh(G)
    if d[0] < -rank_rtol * m[0]:
        raise GeometryError("Indefinite moment Gram matrix.", geometry_diagnostic(m))
    keep = d > rank_rtol * d[-1]
    W = V[:, keep] / np.sqrt(d[keep])
    operator = W.T @ B @ W
    nodes, U = eigh((operator + operator.T) / 2)
    weights = (U.T @ (np.sqrt(d[keep]) * V[0, keep])) ** 2
    reconstructed = np.array([weights @ nodes**k for k in range(6)])
    error = float(
        np.max(np.abs(reconstructed - m) / np.maximum(np.abs(m), m[0] * 1e-14))
    )
    if nodes.min() < -support_atol or nodes.max() > 1 + support_atol or error > 1e-10:
        raise GeometryError(
            "Moments have no numerically resolved quadrature on [0,1].",
            dict(nodes=nodes, reconstruction_error=error),
        )
    # Polynomial remainder works for ranks one and two as well as rank three.
    polynomial = np.polynomial.polynomial.polyfromroots(nodes)
    remainder = np.polynomial.polynomial.polydiv([0.0, 0.0, 0.0, 1.0], polynomial)[1]
    closure = np.pad(remainder, (0, 3 - len(remainder)))
    return dict(
        moments=m,
        nodes=nodes,
        weights=weights,
        closure=closure,
        rank=len(nodes),
        reconstruction_error=error,
    )


def prepare_geometry(m, covariance, *, repair_limit=3.0, covariance_rtol=1e-10):
    """Accept valid geometry or project within its measured uncertainty.

    Covariance describes m1..m5, including the eta-bias proxy. A covariance
    eigenvalue floor stabilizes the metric; it does not alter the trace vector.
    """
    m = np.asarray(m, dtype=float)
    if m.shape != (6,) or not np.all(np.isfinite(m)) or m[0] <= 0:
        raise GeometryError("Expected positive mass and six finite moments.", {})
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (5, 5) or not np.all(np.isfinite(covariance)):
        raise GeometryError("Expected finite 5x5 moment covariance.", {})
    diagnostic = dict(
        raw_moments=m.copy(),
        raw=geometry_diagnostic(m),
        covariance=covariance,
        repaired=False,
        repair_distance=0.0,
    )
    try:
        result = symmetric_closure(m)
        return dict(result, diagnostic=diagnostic)
    except GeometryError as error:
        diagnostic["raw_error"] = str(error)
    d, V = eigh((covariance + covariance.T) / 2)
    if d[-1] <= 0 or d[0] < -1e-10 * d[-1]:
        raise GeometryError(
            "Invalid or zero uncertainty cannot justify a repair.", diagnostic
        )
    floor = max(d[-1] * covariance_rtol, (np.finfo(float).eps * m[0]) ** 2)
    L = V * np.sqrt(np.maximum(d, floor))
    diagnostic["covariance_eigenvalue_floor"] = floor
    # Whiten the optimization variable, avoiding an ill-scaled inverse objective.
    import cvxpy as cp

    z = cp.Variable(5)
    x = cp.hstack([m[0], m[1:] + L @ z]) / m[0]
    G = cp.bmat([[x[i + j] for j in range(3)] for i in range(3)])
    B = cp.bmat([[x[i + j + 1] for j in range(3)] for i in range(3)])
    # A tiny recorded interior margin helps the SDP's boundary solution survive
    # the subsequent generalized eigenproblem. It is not a moment ridge.
    margin = 1e-12
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(z)),
        [
            B >> margin * np.eye(3),
            G - B >> margin * np.eye(3),
            x[:-1] >= x[1:],
            x[-1] >= 0,
        ],
    )
    try:
        problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1e-11,
            tol_gap_rel=1e-11,
            tol_feas=1e-12,
            max_iter=300,
        )
    except cp.error.SolverError as error:
        raise GeometryError("Moment feasibility SDP failed.", diagnostic) from error
    diagnostic.update(solver_status=problem.status, feasibility_margin=margin)
    if z.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        raise GeometryError(
            "Moment feasibility SDP did not return a solution.", diagnostic
        )
    distance = float(np.linalg.norm(z.value))
    accepted = np.r_[m[0], m[1:] + L @ z.value]
    diagnostic.update(
        repair_distance=distance, repaired=True, candidate_moments=accepted
    )
    if not distance < repair_limit:
        raise GeometryError(
            "Moment repair exceeds uncertainty budget; refine traces.", diagnostic
        )
    try:
        result = symmetric_closure(accepted)
    except GeometryError as error:
        diagnostic["candidate_error"] = error.diagnostic
        raise GeometryError(
            "Repaired moments failed quadrature verification.", diagnostic
        ) from error
    return dict(result, diagnostic=diagnostic)


DEFAULT_INTEGRATION_OPTIONS = dict(powers=(18, 20), eta_fractions=(3e-5, 1e-5, 3e-6))


def compute_geometry(
    alpha, rho, delta, *, integration_options=None, max_refinements=2, repair_limit=3.0
):
    """Bounded refinement of coherent RMT inputs; retain every attempted estimate.

    This certifies numerical support/repair size, not finite-hierarchy accuracy.
    Trajectory convergence is assessed by the accompanying validation harness.
    """
    from multistage_pipeline.pretraining.traces import compute_moments_diagnostic

    if max_refinements < 0 or repair_limit <= 0:
        raise ValueError(
            "Nonnegative refinement budget and positive repair limit required."
        )
    options = dict(DEFAULT_INTEGRATION_OPTIONS)
    options.update(integration_options or {})
    if options.get("max_order", 5) != 5:
        raise ValueError("Degree-two geometry requires all five trace moments.")
    attempts = []
    for level in range(max_refinements + 1):
        trace = compute_moments_diagnostic(alpha, rho, delta, **options)
        attempt = dict(trace=trace, integration_options=dict(options))
        attempts.append(attempt)
        try:
            if not trace["converged"]:
                raise GeometryError("Trace integration did not converge.", {})
            geometry = prepare_geometry(
                np.r_[1 - rho, trace["moments"]],
                trace["covariance_total"],
                repair_limit=repair_limit,
            )
            attempt["geometry"] = geometry["diagnostic"]
            return dict(geometry, attempts=attempts, refinement_level=level)
        except GeometryError as error:
            attempt.update(error=str(error), geometry=error.diagnostic)
        powers = options.get("powers", (14, 16, 18))
        etas = options.get("eta_fractions", (0.001, 0.0003, 0.0001, 0.00003))
        options.update(
            powers=tuple(p + 2 for p in powers),
            eta_fractions=tuple(e / 10 for e in etas),
        )
    raise GeometryError(
        "Validated degree-two geometry exhausted trace refinement.",
        dict(attempts=attempts),
    )


@lru_cache(maxsize=128)
def cached_geometry(alpha, rho, delta):
    """Reuse static host geometry across teacher parameters and JAX callbacks."""
    return compute_geometry(float(alpha), float(rho), float(delta))


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
