"""Fit and validate warm-start offsets against RL duration doubling."""

import numpy as np
from scipy.optimize import least_squares


def offset_fit(n, error, model="power"):
    """Bounded raw-error fit, with dimensionless internal parameters."""
    n, error = np.asarray(n, float), np.asarray(error, float)
    base, scale, nref = float(error.min()), float(np.ptp(error)), float(n[0])
    if len(n) < 4 or scale < 1e-14 or base <= 0:
        return None
    x = n / nref
    y = (error - base) / scale
    qmax = base / scale

    def prediction(v):
        q, b, beta = v
        shape = (
            np.exp(-beta * np.log(x)) if model == "power" else np.exp(-beta * (x - 1))
        )
        return -q + b * shape

    best = None
    for beta0 in (0.3, 1.0, 2.0):
        sol = least_squares(
            lambda v: prediction(v) - y,
            [min(0.1, qmax * 0.5), 1.1, beta0],
            bounds=([1e-14, 1e-14, 0.01], [qmax, np.inf, 10.0]),
            max_nfev=1500,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        if sol.success and (best is None or np.sum(sol.fun**2) < np.sum(best.fun**2)):
            best = sol
    if best is None:
        return None
    q, b, beta = best.x
    floor = base - scale * q
    predicted = base + scale * prediction(best.x)
    return dict(
        e_inf=float(max(0.0, floor)),
        amplitude=float(scale * b),
        n_ref=nref,
        beta=float(beta),
        model=model,
        sse=float(np.sum((predicted - error) ** 2)),
        sse_units="raw error",
        boundary=bool(beta <= 0.0101 or beta >= 9.999 or floor < 1e-12 or q < 1e-10),
    )


def predict(fit, n):
    x = np.asarray(n, float) / fit["n_ref"]
    shape = (
        x ** (-fit["beta"])
        if fit["model"] == "power"
        else np.exp(-fit["beta"] * (x - 1))
    )
    return fit["e_inf"] + fit["amplitude"] * shape


def fit_tail(n, error, cutoff=None):
    """Choose the earliest stable post-inflection tail; keep rejection evidence."""
    n, error = np.asarray(n), np.asarray(error)
    result = dict(fit_status="insufficient_tail", accepted=False, candidates=[])
    pos = n > 0
    x, y = n[pos], error[pos]
    if len(x) < 8:
        return result
    spread = float(np.ptp(y))
    if spread < 1e-12:
        result["fit_status"] = "flat"
        return result
    slope = -np.gradient(y, np.log(x))
    inflection = float(x[np.argmax(slope)])
    minimum = float(cutoff) if cutoff is not None else 3 * inflection
    result.update(inflection_n=inflection, cutoff_override=cutoff, minimum_n=minimum)
    monotonic_tolerance = max(1e-11, spread * 1e-4)
    result["nonmonotonic_observed"] = bool(np.any(np.diff(y) > monotonic_tolerance))
    if result["nonmonotonic_observed"]:
        result["fit_status"] = "nonmonotonic"
    fits = []
    for start in np.where(x >= minimum)[0]:
        xx, yy = x[start:], y[start:]
        if len(xx) < 8 or xx[-1] / xx[0] < 10:
            continue
        if np.any(np.diff(yy) > monotonic_tolerance):
            continue
        f = offset_fit(xx, yy)
        if f is None:
            continue
        # Two late points are never used to fit the prediction check.
        holdout = offset_fit(xx[:-2], yy[:-2])
        alternative = offset_fit(xx, yy, "exp")
        deficit = yy - f["e_inf"]
        if np.any(deficit <= 0):
            continue
        residual = yy - predict(f, xx)
        prediction_error = (
            np.inf
            if holdout is None
            else float(
                np.max(np.abs(predict(holdout, xx[-2:]) - yy[-2:]) / deficit[-2:])
            )
        )
        relative_residual = float(np.max(np.abs(residual) / deficit))
        # A quadratic trend must be negligible relative to the local deficit.
        trend = np.polyval(
            np.polyfit(np.log(xx / xx[0]), residual / deficit, 2), np.log(xx / xx[0])
        )
        curvature = float(np.ptp(trend))
        f.update(
            n_min=int(xx[0]),
            n_max=int(xx[-1]),
            n_points=len(xx),
            holdout_relative_error=prediction_error,
            max_relative_residual=relative_residual,
            residual_curvature=curvature,
            exp_sse=alternative["sse"] if alternative else np.nan,
        )
        fits.append(f)
    result["candidates"] = fits
    for i in range(len(fits) - 1):
        f, adjacent = fits[i : i + 2]
        stability = abs(f["beta"] - adjacent["beta"]) / f["beta"]
        f["beta_relative_sensitivity"] = stability
        f["e_inf_sensitivity"] = abs(f["e_inf"] - adjacent["e_inf"])
        # Numerical exact fits can have indistinguishable vanishing SSEs.
        power_better = f["sse"] <= 1e-24 or f["sse"] < 0.95 * f["exp_sse"]
        valid = all(
            not z["boundary"]
            and z["holdout_relative_error"] <= 0.1
            and z["max_relative_residual"] <= 0.1
            and z["residual_curvature"] <= 0.05
            for z in (f, adjacent)
        )
        if valid and stability <= 0.1 and power_better:
            return {
                **result,
                **f,
                "fit_status": "stable_power_law",
                "accepted": True,
                "e_inf_range": sorted([f["e_inf"], adjacent["e_inf"]]),
                "beta_range": sorted([f["beta"], adjacent["beta"]]),
            }
    if fits:
        result.update(fits[-1], fit_status="unstable_or_non_power_law", accepted=False)
    return result


def convergence_errors(curve):
    """Conservative envelopes of the last TWO duration-doubling changes."""
    if curve.history is None or len(curve.history) < 3:
        return None
    if not np.all(curve.durations[-2:] == 2 * curve.durations[-3:-1]):
        return None
    changes = np.diff(curve.history[-3:], axis=0)
    return np.max(np.abs(changes), axis=0)


def empirical_window_fit(n, error, e_inf, floor, minimum):
    """Descriptive finite-range log fit with an independently estimated offset.

    It NEVER estimates E_inf and is never called an asymptotic extrapolation.
    Search latest usable upper endpoints first, then adjacent lower cutoffs.
    """
    deficit = error - e_inf
    keep = (n >= minimum) & (n > 0) & (deficit > 0) & (floor < 0.1 * deficit)
    x, y = n[keep], deficit[keep]
    result = dict(
        accepted=False,
        fit_status="empirical_plateau_no_power_law",
        scope="finite_range",
        candidates=[],
        e_inf=float(e_inf),
        beta=None,
        n_min=float(minimum),
        n_max=int(n[-1]),
    )

    def fit(xx, yy):
        lx = np.log(xx / xx[0])
        ly = np.log(yy)
        slope, intercept = np.polyfit(lx, ly, 1)
        pred = np.exp(intercept + slope * lx)
        train = np.polyfit(lx[:-2], ly[:-2], 1)
        hold = np.exp(np.polyval(train, lx[-2:]))
        residual = (yy - pred) / yy
        trend = np.polyval(np.polyfit(lx, residual, 2), lx)
        exponential = np.polyfit(xx, ly, 1)
        exp_sse = float(np.sum((ly - np.polyval(exponential, xx)) ** 2))
        sse = float(np.sum((ly - np.log(pred)) ** 2))
        return dict(
            accepted=False,
            scope="finite_range",
            model="power",
            e_inf=float(e_inf),
            n_ref=float(xx[0]),
            amplitude=float(np.exp(intercept)),
            beta=float(-slope),
            n_min=int(xx[0]),
            n_max=int(xx[-1]),
            n_points=len(xx),
            holdout_relative_error=float(np.max(abs(hold - yy[-2:]) / yy[-2:])),
            max_relative_residual=float(np.max(abs(residual))),
            residual_curvature=float(np.ptp(trend)),
            sse=sse,
            exp_sse=exp_sse,
            sse_units="log deficit",
        )

    for stop in range(len(x), 8, -1):
        previous = None
        for start in range(stop - 7):
            xx, yy = x[start:stop], y[start:stop]
            if len(xx) < 8 or xx[-1] / xx[0] < 10:
                break
            f = fit(xx, yy)
            if previous is not None:
                stable = abs(f["beta"] - previous["beta"]) / max(
                    previous["beta"], 1e-12
                )
                valid = all(
                    z["beta"] > 0
                    and z["holdout_relative_error"] <= 0.1
                    and z["max_relative_residual"] <= 0.1
                    and z["residual_curvature"] <= 0.05
                    and (z["sse"] < 1e-24 or z["sse"] < 0.95 * z["exp_sse"])
                    for z in (previous, f)
                )
                if valid and stable <= 0.1:
                    return {
                        **previous,
                        "accepted": True,
                        "fit_status": "finite_range_power_law",
                        "beta_relative_sensitivity": stable,
                        "e_inf_sensitivity": 0.0,
                        "beta_range": sorted([f["beta"], previous["beta"]]),
                        "candidates": [],
                    }
            previous = f
    return result


def analyze_curve(c, cutoff=None):
    fit = fit_tail(c.n, c.error, cutoff)
    minimum = float(cutoff) if cutoff is not None else fit.get("minimum_n", 3.0)
    if any(v["rejected"] for v in c.longer_validation):
        fit.update(accepted=False, fit_status="fails_longer_run_validation")
    drift = convergence_errors(c)
    e_zero = float(c.error[0])
    baseline_used = (
        c.baseline is not None and drift is not None and c.baseline["drift"] < drift[0]
    )
    if baseline_used:
        e_zero = c.baseline["error"]
        drift[0] = c.baseline["drift"]
    row = dict(
        config=c.config,
        alpha=c.alpha,
        epsilon=c.epsilon,
        key=c.key,
        rl_steps=c.rl_steps,
        n_sft_max=int(c.n[-1]),
        sources=c.sources,
        e_zero=e_zero,
        method="unresolved",
        status=fit["fit_status"],
        fit=fit,
        n=c.n.tolist(),
        error=c.error.tolist(),
        rl_absolute_drift=None if drift is None else drift.tolist(),
        e_inf=None,
        e_inf_sensitivity=None,
        benefit=None,
        benefit_floor=None,
        beta=fit.get("beta"),
        resolved_mask=[False] * len(c.n),
    )
    row["independent_baseline"] = c.baseline
    row["independent_baseline_used"] = bool(baseline_used)
    row["longer_run_validation"] = c.longer_validation
    if drift is None:
        row["status"] = "rl_unvalidated:" + fit["fit_status"]
        return row
    changes = np.diff(c.history[-3:], axis=0)
    row.update(
        rl_durations=c.durations.tolist(),
        rl_difference_from_zero_drift=np.max(
            np.abs(changes - changes[:, :1]), axis=0
        ).tolist(),
        rl_difference_from_endpoint_drift=np.max(
            np.abs(changes - changes[:, -1:]), axis=0
        ).tolist(),
    )
    # SFT endpoint evidence: distinct checkpoints near M/4, M/2 and M.
    targets = [c.n[-1] / 4, c.n[-1] / 2, c.n[-1]]
    indices = [int(np.argmin(np.abs(c.n - v))) for v in targets]
    endpoint_drift = np.inf
    if len(set(indices)) == 3 and all(
        abs(c.n[k] / t - 1) < 0.15 for k, t in zip(indices, targets)
    ):
        endpoint_drift = float(np.max(np.abs(np.diff(c.error[indices]))))
    gain = float(e_zero - c.error[-1])
    full_tail_fit = fit
    remaining = max(0.0, c.error[-1] - fit["e_inf"]) if fit.get("accepted") else np.inf
    empirical_floor = max(endpoint_drift, remaining)
    empirical_contradicted = any(
        v.get("empirical_rejected", False) for v in c.longer_validation
    )
    # Empirical requires a constrained remaining correction, not flatness alone.
    empirical = (
        not empirical_contradicted
        and fit.get("accepted")
        and abs(gain) > 0
        and empirical_floor + drift[0] + drift[-1] < 0.05 * abs(gain)
    )
    if empirical:
        usable = (
            (c.n >= fit["n_min"])
            & (c.error - c.error[-1] > 0)
            & (drift + drift[-1] + empirical_floor < 0.1 * (c.error - c.error[-1]))
        )
        empirical = bool(usable.sum() >= 6 and c.n[usable][-1] / c.n[usable][0] >= 10)
    if empirical:
        einf, sensitivity, method = float(c.error[-1]), empirical_floor, "empirical"
    elif fit.get("accepted"):
        einf, sensitivity, method = fit["e_inf"], fit["e_inf_sensitivity"], "power_fit"
    elif (
        not empirical_contradicted
        and abs(gain) > 0
        and np.isfinite(endpoint_drift)
        and c.n[indices[0]] >= minimum
        and 10 * endpoint_drift + drift[0] + drift[-1] < 0.01 * abs(gain)
    ):
        # Empirical estimates need not depend on identifying a three-parameter
        # power law. A conservative 10x observed SFT drift supplies the display
        # floor; this is a horizon-limited estimate, not proof of convergence.
        einf, sensitivity, method = (
            float(c.error[-1]),
            max(10 * endpoint_drift, 1e-12),
            "empirical",
        )
        floor = drift + drift[-1] + sensitivity
        fit = empirical_window_fit(c.n, c.error, einf, floor, minimum)
        row.update(
            fit=fit,
            beta=fit.get("beta"),
            full_tail_fit=full_tail_fit,
            empirical_remaining_correction="unidentified; 10x measured SFT extension drift used as a diagnostic floor",
        )
    else:
        return row
    # RL floor includes movement of the fitted asymptote itself under doubling.
    tail_floors = []
    if method == "power_fit":
        for old_error in c.history[-3:-1]:
            mask = c.n >= fit["n_min"]
            old = offset_fit(c.n[mask], old_error[mask])
            if old is None:
                return row
            tail_floors.append(abs(old["e_inf"] - einf))
    einf_drift = max(tail_floors, default=float(drift[-1]))
    numerical_floor = drift + einf_drift + sensitivity
    deficit = c.error - einf
    mask = (c.n > 0) & (deficit > 0) & (numerical_floor < 0.1 * deficit)
    benefit = float(e_zero - einf)
    benefit_floor = float(numerical_floor[0])
    tail_mask = mask & (c.n >= fit.get("n_min", np.inf)) & (c.n <= fit.get("n_max", 0))
    tail_span = (
        float(np.log10(c.n[tail_mask][-1] / c.n[tail_mask][0]))
        if tail_mask.sum() >= 2
        else 0.0
    )
    row.update(
        method=method,
        status=(
            "resolved"
            if abs(benefit) > 0 and benefit_floor < 0.1 * abs(benefit)
            else "benefit_unresolved"
        ),
        e_inf=float(einf),
        e_inf_sensitivity=float(sensitivity),
        e_inf_rl_drift=float(einf_drift),
        benefit=benefit,
        benefit_floor=benefit_floor,
        numerical_floor=numerical_floor.tolist(),
        resolved_mask=mask.tolist(),
        endpoint_sft_drift=endpoint_drift,
        resolved_tail_points=int(tail_mask.sum()),
        resolved_tail_decades=tail_span,
        power_tail_resolved=bool(
            fit.get("accepted") and tail_mask.sum() >= 6 and tail_span >= 1
        ),
        power_law_scope=fit.get("scope", "late_tail"),
    )
    return row
