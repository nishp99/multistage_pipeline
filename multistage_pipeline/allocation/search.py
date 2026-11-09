"""Search feasible pretraining/SFT/RL allocations using the compact dynamics."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..dynamics.pipeline import advance, step_args
from ..pretraining.projector import Geometry, population_geometry, population_moments


@dataclass(frozen=True)
class Prescription:
    alpha: float
    sft: float
    rl: float
    accuracy: float


class TraceTable:
    def __init__(self, alphas, signal_fraction=0.2, delta=0.1, quick=False):
        self.alphas = np.unique(alphas)
        self.geometries = [
            population_geometry(
                population_moments(float(a), signal_fraction, delta, quick)
            )
            for a in self.alphas
        ]
        self.values = np.stack([np.concatenate(g) for g in self.geometries])

    def __call__(self, alpha):
        if alpha < self.alphas[0] - 1e-12 or alpha > self.alphas[-1] + 1e-12:
            raise ValueError("Requested alpha outside interpolation table.")
        values = np.array([np.interp(alpha, self.alphas, col) for col in self.values.T])
        return Geometry(*np.split(values, [4, 6, 8]))


def _fractional_phase(state, geometry, args, epochs, dimension, phase):
    whole = jnp.floor(epochs).astype(int)
    floor = advance(state, geometry, args, whole * dimension, phase)

    def interpolate(_):
        after = advance(floor, geometry, args, dimension, phase)
        return floor + (epochs - whole) * (after - floor)

    return jax.lax.cond(epochs > whole, interpolate, lambda _: floor, None)


def _score(geometry, args, sft, rl, dimension):
    y = _fractional_phase(geometry.initial, geometry, args, sft, dimension, "sft")
    y = _fractional_phase(y, geometry, args, rl, dimension, "rl")
    corr = (y[0] + y[1]) / jnp.sqrt(y[2] + args[1] * y[3])
    return 0.5 + jnp.arcsin(jnp.clip(corr, -1.0, 1.0)) / jnp.pi


scores = jax.jit(jax.vmap(_score, in_axes=(0, None, 0, 0, None)))
score = jax.jit(_score)


def optimize(
    cfg,
    table,
    budget,
    cost,
    mode="mixed",
    fixed_alpha=None,
    grid=21,
    cap=None,
    phase_costs=(1.0, 1.0),
    fixed_spacing=0.0005,
):
    """Optimize alpha and the SFT share of remaining compute; fixed alpha uses FT-only budget."""
    if (
        mode not in ("mixed", "sft", "rl")
        or budget < 0
        or min(phase_costs) <= 0
        or grid < 2
    ):
        raise ValueError("Invalid mode, grid, costs or budget.")
    if fixed_alpha is None:
        if cost <= 0:
            raise ValueError(
                "Pretraining cost must be positive for joint optimization."
            )
        lo, hi = table.alphas[0], min(table.alphas[-1], budget / cost)
    else:
        lo = hi = fixed_alpha
        cost = 0.0
    if hi < lo - 1e-12:
        raise ValueError("Budget cannot pay the minimum pretraining cost.")
    args = step_args(cfg)

    def durations(a, f):
        remaining = np.maximum(0.0, budget - cost * a)
        return remaining * f / phase_costs[0], remaining * (1 - f) / phase_costs[1]

    def evaluate(a, f):
        s, r = durations(a, f)
        return float(
            score(
                table(a),
                args,
                min(s, cap) if cap else s,
                min(r, cap) if cap else r,
                cfg.dimension,
            )
        )

    fractions = (
        np.linspace(0.0, 1.0, grid)
        if mode == "mixed"
        else np.array([float(mode == "sft")])
    )
    alphas = np.array([lo]) if lo == hi else np.linspace(lo, hi, grid)
    candidates = []
    for a in alphas:
        s, r = durations(a, fractions)
        geom = jax.tree.map(lambda v: np.broadcast_to(v, (len(s), len(v))), table(a))
        values = np.asarray(
            scores(
                geom,
                args,
                np.minimum(s, cap) if cap else s,
                np.minimum(r, cap) if cap else r,
                cfg.dimension,
            )
        )
        candidates.extend(zip(values, np.repeat(a, len(values)), fractions))
    if fixed_alpha is not None and mode == "mixed":
        peak = max(v for v, _, _ in candidates)
        center = min(f for v, _, f in candidates if v >= peak - 1e-7)
        width = 1 / (grid - 1)
        fine = np.arange(
            max(0, center - width),
            min(1, center + width) + fixed_spacing / 2,
            fixed_spacing,
        )
        for f in np.clip(fine, 0, 1):
            candidates.append((evaluate(lo, f), lo, f))
    elif hi > lo and budget > cost * lo:
        bounds = [(lo, hi), (fractions[0], fractions[-1])]
        seeds = [
            max(candidates),
            (0, lo, fractions[0]),
            (0, hi, fractions[-1]),
            (0, (lo + hi) / 2, np.mean(fractions)),
        ]
        for _, a, f in seeds:
            opt = minimize(
                lambda x: -evaluate(*x),
                [a, f],
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=20, maxfun=80, eps=0.001),
            )
            if np.isfinite(opt.fun):
                candidates.append((-opt.fun, *opt.x))
    finite = [item for item in candidates if np.isfinite(item[0])]
    if not finite:
        raise ArithmeticError("All feasible prescriptions have non-finite accuracy.")
    peak = max(x[0] for x in finite)
    tolerance = 1e-7 if fixed_alpha is not None else 1e-12
    value, a, f = min(
        (x for x in finite if x[0] >= peak - tolerance), key=lambda x: (x[2], x[1])
    )
    s, r = durations(a, f)
    return Prescription(float(a), float(s), float(r), float(value))
