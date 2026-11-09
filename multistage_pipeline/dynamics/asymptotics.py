"""Finite RL endpoints, SFT fixed points, and warm-start tail extrapolation."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from .pipeline import accuracy, advance, step, step_args
from .tail_fit import analyze_curve


@jax.jit
def _newton_step(state, geometry, args, dimension):
    def residual(y):
        mapped = jax.lax.fori_loop(
            0, dimension, lambda _, z: step(z, geometry, args, "sft")[0], y
        )
        return mapped - y

    change = jnp.linalg.solve(jax.jacfwd(residual)(state), residual(state))
    change *= jnp.minimum(1.0, 0.05 / jnp.maximum(1e-30, jnp.linalg.norm(change)))
    return jnp.where(jnp.isfinite(change).all(), state - change, state)


def sft_fixed_point(cfg, geometry, epochs=1000, iterations=5):
    args = step_args(cfg)
    state = advance(geometry.initial, geometry, args, cfg.dimension * epochs, "sft")
    for _ in range(iterations):
        state = _newton_step(state, geometry, args, cfg.dimension)
    return np.asarray(state)


def warm_start_reference(cfg, geometry, warm_epochs, rl_epochs, alpha):
    warm_epochs = np.asarray(warm_epochs, dtype=int)
    if warm_epochs[0] != 0 or np.any(np.diff(warm_epochs) <= 0) or rl_epochs % 4:
        raise ValueError(
            "Require increasing warm starts from zero and RL epochs divisible by four."
        )
    durations = np.array([rl_epochs // 4, rl_epochs // 2, rl_epochs])
    states = []
    state, previous = geometry.initial, 0
    args = step_args(cfg)
    for epoch in warm_epochs:
        state = advance(
            state, geometry, args, (epoch - previous) * cfg.dimension, "sft"
        )
        states.append(np.asarray(state))
        previous = epoch
    histories = []
    current = jnp.array(states)
    for length in np.diff(np.r_[0, durations]):
        current = _rl_batch(current, geometry, args, length * cfg.dimension)
        histories.append(np.asarray(current))
    errors = 1 - accuracy(histories, cfg.delta)
    curve = SimpleNamespace(
        n=warm_epochs,
        error=errors[-1],
        history=errors,
        durations=durations,
        config="spherical_optimal",
        alpha=alpha,
        epsilon=cfg.epsilon,
        rl_steps=rl_epochs,
        sources=[],
        baseline=None,
        longer_validation=[],
        key=f"alpha={alpha}/epsilon={cfg.epsilon}",
    )
    fit = analyze_curve(curve)
    return dict(
        warm_states=states,
        rl_states=histories,
        durations=durations,
        errors=errors,
        reference=np.nan if fit["e_inf"] is None else fit["e_inf"],
        method=fit["method"],
        status=fit["status"],
        resolved=fit["resolved_mask"],
    )


@jax.jit
def _rl_batch(states, geometry, args, count):
    return jax.vmap(lambda y: advance(y, geometry, args, count, "rl"))(states)
