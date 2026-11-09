"""Trajectory interface for the validated degree-two projector hierarchy."""

from functools import partial

import jax
import numpy as np
from . import degree_two as ode
from .pipeline import Trajectory, sample_steps, step_args
from ..pretraining.spectral import cached_geometry, symmetric_closure


def analytic_initial(alpha, signal_fraction=0.2, delta=0.1):
    """Population moments for a signal-only teacher and independent isotropic student."""
    moments = cached_geometry(alpha, 1 - signal_fraction, delta)["moments"]
    initial = np.r_[np.zeros(3), moments[:3]]
    teacher = moments[1:4] / signal_fraction
    return moments, teacher, initial


@partial(jax.jit, static_argnames=("phase",))
def _advance(state, teacher, closure, diffusion, args, radius, count, phase):
    dt, delta, T, lr, optimal, sphere, _, epsilon = args

    def microstep(_, y):
        if phase == "sft":
            coefficients = ode.sft_coefficients(
                y, 1.0, delta, T, epsilon, teacher, closure, diffusion
            )
        else:
            coefficients = ode.rl_coefficients(
                y, 1.0, delta, T, teacher, closure, diffusion
            )
        return ode._euler_step(
            y, dt, delta, lr, radius, 1.0, coefficients, optimal, sphere
        )[0]

    return jax.lax.fori_loop(0, count, microstep, state)


def trajectory(cfg, moments, teacher, initial, sft_epochs, rl_epochs):
    if cfg.threshold not in (None, cfg.episode_length):
        raise ValueError("The six-state RL closure implements the all-correct reward.")
    geometry = symmetric_closure(moments)
    times, switch = sample_steps(cfg.dimension, sft_epochs, rl_epochs)
    diffusion = ode.diffusion_constants(moments, cfg.delta)
    states = [initial]
    for start, end in zip(times[:-1], times[1:]):
        y = _advance(
            states[-1],
            teacher,
            geometry["closure"],
            diffusion,
            step_args(cfg),
            initial[3],
            end - start,
            "sft" if end <= switch else "rl",
        )
        states.append(np.asarray(y))
    return Trajectory(times, np.stack(states), switch)
