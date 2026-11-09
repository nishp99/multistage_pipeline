"""Integrate SFT followed by RL. One epoch is D microscopic updates."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .compact import compact_step


@dataclass(frozen=True)
class Dynamics:
    dimension: int = 400
    episode_length: int = 6
    delta: float = 0.1
    epsilon: float = 0.2
    learning_rate: float = 1.0
    optimal: bool = True
    spherical: bool = True
    threshold: int | None = None

    def __post_init__(self):
        if self.dimension < 1 or self.episode_length < 1 or not 0 <= self.epsilon <= 1:
            raise ValueError(
                "Invalid dimension, episode length or label-flip probability."
            )
        if (
            self.threshold is not None
            and not 1 <= self.threshold <= self.episode_length
        ):
            raise ValueError("Reward threshold must lie in [1,T].")


class Trajectory(NamedTuple):
    steps: np.ndarray
    states: np.ndarray
    switch_step: int


def accuracy(state, delta=0.1):
    j1, j2, q1, q2 = np.moveaxis(np.asarray(state), -1, 0)
    return 0.5 + np.arcsin(np.clip((j1 + j2) / np.sqrt(q1 + delta * q2), -1, 1)) / np.pi


def step_args(cfg):
    return (
        1 / cfg.dimension,
        cfg.delta,
        cfg.episode_length,
        cfg.learning_rate,
        cfg.optimal,
        cfg.spherical,
        cfg.threshold or cfg.episode_length,
        cfg.epsilon,
    )


def step(state, geometry, args, phase):
    dt, delta, T, rate, optimal, sphere, n, epsilon = args
    radius = geometry.initial[2:].sum()
    *next_state, eta = compact_step(
        state,
        dt,
        delta,
        T,
        radius,
        1.0,
        geometry.teacher,
        geometry.omega,
        geometry.traces,
        rate,
        optimal,
        sphere,
        n,
        epsilon,
        phase,
    )
    return jnp.stack(next_state), eta


@partial(jax.jit, static_argnames=("phase",))
def advance(state, geometry, args, count, phase):
    return jax.lax.fori_loop(
        0, count, lambda _, y: step(y, geometry, args, phase)[0], state
    )


def sample_steps(dimension, sft_epochs, rl_epochs, points=200):
    if min(sft_epochs, rl_epochs) < 0:
        raise ValueError("Phase durations must be nonnegative.")
    switch, rl = (round(t * dimension) for t in (sft_epochs, rl_epochs))
    total = switch + rl
    epochs = np.unique(
        np.rint(np.geomspace(1, max(1, total / dimension), points)).astype(int)
    )
    return (
        np.unique(
            np.r_[0, epochs[epochs * dimension < total] * dimension, switch, total]
        ),
        switch,
    )


def trajectory(cfg, geometry, sft_epochs, rl_epochs, steps=None):
    times, switch = sample_steps(cfg.dimension, sft_epochs, rl_epochs)
    if steps is not None:
        if min(steps) < 0 or max(steps) > times[-1]:
            raise ValueError("Requested checkpoint outside training interval.")
        times = np.unique(np.r_[0, switch, times[-1], steps]).astype(int)
    state = geometry.initial
    states = [np.asarray(state)]
    for start, end in zip(times[:-1], times[1:]):
        phase = "sft" if end <= switch else "rl"
        state = advance(state, geometry, step_args(cfg), end - start, phase)
        states.append(np.asarray(state))
    if not np.isfinite(states).all():
        raise ArithmeticError(
            "Non-finite compact ODE state; inspect phase settings and closure geometry."
        )
    return Trajectory(times, np.stack(states), switch)
