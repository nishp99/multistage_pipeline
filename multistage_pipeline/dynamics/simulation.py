"""Microscopic online learning in the retained PCA coordinates.

Diagonalizing the projected covariance lets us sample the joint Gaussian of
the retained input and teacher field directly. This has the same distribution
as drawing an ambient Gaussian and projecting it, with O(K T) work per update.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh

from ..pretraining.projector import measured_geometry
from .pipeline import sample_steps, step, step_args


class Fields(NamedTuple):
    eigenvalues: np.ndarray
    teacher: np.ndarray
    teacher_regression: np.ndarray
    teacher_residual_sd: float
    input_sd: np.ndarray


def prepare_fields(basis, teacher, student, delta):
    rank = basis.shape[1]
    eigen, rotation = eigh(basis[:rank].T @ basis[:rank], check_finite=False)
    eigen = np.clip(eigen, 0, 1)
    axes = basis @ rotation
    projected_teacher = axes.T @ teacher
    variance = eigen + delta * (1 - eigen)
    # The teacher lives entirely in the signal block of covariance (variance 1).
    regression = projected_teacher / variance
    unexplained = teacher @ teacher - projected_teacher @ regression
    fields = Fields(
        eigen,
        projected_teacher,
        regression,
        np.sqrt(max(0.0, unexplained)),
        np.sqrt(variance),
    )
    return fields, axes.T @ student


def spectral_state(student, fields, dimension):
    e, t = fields.eigenvalues, fields.teacher
    powers = np.stack([np.ones_like(e), e, e**2])
    return np.r_[powers @ (student * t), powers @ (student**2)] / dimension


@partial(jax.jit, static_argnames=("phase", "episode_length"))
def _advance(
    student, theory, geometry, fields, args, key, start, end, phase, episode_length
):
    T = episode_length

    def body(index, carry):
        w, state = carry
        state, rate = step(state, geometry, args, phase)
        episode_key = jax.random.fold_in(key, index)
        data_key, residual_key = jax.random.split(episode_key)
        x = jax.random.normal(data_key, (T, w.size)) * fields.input_sd
        noise = jax.random.normal(residual_key, (T,))
        target = jnp.sign(
            x @ fields.teacher_regression + noise * fields.teacher_residual_sd
        )
        decisions = jnp.sign(x @ w)
        if phase == "sft":
            target *= jnp.where(
                jax.random.bernoulli(
                    jax.random.fold_in(episode_key, 1), args[-1], (T,)
                ),
                -1,
                1,
            )
            gain = jnp.where(target != decisions, -decisions, 0.0)
        else:
            gain = decisions * (jnp.sum(decisions == target) >= args[-2])
        w = w + rate * jnp.sqrt(args[0]) * (gain @ x) / T
        initial_norm = jnp.sqrt(geometry.initial[2:].sum() / args[0])
        w = jnp.where(args[5], w * initial_norm / jnp.linalg.norm(w), w)
        return w, state

    return jax.lax.fori_loop(start, end, body, (student, theory))


class Simulation(NamedTuple):
    steps: np.ndarray
    states: np.ndarray
    conditional: np.ndarray
    spectral: np.ndarray
    switch_step: int


def simulate(cfg, basis, teacher, student, sft_epochs, rl_epochs, seed, steps=None):
    geometry = measured_geometry(basis, teacher, student)
    fields, w = prepare_fields(basis, teacher, student, cfg.delta)
    times, switch = sample_steps(cfg.dimension, sft_epochs, rl_epochs)
    if steps is not None:
        if min(steps) < 0 or max(steps) > times[-1]:
            raise ValueError("Requested simulation checkpoint outside the trajectory.")
        times = np.unique(np.r_[0, switch, times[-1], steps]).astype(int)
    state = geometry.initial
    spectral = [spectral_state(w, fields, cfg.dimension)]
    conditional = [state]
    key = jax.random.PRNGKey(seed)
    for start, end in zip(times[:-1], times[1:]):
        phase = "sft" if end <= switch else "rl"
        w, state = _advance(
            w,
            state,
            geometry,
            fields,
            step_args(cfg),
            key,
            start,
            end,
            phase,
            cfg.episode_length,
        )
        spectral.append(spectral_state(np.asarray(w), fields, cfg.dimension))
        conditional.append(np.asarray(state))
    s = np.stack(spectral)
    compact = np.stack((s[:, 1], s[:, 0] - s[:, 1], s[:, 4], s[:, 3] - s[:, 4]), axis=1)
    return Simulation(times, compact, np.stack(conditional), s, switch)
