"""Compare four and six states on the same finite-D projector and initialization."""

import matplotlib.pyplot as plt
import numpy as np
from jax import random
from experiments.common import run
from multistage_pipeline.dynamics import six_state
from multistage_pipeline.dynamics.pipeline import Dynamics, accuracy
from multistage_pipeline.dynamics.simulation import simulate
from multistage_pipeline.pretraining.projector import (
    vectors,
    pca_basis,
    spectral_moments,
    measured_spectral_state,
)

DEFAULTS = dict(
    dimension=800,
    alpha=0.5,
    signal_fraction=0.2,
    delta=0.1,
    episode_length=5,
    epsilon=0.2,
    optimal=False,
    spherical=True,
    sft_epochs=100,
    rl_epochs=0,
    seed=42,
    include_population=True,
)
QUICK = dict(dimension=50, sft_epochs=3, rl_epochs=2, include_population=False)


def compute(c):
    cfg = Dynamics(
        c["dimension"],
        c["episode_length"],
        c["delta"],
        c["epsilon"],
        optimal=c["optimal"],
        spherical=c["spherical"],
    )
    teacher, student = vectors(cfg.dimension, c["signal_fraction"])
    basis = pca_basis(
        random.PRNGKey(c["seed"]),
        c["alpha"],
        cfg.dimension,
        c["signal_fraction"],
        cfg.delta,
    )
    initial, t = measured_spectral_state(basis, teacher, student)
    moments = spectral_moments(basis)
    s, r = c["sft_epochs"], c["rl_epochs"]
    six = six_state.trajectory(cfg, moments, t, initial, s, r)
    empirical = simulate(cfg, basis, teacher, student, s, r, c["seed"] + 1)
    result = dict(
        moments=moments,
        steps=six.steps,
        switch_step=six.switch_step,
        six=six.states,
        four=empirical.conditional,
        simulation=empirical.states,
    )
    if c["include_population"]:
        m, t, y = six_state.analytic_initial(
            c["alpha"], c["signal_fraction"], cfg.delta
        )
        result["population_six"] = six_state.trajectory(cfg, m, t, y, s, r).states
        result["population_moments"] = m
    return result


def plot(data, c):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    x = data["steps"] / c["dimension"]
    for key in ("simulation", "four", "six", "population_six"):
        if key not in data:
            continue
        state = data[key]
        if state.shape[-1] == 6:
            j0, j1, _, q0, q1, _ = state.T
            state = np.stack([j1, j0 - j1, q1, q0 - q1], axis=-1)
        axs[0].plot(x, accuracy(state, c["delta"]), label=key)
        axs[1].plot(x, state[:, 0], label=key)
    axs[0].set(ylabel="Clean accuracy", xlabel="Completed epochs")
    axs[1].set(ylabel="Signal overlap J", xlabel="Completed epochs")
    axs[0].legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    run(__doc__, "compare_degree_two", DEFAULTS, QUICK, compute, plot)
