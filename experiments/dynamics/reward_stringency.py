"""E06. Compare paired RL trajectories with different reward thresholds."""

import numpy as np
import matplotlib.pyplot as plt
from experiments.common import run
from experiments.dynamics.shared import theory_geometry
from multistage_pipeline.dynamics.pipeline import Dynamics, accuracy, trajectory
from multistage_pipeline.dynamics.simulation import simulate
from multistage_pipeline.pretraining.projector import vectors, pca_basis

DEFAULTS = dict(
    alpha=7.0,
    dimension=800,
    signal_fraction=0.2,
    delta=0.1,
    episode_length=10,
    thresholds=[10, 9, 8, 7, 6, 4],
    epochs=1000,
    pairs=5,
    replicas=3,
)
QUICK = dict(dimension=40, epochs=5, pairs=2, replicas=2)


def compute(c):
    theory, sims, conditional = [], [], []
    d = c["dimension"]
    steps = np.arange(c["epochs"] + 1) * d
    for pair in range(c["pairs"]):
        offset = 1000 * pair
        teacher, student = vectors(d, c["signal_fraction"], (42 + offset, 43 + offset))
        for r in range(c["replicas"]):
            basis = pca_basis(
                np.random.default_rng(100 + offset + r),
                c["alpha"],
                d,
                c["signal_fraction"],
                c["delta"],
            )
            states, predictions, empirical = [], [], []
            for n in c["thresholds"]:
                cfg = Dynamics(
                    d,
                    c["episode_length"],
                    c["delta"],
                    optimal=False,
                    spherical=False,
                    threshold=n,
                )
                g = theory_geometry(c["alpha"], cfg, c, teacher, student)
                predictions.append(trajectory(cfg, g, 0, c["epochs"], steps).states)
                result = simulate(
                    cfg,
                    basis,
                    teacher,
                    student,
                    0,
                    c["epochs"],
                    200 + offset + r,
                    steps,
                )
                states.append(result.states)
                empirical.append(result.conditional)
            sims.append(states)
            theory.append(predictions)
            conditional.append(empirical)
    return dict(
        theory=theory,
        simulation=sims,
        conditional=conditional,
        steps=result.steps,
        switch_step=0,
    )


def plot(data, c):
    fig, ax = plt.subplots()
    x = 1 + data["steps"] / c["dimension"]
    for i, n in enumerate(c["thresholds"]):
        sims = accuracy(data["simulation"][:, i], c["delta"])
        mean, sd = sims.mean(0), sims.std(0, ddof=1)
        ax.plot(
            x,
            accuracy(data["theory"][:, i], c["delta"]).mean(0),
            color=f"C{i}",
            label=str(n),
        )
        ax.plot(x, mean, ":", color=f"C{i}")
        ax.fill_between(x, mean - sd, mean + sd, color=f"C{i}", alpha=0.16)
    ax.set(xscale="log", xlabel="1 + RL epochs", ylabel="Clean accuracy")
    ax.legend(title="Threshold")
    return fig


if __name__ == "__main__":
    run(__doc__, "reward_stringency", DEFAULTS, QUICK, compute, plot)
