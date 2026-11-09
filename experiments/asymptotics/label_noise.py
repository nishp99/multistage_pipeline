"""E05. Converged SFT excess error relative to clean labels at the same alpha."""

import numpy as np
import matplotlib.pyplot as plt
from experiments.common import run
from experiments.dynamics.shared import theory_geometry
from multistage_pipeline.dynamics.pipeline import Dynamics, accuracy
from multistage_pipeline.dynamics.asymptotics import sft_fixed_point

DEFAULTS = dict(
    dimension=400,
    episode_length=6,
    alphas=[0.4, 0.8, 1.6, 3.2],
    delta=0.1,
    signal_fraction=0.2,
    epsilons=[0.0, *np.geomspace(1e-5, 0.4, 80)],
    warm_epochs=1000,
)
QUICK = dict(
    dimension=30, alphas=[0.8, 3.2], epsilons=[0.0, 0.01, 0.1, 0.3], warm_epochs=100
)


def compute(c):
    if c["epsilons"][0] != 0:
        raise ValueError("The first epsilon must be the clean-label reference.")
    states = []
    for a in c["alphas"]:
        row = []
        for eps in c["epsilons"]:
            cfg = Dynamics(c["dimension"], c["episode_length"], c["delta"], eps)
            row.append(
                sft_fixed_point(cfg, theory_geometry(a, cfg, c), c["warm_epochs"])
            )
        states.append(row)
    errors = 1 - accuracy(states, c["delta"])
    return dict(states=states, error=errors, excess=errors - errors[:, :1])


def plot(data, c):
    fig, ax = plt.subplots()
    x = 1 / np.array(c["epsilons"][1:])
    for a, row in zip(c["alphas"], data["excess"]):
        keep = (x <= 20000) & (row[1:] > 0)
        ax.plot(x[keep], row[1:][keep], label=f"α={a}")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Inverse label noise",
        ylabel="Excess SFT error",
    )
    ax.legend()
    return fig


if __name__ == "__main__":
    run(__doc__, "label_noise", DEFAULTS, QUICK, compute, plot)
