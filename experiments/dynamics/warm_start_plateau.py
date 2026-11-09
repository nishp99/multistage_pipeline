"""E07. Pure RL and RL after a short SFT warm start."""

import matplotlib.pyplot as plt
from experiments.common import run
from experiments.dynamics.shared import theory_geometry
from multistage_pipeline.dynamics.pipeline import Dynamics, trajectory, accuracy

DEFAULTS = dict(
    alpha=4.0,
    dimension=2000,
    episode_length=12,
    epsilon=0.2,
    signal_fraction=0.2,
    delta=0.1,
    total_epochs=2000,
    sft_epochs=[0, 1, 2, 3],
)
QUICK = dict(dimension=40, total_epochs=20)


def compute(c):
    cfg = Dynamics(c["dimension"], c["episode_length"], c["delta"], c["epsilon"])
    geom = theory_geometry(c["alpha"], cfg, c)
    arrays = {}
    for i, warm in enumerate(c["sft_epochs"]):
        result = trajectory(cfg, geom, warm, c["total_epochs"] - warm)
        arrays.update({f"{k}_{i}": v for k, v in result._asdict().items()})
    return arrays


def plot(data, c):
    fig, ax = plt.subplots()
    for i, warm in enumerate(c["sft_epochs"]):
        elapsed = (data[f"steps_{i}"] - data[f"switch_step_{i}"]) / c["dimension"]
        keep = elapsed >= 0
        ax.plot(
            elapsed[keep],
            accuracy(data[f"states_{i}"], c["delta"])[keep],
            label=str(warm),
        )
    ax.set(xscale="symlog", xlabel="Completed RL epochs", ylabel="Clean accuracy")
    ax.legend(title="SFT epochs")
    return fig


if __name__ == "__main__":
    run(__doc__, "warm_start_plateau", DEFAULTS, QUICK, compute, plot)
