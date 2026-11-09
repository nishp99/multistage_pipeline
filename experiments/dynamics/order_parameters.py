"""E02. Compare analytic order parameters and independent microscopic ensembles."""

import matplotlib.pyplot as plt
from experiments.common import run
from experiments.dynamics.shared import batch_summary, ensemble
from multistage_pipeline.dynamics.pipeline import Dynamics

DEFAULTS = dict(
    dimension=1000,
    signal_fraction=0.2,
    delta=0.1,
    alpha=0.9,
    episode_length=6,
    epsilon=0.05,
    epochs=1000,
    batches=10,
    replicas=5,
    seed=20260922,
)
QUICK = dict(dimension=40, epochs=5, batches=2, replicas=2)


def compute(c):
    result = {}
    for phase in ("sft", "rl"):
        rl = phase == "rl"
        cfg = Dynamics(
            c["dimension"],
            c["episode_length"],
            c["delta"],
            c["epsilon"],
            optimal=rl,
            spherical=rl,
        )
        group = ensemble(
            c["alpha"], cfg, c, 0 if rl else c["epochs"], c["epochs"] if rl else 0
        )
        result.update({f"{phase}_{k}": v for k, v in group.items()})
    return result


def plot(data, c):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, phase in zip(axs, ("sft", "rl")):
        x = 1 + data[f"{phase}_steps"] / c["dimension"]
        mean, sd = batch_summary(data[f"{phase}_simulation"])
        for i, label in enumerate(("J signal", "J noise", "Q signal", "Q noise")):
            ax.plot(x, data[f"{phase}_theory"][:, i], color=f"C{i}", label=label)
            ax.plot(x, mean[:, i], ":", color=f"C{i}")
            ax.fill_between(
                x,
                mean[:, i] - sd[:, i],
                mean[:, i] + sd[:, i],
                color=f"C{i}",
                alpha=0.16,
            )
        ax.set(xscale="log", xlabel="1 + completed epochs", title=phase.upper())
    axs[0].legend()
    return fig


if __name__ == "__main__":
    run(__doc__, "order_parameters", DEFAULTS, QUICK, compute, plot)
