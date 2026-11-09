"""E03. SFT and RL errors versus alpha, with the leading Bayes asymptote."""

import numpy as np
import matplotlib.pyplot as plt
from experiments.common import run
from experiments.dynamics.shared import theory_geometry
from experiments.asymptotics.grids import ALPHAS, WARM_EPOCHS
from multistage_pipeline.dynamics.pipeline import Dynamics, accuracy, advance, step_args
from multistage_pipeline.dynamics.asymptotics import (
    sft_fixed_point,
    warm_start_reference,
)

DEFAULTS = dict(
    dimension=400,
    episode_length=6,
    alphas=ALPHAS,
    delta=0.1,
    signal_fraction=0.2,
    epsilons=[0.0, 0.3],
    warm_epochs=1000,
    rl_epochs=10000,
    tail_rl_epochs=40000,
    warm_grid=WARM_EPOCHS,
)
QUICK = dict(
    dimension=20,
    alphas=[0.5, 1.6],
    warm_epochs=100,
    rl_epochs=500,
    tail_rl_epochs=1200,
    warm_grid=np.unique(np.r_[0, np.geomspace(1, 4000, 38).astype(int)]).tolist(),
)


def compute(c):
    sft, pure, warm, methods = [], [], [], []
    for a in c["alphas"]:
        cfg = Dynamics(c["dimension"], c["episode_length"], c["delta"], 0.0)
        geom = theory_geometry(a, cfg, c)
        pure.append(
            np.asarray(
                advance(
                    geom.initial,
                    geom,
                    step_args(cfg),
                    cfg.dimension * c["rl_epochs"],
                    "rl",
                )
            )
        )
        grid = c["warm_grid"]
        if not c["quick"] and a not in (0.4, 1.6, 6.4):
            grid = [t for t in grid if t <= 20000]
        tail = warm_start_reference(cfg, geom, grid, c["tail_rl_epochs"], a)
        warm.append(tail["reference"])
        methods.append(tail["method"])
        row = []
        for eps in c["epsilons"]:
            model = Dynamics(cfg.dimension, cfg.episode_length, cfg.delta, eps)
            row.append(sft_fixed_point(model, geom, c["warm_epochs"]))
        sft.append(row)
    bayes = (
        c["delta"]
        * np.sqrt(c["signal_fraction"])
        / (np.pi * (1 - c["delta"]) * np.sqrt(c["alphas"]))
    )
    return dict(
        sft_states=sft,
        rl_states=pure,
        warm_rl_error=warm,
        reference_method=methods,
        bayes=bayes,
    )


def plot(data, c):
    fig, ax = plt.subplots()
    for i, eps in enumerate(c["epsilons"]):
        ax.plot(
            c["alphas"],
            1 - accuracy(data["sft_states"][:, i], c["delta"]),
            label=f"SFT ε={eps}",
        )
    for values, label in [
        (1 - accuracy(data["rl_states"], c["delta"]), "RL"),
        (data["warm_rl_error"], "Warm-start RL"),
        (data["bayes"], "Bayes asymptote"),
    ]:
        ax.plot(c["alphas"], values, label=label)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Pretraining sample ratio α",
        ylabel="Clean error",
    )
    ax.legend()
    return fig


if __name__ == "__main__":
    run(__doc__, "pretraining_quality", DEFAULTS, QUICK, compute, plot)
