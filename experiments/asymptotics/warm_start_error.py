"""E04. Excess RL error relative to an extrapolated long-SFT warm start."""

import numpy as np
import matplotlib.pyplot as plt
from experiments.common import run
from experiments.dynamics.shared import theory_geometry
from experiments.asymptotics.grids import WARM_EPOCHS
from multistage_pipeline.dynamics.pipeline import Dynamics
from multistage_pipeline.dynamics.asymptotics import warm_start_reference

DEFAULTS = dict(
    dimension=400,
    episode_length=6,
    alphas=[0.4, 1.6, 6.4],
    epsilon=0.2,
    delta=0.1,
    signal_fraction=0.2,
    warm_epochs=WARM_EPOCHS,
    rl_epochs=40000,
)
QUICK = dict(
    dimension=20,
    alphas=[1.6],
    warm_epochs=np.unique(np.r_[0, np.geomspace(1, 4000, 38).astype(int)]).tolist(),
    rl_epochs=1200,
)


def compute(c):
    cfg = Dynamics(c["dimension"], c["episode_length"], c["delta"], c["epsilon"])
    rows = [
        warm_start_reference(
            cfg, theory_geometry(a, cfg, c), c["warm_epochs"], c["rl_epochs"], a
        )
        for a in c["alphas"]
    ]
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def plot(data, c):
    fig, ax = plt.subplots()
    x = np.asarray(c["warm_epochs"])[1:]
    for i, a in enumerate(c["alphas"]):
        error = data["errors"][i, -1, 1:]
        ref = data["reference"][i]
        if np.isfinite(ref):
            ax.plot(x, np.where(error > ref, error - ref, np.nan), label=f"α={a}")
        else:
            ax.plot(x, error, ":", label=f"α={a}, raw error (tail unresolved)")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="SFT warm-start epochs",
        ylabel="RL error above fitted reference",
    )
    ax.legend()
    return fig


if __name__ == "__main__":
    run(__doc__, "warm_start_error", DEFAULTS, QUICK, compute, plot)
