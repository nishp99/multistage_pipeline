"""E09. Optimal stage shares versus SFT label noise at fixed total budget."""

from experiments.common import run
import numpy as np
from multistage_pipeline.allocation.search import optimize
from experiments.allocation.shared import (
    BASE,
    QUICK,
    model,
    table,
    row,
    shares,
    shares_plot,
)

DEFAULTS = BASE | dict(
    budget=100000.0,
    cost=20000.0,
    epsilons=np.linspace(0, 0.5, 10).round(6).tolist(),
    trace_points=200,
    cap=6300,
    grid=11,
)
SMALL = QUICK | dict(budget=40.0, cost=25.0, epsilons=[0.0, 0.2, 0.5], cap=40)


def compute(c):
    traces = table(c)
    rows = np.array(
        [
            row(
                optimize(
                    model(c | dict(epsilon=e)),
                    traces,
                    c["budget"],
                    c["cost"],
                    grid=c["grid"],
                    cap=c["cap"],
                )
            )
            for e in c["epsilons"]
        ]
    )
    return dict(
        rows=rows,
        shares=shares(rows, c["cost"], [c["budget"]] * len(rows)),
        cap_active=np.any(rows[:, 1:3] > c["cap"], axis=1),
    )


def plot(data, c):
    return shares_plot(c["epsilons"], data["shares"], "SFT label-flip probability")


if __name__ == "__main__":
    run(__doc__, "noise_shares", DEFAULTS, SMALL, compute, plot)
