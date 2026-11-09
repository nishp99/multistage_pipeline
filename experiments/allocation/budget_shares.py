"""E08, Fig. 5a. Optimize all three stages across total budgets."""

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
from experiments.allocation.grids import BUDGET_KNOTS

DEFAULTS = BASE | dict(
    grid=11,
    audit_grid=201,
    audit_indices=[0, 30, 59],
    cost=1000.0,
    budgets=np.linspace(600, 20000, 60).tolist(),
    knots=BUDGET_KNOTS,
)
SMALL = QUICK | dict(
    audit_grid=5, cost=20.0, budgets=[8.0, 16.0, 32.0], alpha_max=1.6, knots=None
)


def compute(c):
    traces = table(c, c["knots"])
    rows = np.array(
        [
            row(
                optimize(
                    model(c),
                    traces,
                    b,
                    c["cost"],
                    grid=c["audit_grid"] if i in c["audit_indices"] else c["grid"],
                )
            )
            for i, b in enumerate(c["budgets"])
        ]
    )
    return dict(rows=rows, shares=shares(rows, c["cost"], c["budgets"]))


def plot(data, c):
    hits = np.flatnonzero(np.isclose(data["rows"][:, 0], c["alpha_max"]))
    stop = hits[0] + 1 if len(hits) else len(c["budgets"])
    return shares_plot(c["budgets"][:stop], data["shares"][:stop], "Total budget")


if __name__ == "__main__":
    run(__doc__, "budget_shares", DEFAULTS, SMALL, compute, plot)
