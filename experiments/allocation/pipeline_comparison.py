"""E11. Compute-matched SFT, RL and mixed training with empirical endpoints."""

from dataclasses import replace
import matplotlib.pyplot as plt
from experiments.common import run
import numpy as np
from multistage_pipeline.allocation.search import optimize
from experiments.allocation.shared import BASE, QUICK, model, table, row
from experiments.allocation.grids import MODE_BUDGETS
from experiments.dynamics.shared import ensemble, batch_summary
from multistage_pipeline.dynamics.pipeline import accuracy

MODES = ("sft", "rl", "mixed")
DEFAULTS = BASE | dict(
    alpha_min=0.9,
    alpha_max=1.0,
    epsilon=0.3,
    cost=20000.0,
    grid=11,
    audit_grid=201,
    audit_budgets=[18000.001, 18576.271186440677, 20000.0],
    budgets=MODE_BUDGETS,
    empirical_offsets=[1, 10, 100],
    simulation_dimension=1000,
    batches=10,
    replicas=5,
    seed=20260923,
    plot_max_offset=1000,
)
SMALL = QUICK | dict(
    cost=20.0,
    audit_grid=5,
    audit_budgets=[19.0],
    alpha_min=0.9,
    alpha_max=1.5,
    budgets=[18.0, 19.0, 22.0, 28.0, 32.0],
    empirical_offsets=[1, 4],
    simulation_dimension=40,
    batches=2,
    replicas=2,
    plot_max_offset=14,
)


def compute(c):
    cfg, traces = model(c), table(c)
    floor = c["alpha_min"] * c["cost"]
    budgets = np.union1d(c["budgets"], floor + np.array(c["empirical_offsets"]))
    rows = np.array(
        [
            [
                row(
                    optimize(
                        cfg,
                        traces,
                        b,
                        c["cost"],
                        mode=mode,
                        grid=c["audit_grid"] if b in c["audit_budgets"] else c["grid"],
                    )
                )
                for b in budgets
            ]
            for mode in MODES
        ]
    )
    sim = replace(cfg, dimension=c["simulation_dimension"])
    endpoints, samples, cache = [], {}, {}
    for m in range(len(MODES)):
        for offset in c["empirical_offsets"]:
            i = np.flatnonzero(np.isclose(budgets, offset + floor, rtol=0, atol=1e-8))[
                0
            ]
            a, s, r, _ = rows[m, i]
            realized = (
                round(a * sim.dimension) / sim.dimension,
                round(s * sim.dimension) / sim.dimension,
                round(r * sim.dimension) / sim.dimension,
            )
            if realized not in cache:
                group = ensemble(
                    realized[0],
                    sim,
                    c,
                    *realized[1:],
                    balanced=True,
                    seed_offset=len(cache),
                )
                index = len(cache)
                samples.update(
                    {f"replay_{index}_{name}": value for name, value in group.items()}
                )
                batch = accuracy(group["simulation"][:, :, -1, :], c["delta"])
                cache[realized] = (*batch_summary(batch), index)
            endpoints.append([m, offset, *realized, *cache[realized]])
    return dict(rows=rows, budgets=budgets, endpoints=endpoints, **samples)


def plot(data, c):
    fig, ax = plt.subplots()
    x = data["budgets"] - c["cost"] * c["alpha_min"]
    keep = (x > 0) & (x <= c["plot_max_offset"] + 1e-8)
    for m, name in enumerate(MODES):
        ax.plot(x[keep], data["rows"][m, keep, 3], label=name.upper(), color=f"C{m}")
        ends = data["endpoints"][data["endpoints"][:, 0] == m]
        ax.errorbar(ends[:, 1], ends[:, 5], yerr=ends[:, 6], fmt="o", color=f"C{m}")
    ax.set(
        xlabel="Budget above minimum pretraining cost",
        xscale="log",
        ylabel="Clean accuracy",
    )
    ax.legend()
    return fig


if __name__ == "__main__":
    run(__doc__, "pipeline_comparison", DEFAULTS, SMALL, compute, plot)
