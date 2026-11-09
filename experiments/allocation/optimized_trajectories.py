"""E08, Fig. 2d. Optimize prescriptions and replay at finite dimension."""

import matplotlib.pyplot as plt
from dataclasses import replace
from experiments.common import run
import numpy as np
from multistage_pipeline.allocation.search import optimize
from experiments.allocation.shared import BASE, QUICK, model, table, row
from experiments.dynamics.shared import ensemble, batch_summary
from multistage_pipeline.dynamics.pipeline import accuracy

DEFAULTS = BASE | dict(
    cost=2000.0,
    budgets=[1906.78, 5827.12, 13667.8],
    simulation_dimension=1000,
    trace_points=200,
    grid=11,
    batches=20,
    replicas=5,
    seed=20260915,
    cap=6300,
)
SMALL = QUICK | dict(
    cost=30.0,
    alpha_min=0.4,
    alpha_max=1.5,
    budgets=[26.0, 48.0, 75.0],
    simulation_dimension=40,
    batches=2,
    replicas=2,
    cap=100,
)


def compute(c):
    traces = table(c)
    out, rows, realized = {}, [], []
    cfg = model(c)
    sim = replace(cfg, dimension=c["simulation_dimension"])
    for i, b in enumerate(c["budgets"]):
        p = optimize(cfg, traces, b, c["cost"], grid=c["grid"], cap=c["cap"])
        a = round(p.alpha * sim.dimension) / sim.dimension
        s, r = round(p.sft), round(p.rl)
        group = ensemble(
            a, sim, c, s, r, balanced=True, seed_offset=i, every_epoch=True
        )
        out.update({f"{name}_{i}": value for name, value in group.items()})
        rows.append(row(p))
        realized.append([a, s, r, a * c["cost"] + s + r])
    return dict(**out, rows=rows, realized=realized)


def plot(data, c):
    fig, ax = plt.subplots()
    for i, b in enumerate(c["budgets"]):
        x = data[f"steps_{i}"] / c["simulation_dimension"] + 1
        mean, sd = batch_summary(accuracy(data[f"simulation_{i}"], c["delta"]))
        ax.plot(
            x,
            accuracy(data[f"theory_{i}"], c["delta"]),
            color=f"C{i}",
            label=f"B={b:g}",
        )
        ax.plot(x, mean, ":", color=f"C{i}")
        ax.fill_between(x, mean - sd, mean + sd, color=f"C{i}", alpha=0.18)
        ax.axvline(
            1 + data[f"switch_step_{i}"] / c["simulation_dimension"],
            color=f"C{i}",
            alpha=0.4,
        )
    ax.set(xlabel="1 + completed epochs", xscale="log", ylabel="Clean accuracy")
    ax.legend()
    return fig


if __name__ == "__main__":
    run(__doc__, "optimized_trajectories", DEFAULTS, SMALL, compute, plot)
