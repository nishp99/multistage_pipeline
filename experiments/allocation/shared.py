"""Common settings and budget accounting for the allocation panels."""

import numpy as np
import matplotlib.pyplot as plt
from multistage_pipeline.allocation.search import TraceTable
from multistage_pipeline.dynamics.pipeline import Dynamics

BASE = dict(
    dimension=400,
    delta=0.1,
    episode_length=6,
    signal_fraction=0.2,
    epsilon=0.2,
    alpha_min=0.3,
    alpha_max=10.0,
    trace_points=400,
    grid=201,
)
QUICK = dict(dimension=20, trace_points=6, grid=5)


def model(c):
    return Dynamics(c["dimension"], c["episode_length"], c["delta"], c["epsilon"])


def table(c, knots=None):
    if knots is None:
        knots = np.linspace(c["alpha_min"], c["alpha_max"], c["trace_points"])
    return TraceTable(knots, c["signal_fraction"], c["delta"], c["quick"])


def shares_plot(x, shares, xlabel):
    fig, ax = plt.subplots()
    ax.stackplot(x, shares.T, labels=["Pretraining", "SFT", "RL"])
    ax.set(xlabel=xlabel, ylabel="Fraction of budget", ylim=(0, 1))
    ax.legend()
    return fig


def row(p):
    return np.array([p.alpha, p.sft, p.rl, p.accuracy])


def shares(rows, cost, budgets):
    return rows[:, :3] * np.array([cost, 1.0, 1.0]) / np.asarray(budgets)[:, None]
