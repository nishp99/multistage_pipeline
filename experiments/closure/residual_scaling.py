"""E14. Evolution and dimension scaling of exact versus closed ss contractions."""

import matplotlib.pyplot as plt
import numpy as np
from jax import random
from experiments.common import run
from multistage_pipeline.dynamics.pipeline import Dynamics
from multistage_pipeline.dynamics.simulation import simulate
from multistage_pipeline.pretraining.projector import (
    vectors,
    pca_basis,
    measured_geometry,
    population_geometry,
    population_moments,
)

DEFAULTS = dict(
    dimensions=[400, 800, 1600],
    alphas=[0.4, 0.5],
    signal_fraction=0.2,
    delta=0.1,
    episode_length=5,
    epochs=[*range(11), *range(20, 101, 10)],
    disorders=128,
    projections=4,
    bootstrap=5000,
    bootstrap_seed=20260920,
)
QUICK = dict(
    dimensions=[30, 50, 80],
    epochs=[0, 1, 2, 4],
    disorders=4,
    projections=2,
    bootstrap=100,
)


def scaling(values, dimensions, bootstrap, rng):
    """values[D,disorder,time,coefficient source,contraction] already average over P."""
    x = np.log(dimensions)
    weights = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
    slope = np.tensordot(weights, np.log(values.mean(axis=1)), axes=1)
    # Resample disorders independently within each D. Reuse each sampled
    # trajectory across time; the reported percentile intervals are pointwise.
    fits = []
    for start in range(0, bootstrap, 100):
        size = min(100, bootstrap - start)
        boot = np.zeros((size, *slope.shape))
        for v, w in zip(values, weights):
            idx = rng.integers(len(v), size=(size, len(v)))
            boot += w * np.log(v[idx].mean(1))
        fits.append(boot)
    ci = np.percentile(np.concatenate(fits), [2.5, 97.5], axis=0)
    return slope, ci


def compute(c):
    all_values, slopes, intervals = [], [], []
    for a in c["alphas"]:
        by_dim = []
        for d in c["dimensions"]:
            cfg = Dynamics(
                d,
                c["episode_length"],
                c["delta"],
                epsilon=0.0,
                optimal=False,
                spherical=True,
            )
            ana = population_geometry(
                population_moments(a, c["signal_fraction"], cfg.delta, c["quick"])
            )
            disorders = []
            for seed in range(c["disorders"]):
                teacher, student = vectors(
                    d, c["signal_fraction"], (420000 + seed, 430000 + seed)
                )
                residuals = []
                for r in range(c["projections"]):
                    key = random.fold_in(random.PRNGKey(10000 + seed), r)
                    b = pca_basis(key, a, d, c["signal_fraction"], cfg.delta)
                    emp = measured_geometry(b, teacher, student)
                    sim = simulate(
                        cfg,
                        b,
                        teacher,
                        student,
                        max(c["epochs"]),
                        0,
                        20000 + seed + 100000 * r,
                        np.asarray(c["epochs"]) * d,
                    )
                    exact = sim.spectral[:, [2, 5]]
                    blocks = sim.states.reshape(-1, 2, 2)
                    estimated = np.stack([blocks @ g.omega for g in (emp, ana)], axis=1)
                    exact = exact[:, None, :]
                    residuals.append(
                        np.abs(exact - estimated)
                        / (np.abs(exact) + np.abs(estimated) + 1e-12)
                    )
                disorders.append(np.mean(residuals, axis=0))
            by_dim.append(disorders)
        by_dim = np.asarray(by_dim)
        slope, ci = scaling(
            by_dim,
            c["dimensions"],
            c["bootstrap"],
            np.random.default_rng(c["bootstrap_seed"]),
        )
        all_values.append(by_dim)
        slopes.append(slope)
        intervals.append(ci)
    return dict(
        disorder_means=all_values,
        slopes=slopes,
        ci=intervals,
        epochs=c["epochs"],
        microsteps=np.outer(c["dimensions"], c["epochs"]),
    )


def plot(data, c):
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.5), sharex=True)
    x = data["epochs"]
    for h, channel in enumerate(("J", "Q")):
        for m, method in enumerate(("Empirical HS", "Analytic RMT")):
            left, right = axs[h, m], axs[h, m + 2]
            for i, a in enumerate(c["alphas"]):
                left.plot(x, data["slopes"][i, :, m, h], label=f"α={a}")
                left.fill_between(x, *data["ci"][i, :, :, m, h], alpha=0.16)
            left.axhline(-0.5, ls=":", color="gray")
            left.axhline(0.0, ls=":", color="gray")
            for i, d in enumerate(c["dimensions"]):
                curves = data["disorder_means"][-1, i, :, :, m, h]
                right.plot(x, curves.T, color=f"C{i}", alpha=0.06)
                right.plot(x, curves.mean(0), color=f"C{i}", label=f"D={d}")
            left.set(ylabel=f"{channel} scaling exponent", title=method)
            right.set(yscale="log", ylabel=f"{channel} relative residual", title=method)
            if h == 1:
                left.set_xlabel("SFT epochs")
                right.set_xlabel("SFT epochs")
    axs[0, 0].legend()
    axs[0, 2].legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    run(__doc__, "residual_scaling", DEFAULTS, QUICK, compute, plot)
