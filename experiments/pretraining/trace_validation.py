"""E01. RMT, sample PCA, and trained tied-LAE projector moments."""

import matplotlib.pyplot as plt
import numpy as np
from jax import random
from experiments.common import run
from multistage_pipeline.pretraining.projector import pca_basis, spectral_moments
from multistage_pipeline.pretraining.autoencoder import fit_encoder, encoder_basis
from multistage_pipeline.pretraining.traces import five_moments

DEFAULTS = dict(
    dimension=2000,
    signal_fraction=0.2,
    delta=0.1,
    steps=40000,
    learning_rate=5e-4,
    penalty=1e-4,
    lae_trials=20,
    pca_trials=32,
    seed=42,
    pca_seed=921000,
    lae_alphas=[
        0.3,
        0.347,
        0.402,
        0.465,
        0.538,
        0.623,
        0.721,
        0.834,
        0.9655,
        1.1175,
        1.293,
        1.4965,
        1.732,
        2.0045,
        2.32,
        2.685,
        3.107,
        3.596,
        4.162,
        4.8165,
        5.5745,
        6.451,
        7.466,
        8.6405,
        10.0,
    ],
    pca_alphas=[
        0.205,
        0.22,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.625,
        0.65,
        1.0,
        1.2,
        1.4,
        1.6,
        2.0,
        4.0,
        10.0,
    ],
)
QUICK = dict(
    dimension=40,
    steps=3000,
    lae_trials=2,
    pca_trials=3,
    lae_alphas=[0.5, 1.5],
    pca_alphas=[0.5, 1.5],
)


def compute(c):
    d, f, delta = c["dimension"], c["signal_fraction"], c["delta"]
    keys = random.split(random.PRNGKey(c["seed"]), c["lae_trials"])
    lae = [
        [
            spectral_moments(
                encoder_basis(
                    fit_encoder(
                        key,
                        round(a * d),
                        d,
                        round(d * f),
                        delta,
                        c["steps"],
                        c["learning_rate"],
                        c["penalty"],
                    )
                )
            )[1:]
            for key in keys
        ]
        for a in c["lae_alphas"]
    ]
    # Case numbers preserve the Monte Carlo streams from the trace-validation sweep.
    case_ids = [0, 1, 2, 3, 4, 5, 19, 20, 21, 6, 16, 17, 18, 7, 8, 9]
    cases = dict(zip(DEFAULTS["pca_alphas"], case_ids))
    pca = []
    for i, a in enumerate(c["pca_alphas"]):
        rng = np.random.default_rng(
            c["pca_seed"] + 10000 * cases.get(a, len(case_ids) + i) + d
        )
        pca.append(
            [
                spectral_moments(pca_basis(rng, a, d, f, delta, center=False))[1:]
                for _ in range(c["pca_trials"])
            ]
        )
    alphas = np.union1d(c["pca_alphas"], c["lae_alphas"])
    low_cost = (
        dict(
            n_grid1d=256,
            n_grid2=32,
            n_grid3=20,
            integration_options=dict(
                powers=(9, 11, 13), scrambles=4, batch_size=1024, atol=0.003, rtol=0.1
            ),
        )
        if c["quick"]
        else {}
    )
    theory = [five_moments(float(a), f, delta, **low_cost) for a in alphas]
    return dict(alphas=alphas, theory=np.asarray(theory), lae=lae, pca=pca)


def plot(data, c):
    fig, ax = plt.subplots()
    for k in range(5):
        color = f"C{k}"
        ax.plot(data["alphas"], data["theory"][:, k], color=color, label=f"m{k+1}")
        for method, marker in [("lae", "o"), ("pca", "x")]:
            values = data[method][:, :, k]
            ax.errorbar(
                c[f"{method}_alphas"],
                values.mean(1),
                values.std(1),
                color=color,
                fmt=marker,
                markersize=3,
            )
    ax.set(
        xscale="log",
        xlabel="Pretraining sample ratio α",
        ylabel="Normalized trace moment",
        title="RMT (line), tied LAE (circle), PCA (cross)",
    )
    ax.legend()
    return fig


if __name__ == "__main__":
    run(__doc__, "trace_validation", DEFAULTS, QUICK, compute, plot)
