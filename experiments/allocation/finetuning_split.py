"""E10. Optimal RL fraction of finetuning compute at fixed pretraining quality."""

import matplotlib.pyplot as plt
from experiments.common import run
import numpy as np
from multistage_pipeline.allocation.search import optimize
from experiments.allocation.shared import BASE, QUICK, model, table, row

DEFAULTS = BASE | dict(
    epsilon=0.3,
    alphas=[0.4, 1.6, 3.8, 6.4],
    budgets=np.linspace(0, 3000, 35).tolist(),
    audit_indices=[1, 17, 34],
    audit_grid=2001,
    refinement_spacing=0.0005,
)
SMALL = QUICK | dict(
    alphas=[0.4, 1.6], budgets=[0.0, 4.0, 12.0], audit_grid=5, refinement_spacing=0.05
)


def compute(c):
    traces = table(c, c["alphas"])
    rows = np.array(
        [
            [
                row(
                    optimize(
                        model(c),
                        traces,
                        b,
                        0.0,
                        fixed_alpha=a,
                        grid=c["audit_grid"] if i in c["audit_indices"] else c["grid"],
                        fixed_spacing=c["refinement_spacing"],
                    )
                )
                for i, b in enumerate(c["budgets"])
            ]
            for a in c["alphas"]
        ]
    )
    total = rows[:, :, 1] + rows[:, :, 2]
    share = np.divide(
        rows[:, :, 2], total, where=total > 0, out=np.full_like(total, np.nan)
    )
    return dict(rows=rows, rl_share=share)


def plot(data, c):
    fig, ax = plt.subplots()
    for a, y in zip(c["alphas"], data["rl_share"]):
        ax.plot(c["budgets"], y, label=f"α={a}")
    ax.set(xlabel="Finetuning budget", ylabel="RL fraction")
    ax.legend()
    return fig


if __name__ == "__main__":
    run(__doc__, "finetuning_split", DEFAULTS, SMALL, compute, plot)
