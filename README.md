# Multistage training pipelines

Code accompanying **An Analytical Model of Compute-limited Multistage Training Pipelines**. The experiments study how pretraining dataset size, supervised finetuning (SFT), and reward-based finetuning (RL) affect a projected student's learning dynamics.

[Find an experiment by paper figure](experiments/README.md)

## Run a first experiment

Use Python 3.11 and a CPU. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m experiments.dynamics.warm_start_plateau --quick
```

Open `outputs/warm_start_plateau/figure.pdf`. The curves show RL accuracy after zero, one, two, or three SFT epochs. All use the same solver. The horizontal coordinate starts at the switch to RL.

Each script runs its complete scientific calculation on generated inputs. Omit `--quick` for the paper-scale settings. The LAE ensemble trains 20 encoders per sample size, and the allocation and tail sweeps require substantially more compute than the reduced examples. Use `--show-config` to inspect a run before launching it. To vary the plateau experiment:

```bash
python -m experiments.dynamics.warm_start_plateau --quick --show-config > plateau.json
# Edit alpha, sft_epochs, or total_epochs in plateau.json.
python -m experiments.dynamics.warm_start_plateau --quick --config plateau.json --output-dir outputs/my_plateau
python -m experiments.dynamics.warm_start_plateau --plot-only --output-dir outputs/my_plateau
```

The output directory contains `result.npz` with named numeric arrays and the resolved settings in `config_json`, and `figure.pdf` rendered from that result. Use a separate output directory to retain multiple runs. The `dev` extra installs notebook and test tools; the base installation suffices for the experiment scripts.

## Explore the equations

- [SFT followed by RL](notebooks/sft_then_rl.ipynb) exposes the stage durations, label noise, reward threshold, and learning-rate rule.
- [Comparing closures](notebooks/comparing_closures.ipynb) runs the four-state and six-state equations and the microscopic simulator on a shared finite-dimensional projector.

Open these with `jupyter lab` in the installed environment. Both notebooks calculate their own inputs.

| Code | Scientific role |
|---|---|
| `multistage_pipeline/pretraining` | Gaussian data, empirical PCA and tied-LAE row spaces, RMT moments, and validated spectral geometry. |
| `multistage_pipeline/dynamics` | Four-state Hilbert–Schmidt and six-state degree-two equations, microscopic simulation, fixed points, and warm-start tail estimation. |
| `multistage_pipeline/allocation` | Trace interpolation, budget constraints, and search over pretraining quality and SFT/RL shares. |
| `experiments` | Parameter presets, ensemble aggregation, and plotting for each scientific question. |

**Conventions.** `signal_fraction=0.2` gives a retained rank of `round(0.2 * dimension)`; the internal RMT noise fraction is 0.8. The sample ratio is α = N/D. SFT duration and RL duration count completed epochs, with D microscopic updates per epoch. Trajectories retain the initial, switch, and terminal states and their actual update counts. For the four-state model the state order is `(J_signal, J_noise, Q_signal, Q_noise)`. The six-state model uses `(J0, J1, J2, Q0, Q1, Q2)` in the basis of the retained projector, its signal contraction C, and C².

The four-state RL reward is one when at least `threshold` decisions in an episode agree with the teacher. The default threshold is the episode length. Spherical normalization preserves the initial projected norm. With locally optimal rates, each simulated student uses its realized projector's conditional ODE rate at the same microscopic update. The analytic RMT curve evolves independently.

## Verification

Run `python -m pytest -q`. The tests cover phase boundaries, threshold rewards, replay of the microscopic learning rate, spherical normalization, degree-two geometry, budget boundaries, and residual scaling.

Release verification includes reduced runs of all fourteen scripts and execution of both notebooks, plus the paper-setting comparisons in the [experiment guide](experiments/README.md#numerical-checks). Full paper-scale ensemble and optimization sweeps were not rerun for this release. The release covers the projected model and converged-LAE validation; the collaborator-owned skill-composition experiments and OLMo experiments are deferred.
