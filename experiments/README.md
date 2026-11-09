# Paper experiments

Run the module in the last column with `python -m`. Append `--quick` for a small CPU run. The default output directory is `outputs/<module basename>/`. All commands work independently, including the recipes that first optimize a prescription and then simulate it.

| ID | Manuscript panel | Calculation | Module |
|---|---|---|---|
| E01 | Fig. 2a and appendix trace comparison | First five projector–signal trace moments from RMT, PCA Monte Carlo, and trained LAEs | `experiments.pretraining.trace_validation` |
| E02 | Fig. 2b–c | Four order parameters during fixed-rate unconstrained SFT and optimal-rate spherical RL | `experiments.dynamics.order_parameters` |
| E03 | Fig. 3a | SFT and RL errors versus pretraining sample ratio, including the leading Bayes asymptote | `experiments.asymptotics.pretraining_quality` |
| E04 | Fig. 3b | Excess long-run RL error versus SFT warm-start duration | `experiments.asymptotics.warm_start_error` |
| E05 | Fig. 3c | Asymptotic SFT excess error relative to clean labels at the same α | `experiments.asymptotics.label_noise` |
| E06 | Fig. 3d | Paired simulations and theory for six reward thresholds at T=10 | `experiments.dynamics.reward_stringency` |
| E07 | Fig. 4a | Plot RL accuracy against epochs since the SFT→RL switch | `experiments.dynamics.warm_start_plateau` |
| E08 | Fig. 5a | Optimal pretraining, SFT and RL shares versus total budget | `experiments.allocation.budget_shares` |
| E08 | Fig. 2d | Optimize at the three selected budgets and replay each prescription | `experiments.allocation.optimized_trajectories` |
| E09 | Fig. 5b | Optimal stage shares versus label noise | `experiments.allocation.noise_shares` |
| E10 | Fig. 5c | Optimal RL fraction of a fixed finetuning budget | `experiments.allocation.finetuning_split` |
| E11 | Fig. 5d and Fig. 1b preview | Compare three optimized modes and finite-dimensional endpoints | `experiments.allocation.pipeline_comparison` |
| E14 | Appendix projector closure residuals | Measure exact and closed contractions, then fit dimension-scaling exponents | `experiments.closure.residual_scaling` |
| Extension | Optional | Four-state and six-state closures against the same microscopic trajectory | `experiments.closure.compare_degree_two` |

## Reading and changing a result

- **E01.** The selected encoders used 40,000 full-batch Adam steps at `5e-4`, with tied weights and an L2 penalty of `1e-4`. The manuscript reports 10,000 and `1e-4`. The script uses the encoder-run settings, measures all five moments in float64 after QR, and plots population SD across independent runs. The trace-calibration PCA uses the uncentered second moment; downstream simulations use centered covariance.
- **E02 and E06.** In E02, uncertainty is one sample SD across ten PCA-batch means, each averaging five replicas. E06 uses five teacher/student pairs with three replicates per pair, common episodes across thresholds, and descriptive SD over all fifteen trajectories. Its `simulation`, `conditional`, and `theory` arrays distinguish microscopic, realized-projector ODE, and analytic RMT trajectories. The compact closure can retain systematic error, particularly for spherical pure RL. There is no blanket α cutoff for agreement.
- **E03–E05.** SFT fixed points use a warm-up followed by bounded Newton steps. Pure RL in E03 uses its finite 10,000-epoch endpoint. The infinite-SFT references use fresh warm-start grids and RL-duration doubling, retaining the method and resolution status. E04 saves the raw errors as well as the reference. An unresolved offset produces a labeled raw-error curve. E05 measures SFT alone and uses the clean-label result at each α as its baseline.
- **E07.** All curves have α=4, T=12, and the same total training horizon. The elapsed RL clock starts at zero for each SFT duration.
- **E08–E11.** The total budget is `cost × alpha + SFT epochs + RL epochs`. Search interpolates precomputed geometry and, for fractional epochs, interpolates between adjacent epoch endpoints. Endpoints and single-stage choices participate in the search. The compact solver and search implementation are shared across panels. Saved allocation rows contain `[alpha, sft_epochs, rl_epochs, accuracy]`.
- **E08.** The α acquisition costs are 1,000 for Fig. 5a and 2,000 for Fig. 2d. The Fig. 2d script optimizes three budgets independently, rounds the sample count and stage lengths for replay, and saves both the continuous prescription and the realized budget. The empirical replay uses balanced student block norms; E02 uses ordinary random initialization.
- **E09.** The selected run used budget 100,000 and acquisition cost 20,000. The caption gives 50,000. The historical search capped each evaluated phase at 6,300 epochs. `cap_active` identifies prescriptions affected by this cap. The archived result did not record initialization; this release uses the shared population initialization. E09 and E10 historically used earlier spherical equations, so fresh curves can differ.
- **E10 and E11.** E10 charges only finetuning, with an undefined RL fraction at zero budget. E11 accounts for pretraining, uses D=400 for optimization, and realizes the prescriptions at D=1000. Its plotted coordinate is total budget minus the acquisition floor of 18,000. It simulates excess budgets 1, 10, and 100, reusing shared prescriptions. Fresh microscopic ensembles use seeded Gaussian episodes; the sampling implementation can produce different draws from older workers.
- **E14.** The residual is `abs(exact − closed) / (abs(exact) + abs(closed) + 1e-12)`. Average across four PCA projectors within each of 128 teacher/student disorders before fitting log residual versus log D. The intervals are pointwise 95% bootstrap intervals across disorders. The right block uses the final α in the configured grid, 0.5 in the paper.
- **Six-state extension.** The comparison uses measured projector moments and matched initial conditions for both closures. `include_population` adds a separate trajectory with validated analytic geometry and population initialization. Its full-resolution moment integration is the expensive step. The six-state RL equations implement the all-correct reward.

## Source map

These identifiers locate the migration inputs in the research repository's history. The runnable code here has no dependency on those checkouts. The November camera-ready parameter records and manuscript inventory determine the panel mapping.

| Experiments | Source revision and generating code |
|---|---|
| E01 | `d80cacf874`, `Linear_AE/train_autoencoder.py` and `investigations/trace_analysis/higher_moments/validate.py`. Trace integration originated on `e30eeb088f`; the degree-two validation branch `118ea780df` supplies the refined integrator. The untracked November `Linear_AE/plot_trace_moments_v7.py` (SHA-256 prefix `33a229882d5f`) identified the verified alpha grids, QR convention, and SD. |
| E02, E06, E07 | `6eba94724b`, `figure2_comparison.py` and `budget_worker.py` for the E02 prescription and sampling design. `b9d5cafe6b`, `n_or_more/sft_then_rl.py` and `validate_compact.py` for the corrected compact step, threshold reward, and conditional simulator. The November `nmore_run_1000` configurations set the E06 ensemble. E07 uses `d80cacf874`, `run_switchtime_trajectories_fixed_alpha.py` and its shifted-trajectory notebook. |
| E03–E05 | `d80cacf874`, `fixed_point_solvers.py`, the `figure_1a` and `figure_1b` SFT generators, and the `figure_1c` warm-start prescription. The tail analysis comes from `nsft_tail_analysis.py` and the `583dcb4252` extended-tail study. The selected November plot and local Fig. 1a rendering edits identify the displayed curves. |
| E08 | `29eb610331`, `cost_sweep_worker.py` and the `figure_4a_cpa1000` configuration for Fig. 5a. `6eba94724b`, `budget_pipeline.py` and `budget_worker.py`, plus the verified `T6_eps0.2_cpa2000` optimizer settings for Fig. 2d. |
| E09–E11 | `d80cacf874`, the July `eps_sweep_2026-07-17` settings and `fixed_alpha.py`; `29eb610331`, the allocation search/audit code; `b2cd1e1a9a`, `performance_modes.py`, `simulation_worker.py` and the `figure_4d_alpha_min0p9` preset. |
| E14 | `4ed190c0ee`, spherical residual simulation and `analyze_spherical_residual_validation.py`. The untracked final `plot_projector_closure.py` (SHA-256 prefix `13245d292df8`) identifies the selected contractions, dimensions, checkpoints, and uncertainty statistic. |
| Six-state extension | `118ea780df`, `sft_then_rl_degree2.py`, `degree2_geometry.py` and the stable branch's trace provider. |

## Numerical checks

Comparisons used the original research outputs or workers as migration references. No reference data are needed to run the release.

| Check | Result |
|---|---|
| Five analytic trace moments at α=1 versus E01's saved theory | Maximum absolute difference `2.0e-14` |
| All four microscopic order parameters after one epoch of E06, D=800, T=n=10, first realization, versus the October worker | Maximum absolute difference `1.9e-17` on the same machine |
| E05 full SFT solve at α=1.6, ε=0 and 0.4, versus saved errors | Maximum absolute difference `1.7e-10` |
| E11 SFT allocation at budget 18,001 | Both select α=0.9 and one SFT epoch; accuracy differs by `4.3e-14` |
| E14 dimension fits on the original α=0.5 disorder means | Maximum absolute slope difference `9.1e-15` across 20 checkpoints, both contractions, and both coefficient sources |
| Six-state trajectory, D=800, α=0.5, 100 SFT epochs and two RL epochs, versus the stable branch on matched geometry | Maximum absolute terminal-state difference `2.3e-16` |

These checks exercise selected full-setting calculations. The command-level release check uses the reduced presets across all experiments. The complete Fig. 4a preset also finished with finite trajectories for all four warm starts, at four million microscopic updates per trajectory.
