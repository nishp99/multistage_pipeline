"""Geometry, independent PCA batches, and replicated microscopic trajectories."""

import numpy as np
from jax import random
from multistage_pipeline.dynamics.pipeline import trajectory
from multistage_pipeline.dynamics.simulation import simulate
from multistage_pipeline.pretraining.projector import (
    pca_basis,
    population_geometry,
    population_moments,
    vectors,
    block_moments,
)


def theory_geometry(alpha, cfg, settings, teacher=None, student=None):
    f = settings["signal_fraction"]
    blocks = (
        None
        if teacher is None
        else block_moments(teacher, student, round(f * cfg.dimension))
    )
    moments = population_moments(alpha, f, cfg.delta, settings["quick"])
    return population_geometry(moments, blocks)


def ensemble(
    alpha, cfg, settings, sft, rl, *, balanced=False, seed_offset=0, every_epoch=False
):
    f, base = settings["signal_fraction"], settings["seed"]
    teacher, student = vectors(cfg.dimension, f, balanced=balanced)
    steps = np.arange(int(sft + rl) + 1) * cfg.dimension if every_epoch else None
    analytic = trajectory(
        cfg, theory_geometry(alpha, cfg, settings, teacher, student), sft, rl, steps
    )
    simulations, conditionals = [], []
    for batch in range(settings["batches"]):
        batch_id = 100 * seed_offset + batch
        key = random.fold_in(random.PRNGKey(base), batch_id)
        basis = pca_basis(key, alpha, cfg.dimension, f, cfg.delta)
        replicas = [
            simulate(
                cfg,
                basis,
                teacher,
                student,
                sft,
                rl,
                base + 1 + 1000 * batch_id + r,
                steps,
            )
            for r in range(settings["replicas"])
        ]
        simulations.append([run.states for run in replicas])
        conditionals.append(replicas[0].conditional)
    return dict(
        steps=analytic.steps,
        switch_step=analytic.switch_step,
        theory=analytic.states,
        simulation=np.asarray(simulations),
        conditional=np.asarray(conditionals),
    )


def batch_summary(values):
    means = np.mean(values, axis=1)
    return means.mean(axis=0), means.std(axis=0, ddof=1)
