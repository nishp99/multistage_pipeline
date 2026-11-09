import numpy as np
import jax.numpy as jnp
from jax import random
import pytest

from multistage_pipeline.pretraining.projector import (
    vectors,
    pca_basis,
    measured_geometry,
    spectral_moments,
    measured_spectral_state,
)
from multistage_pipeline.dynamics.compact import Prob_correct_rl, _closed_second_order
from multistage_pipeline.dynamics.pipeline import Dynamics, step, step_args, trajectory
from multistage_pipeline.dynamics.simulation import (
    simulate,
    prepare_fields,
    spectral_state,
)
from multistage_pipeline.dynamics import six_state


@pytest.fixture(scope="module")
def problem():
    teacher, student = vectors(40)
    basis = pca_basis(random.PRNGKey(37), 1.0, 40)
    return basis, teacher, student, measured_geometry(basis, teacher, student)


def test_gate_and_closure_identities(problem):
    *_, g = problem
    j1, j2, q1, q2 = g.initial
    p, reward, a, b = Prob_correct_rl(*jnp.asarray(g.initial), 1.0, 0.1, 6, 6)
    np.testing.assert_allclose([reward, a, b], [p**6, p**5, 0.0], atol=1e-14)
    js, jm, jn, qs, qm, qn = _closed_second_order(*g.omega, *g.initial)
    np.testing.assert_allclose(
        [js + jm, jm + jn, qs + qm, qm + qn], g.initial, atol=1e-15
    )


@pytest.mark.parametrize("sft,rl", [(0, 0), (0, 2), (2, 0), (1, 2), (0.25, 0.5)])
def test_phase_boundaries(problem, sft, rl):
    *_, g = problem
    cfg = Dynamics(40)
    result = trajectory(cfg, g, sft, rl)
    assert result.switch_step == round(40 * sft)
    assert result.steps[0] == 0 and result.steps[-1] == round(40 * sft) + round(40 * rl)
    assert result.switch_step in result.steps
    np.testing.assert_array_equal(result.states[0], g.initial)
    np.testing.assert_allclose(
        result.states[:, 2:].sum(1), g.initial[2:].sum(), atol=1e-13
    )
    if sft:
        at_switch = result.states[result.steps == result.switch_step][0]
        np.testing.assert_allclose(
            at_switch, trajectory(cfg, g, sft, 0).states[-1], atol=1e-14
        )


def test_microstep_replays_projector_schedule(problem):
    basis, t, w, g = problem
    cfg = Dynamics(40, epsilon=0.05, optimal=True, spherical=True)
    fields, weights = prepare_fields(basis, t, w, cfg.delta)
    state = jnp.asarray(g.initial)
    key = random.PRNGKey(18)
    for i, phase in enumerate(["sft", "rl"]):
        state, eta = step(state, g, step_args(cfg), phase)
        episode_key = random.fold_in(key, i)
        z_key, noise_key = random.split(episode_key)
        x = np.asarray(random.normal(z_key, (6, len(weights)))) * fields.input_sd
        teacher = np.sign(
            x @ fields.teacher_regression
            + np.asarray(random.normal(noise_key, (6,))) * fields.teacher_residual_sd
        )
        pred = np.sign(x @ weights)
        if phase == "sft":
            flips = np.asarray(
                random.bernoulli(random.fold_in(episode_key, 1), cfg.epsilon, (6,))
            )
            teacher *= np.where(flips, -1, 1)
            gain = (teacher - pred) / 2
        else:
            gain = pred * np.all(pred == teacher)
        weights += float(eta) * (gain @ x) / (6 * np.sqrt(40))
        weights *= np.sqrt(40 * g.initial[2:].sum()) / np.linalg.norm(weights)
    sim = simulate(cfg, basis, t, w, 1 / 40, 1 / 40, 18)
    np.testing.assert_allclose(sim.conditional[-1], state, atol=1e-14)
    np.testing.assert_allclose(
        sim.spectral[-1], spectral_state(weights, fields, 40), atol=1e-14
    )
    np.testing.assert_allclose(
        sim.states[:, 2:].sum(1), g.initial[2:].sum(), atol=1e-13
    )


def test_six_state_measures_same_initialization(problem):
    basis, t, w, g = problem
    initial, teacher = measured_spectral_state(basis, t, w)
    cfg = Dynamics(40, optimal=False)
    six = six_state.trajectory(cfg, spectral_moments(basis), teacher, initial, 1, 1)
    np.testing.assert_allclose(six.states[:, 3], initial[3], atol=1e-13)
    np.testing.assert_allclose(initial[[1, 4]], g.initial[[0, 2]], atol=1e-14)
    assert np.isfinite(six.states).all() and six.steps[-1] == 80
    with pytest.raises(ValueError, match="all-correct"):
        six_state.trajectory(
            Dynamics(40, threshold=3), spectral_moments(basis), teacher, initial, 1, 1
        )
