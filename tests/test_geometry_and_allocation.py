import numpy as np
import pytest
from multistage_pipeline.pretraining.spectral import (
    symmetric_closure,
    GeometryError,
    prepare_geometry,
)
from multistage_pipeline.pretraining.projector import population_geometry
from multistage_pipeline.dynamics.pipeline import (
    Dynamics,
    trajectory,
    accuracy,
    step_args,
)
from multistage_pipeline.allocation.search import optimize, score
from experiments.closure.residual_scaling import scaling


@pytest.mark.parametrize("nodes", [[0.3], [0.0, 1.0], [0.05, 0.5, 0.9]])
def test_quadrature_and_closure(nodes):
    nodes = np.array(nodes)
    moments = np.array([0.2 * np.mean(nodes**k) for k in range(6)])
    g = symmetric_closure(moments)
    np.testing.assert_allclose(g["nodes"], nodes, atol=1e-12)
    np.testing.assert_allclose(
        np.polynomial.polynomial.polyval(nodes, g["closure"]), nodes**3, atol=1e-12
    )
    np.testing.assert_allclose(
        prepare_geometry(moments, np.zeros((5, 5)))["moments"], moments
    )
    with pytest.raises(GeometryError):
        symmetric_closure([0.2, 0.4, 0.8, 1.6, 3.2, 6.4])


class Table:
    alphas = [0.4, 1.0]

    def __call__(self, a):
        nodes = np.array([0.2, 0.5, 0.85]) * a
        return population_geometry([0.2 * np.mean(nodes**k) for k in range(4)])


@pytest.mark.parametrize("mode", ["mixed", "sft", "rl"])
def test_boundary_and_replay(mode):
    cfg = Dynamics(20)
    table = Table()
    floor = optimize(cfg, table, 8.0, 20.0, mode, grid=3)
    assert floor.alpha == 0.4 and floor.sft == floor.rl == 0.0
    p = optimize(cfg, table, 20.0, 20.0, mode, grid=3)
    assert 0.4 <= p.alpha <= 1.0
    assert abs(20 * p.alpha + p.sft + p.rl - 20) < 1e-9
    if mode == "sft":
        assert p.rl == 0.0
    if mode == "rl":
        assert p.sft == 0.0
    np.testing.assert_allclose(
        p.accuracy, score(table(p.alpha), step_args(cfg), p.sft, p.rl, cfg.dimension)
    )
    g = table(0.5)
    np.testing.assert_allclose(
        score(g, step_args(cfg), 1.0, 2.0, 20),
        accuracy(trajectory(cfg, g, 1, 2).states[-1]),
    )
    with pytest.raises(ValueError, match="minimum"):
        optimize(cfg, table, 5.0, 20.0, grid=3)


def test_fixed_alpha_and_residual_fit():
    p = optimize(
        Dynamics(20), Table(), 5.0, 0.0, fixed_alpha=0.5, grid=3, fixed_spacing=0.1
    )
    assert p.alpha == 0.5 and abs(p.sft + p.rl - 5.0) < 1e-12
    dims = np.array([100, 200, 400])
    # An exact inverse-square-root law with repeated disorders at each D.
    values = np.ones((3, 8, 5, 2, 2)) * dims[:, None, None, None, None] ** -0.5
    slopes, ci = scaling(values, dims, 100, np.random.default_rng(123))
    np.testing.assert_allclose(slopes, -0.5, atol=1e-13)
    np.testing.assert_allclose(ci, -0.5, atol=1e-13)


def test_five_moments_use_converged_higher_order_quadrature():
    from multistage_pipeline.pretraining.traces import five_moments

    moments = five_moments(
        0.5,
        0.2,
        0.1,
        n_grid1d=256,
        n_grid2=32,
        n_grid3=20,
        integration_options=dict(
            powers=(9, 11, 13), scrambles=4, batch_size=1024, atol=0.003, rtol=0.1
        ),
    )
    assert moments.shape == (5,)
    assert np.all(np.diff(moments) < 0) and 0 < moments[-1] < moments[0] < 0.2


def test_reduced_allocation_replay_has_finite_schedules():
    from experiments.allocation.optimized_trajectories import DEFAULTS, SMALL, compute

    result = compute(DEFAULTS | SMALL | {"quick": True})
    for i in range(len(SMALL["budgets"])):
        assert np.isfinite(result[f"conditional_{i}"]).all()
        assert np.isfinite(result[f"simulation_{i}"]).all()
        assert result[f"switch_step_{i}"] in result[f"steps_{i}"]
