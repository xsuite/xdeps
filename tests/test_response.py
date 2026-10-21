import numpy as np
import pytest

import xdeps as xd


def make_optimizer(function, x0=(0.0, 0.0), target=(1.0, 2.0)):
    return xd.Optimize.from_callable(
        function,
        x0=list(x0),
        tar=list(target),
        steps=[1e-5] * len(x0),
        tols=[1e-10] * len(target),
        show_call_counter=False,
    )


def test_supplied_response_scaling_masks_and_svd_reuse():
    matrix = np.array([[2.0, 1.0], [1.0, 3.0], [1.0, -1.0]])
    opt = make_optimizer(lambda x: matrix @ x, target=(1.0, 2.0, 0.0))
    for vv, scale in zip(opt.vary, (0.1, 4.0)):
        vv.weight = scale
    for tt, weight in zip(opt.targets, (3.0, 0.2, 5.0)):
        tt.weight = weight
    opt.set_response(matrix)
    weights = np.array([3.0, 0.2, 5.0])
    scales = np.array([0.1, 4.0])
    expected = (
        scales
        * np.linalg.lstsq(
            weights[:, None] * matrix * scales,
            weights * np.array([1.0, 2.0, 0.0]),
            rcond=1e-12,
        )[0]
    )
    np.testing.assert_allclose(opt.get_correction(rcond=1e-12), expected)
    cached = opt.solver._last_jac_svd
    assert opt.response.svd is cached
    np.testing.assert_allclose(opt.get_correction([-1.0, -2.0, 0.0]), expected)
    assert opt.solver._last_jac_svd is cached
    np.testing.assert_array_equal(opt._extract_knob_values(), [0.0, 0.0])
    opt.step(rcond=1e-12)
    np.testing.assert_allclose(opt._extract_knob_values(), expected)
    assert opt.solver._last_jac_svd is cached

    opt.targets[2].active = False
    opt.vary[1].active = False
    step = opt.get_correction([-1.0, -2.0, 0.0])
    assert step[1] == 0
    assert opt.solver._last_jac_svd is not cached
    cached = opt.solver._last_jac_svd
    opt.targets[0].weight = 7.0
    opt.get_correction([-1.0, -2.0, 0.0])
    assert opt.solver._last_jac_svd is not cached


def test_finite_difference_log_response_restores_and_includes_inactive():
    opt = make_optimizer(
        lambda x: [np.exp(x[0]), x[0] + 2 * x[1]], x0=(0.3, 0.4), target=(1.0, 0.0)
    )
    opt.targets[0].optimize_log = True
    opt.targets[0].weight = 7.0
    opt.vary[0].weight = 0.02
    opt.vary[1].active = False
    opt.targets[1].active = False
    response = opt.build_response()
    np.testing.assert_allclose(
        response.matrix, [[1 / np.log(10), 0], [1, 2]], atol=1e-10
    )
    np.testing.assert_array_equal(opt._extract_knob_values(), [0.3, 0.4])
    assert not opt.vary[1].active
    assert not opt.targets[1].active
    assert opt.response is None
    opt.set_response(response)
    opt.solve()
    np.testing.assert_allclose(opt._extract_knob_values(), [0.0, 0.4], atol=1e-10)
    opt.set_response(None)
    assert opt.response is None


def test_failed_probe_restores_knobs_and_activation():
    def function(x):
        if x[0] > 0:
            raise RuntimeError("unstable model")
        return x

    opt = make_optimizer(function)
    opt.vary[0].active = False
    with pytest.raises(RuntimeError, match="unstable"):
        opt.build_response()
    np.testing.assert_array_equal(opt._extract_knob_values(), [0.0, 0.0])
    assert not opt.vary[0].active
    opt.vary[0].active = True
    with pytest.raises(RuntimeError, match="unstable"):
        opt.get_correction()
    np.testing.assert_array_equal(opt._extract_knob_values(), [0.0, 0.0])


def test_response_validation_builder_and_regularization():
    matrix = np.diag([1.0, 1e-8])
    opt = make_optimizer(lambda x: matrix @ x)
    with pytest.raises(ValueError, match="shape"):
        opt.set_response(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        opt.set_response([[np.nan, 0], [0, 1]])
    response = opt.build_response(lambda opt, factor: factor * matrix, factor=1.0)
    opt.set_response(response)
    np.testing.assert_allclose(opt.get_correction(rcond=1e-6), [1.0, 0.0])
    np.testing.assert_allclose(opt.get_correction(sing_val_cutoff=1), [1.0, 0.0])
    other = make_optimizer(lambda x: x)
    with pytest.raises(ValueError, match="differ"):
        other.set_response(response)


def test_limits_and_default_jacobian_after_reset():
    opt = make_optimizer(lambda x: x)
    opt.vary[0].limits = (-2.0, 0.5)
    opt.set_response(np.eye(2))
    opt.step(take_best=False)
    assert opt.vary[0].get_value() <= 0.5
    opt.set_response(None)
    opt.vary[0].limits = (-2.0, 2.0)
    opt.solve()
    np.testing.assert_allclose(opt._extract_knob_values(), [1.0, 2.0], atol=1e-8)


def test_failed_action_is_not_a_zero_response():
    knobs = {"a": 0.0}

    class Action(xd.Action):
        def run(self):
            return "failed" if knobs["a"] != 0.0 else [0.0]

    action = Action()
    opt = xd.Optimize(
        vary=[xd.Vary("a", container=knobs, step=1e-5)],
        targets=[action.target(0, value=1.0, tol=1e-10)],
        show_call_counter=False,
    )
    with pytest.raises(ValueError, match="probe failed"):
        opt.build_response()
    assert knobs["a"] == 0.0


def test_supplied_residuals_skip_evaluation_and_scale_cache_updates():
    calls = []

    def function(x):
        calls.append(1)
        return [x[0] + x[1]]

    opt = make_optimizer(function, target=(1.0,))
    opt.set_response([[1.0, 1.0]])
    before = len(calls)
    first_svd = opt.response.svd
    np.testing.assert_allclose(opt.get_correction([-1.0]), [0.5, 0.5])
    opt.vary[1].weight = 2.0
    np.testing.assert_allclose(opt.get_correction([-1.0]), [0.2, 0.8])
    assert opt.response.svd is not first_svd
    assert len(calls) == before


def test_jacobian_result_and_live_snapshot_allow_explicit_rollback():
    opt = make_optimizer(lambda x: x, x0=(0.0,), target=(1.0,))
    opt.solver.n_bisections = 0
    opt.set_response([[-1.0]])  # Deliberately wrong direction.
    before = opt.get_knob_values()
    info = opt.run_jacobian(take_best=False)
    assert info.degraded and not info.improved
    assert info.penalty_before == 1.0
    assert info.penalty_after > 1.0
    assert info.n_steps == 1 and info.n_calls > 0
    assert not info.within_tol
    assert info.step_scale is not None
    for key, value in before.items():
        opt.vary[0].container[key] = value
    assert opt.get_knob_values() == before
    assert opt.get_knob_values(iteration=-1) != before  # History is preserved.
    opt.set_response([[1.0]])
    info = opt.run_jacobian(take_best=False)
    assert info.improved and not info.degraded
    assert info.within_tol
    assert info.penalty_after == pytest.approx(0.0, abs=1e-12)
    info = opt.run_jacobian()
    assert info.within_tol and not info.improved
    assert info.step_scale is None


def test_jacobian_result_describes_retained_best_and_zero_steps():
    opt = make_optimizer(lambda x: x, x0=(0.0,), target=(1.0,))
    opt.solver.n_bisections = 0
    opt.set_response([[-1.0]])
    info = opt.run_jacobian(take_best=True)
    assert info.penalty_after == info.penalty_before
    assert not info.improved and not info.degraded
    assert opt.get_knob_values() == {0: 0.0}
    info = opt.run_jacobian(n_steps=0)
    assert info.n_steps == 0 and info.step_scale is None
