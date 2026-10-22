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


def test_residual_activation_mask_scales_linearly_and_refreshes(monkeypatch):
    opt = make_optimizer(lambda x: np.full(100, 10.0), target=np.ones(100))
    for target in opt.targets:
        target.optimize_log = True
    merit = opt._err
    original_mask = type(merit).mask_output.fget
    calls = []

    def counted_mask(self):
        calls.append(1)
        return original_mask(self)

    monkeypatch.setattr(type(merit), 'mask_output', property(counted_mask))
    for active in (False, True):
        opt.targets[0].active = active
        calls.clear()
        residuals = merit(merit._get_x())
        np.testing.assert_array_equal(residuals, [float(active)] + [1.0] * 99)
        # Each mask scans all targets, so its call count must stay bounded.
        assert len(calls) <= 3


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


def test_response_reuses_recurring_selections_and_invalidates_scaling(monkeypatch):
    matrix = np.diag([2., 3., 4., 5.])
    opt = make_optimizer(lambda x: matrix @ x, x0=(0.,) * 4,
                         target=(1., 2., 3., 4.))
    opt.set_response(matrix)
    decompositions = []
    original_svd = np.linalg.svd

    def counted_svd(*args, **kwargs):
        decompositions.append(1)
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, 'svd', counted_svd)
    cached = []
    for selected in ([0, 1], [2, 3], [0, 1], [2, 3]):
        opt.disable(target=True, vary=True)
        opt.enable(target=selected, vary=selected)
        step = opt.get_correction([-1., -2., -3., -4.])
        expected = np.zeros(4)
        expected[selected] = np.arange(1., 5.)[selected] / matrix.diagonal()[selected]
        np.testing.assert_allclose(step, expected)
        cached.append(opt.response.svd)
    assert len(decompositions) == 2
    assert cached[0] is cached[2] and cached[1] is cached[3]
    opt.targets[2].weight = 2.
    opt.get_correction([-1., -2., -3., -4.])
    assert len(decompositions) == 3
    assert opt.response.svd is not cached[-1]
    opt.vary[2].weight = 0.5
    opt.get_correction([-1., -2., -3., -4.])
    assert len(decompositions) == 4


def test_response_selection_cache_is_bounded():
    opt = make_optimizer(lambda x: x, x0=(0.,) * 10, target=(1.,) * 10)
    opt.set_response(np.eye(10))
    first = None
    for index in range(10):
        opt.disable(target=True, vary=True).enable(target=index, vary=index)
        if index == 0:
            first = opt.response.svd
        else:
            opt.response.svd
    assert len(opt.response._svd_cache) == 8
    opt.disable(target=True, vary=True).enable(target=0, vary=0)
    assert opt.response.svd is not first


def test_single_shared_action_and_selected_blocks():
    knobs = {'a': 0., 'b': 0.}
    calls = []

    class Measurement(xd.Action):
        def run(self):
            calls.append(1)
            return [2 * knobs['a'], 3 * knobs['b']]

    action = Measurement()
    opt = xd.Optimize(
        vary=[xd.Vary(n, container=knobs, step=1e-6, tag=n) for n in knobs],
        targets=[action.target(i, value=1., tol=1e-10, tag=n)
                 for i, n in enumerate(knobs)],
        show_call_counter=False)
    opt.set_response(np.diag([2., 3.]))
    calls.clear()
    np.testing.assert_allclose(opt.get_correction(), [0.5, 1 / 3])
    assert len(calls) == 1  # All targets share the exact same action object.
    opt.disable(target=True, vary=True).enable(target='b', vary='b')
    np.testing.assert_allclose(opt.get_correction(), [0., 1 / 3])
    assert len(calls) == 2
    np.testing.assert_allclose(opt.get_correction([-1., -1.]), [0., 1 / 3])
    assert len(calls) == 2  # Existing measurements need no model evaluation.
    info = opt.run_jacobian(take_best=False)
    assert info.improved and knobs['a'] == 0.
    assert knobs['b'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('use_blocks', [False, True])
def test_block_response_permuted_rows_columns_masks_and_weights(use_blocks):
    # Two independent rectangular blocks, interleaved in global target order.
    matrix = np.array([[0., 2.], [3., 0.], [0., -1.], [4., 0.]])
    opt = make_optimizer(lambda x: matrix @ x, target=(1., 2., 3., 4.))
    for target, weight in zip(opt.targets, [2., 4., 1., 3.]):
        target.weight = weight
    for vary, scale in zip(opt.vary, [0.1, 2.]):
        vary.weight = scale
    if use_blocks:
        opt.set_response(blocks={
            'b': dict(matrix=[[2.], [-1.]], targets=[0, 2], vary=[1]),
            'a': dict(matrix=[[4.], [3.]], targets=[3, 1], vary=[0]),
        })
        np.testing.assert_array_equal(opt.response.matrix, matrix)
        with pytest.raises(ValueError, match='Select one'):
            opt.response.svd
    else:
        opt.set_response(matrix)
    for rows, columns in [([0, 1, 2, 3], [0, 1]), ([0, 1, 3], [1]),
                          ([1, 2, 3], [0, 1]), ([0, 2], [0, 1])]:
        opt.disable(target=True, vary=True).enable(target=rows, vary=columns)
        weights = np.array([t.weight for t in opt.targets])
        scales = np.array([v.weight for v in opt.vary])
        residuals = -np.arange(1., 5.)
        expected = np.zeros(2)
        expected[columns] = -scales[columns] * np.linalg.lstsq(
            (weights[:, None] * matrix * scales)[np.ix_(rows, columns)],
            (weights * residuals)[rows], rcond=1e-12)[0]
        np.testing.assert_allclose(opt.get_correction(residuals, rcond=1e-12), expected,
                                   atol=1e-14)


def test_block_response_cutoffs_and_steps_do_not_materialize_dense(monkeypatch):
    from xdeps.optimize.response import BlockResponseMatrix

    matrix = np.diag([1., 1e-8, 1e-5, 1e-10])
    opt = make_optimizer(lambda x: matrix @ x, x0=(0.,) * 4, target=(1.,) * 4)
    opt.set_response(blocks={
        'first': dict(matrix=matrix[:2, :2], targets=slice(0, 2), vary=slice(0, 2),
                      rcond=1e-6),
        'second': dict(matrix=matrix[2:, 2:], targets=slice(2, 4), vary=slice(2, 4),
                       rcond=1e-6),
    })

    def no_dense(self):
        raise AssertionError('Ordinary block solves must not assemble a dense matrix')

    monkeypatch.setattr(BlockResponseMatrix, 'matrix', property(no_dense))
    np.testing.assert_allclose(opt.get_correction(), [1., 0., 1e5, 1e10])
    # A scalar override applies relative to each block's own largest singular value.
    np.testing.assert_allclose(opt.get_correction(rcond=1e-4), [1., 0., 1e5, 0.])
    np.testing.assert_allclose(opt.get_correction(sing_val_cutoff=1), [1., 0., 1e5, 0.])
    opt.run_jacobian(take_best=False)
    np.testing.assert_allclose(opt._extract_knob_values(), [1., 0., 1e5, 1e10])
    first_svd = opt.response.blocks['first'].svd
    second_svd = opt.response.blocks['second'].svd
    for selected in ([0, 1], [2, 3], [0, 1], [2, 3]):
        opt.disable(target=True, vary=True).enable(target=selected, vary=selected)
        expected_svd = first_svd if selected[0] == 0 else second_svd
        assert opt.response.svd is expected_svd
    opt.targets[2].weight = 2.
    assert opt.response.svd is not second_svd
    opt.disable(target=True, vary=True).enable(target=[0, 1], vary=[0, 1])
    assert opt.response.svd is first_svd  # Unchanged block retains its factorization.


def test_block_response_limits_reset_and_validation():
    opt = make_optimizer(lambda x: x)
    blocks = {
        'a': dict(matrix=[[1.]], targets=[0], vary=[0]),
        'b': dict(matrix=[[1.]], targets=[1], vary=[1]),
    }
    opt.set_response(blocks=blocks)
    before = opt.get_knob_values()
    with pytest.raises(ValueError, match='Broyden updates require a dense response'):
        opt.step(broyden=True)
    assert opt.get_knob_values() == before
    opt.vary[0].limits = (-2., 0.5)
    opt.step(take_best=False)
    assert opt.vary[0].get_value() <= 0.5
    opt.set_response(None)
    opt.vary[0].limits = (-2., 2.)
    opt.solve()
    np.testing.assert_allclose(opt._extract_knob_values(), [1., 2.], atol=1e-8)
    with pytest.raises(ValueError, match='either'):
        opt.set_response(np.eye(2), blocks=blocks)
    with pytest.raises(ValueError, match='cover every'):
        opt.set_response(blocks={'a': blocks['a']})
    with pytest.raises(ValueError, match='disjoint'):
        opt.set_response(blocks={'a': blocks['a'], 'b': blocks['a']})
    for indices in ([0, 0], [-1], [2], [], [0.5], [[0]]):
        with pytest.raises(ValueError, match='indices'):
            opt.set_response(blocks={'a': dict(matrix=[[1.]], targets=indices, vary=[0])})
    with pytest.raises(ValueError, match='shape'):
        opt.set_response(blocks={'a': dict(matrix=np.eye(2), targets=[0], vary=[0])})
    with pytest.raises(ValueError, match='finite'):
        opt.set_response(blocks={'a': dict(matrix=[[np.nan]], targets=[0], vary=[0])})
    with pytest.raises(ValueError, match='Unknown'):
        opt.set_response(blocks={'a': dict(blocks['a'], typo=0)})
    opt.set_response(blocks=blocks)
    other = make_optimizer(lambda x: x)
    with pytest.raises(ValueError, match='differ'):
        other.set_response(opt.response)


def test_supplied_residuals_do_not_read_machine_knobs(monkeypatch):
    opt = make_optimizer(lambda x: x)
    opt.set_response(blocks={
        'a': dict(matrix=[[1.]], targets=[0], vary=[0]),
        'b': dict(matrix=[[1.]], targets=[1], vary=[1]),
    })

    def no_read():
        raise AssertionError('A fixed linear solve needs only the supplied residuals')

    monkeypatch.setattr(opt._err, '_get_x', no_read)
    np.testing.assert_array_equal(opt.get_correction([-1., -2.]), [1., 2.])


def test_block_solve_diagnostics_include_structural_zero_modes():
    # Each local block is full rank, but the combined matrix has a zero mode.
    opt = make_optimizer(lambda x: x, x0=(0.,) * 3, target=(1.,) * 3)
    opt.set_response(blocks={
        'wide': dict(matrix=[[1., 2.]], targets=[0], vary=[0, 1]),
        'tall': dict(matrix=[[3.], [4.]], targets=[1, 2], vary=[2]),
    })
    matrix = opt.response.matrix
    opt.get_correction([-1.] * 3)
    solver = opt.solver._last_jac_svd
    assert solver.rank == np.linalg.matrix_rank(matrix) == 2
    assert np.isinf(solver.cond)
    # Explicit Jacobian access remains available to other solver frontends.
    np.testing.assert_array_equal(opt._err.get_jacobian(opt._err._get_x()), matrix)


def test_dense_fixed_response_preserves_periodic_broyden_history():
    opt = make_optimizer(lambda x: np.asarray(x)**2, x0=(1.,), target=(2.,))
    opt.set_response([[2.]])
    opt.step(n_steps=2, broyden=2, take_best=False)
    # Newton's first trial is 1.5. The secant response is then 2.5, giving 1.4.
    np.testing.assert_allclose(opt._extract_knob_values(), [1.4], atol=1e-14)


def test_correction_evaluation_preserves_scaled_knobs_exactly():
    opt = make_optimizer(lambda x: x, x0=(0.7,), target=(2.,))
    opt.vary[0].weight = 0.3
    opt.set_response([[1.]])
    before = opt.get_knob_values()
    np.testing.assert_allclose(opt.get_correction(), [1.3])
    assert opt.get_knob_values() == before
