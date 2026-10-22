"""Supplied results bypass starting-point calls; outcome tagging is optional."""

import numpy as np
import pytest
import xdeps as xd


class ExternalAction(xd.Action):
    def prepare(self):
        raise AssertionError('Unexpected preparation')

    def run(self):
        raise AssertionError('Unexpected model evaluation')


def make_external(matrix=np.eye(2)):
    action = ExternalAction()
    knobs = dict(a=0., b=0.)
    opt = xd.Optimize(
        vary=[xd.Vary(k, knobs, step=1e-5) for k in knobs],
        targets=[action.target(i, value=v, tol=1e-9) for i, v in enumerate([1., 2.])],
        action_results={action: np.zeros(2)}, show_call_counter=False,
    )
    opt.set_response(matrix)
    return opt, action, knobs


def test_external_status_tag_and_reload():
    opt, action, knobs = make_external()
    supplied = {action: np.array([0.5, 2.])}
    status = opt.target_status(ret=True, action_results=supplied)
    np.testing.assert_array_equal(status.current_val, [0.5, 2.])
    np.testing.assert_array_equal(status.residue, [-0.5, 0.])
    np.testing.assert_array_equal(status.tol_met, [False, True])
    assert len(opt.target_mismatch(ret=True, action_results=supplied)) == 1
    opt.tag('baseline', action_results=supplied)
    assert opt.log().penalty[-1] == 0.5
    assert knobs == dict(a=0., b=0.)
    assert opt._err.call_counter == 0
    knobs['a'] = 10.
    opt.reload(tag='baseline', action_results=supplied)
    assert knobs == dict(a=0., b=0.)
    opt.clear_log(action_results=supplied)
    assert len(opt.log()) == 1
    with pytest.raises(AssertionError, match='model evaluation'):
        opt.target_status(ret=True)  # Supplied results are not retained implicitly.


def test_run_actions_returns_outputs_without_target_evaluation():
    opt, action, knobs = make_external()
    calls = []
    result = object()  # Cannot be interpreted by the configured targets.
    action.run = lambda: calls.append(action) or result
    before = len(opt.log())
    assert opt.run_actions() == {action: result}
    assert calls == [action]  # Two targets share this action.
    assert len(opt.log()) == before
    assert knobs == dict(a=0., b=0.)


def test_missing_action_results_and_no_jacobian_are_errors():
    opt, action, knobs = make_external()
    before = len(opt.log())
    for call in (opt.tag, opt.target_status, opt.run_lstsq):
        with pytest.raises(ValueError, match='Missing action_results'):
            call(action_results={})
    assert len(opt.log()) == before
    assert knobs == dict(a=0., b=0.)
    opt.set_response(None)
    with pytest.raises(ValueError, match='Jacobian or fixed response'):
        opt.run_lstsq(tag_outcome=False, action_results={action: [0., 0.]})


def test_lstsq_weights_scales_masks_and_retry():
    matrix = np.array([[2., 1.], [1., 3.]])
    opt, action, knobs = make_external(matrix)
    opt.targets[0].weight = 3.
    opt.vary[0].weight = 0.2
    opt.vary[1].weight = 4.
    supplied = {action: [0., 0.]}
    weights, scales = np.array([3., 1.]), np.array([0.2, 4.])
    expected = scales * np.linalg.lstsq(
        weights[:, None] * matrix * scales,
        weights * np.array([1., 2.]), rcond=1e-12)[0]
    assert opt.run_lstsq(tag_outcome=False, action_results=supplied, rcond=1e-12) is opt
    np.testing.assert_allclose(list(knobs.values()), expected)
    cached = opt.response.svd
    assert len(opt.log()) == 1  # No unevaluated post-step point is logged.
    knobs.update(a=0., b=0.)
    opt.run_lstsq(tag_outcome=False, action_results=supplied, rcond=1e-12, step_scale=0.5)
    np.testing.assert_allclose(list(knobs.values()), 0.5 * expected)
    assert opt.response.svd is cached
    knobs.update(a=0., b=0.)
    opt.targets[1].active = False
    opt.vary[1].active = False
    opt.run_lstsq(tag_outcome=False, action_results=supplied)
    np.testing.assert_allclose(list(knobs.values()), [0.5, 0.])


@pytest.mark.parametrize('supply_results', [False, True])
def test_lstsq_tags_fresh_outcome_by_default(supply_results):
    opt, action, knobs = make_external()
    calls = []

    def run():
        values = np.array(list(knobs.values()))
        calls.append(values.copy())
        return values

    action.run = run
    before = len(opt.log())
    supplied = {action: [0., 0.]} if supply_results else None
    assert opt.run_lstsq(action_results=supplied, step_scale=0.5) is opt
    np.testing.assert_allclose(calls, [[0.5, 1.]] if supply_results
                               else [[0., 0.], [0.5, 1.]])
    log = opt.log()
    assert len(log) == before + 1
    assert log.tag[-1] == 'lstsq'
    np.testing.assert_allclose(
        list(opt.get_knob_values(iteration=-1).values()), [0.5, 1.])
    np.testing.assert_allclose(log.penalty[-1], np.linalg.norm([0.5, 1.]))
    np.testing.assert_allclose(opt._err.last_res_values, [0.5, 1.])


def test_temporary_jacobian_and_limits():
    opt, action, knobs = make_external()
    response = opt.response
    supplied = {action: [0., 0.]}
    opt.run_lstsq(tag_outcome=False, action_results=supplied, jacobian=2 * np.eye(2))
    np.testing.assert_allclose(list(knobs.values()), [0.5, 1.])
    assert opt.response is response
    knobs.update(a=0., b=0.)
    opt.vary[0].limits = (-1., 0.25)
    opt.run_lstsq(tag_outcome=False, action_results=supplied)
    np.testing.assert_allclose(list(knobs.values()), [0.25, 0.5])
    knobs.update(a=0., b=0.)
    opt.vary[0].max_step = 0.1
    opt.vary[1].max_step = 0.4
    opt.run_lstsq(tag_outcome=False, action_results=supplied, step_scale=0.5)
    np.testing.assert_allclose(list(knobs.values()), [0.05, 0.1])
    for invalid in (-1, np.nan, np.inf, 1.1):
        with pytest.raises(ValueError, match='step_scale'):
            opt.run_lstsq(tag_outcome=False, action_results=supplied, step_scale=invalid)


def test_log_targets_and_failed_data():
    opt, action, knobs = make_external()
    opt.targets[0].optimize_log = True
    opt.run_lstsq(tag_outcome=False, action_results={action: [10., 2.]})
    np.testing.assert_allclose(list(knobs.values()), [-1., 0.])
    baseline = knobs.copy()
    with pytest.raises(ValueError, match='Residual evaluation failed'):
        opt.run_lstsq(tag_outcome=False, action_results={action: 'failed'})
    assert knobs == baseline


def test_default_evaluation_and_failed_numerical_probe():
    opt = xd.Optimize.from_callable(
        lambda x: [2*x[0]], x0=[0.], tar=[1.], steps=[1e-5],
        tols=[1e-9], show_call_counter=False)
    opt.run_lstsq()
    np.testing.assert_allclose(list(opt.get_knob_values().values()), [0.5])
    action = opt.actions[0]
    original = action.run

    def fail_probe():
        if list(opt.get_knob_values().values())[0] != 0.5:
            raise RuntimeError('Probe failed')
        return original()

    action.run = fail_probe
    with pytest.raises(RuntimeError, match='Probe failed'):
        opt.run_lstsq()
    np.testing.assert_allclose(list(opt.get_knob_values().values()), [0.5])


@pytest.mark.parametrize('method', ['step', 'solve', 'run_jacobian'])
def test_automatic_methods_use_supplied_results_only_at_start(method):
    opt, action, knobs = make_external()
    calls = []

    def run():
        values = np.array(list(knobs.values()))
        calls.append(values.copy())
        return values

    action.run = run
    getattr(opt, method)(action_results={action: [0., 0.]})
    assert calls
    assert all(np.any(values != 0) for values in calls)
    np.testing.assert_allclose(list(knobs.values()), [1., 2.])


def test_multiple_actions_require_complete_results():
    a, b = ExternalAction(), ExternalAction()
    knobs = dict(k=0.)
    opt = xd.Optimize(
        vary=[xd.Vary('k', knobs)],
        targets=[a.target('u', value=1., tol=1e-9),
                 b.target('v', value=2., tol=1e-9)],
        action_results={a: {'u': 0.}, b: {'v': 0.}}, show_call_counter=False)
    opt.set_response([[1.], [2.]])
    with pytest.raises(ValueError, match='Missing action_results'):
        opt.run_lstsq(tag_outcome=False, action_results={a: {'u': 0.}})
    assert knobs['k'] == 0.
    calls = []
    a.run = lambda: calls.append('a') or {'u': knobs['k']}
    b.run = lambda: calls.append('b') or {'v': 2 * knobs['k']}
    data = opt.run_actions()
    assert calls == ['a', 'b']
    opt.run_lstsq(tag_outcome=False, action_results=data)
    assert calls == ['a', 'b']
    assert knobs['k'] == pytest.approx(1.)
