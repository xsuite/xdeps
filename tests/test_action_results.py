"""Supplied results bypass starting-point calls; outcome tagging is optional."""

import numpy as np
import pytest
import xdeps as xd


class ExternalAction(xd.Action):
    def prepare(self):
        raise AssertionError('Unexpected preparation')

    def run(self):
        raise AssertionError('Unexpected model evaluation')


def make_external(matrix=np.eye(2), result_format=dict):
    action = ExternalAction()
    knobs = dict(a=0., b=0.)
    opt = xd.Optimize(
        vary=[xd.Vary(k, knobs, step=1e-5) for k in knobs],
        targets=[action.target(i, value=v, tol=1e-9) for i, v in enumerate([1., 2.])],
        action_results=(
            {action: np.zeros(2)} if result_format is dict
            else result_format([np.zeros(2)])), show_call_counter=False,
    )
    opt.set_response(matrix)
    return opt, action, knobs


@pytest.mark.parametrize('result_format', [dict, list, tuple])
def test_external_status_tag_and_reload(result_format):
    opt, action, knobs = make_external(result_format=result_format)
    supplied = {action: np.array([0.5, 2.])}
    if result_format is not dict:
        supplied = result_format(supplied.values())
    status = opt.target_status(ret=True, action_results=supplied)
    np.testing.assert_array_equal(status.current_val, [0.5, 2.])
    np.testing.assert_array_equal(status.residue, [-0.5, 0.])
    np.testing.assert_array_equal(status.tol_met, [False, True])
    assert status.penalty == 0.5
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


@pytest.mark.parametrize('supply_results', [False, True])
def test_status_penalty_uses_optimizer_objective(supply_results):
    action = ExternalAction()
    knobs = dict(k=0.)
    transformed = action.target(1, value=4., tol=1e-9, weight=3.)
    transformed.transform = lambda x: x**2
    opt = xd.Optimize(
        vary=[xd.Vary('k', knobs)],
        targets=[action.target(0, value=1., tol=1e-9, weight=2., optimize_log=True),
                 transformed,
                 action.target(2, value=0., tol=1e-9, weight=5.)],
        action_results={action: [1., 2., 0.]}, show_call_counter=False)
    opt.targets[2].active = False
    calls = []
    action.run = lambda: calls.append(1) or [10., 3., 100.]
    supplied = {action: [10., 3., 100.]} if supply_results else None
    log_length = len(opt.log())
    status = opt.target_status(ret=True, action_results=supplied)
    assert len(calls) == (0 if supply_results else 1)
    assert len(opt.log()) == log_length
    assert knobs == dict(k=0.)
    assert isinstance(status.penalty, float)
    assert 'penalty' not in status._col_names
    assert status.penalty == pytest.approx(np.hypot(2., 15.))
    np.testing.assert_array_equal(status.residue, [9., 5., 100.])
    # Scalar metadata describes the original evaluation, even on filtered views.
    assert status.rows[:1].penalty == status.penalty
    opt.tag(action_results=supplied)
    assert opt.log().penalty[-1] == pytest.approx(status.penalty)
    # Status also works when the merit function normally returns a scalar.
    opt._err.return_scalar = True
    assert opt.target_status(ret=True, action_results=supplied).penalty == status.penalty


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


@pytest.mark.parametrize('result_format', [dict, list, tuple])
def test_lstsq_weights_scales_masks_and_retry(result_format):
    matrix = np.array([[2., 1.], [1., 3.]])
    opt, action, knobs = make_external(matrix)
    opt.targets[0].weight = 3.
    opt.vary[0].weight = 0.2
    opt.vary[1].weight = 4.
    supplied = {action: [0., 0.]}
    if result_format is not dict:
        supplied = result_format(supplied.values())
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
@pytest.mark.parametrize('result_format', [dict, list, tuple])
def test_automatic_methods_use_supplied_results_only_at_start(method, result_format):
    opt, action, knobs = make_external()
    calls = []

    def run():
        values = np.array(list(knobs.values()))
        calls.append(values.copy())
        return values

    action.run = run
    supplied = {action: [0., 0.]}
    if result_format is not dict:
        supplied = result_format(supplied.values())
    getattr(opt, method)(action_results=supplied)
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


@pytest.mark.parametrize('result_format', [list, tuple])
def test_ordered_results_follow_unique_actions_not_targets(result_format):
    a, b = ExternalAction(), ExternalAction()
    knobs = dict(k=0.)
    supplied = result_format([{'v': 0.}, {'u': 0., 'w': 0.}])
    opt = xd.Optimize(
        vary=[xd.Vary('k', knobs)],
        targets=[b.target('v', value=2., tol=1e-9),
                 a.target('u', value=1., tol=1e-9),
                 b.target('v', value=2., tol=1e-9),
                 a.target('w', value=3., tol=1e-9)],
        action_results=supplied, show_call_counter=False)
    assert opt.actions == [b, a]
    opt.set_response([[2.], [1.], [2.], [3.]])
    status = opt.target_status(ret=True, action_results=supplied)
    np.testing.assert_array_equal(status.residue, [-2., -1., -2., -3.])
    for invalid in ([], [supplied[0]], list(supplied) + [supplied[0]]):
        for call in (opt.tag, opt.target_status, opt.run_lstsq):
            with pytest.raises(ValueError, match='requires 2 results in action order'):
                call(action_results=result_format(invalid))
    assert knobs['k'] == 0.
    assert len(opt.log()) == 1
    opt.run_lstsq(action_results=supplied, tag_outcome=False)
    assert knobs['k'] == pytest.approx(1.)
    assert opt._err.call_counter == 0
