# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2023.                 #
# ######################################### #
import numpy as np
import pickle
import pytest

import xdeps as xd
import xdeps.tasks


def test_manager_refs():
    manager = xd.Manager()
    container1 = {'b': 3}
    container2 = {}
    ref1 = manager.ref(container1, 'ref1')
    ref2 = manager.ref(container2, 'ref2')

    ref1['a'] = 1
    ref2['a'] = 2

    assert container1['a'] == 1
    assert ref1['a']._get_value() == 1
    assert container2['a'] == 2
    assert ref2['a']._get_value() == 2

    assert container1['b'] == 3
    assert ref1['b']._get_value() == 3


def test_expr_task():
    manager = xd.Manager()
    container = {'b': 3, 'c': 4}
    ref = manager.ref(container, 'ref')
    target = ref['a']
    expr = 2 * ref['b'] + 3 * ref['c']

    expr_task = xdeps.tasks.ExprTask(target, expr)

    assert expr_task.expr == expr
    assert expr_task.taskid == target
    assert expr_task.dependencies == {ref['b'], ref['c']}
    assert expr_task.targets == {target}

    assert container == {'b': 3, 'c': 4}
    expr_task.run()
    assert container == {'a': 18, 'b': 3, 'c': 4}


def test_depenv():
    manager = xd.Manager()
    container = {'b': 3, 'c': 4}
    ref = manager.ref(container, 'ref')
    ref['a'] = ref['b'] + ref['c']

    depenv = xd.tasks.DepEnv(container, ref)

    assert (depenv['b'] == 3) is True
    assert (depenv['c'] == 4) is True
    assert (depenv['a'] == 7) is True
    assert set(depenv.keys()) == {'a', 'b', 'c'}

    depenv['b'] = 5
    depenv.c = 6
    assert (depenv['a'] == 11) is True

    assert (depenv._eval('a + b')._value == 16) is True


@pytest.fixture(scope='function')
def example_manager():
    manager = xd.Manager()

    container1 = {'a': 12, 'b': 8, 'c': -4, 'g': 42}
    ref1 = manager.ref(container1, 'ref1')

    container2 = {'d': [2, 3]}
    ref2 = manager.ref(container2, 'ref2')

    # We are realising the following setup in the test:
    #
    #            ╭ [a]─┐                                      Legend:
    #            │     │                                      [x] ref/expr x
    # container1 │ [b]─┴─►(f=a+b)─►[f]─┐                      (x) task x
    #            │                     │
    #            ╰ [c]───►(g=-c)─►[g]──┴─►(h=√(f-g))─►[h]─┐
    #                                                     │
    #                                                     │
    #            ╭ [d[0]]─┐                               │
    # container2 │        │                               │
    #            ╰ [d[1]]─┴───►(e[0]=d[0]*d[1])───►[e[0]]─┴─►(j=h+e[0])─►[j]

    ref1['f'] = ref1['a'] + ref1['b']

    ref1['h'] = (ref1['f'] - ref1['g']) ** 0.5

    # Define [g] after use to test to fully test manager.register with rtasks.
    ref1['g'] = -ref1['c']

    ref2['e'] = [None]
    ref2['e'][0] = ref2['d'][0] * ref2['d'][1]

    ref2['j'] = ref1['h'] + ref2['e'][0]

    return manager, [ref1, ref2], [container1, container2]


def test_manager(example_manager):
    manager, refs, containers = example_manager
    ref1, ref2 = refs
    container1, container2 = containers

    # First we check that everything was registered correctly
    f_expr = ref1['a'] + ref1['b']
    h_expr = (ref1['f'] - ref1['g']) ** 0.5
    g_expr = -ref1['c']
    e0_expr = ref2['d'][0] * ref2['d'][1]
    j_expr = ref1['h'] + ref2['e'][0]

    # Check that tasks are correctly created.
    assert len(manager.tasks) == 5
    assert manager.tasks[ref1['f']].expr.equals(f_expr)
    assert manager.tasks[ref1['g']].expr.equals(g_expr)
    assert manager.tasks[ref1['h']].expr.equals(h_expr)
    assert manager.tasks[ref2['e'][0]].expr.equals(e0_expr)
    assert manager.tasks[ref2['j']].expr.equals(j_expr)

    # Check that rdeps are correctly created.
    assert len(manager.rdeps) == 11
    assert set(manager.rdeps[ref1['a']]) == {ref1['f']}  # [f] depends on [a]
    assert set(manager.rdeps[ref1['b']]) == {ref1['f']}  # [f] depends on [b]
    assert set(manager.rdeps[ref1['c']]) == {ref1['g']}  # [g] depends on [c]
    assert set(manager.rdeps[ref1['f']]) == {ref1['h']}  # [h] depends on [f]
    assert set(manager.rdeps[ref1['g']]) == {ref1['h']}  # [h] depends on [g]
    # [e[0]] and thus [e] depend on [d[0]], [d[1]], and thus [d]:
    assert set(manager.rdeps[ref2['d']]) == {ref2['e'], ref2['e'][0]}
    assert set(manager.rdeps[ref2['d'][0]]) == {ref2['e'], ref2['e'][0]}
    assert set(manager.rdeps[ref2['d'][1]]) == {ref2['e'], ref2['e'][0]}
    assert set(manager.rdeps[ref1['h']]) == {ref2['j']}  # [j] depends on [h]
    assert set(manager.rdeps[ref2['e'][0]]) == {ref2['j']}  # [j] depends on [e[0]]
    assert set(manager.rdeps[ref2['e']]) == {ref2['j']}  # thus [j] depends on [e]

    # Check that rtasks are correctly created.
    # This is a special case as all tasks so far output one ref.
    # In this case is subset of rdeps as it has taskids on the lhs (and since
    # not all refs are tasks, we have fewer entries than in rdeps).
    assert len(manager.rtasks) == 4
    assert set(manager.rtasks[ref1['h']]) == {ref2['j']}
    assert set(manager.rtasks[ref2['e'][0]]) == {ref2['j']}
    assert set(manager.rtasks[ref1['f']]) == {ref1['h']}
    assert set(manager.rtasks[ref1['g']]) == {ref1['h']}

    # Check that deptasks are correctly created.
    # Almost the same as rdeps but maps to taskids instead of refs, so:
    # - We have an entry for [j],
    # - [e[0]] etc. are on the rhs but not [e].
    assert len(manager.deptasks) == 12
    assert set(manager.deptasks[ref1['a']]) == {ref1['f']}
    assert set(manager.deptasks[ref1['b']]) == {ref1['f']}
    assert set(manager.deptasks[ref1['c']]) == {ref1['g']}
    assert set(manager.deptasks[ref1['f']]) == {ref1['h']}
    assert set(manager.deptasks[ref1['g']]) == {ref1['h']}
    assert set(manager.deptasks[ref2['d']]) == {ref2['e'][0]}
    assert set(manager.deptasks[ref2['d'][0]]) == {ref2['e'][0]}
    assert set(manager.deptasks[ref2['d'][1]]) == {ref2['e'][0]}
    assert set(manager.deptasks[ref1['h']]) == {ref2['j']}
    assert set(manager.deptasks[ref2['e'][0]]) == {ref2['j']}
    assert set(manager.deptasks[ref2['e']]) == {ref2['j']}
    assert set(manager.deptasks[ref2['j']]) == set()

    # Check that tartasks are correctly created.
    # In this case it is (roughly) an identity on refs that are tasks,
    # and zero otherwise.
    assert len(manager.tartasks) == 12
    # Tasks:
    for ref in [ref1['f'], ref1['g'], ref1['h'], ref2['e'][0], ref2['j']]:
        assert set(manager.tartasks[ref]) == {ref}
    # Extra considerations for e for e[0] etc.:
    assert set(manager.tartasks[ref2['e']]) == {ref2['e'][0]}
    # Refs that aren't tasks:
    for ref in [ref1['a'], ref1['b'], ref1['c'], ref2['d'], ref2['d'][0], ref2['d'][1]]:
        assert set(manager.tartasks[ref]) == set()

    # Check that containers are correctly created.
    assert manager.containers == {'ref1': ref1, 'ref2': ref2}

    # Now we check that the manager can correctly execute the tasks
    assert container1['f'] == 20
    assert container1['g'] == 4
    assert container1['h'] == 4
    assert container2['e'][0] == 6
    assert container2['j'] == 10

    # Test that values are updated correctly
    ref1['a'] = 21
    assert container2['j'] == 11


def test_manager_unregister(example_manager):
    manager, refs, containers = example_manager
    ref1, ref2 = refs
    container1, container2 = containers

    # Test unregistering a task
    ref1['h'] = 4
    assert container2['j'] == 10


def test_manager_clone_verify_refresh(example_manager):
    manager, refs, _ = example_manager
    ref1, ref2 = refs

    # Test consistency and make a copy
    manager.verify()
    manager_copy = manager.copy()

    # Mess up consistency and check again
    manager.rtasks[ref1['f']].append(ref2['e'][0])
    manager.rdeps[ref1['b']].remove(ref1['f'])

    with pytest.raises(ValueError) as error:
        manager.verify()
    assert 'not consistent' in str(error.value)

    manager_copy.verify()  # the copy should still be okay

    # Refresh and check again
    manager.refresh()
    manager.verify()


def test_manager_dump_and_load(example_manager):
    manager, _, _ = example_manager

    dumped = manager.dump()

    new_manager = xd.Manager()
    new_container1 = {'a': 10, 'b': 10, 'c': -4}
    new_ref1 = new_manager.ref(new_container1, 'new_ref1')
    new_container2 = {'d': [3, 2], 'e': [-99]}
    new_ref2 = new_manager.ref(new_container2, 'new_ref2')

    old_expr = new_ref2['j'] = 2 * new_ref1['a']  # this will be removed by load

    new_containers = {
        'ref1': new_ref1,
        'ref2': new_ref2,
    }
    new_manager.load(dumped, new_containers)

    assert new_containers['ref2']['j']._get_value() == 20
    assert not new_manager.tasks[new_ref2['j']].expr == old_expr


def test_manager_pickle(example_manager):
    manager, _, original_containers = example_manager
    original1, original2 = original_containers

    pickled = pickle.dumps(manager)
    new_manager = pickle.loads(pickled)

    ref1 = new_manager.containers['ref1']
    container1 = ref1._owner
    ref2 = new_manager.containers['ref2']
    container2 = ref2._owner

    ref1['a'] = 32
    ref2['d'][1] = 5

    assert container1 is not original1
    assert container2 is not original2
    assert container1['f'] == 40
    assert container1['h'] == 6
    assert container2['e'][0] == 10
    assert container2['j'] == 16


def test_ref_count():
    manager = xd.Manager()
    container = {'a': 3, 'b': [None], 'c': None}
    ref = manager.ref(container, 'ref')

    ref['c'] = ref['b'][0]
    ref['b'][0] = ref['a']

    assert len(manager.rtasks) == 1
    assert len(manager.rtasks[ref['b'][0]]) == 1
    assert manager.rtasks[ref['b'][0]][ref['c']] == 2
    # This is a bit weird, the task (b[0]) influences [c] twice.
    # One time directly, and the other comes from the fact that the task
    # (b[0]) influences both the ref [b[0]] and [b]. Therefore both of these
    # refs, in turn, influence [c].
    # The setup of this example requires [a], as we need two tasks, the first
    # one being (b[0]) and the second one being (a).

    assert container == {'a': 3, 'b': [3], 'c': 3}


def test_function_task():
    manager = xd.Manager()
    container = {'a': 3, 'b': 4}
    ref = manager.ref(container, 'ref')

    def action():
        ref['c'] = ref['a']._get_value() + ref['b']._get_value()
        ref['d'] = ref['a']._get_value() * ref['b']._get_value()

    task = xd.tasks.FunctionTask(
        taskid='my_action',
        action=action,
        targets={ref['c'], ref['d']},
        dependencies={ref['a'], ref['b']},
    )

    manager.register(task)
    task.run()

    assert container == {'a': 3, 'b': 4, 'c': 7, 'd': 12}

    ref['a'] = 4
    assert container == {'a': 4, 'b': 4, 'c': 8, 'd': 16}

    assert len(manager.tasks) == 1
    assert manager.tasks['my_action'] is task

    assert len(manager.rdeps) == 2
    assert set(manager.rdeps[ref['a']]) == {ref['c'], ref['d']}
    assert set(manager.rdeps[ref['b']]) == {ref['c'], ref['d']}

    assert len(manager.deptasks) == 4
    assert set(manager.deptasks[ref['a']]) == {'my_action'}
    assert set(manager.deptasks[ref['b']]) == {'my_action'}
    assert set(manager.deptasks[ref['c']]) == set()
    assert set(manager.deptasks[ref['d']]) == set()

    assert len(manager.tartasks) == 4
    assert set(manager.tartasks[ref['a']]) == set()
    assert set(manager.tartasks[ref['b']]) == set()
    assert set(manager.tartasks[ref['c']]) == {'my_action'}
    assert set(manager.tartasks[ref['d']]) == {'my_action'}


@pytest.mark.xfail(reason='Not implemented yet, unclear if it should be')
def test_function_task_depends_on_container():
    manager = xd.Manager()
    container_source = {'a': 3, 'b': 4}
    container_target = {}
    ref_source = manager.ref(container_source, 'ref_source')
    ref_target = manager.ref(container_target, 'ref_target')

    def action():
        ref_target['c'] = ref_source['a']._get_value() + ref_source['b']._get_value()
        ref_target['d'] = ref_source['a']._get_value() * ref_source['b']._get_value()

    task = xd.tasks.FunctionTask(
        taskid='my_action',
        action=action,
        targets={ref_target['c'], ref_target['d']},
        dependencies={ref_source},
    )

    manager.register(task)
    task.run()

    assert container_target == {'c': 7, 'd': 12}
    ref_source['a'] = 4
    assert container_target == {'c': 8, 'd': 16}


def test_lsa_knob_example():
    manager = xd.Manager()
    container = {'src': 10, 'tar': list(range(5))}
    ref = manager.ref(container, 'ref')

    weights = np.linspace(0, 1, 5)
    lsa = xd.tasks.LinearKnob(
        taskid='lsa',
        source=ref['src'],
        weights=weights,
        targets=[ref['tar'][ii] for ii in range(5)],
    )

    manager.register(lsa)
    lsa.run()

    assert container['tar'] == [0, 1, 2, 3, 4]

    ref['src'] = 20
    assert container['tar'] == [0.0, 3.5, 7.0, 10.5, 14.0]
