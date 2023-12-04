# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
import pickle

import xdeps
from xdeps.tasks import AttrDict


def test_set():
    v = {'a': 1, 'b': 3}
    mgr = xdeps.Manager()
    r = mgr.ref(v)
    r['c'] = 0.1 * r['a'] + 0.2 * r['b']
    assert v['c'] == 0.1 * v['a'] + 0.2 * v['b']
    r['a'] = 1.2
    assert v['c'] == 0.1 * v['a'] + 0.2 * v['b']


def test_attrref():
    class E:
        pass

    e = E()
    e.knl = [0, 1, 2, 3]
    v = {4: e, 2: 1.2}

    mgr = xdeps.Manager()
    v_ = mgr.ref(v, 'v')
    v_[4].knl[1] = v_[2] * 3

    assert v[4].knl[1] == v[2] * 3
    v_[2] = 1.1
    assert v[4].knl[1] == v[2] * 3


def test_inplace():
    m = xdeps.Manager()

    s = {'a': 1, 'b': 2}
    s_ = m.ref(s, 's')
    s_._exec('c=a+b')
    assert s['c'] == s['a'] + s['b']
    s_['c'] += 1  ## s_['a']= s_['c']._expr + 1
    assert s['c'] == s['a'] + s['b'] + 1
    s_['a'] += 1  ## s_['a']= s['a'] + 1
    assert s['c'] == s['a'] + s['b'] + 1


def test_unregister_implicit():
    mgr = xdeps.Manager()
    vd = {}
    v = mgr.ref(vd, 'v')
    v['a'] = 1
    v['b'] = v['a'] * 3

    assert v['b'] in mgr.tasks
    assert v['b'] in mgr.rdeps[v['a']]
    assert v['b'] in mgr.deptasks[v['a']]
    assert v['b'] in mgr.tartasks[v['b']]

    v['b'] = 0

    assert v['b'] not in mgr.tasks
    assert v['b'] not in mgr.rdeps[v['a']]
    assert v['b'] not in mgr.deptasks[v['a']]
    assert v['b'] not in mgr.tartasks[v['b']]


def test_unregister():
    mgr = xdeps.Manager()

    container = {
        'a': 1,
        'b': 2,
        'c': {
            'x': 4,
            'y': 6,
        }
    }

    ref = mgr.ref(container, 'ref')
    ref['c']['x'] = 2 * ref['a'] * ref['b']
    ref['c']['y'] = 3 * ref['a'] * ref['b']

    ref['a'] = 2
    ref['b'] = 2

    assert container['c']['x'] == 8
    assert container['c']['y'] == 12

    mgr.unregister(ref['c']['y'])

    ref['a'] = 1

    assert container['c']['x'] == 4
    assert container['c']['y'] == 12  # unchanged


def test_collisions():
    num_elements = 100000

    mgr = xdeps.Manager()

    elements = {f'bend{ii}': AttrDict(k1=0) for ii in range(num_elements)}
    element_refs = mgr.ref(elements, 'element_refs')

    vars = {}
    var_refs = mgr.ref(vars, 'var_refs')

    for ii in range(num_elements):
        var_refs[f'k{ii}'] = 1
        element_refs[f'bend{ii}'].k0 = var_refs[f'k{ii}']
        assert len(mgr.tasks) == ii + 1

    assert len(mgr.tasks) == num_elements
