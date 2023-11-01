import math

import numpy as np
import pytest

from xdeps import refs, tasks


def test_zero_divisions():
    one = refs.PosExpr(1)
    zero = refs.PosExpr(0)

    assert np.isnan((one / zero)._get_value())
    assert np.isnan((one // zero)._get_value())
    assert np.isnan((one % zero)._get_value())


def test_unary_expressions():
    arg = refs.PosExpr(7)

    assert (+arg)._get_value() == 7
    assert (-arg)._get_value() == -7
    assert (~arg)._get_value() == ~7


def test_binary_expressions():
    lhs = refs.PosExpr(7)
    rhs = refs.PosExpr(13)

    # Test standard versions
    assert (lhs + rhs)._get_value() == 7 + 13
    assert (lhs - rhs)._get_value() == 7 - 13
    assert (lhs * rhs)._get_value() == 7 * 13
    assert (lhs / rhs)._get_value() == 7 / 13
    assert (lhs // rhs)._get_value() == 7 // 13
    assert (lhs % rhs)._get_value() == 7 % 13
    assert (lhs ** rhs)._get_value() == 7 ** 13
    assert (lhs << rhs)._get_value() == 7 << 13
    assert (lhs >> rhs)._get_value() == 7 >> 13
    assert (lhs & rhs)._get_value() == 7 & 13
    assert (lhs | rhs)._get_value() == 7 | 13
    assert (lhs ^ rhs)._get_value() == 7 ^ 13
    assert (lhs > rhs)._get_value() == (7 > 13)
    assert (lhs < rhs)._get_value() == (7 < 13)
    assert (lhs >= rhs)._get_value() == (7 >= 13)
    assert (lhs <= rhs)._get_value() == (7 <= 13)
    assert (lhs == rhs)._get_value() == (7 == 13)
    assert (lhs != rhs)._get_value() == (7 != 13)

    # Test r-versions
    assert (7 + rhs)._get_value() == 7 + 13
    assert (7 - rhs)._get_value() == 7 - 13
    assert (7 * rhs)._get_value() == 7 * 13
    assert (7 / rhs)._get_value() == 7 / 13
    assert (7 // rhs)._get_value() == 7 // 13
    assert (7 % rhs)._get_value() == 7 % 13
    assert (7 ** rhs)._get_value() == 7 ** 13
    assert (7 << rhs)._get_value() == 7 << 13
    assert (7 >> rhs)._get_value() == 7 >> 13
    assert (7 & rhs)._get_value() == 7 & 13
    assert (7 | rhs)._get_value() == 7 | 13
    assert (7 ^ rhs)._get_value() == 7 ^ 13
    assert (7 > rhs)._get_value() == (7 > 13)
    assert (7 < rhs)._get_value() == (7 < 13)
    assert (7 >= rhs)._get_value() == (7 >= 13)
    assert (7 <= rhs)._get_value() == (7 <= 13)
    assert (7 == rhs)._get_value() == (7 == 13)
    assert (7 != rhs)._get_value() == (7 != 13)


def test_matmul_expression():
    class DummyHashableMatrix:
        def __init__(self, value):
            self._matrix = np.array(value)

        def __matmul__(self, other):
            return DummyHashableMatrix(self._matrix @ other._matrix)

        def __pos__(self):
            return self

        def __hash__(self):
            return hash(self._matrix.tobytes())

        def __eq__(self, other):
            return np.all(self._matrix == other._matrix)

    matlhs = refs.PosExpr(DummyHashableMatrix([[1, 2], [3, 4]]))
    matrhs = refs.PosExpr(DummyHashableMatrix([[5], [6]]))
    assert (matlhs @ matrhs)._get_value() == DummyHashableMatrix([[17], [39]])


def test_matmul_expression_ref():
    container = {
        'a': np.array([[1, 2], [3, 4]]),
        'b': np.array([[5], [6]]),
    }

    manager = tasks.Manager()
    ref = manager.ref(container, 'ref')
    ref['c'] = ref['a'] @ ref['b']

    assert np.all(container['c'] == np.array([[17], [39]]))


def test_builtin_expression():
    lhs = refs.PosExpr(7)
    rhs = refs.PosExpr(13)
    fp = refs.PosExpr(-7.13)

    assert divmod(lhs, rhs)._get_value() == divmod(7, 13)
    assert round(fp)._get_value() == round(-7.13)
    assert math.trunc(fp)._get_value() == math.trunc(-7.13)
    assert math.floor(fp)._get_value() == math.floor(-7.13)
    assert math.ceil(fp)._get_value() == math.ceil(-7.13)
    assert abs(fp)._get_value() == abs(-7.13)
    # These will not work, as Python verifies the type of the result:
    # assert complex(lhs, rhs)._get_value() == complex(7, 13)
    # assert int(fp)._get_value() == int(-7.13)
    # assert float(lhs)._get_value() == float(7)


@pytest.mark.xfail(reason='Undefined behaviour')
def test_ref_inplace_ops():
    manager = tasks.Manager()
    container = np.array([1, 2])
    ref = manager.ref(container, 'ref')

    assert np.all(ref._get_value() == [1, 2])
    ref += 3
    assert np.all(ref._get_value() == [4, 5])

    # The below does not work, as it is not clear how to implement this in a way
    # that is universal. Is it even desired?
    assert np.all(container == [4, 5])


def test_itemref():
    manager = tasks.Manager()
    container = {'a': 1, 'b': 2}
    ref = manager.ref(container, 'ref')

    item_ref_a = ref['a']
    assert item_ref_a._get_value() == 1

    item_ref_b = ref['b']
    assert item_ref_b._get_value() == 2
    item_ref_b._set_value(3)
    assert item_ref_b._get_value() == 3


def test_attrref():
    class DummyClass:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    manager = tasks.Manager()
    container = DummyClass(1, 2)
    ref = manager.ref(container, 'ref')

    attr_ref_a = ref.a
    assert attr_ref_a._get_value() == 1

    attr_ref_b = ref.b
    assert attr_ref_b._get_value() == 2
    attr_ref_b._set_value(3)
    assert attr_ref_b._get_value() == 3


def test_call_ref():
    def func(arg1, arg2, *, kwarg1, kwarg2):
        return arg1, arg2, kwarg1, kwarg2

    manager = tasks.Manager()
    container = {'arg1': 1, 'kwarg1': 3}
    data_ref = manager.ref(container, 'container')

    manager = tasks.Manager()
    ref = manager.ref(func, 'func')

    call_ref = ref(data_ref['arg1'], 4, kwarg2=2, kwarg1=data_ref['kwarg1'])
    assert call_ref._get_value() == (1, 4, 3, 2)

    assert call_ref._get_dependencies() == {data_ref['arg1'], data_ref['kwarg1']}


def test_objectattrref():
    container = {'a': 3}
    manager = tasks.Manager()
    ref = manager.refattr(container, 'ref')

    assert ref.a._get_value() == 3

    ref.b = 42
    assert ref.b._get_value() == 42
    assert container['b'] == 42


def test_refcount():
    ref_count = refs.RefCount()
    ref_count.extend(['a', 'b'])
    ref_count.append('b')

    assert ref_count['a'] == 1
    assert ref_count['b'] == 2

    ref_count.append('b')
    assert ref_count['a'] == 1
    assert ref_count['b'] == 3

    ref_count.remove('a')
    assert 'a' not in ref_count
    assert ref_count['b'] == 3

    ref_count.remove('b')
    assert 'a' not in ref_count
    assert ref_count['b'] == 2


def test_cythonized():
    assert refs.is_cythonized()