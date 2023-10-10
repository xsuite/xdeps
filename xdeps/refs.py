# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

# cython: language_level=3

import cython
import operator
import builtins
import math


special_methods = {
    '__dict__',
    '__getstate__',
    '__setstate__',
    '__reduce__',
    '__reduce_cython__',
    '__wrapped__',
    '_ipython_canary_method_should_not_exist_',
    '__array_ufunc__',
    '__array_function__',
    '__array_struct__',
    '__array_interface__',
    '__array_prepare__',
    '__array_wrap__',
    '__array_finalize__',
    '__array__',
    '__array_priority__',
}


OPERATOR_SYMBOLS = {
    # Binary
    operator.add: "+",
    operator.sub: "-",
    operator.mul: "*",
    operator.matmul: "@",
    operator.truediv: "/",
    operator.floordiv: "//",
    operator.mod: "%",
    operator.pow: "**",
    operator.and_: "&",
    operator.or_: "|",
    operator.xor: "^",
    operator.lt: "<",
    operator.le: "<=",
    operator.eq: "==",
    operator.ne: "!=",
    operator.ge: ">=",
    operator.gt: ">",
    operator.rshift: ">>",
    operator.lshift: "<<",
    # Unary
    operator.neg: "-",
    operator.pos: "+",
    operator.invert: "~",
    # Inplace
    operator.iadd: "+",
    operator.isub: "-",
    operator.imul: "*",
    operator.imatmul: "@",
    operator.itruediv: "/",
    operator.ifloordiv: "//",
    operator.imod: "%",
    operator.ipow: "**",
    operator.ilshift: "<<",
    operator.irshift: ">>",
    operator.iand: "&",
    operator.ior: "|",
    operator.ixor: "^",
}


def _isref(obj):
    return isinstance(obj, ARef)


@cython.cclass
class ARef:
    _manager = cython.declare(object, visibility='public', value=None)
    _hash = cython.declare(int, visibility='private')

    def __init__(self, *args, **kwargs):
        raise TypeError("Cannot instantiate abstract class ARef")

    def __hash__(self):
        return self._hash

    @staticmethod
    def _mk_value(value):
        if isinstance(value, ARef):
            return value._get_value()
        else:
            return value

    def _get_value(self):
        raise NotImplementedError()

    @property
    def _value(self):
        return self._get_value()

    def _get_dependencies(self, out=None):
        return out or set()

    # order of precedence
    def __call__(self, *args, **kwargs):
        return CallRef(self, args, kwargs)

    def __getitem__(self, item):
        return ItemRef(self, item, self._manager)

    def __getattr__(self, attr):
        if attr in special_methods:
            raise AttributeError(attr)

        return AttrRef(self, attr, self._manager)

    # numerical unary  operator
    def __neg__(self):
        return UnOpRef(self, operator.neg)

    def __pos__(self):
        return UnOpRef(self, operator.pos)

    def __invert__(self):
        return UnOpRef(self, operator.invert)

    # numerical binary operators

    def __add__(self, other):
        return AddExpr(self, other)

    def __radd__(self, other):
        return BinOpRef(other, self, operator.add)

    def __sub__(self, other):
        return BinOpRef(self, other, operator.sub)

    def __rsub__(self, other):
        return BinOpRef(other, self, operator.sub)

    def __mul__(self, other):
        return BinOpRef(self, other, operator.mul)

    def __rmul__(self, other):
        return BinOpRef(other, self, operator.mul)

    def __matmul__(self, other):
        return BinOpRef(self, other, operator.matmul)

    def __rmatmul__(self, other):
        return BinOpRef(other, self, operator.matmul)

    def __truediv__(self, other):
        return BinOpRef(self, other, operator.truediv)

    def __rtruediv__(self, other):
        return BinOpRef(other, self, operator.truediv)

    def __floordiv__(self, other):
        return BinOpRef(self, other, operator.floordiv)

    def __rfloordiv__(self, other):
        return BinOpRef(other, self, operator.floordiv)

    def __mod__(self, other):
        return BinOpRef(self, other, operator.mod)

    def __rmod__(self, other):
        return BinOpRef(other, self, operator.mod)

    def __pow__(self, other):
        return BinOpRef(self, other, operator.pow)

    def __rpow__(self, other):
        return BinOpRef(other, self, operator.pow)

    def __and__(self, other):
        return BinOpRef(self, other, operator.and_)

    def __rand__(self, other):
        return BinOpRef(other, self, operator.and_)

    def __or__(self, other):
        return BinOpRef(self, other, operator.or_)

    def __ror__(self, other):
        return BinOpRef(other, self, operator.or_)

    def __xor__(self, other):
        return BinOpRef(self, other, operator.xor)

    def __rxor__(self, other):
        return BinOpRef(other, self, operator.xor)

    def __lt__(self, other):
        return BinOpRef(self, other, operator.lt)

    def __rlt__(self, other):
        return BinOpRef(other, self, operator.lt)

    def __le__(self, other):
        return BinOpRef(self, other, operator.le)

    def __rle__(self, other):
        return BinOpRef(other, self, operator.le)

    def __eq__(self, other):
        return BinOpRef(self, other, operator.eq)

    def __req__(self, other):
        return BinOpRef(other, self, operator.eq)

    def __ne__(self, other):
        return BinOpRef(self, other, operator.ne)

    def __rne__(self, other):
        return BinOpRef(other, self, operator.ne)

    def __ge__(self, other):
        return BinOpRef(self, other, operator.ge)

    def __rge__(self, other):
        return BinOpRef(other, self, operator.ge)

    def __gt__(self, other):
        return BinOpRef(self, other, operator.gt)

    def __rgt__(self, other):
        return BinOpRef(other, self, operator.gt)

    def __rshift__(self, other):
        return BinOpRef(self, other, operator.rshift)

    def __rrshift__(self, other):
        return BinOpRef(other, self, operator.rshift)

    def __lshift__(self, other):
        return BinOpRef(self, other, operator.lshift)

    def __rlshift__(self, other):
        return BinOpRef(other, self, operator.lshift)

    def __divmod__(self, other):
        return BuiltinRef(self, builtins.divmod, (other,))

    def __round__(self, other=0):
        return BuiltinRef(self, builtins.round, (other,))

    def __trunc__(self):
        return BuiltinRef(self, math.trunc)

    def __floor__(self):
        return BuiltinRef(self, math.floor)

    def __ceil__(self):
        return BuiltinRef(self, math.ceil)

    def __abs__(self):
        return BuiltinRef(self, builtins.abs)

    def __complex__(self):
        return BuiltinRef(self, builtins.complex)

    def __int__(self):
        return BuiltinRef(self, builtins.int)

    def __float__(self):
        return BuiltinRef(self, builtins.float)


@cython.cclass
class MutableRef(ARef):
    _owner = cython.declare(object, visibility='public', value=None)
    _key = cython.declare(object, visibility='public', value=None)

    def __init__(self, _owner, _key, _manager):
        self._owner = _owner
        self._key = _key
        self._manager = _manager
        self._hash = hash((self.__class__.__name__, _owner, _key))

    def __setitem__(self, key, value):
        ref = ItemRef(self, key, self._manager)
        self._manager.set_value(ref, value)

    def __setattr__(self, attr, value):
        """Set a built-in attribute of the object or create an AttrRef.

        This should only be called during __init__, when unpickling, or to
        create an AttrRef for attrs that are not already defined on the object.
        In other cases the behaviour between python and cython is different,
        due to the way cython handles __setattr__.

        For this method to work correctly, all subclasses of MutableRef must
        be cython cdef classes (decorated with @cython.cclass).
        """
        if attr in dir(self):
            if not cython.compiled:
                object.__setattr__(self, attr, value)
                return
            else:
                # The above way of setting attributes does not work in Cython,
                # as the object does not have a __dict__. We do not really need
                # a setter for those though, as the only time we need to
                # set a "built-in" attribute is during __init__ or when
                # unpickling, and both of those cases are handled by Cython
                # without the usual pythonic call to __setattr__.
                raise AttributeError(f"Attribute {attr} is read-only.")

        ref = AttrRef(self, attr, self._manager)
        self._manager.set_value(ref, value)

    def _get_dependencies(self, out=None):
        if out is None:
            out = set()
        if isinstance(self._owner, ARef):
            self._owner._get_dependencies(out)
        if isinstance(self._key, ARef):
            self._key._get_dependencies(out)
        out.add(self)
        return out

    def _eval(self, expr, gbl=None):
        if gbl is None:
            gbl = {}
        return eval(expr, gbl, self)  # noqa

    def _exec(self, expr, gbl=None):
        if gbl is None:
            gbl = {}
        exec(expr, gbl, self)  # noqa

    @property
    def _tasks(self):
        return self._manager.tartasks[self]

    def _find_dependant_targets(self):
        return self._manager.find_deps([self])

    @property
    def _expr(self):
        if self in self._manager.tasks:
            task = self._manager.tasks[self]
            if hasattr(task, "expr"):
                return task.expr

    def _info(self, limit=10):

        print(f"#  {self}._get_value()")
        try:
            value = self._get_value()
            print(f"   {self} = {value}")
        except NotImplementedError:
            print(f"#  {self} has no value")
        print()

        if self in self._manager.tasks:
            task = self._manager.tasks[self]
            print(f"#  {self}._expr")
            print(f"   {task}")
            print()
            if hasattr(task, "info"):
                task.info()
        else:
            print(f"#  {self}._expr is None")
            print()

        refs = self._manager.find_deps([self])[1:]
        limit = limit or len(refs)
        if len(refs) == 0:
            print(f"#  {self} does not influence any target")
            print()
        else:
            print(f"#  {self}._find_dependant_targets()")
            for tt in refs[:limit]:
                if tt._expr is not None:
                    print(f"   {tt}")
            if len(refs) > limit:
                print(f"   ... set _info(limit=None) to get all lines")
            print()

    def __iadd__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr + other
        else:
            return self._get_value() + other

    def __ifloordiv__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr // other
        else:
            return self._get_value() // other

    def __ilshift__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr << other
        else:
            return self._get_value() << other

    def __imatmul__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr @ other
        else:
            return self._get_value() @ other

    def __imod__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr % other
        else:
            return self._get_value() % other

    def __imul__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr * other
        else:
            return self._get_value() * other

    def __ipow__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr**other
        else:
            return self._get_value() ** other

    def __irshift__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr >> other
        else:
            return self._get_value() >> other

    def __isub__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr - other
        else:
            return self._get_value() - other

    def __itruediv__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr / other
        else:
            return self._get_value() / other

    def __ixor__(self, other):
        newexpr = self._expr
        if newexpr:
            return newexpr ^ other
        else:
            return self._get_value() ^ other


@cython.cclass
class Ref(MutableRef):
    """
    A reference in the top-level container.
    """
    def __init__(self, _owner, _key, _manager):
        self._owner = _owner
        self._key = _key
        self._manager = _manager
        self._hash = hash((self.__class__.__name__, _key))

    def __repr__(self):
        return self._key

    def _get_value(self):
        return ARef._mk_value(self._owner)

    def _get_dependencies(self, out=None):
        return out


@cython.cclass
class AttrRef(MutableRef):
    def _get_value(self):
        owner = ARef._mk_value(self._owner)
        attr = ARef._mk_value(self._key)
        return getattr(owner, attr)

    def _set_value(self, value):
        owner = ARef._mk_value(self._owner)
        attr = ARef._mk_value(self._key)
        setattr(owner, attr, value)

    def __repr__(self):
        return f"{self._owner}.{self._key}"


@cython.cclass
class ItemRef(MutableRef):
    def _get_value(self):
        owner = ARef._mk_value(self._owner)
        item = ARef._mk_value(self._key)
        return owner[item]

    def _set_value(self, value):
        owner = ARef._mk_value(self._owner)
        item = ARef._mk_value(self._key)
        owner[item] = value

    def __repr__(self):
        return f"{self._owner}[{repr(self._key)}]"


@cython.cclass
class BinOpRef(ARef):
    _a = cython.declare(object, visibility='public')
    _b = cython.declare(object, visibility='public')
    _op = cython.declare(object, visibility='public')  # callable

    def __init__(self, _a, _b, _op):
        self._a = _a
        self._b = _b
        self._op = _op
        self._hash = hash((self._op, self._a, self._b))

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        try:
            ret = self._op(_a, _b)
        except ZeroDivisionError:
            ret = float("nan")
        return ret

    def _get_dependencies(self, out=None):
        _a = self._a
        _b = self._b
        if out is None:
            out = set()
        if isinstance(_a, ARef):
            _a._get_dependencies(out)
        if isinstance(_b, ARef):
            _b._get_dependencies(out)
        return out

    def __repr__(self):
        op_symbol = OPERATOR_SYMBOLS[self._op]
        return f"({self._a} {op_symbol} {self._b})"


@cython.cclass
class BinOpExpr(ARef):
    _a = cython.declare(object, visibility='public')
    _b = cython.declare(object, visibility='public')

    def __init__(self, _a, _b):
        self._a = _a
        self._b = _b
        self._hash = hash((self.__class__, self._a, self._b))

    def _get_value(self):
        raise NotImplementedError()

    def _get_dependencies(self, out=None):
        _a = self._a
        _b = self._b
        if out is None:
            out = set()
        if isinstance(_a, ARef):
            _a._get_dependencies(out)
        if isinstance(_b, ARef):
            _b._get_dependencies(out)
        return out

    def __repr__(self):
        return f"({self._a} {self._op_str} {self._b})"


@cython.cclass
class AddExpr(BinOpExpr):
    _op_str = '+'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a + _b


@cython.cclass
class SubExpr(BinOpExpr):
    _op_str = '-'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a - _b


@cython.cclass
class MulExpr(BinOpExpr):
    _op_str = '*'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a * _b


@cython.cclass
class MatmulExpr(BinOpExpr):
    _op_str = '@'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a @ _b


@cython.cclass
class TruedivExpr(BinOpExpr):
    _op_str = '/'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a / _b


@cython.cclass
class FloordivExpr(BinOpExpr):
    _op_str = '//'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a // _b


@cython.cclass
class ModExpr(BinOpExpr):
    _op_str = '%'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a % _b


@cython.cclass
class PowExpr(BinOpExpr):
    _op_str = '**'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a ** _b


@cython.cclass
class And_Expr(BinOpExpr):
    _op_str = '&'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a & _b


@cython.cclass
class Or_Expr(BinOpExpr):
    _op_str = '|'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a | _b


@cython.cclass
class XorExpr(BinOpExpr):
    _op_str = '^'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a ^ _b


@cython.cclass
class LtExpr(BinOpExpr):
    _op_str = '<'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a < _b


@cython.cclass
class LeExpr(BinOpExpr):
    _op_str = '<='

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a <= _b


@cython.cclass
class EqExpr(BinOpExpr):
    _op_str = '=='

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a == _b


@cython.cclass
class NeExpr(BinOpExpr):
    _op_str = '!='

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a != _b


@cython.cclass
class GeExpr(BinOpExpr):
    _op_str = '>='

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a >= _b


@cython.cclass
class GtExpr(BinOpExpr):
    _op_str = '>'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a > _b


@cython.cclass
class RshiftExpr(BinOpExpr):
    _op_str = '>>'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a >> _b


@cython.cclass
class LshiftExpr(BinOpExpr):
    _op_str = '<<'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a << _b


@cython.cclass
class UnOpRef(ARef):
    _a = cython.declare(object, visibility='public')
    _op = cython.declare(object, visibility='public')

    def __init__(self, a, op):
        self._a = a
        self._op = op
        self._hash = hash((self._op, self._a))

    def _get_value(self):
        a = ARef._mk_value(self._a)
        return self._op(a)

    def _get_dependencies(self, out=None):
        a = self._a
        if out is None:
            out = set()
        if isinstance(a, ARef):
            a._get_dependencies(out)
        return out

    def __repr__(self):
        op_symbol = OPERATOR_SYMBOLS[self._op]
        return f"({op_symbol}{self._a})"


@cython.cclass
class BuiltinRef(ARef):
    _a = cython.declare(object, visibility='public')
    _op = cython.declare(object, visibility='public')
    _params = cython.declare(tuple, visibility='public')

    def __init__(self, a, op, params=()):
        self._a = a
        self._op = op
        self._params = params
        self._hash = hash((self._op, self._a, self._params))

    def _get_value(self):
        a = ARef._mk_value(self._a)
        return self._op(a, *self._params)

    def _get_dependencies(self, out=None):
        a = self._a
        if out is None:
            out = set()
        if isinstance(a, ARef):
            a._get_dependencies(out)
        return out

    def __repr__(self):
        op_symbol = OPERATOR_SYMBOLS.get(self._op, self._op.__name__)
        return f"{op_symbol}({self._a})"


@cython.cclass
class CallRef(ARef):
    _func = cython.declare(object, visibility='public')
    _args = cython.declare(tuple, visibility='public')
    _kwargs = cython.declare(tuple, visibility='public')

    def __init__(self, func, args, kwargs):
        self._func = func
        self._args = args
        self._kwargs = tuple(kwargs.items())
        self._hash = hash((self._func, self._args, self._kwargs))

    def _get_value(self):
        func = ARef._mk_value(self._func)
        args = [ARef._mk_value(a) for a in self._args]
        kwargs = {n: ARef._mk_value(v) for n, v in self._kwargs}
        return func(*args, **kwargs)

    def _get_dependencies(self, out=None):
        if out is None:
            out = set()
        if isinstance(self._func, ARef):
            self._func._get_dependencies(out)
        for arg in self._args:
            if isinstance(arg, ARef):
                arg._get_dependencies(out)
        for name, arg in self._kwargs:
            if isinstance(arg, ARef):
                arg._get_dependencies(out)
        return out

    def __repr__(self):
        args = []
        for aa in self._args:
            args.append(repr(aa))
        for k, v in self._kwargs:
            args.append(f"{k}={v}")
        args = ", ".join(args)
        if isinstance(self._func, ARef):
            fname = repr(self._func)
        else:
            fname = self._func.__name__
        return f"{fname}({args})"


class RefContainer:
    """
    A list implementation that does not use __eq__ for comparisons. It is used
    for storing tasks, which need to be compared by their hash, as the usual
    == operator yields an expression, which is always True.
    """

    def __init__(self, *args, **kwargs):
        self.list = list(*args, **kwargs)

    def __repr__(self):
        return f"RefContainer({self.list})"

    def __contains__(self, item):
        try:
            self.index(item)
            return True
        except ValueError:
            return False

    def __getitem__(self, item):
        return self.list[item]

    def __delitem__(self, index):
        self.list.pop(index)

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(self.list)

    def index(self, item):
        for ii, x in enumerate(self.list):
            if hash(item) == hash(x):
                return ii
        raise ValueError(f"{item} is not in list")

    def extend(self, other):
        if isinstance(other, RefContainer):
            other = other.list
        self.list.extend(other)

    def append(self, item):
        self.list.append(item)

    def remove(self, item):
        del self[self.index(item)]


class RefCount(dict):
    def append(self, item):
        self[item] = self.get(item, 0) + 1

    def extend(self, other):
        for kk in other:
            self.append(kk)

    def remove(self, item):
        occ = self[item]
        if occ > 1:
            self[item] = occ - 1
        else:
            del self[item]
