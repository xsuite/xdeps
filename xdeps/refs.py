# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

# cython: language_level=3

import builtins
import cython
import logging
import math
import operator

logger = logging.getLogger(__name__)

special_methods = {
    '__copy__',
    '__deepcopy__',
    '__dict__',
    '__getstate__',
    '__setstate__',
    '__reduce_cython__',
    '__wrapped__',
    '__array_ufunc__',
    '__array_function__',
    '__array_struct__',
    '__array_interface__',
    '__array_prepare__',
    '__array_wrap__',
    '__array_finalize__',
    '__array__',
    '__array_priority__',
    # _ipython_canary_method_should_not_exist_ is used by IPython to detect
    # if an object 'lies' about its attributes due to its __getattr__. We should
    # not try to intercept it.
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


def is_ref(obj):
    return isinstance(obj, BaseRef)


def _isref(obj):
    # This is never used within this module, let's not prefix it with '_'.
    logger.warning("xdeps.refs._isref is deprecated, use is_ref instead.")
    return is_ref(obj)


def is_cythonized():
    return cython.compiled


@cython.cclass
class BaseRef:
    """
    An abstract base class for all reference/expression objects. Such an object
    represents a value computation, based on dependent values/other refs. E.g.,
    a simple reference to a value contained in a Python object (or a dictionary,
    list field), or a more complex expression, such as a binary operation, whose
    arguments can be refs themselves.

    Notes:
        Cannot override __complex__, because Python requires it to return a
        complex value, but we want it to be a Ref object. The same is true for
        __int__, __float__, __bool__. The previous implementations are therefore
        removed, and no workaround is provided until an explicit need for these
        methods arises.
    """
    _manager = cython.declare(object, visibility='readonly', value=None)
    _hash = cython.declare(int, visibility='readonly')

    def __init__(self, *args, **kwargs):
        # To keep compatibility with pure Python (useful for debugging simpler
        # issues), we simulate Cython __cinit__ behaviour with this __init__:
        if not is_cythonized():
            for base in type(self).__mro__:
                cinit = getattr(base, '__cinit__', None)
                if cinit:
                    cinit(self, *args, **kwargs)

    def __hash__(self):
        return self._hash

    def __reduce__(self):
        raise TypeError("Cannot pickle an abstract class")

    def __eq__(self, other):
        """Check equality of the expressions `self` and `other`.

        Check if `self` and `other` are the same. Crucially, even if self._value
        and other._value are the same, this does not necessarily imply that
        self == other. Python requires that if a == b then hash(a) == hash(b).
        Since in our case two different ref objects may have the same hash (by
        virtue of representing the same expression!), we must not use __eq__ to
        create expressions.

        If a deferred equality expression is desired, use `self._eq`.
        """
        return str(self) == str(other)

    def _eq(self, other):
        """Deferred expression for the equality of values of `self` and `other`."""
        return EqExpr(self, other)

    def _neq(self, other):
        """Deferred expression for the inequality of values of `self` and `other`."""
        return NeExpr(self, other)

    @staticmethod
    def _mk_value(value):
        if isinstance(value, BaseRef):
            return value._get_value()
        else:
            return value

    def _get_value(self):
        raise NotImplementedError

    @property
    def _value(self):
        try:
            return self._get_value()
        except AttributeError:
            # Python's default __getattribute__ implementation falls back to
            # __getattr__ if a property getter raises AttributeError. This will
            # have an effect of silencing the error in our special case of
            # ItemRef and AttrRef, so here we reraise it as a different type.
            raise LookupError(
                f"The reference '{self}' points to a non-existent field."
            )

    def _get_dependencies(self, out=None):
        return out or set()

    # order of precedence
    def __call__(self, *args, **kwargs):
        return CallRef(self, args, kwargs)

    def __getitem__(self, item):
        return ItemRef(self, item, self._manager)

    def __getattr__(self, attr):
        if attr in special_methods:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'")

        return AttrRef(self, attr, self._manager)

    # numerical unary  operator
    def __neg__(self):
        return NegExpr(self)

    def __pos__(self):
        return PosExpr(self)

    def __invert__(self):
        return InvertExpr(self)

    # numerical binary operators
    def __add__(self, other):
        return AddExpr(self, other)

    def __radd__(self, other):
        return AddExpr(other, self)

    def __sub__(self, other):
        return SubExpr(self, other)

    def __rsub__(self, other):
        return SubExpr(other, self)

    def __mul__(self, other):
        return MulExpr(self, other)

    def __rmul__(self, other):
        return MulExpr(other, self)

    def __matmul__(self, other):
        return MatmulExpr(self, other)

    def __rmatmul__(self, other):
        return MatmulExpr(other, self)

    def __truediv__(self, other):
        return TruedivExpr(self, other)

    def __rtruediv__(self, other):
        return TruedivExpr(other, self)

    def __floordiv__(self, other):
        return FloordivExpr(self, other)

    def __rfloordiv__(self, other):
        return FloordivExpr(other, self)

    def __mod__(self, other):
        return ModExpr(self, other)

    def __rmod__(self, other):
        return ModExpr(other, self)

    def __pow__(self, other):
        return PowExpr(self, other)

    def __rpow__(self, other):
        return PowExpr(other, self)

    def __and__(self, other):
        return BitwiseAndExpr(self, other)

    def __rand__(self, other):
        return BitwiseAndExpr(other, self)

    def __or__(self, other):
        return BitwiseOrExpr(self, other)

    def __ror__(self, other):
        return BitwiseOrExpr(other, self)

    def __xor__(self, other):
        return XorExpr(self, other)

    def __rxor__(self, other):
        return XorExpr(other, self)

    def __lt__(self, other):
        return LtExpr(self, other)

    def __rlt__(self, other):
        return LtExpr(other, self)

    def __le__(self, other):
        return LeExpr(self, other)

    def __rle__(self, other):
        return LeExpr(other)

    def __ge__(self, other):
        return GeExpr(self, other)

    def __rge__(self, other):
        return GeExpr(other, self)

    def __gt__(self, other):
        return GtExpr(self, other)

    def __rgt__(self, other):
        return GtExpr(other, self)

    def __rshift__(self, other):
        return RshiftExpr(self, other)

    def __rrshift__(self, other):
        return RshiftExpr(other, self)

    def __lshift__(self, other):
        return LshiftExpr(self, other)

    def __rlshift__(self, other):
        return LshiftExpr(other, self)

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


@cython.cclass
class MutableRef(BaseRef):
    """
    An (abstract) class representing a reference to a mutable object.
    """
    _owner = cython.declare(object, visibility='readonly', value=None)
    _key = cython.declare(object, visibility='readonly', value=None)

    def __cinit__(self, _owner, _key, _manager):
        self._owner = _owner
        self._key = _key
        self._manager = _manager
        # _hash will depend on the particularities of the subclass, for now
        # it is None, which does not matter, as this class should never be
        # instantiated.

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
                # set a "built-in" attribute is during __cinit__ or when
                # unpickling, and both of those cases are handled by Cython
                # without the usual pythonic call to __setattr__.
                raise AttributeError(f"Attribute {attr} is read-only.")

        ref = AttrRef(self, attr, self._manager)
        self._manager.set_value(ref, value)

    def __reduce__(self):
        """Do not store the hash when pickling.

        The hash is only guaranteed to be the same for the 'same' refs within
        the same python instance, therefore serialising hashes makes no sense.
        """
        return type(self), (self._owner, self._key, self._manager)

    def _get_dependencies(self, out=None):
        if out is None:
            out = set()
        if isinstance(self._owner, BaseRef):
            self._owner._get_dependencies(out)
        if isinstance(self._key, BaseRef):
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
        """All tasks that influence this expression."""
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
        try:
            value = self._get_value()
            print(f"#  {self}._get_value()")
            print(f"   {self} = {value}")
        except AttributeError:
            print(f"#  The field '{self}' does not exist!!!")
        except NotImplementedError:
            print(f"#  '{self}' does not implement _get_value()!!!")
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
    def __cinit__(self, _owner, _key, _manager):
        # Cython automatically calls __cinit__ in the base classes
        self._hash = hash((type(self).__name__, _key))

    def __repr__(self):
        return self._key

    def _get_value(self):
        return BaseRef._mk_value(self._owner)

    def _get_dependencies(self, out=None):
        return out


@cython.cclass
class ObjectAttrRef(Ref):
    """Like `Ref`, but translates attribute access to item access."""

    def __getattr__(self, attr):
        if attr in special_methods:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'")

        return ItemRef(self, attr, self._manager)

    def __setattr__(self, attr, value):
        """Set a built-in attribute of the object or create an ItemRef.

        For notes on why the implementation is different in Cython, see
        the __setattr__ method of `Ref`.
        """
        if attr in dir(self):
            if not cython.compiled:
                object.__setattr__(self, attr, value)
                return
            else:
                raise AttributeError(f"Attribute {attr} is read-only.")

        ref = ItemRef(self, attr, self._manager)
        self._manager.set_value(ref, value)


@cython.cclass
class AttrRef(MutableRef):
    def __cinit__(self, _owner, _key, _manager):
        # Cython automatically calls __cinit__ in the base classes
        self._hash = hash((type(self).__name__, _owner, _key))

    def _get_value(self):
        owner = BaseRef._mk_value(self._owner)
        attr = BaseRef._mk_value(self._key)
        return getattr(owner, attr)

    def _set_value(self, value):
        owner = BaseRef._mk_value(self._owner)
        attr = BaseRef._mk_value(self._key)
        setattr(owner, attr, value)

    def __repr__(self):
        assert self._owner is not None
        assert self._key is not None
        return f"{self._owner}.{self._key}"


@cython.cclass
class ItemRef(MutableRef):
    def __cinit__(self, _owner, _key, _manager):
        # Cython automatically calls __cinit__ in the base classes
        self._hash = hash((type(self).__name__, _owner, _key))

    def _get_value(self):
        owner = BaseRef._mk_value(self._owner)
        item = BaseRef._mk_value(self._key)
        return owner[item]

    def _set_value(self, value):
        owner = BaseRef._mk_value(self._owner)
        item = BaseRef._mk_value(self._key)
        owner[item] = value

    def __repr__(self):
        assert self._owner is not None
        return f"{self._owner}[{repr(self._key)}]"


@cython.cclass
class BinOpExpr(BaseRef):
    """
    This _abstract_ class represents a binary operation on two refs/literals.

    When overriden, the subclass must define the following:

    1. A method `_get_value` which returns the value of the expression. Tests
       have determined that this way of implementing operators (by specialising)
       classes instead of using the generic `operator` methods is ~15% faster
       in both evaluating the expressions and building the dependency graph.
    2. A property `_op_str` which returns the string representation of the
       operator for pretty printing.
    """
    _lhs = cython.declare(object, visibility='readonly')
    _rhs = cython.declare(object, visibility='readonly')

    def __cinit__(self, lhs, rhs):
        self._lhs = lhs
        self._rhs = rhs
        self._hash = hash((self.__class__, lhs, rhs))

    def _get_value(self):
        raise NotImplementedError()

    def _get_dependencies(self, out=None):
        if out is None:
            out = set()
        if isinstance(self._lhs, BaseRef):
            self._lhs._get_dependencies(out)
        if isinstance(self._rhs, BaseRef):
            self._rhs._get_dependencies(out)
        return out

    def __reduce__(self):
        """Instruct pickle to not pickle the hash."""
        return type(self), (self._lhs, self._rhs)

    def __repr__(self):
        return f"({self._lhs} {self._op_str} {self._rhs})"


@cython.cclass
class UnaryOpExpr(BaseRef):
    """
    This _abstract_ class represents a unary operation on a ref.

    When overriden, the subclass must define the following:

    1. A method `_get_value` which returns the value of the expression. Tests
       have determined that this way of implementing operators (by specialising)
       classes instead of using the generic `operator` methods is ~15% faster
       in both evaluating the expressions and building the dependency graph.
    2. A cython property `_op_str` which returns the string representation of
       the operator for pretty printing.
    """
    _arg = cython.declare(object, visibility='readonly')

    def __cinit__(self, arg):
        self._arg = arg
        self._hash = hash((self.__class__, self._arg))

    def _get_value(self):
        raise NotImplementedError()

    def _get_dependencies(self, out=None):
        # The check for if `self._arg` is a ref (as for BinOpExpr) is not
        # necessary, as unless constructed manually (which should not ever
        # be done in the wild), the hypothetical literal would have been
        # evaluated to a different literal, and not this ref. Thus, for
        # performance reasons we skip the check.
        return self._arg._get_dependencies(out)

    def __reduce__(self):
        """Instruct pickle to not pickle the hash."""
        return type(self), (self._arg,)

    def __repr__(self):
        return f"({self._op_str}{self._arg})"


@cython.cclass
class AddExpr(BinOpExpr):
    _op_str = '+'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs + rhs


@cython.cclass
class SubExpr(BinOpExpr):
    _op_str = '-'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs - rhs


@cython.cclass
class MulExpr(BinOpExpr):
    _op_str = '*'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs * rhs


@cython.cclass
class MatmulExpr(BinOpExpr):
    _op_str = '@'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs @ rhs


@cython.cclass
class TruedivExpr(BinOpExpr):
    _op_str = '/'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        try:
            return lhs / rhs
        except ZeroDivisionError:
            return float('nan')


@cython.cclass
class FloordivExpr(BinOpExpr):
    _op_str = '//'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        try:
            return lhs // rhs
        except ZeroDivisionError:
            return float('nan')


@cython.cclass
class ModExpr(BinOpExpr):
    _op_str = '%'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        try:
            return lhs % rhs
        except ZeroDivisionError:
            return float('nan')


@cython.cclass
class PowExpr(BinOpExpr):
    _op_str = '**'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs ** rhs


@cython.cclass
class BitwiseAndExpr(BinOpExpr):
    _op_str = '&'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs & rhs


@cython.cclass
class BitwiseOrExpr(BinOpExpr):
    _op_str = '|'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs | rhs


@cython.cclass
class XorExpr(BinOpExpr):
    _op_str = '^'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs ^ rhs


@cython.cclass
class LtExpr(BinOpExpr):
    _op_str = '<'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs < rhs


@cython.cclass
class LeExpr(BinOpExpr):
    _op_str = '<='

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs <= rhs


@cython.cclass
class EqExpr(BinOpExpr):
    _op_str = '=='

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs == rhs


@cython.cclass
class NeExpr(BinOpExpr):
    _op_str = '!='

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs != rhs


@cython.cclass
class GeExpr(BinOpExpr):
    _op_str = '>='

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs >= rhs


@cython.cclass
class GtExpr(BinOpExpr):
    _op_str = '>'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs > rhs


@cython.cclass
class RshiftExpr(BinOpExpr):
    _op_str = '>>'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs >> rhs


@cython.cclass
class LshiftExpr(BinOpExpr):
    _op_str = '<<'

    def _get_value(self):
        lhs = BaseRef._mk_value(self._lhs)
        rhs = BaseRef._mk_value(self._rhs)
        return lhs << rhs


@cython.cclass
class NegExpr(UnaryOpExpr):
    _op_str = '-'

    def _get_value(self):
        arg = BaseRef._mk_value(self._arg)
        return -arg


@cython.cclass
class PosExpr(UnaryOpExpr):
    _op_str = '+'

    def _get_value(self):
        arg = BaseRef._mk_value(self._arg)
        return +arg


@cython.cclass
class InvertExpr(UnaryOpExpr):
    _op_str = '~'

    def _get_value(self):
        arg = BaseRef._mk_value(self._arg)
        return ~arg


@cython.cclass
class BuiltinRef(BaseRef):
    _arg = cython.declare(object, visibility='readonly')
    _op = cython.declare(object, visibility='readonly')
    _params = cython.declare(tuple, visibility='readonly')

    def __cinit__(self, arg, op, params=()):
        self._arg = arg
        self._op = op
        self._params = params
        self._hash = hash((self._op, self._arg, self._params))

    def _get_value(self):
        arg = BaseRef._mk_value(self._arg)
        return self._op(
            arg,
            *(BaseRef._mk_value(param) for param in self._params),
        )

    def _get_dependencies(self, out=None):
        arg = self._arg
        if out is None:
            out = set()
        if isinstance(arg, BaseRef):
            arg._get_dependencies(out)
        return out

    def __reduce__(self):
        """Instruct pickle to not pickle the hash."""
        return type(self), (self._op, self._args)

    def __repr__(self):
        op_symbol = OPERATOR_SYMBOLS.get(self._op, self._op.__name__)
        return f"{op_symbol}({self._arg})"


@cython.cclass
class CallRef(BaseRef):
    _func = cython.declare(object, visibility='readonly')
    _args = cython.declare(tuple, visibility='readonly')
    _kwargs = cython.declare(tuple, visibility='readonly')

    def __cinit__(self, func, args, kwargs):
        self._func = func
        self._args = args
        if isinstance(kwargs, dict):
            self._kwargs = tuple(kwargs.items())
        else:
            self._kwargs = tuple(kwargs)
        self._hash = hash((self._func, self._args, self._kwargs))

    def _get_value(self):
        func = BaseRef._mk_value(self._func)
        args = [BaseRef._mk_value(a) for a in self._args]
        kwargs = {n: BaseRef._mk_value(v) for n, v in self._kwargs}
        return func(*args, **kwargs)

    def _get_dependencies(self, out=None):
        if out is None:
            out = set()
        if isinstance(self._func, BaseRef):
            self._func._get_dependencies(out)
        for arg in self._args:
            if isinstance(arg, BaseRef):
                arg._get_dependencies(out)
        for name, arg in self._kwargs:
            if isinstance(arg, BaseRef):
                arg._get_dependencies(out)
        return out

    def __reduce__(self):
        """Instruct pickle to not pickle the hash."""
        return type(self), (self._func, self._args, self._kwargs)

    def __repr__(self):
        args = [repr(arg) for arg in self._args]
        args += [f"{k}={v}" for k, v in self._kwargs]
        args = ", ".join(args)

        if isinstance(self._func, BaseRef):
            fname = repr(self._func)
        else:
            fname = self._func.__name__

        return f"{fname}({args})"


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
