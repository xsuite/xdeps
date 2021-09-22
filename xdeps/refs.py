from dataclasses import dataclass, field
import operator, builtins, math

_binops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "@": operator.matmul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
    "&": operator.and_,
    "|": operator.or_,
    "^": operator.xor,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
    ">>": operator.rshift,
    "<<": operator.lshift,
}

_unops = {"-": operator.neg, "+": operator.pos, "~": operator.invert}

_builtins = {
    "divmod": builtins.divmod,
    "abs": builtins.abs,
    "complex": builtins.complex,
    "int": builtins.int,
    "float": builtins.float,
    "pow": math.pow,
    "round": builtins.round,
    "trunc": math.trunc,
    "floor": math.floor,
    "ceil": math.ceil,
}


def _pr_binop():
    for sy, op in _binops.items():
        fn = op.__name__.replace("_", "")
        rr = fn.capitalize()
        fmt = f"""
       def __{fn}__(self, other):
           return {rr}Ref(self, other)

       def __r{fn}__(self, other):
           return {rr}Ref(other, self)"""
        print(fmt)


def _pr_builtins():
    for sy, op in _builtins.items():
        fn = op.__name__.replace("_", "")
        rr = fn.capitalize()
        fmt = f"""
       def __{fn}__(self, other):
           return {rr}Ref(self, other)"""
        print(fmt)


class Ref:
    @staticmethod
    def _mk_value(value):
        if isinstance(value, Ref):
            return value._get_value()
        else:
            return value

    # order of precedence
    def __call__(self, *args, **kwargs):
        return CallRef(self, args, kwargs)

    def __getitem__(self, item):
        return ItemRef(self, item, self._manager)

    def __getattr__(self, attr):
        return AttrRef(self, attr, self._manager)

    # numerical unary  operator
    def __neg__(self):
        return NegRef(self)

    def __pos__(self):
        return PosRef(self)

    def __invert__(self):
        return InvertRef(self)

    # numerical binary operators

    def __add__(self, other):
        return AddRef(self, other)

    def __radd__(self, other):
        return AddRef(other, self)

    def __sub__(self, other):
        return SubRef(self, other)

    def __rsub__(self, other):
        return SubRef(other, self)

    def __mul__(self, other):
        return MulRef(self, other)

    def __rmul__(self, other):
        return MulRef(other, self)

    def __matmul__(self, other):
        return MatmulRef(self, other)

    def __rmatmul__(self, other):
        return MatmulRef(other, self)

    def __truediv__(self, other):
        return TruedivRef(self, other)

    def __rtruediv__(self, other):
        return TruedivRef(other, self)

    def __floordiv__(self, other):
        return FloordivRef(self, other)

    def __rfloordiv__(self, other):
        return FloordivRef(other, self)

    def __mod__(self, other):
        return ModRef(self, other)

    def __rmod__(self, other):
        return ModRef(other, self)

    def __pow__(self, other):
        return PowRef(self, other)

    def __rpow__(self, other):
        return PowRef(other, self)

    def __and__(self, other):
        return AndRef(self, other)

    def __rand__(self, other):
        return AndRef(other, self)

    def __or__(self, other):
        return OrRef(self, other)

    def __ror__(self, other):
        return OrRef(other, self)

    def __xor__(self, other):
        return XorRef(self, other)

    def __rxor__(self, other):
        return XorRef(other, self)

    def __lt__(self, other):
        return LtRef(self, other)

    def __rlt__(self, other):
        return LtRef(other, self)

    def __le__(self, other):
        return LeRef(self, other)

    def __rle__(self, other):
        return LeRef(other, self)

    def __eq__(self, other):
        return EqRef(self, other)

    def __req__(self, other):
        return EqRef(other, self)

    def __ne__(self, other):
        return NeRef(self, other)

    def __rne__(self, other):
        return NeRef(other, self)

    def __ge__(self, other):
        return GeRef(self, other)

    def __rge__(self, other):
        return GeRef(other, self)

    def __gt__(self, other):
        return GtRef(self, other)

    def __rgt__(self, other):
        return GtRef(other, self)

    def __rshift__(self, other):
        return RshiftRef(self, other)

    def __rrshift__(self, other):
        return RshiftRef(other, self)

    def __lshift__(self, other):
        return LshiftRef(self, other)

    def __rlshift__(self, other):
        return LshiftRef(other, self)

    def __divmod__(self, other):
        return DivmodRef(self, other)

    def __abs__(self, other):
        return AbsRef(self, other)

    def __complex__(self, other):
        return ComplexRef(self, other)

    def __int__(self, other):
        return IntRef(self, other)

    def __float__(self, other):
        return FloatRef(self, other)

    def __pow__(self, other):
        return PowRef(self, other)

    def __round__(self, other):
        return RoundRef(self, other)

    def __trunc__(self, other):
        return TruncRef(self, other)

    def __floor__(self, other):
        return FloorRef(self, other)

    def __ceil__(self, other):
        return CeilRef(self, other)


@dataclass(frozen=True, unsafe_hash=True)
class AttrRef(Ref):
    _owner: object
    _attr: str
    _manager: object

    def _get_value(self):
        owner = Ref._mk_value(self._owner)
        attr = Ref._mk_value(self._attr)
        return getattr(owner, attr)

    def _get_dependencies(self, out=None):
        if out is None:
            out = []
        if isinstance(self._owner, Ref):
            self._owner._get_dependencies(out)
        if isinstance(self._attr, Ref):
            self._attr._get_dependencies(out)
        out.append(self)
        return out

    def __repr__(self):
        return f"{self._owner}.{self._attr}"


@dataclass(frozen=True)
class ItemRef(Ref):
    _owner: object
    _item: int
    _manager: object

    def __hash__(self):
        return hash((id(self._owner),self._item))

    def _get_value(self):
        owner = Ref._mk_value(self._owner)
        item = Ref._mk_value(self._item)
        return owner[item]

    def _get_dependencies(self, out=None):
        if out is None:
            out = []
        if isinstance(self._owner, Ref):
            self._owner._get_dependencies(out)
        if isinstance(self._item, Ref):
            self._item._get_dependencies(out)
        out.append(self)
        return out

    def _set_value(self, value):
        self._manager.set_value(self,value)

    def __repr__(self):
        return f"{self._owner}[{self._item}]"



@dataclass(frozen=True, unsafe_hash=True)
class BinOpRef(Ref):
    _a: object
    _b: object

    def _get_value(self):
        a = Ref._mk_value(self._a)
        b = Ref._mk_value(self._b)
        return self._op(a, b)

    def _get_dependencies(self, out=None):
        a = self._a
        b = self._b
        if out is None:
            out = []
        if isinstance(a, Ref):
            a._get_dependencies(out)
        if isinstance(b, Ref):
            b._get_dependencies(out)
        return out

    def __repr__(self):
        return f"({self._a}{self._st}{self._b})"


@dataclass(frozen=True, unsafe_hash=True)
class UnOpRef(Ref):
    _a: object

    def _get_value(self):
        a = Ref._mk_value(self._a)
        return self._op(a)

    def _get_dependencies(self, out=None):
        a = self._a
        if out is None:
            out = []
        if isinstance(a, Ref):
            a._get_dependencies(out)
        return out

    def __repr__(self):
        return f"({self._st}{self._a})"


@dataclass(frozen=True, unsafe_hash=True)
class BuiltinRef(Ref):
    _a: object

    def _get_value(self):
        a = Ref._mk_value(self._a)
        return self._op(a)

    def _get_dependencies(self, out=None):
        a = self._a
        if out is None:
            out = []
        if isinstance(a, Ref):
            a._get_dependencies(out)
        return out

    def __repr__(self):
        return f"{self._st}({self._a})"


gbl = globals()
for st, op in _binops.items():
    fn = op.__name__.replace("_", "")
    cn = f"{fn.capitalize()}Ref"
    mn = f"__{fn}__"
    gbl[cn] = type(cn, (BinOpRef,), {"_op": op, "_st": st})

for st, op in _unops.items():
    fn = op.__name__.replace("_", "")
    cn = f"{fn.capitalize()}Ref"
    if cn in gbl:
        raise ValueError
    mn = f"__{fn}__"
    gbl[cn] = type(cn, (UnOpRef,), {"_op": op, "_st": st})

for st, op in _builtins.items():
    fn = op.__name__.replace("_", "")
    cn = f"B{fn.capitalize()}Ref"
    if cn in gbl:
        raise ValueError
    mn = f"__{fn}__"
    gbl[cn] = type(cn, (BuiltinRef,), {"_op": op, "_st": st})


@dataclass(frozen=True, unsafe_hash=True)
class CallRef(Ref):
    _func: object
    _args: tuple
    _kwargs: tuple

    def _get_value(self):
        func = Ref._mk_value(self._func)
        args = [Ref._mk_value(a) for a in self._args]
        kwargs = {n: Ref._mk_value(v) for n, v in self._kwargs}
        return self._func(*args, **dict(kwargs))

    def _get_dependencies(self, out=None):
        if out is None:
            out = []
        if isinstance(self._func, Ref):
            self._func._get_dependencies(out)
        for arg in self._args:
            if isinstance(arg, Ref):
                arg._get_dependencies(out)
        for name, arg in self._kwargs:
            if isinstance(arg, Ref):
                arg._get_dependencies(out)
        return out

    def __repr__(self):
        args=[]
        for aa in self._args:
            args.append(repr(aa))
        for k,v in self._kwargs:
            args.append(f"{k}={v}")
        args=','.join(args)
        return f"{self._func}({args})"

