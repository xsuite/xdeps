# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from collections import defaultdict
import math
import numpy as np

from lark import Lark, Transformer, v_args
from .tasks import Manager
from .utils import AttrDict
from .table import Table

calc_grammar = """
    ?start: sum
        | NAME "=" sum      -> assign_var

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: power
        | product "*" power  -> mul
        | product "/" power  -> div

    ?power: atom
        | power "^" atom    -> pow

    ?atom: NUMBER           -> number
         | "-" atom         -> neg
         | "+" atom         -> pos
         | NAME             -> var
         | NAME "->" NAME   -> getitem
         | NAME "(" sum ("," sum)* ")" -> call
         | "(" sum ")"

    NAME: /[a-z_\\.][a-z0-9_\\.%]*/
    %import common.NUMBER
    %import common.WS_INLINE
    %ignore WS_INLINE
"""


@v_args(inline=True)
class MadxEval(Transformer):
    from operator import add, sub, mul, truediv as div
    from operator import neg, pos, pow

    number = float

    def __init__(self, variables, functions, elements, get="item"):
        """
        variables: dict of variables
        elements: dict of elements
        functions: module of functions such as `math` or `numpy`

        get: 'item' if items in elements support el['key'], 'attr' if they support el.key
        """
        self.variables = variables
        self.functions = functions
        self.elements = elements
        grammar = calc_grammar
        if get == "attr":
            grammar = grammar.replace("getitem", "getattr")
        self.eval = Lark(grammar, parser="lalr", transformer=self).parse

    def assign_var(self, name, value):
        self.variables[name] = value
        return value

    def call(self, name, *args):
        ff = getattr(self.functions, name)
        return ff(*args)

    def getitem(self, name, key):
        return self.elements[name.value][key.value]

    def getattr(self, name, key):
        return getattr(self.elements[name], key)

    def var(self, name):
        try:
            return self.variables[name.value]
        except KeyError:
            raise Exception("Variable not found: %s" % name)


def test():
    import math
    from collections import defaultdict

    madx = MadxEval(defaultdict(lambda: 0), math, {})
    print(madx.eval("+1+2^-2"))
    print(madx.eval("a.b"))
    print(madx.eval("1+a.b*-3"))
    print(madx.eval("sin(3)^2"))


class Mix:
    __slots__ = ("_v", "_r")

    def __init__(self, data, refs):
        object.__setattr__(self, "_v", data)
        object.__setattr__(self, "_r", refs)

    def __getattr__(self, key):
        return self._v[key]

    def __getitem__(self, key):
        return self._v[key]

    def __setattr__(self, key, value):
        self._r[key] = value

    def __setitem__(self, key, value):
        self._r[key] = value

    def _eval(self, expr):
        return self._r._eval(expr)


class View:
    def __init__(self, obj, ref, evaluator):
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_ref", ref)
        object.__setattr__(self, "_eval", evaluator)

    def _get_viewed_object(self):
        return object.__getattribute__(self, "_obj")

    @property
    def __class__(self):
        # return type("View",(self._obj.__class__,),{})
        return self._obj.__class__

    def get_value(self, key=None):
        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            raise ValueError("get_value not supported for this object")

        if key is None:
            if hasattr(self._obj, "_xofields"):
                return {kk: self.get_value(kk) for kk in self._obj._xofields}
            else:
                return {kk: self.get_value(kk) for kk in dir(self._obj)}

        if hasattr(self._obj, "__iter__"):
            return self._obj[key]
        else:
            return getattr(self._obj, key)

    def get_expr(self, key=None, index=None):
        if index is not None:
            if key is None:
                raise ValueError('`key` must be provided when `index` is provided.')
            return getattr(self._ref, key)[index]._expr

        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            # is an array
            return self._ref[key]._expr

        if key is None:
            if hasattr(self._obj, "_xofields"):
                return {kk: self.get_expr(kk) for kk in self._obj._xofields}
            else:
                return {kk: self.get_expr(kk) for kk in dir(self._obj)}

        if hasattr(self._obj, "__iter__"):
            return self._ref[key]._expr
        else:
            return getattr(self._ref, key)._expr


    def get_info(self, key=None):
        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            raise ValueError("get_info not supported for this object")

        if key is None:
            print("Element of type: ", self._obj.__class__.__name__)
            self.get_table().show(header=False)
        else:
            if hasattr(self._obj, "__iter__"):
                return self._ref[key]._info()
            else:
                return getattr(self._ref, key)._info()

    def get_table(self):

        if not hasattr(self._obj, "keys") and not hasattr(self._obj, "_xofields"):
            raise ValueError("get_table not supported for this object")

        out_expr = self.get_expr()
        out_value = self.get_value()

        value = [out_value[kk] for kk in out_expr.keys()]
        for ii, vv in enumerate(value):
            if not(np.isscalar(vv)):
                value[ii] = str(vv)

        data = {
            "name": np.array(list(out_expr.keys()), dtype=object),
            "value": np.array(value, dtype=object),
            "expr": np.array(
                [str(out_expr[kk]) for kk in out_expr.keys()], dtype=object
            ),
        }
        return Table(data)

    def __getattr__(self, key):
        val = getattr(self._obj, key)
        if hasattr(val, "__setitem__"):
            return View(val, getattr(self._ref, key), self._eval)
        else:
            return val

    def __getitem__(self, key):
        val = self._obj[key]
        if hasattr(val, "__setitem__"):
            return View(val, self._ref[key], self._eval)
        else:
            return val

    def __setattr__(self, key, value):
        if isinstance(value, str):
            value = self._eval(value)
        setattr(self._ref, key, value)

    def __setitem__(self, key, value):
        if isinstance(value, str):
            value = self._eval(value)
        self._ref[key] = value

    def __repr__(self):
        return f"View of {self._obj!r}"

    def __dir__(self):
        return dir(self._obj)

    def __len__(self):
        return len(self._obj)

    def __add__(self, other):
        return self._obj + other

    def __radd__(self, other):
        return other + self._obj

    def __sub__(self, other):
        return self._obj - other

    def __rsub__(self, other):
        return other - self._obj

    def __mul__(self, other):
        return self._obj * other

    def __rmul__(self, other):
        return other * self._obj

    def __truediv__(self, other):
        return self._obj / other

    def __rtruediv__(self, other):
        return other / self._obj

    def __floordiv__(self, other):
        return self._obj // other

    def __rfloordiv__(self, other):
        return other // self._obj

    def __mod__(self, other):
        return self._obj % other

    def __rmod__(self, other):
        return other % self._obj

    def __pow__(self, other):
        return self._obj**other

    def __rpow__(self, other):
        return other**self._obj

    def __eq__(self, value: object) -> bool:
        return self._obj == value

    def __ne__(self, value: object) -> bool:
        return self._obj != value

    def __lt__(self, value: object) -> bool:
        return self._obj < value

    def __le__(self, value: object) -> bool:
        return self._obj <= value

    def __gt__(self, value: object) -> bool:
        return self._obj > value

    def __ge__(self, value: object) -> bool:
        return self._obj >= value


class MadxEnv:
    def __init__(self, mad=None):
        self._variables = defaultdict(lambda: 0)
        self._elements = {}
        self.manager = Manager()
        self._vref = self.manager.ref(self._variables, "v")
        self._eref = self.manager.ref(self._elements, "e")
        self._fref = self.manager.ref(math, "f")
        self.madexpr = MadxEval(self._vref, self._fref, self._eref).eval
        self.madeval = MadxEval(self._variables, math, self._elements).eval
        self.v = Mix(self._variables, self._vref)
        self.e = Mix(self._elements, self._eref)
        if mad is not None:
            self.read_state(mad)

    def dump(self):
        return {
            "variables": self._variables,
            "elements": self._elements,
            "xdeps": self.manager.dump(),
        }

    def load(self, data):
        self._variables.update(data["variables"])
        self._elements.update(data["elements"])
        self.manager.load(data["xdeps"])

    def to_json(self, filename):
        import json

        json.dump(self.dump(), open(filename, "w"))

    @classmethod
    def from_json(cls, filename):
        import json

        self = cls()
        self.load(json.load(open(filename)))
        return self

    def read_state(self, mad):
        elemdata = AttrDict()
        elem = mad.beam
        for parname, par in elem.cmdpar.items():
            elemdata[parname] = par.value
        self._elements["beam"] = elemdata

        for name, par in mad.globals.cmdpar.items():
            if par.expr is None:
                self._variables[name] = par.value
            else:
                self._vref[name] = self.madexpr(par.expr)

        for name, elem in mad.elements.items():
            elemdata = AttrDict()
            for parname, par in elem.cmdpar.items():
                elemdata[parname] = par.value
            self._elements[name] = elemdata

        for name, elem in mad.elements.items():
            for parname, par in elem.cmdpar.items():
                if par.expr is not None:
                    if par.dtype == 12:  # handle lists
                        for ii, ee in enumerate(par.expr):
                            if ee is not None:
                                self._eref[name][parname][ii] = self.madexpr(ee)
                    else:
                        self._eref[name][parname] = self.madexpr(par.expr)


def to_madx(expr):
    if expr.__class__.__name__ == "NegExpr":
        return f"(-{to_madx(expr._arg)})"
    elif expr.__class__.__name__ == "PosExpr":
        return f"{to_madx(expr._arg)}"
    elif expr.__class__.__name__ == "AddExpr":
        return f"({to_madx(expr._lhs)}+{to_madx(expr._rhs)})"
    elif expr.__class__.__name__ == "SubExpr":
        return f"({to_madx(expr._lhs)}-{to_madx(expr._rhs)})"
    elif expr.__class__.__name__ == "MulExpr":
        return f"({to_madx(expr._lhs)}*{to_madx(expr._rhs)})"
    elif expr.__class__.__name__ == "DivExpr":
        return f"({to_madx(expr._lhs)}/{to_madx(expr._rhs)})"
    elif expr.__class__.__name__ == "PowExpr":
        return f"({to_madx(expr._lhs)}^{to_madx(expr._rhs)})"
    elif expr.__class__.__name__ == "ItemRef":  # shortcut
        return f"{expr._key}"
    else:
        return repr(expr)


if __name__ == "__main__":
    test()
