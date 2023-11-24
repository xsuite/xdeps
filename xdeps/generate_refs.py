import builtins
import math
import operator

BINARY_OPS = {
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
    operator.ne: "!=",
    operator.ge: ">=",
    operator.gt: ">",
    operator.rshift: ">>",
    operator.lshift: "<<",
}

UNARY_OPS = {
    operator.neg: "-",
    operator.pos: "+",
    operator.invert: "~",
}

GUARD_ZERO_DIVISION = ['__truediv__', '__floordiv__', '__mod__']


def generate_binary_refs():
    for op, op_str in BINARY_OPS.items():
        op_name = op.__name__
        if op_name == 'and_':
            op_name = 'BitwiseAnd'
        elif op_name == 'or_':
            op_name = 'BitwiseOr'
        else:
            op_name = op_name.title()
        print(f"""\
@cython.cclass
class {op_name}Expr(BinOpExpr):
    _op_str = '{op_str}'

    def _get_value(self):
        lhs = ARef._mk_value(self._lhs)
        rhs = ARef._mk_value(self._rhs)
""")
        if op_name not in GUARD_ZERO_DIVISION:
            print(f"""\
        return lhs {op_str} rhs
""")
        else:
            print(f"""\
        try:
            return lhs {op_str} rhs
        except ZeroDivisionError:
            return float('nan')
""")


def generate_unary_refs():
    for op, op_str in UNARY_OPS.items():
        print(f"""\
@cython.cclass
class {op.__name__.title()}Expr(UnaryOpExpr):
    _op_str = '{op_str}'
    
    def _get_value(self):
        arg = ARef._mk_value(self._arg)
        return {op_str}arg
        
""")


generate_binary_refs()
generate_unary_refs()
