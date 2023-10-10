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
    operator.eq: "==",
    operator.ne: "!=",
    operator.ge: ">=",
    operator.gt: ">",
    operator.rshift: ">>",
    operator.lshift: "<<",
}


def generate_binary_refs():
    for op, op_str in BINARY_OPS.items():
        print(f"""\
@cython.cclass
class {op.__name__.title()}Expr(BinOpExpr):
    _op_str = '{op_str}'

    def _get_value(self):
        _a = ARef._mk_value(self._a)
        _b = ARef._mk_value(self._b)
        return _a {op_str} _b

""")


generate_binary_refs()
