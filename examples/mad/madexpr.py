from lark import Lark, Transformer, v_args


try:
    input = raw_input   # For Python2 compatibility
except NameError:
    pass

calc_grammar = """
    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: power
        | product "*" atom  -> mul
        | product "/" atom  -> div

    ?power: atom
        | power "^" atom    -> pow

    ?atom: NUMBER           -> number
         | "-" atom         -> neg
         | "+" atom         -> pos
         | NAME             -> var
         | NAME "(" sum ("," sum)* ")" -> function
         | "(" sum ")"

    NAME: /[a-z_\.][a-z0-9_\.%]*/
    %import common.NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
"""


@v_args(inline=True)    # Affects the signatures of the methods
class CalculateTree(Transformer):
    from operator import add, sub, mul, truediv as div, neg, pos, pow
    number = float
    import math

    def __init__(self,ns):
        self.vars = {}

    def assign_var(self, name, value):
        self.vars[name] = value
        return value

    def function(self,name,*args):
        ff=getattr(self.math,name)
        return ff(*args)

    def var(self, name):
        try:
            return self.vars[name]
        except KeyError:
            raise Exception("Variable not found: %s" % name)

calc_parser = Lark(calc_grammar, parser='lalr', transformer=CalculateTree(ns))
calc = calc_parser.parse

def test():
    print(calc("+1+2^-2"))
    print(calc("1+a.b*-3"))
    print(calc("a.b"))
    print(calc("sin(3)^2"))


if __name__ == '__main__':
    test()

