from collections import defaultdict
import math

from lark import Lark, Transformer, v_args
from .tasks import Manager
from .utils import AttrDict

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

    NAME: /[a-z_\.][a-z0-9_\.%]*/
    %import common.NUMBER
    %import common.WS_INLINE
    %ignore WS_INLINE
"""

@v_args(inline=True)
class MadxEval(Transformer):
    from operator import add, sub, mul, truediv as div
    from operator import neg, pos, pow
    number = float

    def __init__(self,variables,functions,elements,get='item'):
        self.variables = variables
        self.functions = functions
        self.elements  = elements
        grammar=calc_grammar
        if get == 'attr':
            grammar=grammar.replace('getitem','getattr')
        self.eval=Lark(grammar, parser='lalr',
                         transformer=self).parse

    def assign_var(self, name, value):
        self.variables[name] = value
        return value

    def call(self,name,*args):
        ff=getattr(self.functions,name)
        return ff(*args)

    def getitem(self,name,key):
        return self.elements[name[1]][key]

    def getattr(self,name,key):
        return getattr(self.elements[name[1]],key)

    def var(self, name):
        try:
            return self.variables[name.value]
        except KeyError:
            raise Exception("Variable not found: %s" % name)

def test():
    import math
    from collections import defaultdict
    madx=MadxEval(defaultdict(lambda :0),{},math)
    print(madx.eval("+1+2^-2"))
    print(madx.eval("a.b"))
    print(madx.eval("1+a.b*-3"))
    print(madx.eval("sin(3)^2"))


class Mix:
    __slots__=('_v','_r')
    def __init__(self,data,refs):
        object.__setattr__(self,'_v',data)
        object.__setattr__(self,'_r',refs)

    def __getattr__(self,key):
        return self._v[key]

    def __getitem__(self,key):
        return self._v[key]

    def __setattr__(self,key,value):
        self._r[key]=value

    def __setitem__(self,key,value):
        self._r[key]=value

    def _eval(self,expr):
        return self._r._eval(expr)

class MadxEnv:
    def __init__(self,mad=None):
        self._variables=defaultdict(lambda :0)
        self._elements={}
        self.manager=Manager()
        self._vref=self.manager.ref(self._variables,'v')
        self._eref=self.manager.ref(self._elements,'e')
        self._fref=self.manager.ref(math,'f')
        self.madexpr=MadxEval(self._vref,self._fref,self._eref).eval
        self.madeval=MadxEval(self._variables,math,self._elements).eval
        self.v=Mix(self._variables,self._vref)
        self.e=Mix(self._elements,self._eref)
        if mad is not None:
           self.read_state(mad)

    def dump(self):
        return {'variables':self._variables,
             'elements':self._elements,
             'xdeps':self.manager.dump()}

    def load(self,data):
        self._variables.update(data['variables'])
        self._elements.update(data['elements'])
        self.manager.load(data['xdeps'])

    def to_json(self,filename):
        import json
        json.dump(self.dump(),open(filename,'w'))

    @classmethod
    def from_json(cls,filename):
        import json
        self=cls()
        self.load(json.load(open(filename)))
        return self

    def read_state(self,mad):
        for name,par in mad.globals.cmdpar.items():
            if par.expr is None:
                self._variables[name]=par.value
            else:
                self._vref[name]=self.madexpr(par.expr)

        for name,elem in mad.elements.items():
            elemdata=AttrDict()
            for parname, par in elem.cmdpar.items():
                elemdata[parname]=par.value
            self._elements[name]=elemdata

        for name,elem in mad.elements.items():
            for parname, par in elem.cmdpar.items():
                if par.expr is not None:
                    if par.dtype==12: # handle lists
                        for ii,ee in enumerate(par.expr):
                            if ee is not None:
                                self._eref[name][parname][ii]=self.madexpr(ee)
                    else:
                        self._eref[name][parname]=self.madexpr(par.expr)


if __name__ == '__main__':
    test()

