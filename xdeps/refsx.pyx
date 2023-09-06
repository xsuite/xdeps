import cython

@cython.cclass
class ARef:
    def __add__(self,other):
        return AddRef(self,other)

@cython.cclass
class AddRef(ARef):
    a: cython.declare(object, visibility='public')
    b: cython.declare(object, visibility='public')

    def __init__(self,a,b):
        self.a=a
        self.b=b

    def __repr__(self):
        return f"({self.a}+{self.b})"

    def __hash__(self):
        return hash(('AddRef',self.a,self.b))

    def _get_value(self):
        a=self.a
        b=self.b
        if hasattr(a,'_get_value'):
            a=a._get_value()
        if hasattr(b,'_get_value'):
            b=b._get_value()
        return a+b

