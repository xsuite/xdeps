import time


def timeit():
    if hasattr(timeit,'start'):
        newstart=time.time()
        print(f"Elapsed: {newstart-timeit.start:13.9f} sec")
    timeit.start=time.time()



timeit()
from cpymad.madx import Madx
mad=Madx(stdout=False)
mad.call("lhc.seq")
mad.call("optics.madx")
timeit()

variables={}
for name,par in mad.globals.cmdpar.items():
    variables[name]=par.value

elements={}
for name,elem in mad.elements.items():
    elemdata={}
    for parname, par in elem.cmdpar.items():
        elemdata[parname]=par.value
    elements[name]=elemdata

timeit()

import xdeps
import math

manager=xdeps.DepManager()
vref=manager.ref(variables)
eref=manager.ref(elements)
fref=manager.ref(math)
madeval=xdeps.MadxEval(vref,fref,eref).eval

for name,par in mad.globals.cmdpar.items():
    if par.expr is not None:
        vref[name]=madeval(par.expr)

for name,elem in mad.elements.items():
    for parname, par in elem.cmdpar.items():
        if par.expr is not None:
            if par.dtype==12: # handle lists
                for ii,ee in enumerate(par.expr):
                    if ee is not None:
                        eref[name][parname][ii]=madeval(ee)
            else:
                eref[name][parname]=madeval(par.expr)

timeit()


import time
start=time.time()
print(elements['mcbcv.5r1.b2']['kick'])
vref['on_x1']=2
print(elements['mcbcv.5r1.b2']['kick'])
print(time.time()-start)

