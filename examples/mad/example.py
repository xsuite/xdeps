from cpymad.madx import Madx
mad=Madx()
mad.call("lhc.seq")
mad.call("optics.madx")

variables={}
for name,par in mad.globals.cmdpar.items():
    variables[name]=par.value

elements={}
for name,elem in mad.elements.items():
    elemdata={}
    for parname, par in elem.cmdpar.items():
        elemdata[parname]=par.value
    elements[name]=elemdata


import xdeps
import math

vref=xdeps.manager.ref(variables)
eref=xdeps.manager.ref(elements)
fref=xdeps.manager.ref(math)
madeval=xdeps.MadxEval(vref,fref,eref).eval

for name,par in mad.globals.cmdpar.items():
    if par.expr is not None:
        #print(name,par.expr)
        vref[name]=madeval(par.expr)

for name,elem in mad.elements.items():
    for parname, par in elem.cmdpar.items():
        if par.expr is not None:
            if par.dtype==12:
                for ii,ee in enumerate(par.expr):
                    if ee is not None:
                        #print(name,parname,ii,par.expr)
                        eref[name][parname][ii]=madeval(ee)
            else:
                #print(name,parname,par.expr)
                eref[name][parname]=madeval(par.expr)


