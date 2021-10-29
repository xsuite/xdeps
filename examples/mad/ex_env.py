import time

from cpymad.madx import Madx
mad=Madx(stdout=False)
mad.call("lhc.seq")
mad.call("optics.madx")

import xdeps.madxutils

m=xdeps.madxutils.MadxEnv(mad)
m.manager.find_deps([m._vref.on_x1])
m.manager.find_tasks([m._vref.on_x1])

for aa in range(100,106):
    m.v.on_x1=aa
    print(m.v.on_x1, m.e['mcbcv.5r1.b2'].kick)

myf=m.manager.gen_fun("myset",on_x1=m._vref['on_x1'])

for aa in range(100,106):
    myf(aa)
    print(m.v.on_x1, m.e['mcbcv.5r1.b2'].kick)

#%timeit m.v.on_x1=aa
#%timeit myf(aa)




