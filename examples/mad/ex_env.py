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

set_on_x1=m.manager.gen_fun("set_on_x1",on_x1=m._vref['on_x1'])

for aa in range(100,106):
    set_on_x1(aa)
    print(m.v.on_x1, m.e['mcbcv.5r1.b2'].kick)

#%timeit m.v.on_x1=aa
#%timeit myf(aa)




