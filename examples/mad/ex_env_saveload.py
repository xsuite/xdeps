import time

from cpymad.madx import Madx
mad=Madx(stdout=False)
mad.call("lhc.seq")
mad.call("optics.madx")

import xdeps.madxutils

m=xdeps.madxutils.MadxEnv(mad)

m.to_json('data.json')

m2=xdeps.madxutils.MadxEnv.from_json('data.json')
m2.v.on_x1+=1
m2.e['mcbcv.5r1.b2']['kick']

