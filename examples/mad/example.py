from cpymad.madx import Madx

mad=Madx()
mad.call("lhc.seq")
mad.call("optics.madx")

def get_all_expr(mad):
    pass

import xdeps as xdeps

ref=xdeps.manager.ref()




