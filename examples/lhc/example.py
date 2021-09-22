from cpymad.madx import Madx

mad=Madx()
mad.call("lhc.seq")
mad.call("optics.madx")

