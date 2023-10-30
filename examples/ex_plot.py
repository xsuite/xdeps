# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import matplotlib.pyplot as plt
import numpy as np

a = 1
b = 1
c = 0
x = np.arange(0, 3.0, 0.1)
y = a * np.sin(b * x + c)
(pl,) = plt.plot(x, y)


def update(pl, x, y):
    pl.set_xdata(x)
    pl.set_ydata(y)


import xdeps

mgr = xdeps.Manager()
gbl = mgr.refattr(globals(), "gbl")

gbl._eval("y=a*np.sin(b*x+c)")
gbl._eval("myup=update(pl,x,y)")

gbl.a = 2.1
gbl.x[3] = 2.1

mgr.to_pydot([gbl.x])
