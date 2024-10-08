# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xdeps

manager=xdeps.Manager()
m=manager.refattr(globals(),'m')

a=b=0
m.c=0.1*m.a+0.2*m.b
m.a=2
print(m.c)
m.b=3
print(m.c)






