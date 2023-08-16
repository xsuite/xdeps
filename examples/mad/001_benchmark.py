# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import time
import json
import xdeps.madxutils


st0=time.time()
data=json.load(open('data.json'))
st1=time.time()
print(f"Loading json {st1-st0} sec")

m=xdeps.madxutils.MadxEnv()
st0=time.time()
m.load(data)
st1=time.time()
print(f"Loading expressions {st1-st0} sec")

st0=time.time()
m.manager.refresh()
st1=time.time()
print(f"Rebuild dict {st1-st0} sec")






