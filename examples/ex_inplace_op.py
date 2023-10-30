# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xdeps

m = xdeps.Manager()

s = {}
s_ = m.ref(s, "s")

s["a"] = 1
s["b"] = 2
s_._exec("c=a+b")
assert s["c"] == s["a"] + s["b"]
s_["c"] += 1
assert s["c"] == s["a"] + s["b"] + 1
s_["a"] += 1
assert s["c"] == s["a"] + s["b"] + 1
