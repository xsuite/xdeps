# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xdeps
import math

el = {"el": {"a": 1, "b": 2}}
va = {"a": 2, "b.c": 4}


def test_madxeval():
    ma = xdeps.madxutils.MadxEval(va, math, el)
    assert ma.eval("+1+2^-2") == 1 + 2**-2
    assert ma.eval("b.c") == va["b.c"]
    assert ma.eval("sin(3)^2") == math.sin(3) ** 2
    assert ma.eval("el->a*el->b") == el["el"]["a"] * el["el"]["b"]


def test_madxeval_attr():
    el2 = {"el": type("el", (), el["el"])}
    ma = xdeps.madxutils.MadxEval(va, math, el2, get="attr")
    assert ma.eval("el->a*el->b") == el2["el"].a * el2["el"].b
