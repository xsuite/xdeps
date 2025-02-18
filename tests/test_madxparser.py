# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xdeps
import math

elements = {"el": {"a": 1, "b": 2}}
vars = {"a": 2, "b.c": 4}


def test_madxeval():
    ma = xdeps.madxutils.MadxEval(vars, math, elements)
    assert ma.eval("+1+2^-2") == 1 + 2**-2
    assert ma.eval("b.c") == vars["b.c"]
    assert ma.eval("sin(3)^2") == math.sin(3) ** 2
    assert ma.eval("el->a*el->b") == elements["el"]["a"] * elements["el"]["b"]


def test_madxeval_attr():
    elements2 = {"el": type("el", (), elements["el"])}
    ma = xdeps.madxutils.MadxEval(vars, math, elements2, get="attr")
    assert ma.eval("el->a*el->b") == elements2["el"].a * elements2["el"].b


def test_madxeval_value_of():
    ma = xdeps.madxutils.MadxEval(vars, math, {})
    assert ma.eval("immediate_eval(42)") == 42
    assert ma.eval("immediate_eval(a)") == vars["a"]
    assert ma.eval("immediate_eval(b.c)") == vars["b.c"]
    assert ma.eval("immediate_eval(a * b.c)") == vars["a"] * vars["b.c"]
