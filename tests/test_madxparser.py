# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xdeps
import math
import pytest

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


@pytest.mark.parametrize('power', ['^', '**'])
@pytest.mark.parametrize('expression,expected', [
    ('-a^2', -0.3**2),
    ('-a^2 + 1', -0.3**2 + 1),
    ('b + (-a^2)', 0.7 + (-0.3**2)),
    ('-2^2', -4.),
    ('1 - a^2', 1 - 0.3**2),
    ('-(a^2)', -0.3**2),
    ('(-a)^2', (-0.3)**2),
    ('-a^3', -0.3**3),
    ('-a*b', -0.3*0.7),
    ('a*-b^2', 0.3*(-0.7**2)),
    ('a/-b^2', 0.3/(-0.7**2)),
    ('+-a^2', -0.3**2),
    ('-sin(a)^2', -math.sin(0.3)**2),
    ('a^-2', 0.3**-2),
    ('a^+2', 0.3**2),
    ('a^(-b^2)', 0.3**(-0.7**2)),
    ('1e-2^2', 1e-4),
    # Chained powers retain MAD-X's left associativity.
    ('a^b^2', (0.3**0.7)**2),
    ('-a^2^2', -((0.3**2)**2)),
    ('a^-2^2', (0.3**-2)**2),
])
def test_madxeval_power_precedence(power, expression, expected):
    evaluator = xdeps.madxutils.MadxEval({'a': 0.3, 'b': 0.7}, math, {})
    assert evaluator.eval(expression.replace('^', power)) == pytest.approx(expected)


@pytest.mark.parametrize('power', ['^', '**'])
def test_madxeval_power_precedence_deferred(power):
    values = {'a': 0.3}
    manager = xdeps.Manager()
    variables = manager.ref(values, 'vars')
    evaluator = xdeps.madxutils.MadxEval(variables, math, {})
    variables['negative_square'] = evaluator.eval(f'-a {power} 2')
    variables['squared_negative'] = evaluator.eval(f'(-a) {power} 2')
    for a in (0.3, -0.4):
        variables['a'] = a
        assert values['negative_square'] == pytest.approx(-a**2)
        assert values['squared_negative'] == pytest.approx(a**2)
