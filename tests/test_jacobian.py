import numpy as np

from xdeps.optimize.jacobian import JacobianSolver


def test_lin_solve():
    def ftosolve(x):
        x1, x2 = x
        return np.array([x1 + 3 * x2 - 1, x1 + 1 * x2 - 1])

    x = JacobianSolver(ftosolve).solve(x0=[0.1, 0.2])
    assert np.allclose(ftosolve(x), 0)


def test_lin2_solve():
    def ftosolve(x):
        x1, x2, x3, x4 = x
        return np.array(
            [
                x1 + 2 * x2 - 3 * x3 + 4 * x4 - 1,
                5 * x1 + 2 * x2 + 7 * x3 + 1 * x4 - 2,
                6 * x1 + 2 * x2 + 3 * x3 + 3 * x4 - 3,
            ]
        )

    x0 = [0, 0, 0, 0]
    x = JacobianSolver(ftosolve).solve(x0)
    assert ((x - x0) ** 2).sum() < 0.3
    x0 = [-1, -5, 2, 4]
    x = JacobianSolver(ftosolve).solve(x0)
    assert ((x - x0) ** 2).sum() < 0.3
    assert np.allclose(ftosolve(x), 0)
