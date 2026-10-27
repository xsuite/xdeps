import numpy as np
import pytest

from xdeps.optimize.jacobian import JacobianSolver


class LinearMerit:
    def __init__(self, target, limits):
        self.target = np.array(target)
        self.limits = np.array(limits)
        self.mask_input = np.ones(len(target), dtype=bool)
        self.mask_output = np.ones(len(target), dtype=bool)
        self.last_point_within_tol = False
        self.evaluated = []

    def __call__(self, x):
        self.evaluated.append(x.copy())
        return x - self.target

    def get_jacobian(self, x, f0=None):
        return np.eye(len(x))

    def _clip_to_max_steps(self, step):
        return step

    def _get_x_limits(self):
        return self.limits


@pytest.mark.parametrize('sign', [1, -1])
def test_limit_roundoff(sign):
    # This step used to land one ulp beyond the bound, poisoning the next
    # step when the limited variable was excluded from the least-squares solve.
    merit = LinearMerit([sign * 24.35903027290258, 2.],
                        [[-5., 5.], [-10., 10.]])
    solver = JacobianSolver(merit)
    solver.x = [sign * 1.0333333333333334, 0.]

    with np.errstate(divide='raise', invalid='raise'):
        solver.step()
        assert solver.x[0] == sign * 5.
        np.testing.assert_array_equal(solver.mask_from_limits, [False, True])
        np.testing.assert_array_equal(merit.evaluated[-1], solver.x)
        np.testing.assert_array_equal(solver._xbest, solver.x)
        solver.step()

    np.testing.assert_array_equal(solver.x, [sign * 5., 2.])
    assert all(np.all(np.isfinite(x)) for x in merit.evaluated)


@pytest.mark.parametrize('sign', [1, -1])
def test_zero_step_just_outside_limit(sign):
    merit = LinearMerit([sign * 10., 2.], [[-5., 5.], [-10., 10.]])
    solver = JacobianSolver(merit)
    solver.x = [np.nextafter(sign * 5., sign * np.inf), 0.]
    solver.mask_from_limits[0] = False

    with np.errstate(divide='raise', invalid='raise'):
        solver.step()

    np.testing.assert_array_equal(solver.x, [sign * 5., 2.])
    np.testing.assert_array_equal(merit.evaluated[-1], solver.x)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('sign', [1, -1])
def test_most_restrictive_limit(reverse, sign):
    order = slice(None, None, -1 if reverse else 1)
    merit = LinearMerit(np.array([sign * 10., sign * 10.])[order],
                        np.array([[-5., 5.], [-2., 2.]])[order])
    solver = JacobianSolver(merit)
    solver.x = [0., 0.]
    solver.step()

    # Only the second variable reaches its bound; snapping the first to 5
    # would change the direction and no longer match the evaluated point.
    np.testing.assert_array_equal(solver.x, [sign * 2., sign * 2.])
    np.testing.assert_array_equal(solver.mask_from_limits,
                                  np.array([True, False])[order])
    np.testing.assert_array_equal(merit.evaluated[-1], solver.x)


@pytest.mark.parametrize('sign', [1, -1])
def test_tied_limits(sign):
    merit = LinearMerit([sign * 10., sign * 20.],
                        [[-5., 5.], [-10., 10.]])
    solver = JacobianSolver(merit)
    solver.x = [0., 0.]
    solver.step()

    np.testing.assert_array_equal(solver.x, [sign * 5., sign * 10.])
    np.testing.assert_array_equal(solver.mask_from_limits, [False, False])
    np.testing.assert_array_equal(merit.evaluated[-1], solver.x)
