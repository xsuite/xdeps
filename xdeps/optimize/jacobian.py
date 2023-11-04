import numpy as np
from ..general import _print

from numpy.linalg import lstsq

class JacobianSolver:

    def __init__(self, func, n_steps_max=20, tol=1e-20, n_bisections=3,
                 min_step=1e-20, error_on_penalty_increase=100,
                 max_rel_penalty_increase=10., verbose=False):
        self.func = func
        self.n_steps_max = n_steps_max
        self.tol = tol
        self.n_bisections = n_bisections
        self.min_step = min_step

        self._penalty_best = None
        self._xbest = None
        self._step = 0
        self.verbose = verbose
        self._penalty_best = 1e200
        self.ncalls = 0
        self.stopped = None
        self._x = None
        self.alpha_last_step = None
        self.error_on_penalty_increase = error_on_penalty_increase
        self.max_rel_penalty_increase = max_rel_penalty_increase

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        # Limit check to be added
        self._x = np.atleast_1d(np.float_(value))
        self.mask_from_limits = np.ones(len(self._x), dtype=bool)

    def step(self, n_steps=1):

        myf = self.func
        self.stopped = None

        for step in range(n_steps):

            self._step += 1

            # test penalty
            y, penalty = self.eval(self.x) # will need to handle mask
            self.penalty_before_last_step = penalty
            self.penalty_after_last_step = penalty
            if penalty < self.tol:
                self.stopped = 'jacobian tolerance met'
                if self.verbose:
                    _print("Jacobian tolerance met")
                break
            if myf.last_point_within_tol:
                self.stopped = 'function tolerance met'
                if self.verbose:
                    _print("Function tolerance met")
                break
            # Equation search
            jac = myf.get_jacobian(self.x, f0=y)
            self._last_jac_x = self.x.copy()
            self._last_jac = jac.copy()

            # lstsq using only the the variables that were not at the limit
            # in the previous step
            xstep = np.zeros(len(self.x))

            assert len(self.func.mask_input) > 0, "At least one vary should be present"
            assert np.any(self.func.mask_input), "At least one vary should be active"

            mask_input = self.func.mask_input & self.mask_from_limits
            mask_output = self.func.mask_output.copy()

            xstep[mask_input] = lstsq(
                jac[mask_output, :][:, mask_input], y[mask_output], rcond=None)[0]  # newton step

            xstep = myf._clip_to_max_steps(xstep)

            self.mask_from_limits[:] = True
            self._last_xstep = xstep.copy()

            alpha = -1
            limits = self.func._get_x_limits()

            while True:  # bisec search
                if (alpha > self.n_bisections
                    and (self.max_rel_penalty_increase is None
                        or newpen < self.max_rel_penalty_increase * penalty)):
                    break
                alpha += 1
                l = 2.0**-alpha
                if self.verbose:
                    _print(f"\n--> step {step} alpha {alpha}\n")

                # Substep
                this_xstep = l * xstep

                # Check limits
                mask_hit_limit = np.zeros(len(self.x), dtype=bool)
                for ii in range(len(self.x)):
                    if self.x[ii] - this_xstep[ii] < limits[ii][0]:
                        this_xstep[ii] = 0
                        mask_hit_limit[ii] = True
                    elif self.x[ii] - this_xstep[ii] > limits[ii][1]:
                        this_xstep[ii] = 0
                        mask_hit_limit[ii] = True

                # Eval function at substep
                y, newpen = self.eval(self.x - this_xstep)

                if self.verbose:
                    _print(f"penalty {penalty} newpen {newpen}")

                self.ncalls += 1

                # Stop if improvement wrt to previous full step
                if newpen < penalty:
                    if self.verbose:
                        print("newpen < penalty")
                    break

            if (self.error_on_penalty_increase
                    and newpen > penalty * self.error_on_penalty_increase):

                # Put things back
                self.eval(self.x)

                raise ValueError(
                    f"penalty increased by more than {self.error_on_penalty_increase} times")

            penalty = newpen
            self.x -= this_xstep  # update solution
            self.mask_from_limits = ~mask_hit_limit
            self.penalty_after_last_step = penalty
            self.alpha_last_step = alpha

            if myf.last_point_within_tol:
                self.stopped = 'function tolerance met'
                if self.verbose:
                    _print("Function tolerance met")
                break

            if self.verbose:
                _print(f"step {step} step_best {self._step_best} {this_xstep}")
            if np.sqrt(np.dot(this_xstep, this_xstep)) < self.min_step:
                if self.verbose:
                    _print("No progress, stopping")
                self.stopped = 'no progress'
                break
        else:
            if self.verbose:
                _print("N. steps reached")

        return self._xbest

    def solve(self, x0):

        self.x = x0.copy()

        self.step(self.n_steps_max)

        return self._xbest

    def eval(self, x):
        y = self.func(x)
        penalty = np.sqrt(np.dot(y, y))
        if self.verbose:
            _print(f"penalty: {penalty}")
        if penalty < self._penalty_best:
            if self._penalty_best - penalty > 1e-20: #????????????
                self._step_best = self._step
            self._penalty_best = penalty
            self._xbest = x.copy()
            if self.verbose:
                _print(f"new best: {self._penalty_best}")
        return y, penalty