import copy
import logging
import re
from warnings import warn

import numpy as np
from scipy.optimize import direct, least_squares, minimize

from ..general import _print
from ..table import Table
from .jacobian import JacobianSolver

log = logging.getLogger(__name__)

LIMITS_DEFAULT = (-1e200, 1e200)
STEP_DEFAULT = 1e-10
TOL_DEFAULT = 1e-10


class Vary:
    def __init__(
        self,
        name,
        container,
        limits=None,
        step=None,
        weight=None,
        max_step=None,
        tag="",
        active=True,
    ):

        if weight is None:
            weight = 1.0

        if limits is not None:
            assert len(limits) == 2, "`limits` must have length 2."
            limits = np.array(limits)

        assert weight > 0, "`weight` must be positive."

        self.name = name
        self.limits = limits
        self.step = step
        self.weight = weight
        self.container = container
        self.max_step = max_step
        self.active = active
        self.tag = tag

        self._complete_limits_and_step_from_defaults()

    def get_value(self):
        val = self.container[self.name]
        if hasattr(val, "_value"):
            return val._value
        else:
            return val

    def _complete_limits_and_step_from_defaults(self):
        if (
            self.limits is None
            and hasattr(self.container, "vary_default")
            and self.name in self.container.vary_default
        ):
            self.limits = self.container.vary_default[self.name]["limits"]

        if (
            self.step is None
            and hasattr(self.container, "vary_default")
            and self.name in self.container.vary_default
        ):
            self.step = self.container.vary_default[self.name]["step"]

    def __repr__(self):
        try:
            lim = f"({self.limits[0]:.4g}, {self.limits[1]:.4g})"
        except (IndexError, TypeError):
            lim = self.limits
        try:
            step = f"{self.step:.4g}"
        except (ValueError, TypeError):
            step = self.step
        try:
            weight = f"{self.weight:.4g}"
        except (ValueError, TypeError):
            weight = self.weight
        return f"Vary(name={self.name!r}, limits={lim}, step={step}, weight={weight})"


class VaryList:
    def __init__(self, vars, container, **kwargs):
        self.vary_objects = [Vary(vv, container, **kwargs) for vv in vars]


class Target:
    def __init__(
        self,
        tar,
        value,
        tol=None,
        weight=None,
        scale=None,
        action=None,
        tag="",
        optimize_log=False,
    ):

        if scale is not None and weight is not None:
            raise ValueError("Cannot specify both `weight` and `scale` for a target.")

        if scale is not None:
            weight = scale

        self.tar = tar
        self.action = action
        self.value = value
        self.tol = tol
        self.weight = weight
        self.active = True
        self.tag = tag
        self.optimize_log = optimize_log

    def __repr__(self):
        out = "Target("
        if callable(self.tar):
            tar_repr = "callable"
        else:
            tar_repr = repr(self.tar)
        try:
            val_str = f"{self.value:.6g}"
        except (ValueError, TypeError):
            val_str = self.value
        try:
            tol_str = f"{self.tol:.4g}"
        except (ValueError, TypeError):
            tol_str = self.tol
        try:
            weight_str = f"{self.weight:.4g}"
        except (ValueError, TypeError):
            weight_str = self.weight
        out += f"{tar_repr}, val={val_str}, tol={tol_str}, weight={weight_str}"
        if self.optimize_log:
            out += ", optimize_log=True"
        out += ")"
        return out

    def copy(self):
        return copy.copy(self)

    @property
    def scale(self):
        return self.weight

    @scale.setter
    def scale(self, value):
        self.weight = value

    def eval(self, data):
        res = data[self.action]
        if callable(self.tar):
            return self.tar(res)
        else:
            return res[self.tar]

    def runeval(self):
        return self.eval({self.action: self.action.run()})


class TargetList:
    def __init__(self, tars, **kwargs):
        self.targets = [Target(tt, **kwargs) for tt in tars]


class Action:

    _target_class = Target # so that it can be overridden by subclasses

    def prepare(self):
        pass

    def run(self):
        return dict()

    def target(self, tar, value, **kwargs):
        return self._target_class(tar, value, action=self, **kwargs)


class MeritFunctionForMatch:

    def __init__(
        self,
        vary,
        targets,
        actions,
        return_scalar,
        call_counter,
        verbose,
        tw_kwargs,
        steps_for_jacobian,
        check_limits,
        show_call_counter=True,
    ):

        self.vary = vary
        self.targets = targets
        self.actions = actions
        self.return_scalar = return_scalar
        self.call_counter = call_counter
        self.verbose = verbose
        self.tw_kwargs = tw_kwargs
        self.steps_for_jacobian = steps_for_jacobian
        self.found_point_within_tol = False
        self.zero_if_met = False
        self.show_call_counter = show_call_counter
        self.check_limits = check_limits

    def _x_to_knobs(self, x):
        knob_values = np.array(x).copy()
        for ii, vv in enumerate(self.vary):
            if vv.weight is not None:
                knob_values[ii] *= vv.weight
        return knob_values

    def _knobs_to_x(self, knob_values):
        x = np.array(knob_values, dtype=np.float64).copy()
        for ii, vv in enumerate(self.vary):
            if vv.weight is not None:
                x[ii] /= vv.weight
        return x

    def _extract_knob_values(self):
        return [ vv.get_value() for vv in self.vary]

    def _get_x(self):
        return self._knobs_to_x(self._extract_knob_values())

    def _set_x(self, x):
        self(x)  # this sets the knobs

    def _get_x_limits(self):
        knob_limits = []
        for vv in self.vary:
            if vv.limits is None:
                knob_limits.append(LIMITS_DEFAULT)
            else:
                knob_limits.append(vv.limits)
        knob_limits = np.array(knob_limits)
        x_lim_low = self._knobs_to_x(np.atleast_1d(np.squeeze(knob_limits[:, 0])))
        x_lim_high = self._knobs_to_x(np.atleast_1d(np.squeeze(knob_limits[:, 1])))
        x_limits = np.array([[hh, ll] for hh, ll in zip(x_lim_low, x_lim_high)])
        return x_limits

    def get_merit_function(self, check_limits=True, return_scalar=None, rescale_x=None):
        return MeritFuctionView(
            self, check_limits=check_limits, return_scalar=return_scalar, rescale_x=rescale_x
        )

    def __call__(self, x=None, check_limits=None, return_scalar=None, zero_if_met=None):

        if zero_if_met is None:
            zero_if_met = self.zero_if_met

        if x is None:
            knob_values = self._extract_knob_values()
        else:
            knob_values = self._x_to_knobs(x)

        if check_limits is None:
            check_limits = self.check_limits

        # Set knobs
        for vv, val in zip(self.vary, knob_values):
            if vv.active:
                if check_limits and vv.limits is not None and vv.limits[0] is not None:
                    if val < vv.limits[0] - 1e-3 * (vv.limits[1] - vv.limits[0]):
                        raise ValueError(f"Knob {vv.name} is below lower limit.")
                if check_limits and vv.limits is not None and vv.limits[1] is not None:
                    if val > vv.limits[1] + 1e-3 * (vv.limits[1] - vv.limits[0]):
                        raise ValueError(f"Knob {vv.name} is above upper limit.")
                vv.container[vv.name] = val

        # if self.verbose:
        #     _print(f'x = {knob_values}')

        # Run actions
        res_data = {}
        failed = False
        for aa in self.actions:
            res_data[aa] = aa.run()
            if isinstance(res_data[aa], str) and res_data[aa] == "failed":
                failed = True
                break

        if failed:
            err_values = np.full(len(self.targets), 1e100)
        else:
            res_values = []
            target_values = []
            for tt in self.targets:
                res_values.append(tt.eval(res_data))
                if hasattr(tt.value, "_value"):
                    target_values.append(tt.value._value)
                else:
                    target_values.append(tt.value)

            self._last_data = res_data  # for debugging

            res_values = np.array(res_values)
            target_values = np.array(target_values)

            transformed_res_values = res_values * 0
            for ii, tt in enumerate(self.targets):
                if hasattr(tt, "transform"):
                    transformed_res_values[ii] = tt.transform(res_values[ii])
                else:
                    transformed_res_values[ii] = res_values[ii]

            err_values = transformed_res_values - target_values

            # if self.verbose:
            #     _print(f'   f(x) = {res_values}')

            tols = 0 * err_values
            for ii, tt in enumerate(self.targets):
                tols[ii] = tt.tol

            # if self.verbose:
            #     _print(f'   err/tols = {err_values/tols}')

            targets_within_tol = np.abs(err_values) < tols
            self.last_targets_within_tol = targets_within_tol
            self.last_res_values = res_values
            self.last_residue_values = err_values.copy()

            err_values[~self.mask_output] = 0


            if np.all(targets_within_tol | (~self.mask_output)):
                if zero_if_met:
                    err_values *= 0
                self.last_point_within_tol = True
                self.found_point_within_tol = True
                if self.verbose:
                    _print("Found point within tolerance!")
            else:
                self.last_point_within_tol = False

            # handle optimize logarithm
            err_values = err_values.copy()
            for ii, tt in enumerate(self.targets):
                if self.mask_output[ii] and tt.optimize_log:
                    assert (
                        res_values[ii] > 0
                    ), "Cannot use optimize_log with negative values"
                    assert tt.value > 0, "Cannot use optimize_log with negative targets"
                    if hasattr(tt, "transform"):
                        assert (
                            tt.transform(res_values[ii]) == res_values[ii]
                        ), "Cannot use optimize_log with transformed values"
                    vvv = np.log10(res_values[ii])
                    vvv_tar = np.log10(tt.value)
                    err_values[ii] = vvv - vvv_tar

            for ii, tt in enumerate(self.targets):
                if tt.weight is not None:
                    err_values[ii] *= tt.weight

        if return_scalar is None:
            return_scalar = self.return_scalar

        out_scalar = np.sum(err_values * err_values)

        if return_scalar:
            out = out_scalar
        else:
            out = np.array(err_values)

        if self.show_call_counter:
            _print(
                f"Matching: model call n. {self.call_counter} "
                + f"penalty = {np.sqrt(out_scalar):.4e}"
                + "              ",
                end="\r",
                flush=True,
            )
        self.call_counter += 1

        return out

    def get_jacobian(self, x, f0=None):
        if hasattr(self, "_force_jacobian"):
            return self._force_jacobian
        x = np.array(x).copy()
        steps = self._knobs_to_x(self.steps_for_jacobian)
        assert len(x) == len(steps)
        if f0 is None:
            f0 = self(x)
        if np.isscalar(f0):
            jac = np.zeros((1, len(x)))
        else:
            jac = np.zeros((len(f0), len(x)))
        mask_input = self.mask_input
        for ii in range(len(x)):
            if not mask_input[ii]:
                continue
            x[ii] += steps[ii]
            jac[:, ii] = (self(x, check_limits=False) - f0) / steps[ii]
            x[ii] -= steps[ii]

        self._last_jac = jac
        self._set_x(x)
        return jac

    def _clip_to_max_steps(self, x_step):
        max_steps = np.array([vv.max_step for vv in self.vary])
        out = x_step.copy()
        for ii in range(len(x_step)):
            if max_steps[ii] is None:
                continue
            if np.abs(x_step[ii]) > max_steps[ii]:
                out *= max_steps[ii] / np.abs(out[ii])
        return out

    @property
    def mask_input(self):
        mask = []
        for vv in self.vary:
            if hasattr(vv, "active"):
                mask.append(vv.active)
            else:
                mask.append(True)
        return np.array(mask)

    @property
    def mask_output(self):
        mask = []
        for tt in self.targets:
            if hasattr(tt, "active"):
                mask.append(tt.active)
            else:
                mask.append(True)
        return np.array(mask)


class MeritFuctionView:

    def __init__(self, merit_function, check_limits=True, return_scalar=None, rescale_x=None,
                 zero_if_met=False):

        self.merit_function = merit_function
        self.check_limits = check_limits
        self.return_scalar = return_scalar
        self.rescale_x = rescale_x
        self.zero_if_met = zero_if_met

    def __call__(self, x):
        x = np.array(x)
        if self.rescale_x:
            x = self._scaled_to_native(x)

        return self.merit_function(
            x, check_limits=self.check_limits, return_scalar=self.return_scalar,
            zero_if_met=self.zero_if_met
        )

    def get_jacobian(self, x):
        x = np.array(x)

        if self.rescale_x:
            x = self._scaled_to_native(x)

        jac_native = self.merit_function.get_jacobian(x)

        if self.rescale_x:
            zzz = 0 * x
            ttt = 1. + 0 * x
            dx_native_dx_scaled = self._scaled_to_native(ttt) - self._scaled_to_native(zzz)
            jac = jac_native.copy()
            for jj in range(jac_native.shape[1]):
                jac[:, jj] *= dx_native_dx_scaled[jj]
        else:
            jac = jac_native

        if self.return_scalar:
            f0 = self.merit_function(x, check_limits=self.check_limits,
                                     zero_if_met=self.zero_if_met)
            return 2 * np.dot(f0, jac)
        else:
            return jac

    def _scaled_to_native(self, x):
        """
        From scaled to native space
        """
        bounds = self.merit_function._get_x_limits()
        self._check_for_scalability(bounds)
        scaled_range = self.rescale_x

        transformed_x = bounds[:,0] + (
            ((x - scaled_range[0]) * (bounds[:,1] - bounds[:,0]))
            / (scaled_range[1] - scaled_range[0]))
        return transformed_x

    def _scaled_from_native(self, x):
        """
        From native to scaled space
        """
        bounds = self.merit_function._get_x_limits()
        self._check_for_scalability(bounds)
        scaled_range = self.rescale_x

        transformed_x = scaled_range[0] + (
            ((x - bounds[:,0]) * (scaled_range[1] - scaled_range[0]))
            / (bounds[:,1] - bounds[:,0]))
        return transformed_x

    def _check_for_scalability(self, bounds):
        if self.rescale_x is not None and not isinstance(self.rescale_x, tuple):
            raise TypeError("Normalized Space must be a tuple")
        elif self.rescale_x[0] < -1e20 or self.rescale_x[1] > 1e20 or self.rescale_x[1] - self.rescale_x[0] > 1e20:
            raise ValueError("Normalized Interval is too large")
        elif np.any(bounds[:,0] < -1e20) or np.any(bounds[:,1] > 1e20) or np.any(bounds[:,1] - bounds[:,0] > 1e20):
            raise ValueError("Bounds are not given or too large to normalize")

    def get_x_limits(self):
        bounds = self.merit_function._get_x_limits()
        if self.rescale_x:
            self._check_for_scalability(bounds)
            for ii in range(len(bounds)):
                bounds[ii, :] = np.array([self.rescale_x[0], self.rescale_x[1]])
        return bounds

    def get_x(self):
        x = self.merit_function._get_x()
        if self.rescale_x:
            x = self._scaled_from_native(x)
        return x

    def set_x(self, x):
        x = np.array(x)
        if self.rescale_x:
            x = self._scaled_to_native(x)
        self.merit_function._set_x(x)

class Optimize:

    def __init__(
        self,
        vary,
        targets,
        restore_if_fail=True,
        solver=None,
        verbose=None,
        assert_within_tol=True,
        n_steps_max=20,
        solver_options={},
        show_call_counter=True,
        check_limits=True,
        name="",
        **kwargs,
    ):
        """
        Numerical optimizer for matching.

        The optimizer changes the variables described by ``vary`` until the
        quantities described by ``targets`` are within their tolerances. In
        Xtrack this object is returned by :meth:`xtrack.Line.match` when
        ``solve=False``. It can also be used directly with xdeps
        :class:`Vary`, :class:`Target` and :class:`Action` objects.

        Parameters
        ----------
        vary : list of Vary
            List of knobs to vary.
        targets : list of Target
            List of targets to match.
        restore_if_fail : bool, optional
            If True, restore the initial knob values if the optimization fails.
            Defaults to True.
        solver : str, optional
            Solver to use. Can be 'fsolve', 'bfgs', or 'jacobian'. Defaults to
            'jacobian'.
        verbose : bool, optional
            If True, print information during the optimization. Defaults to False.
        assert_within_tol : bool, optional
            If True, raise an error if the optimization fails. Defaults to True.
        n_steps_max : int, optional
            Maximum number of steps to take. Defaults to 20.
        solver_options : dict, optional
            Options to pass to the solver. Defaults to {}.
        show_call_counter : bool, optional
            If True, show the number of merit-function calls while optimizing.
            Defaults to True.
        check_limits : bool, optional
            If True, reject knob values outside their limits. If False, values
            outside limits are clipped before a Jacobian step. Defaults to True.
        name : str, optional
            Name printed in optimization progress messages.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)

            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON          False            -1            -1          0 0, val=0, tol=1e-12, weight=1
            # 1  ON          False             2             2          0 1, val=0, tol=1e-12, weight=1

            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             0        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200             0        1e+200             0         1e-06             1

            opt.solve(verbose=False)

            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1

            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             1        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200            -2        1e+200             0         1e-06             1

            opt.log().show()
            # iteration                   penalty alpha tag tol_met target_active hit_limits vary_active ...
            # 0                           2.23607    -1     nn      yy            nn         yy
            # 1                           2.23607    -1     nn      yy            nn         yy
            # 2                       2.81031e-10     0     nn      yy            nn         yy
            # 3                                 0     0     yy      yy            nn         yy

        """
        self.name = name

        if isinstance(vary, (str, Vary)):
            vary = [vary]

        input_vary = vary
        vary = []
        for ii, rr in enumerate(input_vary):
            if isinstance(rr, Vary):
                vary.append(rr)
            elif isinstance(rr, str):
                vary.append(Vary(rr))
            elif isinstance(rr, (list, tuple)):
                raise ValueError("Not supported")
            elif isinstance(rr, VaryList):
                vary += rr.vary_objects
            else:
                raise ValueError(f"Invalid vary setting {rr}")

        input_targets = targets
        targets = []
        for ii, tt in enumerate(input_targets):
            if isinstance(tt, Target):
                targets.append(tt)
            elif isinstance(tt, (list, tuple)):
                targets.append(Target(*tt))
            elif isinstance(tt, TargetList):
                targets += tt.targets
            else:
                raise ValueError(f"Invalid target element {tt}")

        actions = []
        for tt in targets:
            if tt.weight is None:
                tt.weight = 1.0
            if tt.weight <= 0:
                raise ValueError("`weight` must be positive.")

            if tt.action not in actions:
                actions.append(tt.action)

        for aa in actions:
            aa.prepare()

        data0 = {}
        for aa in actions:
            data0[aa] = aa.run()
            if isinstance(data0[aa], str):
                assert (
                    data0[aa] != "failed"
                ), f"Action {aa} failed to compute initial data."

        for tt in targets:
            if tt.value == "preserve":
                tt.value = tt.eval(data0)

        if solver is None:
            solver = "jacobian"

        if verbose:
            _print(f"Using solver {solver}")

        steps = []
        for vv in vary:
            if vv.step is None:
                steps.append(STEP_DEFAULT)
            else:
                steps.append(vv.step)

        assert solver in ["fsolve", "bfgs", "jacobian"], f"Invalid solver {solver}."

        return_scalar = {"fsolve": False, "bfgs": True, "jacobian": False}[solver]

        _err = MeritFunctionForMatch(
            vary=vary,
            targets=targets,
            actions=actions,
            return_scalar=return_scalar,
            call_counter=0,
            verbose=verbose,
            tw_kwargs=kwargs,
            steps_for_jacobian=steps,
            check_limits=check_limits,
            show_call_counter=False,
        )

        if solver == "jacobian":
            self.solver = JacobianSolver(func=_err, verbose=verbose, **solver_options)
        else:
            raise NotImplementedError(f"Solver {solver} not implemented.")

        self.assert_within_tol = assert_within_tol
        self.restore_if_fail = restore_if_fail
        self.n_steps_max = n_steps_max
        self._log = dict(
            penalty=[],
            hit_limits=[],
            alpha=[],
            tol_met=[],
            knobs=[],
            targets=[],
            vary_active=[],
            target_active=[],
            tag=[],
            last_jac_rank=[],
            last_jac_cond=[],
        )

        if not self.check_limits:
            self.add_point_to_log()
            self._clip_to_limits()

        self.add_point_to_log()

        self._err.show_call_counter = show_call_counter

    @classmethod
    def from_callable(cls, function, x0, tar, steps=None, tols=None,
                      limits=None,
                      show_call_counter=True):

        """
        Build an optimizer from a Python callable.

        The callable is evaluated as ``function(x)`` and must return a sequence
        with the same length as ``tar``. The input vector ``x`` is optimized so
        that each returned value reaches the corresponding target value within
        the corresponding tolerance.

        Parameters
        ----------
        function : callable
            Callable receiving the current input vector and returning the
            quantities to match.
        x0 : array-like
            Initial input vector.
        tar : array-like
            Target values for the callable output.
        steps : array-like, optional
            Finite-difference steps used to build the Jacobian. If not provided,
            ``STEP_DEFAULT`` is used for all variables.
        tols : array-like, optional
            Tolerance for each target. If not provided, ``TOL_DEFAULT`` is used
            for all targets.
        limits : array-like, optional
            Lower and upper limits for each input variable, as
            ``[(lower, upper), ...]``.
        show_call_counter : bool, optional
            If True, show the number of merit-function calls while optimizing.

        Returns
        -------
        Optimize
            Optimizer configured for the provided callable.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)

            opt.solve(verbose=False)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             1        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200            -2        1e+200             0         1e-06             1
            opt.log().show()
            # iteration                   penalty alpha tag tol_met target_active hit_limits vary_active ...
            # 0                           2.23607    -1     nn      yy            nn         yy
            # 1                           2.23607    -1     nn      yy            nn         yy
            # 2                       2.81031e-10     0     nn      yy            nn         yy
            # 3                                 0     0     yy      yy            nn         yy
        """

        x0 = np.array(x0)

        if steps is None:
            steps = np.ones(len(x0)) * STEP_DEFAULT
        if tols is None:
            tols = np.ones(len(tar)) * TOL_DEFAULT
        if limits is None:
            limits = [[-1e200, 1e200]] * len(x0)

        x = x0.copy()
        vary = [Vary(ii, container=x, step=steps[ii], limits=limits[ii])
                for ii in range(len(x))]
        targets=ActionCall(function, vary).get_targets(tar)

        for ttt, tttol in zip(targets, tols):
            ttt.tol = tttol

        opt = Optimize(
            vary=vary,
            targets=targets,
            show_call_counter=show_call_counter,
        )

        for ii, tt in enumerate(opt.targets):
            tt.tol = tols[ii]

        return opt

    def run_jacobian(self, n_steps=1):
        """
        Perform optimization steps using the Jacobian solver.

        This is equivalent to calling :meth:`step` with the provided number of
        steps.

        Parameters
        ----------
        n_steps : int, optional
            Number of steps to perform. Defaults to 1.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], limits=[[-5, 5], [-5, 5]],
                show_call_counter=False)
            opt.verbose = False

            opt.run_jacobian(n_steps=3)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name lower_limit   current_val upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0          -5             1           5             0         1e-06             1
            # 1  ON        OK     1          -5            -2           5             0         1e-06             1
        """
        self.step(n_steps)


    def run_ls_trf(self, n_steps=1000, ftol=1e-12, gtol=None, xtol=1e-12, verbose=None):
        """
        Perform the least squares optimization using the Trust Region Reflective algorithm.

        This method uses :func:`scipy.optimize.least_squares` with
        ``method="trf"``.

        Parameters
        ----------
        n_steps : int, optional
            Maximum number of steps to perform. Defaults to 1000.
        ftol : float, optional
            Tolerance for the cost function. Defaults to 1e-12.
        gtol : float, optional
            Tolerance for the gradient. Defaults to None.
        xtol : float, optional
            Tolerance for the step. Defaults to 1e-12.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], limits=[[-5, 5], [-5, 5]],
                show_call_counter=False)

            opt.run_ls_trf(n_steps=200)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name lower_limit   current_val upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0          -5             1           5             0         1e-06             1
            # 1  ON        OK     1          -5            -2           5             0         1e-06             1
        """
        merit_function = self.get_merit_function(return_scalar=False, check_limits=False)
        bounds = merit_function.get_x_limits()
        res = least_squares(merit_function, merit_function.get_x(), method="trf",
                        bounds=bounds.T, ftol=ftol, gtol=gtol, xtol=xtol,
                        jac=merit_function.get_jacobian, max_nfev=n_steps,
                        verbose=(verbose or 0))
        merit_function.set_x(res.x)
        self.tag('trf')

    def run_ls_dogbox(self, n_steps=1000, ftol=1e-12, gtol=None, xtol=1e-12, verbose=None):
        """
        Perform the least squares optimization using the Dogbox algorithm.

        This method uses :func:`scipy.optimize.least_squares` with
        ``method="dogbox"``.

        Parameters
        ----------
        n_steps : int, optional
            Maximum number of steps to perform. Defaults to 1000.
        ftol : float, optional
            Tolerance for the cost function. Defaults to 1e-12.
        gtol : float, optional
            Tolerance for the gradient. Defaults to None.
        xtol : float, optional
            Tolerance for the step. Defaults to 1e-12.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], limits=[[-5, 5], [-5, 5]],
                show_call_counter=False)

            opt.run_ls_dogbox(n_steps=200)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name lower_limit   current_val upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0          -5             1           5             0         1e-06             1
            # 1  ON        OK     1          -5            -2           5             0         1e-06             1
        """

        merit_function = self.get_merit_function(return_scalar=False, check_limits=False)
        bounds = merit_function.get_x_limits()
        res = least_squares(merit_function, merit_function.get_x(), method="dogbox",
                        bounds=bounds.T, ftol=ftol, gtol=gtol, xtol=xtol,
                        jac=merit_function.get_jacobian, max_nfev=n_steps,
                        verbose=(verbose or 0))
        merit_function.set_x(res.x)
        self.tag('dogbox')

    def run_l_bfgs_b(self, n_steps=1000, ftol=1e-24, gtol=1e-24, disp=False):
        """
        Perform the optimization using the L-BFGS-B algorithm.

        This method uses :func:`scipy.optimize.minimize` with
        ``method="L-BFGS-B"``.

        Parameters
        ----------
        n_steps : int, optional
            Maximum number of steps to perform. Defaults to 1000.
        ftol : float, optional
            Tolerance for the cost function. Defaults to 1e-24.
        gtol : float, optional
            Tolerance for the gradient. Defaults to 1e-24.
        disp : bool, optional
            If True, display convergence messages. Defaults to False.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], limits=[[-5, 5], [-5, 5]],
                show_call_counter=False)

            opt.run_l_bfgs_b(n_steps=200)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name lower_limit   current_val upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0          -5             1           5             0         1e-06             1
            # 1  ON        OK     1          -5            -2           5             0         1e-06             1
        """

        merit_function = self.get_merit_function(return_scalar=True, check_limits=False)
        bounds = merit_function.get_x_limits()
        res = minimize(merit_function, merit_function.get_x(), method='L-BFGS-B',
                        jac=merit_function.get_jacobian,
                        bounds=bounds,
                        options=dict(
                        maxiter=n_steps,
                        ftol=ftol,
                        gtol=gtol,
                        disp=disp,
                        ))
        merit_function.set_x(res.x)
        self.tag('l-bfgs-b')

    def run_bfgs(self, n_steps=1000, xrtol=1e-10, gtol=1e-18, disp=False):
        """
        Perform the optimization using the BFGS algorithm.

        This method uses :func:`scipy.optimize.minimize` with
        ``method="BFGS"``.

        Parameters
        ----------
        n_steps : int, optional
            Maximum number of steps to perform. Defaults to 1000.
        xrtol : float, optional
            Relative tolerance for the step. Defaults to 1e-10.
        gtol : float, optional
            Tolerance for the gradient. Defaults to 1e-18.
        disp : bool, optional
            If True, display convergence messages. Defaults to False.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-11, 1e-11], limits=[[-5, 5], [-5, 5]],
                show_call_counter=False)

            opt.run_bfgs(n_steps=200)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True  -1.41642e-12  -1.41642e-12          0 0, val=0, tol=1e-11, weight=1
            # 1  ON           True   3.23319e-12   3.23319e-12          0 1, val=0, tol=1e-11, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name lower_limit   current_val upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0          -5             1           5             0         1e-06             1
            # 1  ON        OK     1          -5            -2           5             0         1e-06             1
        """

        merit_function = self.get_merit_function(return_scalar=True, check_limits=False)
        res = minimize(merit_function, merit_function.get_x(), method='BFGS',
                        jac=merit_function.get_jacobian,
                        options=dict(
                        maxiter=n_steps,
                        xrtol=xrtol,
                        gtol=gtol,
                        disp=disp,
                        ))
        merit_function.set_x(res.x)
        self.tag('bfgs')

    def run_nelder_mead(self, n_steps=1000, fatol=1e-11, xatol=1e100,
                            adaptive=True, disp=False, verbose=None):
        """
        Perform the optimization using the Nelder-Mead Simplex algorithm.

        This method uses :func:`scipy.optimize.minimize` with
        ``method="Nelder-Mead"``.

        Parameters
        ----------
        n_steps : int, optional
            Maximum number of steps to perform. Defaults to 1000.
        fatol : float, optional
            Absolute tolerance for the cost function. Defaults to 1e-11.
        xatol : float, optional
            Absolute tolerance for the step. Defaults to 1e100 (no effect).
        adaptive : bool, optional
            If True, adapt algorithm parameters to dimensionality of problem. Defaults to True.
        disp : bool, optional
            If True, display convergence messages. Defaults to False.
        verbose : bool or int, optional
            Verbosity used for xdeps progress messages.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-5, 1e-5], limits=[[-5, 5], [-5, 5]],
                show_call_counter=False)

            opt.run_nelder_mead(n_steps=500, verbose=False)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True  -8.87276e-06  -8.87276e-06          0 0, val=0, tol=1e-05, weight=1
            # 1  ON           True   2.76274e-06   2.76274e-06          0 1, val=0, tol=1e-05, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name lower_limit   current_val upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0          -5      0.999991           5             0         1e-06             1
            # 1  ON        OK     1          -5            -2           5             0         1e-06             1
        """

        self._add_starting_point_to_log_and_print(verbose=verbose)

        fff = self.get_merit_function(return_scalar=True)
        fff.zero_if_met = True
        bounds = fff.get_x_limits()
        res = minimize(fff, fff.get_x(), method='Nelder-Mead',
                    bounds=bounds,
                    options=dict(
                        maxiter=n_steps,
                        fatol=fatol,
                        xatol=xatol,
                        adaptive=adaptive,
                        disp=disp,
                    ))
        self._last_symplex_res = res
        fff.set_x(res.x)
        self.tag('simplex')

        self._print_end(verbose)

    def run_simplex(self, n_steps=1000, fatol=1e-11, xatol=1e100,
                             adaptive=True, disp=False, verbose=None):
        """
        Run the Nelder-Mead Simplex optimizer.

        This is a convenience wrapper around :meth:`run_nelder_mead`, which uses
        :func:`scipy.optimize.minimize` with ``method="Nelder-Mead"``.

        Parameters
        ----------
        n_steps : int, optional
            Maximum number of steps to perform. Defaults to 1000.
        fatol : float, optional
            Absolute tolerance for the cost function. Defaults to 1e-11.
        xatol : float, optional
            Absolute tolerance for the step. Defaults to 1e100 (no effect).
        adaptive : bool, optional
            If True, adapt algorithm parameters to dimensionality of problem.
            Defaults to True.
        disp : bool, optional
            If True, display convergence messages. Defaults to False.
        verbose : bool or int, optional
            Verbosity used for xdeps progress messages.
        """
        self.run_nelder_mead(n_steps=n_steps, fatol=fatol, xatol=xatol,
                                adaptive=adaptive, disp=disp, verbose=verbose)

    def run_direct(self, n_steps=1000, verbose=None, **kwargs):
        """
        Perform the optimization using the DIRECT algorithm.

        This method uses :func:`scipy.optimize.direct`.

        Parameters
        ----------
        n_steps : int, optional
            Maximum number of steps to perform. Defaults to 1000.
        verbose : bool or int, optional
            Verbosity used for xdeps progress messages.
        **kwargs
            Additional keyword arguments passed to
            :func:`scipy.optimize.direct`.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], limits=[[-5, 5], [-5, 5]],
                show_call_counter=False)

            opt.run_direct(n_steps=1000, verbose=False)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True   -4.4631e-14   -4.4631e-14          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True   8.74856e-14   8.74856e-14          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name lower_limit   current_val upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0          -5             1           5             0         1e-06             1
            # 1  ON        OK     1          -5            -2           5             0         1e-06             1
        """

        kwargs['f_min_rtol'] = kwargs.get('f_min_rtol', 1e-30)
        kwargs['vol_tol'] = kwargs.get('vol_tol', 1e-30)
        kwargs['len_tol'] = kwargs.get('len_tol', 1e-30)

        self._add_starting_point_to_log_and_print(verbose)
        i_log_start = len(self._log["penalty"]) - 1

        merit_function = self.get_merit_function(return_scalar=True)
        bounds = merit_function.get_x_limits()
        res = direct(merit_function, bounds=list(bounds), maxiter=n_steps,
                     maxfun=len(bounds) * n_steps,
                    **kwargs)
        self._last_direct_res = res
        merit_function.set_x(res.x)
        self.tag('direct')

        ll = self.log()
        if ll['penalty'][-1] > ll['penalty'][i_log_start]:
            self.reload(iteration=i_log_start)

        self._print_end(verbose)

    def _step_simplex(self, n_steps=1, fatol=1e-11, xatol=1e-11,
                                adaptive=True, disp=False):
        # for backwards compatibility
        self.run_simplex(n_steps=n_steps, fatol=fatol, xatol=xatol,
                            adaptive=adaptive, disp=disp)

    def step(
        self,
        n_steps=1,
        take_best=True,
        enable_target=None,
        enable_vary=None,
        enable_vary_name=None,
        disable_target=None,
        disable_vary=None,
        disable_vary_name=None,
        rcond=None,
        sing_val_cutoff=None,
        verbose=None,
        broyden=False,
    ):
        """
        Perform one or more optimization steps.

        A step updates the active variables using the active targets and records
        the result in the optimization log. Variables and targets can be enabled
        or disabled only for the duration of these steps using the corresponding
        keyword arguments.

        Parameters
        ----------
        n_steps : int, optional
            Number of steps to perform. Defaults to 1.
        take_best : bool, optional
            If True and the final point is not within tolerance, reload the best
            point found during this call. Defaults to True.
        enable_target: list of int or strings, optional
            For the performed steps, enable target with corresponding id or tag
        enable_vary: list of int or strings, optional
            For the performed steps, enable variables with corresponding id or tag
        enable_vary_name: list of str, optional
            For the performed steps, enable variables with corresponding name
        disable_target: list of int or strings, optional
            For the performed steps, disable target with corresponding id or tag
        disable_vary: list of int or strings, optional
            For the performed steps, disable variables with corresponding id or tag
        disable_vary_name: list of str, optional
            For the performed steps, disable variables with corresponding name
        rcond : float, optional
            Cutoff passed to the Jacobian linear solve.
        sing_val_cutoff : float, optional
            Singular-value cutoff passed to the Jacobian linear solve.
        verbose : bool or int, optional
            Verbosity for progress messages. If not provided, ``self.verbose``
            is used.
        broyden : bool or int, optional
            If True, update the Jacobian with Broyden updates between full
            finite-difference Jacobian evaluations. If an integer is provided,
            a full Jacobian is recomputed every ``broyden + 1`` steps.

        Returns
        -------
        Optimize
            The optimizer itself.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)

            opt.step(3, verbose=False)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             1        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200            -2        1e+200             0         1e-06             1
            opt.log().show()
            # iteration                   penalty alpha tag tol_met target_active hit_limits vary_active ...
            # 0                           2.23607    -1     nn      yy            nn         yy
            # 1                           2.23607    -1     nn      yy            nn         yy
            # 2                       2.81031e-10     0     nn      yy            nn         yy
            # 3                                 0     0     yy      yy            nn         yy

            opt.step(2, disable_vary=0, verbose=False)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             1        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200            -2        1e+200             0         1e-06             1
            opt.log().show()
            # iteration                   penalty alpha tag           tol_met target_active hit_limits ...
            # 0                           2.23607    -1               nn      yy            nn
            # 1                          0.447214    -1               nn      yy            nn
            # 2                       5.62063e-11     0               nn      yy            nn
            # 3                                 0     0               yy      yy            nn
            # 4                                 0    -1 Homotopy it 0 yy      yy            nn
            # 5                          0.447214    -1               nn      yy            nn
            # 3                                 0     0     yy      yy            nn         yy
            # 4                                 0    -1     yy      yy            nn         ny
            # 5                                 0     0     yy      yy            nn         ny
        """
        if not self.check_limits:
            self._clip_to_limits()

        if enable_target is not None:
            self.enable(target=enable_target)

        if enable_vary is not None:
            self.enable(vary=enable_vary)

        if disable_target is not None:
            self.disable(target=disable_target)

        if disable_vary is not None:
            self.disable(vary=disable_vary)

        if disable_vary_name is not None:
            self.disable(vary_name=disable_vary_name)

        if enable_vary_name is not None:
            self.enable(vary_name=enable_vary_name)

        if verbose is None:
            verbose = self.verbose

        self._add_starting_point_to_log_and_print(verbose)
        i_log_start = len(self._log["penalty"]) - 1

        for i_step in range(n_steps):
            knobs_before = self._extract_knob_values()

            x = self._err._knobs_to_x(knobs_before)
            mskinp = self._err.mask_input
            if self.solver.x is None or not np.allclose(
                x[mskinp], self.solver.x[mskinp], rtol=0, atol=1e-12
            ):
                self.solver.x = x  # this resets solver.mask_from_limits

            # self.solver.x = self._err._knobs_to_x(self._extract_knob_values())

            this_broyden = False
            if broyden:
                if isinstance(broyden, bool):
                    this_broyden = True
                else:
                    assert isinstance(broyden, int)
                    this_broyden = True
                    if i_step % (broyden + 1) == 0:
                        this_broyden = False

            self.solver.step(
                rcond=rcond, sing_val_cutoff=sing_val_cutoff, broyden=this_broyden)
            self._log["penalty"].append(self.solver.penalty_after_last_step)
            if hasattr(self.solver, "_last_jac_svd"):
                self._log["last_jac_rank"].append(self.solver._last_jac_svd.rank)
                self._log["last_jac_cond"].append(self.solver._last_jac_svd.cond)
            else:
                self._log["last_jac_rank"].append(-1)
                self._log["last_jac_cond"].append(-1)

            self.set_knobs_from_x(self.solver.x)

            knobs_after = self._extract_knob_values()
            self._log["knobs"].append(knobs_after)
            self._log["targets"].append(self._err.last_res_values)
            self._log["hit_limits"].append(
                _bool_array_to_string(~self.solver.mask_from_limits)
            )
            self._log["vary_active"].append(_bool_array_to_string(self._err.mask_input))
            self._log["target_active"].append(
                _bool_array_to_string(self._err.mask_output)
            )
            self._log["tol_met"].append(
                _bool_array_to_string(self._err.last_targets_within_tol)
            )
            self._log["alpha"].append(self.solver.alpha_last_step)
            self._log["tag"].append("")

            if self._err.last_point_within_tol:
                break

        if take_best and not self._err.last_point_within_tol:
            penalty_step = self._log["penalty"][i_log_start:]
            i_best = np.argmin(penalty_step)
            if i_best != len(penalty_step) - 1:
                self.reload(iteration=i_best + i_log_start)
                self._log["tag"][-1] = "take_best"

        self._print_end(verbose)

        if enable_target is not None:
            self.disable(target=enable_target)

        if enable_vary is not None:
            self.disable(vary=enable_vary)

        if disable_target is not None:
            self.enable(target=disable_target)

        if disable_vary is not None:
            self.enable(vary=disable_vary)

        if disable_vary_name is not None:
            self.enable(vary_name=disable_vary_name)

        if enable_vary_name is not None:
            self.disable(vary_name=enable_vary_name)

        return self

    def solve(self, n_steps=None, verbose=None, take_best=True, rcond=None, sing_val_cutoff=None, broyden=False):
        """
        Run the optimizer until convergence or until the step limit is reached.

        The optimizer performs up to ``n_steps`` Jacobian steps. If ``n_steps``
        is not provided, ``self.n_steps_max`` is used. If
        ``assert_within_tol`` is True, an error is raised when no point within
        tolerance is found. If ``restore_if_fail`` is True, the initial knob
        values are restored when an error is raised.

        Parameters
        ----------
        n_steps : int, optional
            Maximum number of Jacobian steps. If not provided,
            ``self.n_steps_max`` is used.
        verbose : bool or int, optional
            Verbosity for progress messages. If not provided, ``self.verbose``
            is used.
        take_best : bool, optional
            If True and the final point is not within tolerance, reload the best
            point found during this call. Defaults to True.
        rcond : float, optional
            Cutoff passed to the Jacobian linear solve.
        sing_val_cutoff : float, optional
            Singular-value cutoff passed to the Jacobian linear solve.
        broyden : bool or int, optional
            If True, use Broyden updates between full finite-difference
            Jacobian evaluations. If an integer is provided, a full Jacobian is
            recomputed every ``broyden + 1`` steps.

        Returns
        -------
        Optimize
            The optimizer itself.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)

            opt.solve(verbose=False)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             1        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200            -2        1e+200             0         1e-06             1
            opt.log().show()
            # iteration                   penalty alpha tag tol_met target_active hit_limits vary_active ...
            # 0                           2.23607    -1     nn      yy            nn         yy
            # 1                           2.23607    -1     nn      yy            nn         yy
            # 2                       2.81031e-10     0     nn      yy            nn         yy
        """

        if verbose is None:
            verbose = self.verbose

        if n_steps is None:
            n_steps = self.n_steps_max

        try:
            self.solver.x = self._err._knobs_to_x(self._extract_knob_values())
            self.step(n_steps, verbose=verbose, take_best=take_best, rcond=rcond, sing_val_cutoff=sing_val_cutoff, broyden=broyden)

            if not self._err.last_point_within_tol:
                _print("\n")
                _print("Could not find point within tolerance.")

            if self.assert_within_tol and not self._err.last_point_within_tol:
                raise RuntimeError("Could not find point within tolerance.")

        except Exception as err:
            if self.restore_if_fail:
                self.reload(iteration=0)
            _print("\n")
            raise err
        # if self._err.show_call_counter:
        #     _print("\n")
        return self

    def solve_homotopy(self, n_steps=10):
        """
        Solve a sequence of intermediate target problems.

        The target values are approached in equidistant linear steps from the
        current values to the requested target values. This can improve
        convergence when the full problem is too far from the initial point.
        If a subproblem fails, the last successful tagged point is reloaded.

        Parameters
        ----------
        n_steps : int, optional
            Number of intermediate subproblems to solve.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)
            opt.verbose = False

            opt.solve_homotopy(n_steps=5)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             1        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200            -2        1e+200             0         1e-06             1
            opt.log().show()
            # iteration                   penalty alpha tag tol_met target_active hit_limits vary_active ...
            # 0                           2.23607    -1     nn      yy            nn         yy
            # 1                           2.23607    -1     nn      yy            nn         yy
            # 2                       2.81031e-10     0     nn      yy            nn         yy
        """

        steps = np.linspace(0, 1, n_steps + 1)[1:]
        init_res_values = self._err.last_res_values
        target_values = np.array([tt.value for tt in self.targets])

        for i in range(n_steps):
            sub_targets = (1 - steps[i]) * init_res_values + steps[i] * target_values

            for oldtar, newval in zip(self.targets, sub_targets):
                oldtar.value = newval

            try:
                self.solve()
            except RuntimeError:
                # Reverting values
                print("Reverting Values")
                self.reload(tag=f"Homotopy it {i-1}")
                return

            self.tag(f"Homotopy it {i}")

    def _add_starting_point_to_log_and_print(self, verbose):
        if verbose is None or verbose > 0:
            _print("                                             ")
        self.tag()
        pen_start = self._log["penalty"][-1]
        to_print = 'Optimize'
        if self.name:
            to_print += f" [{self.name}]"
        to_print += f" - start penalty: {pen_start:.4g}                         "
        if verbose is None or verbose > 0:
            _print(to_print)

    def _print_end(self, verbose):
        pen_end = self._log["penalty"][-1]
        to_print = '\nOptimize'
        if self.name:
            to_print += f" [{self.name}]"
        to_print += f" - end penalty:  {pen_end:-4g}                            "
        if verbose is None or verbose > 0:
            _print(to_print)

    def vary_status(self, ret=False, max_col_width=40, iter_ref=0, maxwidth=1000):
        """
        Display or return the status of the variables.

        The returned table includes current values, limits, finite-difference
        steps, weights, tags, active state, and values at a reference log
        iteration.

        Parameters
        ----------
        ret : bool, optional
            If True, return the status as a Table. Defaults to False.
        max_col_width : int, optional
            Maximum column width. Defaults to 40.
        iter_ref : int, optional
            Iteration to use as reference. Defaults to 0.
        maxwidth : int, optional
            Maximum width used when printing the table. Defaults to 1000.

        Returns
        -------
        Table or None
            Status table when ``ret=True``; otherwise prints the table.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)
            opt.solve(verbose=False)

            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             1        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200            -2        1e+200             0         1e-06             1

            vary_table = opt.vary_status(ret=True)
            vary_table.cols["id state met name current_val"]
            # id state met name current_val
            # 0  ON    OK  0               1
            # 1  ON    OK  1              -2
        """

        vvv = self._vary_table()
        vvv["name"] = np.array([vv.name for vv in self.vary])
        vvv["current_val"] = np.array(self._err._extract_knob_values())
        vvv["lower_limit"] = np.array(
            [(vv.limits[0] if vv.limits is not None else None) for vv in self.vary]
        )
        vvv["upper_limit"] = np.array(
            [(vv.limits[1] if vv.limits is not None else None) for vv in self.vary]
        )
        vvv[f"val_at_iter_{iter_ref}"] = self.log().vary[iter_ref, :]
        vvv["step"] = np.array([vv.step for vv in self.vary])
        vvv["weight"] = np.array([vv.weight for vv in self.vary])

        # check if variable is in limits
        in_lim = []
        for vv, cv, lo, hi in zip(
            self.vary, vvv["current_val"], vvv["lower_limit"], vvv["upper_limit"]
        ):
            good = "OK"
            if lo is not None and cv < lo:
                good = "LOW"
            if hi is not None and cv > hi:
                good = "HIGH"
            in_lim.append(good)
        vvv["met"] = np.array(in_lim)

        vvv._col_names = [
            "id",
            "state",
            "tag",
            "met",
            "name",
            "lower_limit",
            "current_val",
            "upper_limit",
            f"val_at_iter_{iter_ref}",
            "step",
            "weight",
        ]


        if ret:
            return vvv
        else:
            print("Vary status:                 ")
            vvv.show(max_col_width=max_col_width, maxwidth=maxwidth)


    def target_status(self, ret=False, max_col_width=40, maxwidth=1000):
        """
        Display or return the status of the targets.

        The returned table includes target values, current values, residuals,
        tolerance flags, tags, active state, and target descriptions.

        Parameters
        ----------
        ret : bool, optional
            If True, return the status as a Table. Defaults to False.
        max_col_width : int, optional
            Maximum column width. Defaults to 40.
        maxwidth : int, optional
            Maximum width used when printing the table. Defaults to 1000.

        Returns
        -------
        Table or None
            Status table when ``ret=True``; otherwise prints the table.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)
            opt.solve(verbose=False)

            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON           True             0             0          0 0, val=0, tol=1e-12, weight=1
            # 1  ON           True             0             0          0 1, val=0, tol=1e-12, weight=1

            target_table = opt.target_status(ret=True)
            target_table.cols["id state tol_met residue current_val target_val"]
            # id state tol_met residue current_val target_val
            # 0  ON    True          0           0          0
            # 1  ON    True          0           0          0
        """

        ttt = self._targets_table()
        self._err(None, check_limits=False)
        ttt["tol_met"] = self._err.last_targets_within_tol
        ttt["residue"] = self._err.last_residue_values
        ttt["current_val"] = np.array(self._err.last_res_values)
        ttt["target_val"] = np.array([tt.value for tt in self.targets])

        ttt._col_names = [
            "id",
            "state",
            "tag",
            "tol_met",
            "residue",
            "current_val",
            "target_val",
            "description",
        ]

        if ret:
            return ttt
        else:
            print("Target status:               ")
            ttt.show(max_col_width=max_col_width, maxwidth=maxwidth)

    def target_mismatch(self, ret=False, max_col_width=40, maxwidth=1000):
        """
        Display or return only targets that are not within tolerance.

        Parameters
        ----------
        ret : bool, optional
            If True, return the status as a Table. Defaults to False.
        max_col_width : int, optional
            Maximum column width. Defaults to 40.
        maxwidth : int, optional
            Maximum width used when printing the table. Defaults to 1000.

        Returns
        -------
        Table or None
            Mismatch table when ``ret=True``; otherwise prints the table.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)

            opt.target_mismatch()
            # Target mismatch:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON          False            -1            -1          0 0, val=0, tol=1e-12, weight=1
            # 1  ON          False             2             2          0 1, val=0, tol=1e-12, weight=1
        """

        out = self.target_status(ret=True)
        out = out.rows[out.tol_met == False]
        if ret:
            return out
        else:
            print("Target mismatch:             ")
            out.show(max_col_width=max_col_width, maxwidth=maxwidth)

    def get_knob_values(self, iteration=None):
        """
        Get the knob values at a given iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration to use. Defaults to None, i.e. the last iteration.

        Returns
        -------
        dict
            Dictionary of knob values.
        """

        if iteration is None:
            iteration = len(self._log["penalty"]) - 1
        out = dict()
        for ii, vv in enumerate(self.vary):
            out[vv.name] = self._log["knobs"][iteration][ii]

        return out

    def show(self, vary=True, targets=True, maxwidth=1000, max_col_width=80):
        """
        Display the knobs and targets used in the optimization.

        Parameters
        ----------
        vary : bool, optional
            If True, display the knobs. Defaults to True.
        targets : bool, optional
            If True, display the targets. Defaults to True.
        maxwidth : int, optional
            Maximum width of the table. Defaults to 1000.
        max_col_width : int, optional
            Maximum column width. Defaults to 80.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)

            opt.show()
            # Vary:
            # id tag state description
            # 0      ON    name=0, limits=(-1e+200, 1e+200), step=1e-06, weight=1
            # 1      ON    name=1, limits=(-1e+200, 1e+200), step=1e-06, weight=1
            # Targets:
            # id tag state description
            # 0      ON    0, val=0, tol=1e-12, weight=1
            # 1      ON    1, val=0, tol=1e-12, weight=1

            opt.show(targets=False)
            # Vary:
            # id tag state description
            # 0      ON    name=0, limits=(-1e+200, 1e+200), step=1e-06, weight=1
            # 1      ON    name=1, limits=(-1e+200, 1e+200), step=1e-06, weight=1
        """

        if vary:
            print("Vary:")
            self._vary_table().show(maxwidth=maxwidth, max_col_width=max_col_width)
        if targets:
            print("Targets:")
            self._targets_table().show(maxwidth=maxwidth, max_col_width=max_col_width)

    def log(self):
        """
        Return the optimization log as a Table.

        The log contains the penalty, step length, active masks, limit hits,
        target values, variable values, and any tags attached to logged points.

        Returns
        -------
        Table
            Optimization log.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)
            opt.solve(verbose=False)

            log = opt.log()
            log.cols["iteration penalty tag"]
            # Table: 4 rows, 3 cols
            # iteration                   penalty tag
            # 0                           2.23607
            # 1                           2.23607
            # 2                       2.81031e-10
            # 3                                 0
        """

        out_dct = dict()
        out_dct["penalty"] = np.array(self._log["penalty"])
        out_dct["alpha"] = np.array(self._log["alpha"])
        out_dct["tag"] = np.array(self._log["tag"])
        out_dct["tol_met"] = np.array(self._log["tol_met"])
        out_dct["target_active"] = np.array(self._log["target_active"])
        out_dct["hit_limits"] = np.array(self._log["hit_limits"])
        out_dct["vary_active"] = np.array(self._log["vary_active"])
        out_dct["iteration"] = np.arange(len(out_dct["penalty"]))
        out_dct["last_jac_rank"] = np.array(self._log["last_jac_rank"])
        out_dct["last_jac_cond"] = np.array(self._log["last_jac_cond"])

        knob_array = np.array(self._log["knobs"])
        for ii, vv in enumerate(self.vary):
            out_dct[f"vary_{ii}"] = knob_array[:, ii]

        target_array = np.array(self._log["targets"])
        for ii, tt in enumerate(self.targets):
            out_dct[f"target_{ii}"] = target_array[:, ii]

        out_dct["vary"] = knob_array
        out_dct["targets"] = target_array

        out = Table(out_dct, index="iteration")
        return out

    def reload(self, iteration=None, tag=None):
        """
        Reload the knob values from a given iteration in the optimization log.

        Parameters
        ----------
        iteration : int, optional
            Iteration to use.
        tag : str, optional
            Reload the last log entry with the given tag. Exactly one of
            ``iteration`` and ``tag`` must be provided.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)
            opt.solve(verbose=False)
            opt.tag("matched")

            opt.reload(iteration=0)
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             0        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200             0        1e+200             0         1e-06             1

            opt.reload(tag="matched")
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             1        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200            -2        1e+200             0         1e-06             1
        """
        assert iteration is not None or tag is not None
        if tag is not None:
            assert iteration is None
            if tag not in self._log["tag"]:
                raise ValueError(f"Tag `{tag}` not found.")
            iteration = np.where(np.array(self._log["tag"]) == tag)[0][-1]

        assert iteration < len(self._log["penalty"])
        knob_values = self._log["knobs"][iteration]
        mask_input = _bool_array_from_string(self._log["vary_active"][iteration])
        for vv, rr, aa in zip(self.vary, knob_values, mask_input):
            vv.container[vv.name] = rr
            vv.active = aa
        mask_output = _bool_array_from_string(self._log["target_active"][iteration])
        for tt, aa in zip(self.targets, mask_output):
            tt.active = aa
        self.add_point_to_log()

    def clear_log(self):
        """
        Clear the optimization log.

        A new log entry for the current point is added immediately after the log
        is cleared.
        """

        for kk in self._log:
            self._log[kk].clear()
        self.add_point_to_log()

    def add_point_to_log(self, tag=""):
        """
        Add the current point to the optimization log.

        Parameters
        ----------
        tag : str, optional
            Tag to add to the point. Defaults to ''.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)

            opt.add_point_to_log(tag="before_new_stage")
            opt.log().cols["iteration penalty tag"]
            # Table: 2 rows, 3 cols
            # iteration                   penalty tag
            # 0                           2.23607
            # 1                           2.23607 before_new_stage
        """

        knobs = self._extract_knob_values()
        self._log["knobs"].append(knobs)
        x = self._err._knobs_to_x(knobs)
        _, penalty = self.solver.eval(x)
        self._log["targets"].append(self._err.last_res_values)
        self._log["penalty"].append(penalty)
        self._log["tol_met"].append(
            _bool_array_to_string(self._err.last_targets_within_tol)
        )
        self._log["hit_limits"].append("".join(["n"] * len(knobs)))
        self._log["vary_active"].append(_bool_array_to_string(self._err.mask_input))
        self._log["target_active"].append(_bool_array_to_string(self._err.mask_output))
        self._log["alpha"].append(-1)
        self._log["tag"].append(tag)
        self._log["last_jac_rank"].append(-1)
        self._log["last_jac_cond"].append(-1)

    def tag(self, tag=""):
        """
        Tag the current point in the optimization log.

        Parameters
        ----------
        tag : str, optional
            Tag to add to the point. Defaults to ''.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)
            opt.solve(verbose=False)

            opt.tag("matched")
            opt.log().cols["iteration penalty tag"]
            # Table: 5 rows, 3 cols
            # iteration                   penalty tag
            # 0                           2.23607
            # 1                           2.23607
            # 2                       2.81031e-10
            # 3                                 0
            # 4                                 0 matched
        """
        self.add_point_to_log(tag=tag)

    def enable(self, target=None, vary=None, vary_name=None):
        """
        Enable a list of variables and targets.

        Selectors can be a single id/tag/name, a list of selectors, ``True`` to
        select all, or ``False`` to select none. String selectors are matched as
        regular expressions.

        Parameters
        ----------
        target: str, int, list of int or string, True, False.
            If target is True, enable all targets, if False, disable all targets,
            else enable the targets with corresponding id if target is int or tag if target is str.
            String are matched as regular expression.
        vary: list of int or string
            If True, enable all variables. If False, disable all variables.
            Else enable the variables with corresponding id or tag or all if True.
            String are matched as regular expression.
        vary_name: list of str
            Enable the variables with corresponding name.
            String are matched as regular expression.

        Returns
        -------
        Optimize
            The optimizer itself.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)
            opt.disable(target=True, vary=True)

            opt.enable(target=0, vary=0)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON          False            -1            -1          0 0, val=0, tol=1e-12, weight=1
            # 1  OFF         False             2             2          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             0        1e+200             0         1e-06             1
            # 1  OFF       OK     1       -1e+200             0        1e+200             0         1e-06             1
        """
        _set_state(self.targets, True, target)
        _set_state(self.vary, True, vary, attr="tag")
        _set_state(self.vary, True, vary_name, attr="name")
        return self

    def disable(self, target=None, vary=None, vary_name=None):
        """
        Disable a list of variables and targets.

        Selectors can be a single id/tag/name, a list of selectors, ``True`` to
        select all, or ``False`` to select none. String selectors are matched as
        regular expressions.

        Parameters
        ----------
        target: list of int or string
            If True, disable all targets. If False, enable all targets.
            Else enable the targets with corresponding id or tag or all if True.
            String are matched as regular expression.
        vary: list of int or string
            If True, disable all variables. If False, enable all variables.
            Else enable the variables with corresponding id or tag or all if True.
            String are matched as regular expression.
        vary_name: list of str
            Disable the variables with corresponding name.
            String are matched as regular expression.

        Returns
        -------
        Optimize
            The optimizer itself.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)

            opt.disable(target=0, vary=0)
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  OFF         False            -1            -1          0 0, val=0, tol=1e-12, weight=1
            # 1  ON          False             2             2          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  OFF       OK     0       -1e+200             0        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200             0        1e+200             0         1e-06             1
        """
        _set_state(self.targets, False, target)
        _set_state(self.vary, False, vary, attr="tag")
        _set_state(self.vary, False, vary_name, attr="name")
        return self

    def get_merit_function(self, check_limits=True, return_scalar=None, rescale_x=None):
        """
        Get the merit function that can be used with a different optimizer.

        The returned object exposes the current optimizer variables as a vector
        and evaluates the residuals used by the optimizer. It also provides a
        finite-difference Jacobian through ``get_jacobian``.

        Parameters
        ----------
        check_limits : bool, optional
            If True, enforce that the knob values are within the limits.
            An error is raised if a knob value is outside the limits.
            Defaults to True.
        return_scalar : bool, optional
            If True, return a scalar value. If False, return an array.
            If None, use the default value for the solver. Defaults to None.
        rescale_x : tuple, optional
            If set, merit_function normalizes x to the given interval.
            If None, x is used as is.

        Returns
        -------
        MeritFunctionView
            Callable merit-function view.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)

            merit = opt.get_merit_function(return_scalar=False)
            x = merit.get_x()
            residuals = merit(x)
            # [-1.  2.]
            jacobian = merit.get_jacobian(x)
            # [[1. 0.]
            #  [0. 1.]]
            opt.target_status()
            # Target status:
            # id state tag tol_met       residue   current_val target_val description
            # 0  ON          False            -1            -1          0 0, val=0, tol=1e-12, weight=1
            # 1  ON          False             2             2          0 1, val=0, tol=1e-12, weight=1
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             0        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200             0        1e+200             0         1e-06             1
        """

        return self._err.get_merit_function(
            check_limits=check_limits, return_scalar=return_scalar, rescale_x=rescale_x
        )

    def _clip_to_limits(self):
        for vv in self.vary:
            if vv.active:
                cv = vv.get_value()
                if vv.limits is None:
                    continue
                if vv.limits[0] is not None:
                    if cv < vv.limits[0]:
                        vv.container[vv.name] = vv.limits[0]
                if vv.limits[1] is not None:
                    if cv > vv.limits[1]:
                        vv.container[vv.name] = vv.limits[1]

    #### DEPRECATED METHODS ####

    def enable_vary(self, id=None, tag=None):
        """
        Enable one or more variables.

        .. warning:: Deprecated. Please use :meth:`enable` passing ``id`` or ``tag``.

        Parameters
        ----------
        id : int or list of int, optional
            Index of the variable to disable. Defaults to None.
        tag : str or list of str, optional
            Tag of the variable to disable.
        """
        warn(
            "The function `enable_vary` is deprecated, use `enable(vary=id_or_tags)`.",
            FutureWarning,
        )
        self.enable(vary=_add_id_tag(id, tag))

    def disable_vary(self, id=None, tag=None):
        """
        Disable one or more variables.

        .. warning:: Deprecated. Please use :meth:`disable` passing ``id`` or ``tag``.

        Parameters
        ----------
        id : int or list of int, optional
            Index of the variable to disable. Defaults to None.
        tag : str or list of str, optional
            Tag of the variable to disable.
            Str is interpreted as regular expression. Defaults to None.
        """
        warn(
           "The function `disable_vary` is deprecated, use `disable(vary=id_or_tags)`.",
            FutureWarning,
        )
        self.disable(vary=_add_id_tag(id, tag))

    def enable_targets(self, *id_or_tag, id=None, tag=None):
        """
        Enable one or more targets.

        .. warning:: Deprecated. Please use :meth:`enable`.

        Parameters
        ----------
        id_or_tag : int or string
            Disable the targets with corresponding id or tag.
        Depraecated:
        id : int or list of int, optional
            Index of the targets to disable. Defaults to None.
        tag : str or list of str, optional
            Tag of the targets to disable.
            Str is interpreted as regular expression. Defaults to None.
        """
        warn(
           "The function `enable_targets` is deprecated, use `enable(target=id_or_tags)`.",
            FutureWarning,
        )
        self.enable(target=_add_id_tag(id, tag))

    def disable_targets(self, id=None, tag=None):
        """
        Disable one or more targets.

        .. warning:: Deprecated. Please use :meth:`disable` passing ``id`` or ``tag``.

        Parameters
        ----------
        id : int or list of int, optional
            Index of the targets to disable. Defaults to None.
        tag : str or list of str, optional
            Tag of the targets to disable.
            Str is interpreted as regular expression. Defaults to None.
        """
        warn(
           "The function `disable_targets` is deprecated, use `disable(target=ids_or_tags)`.",
            FutureWarning,
        )
        self.disable(target=_add_id_tag(id, tag))

    def disable_all_targets(self):
        """
        Disable all targets.

        .. warning:: Deprecated. Please use :meth:`disable` with ``target=True``.
        """
        warn(
           "The function `disable_all_targets` is deprecated, use `disable(target=True).",
            FutureWarning,
        )
        for tt in self.targets:
            tt.active = False

    def enable_all_targets(self):
        """
        Enable all targets.

        .. warning:: Deprecated. Please use :meth:`enable` with ``target=True``.
        """
        warn(
           "The function `enable_all_targets` is deprecated, use `enable(target=True).",
            FutureWarning,
        )
        for tt in self.targets:
            tt.active = True
        return self

    def disable_all_vary(self):
        """
        Disable all knobs.

        .. warning:: Deprecated. Please use :meth:`disable` with ``vary=True``.
        """
        warn(
           "The function `disable_all_vary` is deprecated, use `disable(vary=True).",
            FutureWarning,
        )
        for vv in self.vary:
            vv.active = False
        return self

    def enable_all_vary(self):
        """
        Enable all knobs.

        .. warning:: Deprecated. Please use :meth:`enable` with ``vary=True``.
        """
        warn(
           "The function `enable_all_vary` is deprecated, use `enable(vary=True).",
            FutureWarning,
        )
        for vv in self.vary:
            vv.active = True
        return self

    #### END DEPRECATED METHODS ####

    @property
    def check_limits(self):
        """
        Whether variable limits are enforced when evaluating the merit function.

        If True, evaluating a point outside the configured limits raises an
        error. If False, Jacobian steps clip active variables to their limits.
        """
        return self._err.check_limits

    @check_limits.setter
    def check_limits(self, value):
        self._err.check_limits = value

    @property
    def _err(self):
        return self.solver.func

    @_err.setter
    def _err(self, value):
        self.solver.func = value

    @property
    def actions(self):
        """
        Actions used to compute target quantities.

        Each action is prepared during optimizer construction and run whenever
        the merit function needs fresh target data.
        """
        return self._err.actions

    @property
    def verbose(self):
        """
        Verbosity used by the optimizer and its Jacobian solver.
        """
        return self.solver.verbose

    @verbose.setter
    def verbose(self, value):
        self.solver.verbose = value

    @property
    def vary(self):
        """
        Container with the active and inactive variables of the optimizer.

        The container can be inspected directly and supports convenience display
        through its representation in notebooks and terminals. Use
        :meth:`vary_status` for a tabular view with current values and limits.
        """
        return OptContainer(self, 'vary')

    @property
    def targets(self):
        """
        Container with the active and inactive targets of the optimizer.

        Use :meth:`target_status` for a tabular view with current values,
        target values, residuals, and tolerance flags.
        """
        return OptContainer(self, 'targets')

    def _is_within_tol(self):
        return self._err()

    def set_knobs_from_x(self, x):
        """
        Set active variable values from an optimizer vector.

        Parameters
        ----------
        x : array-like
            Vector in the optimizer coordinates. Values are converted to native
            knob values using the internal variable scaling before assignment.

        Examples
        --------
        .. code-block:: python

            import xdeps as xd

            def f(x):
                return [x[0] - 1, x[1] + 2]

            opt = xd.Optimize.from_callable(
                f, x0=[0., 0.], tar=[0, 0], steps=[1e-6, 1e-6],
                tols=[1e-12, 1e-12], show_call_counter=False)

            merit = opt.get_merit_function()
            x = merit.get_x()
            x[0] = 1.
            x[1] = -2.
            opt.set_knobs_from_x(x)
            opt.vary_status()
            # Vary status:
            # id state tag met name   lower_limit   current_val   upper_limit val_at_iter_0          step        weight
            # 0  ON        OK     0       -1e+200             1        1e+200             0         1e-06             1
            # 1  ON        OK     1       -1e+200            -2        1e+200             0         1e-06             1
        """
        for vv, rr in zip(self.vary, self._err._x_to_knobs(x)):
            if vv.active:
                vv.container[vv.name] = rr

    def _vary_table(self):
        return _make_table(self.vary)

    def _targets_table(self):
        return _make_table(self.targets)

    def _extract_knob_values(self):
        return self._err._extract_knob_values()


def _bool_array_to_string(arr, dct={True: "y", False: "n"}):
    return "".join([dct[aa] for aa in arr])


def _bool_array_from_string(strng, dct={"y": True, "n": False}):
    for ss in strng:
        assert ss in dct, f"Invalid character {ss}"
    return np.array([dct[ss] for ss in strng])


def _make_table(vary):
    id = []
    tag = []
    state = []
    description = []
    for ii, vv in enumerate(vary):
        id.append(ii)
        tag.append(vv.tag)
        state.append("ON" if vv.active else "OFF")
        vv_repr = vv.__repr__()
        vv_repr = vv_repr.replace("Vary(", "")
        vv_repr = vv_repr.replace("Vary", "")
        vv_repr = vv_repr.replace("TargetPhaseAdv(", "")
        vv_repr = vv_repr.replace("Target(", "")
        vv_repr = vv_repr.replace("Target", "")
        if vv_repr[-1] == ")":
            vv_repr = vv_repr[:-1]
        description.append(vv_repr)
    id = np.array(id)
    tag = np.array(tag)
    state = np.array(state)
    description = np.array(description)
    return Table(dict(id=id, tag=tag, state=state, description=description), index="id")


def _set_state(lst, state, entries, attr="tag"):
    if entries is None:
        return
    elif entries is True:
        for vv in lst:
            vv.active = state
        return
    elif entries is False:
        for vv in lst:
            vv.active = not state
        return
    if isinstance(entries, int) or isinstance(entries, str):
        entries = [entries]
    for entry in entries:
            if isinstance(entry, int):
                lst[entry].active = state
            elif isinstance(entry, str):
                for vv in lst:
                    if re.fullmatch(entry, getattr(vv, attr)):
                        vv.active = state


def _add_id_tag(id, tag):
    id_or_tag = ()
    if isinstance(id, int):
        id_or_tag += (id,)
    elif isinstance(id, (list, tuple, np.ndarray)):
        id_or_tag += tuple(id)
    if isinstance(tag, str):
        id_or_tag += (tag,)
    elif isinstance(tag, (list, tuple, np.ndarray)):
        id_or_tag += tuple(tag)
    return id_or_tag

class OptContainer:

    def __init__(self, optimize, what):
        self.optimize = optimize
        assert what in ['targets', 'vary']
        self.what = what

    def __repr__(self):
        if self.what == 'vary':
            return self.optimize._vary_table().show(output=str)
        else:
            return self.optimize._targets_table().show(output=str)

    def status(self, *args, **kwargs):
        if self.what == 'vary':
            return self.optimize.vary_status(*args, **kwargs)
        else:
            return self.optimize.target_status(*args, **kwargs)

    def __getitem__(self, key):
        if self.what == 'targets':
            container = self.optimize._err.targets
        else:
            container = self.optimize._err.vary
        if isinstance(key, int):
            return container[key]
        if isinstance(key, str):
            out = []
            for vv in container:
                if re.fullmatch(key, vv.tag):
                    out.append(vv)
            if len(out) == 1:
                return out[0]
            if len(out) == 0:
                raise ValueError(f"Tag {key} not found.")
            return out
        if isinstance(key, slice):
            return container[key]

    def __setitem__(self, key, value):
        raise ValueError(f"Cannot replace {self.what}.")

    def __delitem__(self, key):
        raise ValueError(f"Cannot delete {self.what}.")

    def __len__(self):
        if self.what == 'vary':
            return len(self.optimize._err.vary)
        else:
            return len(self.optimize._err.targets)

    def __iter__(self):
        if self.what == 'vary':
            return iter(self.optimize._err.vary)
        else:
            return iter(self.optimize._err.targets)

    def extend(self, *args, **kwargs):
        raise ValueError(f"Cannot extend {self.what}.")

    def copy(self):
        if self.what == 'vary':
            return self.optimize._err.vary.copy()
        else:
            return self.optimize._err.targets.copy()

class ActionCall(Action):
    def __init__(self, function, vary):
        self.vary = vary
        self.function = function

    def run(self):
        x = [vv.container[vv.name] for vv in self.vary]
        return self.function(x)

    def get_targets(self, ftar):
        tars = []
        for ii in range(len(ftar)):
            tars.append(Target(ii, ftar[ii], action=self))

        return tars
