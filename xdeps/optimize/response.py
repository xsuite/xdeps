import numpy as np

from .matrixutils import SVD


class ResponseMatrix:
    """A fixed, local derivative of unweighted residuals w.r.t. physical knobs.

    Rows and columns follow the optimizer's full target and vary lists, including
    inactive entries. Target transformations (including log10 for optimize_log)
    belong in this derivative; target weights and Vary weights do not. Changing
    weights or activation does not require rebuilding the response. Changing the
    observable definitions does.

    Construct through ``opt.set_response(matrix)`` or ``opt.build_response()``.
    This object is bound to that optimizer's targets and varies. It intentionally
    remains fixed when the machine changes; rebuild explicitly when appropriate.

    Examples
    --------
    opt.set_response(opt.build_response())  # Freeze a numerical response.
    svd = opt.response.svd                 # Inspect U, s, Vh, rank and cond.
    delta = opt.get_correction(rcond=1e-5)  # Propose, without changing knobs.
    opt.step(rcond=1e-5)                   # Or use the existing step policy.
    opt.set_response(None)                # Return to the default Jacobian.
    """

    def __init__(self, matrix, optimizer, metadata=None):
        matrix = np.array(matrix, dtype=float, copy=True)
        expected = (len(optimizer.targets), len(optimizer.vary))
        if matrix.shape != expected:
            raise ValueError(f"Response shape {matrix.shape}; expected {expected}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Response matrix must contain only finite values")
        matrix.flags.writeable = False
        self._matrix = matrix
        self._targets = tuple(optimizer.targets)
        self._vary = tuple(optimizer.vary)
        self.rows = tuple(tt.tag or str(tt.tar) for tt in self._targets)
        self.columns = tuple(vv.name for vv in self._vary)
        self.reference_knobs = tuple(vv.get_value() for vv in self._vary)
        self.metadata = dict(metadata or {})
        self._svd = None
        self._scaled_matrix = None
        self._scaling = None
        self._svd_source = None
        self._svd_masks = None

    @property
    def matrix(self):
        """Read-only unweighted response; install a new response to replace it."""
        return self._matrix

    @property
    def svd(self):
        """Cached SVD of the currently weighted, scaled, active response.

        Exposes U, s, Vh and lstsq(). Changing the solve cutoff reuses this
        object; changing weights, scales or activation rebuilds it lazily.
        """
        outputs = np.array([tt.active for tt in self._targets], dtype=bool)
        inputs = np.array([vv.active for vv in self._vary], dtype=bool)
        return self._get_svd(self._scaled(), inputs, outputs)

    def _get_svd(self, jac, inputs, outputs):
        masks = (inputs.tobytes(), outputs.tobytes())
        if jac is not self._svd_source or masks != self._svd_masks:
            matrix = (
                jac
                if np.all(inputs) and np.all(outputs)
                else jac[outputs, :][:, inputs]
            )
            if self._svd is None or not np.array_equal(self._svd.matrix, matrix):
                self._svd = SVD(matrix)
            self._svd_source = jac
            self._svd_masks = masks
        return self._svd

    def _validate(self, merit):
        if tuple(merit.targets) != self._targets or tuple(merit.vary) != self._vary:
            raise ValueError("Response targets/varies differ from this optimizer")

    def _scaled(self, merit=None):
        if merit is not None:
            self._validate(merit)
        weights = tuple(
            tt.weight if tt.weight is not None else 1.0 for tt in self._targets
        )
        scales = tuple(vv.weight if vv.weight is not None else 1.0 for vv in self._vary)
        if (weights, scales) != self._scaling:
            self._scaled_matrix = (
                self.matrix * np.array(weights)[:, None] * np.array(scales)[None, :]
            )
            self._scaled_matrix.flags.writeable = False
            self._scaling = (weights, scales)
        return self._scaled_matrix
