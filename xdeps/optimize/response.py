from collections import OrderedDict
from types import MappingProxyType

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
        self._bind(optimizer.targets, optimizer.vary, metadata)
        self._set_matrix(matrix)

    def _set_matrix(self, matrix):
        matrix = np.array(matrix, dtype=float, copy=True)
        expected = (len(self._targets), len(self._vary))
        if matrix.shape != expected:
            raise ValueError(f"Response shape {matrix.shape}; expected {expected}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Response matrix must contain only finite values")
        matrix.flags.writeable = False
        self._matrix = matrix

    def _bind(self, targets, vary, metadata=None):
        self._targets = tuple(targets)
        self._vary = tuple(vary)
        self.rows = tuple(tt.tag or str(tt.tar) for tt in self._targets)
        self.columns = tuple(vv.name for vv in self._vary)
        self.reference_knobs = tuple(vv.get_value() for vv in self._vary)
        self.metadata = dict(metadata or {})
        self._svd = None
        self._scaled_matrix = None
        self._scaling = None
        self._svd_source = None
        self._svd_masks = None
        self._svd_cache = OrderedDict()

    @classmethod
    def _from_targets(cls, matrix, targets, vary):
        response = cls.__new__(cls)
        response._bind(targets, vary)
        response._set_matrix(matrix)
        return response

    @property
    def matrix(self):
        """Read-only unweighted response; install a new response to replace it."""
        return self._matrix

    @property
    def svd(self):
        """Cached SVD of the currently weighted, scaled, active response.

        Exposes U, s, Vh and lstsq(). Changing the solve cutoff reuses this
        object. The last eight activation selections are cached; changing
        weights or knob scales clears those decompositions lazily.
        """
        outputs = np.array([tt.active for tt in self._targets], dtype=bool)
        inputs = np.array([vv.active for vv in self._vary], dtype=bool)
        return self._get_svd(self._scaled(), inputs, outputs)

    def _get_svd(self, jac, inputs, outputs):
        masks = (inputs.tobytes(), outputs.tobytes())
        if jac is not self._svd_source:
            self._svd_cache.clear()
        if jac is not self._svd_source or masks != self._svd_masks:
            if masks in self._svd_cache:
                self._svd = self._svd_cache.pop(masks)
            else:
                matrix = (
                    jac
                    if np.all(inputs) and np.all(outputs)
                    else jac[np.ix_(outputs, inputs)]
                )
                self._svd = SVD(matrix)
            self._svd_cache[masks] = self._svd
            if len(self._svd_cache) > 8:
                self._svd_cache.popitem(last=False)
            self._svd_source = jac
            self._svd_masks = masks
        return self._svd

    def _get_solver(self, inputs, outputs):
        return self._get_svd(self._scaled(), inputs, outputs)

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


class BlockResponseMatrix(ResponseMatrix):
    """A fixed response with disjoint row/column blocks and zero cross terms.

    Construct with ``opt.set_response(blocks={name: dict(matrix=..., targets=...,
    vary=..., rcond=...)})``. ``targets`` and ``vary`` are slices or sequences of
    indices in the optimizer's full lists. Every target and vary must belong to
    exactly one block. Repeated physical observables may be represented by
    distinct Target objects, for example tunes fitted by different knob families.

    Weights, knob scales and activation are read from those Target/Vary objects.
    Each block caches its own SVDs. By default each block uses its configured
    ``rcond`` (1e-14) and ``sing_val_cutoff`` (all modes); explicit scalar solve
    arguments override those defaults for every active block.

    ``blocks[name]`` exposes a block's ResponseMatrix. ``svd`` is available when
    only one block is selected, in that block's row/column order. ``matrix``
    explicitly constructs the full dense matrix for inspection; ordinary
    Jacobian steps and correction directions never materialize it. Cross responses require a dense response
    or a larger block containing the coupled rows and columns. Broyden updates
    require a dense response, as they can introduce cross-block terms.
    """

    def __init__(self, blocks, optimizer):
        self._bind(optimizer.targets, optimizer.vary)
        parts, indices, cutoffs = {}, {}, {}
        covered_rows = set()
        covered_columns = set()
        for name, specification in blocks.items():
            extra = set(specification) - {
                'matrix', 'targets', 'vary', 'rcond', 'sing_val_cutoff'}
            if extra:
                raise ValueError(f'Unknown response block fields: {sorted(extra)}')
            rows = self._indices(specification['targets'], len(self._targets))
            columns = self._indices(specification['vary'], len(self._vary))
            if covered_rows.intersection(rows) or covered_columns.intersection(columns):
                raise ValueError('Response blocks must have disjoint targets and varies')
            covered_rows.update(rows)
            covered_columns.update(columns)
            parts[name] = ResponseMatrix._from_targets(
                specification['matrix'], [self._targets[i] for i in rows],
                [self._vary[i] for i in columns])
            indices[name] = rows, columns
            cutoffs[name] = (specification.get('rcond', 1e-14),
                             specification.get('sing_val_cutoff'))
        if (covered_rows != set(range(len(self._targets)))
                or covered_columns != set(range(len(self._vary)))):
            raise ValueError('Response blocks must cover every target and vary')
        self.blocks = MappingProxyType(parts)
        self._indices_by_block = indices
        self._cutoffs = cutoffs

    @staticmethod
    def _indices(selection, size):
        if isinstance(selection, slice):
            result = np.arange(size)[selection]
        else:
            result = np.asarray(selection)
            if result.ndim != 1 or result.dtype.kind not in 'iu':
                raise ValueError('Block indices must be a slice or a sequence of integers')
        if (len(result) == 0 or np.any(result < 0) or np.any(result >= size)
                or len(np.unique(result)) != len(result)):
            raise ValueError('Block indices must be nonempty, unique and within range')
        result = np.array(result, dtype=int, copy=True)
        result.flags.writeable = False
        return result

    @property
    def matrix(self):
        matrix = np.zeros((len(self._targets), len(self._vary)))
        for name, block in self.blocks.items():
            matrix[np.ix_(*self._indices_by_block[name])] = block.matrix
        matrix.flags.writeable = False
        return matrix

    @property
    def svd(self):
        inputs = np.array([v.active for v in self._vary], dtype=bool)
        outputs = np.array([t.active for t in self._targets], dtype=bool)
        solver = self._get_solver(inputs, outputs)
        if (len(solver.parts) != 1
                or len(solver.parts[0][1]) != np.sum(inputs)
                or len(solver.parts[0][2]) != np.sum(outputs)):
            raise ValueError('Select one response block, or inspect response.blocks[name].svd')
        return solver.parts[0][0]

    def _get_solver(self, inputs, outputs):
        # Maps full target/vary indices to the compact active residual/direction.
        input_positions = np.cumsum(inputs) - 1
        output_positions = np.cumsum(outputs) - 1
        parts = []
        for name, block in self.blocks.items():
            rows, columns = self._indices_by_block[name]
            active_inputs, active_outputs = inputs[columns], outputs[rows]
            if not np.any(active_inputs) or not np.any(active_outputs):
                continue
            svd = block._get_solver(active_inputs, active_outputs)
            parts.append((svd, input_positions[columns[active_inputs]],
                          output_positions[rows[active_outputs]], *self._cutoffs[name]))
        return _BlockSolve(parts, int(np.sum(inputs)), int(np.sum(outputs)))


class _BlockSolve:
    """Least-squares solves on disjoint blocks in the active vector ordering."""

    def __init__(self, parts, n_inputs, n_outputs):
        self.parts = parts
        self.n_inputs = n_inputs
        # Report diagnostics for the full active matrix, including zero modes
        # from rectangular blocks or blocks with no active rows/columns.
        singular_values = np.zeros(min(n_inputs, n_outputs))
        values = np.concatenate([part[0].s for part in parts]) if parts else np.array([])
        singular_values[:len(values)] = values
        if len(singular_values):
            largest, smallest = singular_values.max(), singular_values.min()
            tolerance = largest * max(n_inputs, n_outputs) * np.finfo(float).eps
            self.rank = int(np.count_nonzero(singular_values > tolerance))
            self.cond = np.inf if smallest < 1e-16 else largest / smallest
        else:
            self.rank = self.cond = -1

    def lstsq(self, residuals, rcond=None, sing_val_cutoff=None):
        direction = np.zeros((self.n_inputs,) + np.shape(residuals)[1:])
        for svd, inputs, outputs, block_rcond, block_cutoff in self.parts:
            direction[inputs] = svd.lstsq(
                residuals[outputs],
                rcond=block_rcond if rcond is None else rcond,
                sing_val_cutoff=block_cutoff if sing_val_cutoff is None else sing_val_cutoff)
        return direction
