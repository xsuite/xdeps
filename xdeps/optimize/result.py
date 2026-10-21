from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JacobianResult:
    """Outcome of run_jacobian, describing the state actually left in place.

    Penalties are norms of the weighted residual vector. With take_best=True,
    penalty_after includes any restoration performed by the optimizer. This
    result reports progress; acceptance of a larger workflow is up to the caller.
    Model/solver exceptions still propagate normally.
    """

    penalty_before: float
    penalty_after: float
    n_steps: int
    n_calls: int
    step_scale: float | None
    stop_reason: str | None
    within_tol: bool

    @property
    def improved(self):
        return self.penalty_after < self.penalty_before

    @property
    def degraded(self):
        return self.penalty_after > self.penalty_before
