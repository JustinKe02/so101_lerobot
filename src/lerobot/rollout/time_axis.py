"""Realtime-VLA V2 time-axis action planning.

The planner follows the upstream time-parameterisation formulation: each QP
variable is an inverse segment duration ``s = 1 / dt``.  The resulting path is
sampled back onto the policy control period.  OSQP is imported lazily so the
planner remains an optional rollout feature; any import, solver, or numerical
failure returns the unmodified model chunk.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .joint_constraints import JointConstraintArtifactConfig
from .speed_adapter import SpeedAdapter, beta_to_segment_dt_ref

logger = logging.getLogger(__name__)


@dataclass
class TimeAxisPlannerConfig:
    """Configuration for per-chunk time re-parameterisation."""

    enabled: bool = False
    dt_ref: float = 0.05
    dt_min: float = 0.01
    dt_max: float = 0.30
    lambda_acc: float = 1.0
    lambda_time: float = 0.1
    lambda_velocity: float = 10.0
    joint_constraints: JointConstraintArtifactConfig = field(default_factory=JointConstraintArtifactConfig)
    max_velocity: float | list[float] | None = None
    max_acceleration: float | list[float] | None = None
    horizon: int = 20
    stride: int = 10
    optimization_dims: list[int] = field(default_factory=list)
    max_iterations: int = 80

    def __post_init__(self) -> None:
        if not isinstance(self.joint_constraints, JointConstraintArtifactConfig):
            raise ValueError("time_axis_planner.joint_constraints must be a JointConstraintArtifactConfig")
        if self.joint_constraints.enabled and not self.enabled:
            raise ValueError("joint constraint artifact requires time_axis_planner.enabled=true")
        for name in ("dt_ref", "dt_min", "dt_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"time_axis_planner.{name} must be finite and positive")
        if self.dt_min > self.dt_max:
            raise ValueError("time_axis_planner.dt_min must be <= dt_max")
        for name in ("lambda_acc", "lambda_time", "lambda_velocity"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"time_axis_planner.{name} must be finite and non-negative")
        for name in ("max_velocity", "max_acceleration"):
            value = getattr(self, name)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            if not values or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) <= 0
                for item in values
            ):
                raise ValueError(f"time_axis_planner.{name} must contain finite positive values")
        if isinstance(self.horizon, bool) or self.horizon < 2:
            raise ValueError("time_axis_planner.horizon must be at least 2")
        if isinstance(self.stride, bool) or self.stride < 1:
            raise ValueError("time_axis_planner.stride must be positive")
        if isinstance(self.max_iterations, bool) or self.max_iterations < 1:
            raise ValueError("time_axis_planner.max_iterations must be positive")
        if any(isinstance(dim, bool) or int(dim) < 0 for dim in self.optimization_dims):
            raise ValueError("time_axis_planner.optimization_dims must contain non-negative integers")


@dataclass(frozen=True)
class TimeAxisPlan:
    """Result and diagnostics for one action chunk."""

    actions: np.ndarray
    segment_durations: np.ndarray
    used_fallback: bool
    reason: str | None = None
    reference_segment_durations: np.ndarray | None = None
    speed_factors: np.ndarray | None = None


def _load_osqp_modules() -> tuple[Any, Any]:
    """Load optional solver modules only when time-axis planning is used."""

    try:
        import osqp
        from scipy import sparse
    except ImportError as exc:
        raise RuntimeError(
            "OSQP time-axis planning requires the optional 'osqp' and 'scipy' packages"
        ) from exc
    return osqp, sparse


class TimeAxisPlanner:
    """OSQP time re-parameterisation for action chunks.

    ``committed_prefix_steps`` is the number of leading actions that have
    already been committed to the robot.  Those rows are copied without any
    arithmetic conversion and only the postfix is re-sampled.  ``start_index``
    is an equivalent spelling for callers that track a model-chunk cursor.
    """

    def __init__(
        self,
        config: TimeAxisPlannerConfig,
        *,
        speed_adapter: SpeedAdapter | None = None,
    ) -> None:
        self.config = config
        self.speed_adapter = speed_adapter
        self._warned_fallback = False

    @property
    def feature_coordinate_space(self) -> str | None:
        if self.speed_adapter is None:
            return None
        return self.speed_adapter.config.feature_coordinate_space

    def plan(
        self,
        actions,
        *,
        start_index: int = 0,
        committed_prefix_steps: int | None = None,
        state_embeddings=None,
        phase_embeddings=None,
    ) -> TimeAxisPlan:
        if hasattr(actions, "detach"):
            actions = actions.detach().to(device="cpu")
        input_array = np.asarray(actions)
        output_dtype = input_array.dtype if np.issubdtype(input_array.dtype, np.floating) else np.float64
        original = np.asarray(input_array, dtype=np.float64)
        segment_count = max(original.shape[0] - 1, 0) if original.ndim >= 1 else 0

        if original.ndim != 2 or original.shape[0] < 2:
            return self._fallback(original, output_dtype, segment_count, "invalid_shape")
        if not np.isfinite(original).all():
            return self._fallback(original, output_dtype, segment_count, "non_finite_input")
        if not self.config.enabled:
            return TimeAxisPlan(
                original.astype(output_dtype, copy=True),
                np.full(segment_count, self.config.dt_ref),
                False,
            )

        reference_diagnostics: np.ndarray | None = None
        speed_factor_diagnostics: np.ndarray | None = None
        try:
            planning_start = self._resolve_start_index(
                len(original),
                start_index=start_index,
                committed_prefix_steps=committed_prefix_steps,
            )
            if planning_start >= len(original):
                return TimeAxisPlan(
                    original.astype(output_dtype, copy=True),
                    np.full(segment_count, self.config.dt_ref),
                    False,
                )

            # Include the last committed action as a fixed interpolation anchor.
            anchor_index = max(0, planning_start - 1)
            waypoints = original[anchor_index:]
            reference_durations, speed_factors = self._reference_profile(
                waypoints,
                state_embeddings=self._slice_path_embedding(
                    state_embeddings,
                    anchor_index=anchor_index,
                    action_count=len(original),
                ),
                phase_embeddings=self._slice_path_embedding(
                    phase_embeddings,
                    anchor_index=anchor_index,
                    action_count=len(original),
                ),
            )
            if speed_factors is not None:
                reference_diagnostics = np.full(segment_count, self.config.dt_ref, dtype=np.float64)
                speed_factor_diagnostics = np.ones(segment_count, dtype=np.float64)
                reference_diagnostics[anchor_index : anchor_index + len(reference_durations)] = (
                    reference_durations
                )
                speed_factor_diagnostics[anchor_index : anchor_index + len(speed_factors)] = speed_factors
            dims = self._resolve_dims(waypoints.shape[1])
            velocity_limits = self._resolve_limits(
                self.config.max_velocity,
                action_dim=waypoints.shape[1],
                dims=dims,
                name="max_velocity",
            )
            acceleration_limits = self._resolve_limits(
                self.config.max_acceleration,
                action_dim=waypoints.shape[1],
                dims=dims,
                name="max_acceleration",
            )
            speeds = self._solve_rolling(
                waypoints[:, dims],
                reference_durations=reference_durations,
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
            )
            optimized_dims = self._resample(waypoints[:, dims], speeds)

            output = original.copy()
            local_output = waypoints.copy()
            local_output[:, dims] = optimized_dims
            local_start = planning_start - anchor_index
            output[planning_start:] = local_output[local_start:]
            if not np.isfinite(output).all():
                raise FloatingPointError("time-axis output is non-finite")

            durations = np.full(segment_count, self.config.dt_ref, dtype=np.float64)
            durations[anchor_index : anchor_index + len(speeds)] = 1.0 / speeds
            optimized_durations = 1.0 / speeds
            if np.any(optimized_durations < self.config.dt_min - 1e-9) or np.any(
                optimized_durations > self.config.dt_max + 1e-9
            ):
                raise FloatingPointError("time-axis duration is outside the configured bounds")

            cast_output = output.astype(output_dtype, copy=False)
            # Restore the committed bytes after all interpolation and casting.
            cast_output[:planning_start] = input_array[:planning_start]
            return TimeAxisPlan(
                cast_output,
                durations,
                False,
                reference_segment_durations=reference_diagnostics,
                speed_factors=speed_factor_diagnostics,
            )
        except Exception as exc:  # fail closed at the control boundary
            if not self._warned_fallback:
                logger.warning("Time-axis planner fell back to the model chunk: %s", exc)
                self._warned_fallback = True
            return self._fallback(
                original,
                output_dtype,
                segment_count,
                str(exc),
                reference_segment_durations=reference_diagnostics,
                speed_factors=speed_factor_diagnostics,
            )

    def _fallback(
        self,
        original: np.ndarray,
        output_dtype: np.dtype,
        segment_count: int,
        reason: str,
        *,
        reference_segment_durations: np.ndarray | None = None,
        speed_factors: np.ndarray | None = None,
    ) -> TimeAxisPlan:
        return TimeAxisPlan(
            original.astype(output_dtype, copy=True),
            np.full(segment_count, self.config.dt_ref),
            True,
            reason,
            reference_segment_durations,
            speed_factors,
        )

    @staticmethod
    def _slice_path_embedding(value, *, anchor_index: int, action_count: int):
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] in (action_count, action_count - 1):
            return value[anchor_index:]
        return value

    def _reference_profile(
        self,
        waypoints: np.ndarray,
        *,
        state_embeddings,
        phase_embeddings,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        segment_count = len(waypoints) - 1
        if self.speed_adapter is None:
            return np.full(segment_count, self.config.dt_ref, dtype=np.float64), None
        speed_factors_tensor = self.speed_adapter.predict_path(
            waypoints,
            state_embeddings=state_embeddings,
            phase_embeddings=phase_embeddings,
        )
        speed_factors = np.asarray(
            speed_factors_tensor.detach().to(device="cpu").float().numpy(),
            dtype=np.float64,
        )
        if speed_factors.shape != (segment_count,) or not np.isfinite(speed_factors).all():
            raise ValueError(
                "speed adapter returned an invalid beta trajectory: "
                f"expected shape {(segment_count,)}, got {speed_factors.shape}"
            )
        reference_tensor = beta_to_segment_dt_ref(
            speed_factors_tensor,
            base_dt_s=self.config.dt_ref,
            min_dt_s=self.config.dt_min,
            max_dt_s=self.config.dt_max,
        )
        reference_durations = np.asarray(
            reference_tensor.detach().to(device="cpu").float().numpy(),
            dtype=np.float64,
        )
        return reference_durations, speed_factors

    @staticmethod
    def _resolve_start_index(
        action_count: int,
        *,
        start_index: int,
        committed_prefix_steps: int | None,
    ) -> int:
        if isinstance(start_index, bool) or not isinstance(start_index, (int, np.integer)):
            raise ValueError("start_index must be an integer")
        if committed_prefix_steps is not None:
            if isinstance(committed_prefix_steps, bool) or not isinstance(
                committed_prefix_steps, (int, np.integer)
            ):
                raise ValueError("committed_prefix_steps must be an integer")
            if start_index != 0 and start_index != committed_prefix_steps:
                raise ValueError("start_index and committed_prefix_steps disagree")
            start_index = int(committed_prefix_steps)
        if start_index < 0 or start_index > action_count:
            raise ValueError(f"planning start index must be in [0, {action_count}]")
        return int(start_index)

    def _resolve_dims(self, action_dim: int) -> np.ndarray:
        if not self.config.optimization_dims:
            return np.arange(action_dim, dtype=np.int64)
        dims = np.asarray(self.config.optimization_dims, dtype=np.int64)
        if (dims >= action_dim).any():
            raise ValueError(f"optimization_dims contains an index >= action_dim={action_dim}")
        if len(np.unique(dims)) != len(dims):
            raise ValueError("optimization_dims must not contain duplicate indices")
        return dims

    @staticmethod
    def _resolve_limits(
        value: float | list[float] | None,
        *,
        action_dim: int,
        dims: np.ndarray,
        name: str,
    ) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return np.full(len(dims), float(value), dtype=np.float64)
        limits = np.asarray(value, dtype=np.float64)
        if limits.ndim != 1:
            raise ValueError(f"{name} must be a scalar or one-dimensional list")
        if len(limits) == action_dim:
            limits = limits[dims]
        elif len(limits) != len(dims):
            raise ValueError(
                f"{name} must have action_dim={action_dim} or optimization_dim={len(dims)} values"
            )
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError(f"{name} must contain finite positive values")
        return limits

    def _solve_rolling(
        self,
        waypoints: np.ndarray,
        *,
        reference_durations: np.ndarray,
        velocity_limits: np.ndarray | None,
        acceleration_limits: np.ndarray | None,
    ) -> np.ndarray:
        deltas = np.diff(waypoints, axis=0)
        segment_count = len(deltas)
        if segment_count == 0:
            return np.empty(0, dtype=np.float64)

        speeds = np.empty(segment_count, dtype=np.float64)
        cursor = 0
        while cursor < segment_count:
            window_end = min(segment_count, cursor + int(self.config.horizon))
            window_speeds = self._solve_qp(
                deltas[cursor:window_end],
                reference_inverse_speeds=1.0 / reference_durations[cursor:window_end],
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
                previous_delta=deltas[cursor - 1] if cursor > 0 else None,
                previous_inverse_duration=float(speeds[cursor - 1]) if cursor > 0 else None,
            )
            commit_count = min(int(self.config.stride), len(window_speeds))
            if commit_count < 1:
                raise RuntimeError("time-axis rolling solve made no progress")
            speeds[cursor : cursor + commit_count] = window_speeds[:commit_count]
            cursor += commit_count
        return speeds

    def _solve_qp(
        self,
        deltas: np.ndarray,
        *,
        reference_inverse_speeds: np.ndarray,
        velocity_limits: np.ndarray | None,
        acceleration_limits: np.ndarray | None,
        previous_delta: np.ndarray | None = None,
        previous_inverse_duration: float | None = None,
    ) -> np.ndarray:
        osqp, sparse = _load_osqp_modules()
        segment_count = len(deltas)
        if segment_count == 0:
            return np.empty(0, dtype=np.float64)

        has_previous_boundary = previous_delta is not None or previous_inverse_duration is not None
        if has_previous_boundary and (previous_delta is None or previous_inverse_duration is None):
            raise ValueError(
                "previous delta and inverse duration must be provided together for a rolling boundary"
            )
        previous_velocity: np.ndarray | None = None
        if previous_delta is not None and previous_inverse_duration is not None:
            previous_delta = np.asarray(previous_delta, dtype=np.float64)
            if previous_delta.shape != (deltas.shape[1],) or not np.isfinite(previous_delta).all():
                raise ValueError("previous delta must be a finite vector matching the action dimension")
            previous_inverse_duration = float(previous_inverse_duration)
            if not math.isfinite(previous_inverse_duration) or previous_inverse_duration <= 0.0:
                raise ValueError("previous inverse duration must be finite and positive")
            previous_velocity = previous_delta * previous_inverse_duration

        norms = np.linalg.norm(deltas, axis=1)
        s_ref = np.asarray(reference_inverse_speeds, dtype=np.float64)
        if s_ref.shape != (segment_count,) or not np.isfinite(s_ref).all() or np.any(s_ref <= 0):
            raise ValueError("reference inverse durations must be finite and positive per segment")
        lower = np.full(segment_count, 1.0 / self.config.dt_max, dtype=np.float64)
        upper = np.full(segment_count, 1.0 / self.config.dt_min, dtype=np.float64)
        if velocity_limits is not None:
            moving = np.abs(deltas) > 1e-12
            coordinate_bounds = np.full_like(deltas, np.inf, dtype=np.float64)
            np.divide(
                velocity_limits[None, :],
                np.abs(deltas),
                out=coordinate_bounds,
                where=moving,
            )
            upper = np.minimum(upper, coordinate_bounds.min(axis=1))
        if np.any(upper < lower - 1e-10):
            raise ValueError("velocity limit is infeasible within the configured dt bounds")
        upper = np.maximum(upper, lower)

        scale_time = float(np.mean(s_ref**2)) + 1e-6
        scale_acc = float(np.mean((norms * s_ref) ** 2)) + 1e-6
        time_weight = (self.config.lambda_time + self.config.lambda_velocity) / scale_time
        acc_weight = self.config.lambda_acc / scale_acc

        # OSQP minimizes 0.5*x.T*P*x + q.T*x.
        p_matrix = 2.0 * time_weight * np.eye(segment_count, dtype=np.float64)
        q = -2.0 * time_weight * s_ref
        if previous_velocity is not None:
            first_delta = deltas[0]
            p_matrix[0, 0] += 2.0 * acc_weight * float(first_delta @ first_delta)
            q[0] -= 2.0 * acc_weight * float(first_delta @ previous_velocity)
        for index in range(segment_count - 1):
            left = deltas[index]
            right = deltas[index + 1]
            p_matrix[index, index] += 2.0 * acc_weight * float(left @ left)
            p_matrix[index + 1, index + 1] += 2.0 * acc_weight * float(right @ right)
            coupling = -2.0 * acc_weight * float(left @ right)
            p_matrix[index, index + 1] += coupling
            p_matrix[index + 1, index] += coupling

        constraint_blocks = [sparse.eye(segment_count, format="csc")]
        lower_blocks = [lower]
        upper_blocks = [upper]
        if acceleration_limits is not None and (segment_count > 1 or previous_velocity is not None):
            # This is the linear, per-coordinate acceleration proxy used by
            # the previous planner: bound adjacent segment-velocity changes
            # over one nominal control period.
            rows = []
            acceleration_lower: list[float] = []
            acceleration_upper: list[float] = []
            if previous_velocity is not None:
                for dim in range(deltas.shape[1]):
                    row = np.zeros(segment_count, dtype=np.float64)
                    row[0] = deltas[0, dim]
                    rows.append(row)
                    allowed_delta = acceleration_limits[dim] * self.config.dt_ref
                    acceleration_lower.append(previous_velocity[dim] - allowed_delta)
                    acceleration_upper.append(previous_velocity[dim] + allowed_delta)
            for index in range(segment_count - 1):
                for dim in range(deltas.shape[1]):
                    row = np.zeros(segment_count, dtype=np.float64)
                    row[index] = -deltas[index, dim]
                    row[index + 1] = deltas[index + 1, dim]
                    rows.append(row)
                    allowed_delta = acceleration_limits[dim] * self.config.dt_ref
                    acceleration_lower.append(-allowed_delta)
                    acceleration_upper.append(allowed_delta)
            acceleration_matrix = sparse.csc_matrix(np.asarray(rows))
            constraint_blocks.append(acceleration_matrix)
            lower_blocks.append(np.asarray(acceleration_lower, dtype=np.float64))
            upper_blocks.append(np.asarray(acceleration_upper, dtype=np.float64))

        constraints = sparse.vstack(constraint_blocks, format="csc")
        constraint_lower = np.concatenate(lower_blocks)
        constraint_upper = np.concatenate(upper_blocks)
        problem = osqp.OSQP()
        problem.setup(
            P=sparse.csc_matrix(np.triu(p_matrix)),
            q=q,
            A=constraints,
            l=constraint_lower,
            u=constraint_upper,
            verbose=False,
            max_iter=int(self.config.max_iterations),
            eps_abs=1e-6,
            eps_rel=1e-6,
            polishing=False,
        )
        result = problem.solve()
        status = str(getattr(result.info, "status", "")).lower()
        if not status.startswith("solved"):
            raise RuntimeError(f"OSQP failed with status {status or 'unknown'}")
        solution = np.asarray(result.x, dtype=np.float64)
        if solution.shape != (segment_count,) or not np.isfinite(solution).all():
            raise FloatingPointError("OSQP returned an invalid inverse-duration trajectory")
        if np.any(solution < lower - 1e-6) or np.any(solution > upper + 1e-6):
            raise FloatingPointError("OSQP returned an inverse duration outside its bounds")
        if acceleration_limits is not None:
            velocities = deltas * solution[:, None]
            velocity_deltas = np.diff(velocities, axis=0)
            if previous_velocity is not None:
                velocity_deltas = np.vstack((velocities[0] - previous_velocity, velocity_deltas))
            allowed = acceleration_limits * self.config.dt_ref
            if len(velocity_deltas) and np.any(np.abs(velocity_deltas) > allowed[None, :] + 1e-5):
                raise FloatingPointError("OSQP returned an acceleration trajectory outside its bounds")
        return np.clip(solution, lower, upper)

    def _resample(self, waypoints: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        durations = 1.0 / speeds
        source_t = np.concatenate(([0.0], np.cumsum(durations)))
        target_t = np.arange(len(waypoints), dtype=np.float64) * self.config.dt_ref
        output = np.stack(
            [np.interp(target_t, source_t, waypoints[:, dim]) for dim in range(waypoints.shape[1])],
            axis=1,
        )
        output[0] = waypoints[0]
        return output
