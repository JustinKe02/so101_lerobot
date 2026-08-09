# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure NumPy smooth execution core for timestamped realtime trajectories.

The executor deliberately has no robot or motor-bus dependency. A deployment
loop owns the heartbeat and sends the returned array to its actuator. Keeping
that boundary explicit makes interpolation, delay compensation, and safety
limits deterministic and testable without hardware.
"""

from __future__ import annotations

import math
import threading
import time
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_timestamp(timestamp: float, *, name: str = "timestamp") -> float:
    if isinstance(timestamp, bool):
        raise TypeError(f"{name} must be a finite number")
    value = float(timestamp)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _vector(value: ArrayLike, size: int, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    array = np.full(size, float(array), dtype=np.float64) if array.ndim == 0 else array.reshape(-1).copy()
    if array.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_limit(value: ArrayLike, size: int, *, name: str) -> NDArray[np.float64]:
    limit = _vector(value, size, name=name)
    if np.any(limit <= 0.0):
        raise ValueError(f"{name} values must be positive")
    limit.setflags(write=False)
    return limit


def _nonnegative_limit(value: ArrayLike, size: int, *, name: str) -> NDArray[np.float64]:
    limit = _vector(value, size, name=name)
    if np.any(limit < 0.0):
        raise ValueError(f"{name} values must be non-negative")
    limit.setflags(write=False)
    return limit


@dataclass(frozen=True)
class TimedWaypoint:
    """One immutable action waypoint in the monotonic clock domain."""

    timestamp: float
    action: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _finite_timestamp(self.timestamp))
        action = np.asarray(self.action, dtype=np.float64).reshape(-1).copy()
        if not np.isfinite(action).all():
            raise ValueError("waypoint action must contain only finite values")
        action.setflags(write=False)
        object.__setattr__(self, "action", action)


@dataclass(frozen=True)
class RealtimeExecutorConfig:
    """Configuration for :class:`RealtimeExecutor`.

    Limits use the same units as the waypoint actions. ``forward_lead_s``
    defaults to ``actuator_tau_s + command_delay_s`` per joint. This is the
    first-order preview approximation for a delayed actuator,
    ``q + (tau + delay) * q_dot``.
    """

    action_dim: int
    heartbeat_dt_s: float
    max_velocity: ArrayLike
    max_acceleration: ArrayLike
    actuator_tau_s: ArrayLike = 0.15
    command_delay_s: ArrayLike = 0.0
    enable_forward_tracking: bool = True
    forward_lead_s: ArrayLike | None = None
    forward_feedback_gain: float = 0.0
    savgol_window_length: int = 1
    savgol_polyorder: int = 2
    max_waypoints: int = 4096
    max_command_history: int = 512

    def __post_init__(self) -> None:
        if isinstance(self.action_dim, bool) or not isinstance(self.action_dim, int) or self.action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        heartbeat_dt_s = float(self.heartbeat_dt_s)
        if not math.isfinite(heartbeat_dt_s) or heartbeat_dt_s <= 0.0:
            raise ValueError("heartbeat_dt_s must be finite and positive")
        object.__setattr__(self, "heartbeat_dt_s", heartbeat_dt_s)

        tau = _positive_limit(self.actuator_tau_s, self.action_dim, name="actuator_tau_s")
        delay = _nonnegative_limit(self.command_delay_s, self.action_dim, name="command_delay_s")
        lead = (
            tau + delay
            if self.forward_lead_s is None
            else _nonnegative_limit(self.forward_lead_s, self.action_dim, name="forward_lead_s")
        )
        lead.setflags(write=False)
        object.__setattr__(self, "actuator_tau_s", tau)
        object.__setattr__(self, "command_delay_s", delay)
        object.__setattr__(self, "forward_lead_s", lead)

        gain = float(self.forward_feedback_gain)
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("forward_feedback_gain must be finite and non-negative")
        object.__setattr__(self, "forward_feedback_gain", gain)
        object.__setattr__(
            self,
            "max_velocity",
            _positive_limit(self.max_velocity, self.action_dim, name="max_velocity"),
        )
        object.__setattr__(
            self,
            "max_acceleration",
            _positive_limit(self.max_acceleration, self.action_dim, name="max_acceleration"),
        )

        window = self.savgol_window_length
        order = self.savgol_polyorder
        if isinstance(window, bool) or not isinstance(window, int) or window < 1 or window % 2 == 0:
            raise ValueError("savgol_window_length must be one or an odd positive integer")
        if isinstance(order, bool) or not isinstance(order, int) or order < 0:
            raise ValueError("savgol_polyorder must be a non-negative integer")
        if window > 1 and order >= window:
            raise ValueError("savgol_polyorder must be smaller than savgol_window_length")
        if isinstance(self.max_waypoints, bool) or self.max_waypoints < 2:
            raise ValueError("max_waypoints must be at least two")
        if isinstance(self.max_command_history, bool) or self.max_command_history < 2:
            raise ValueError("max_command_history must be at least two")


@dataclass(frozen=True)
class _CommandStamped:
    timestamp: float
    command: NDArray[np.float64]


@dataclass(frozen=True)
class ExecutorCommandPreview:
    """One command predicted by a non-mutating executor preview."""

    timestamp: float
    command: NDArray[np.float64]
    reference: NDArray[np.float64]
    underrun: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _finite_timestamp(self.timestamp))
        command = np.asarray(self.command, dtype=np.float64).reshape(-1).copy()
        reference = np.asarray(self.reference, dtype=np.float64).reshape(-1).copy()
        if (
            command.shape != reference.shape
            or not np.isfinite(command).all()
            or not np.isfinite(reference).all()
        ):
            raise ValueError("preview command and reference must be finite vectors with matching shapes")
        command.setflags(write=False)
        reference.setflags(write=False)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "reference", reference)


@dataclass(frozen=True)
class _ReplaySnapshot:
    commands: tuple[_CommandStamped, ...]
    anchor_timestamp: float | None
    anchor_position: NDArray[np.float64] | None
    last_observation_timestamp: float | None


class FirstOrderReplayEstimator:
    """Estimate current joint state by replaying commands from an observation.

    The actuator model is ``x_dot = (command - x) / tau``. Observations and
    commands must each arrive in non-decreasing monotonic timestamp order.
    """

    def __init__(
        self,
        action_dim: int,
        tau_s: ArrayLike,
        max_command_history: int = 512,
        command_delay_s: ArrayLike = 0.0,
    ) -> None:
        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if isinstance(max_command_history, bool) or max_command_history < 2:
            raise ValueError("max_command_history must be at least two")
        self.action_dim = action_dim
        self.tau_s = _positive_limit(tau_s, action_dim, name="tau_s")
        self.command_delay_s = _nonnegative_limit(
            command_delay_s,
            action_dim,
            name="command_delay_s",
        )
        self._commands: deque[_CommandStamped] = deque(maxlen=max_command_history)
        self._anchor_timestamp: float | None = None
        self._anchor_position: NDArray[np.float64] | None = None
        self._last_observation_timestamp: float | None = None
        self._lock = threading.RLock()

    @property
    def initialized(self) -> bool:
        with self._lock:
            return self._anchor_position is not None

    def reset(self, position: ArrayLike, timestamp: float) -> None:
        position_array = _vector(position, self.action_dim, name="position")
        stamp = _finite_timestamp(timestamp)
        with self._lock:
            self._commands.clear()
            self._anchor_timestamp = stamp
            self._anchor_position = position_array
            self._last_observation_timestamp = stamp

    def push_command(self, command: ArrayLike, timestamp: float) -> None:
        command_array = _vector(command, self.action_dim, name="command")
        stamp = _finite_timestamp(timestamp)
        with self._lock:
            if self._commands and stamp < self._commands[-1].timestamp:
                raise ValueError("command timestamps must be monotonic")
            self._commands.append(_CommandStamped(stamp, command_array))

    def push_observation(self, position: ArrayLike, timestamp: float) -> None:
        position_array = _vector(position, self.action_dim, name="position")
        stamp = _finite_timestamp(timestamp)
        with self._lock:
            if self._last_observation_timestamp is not None and stamp < self._last_observation_timestamp:
                raise ValueError("observation timestamps must be monotonic")
            self._anchor_timestamp = stamp
            self._anchor_position = position_array
            self._last_observation_timestamp = stamp

    def _snapshot(self) -> _ReplaySnapshot:
        """Capture estimator state while an owning executor previews commands."""

        with self._lock:
            return _ReplaySnapshot(
                commands=tuple(
                    _CommandStamped(record.timestamp, record.command.copy()) for record in self._commands
                ),
                anchor_timestamp=self._anchor_timestamp,
                anchor_position=(None if self._anchor_position is None else self._anchor_position.copy()),
                last_observation_timestamp=self._last_observation_timestamp,
            )

    def _restore(self, snapshot: _ReplaySnapshot) -> None:
        with self._lock:
            self._commands = deque(
                (_CommandStamped(record.timestamp, record.command.copy()) for record in snapshot.commands),
                maxlen=self._commands.maxlen,
            )
            self._anchor_timestamp = snapshot.anchor_timestamp
            self._anchor_position = (
                None if snapshot.anchor_position is None else snapshot.anchor_position.copy()
            )
            self._last_observation_timestamp = snapshot.last_observation_timestamp

    def estimate(self, timestamp: float) -> NDArray[np.float64]:
        stamp = _finite_timestamp(timestamp)
        with self._lock:
            if self._anchor_position is None or self._anchor_timestamp is None:
                raise RuntimeError("replay estimator must be reset or observed before estimate()")
            if stamp < self._anchor_timestamp:
                raise ValueError("estimate timestamp precedes the latest observation")
            commands = tuple(self._commands)
            anchor_timestamp = self._anchor_timestamp
            position = self._anchor_position.copy()

        for joint_index in range(self.action_dim):
            position[joint_index] = self._estimate_joint(
                joint_index,
                float(position[joint_index]),
                commands,
                anchor_timestamp,
                stamp,
            )
        return position

    def _estimate_joint(
        self,
        joint_index: int,
        position: float,
        commands: tuple[_CommandStamped, ...],
        anchor_timestamp: float,
        estimate_timestamp: float,
    ) -> float:
        delay_s = float(self.command_delay_s[joint_index])
        tau_s = float(self.tau_s[joint_index])
        active_command = position
        for record in commands:
            effective_timestamp = record.timestamp + delay_s
            if effective_timestamp <= anchor_timestamp:
                active_command = float(record.command[joint_index])
            else:
                break

        replay_timestamp = anchor_timestamp
        for record in commands:
            effective_timestamp = record.timestamp + delay_s
            if effective_timestamp <= anchor_timestamp:
                continue
            if effective_timestamp > estimate_timestamp:
                break
            position = self._advance_scalar(
                position,
                active_command,
                effective_timestamp - replay_timestamp,
                tau_s,
            )
            replay_timestamp = effective_timestamp
            active_command = float(record.command[joint_index])
        return self._advance_scalar(
            position,
            active_command,
            estimate_timestamp - replay_timestamp,
            tau_s,
        )

    @staticmethod
    def _advance_scalar(position: float, command: float, duration_s: float, tau_s: float) -> float:
        if duration_s <= 0.0:
            return position
        decay = math.exp(-duration_s / tau_s)
        return command + (position - command) * decay

    def _advance(
        self,
        position: NDArray[np.float64],
        command: NDArray[np.float64],
        duration_s: float,
    ) -> NDArray[np.float64]:
        if duration_s <= 0.0:
            return position.copy()
        decay = np.exp(-duration_s / self.tau_s)
        return command + (position - command) * decay


class ForwardTracker:
    """First-order inverse feed-forward with optional estimated-state feedback."""

    def __init__(self, action_dim: int, lead_s: ArrayLike, feedback_gain: float = 0.0) -> None:
        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if not math.isfinite(feedback_gain) or feedback_gain < 0.0:
            raise ValueError("feedback_gain must be finite and non-negative")
        self.action_dim = action_dim
        self.lead_s = _nonnegative_limit(lead_s, action_dim, name="lead_s")
        self.feedback_gain = float(feedback_gain)

    def compensate(
        self,
        target: ArrayLike,
        target_velocity: ArrayLike,
        estimated_position: ArrayLike,
    ) -> NDArray[np.float64]:
        target_array = _vector(target, self.action_dim, name="target")
        velocity_array = _vector(target_velocity, self.action_dim, name="target_velocity")
        estimate_array = _vector(estimated_position, self.action_dim, name="estimated_position")
        return (
            target_array + self.lead_s * velocity_array + self.feedback_gain * (target_array - estimate_array)
        )


@dataclass(frozen=True)
class _ExecutorSnapshot:
    waypoints: tuple[TimedWaypoint, ...]
    last_enqueued_timestamp: float | None
    origin_timestamp: float | None
    next_heartbeat_timestamp: float | None
    last_heartbeat_timestamp: float | None
    tick_index: int
    last_command: NDArray[np.float64] | None
    command_velocity: NDArray[np.float64]
    last_reference: NDArray[np.float64] | None
    underrun_count: int
    last_tick_was_underrun: bool
    replay: _ReplaySnapshot


def _savgol_weights(window_length: int, polyorder: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if window_length == 1:
        return np.ones(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    half = window_length // 2
    positions = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(positions, polyorder + 1, increasing=True)
    coefficients = np.linalg.pinv(design)
    value_weights = coefficients[0]
    derivative_weights = coefficients[1] if polyorder >= 1 else np.zeros(window_length)
    return value_weights, derivative_weights


class RealtimeExecutor:
    """Thread-safe fixed-heartbeat executor for timestamped action waypoints.

    ``heartbeat()`` advances one logical tick every call; it does not sleep and
    never accesses hardware. The first tick is the timestamp passed to
    ``reset`` and subsequent ticks are exactly ``heartbeat_dt_s`` apart.
    """

    def __init__(self, config: RealtimeExecutorConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._waypoints: deque[TimedWaypoint] = deque()
        self._last_enqueued_timestamp: float | None = None
        self._origin_timestamp: float | None = None
        self._next_heartbeat_timestamp: float | None = None
        self._last_heartbeat_timestamp: float | None = None
        self._tick_index = 0
        self._last_command: NDArray[np.float64] | None = None
        self._command_velocity = np.zeros(config.action_dim, dtype=np.float64)
        self._last_reference: NDArray[np.float64] | None = None
        self._underrun_count = 0
        self._last_tick_was_underrun = False
        self._value_weights, self._derivative_weights = _savgol_weights(
            config.savgol_window_length,
            config.savgol_polyorder,
        )
        self._replay = FirstOrderReplayEstimator(
            config.action_dim,
            config.actuator_tau_s,
            config.max_command_history,
            config.command_delay_s,
        )
        self._tracker = ForwardTracker(
            config.action_dim,
            config.forward_lead_s,
            config.forward_feedback_gain,
        )

    @property
    def initialized(self) -> bool:
        with self._lock:
            return self._last_command is not None

    @property
    def next_heartbeat_timestamp(self) -> float | None:
        with self._lock:
            return self._next_heartbeat_timestamp

    @property
    def last_heartbeat_timestamp(self) -> float | None:
        with self._lock:
            return self._last_heartbeat_timestamp

    @property
    def last_command(self) -> NDArray[np.float64] | None:
        with self._lock:
            return None if self._last_command is None else self._last_command.copy()

    @property
    def last_reference(self) -> NDArray[np.float64] | None:
        with self._lock:
            return None if self._last_reference is None else self._last_reference.copy()

    @property
    def pending_waypoint_count(self) -> int:
        with self._lock:
            if self._next_heartbeat_timestamp is None:
                return len(self._waypoints)
            return sum(waypoint.timestamp >= self._next_heartbeat_timestamp for waypoint in self._waypoints)

    @property
    def underrun_count(self) -> int:
        with self._lock:
            return self._underrun_count

    @property
    def last_tick_was_underrun(self) -> bool:
        with self._lock:
            return self._last_tick_was_underrun

    def reset(
        self,
        initial_position: ArrayLike,
        timestamp: float | None = None,
        *,
        observation_timestamp: float | None = None,
        clear_waypoints: bool = True,
    ) -> None:
        """Reset command, estimator, heartbeat, and optionally queued waypoints."""

        position = _vector(initial_position, self.config.action_dim, name="initial_position")
        stamp = time.monotonic_ns() * 1e-9 if timestamp is None else _finite_timestamp(timestamp)
        observation_stamp = (
            stamp
            if observation_timestamp is None
            else _finite_timestamp(observation_timestamp, name="observation_timestamp")
        )
        if observation_stamp > stamp:
            raise ValueError("observation_timestamp cannot be later than the heartbeat timestamp")
        with self._lock:
            if clear_waypoints:
                self._waypoints.clear()
                self._last_enqueued_timestamp = None
            self._origin_timestamp = stamp
            self._next_heartbeat_timestamp = stamp
            self._last_heartbeat_timestamp = None
            self._tick_index = 0
            self._last_command = position.copy()
            self._command_velocity.fill(0.0)
            self._last_reference = position.copy()
            self._underrun_count = 0
            self._last_tick_was_underrun = False
            self._replay.reset(position, observation_stamp)

    def clear_waypoints(self) -> None:
        """Atomically clear pending waypoints without changing executor state."""

        with self._lock:
            self._waypoints.clear()
            self._last_enqueued_timestamp = None

    def rebase_next_heartbeat(self, timestamp: float) -> None:
        """Move the next logical tick forward after a wall-clock deadline miss.

        A control loop must not burst commands to catch up after an overrun. The
        loop skips elapsed deadlines and rebases the next executor tick instead;
        command and estimator state are preserved, and the caller replaces the
        future waypoint plan before emitting the rebased tick.
        """

        stamp = _finite_timestamp(timestamp)
        with self._lock:
            if self._last_command is None:
                raise RuntimeError("executor must be reset before rebasing its heartbeat")
            if self._last_heartbeat_timestamp is not None and stamp <= self._last_heartbeat_timestamp:
                raise ValueError("rebased heartbeat must follow the last emitted heartbeat")
            if self._next_heartbeat_timestamp is not None and stamp < self._next_heartbeat_timestamp:
                raise ValueError("rebased heartbeat cannot move the schedule backwards")
            self._origin_timestamp = stamp
            self._next_heartbeat_timestamp = stamp
            self._tick_index = 0

    def enqueue_waypoints(self, timestamps: ArrayLike, actions: ArrayLike) -> None:
        """Atomically append a strictly time-ordered waypoint batch."""

        new_waypoints = self._validated_waypoints(timestamps, actions)
        if not new_waypoints:
            return
        with self._lock:
            if (
                self._last_enqueued_timestamp is not None
                and new_waypoints[0].timestamp <= self._last_enqueued_timestamp
            ):
                raise ValueError("waypoint timestamps must increase across appended batches")
            if len(self._waypoints) + len(new_waypoints) > self.config.max_waypoints:
                raise OverflowError("waypoint queue capacity exceeded")
            self._waypoints.extend(new_waypoints)
            self._last_enqueued_timestamp = new_waypoints[-1].timestamp

    def replace_waypoints_from(
        self,
        replace_timestamp: float,
        timestamps: ArrayLike,
        actions: ArrayLike,
    ) -> None:
        """Replace the current/future trajectory while retaining smoothing history.

        RTC calls this once per heartbeat after atomically popping the action for
        that tick. Waypoints older than ``replace_timestamp`` remain available to
        the Savitzky-Golay window; all current and future waypoints are replaced
        by the latest queue snapshot in one lock acquisition.
        """

        replace_stamp = _finite_timestamp(replace_timestamp, name="replace_timestamp")
        new_waypoints = self._validated_waypoints(timestamps, actions)
        if new_waypoints and new_waypoints[0].timestamp < replace_stamp:
            raise ValueError("replacement waypoints cannot precede replace_timestamp")
        with self._lock:
            self._replace_waypoints_from_locked(replace_stamp, new_waypoints)

    def _replace_waypoints_from_locked(
        self,
        replace_stamp: float,
        new_waypoints: tuple[TimedWaypoint, ...],
    ) -> None:
        retained = [waypoint for waypoint in self._waypoints if waypoint.timestamp < replace_stamp]
        if retained and new_waypoints and new_waypoints[0].timestamp <= retained[-1].timestamp:
            raise ValueError("replacement waypoints must follow retained history")
        if len(retained) + len(new_waypoints) > self.config.max_waypoints:
            raise OverflowError("waypoint queue capacity exceeded")
        self._waypoints = deque([*retained, *new_waypoints])
        self._last_enqueued_timestamp = self._waypoints[-1].timestamp if self._waypoints else None

    def control_step(
        self,
        replace_timestamp: float,
        timestamps: ArrayLike,
        actions: ArrayLike,
        observation: ArrayLike | None = None,
        *,
        observation_timestamp: float | None = None,
    ) -> NDArray[np.float64]:
        """Atomically replace the future plan and emit its first heartbeat.

        Keeping replacement and heartbeat under one lock ensures a concurrent
        prefill preview observes either the complete old plan or the complete
        new command state, never a half-updated executor.
        """

        replace_stamp = _finite_timestamp(replace_timestamp, name="replace_timestamp")
        new_waypoints = self._validated_waypoints(timestamps, actions)
        if new_waypoints and new_waypoints[0].timestamp < replace_stamp:
            raise ValueError("replacement waypoints cannot precede replace_timestamp")
        with self._lock:
            self._replace_waypoints_from_locked(replace_stamp, new_waypoints)
            return self._heartbeat_locked(
                observation,
                observation_timestamp=observation_timestamp,
            )

    def _validated_waypoints(
        self,
        timestamps: ArrayLike,
        actions: ArrayLike,
    ) -> tuple[TimedWaypoint, ...]:
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
        action_array = np.asarray(actions, dtype=np.float64)
        if timestamp_array.ndim != 1:
            raise ValueError("timestamps must be a one-dimensional array")
        if action_array.ndim != 2 or action_array.shape[1] != self.config.action_dim:
            raise ValueError(f"actions must have shape (T, {self.config.action_dim})")
        if action_array.shape[0] != timestamp_array.shape[0]:
            raise ValueError("timestamps and actions must contain the same number of rows")
        if not np.isfinite(timestamp_array).all() or np.any(timestamp_array < 0.0):
            raise ValueError("timestamps must be finite and non-negative")
        if not np.isfinite(action_array).all():
            raise ValueError("actions must contain only finite values")
        if len(timestamp_array) == 0:
            return ()
        if np.any(np.diff(timestamp_array) <= 0.0):
            raise ValueError("waypoint timestamps must be strictly increasing")

        return tuple(
            TimedWaypoint(float(stamp), action)
            for stamp, action in zip(timestamp_array, action_array, strict=True)
        )

    def record_observation(self, position: ArrayLike, timestamp: float) -> None:
        """Add a delayed measured state anchor for subsequent forward replay."""

        self._replay.push_observation(position, timestamp)

    def heartbeat(
        self,
        observation: ArrayLike | None = None,
        *,
        observation_timestamp: float | None = None,
    ) -> NDArray[np.float64]:
        """Run exactly one fixed-period logical control tick and return a command."""

        with self._lock:
            return self._heartbeat_locked(
                observation,
                observation_timestamp=observation_timestamp,
            )

    def _heartbeat_locked(
        self,
        observation: ArrayLike | None = None,
        *,
        observation_timestamp: float | None = None,
    ) -> NDArray[np.float64]:
        if (
            self._last_command is None
            or self._next_heartbeat_timestamp is None
            or self._origin_timestamp is None
        ):
            raise RuntimeError("executor must be reset before heartbeat()")
        tick_timestamp = self._next_heartbeat_timestamp
        if observation is not None:
            observation_stamp = (
                tick_timestamp
                if observation_timestamp is None
                else _finite_timestamp(observation_timestamp, name="observation_timestamp")
            )
            if observation_stamp > tick_timestamp:
                raise ValueError("observation_timestamp cannot be later than the current heartbeat")
            self._replay.push_observation(observation, observation_stamp)
        elif observation_timestamp is not None:
            raise ValueError("observation_timestamp requires an observation")

        if not self._has_reference_at(tick_timestamp):
            command = self._last_command.copy()
            self._command_velocity.fill(0.0)
            self._last_reference = command.copy()
            self._last_tick_was_underrun = True
            self._underrun_count += 1
            if self._waypoints and tick_timestamp > self._waypoints[-1].timestamp:
                self._waypoints.clear()
        else:
            target, target_velocity = self._smoothed_reference(tick_timestamp)
            self._last_reference = target.copy()
            requested = target
            if self.config.enable_forward_tracking:
                estimate = self._replay.estimate(tick_timestamp)
                requested = self._tracker.compensate(target, target_velocity, estimate)
            command = self._limit_command(requested)
            self._last_tick_was_underrun = False

        self._last_command = command.copy()
        self._last_heartbeat_timestamp = tick_timestamp
        self._replay.push_command(command, tick_timestamp)
        self._prune_waypoints(tick_timestamp)
        self._tick_index += 1
        self._next_heartbeat_timestamp = (
            self._origin_timestamp + self._tick_index * self.config.heartbeat_dt_s
        )
        return command.copy()

    def preview(self, steps: int) -> tuple[ExecutorCommandPreview, ...]:
        """Predict future applied commands without changing executor state."""

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("preview steps must be a non-negative integer")
        with self._lock:
            return self._preview_locked(steps)

    def preview_through(self, end_timestamp: float) -> tuple[ExecutorCommandPreview, ...]:
        """Predict every scheduled command through an inclusive timestamp."""

        end = _finite_timestamp(end_timestamp, name="end_timestamp")
        with self._lock:
            if self._next_heartbeat_timestamp is None:
                raise RuntimeError("executor must be reset before previewing commands")
            if end < self._next_heartbeat_timestamp:
                return ()
            steps = (
                int(math.floor((end - self._next_heartbeat_timestamp) / self.config.heartbeat_dt_s + 1e-12))
                + 1
            )
            return self._preview_locked(steps)

    def _preview_locked(self, steps: int) -> tuple[ExecutorCommandPreview, ...]:
        if self._last_command is None or self._next_heartbeat_timestamp is None:
            raise RuntimeError("executor must be reset before previewing commands")
        snapshot = self._snapshot_locked()
        result: list[ExecutorCommandPreview] = []
        try:
            for _ in range(steps):
                timestamp = self._next_heartbeat_timestamp
                command = self._heartbeat_locked()
                reference = self._last_reference
                if timestamp is None or reference is None:
                    raise RuntimeError("executor preview lost its heartbeat state")
                result.append(
                    ExecutorCommandPreview(
                        timestamp=timestamp,
                        command=command,
                        reference=reference,
                        underrun=self._last_tick_was_underrun,
                    )
                )
        finally:
            self._restore_locked(snapshot)
        return tuple(result)

    def _snapshot_locked(self) -> _ExecutorSnapshot:
        return _ExecutorSnapshot(
            waypoints=tuple(self._waypoints),
            last_enqueued_timestamp=self._last_enqueued_timestamp,
            origin_timestamp=self._origin_timestamp,
            next_heartbeat_timestamp=self._next_heartbeat_timestamp,
            last_heartbeat_timestamp=self._last_heartbeat_timestamp,
            tick_index=self._tick_index,
            last_command=None if self._last_command is None else self._last_command.copy(),
            command_velocity=self._command_velocity.copy(),
            last_reference=None if self._last_reference is None else self._last_reference.copy(),
            underrun_count=self._underrun_count,
            last_tick_was_underrun=self._last_tick_was_underrun,
            replay=self._replay._snapshot(),
        )

    def _restore_locked(self, snapshot: _ExecutorSnapshot) -> None:
        self._waypoints = deque(snapshot.waypoints)
        self._last_enqueued_timestamp = snapshot.last_enqueued_timestamp
        self._origin_timestamp = snapshot.origin_timestamp
        self._next_heartbeat_timestamp = snapshot.next_heartbeat_timestamp
        self._last_heartbeat_timestamp = snapshot.last_heartbeat_timestamp
        self._tick_index = snapshot.tick_index
        self._last_command = None if snapshot.last_command is None else snapshot.last_command.copy()
        self._command_velocity = snapshot.command_velocity.copy()
        self._last_reference = None if snapshot.last_reference is None else snapshot.last_reference.copy()
        self._underrun_count = snapshot.underrun_count
        self._last_tick_was_underrun = snapshot.last_tick_was_underrun
        self._replay._restore(snapshot.replay)

    def _has_reference_at(self, timestamp: float) -> bool:
        if not self._waypoints:
            return False
        tolerance = max(1e-12, self.config.heartbeat_dt_s * 1e-9)
        return (
            self._waypoints[0].timestamp - tolerance <= timestamp <= self._waypoints[-1].timestamp + tolerance
        )

    def _smoothed_reference(self, timestamp: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        window = self.config.savgol_window_length
        if window == 1:
            return self._interpolate_reference(timestamp)
        half = window // 2
        samples = np.stack(
            [
                self._interpolate_position(timestamp + offset * self.config.heartbeat_dt_s)
                for offset in range(-half, half + 1)
            ]
        )
        target = self._value_weights @ samples
        velocity = self._derivative_weights @ samples / self.config.heartbeat_dt_s
        if timestamp <= self._waypoints[0].timestamp or timestamp >= self._waypoints[-1].timestamp:
            velocity.fill(0.0)
        return target, velocity

    def _interpolate_reference(self, timestamp: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        position = self._interpolate_position(timestamp)
        if len(self._waypoints) < 2:
            return position, np.zeros(self.config.action_dim, dtype=np.float64)
        waypoints = tuple(self._waypoints)
        times = [waypoint.timestamp for waypoint in waypoints]
        index = bisect_right(times, timestamp)
        if index == 0 or index >= len(waypoints):
            return position, np.zeros(self.config.action_dim, dtype=np.float64)
        left = waypoints[index - 1]
        right = waypoints[index]
        velocity = (right.action - left.action) / (right.timestamp - left.timestamp)
        return position, velocity

    def _interpolate_position(self, timestamp: float) -> NDArray[np.float64]:
        waypoints = tuple(self._waypoints)
        if timestamp <= waypoints[0].timestamp:
            return waypoints[0].action.copy()
        if timestamp >= waypoints[-1].timestamp:
            return waypoints[-1].action.copy()
        times = [waypoint.timestamp for waypoint in waypoints]
        right_index = bisect_right(times, timestamp)
        left = waypoints[right_index - 1]
        right = waypoints[right_index]
        alpha = (timestamp - left.timestamp) / (right.timestamp - left.timestamp)
        return left.action + alpha * (right.action - left.action)

    def _limit_command(self, requested: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._last_command is None:
            raise RuntimeError("executor command state is not initialized")
        dt = self.config.heartbeat_dt_s
        velocity_target = np.clip(
            (requested - self._last_command) / dt,
            -self.config.max_velocity,
            self.config.max_velocity,
        )
        velocity = np.clip(
            velocity_target,
            self._command_velocity - self.config.max_acceleration * dt,
            self._command_velocity + self.config.max_acceleration * dt,
        )
        velocity = np.clip(velocity, -self.config.max_velocity, self.config.max_velocity)
        command = self._last_command + velocity * dt
        self._command_velocity = velocity
        return command

    def _prune_waypoints(self, timestamp: float) -> None:
        half_window = self.config.savgol_window_length // 2
        oldest_required = timestamp - half_window * self.config.heartbeat_dt_s
        while len(self._waypoints) > 2 and self._waypoints[1].timestamp <= oldest_required:
            self._waypoints.popleft()
