"""Timestamped state/action history and deployment trace writing."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from bisect import bisect_right
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

MONOTONIC_CLOCK_DOMAIN = "monotonic"
UNIX_CLOCK_DOMAIN = "unix"
TRACE_TERMINAL_STATUSES = frozenset({"completed", "abnormal"})

_CLOCK_DOMAIN_ALIASES = {
    "monotonic_ns": MONOTONIC_CLOCK_DOMAIN,
    "system_monotonic": MONOTONIC_CLOCK_DOMAIN,
    "time.monotonic": MONOTONIC_CLOCK_DOMAIN,
    "unix_wall": UNIX_CLOCK_DOMAIN,
    "wall": UNIX_CLOCK_DOMAIN,
    "wall_clock": UNIX_CLOCK_DOMAIN,
}
_TIMESTAMP_KEYS = ("timestamp", "state_timestamp", "image_timestamp")
_TIMESTAMP_NS_KEYS = ("timestamp_ns", "monotonic_timestamp_ns")
_ACTION_TIME_METADATA_KEYS = {
    "clock_domain",
    "image_timestamp",
    "monotonic_timestamp_ns",
    "state_timestamp",
    "timestamp",
    "timestamp_ns",
}


def _normalize_clock_domain(clock_domain: str) -> str:
    if not isinstance(clock_domain, str) or not clock_domain.strip():
        raise ValueError("clock_domain must be a non-empty string")
    normalized = clock_domain.strip().lower()
    return _CLOCK_DOMAIN_ALIASES.get(normalized, normalized)


def _seconds_from_ns(timestamp_ns: int | float, *, name: str) -> float:
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, (int, float)):
        raise TypeError(f"{name} must be an integer or float number of nanoseconds")
    timestamp = float(timestamp_ns) * 1e-9
    if not math.isfinite(timestamp):
        raise ValueError(f"{name} must be finite")
    return timestamp


def _coerce_timestamp(timestamp: int | float, *, name: str) -> float:
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        raise TypeError(f"{name} must be an integer or float number of seconds")
    result = float(timestamp)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _now_seconds(clock_domain: str) -> float:
    if clock_domain == MONOTONIC_CLOCK_DOMAIN:
        return time.monotonic_ns() * 1e-9
    if clock_domain == UNIX_CLOCK_DOMAIN:
        return time.time_ns() * 1e-9
    raise ValueError(
        f"clock domain {clock_domain!r} has no local clock source; provide an explicit timestamp"
    )


_TRACE_INLINE_ARRAY_VALUES = 4096


def _jsonable(value: Any) -> Any:
    """Convert tensors/arrays and nested mappings into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if tensor.numel() > _TRACE_INLINE_ARRAY_VALUES:
            finite = tensor.float()[tensor.isfinite()] if tensor.is_floating_point() else None
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "storage": "summary",
                "min": (float(finite.min()) if finite is not None and finite.numel() else None),
                "max": (float(finite.max()) if finite is not None and finite.numel() else None),
                "mean": (float(finite.mean()) if finite is not None and finite.numel() else None),
            }
        value = tensor.tolist()
    elif hasattr(value, "shape") and hasattr(value, "tolist"):
        if int(getattr(value, "size", 0)) > _TRACE_INLINE_ARRAY_VALUES:
            array = value
            return {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "storage": "summary",
                "min": float(array.min()) if np.isfinite(array).any() else None,
                "max": float(array.max()) if np.isfinite(array).any() else None,
                "mean": float(array[np.isfinite(array)].mean()) if np.isfinite(array).any() else None,
            }
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class TimedRecord:
    timestamp: float
    value: dict[str, Any]
    clock_domain: str = MONOTONIC_CLOCK_DOMAIN

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _coerce_timestamp(self.timestamp, name="record timestamp"))
        object.__setattr__(self, "clock_domain", _normalize_clock_domain(self.clock_domain))


class PrefillCapacityError(ValueError):
    """Raised when a committed prefix exceeds the checkpoint's trained capacity."""

    def __init__(self, required_steps: int, max_steps: int) -> None:
        self.required_steps = int(required_steps)
        self.max_steps = int(max_steps)
        super().__init__(
            "committed action prefill exceeds capacity: "
            f"required_steps={self.required_steps}, max_steps={self.max_steps}"
        )


@dataclass(frozen=True)
class CommittedActionPrefill:
    """A clock-safe action prefix committed before an inference can finish.

    ``anchor_timestamp`` is the capture timestamp shifted backwards by the
    calibrated observation delay. ``timestamps`` is the control-rate grid
    starting at that anchor. ``source_timestamps`` identifies the actual or
    queued command held at every grid point, which makes alignment decisions
    auditable in deployment traces.
    """

    clock_domain: str
    image_capture_timestamp: float
    calibrated_image_delay_s: float
    anchor_timestamp: float
    request_timestamp: float
    predicted_completion_timestamp: float
    timestamps: tuple[float, ...]
    actions: tuple[dict[str, Any], ...]
    source_timestamps: tuple[float, ...]
    required_steps: int
    max_steps: int
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.actions)

    def as_records(self) -> list[TimedRecord]:
        return [
            TimedRecord(timestamp, dict(action), self.clock_domain)
            for timestamp, action in zip(self.timestamps, self.actions, strict=True)
        ]


class DelayAlignedTrajectory:
    """Bounded history for delayed image/state and action alignment.

    Numeric state fields are linearly interpolated when both surrounding
    samples are available.  Non-numeric fields use the latest sample.  This
    keeps the component useful for SO-101 dict observations without imposing a
    fixed AIRBOT vector layout.
    """

    def __init__(self, maxlen: int = 512, *, clock_domain: str = MONOTONIC_CLOCK_DOMAIN) -> None:
        if maxlen < 2:
            raise ValueError("trajectory maxlen must be at least 2")
        self.clock_domain = _normalize_clock_domain(clock_domain)
        self._states: deque[TimedRecord] = deque(maxlen=maxlen)
        self._actions: deque[TimedRecord] = deque(maxlen=maxlen)
        self._lock = threading.RLock()

    def _timestamp_and_domain(
        self,
        record: Mapping[str, Any] | None,
        *,
        timestamp: float | None,
        clock_domain: str | None,
    ) -> tuple[float, str]:
        record_domain = record.get("clock_domain") if record else None
        if record_domain is not None and clock_domain is not None:
            normalized_record_domain = _normalize_clock_domain(record_domain)
            normalized_argument_domain = _normalize_clock_domain(clock_domain)
            if normalized_record_domain != normalized_argument_domain:
                raise ValueError(
                    "conflicting clock domains: "
                    f"record={normalized_record_domain!r}, argument={normalized_argument_domain!r}"
                )
        domain = _normalize_clock_domain(
            clock_domain if clock_domain is not None else record_domain or self.clock_domain
        )
        self._validate_clock_domain(domain)
        if timestamp is not None:
            return _coerce_timestamp(timestamp, name="timestamp"), domain
        if record:
            for key in _TIMESTAMP_KEYS:
                value = record.get(key)
                if value is not None:
                    return _coerce_timestamp(value, name=key), domain
            for key in _TIMESTAMP_NS_KEYS:
                value = record.get(key)
                if value is not None:
                    return _seconds_from_ns(value, name=key), domain
        return _now_seconds(domain), domain

    def _validate_clock_domain(self, clock_domain: str) -> None:
        if clock_domain != self.clock_domain:
            raise ValueError(
                "clock domain mismatch: "
                f"trajectory={self.clock_domain!r}, value={clock_domain!r}; "
                "timestamps must be converted before alignment"
            )

    def _query_timestamp(self, timestamp: float | None, clock_domain: str | None) -> float:
        domain = _normalize_clock_domain(clock_domain or self.clock_domain)
        self._validate_clock_domain(domain)
        if timestamp is None:
            return _now_seconds(domain)
        return _coerce_timestamp(timestamp, name="query timestamp")

    def append_state(
        self,
        state: dict[str, Any],
        timestamp: float | None = None,
        *,
        clock_domain: str | None = None,
    ) -> None:
        stamp, domain = self._timestamp_and_domain(state, timestamp=timestamp, clock_domain=clock_domain)
        with self._lock:
            self._states.append(TimedRecord(stamp, dict(state), domain))

    def append_action(
        self,
        action: dict[str, Any],
        timestamp: float | None = None,
        *,
        clock_domain: str | None = None,
    ) -> None:
        stamp, domain = self._timestamp_and_domain(action, timestamp=timestamp, clock_domain=clock_domain)
        with self._lock:
            self._actions.append(TimedRecord(stamp, dict(action), domain))

    def estimate_state(
        self,
        timestamp: float | None = None,
        *,
        clock_domain: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            if not self._states:
                return None
            target = self._query_timestamp(timestamp, clock_domain)
            records = sorted(self._states, key=lambda record: record.timestamp)
        if target <= records[0].timestamp:
            return dict(records[0].value)
        if target >= records[-1].timestamp:
            return dict(records[-1].value)
        right_index = bisect_right([record.timestamp for record in records], target)
        before, after = records[right_index - 1], records[right_index]
        span = max(after.timestamp - before.timestamp, 1e-9)
        alpha = (target - before.timestamp) / span
        result: dict[str, Any] = {}
        for key in before.value.keys() | after.value.keys():
            left, right = before.value.get(key), after.value.get(key)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                result[key] = float(left) + alpha * (float(right) - float(left))
            else:
                result[key] = right if alpha >= 0.5 else left
        return result

    def action_history(self) -> list[TimedRecord]:
        with self._lock:
            return list(self._actions)

    def state_history(self) -> list[TimedRecord]:
        with self._lock:
            return list(self._states)

    def future_action_trajectory(
        self,
        start_timestamp: float,
        dt: float,
        horizon: int,
        *,
        clock_domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """Sample the last known command with zero-order hold on a future grid."""
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("trajectory dt must be finite and positive")
        if horizon < 1:
            return []
        start = self._query_timestamp(start_timestamp, clock_domain)
        with self._lock:
            records = sorted(self._actions, key=lambda record: record.timestamp)
        if not records:
            return []
        result: list[dict[str, Any]] = []
        index = 0
        for step in range(horizon):
            target = start + step * float(dt)
            while index + 1 < len(records) and records[index + 1].timestamp <= target:
                index += 1
            result.append(dict(records[index].value))
        return result

    @staticmethod
    def _action_payload(action: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in action.items() if key not in _ACTION_TIME_METADATA_KEYS}

    def _future_records(
        self,
        future_action_queue: Sequence[TimedRecord | Mapping[str, Any]],
        *,
        request_timestamp: float,
        dt: float,
        history: Sequence[TimedRecord],
        clock_domain: str,
    ) -> list[TimedRecord]:
        next_implicit_timestamp = (
            max(request_timestamp, history[-1].timestamp + dt) if history else request_timestamp
        )
        result: list[TimedRecord] = []
        previous_timestamp = -math.inf
        for index, item in enumerate(future_action_queue):
            if isinstance(item, TimedRecord):
                domain = _normalize_clock_domain(item.clock_domain)
                self._validate_clock_domain(domain)
                timestamp = item.timestamp
                action = item.value
            elif isinstance(item, Mapping):
                item_domain = item.get("clock_domain", clock_domain)
                domain = _normalize_clock_domain(item_domain)
                self._validate_clock_domain(domain)
                timestamp = None
                for key in _TIMESTAMP_KEYS:
                    if item.get(key) is not None:
                        timestamp = _coerce_timestamp(item[key], name=f"future_action_queue[{index}].{key}")
                        break
                if timestamp is None:
                    for key in _TIMESTAMP_NS_KEYS:
                        if item.get(key) is not None:
                            timestamp = _seconds_from_ns(
                                item[key], name=f"future_action_queue[{index}].{key}"
                            )
                            break
                if timestamp is None:
                    timestamp = next_implicit_timestamp
                action = item
            else:
                raise TypeError(
                    "future_action_queue entries must be TimedRecord or action mappings, "
                    f"got {type(item).__name__} at index {index}"
                )
            if timestamp < request_timestamp:
                raise ValueError(
                    "future action timestamp precedes the inference request: "
                    f"index={index}, action={timestamp:.9f}, request={request_timestamp:.9f}"
                )
            if timestamp < previous_timestamp:
                raise ValueError("future_action_queue timestamps must be non-decreasing")
            result.append(TimedRecord(timestamp, self._action_payload(action), domain))
            previous_timestamp = timestamp
            next_implicit_timestamp = timestamp + dt
        return result

    def build_committed_prefill(
        self,
        *,
        image_capture_timestamp: float,
        calibrated_image_delay_s: float,
        request_timestamp: float,
        predicted_completion_timestamp: float,
        dt: float,
        future_action_queue: Sequence[TimedRecord | Mapping[str, Any]],
        max_prefill_steps: int,
        clock_domain: str | None = None,
        overflow: Literal["raise", "truncate"] = "raise",
    ) -> CommittedActionPrefill:
        """Build the action prefix committed between an image anchor and inference completion.

        All input timestamps must already be expressed in this trajectory's clock
        domain. The calibrated image delay is subtracted from the capture time,
        matching the Realtime-VLA V2 observation alignment convention. Executed
        history and the future command queue are sampled with zero-order hold on
        a control-rate grid. The first sample is always the calibrated image
        anchor, so silently dropping old samples would invalidate conditioning;
        capacity overflow therefore raises by default.
        """

        domain = _normalize_clock_domain(clock_domain or self.clock_domain)
        self._validate_clock_domain(domain)
        capture = _coerce_timestamp(image_capture_timestamp, name="image_capture_timestamp")
        request = _coerce_timestamp(request_timestamp, name="request_timestamp")
        completion = _coerce_timestamp(predicted_completion_timestamp, name="predicted_completion_timestamp")
        delay = _coerce_timestamp(calibrated_image_delay_s, name="calibrated_image_delay_s")
        control_dt = _coerce_timestamp(dt, name="dt")
        if delay < 0:
            raise ValueError("calibrated_image_delay_s must be non-negative")
        if control_dt <= 0:
            raise ValueError("dt must be positive")
        if isinstance(max_prefill_steps, bool) or not isinstance(max_prefill_steps, int):
            raise TypeError("max_prefill_steps must be an integer")
        if max_prefill_steps < 0:
            raise ValueError("max_prefill_steps must be non-negative")
        if capture > request:
            raise ValueError("image_capture_timestamp cannot be later than request_timestamp")
        if completion < request:
            raise ValueError("predicted_completion_timestamp cannot precede request_timestamp")
        if overflow not in {"raise", "truncate"}:
            raise ValueError("overflow must be either 'raise' or 'truncate'")

        anchor = capture - delay
        committed_duration = max(0.0, completion - anchor)
        # The completion anchor is inclusive: the first generated postfix
        # action follows the last committed action. This matches the upstream
        # ``timeline[:anchor_index + 1]`` contract and avoids an off-by-one
        # whenever completion lands exactly on a control tick.
        required_steps = int(math.floor(committed_duration / control_dt + 1e-12)) + 1
        truncated = required_steps > max_prefill_steps
        if truncated and overflow == "raise":
            raise PrefillCapacityError(required_steps, max_prefill_steps)
        returned_steps = min(required_steps, max_prefill_steps)
        grid = tuple(anchor + step * control_dt for step in range(returned_steps))

        with self._lock:
            history = sorted(self._actions, key=lambda record: record.timestamp)
        future = self._future_records(
            future_action_queue,
            request_timestamp=request,
            dt=control_dt,
            history=history,
            clock_domain=domain,
        )

        if not grid:
            return CommittedActionPrefill(
                clock_domain=domain,
                image_capture_timestamp=capture,
                calibrated_image_delay_s=delay,
                anchor_timestamp=anchor,
                request_timestamp=request,
                predicted_completion_timestamp=completion,
                timestamps=(),
                actions=(),
                source_timestamps=(),
                required_steps=required_steps,
                max_steps=max_prefill_steps,
                truncated=truncated,
            )

        # Planned commands are inserted first so an actually dispatched command
        # at the same timestamp wins the zero-order-hold lookup.
        timeline = sorted(
            [*(record for record in future), *(record for record in history)],
            key=lambda record: record.timestamp,
        )
        if not timeline:
            raise ValueError("cannot build a non-empty committed prefill without action history or queue")

        timeline_timestamps = [record.timestamp for record in timeline]
        actions: list[dict[str, Any]] = []
        source_timestamps: list[float] = []
        for target in grid:
            source_index = bisect_right(timeline_timestamps, target) - 1
            if source_index < 0:
                source_index = 0
            source = timeline[source_index]
            actions.append(self._action_payload(source.value))
            source_timestamps.append(source.timestamp)

        return CommittedActionPrefill(
            clock_domain=domain,
            image_capture_timestamp=capture,
            calibrated_image_delay_s=delay,
            anchor_timestamp=anchor,
            request_timestamp=request,
            predicted_completion_timestamp=completion,
            timestamps=grid,
            actions=tuple(actions),
            source_timestamps=tuple(source_timestamps),
            required_steps=required_steps,
            max_steps=max_prefill_steps,
            truncated=truncated,
        )

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._actions.clear()


class RealtimeTraceWriteError(RuntimeError):
    """Raised when an enabled deployment trace can no longer be written reliably."""


class RealtimeTraceWriter:
    """Append-only JSONL trace for raw/prefix/planned/applied trajectories."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        clock_domain: str = MONOTONIC_CLOCK_DOMAIN,
        session_id: str | None = None,
        config_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.clock_domain = _normalize_clock_domain(clock_domain)
        if self.clock_domain not in {MONOTONIC_CLOCK_DOMAIN, UNIX_CLOCK_DOMAIN}:
            raise ValueError("trace writer clock_domain must be 'monotonic' or 'unix'")
        self._lock = threading.Lock()
        self._handle = None
        self._closed = False
        self._write_failure: BaseException | None = None
        self._terminal_status: Literal["completed", "abnormal"] | None = None
        self._terminal_reason: str | None = None
        self.session_id = str(session_id or uuid.uuid4())
        self._sequence = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8", buffering=1)
            try:
                with self._lock:
                    self._write_locked(
                        "session_start",
                        schema_version=2,
                        config_snapshot=dict(config_snapshot) if config_snapshot is not None else None,
                    )
            except BaseException as write_error:
                try:
                    self.close(status="abnormal", reason="session_start write failed")
                except BaseException as close_error:
                    raise BaseExceptionGroup(
                        "realtime trace session_start and cleanup both failed",
                        [write_error, close_error],
                    ) from write_error
                raise

    @property
    def enabled(self) -> bool:
        return self._handle is not None and not self._closed and self._write_failure is None

    @property
    def closed(self) -> bool:
        return self._closed

    def mark_abnormal(self, reason: object | None = None) -> None:
        """Ensure the eventual terminal record cannot claim successful completion."""

        with self._lock:
            if self._closed:
                return
            self._terminal_status = "abnormal"
            if reason is not None and self._terminal_reason is None:
                normalized = str(reason).strip()
                self._terminal_reason = normalized or None

    def write(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        if event in {"session_start", "session_end"}:
            raise ValueError(f"{event!r} is reserved for RealtimeTraceWriter lifecycle records")
        with self._lock:
            if self._closed:
                raise RealtimeTraceWriteError("deployment trace is already closed")
            if self._write_failure is not None:
                raise RealtimeTraceWriteError(
                    "deployment trace is unusable after an earlier write failure"
                ) from self._write_failure
            self._write_locked(event, **fields)

    def _write_locked(self, event: str, **fields: Any) -> None:
        if self._handle is None:
            raise RealtimeTraceWriteError("enabled deployment trace has no open file handle")
        monotonic_timestamp = _now_seconds(MONOTONIC_CLOCK_DOMAIN)
        wall_timestamp = _now_seconds(UNIX_CLOCK_DOMAIN)
        record = {
            "event": event,
            "session_id": self.session_id,
            "sequence": self._sequence,
            **fields,
            "timestamp": (
                monotonic_timestamp if self.clock_domain == MONOTONIC_CLOCK_DOMAIN else wall_timestamp
            ),
            "clock_domain": self.clock_domain,
            "monotonic_timestamp": monotonic_timestamp,
            "monotonic_clock_domain": MONOTONIC_CLOCK_DOMAIN,
            "wall_timestamp": wall_timestamp,
            "wall_clock_domain": UNIX_CLOCK_DOMAIN,
        }
        try:
            line = json.dumps(_jsonable(record), separators=(",", ":"), ensure_ascii=True)
            self._handle.write(line + "\n")
        except Exception as exc:
            self._write_failure = exc
            self._terminal_status = "abnormal"
            if self._terminal_reason is None:
                self._terminal_reason = f"trace write failed: {type(exc).__name__}: {exc}"
            raise RealtimeTraceWriteError(
                f"failed to append realtime trace event {event!r} to {self.path}"
            ) from exc
        self._sequence += 1

    def close(
        self,
        *,
        status: Literal["completed", "abnormal"] | None = None,
        reason: object | None = None,
    ) -> None:
        """Write one terminal record and durably close the trace; repeated calls are no-ops."""

        with self._lock:
            if self._closed:
                return
            if status is not None and status not in TRACE_TERMINAL_STATUSES:
                raise ValueError(f"trace terminal status must be one of {sorted(TRACE_TERMINAL_STATUSES)}")
            if status == "abnormal":
                self._terminal_status = "abnormal"
            elif self._terminal_status is None:
                self._terminal_status = status or "completed"
            if reason is not None and self._terminal_reason is None:
                normalized = str(reason).strip()
                self._terminal_reason = normalized or None

            close_error: BaseException | None = None
            handle = self._handle
            try:
                if handle is not None and self._write_failure is None:
                    terminal_fields: dict[str, Any] = {
                        "status": self._terminal_status or "completed",
                        "records_before_end": self._sequence,
                    }
                    if self._terminal_reason is not None:
                        terminal_fields["reason"] = self._terminal_reason
                    self._write_locked("session_end", **terminal_fields)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException as exc:
                close_error = exc
                if self._write_failure is None:
                    self._write_failure = exc
            finally:
                if handle is not None:
                    try:
                        handle.close()
                    except BaseException as exc:
                        if close_error is None:
                            close_error = exc
                self._handle = None
                self._closed = True

            if close_error is not None:
                if isinstance(close_error, RealtimeTraceWriteError):
                    raise close_error
                raise RealtimeTraceWriteError(
                    f"failed to finalize realtime trace {self.path}"
                ) from close_error
