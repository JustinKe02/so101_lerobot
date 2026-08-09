from __future__ import annotations

import json

import pytest

from lerobot.rollout.trajectory import (
    MONOTONIC_CLOCK_DOMAIN,
    CommittedActionPrefill,
    DelayAlignedTrajectory,
    PrefillCapacityError,
    RealtimeTraceWriteError,
    RealtimeTraceWriter,
    TimedRecord,
)


def test_default_timestamps_use_monotonic_nanoseconds_in_seconds(monkeypatch) -> None:
    monkeypatch.setattr("lerobot.rollout.trajectory.time.monotonic_ns", lambda: 12_345_678_900)
    history = DelayAlignedTrajectory()

    history.append_action({"shoulder_pan": 1.0})

    record = history.action_history()[0]
    assert record.timestamp == pytest.approx(12.3456789)
    assert record.clock_domain == MONOTONIC_CLOCK_DOMAIN


def test_state_interpolation_is_order_independent_and_clamps_boundaries() -> None:
    history = DelayAlignedTrajectory()
    history.append_state({"shoulder_pan": 2.0, "mode": "right"}, timestamp=11.0)
    history.append_state({"shoulder_pan": 0.0, "mode": "left"}, timestamp=10.0)

    assert history.estimate_state(9.0) == {"shoulder_pan": 0.0, "mode": "left"}
    assert history.estimate_state(12.0) == {"shoulder_pan": 2.0, "mode": "right"}
    assert history.estimate_state(10.25) == {"shoulder_pan": 0.5, "mode": "left"}
    assert history.estimate_state(10.75) == {"shoulder_pan": 1.5, "mode": "right"}


def test_build_committed_prefill_aligns_executed_history_and_future_queue() -> None:
    history = DelayAlignedTrajectory()
    history.append_action({"shoulder_pan": 0.0}, timestamp=9.8)
    history.append_action({"shoulder_pan": 1.0}, timestamp=9.9)
    history.append_action({"shoulder_pan": 2.0}, timestamp=10.0)

    result = history.build_committed_prefill(
        image_capture_timestamp=10.0,
        calibrated_image_delay_s=0.1,
        request_timestamp=10.05,
        predicted_completion_timestamp=10.25,
        dt=0.1,
        future_action_queue=[{"shoulder_pan": 3.0}, {"shoulder_pan": 4.0}],
        max_prefill_steps=4,
    )

    assert isinstance(result, CommittedActionPrefill)
    assert result.clock_domain == MONOTONIC_CLOCK_DOMAIN
    assert result.anchor_timestamp == pytest.approx(9.9)
    assert result.timestamps == pytest.approx((9.9, 10.0, 10.1, 10.2))
    assert result.source_timestamps == pytest.approx((9.9, 10.0, 10.1, 10.2))
    assert [action["shoulder_pan"] for action in result.actions] == [1.0, 2.0, 3.0, 4.0]
    assert result.required_steps == 4
    assert not result.truncated


def test_build_committed_prefill_uses_actual_action_at_duplicate_timestamp() -> None:
    history = DelayAlignedTrajectory()
    history.append_action({"shoulder_pan": 1.0}, timestamp=10.0)
    history.append_action({"shoulder_pan": 2.0}, timestamp=10.1)

    result = history.build_committed_prefill(
        image_capture_timestamp=10.0,
        calibrated_image_delay_s=0.0,
        request_timestamp=10.0,
        predicted_completion_timestamp=10.2,
        dt=0.1,
        future_action_queue=[
            {"timestamp": 10.1, "shoulder_pan": 99.0},
            {"timestamp": 10.2, "shoulder_pan": 3.0},
        ],
        max_prefill_steps=3,
    )

    assert [action["shoulder_pan"] for action in result.actions] == [1.0, 2.0, 3.0]
    assert all("timestamp" not in action for action in result.actions)


def test_build_committed_prefill_capacity_is_a_hard_boundary_by_default() -> None:
    history = DelayAlignedTrajectory()
    history.append_action({"shoulder_pan": 1.0}, timestamp=9.9)

    with pytest.raises(PrefillCapacityError, match=r"required_steps=4, max_steps=3") as exc_info:
        history.build_committed_prefill(
            image_capture_timestamp=10.0,
            calibrated_image_delay_s=0.1,
            request_timestamp=10.05,
            predicted_completion_timestamp=10.25,
            dt=0.1,
            future_action_queue=[],
            max_prefill_steps=3,
        )

    assert exc_info.value.required_steps == 4
    assert exc_info.value.max_steps == 3


def test_build_committed_prefill_can_return_a_marked_diagnostic_truncation() -> None:
    history = DelayAlignedTrajectory()
    history.append_action({"shoulder_pan": 1.0}, timestamp=9.9)

    result = history.build_committed_prefill(
        image_capture_timestamp=10.0,
        calibrated_image_delay_s=0.1,
        request_timestamp=10.05,
        predicted_completion_timestamp=10.25,
        dt=0.1,
        future_action_queue=[],
        max_prefill_steps=3,
        overflow="truncate",
    )

    assert result.truncated
    assert result.required_steps == 4
    assert len(result) == 3
    assert result.timestamps == pytest.approx((9.9, 10.0, 10.1))


def test_prefill_includes_the_completion_anchor_on_an_exact_control_tick() -> None:
    history = DelayAlignedTrajectory()
    history.append_action({"shoulder_pan": 1.0}, timestamp=10.0)

    result = history.build_committed_prefill(
        image_capture_timestamp=10.0,
        calibrated_image_delay_s=0.0,
        request_timestamp=10.0,
        predicted_completion_timestamp=10.0,
        dt=0.1,
        future_action_queue=[],
        max_prefill_steps=6,
    )

    assert len(result) == 1
    assert result.timestamps == pytest.approx((10.0,))
    assert result.actions == ({"shoulder_pan": 1.0},)


def test_clock_domain_mismatch_is_rejected_at_record_query_and_queue_boundaries() -> None:
    history = DelayAlignedTrajectory()
    with pytest.raises(ValueError, match="clock domain mismatch"):
        history.append_action({"shoulder_pan": 1.0}, timestamp=10.0, clock_domain="unix")
    with pytest.raises(ValueError, match="conflicting clock domains"):
        history.append_action(
            {"clock_domain": "unix", "shoulder_pan": 1.0},
            timestamp=10.0,
            clock_domain="monotonic",
        )

    history.append_state({"shoulder_pan": 1.0}, timestamp=10.0)
    with pytest.raises(ValueError, match="clock domain mismatch"):
        history.estimate_state(10.0, clock_domain="unix")

    with pytest.raises(ValueError, match="clock domain mismatch"):
        history.build_committed_prefill(
            image_capture_timestamp=10.0,
            calibrated_image_delay_s=0.0,
            request_timestamp=10.0,
            predicted_completion_timestamp=10.1,
            dt=0.1,
            future_action_queue=[TimedRecord(10.0, {"shoulder_pan": 1.0}, "unix")],
            max_prefill_steps=2,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image_capture_timestamp": 10.1}, "image_capture_timestamp"),
        ({"predicted_completion_timestamp": 9.9}, "predicted_completion_timestamp"),
        ({"calibrated_image_delay_s": -0.1}, "calibrated_image_delay_s"),
        ({"dt": 0.0}, "dt must be positive"),
    ],
)
def test_build_committed_prefill_rejects_invalid_time_boundaries(overrides, message) -> None:
    kwargs = {
        "image_capture_timestamp": 10.0,
        "calibrated_image_delay_s": 0.0,
        "request_timestamp": 10.0,
        "predicted_completion_timestamp": 10.1,
        "dt": 0.1,
        "future_action_queue": [{"shoulder_pan": 1.0}],
        "max_prefill_steps": 2,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        DelayAlignedTrajectory().build_committed_prefill(**kwargs)


def test_trace_records_explicit_monotonic_and_wall_clock_domains(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("lerobot.rollout.trajectory.time.monotonic_ns", lambda: 2_500_000_000)
    monkeypatch.setattr("lerobot.rollout.trajectory.time.time_ns", lambda: 20_000_000_000)
    path = tmp_path / "trace.jsonl"
    trace = RealtimeTraceWriter(path)

    trace.write("tick")
    trace.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["event"] == "session_start"
    record = records[1]
    assert record["sequence"] == 1
    assert record["session_id"] == records[0]["session_id"]
    assert record["timestamp"] == 2.5
    assert record["clock_domain"] == "monotonic"
    assert record["monotonic_timestamp"] == 2.5
    assert record["monotonic_clock_domain"] == "monotonic"
    assert record["wall_timestamp"] == 20.0
    assert record["wall_clock_domain"] == "unix"
    terminal = records[2]
    assert terminal["event"] == "session_end"
    assert terminal["sequence"] == 2
    assert terminal["records_before_end"] == 2
    assert terminal["status"] == "completed"
    assert terminal["session_id"] == records[0]["session_id"]


def test_trace_close_is_idempotent_and_writes_one_terminal_record(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = RealtimeTraceWriter(path)

    trace.close()
    trace.close(status="abnormal", reason="must not replace the first terminal record")

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    terminal_records = [record for record in records if record["event"] == "session_end"]
    assert trace.closed is True
    assert len(terminal_records) == 1
    assert terminal_records[0]["status"] == "completed"
    assert terminal_records[0]["records_before_end"] == 1


def test_trace_marked_abnormal_never_records_completed_terminal_status(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = RealtimeTraceWriter(path)
    trace.write("tick")

    trace.mark_abnormal("synthetic rollout failure")
    trace.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    terminal_records = [record for record in records if record["event"] == "session_end"]
    assert len(terminal_records) == 1
    assert terminal_records[0]["status"] == "abnormal"
    assert terminal_records[0]["reason"] == "synthetic rollout failure"
    assert not any(
        record["event"] == "session_end" and record.get("status") == "completed" for record in records
    )


def test_trace_write_failure_propagates_and_prevents_success_terminal_record(tmp_path) -> None:
    class _FailingWriteHandle:
        def __init__(self, delegate) -> None:
            self._delegate = delegate

        def write(self, _value: str) -> None:
            raise OSError("synthetic trace write failure")

        def close(self) -> None:
            self._delegate.close()

    path = tmp_path / "trace.jsonl"
    trace = RealtimeTraceWriter(path)
    assert trace._handle is not None
    trace._handle = _FailingWriteHandle(trace._handle)

    with pytest.raises(RealtimeTraceWriteError, match="failed to append realtime trace event"):
        trace.write("tick")
    with pytest.raises(RealtimeTraceWriteError, match="unusable after an earlier write failure"):
        trace.write("second_tick")
    trace.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert trace.closed is True
    assert [record["event"] for record in records] == ["session_start"]
