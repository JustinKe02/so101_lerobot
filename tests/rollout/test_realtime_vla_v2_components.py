from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import sparse

import lerobot.rollout.time_axis as time_axis_module
from lerobot.rollout.speed_adapter import SpeedAdapter, SpeedAdapterConfig
from lerobot.rollout.time_axis import TimeAxisPlanner, TimeAxisPlannerConfig
from lerobot.rollout.trajectory import DelayAlignedTrajectory, RealtimeTraceWriter


class _BoundedQuadraticOSQP:
    """Small OSQP test double for diagonal-objective planner cases."""

    def setup(self, **kwargs) -> None:
        p_matrix = kwargs["P"]
        q = kwargs["q"]
        lower = kwargs["l"]
        upper = kwargs["u"]
        diagonal = np.asarray(p_matrix.diagonal(), dtype=np.float64)
        unconstrained = np.divide(-q, diagonal, out=np.zeros_like(q), where=diagonal != 0)
        variable_count = len(q)
        self.solution = np.clip(unconstrained, lower[:variable_count], upper[:variable_count])

    def solve(self):
        return SimpleNamespace(
            x=self.solution,
            info=SimpleNamespace(status="solved"),
        )


@pytest.fixture
def fake_osqp(monkeypatch):
    module = SimpleNamespace(OSQP=_BoundedQuadraticOSQP)
    monkeypatch.setattr(time_axis_module, "_load_osqp_modules", lambda: (module, sparse))


def test_time_axis_planner_preserves_first_waypoint_and_bounds(fake_osqp) -> None:
    actions = np.stack(
        [np.linspace(0.0, 1.0, 10), np.sin(np.linspace(0.0, 2.0, 10))],
        axis=1,
    )
    planner = TimeAxisPlanner(
        TimeAxisPlannerConfig(
            enabled=True,
            dt_ref=0.05,
            dt_min=0.02,
            dt_max=0.1,
            lambda_acc=0.0,
            horizon=10,
            max_velocity=3.0,
        )
    )
    result = planner.plan(actions)
    assert not result.used_fallback
    np.testing.assert_allclose(result.actions[0], actions[0])
    assert np.all(result.segment_durations >= 0.02 - 1e-9)
    assert np.all(result.segment_durations <= 0.1 + 1e-9)


def test_time_axis_planner_accepts_torch_tensor(fake_osqp) -> None:
    planner = TimeAxisPlanner(TimeAxisPlannerConfig(enabled=True, horizon=4, lambda_acc=0.0))
    result = planner.plan(torch.arange(8, dtype=torch.float32).reshape(4, 2))
    assert result.actions.shape == (4, 2)


def test_time_axis_planner_keeps_committed_prefix_bytes_unchanged(fake_osqp) -> None:
    actions = np.stack([np.arange(10, dtype=np.float32) * 0.1, np.zeros(10, dtype=np.float32)], axis=1)
    planner = TimeAxisPlanner(
        TimeAxisPlannerConfig(
            enabled=True,
            dt_ref=0.05,
            dt_min=0.02,
            dt_max=0.5,
            lambda_acc=0.0,
            max_velocity=0.5,
            horizon=3,
            stride=2,
        )
    )

    result = planner.plan(actions, committed_prefix_steps=4)

    assert not result.used_fallback
    assert result.actions.dtype == actions.dtype
    assert result.actions[:4].tobytes() == actions[:4].tobytes()
    assert not np.array_equal(result.actions[4:], actions[4:])


def test_time_axis_planner_uses_adapter_reference_only_after_committed_prefix(fake_osqp) -> None:
    adapter = SpeedAdapter(
        SpeedAdapterConfig(
            action_dim=2,
            hidden_dims=(4,),
            beta_min=1.5,
            beta_max=2.5,
        )
    )
    for parameter in adapter.parameters():
        torch.nn.init.zeros_(parameter)
    planner = TimeAxisPlanner(
        TimeAxisPlannerConfig(
            enabled=True,
            dt_ref=0.05,
            dt_min=0.01,
            dt_max=0.1,
            lambda_acc=0.0,
            horizon=10,
        ),
        speed_adapter=adapter,
    )
    actions = np.stack([np.arange(5, dtype=np.float64) * 0.01, np.zeros(5)], axis=1)

    result = planner.plan(actions, committed_prefix_steps=2)

    assert not result.used_fallback
    assert result.reference_segment_durations is not None
    assert result.speed_factors is not None
    np.testing.assert_allclose(result.speed_factors, [1.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(result.reference_segment_durations, [0.05, 0.025, 0.025, 0.025])
    np.testing.assert_allclose(result.segment_durations, [0.05, 0.025, 0.025, 0.025])
    assert result.actions[:2].tobytes() == actions[:2].tobytes()


def test_time_axis_planner_falls_back_when_adapter_features_are_incompatible(fake_osqp) -> None:
    adapter = SpeedAdapter(SpeedAdapterConfig(action_dim=3, hidden_dims=(4,)))
    planner = TimeAxisPlanner(
        TimeAxisPlannerConfig(enabled=True, horizon=4),
        speed_adapter=adapter,
    )
    actions = np.arange(12, dtype=np.float32).reshape(6, 2)

    result = planner.plan(actions)

    assert result.used_fallback
    assert "dimension 6" in result.reason
    assert result.reference_segment_durations is None
    assert result.speed_factors is None
    assert result.actions.tobytes() == actions.tobytes()


def test_time_axis_planner_rolls_horizon_and_enforces_velocity(fake_osqp) -> None:
    actions = np.stack([np.arange(12, dtype=np.float64) * 0.2, np.zeros(12)], axis=1)
    planner = TimeAxisPlanner(
        TimeAxisPlannerConfig(
            enabled=True,
            dt_ref=0.05,
            dt_min=0.02,
            dt_max=0.4,
            lambda_acc=0.0,
            max_velocity=1.0,
            horizon=3,
            stride=2,
        )
    )

    result = planner.plan(actions)

    assert not result.used_fallback
    assert len(result.segment_durations) == len(actions) - 1
    assert np.all(result.segment_durations >= 0.02 - 1e-9)
    assert np.all(result.segment_durations <= 0.4 + 1e-9)
    source_velocity = np.linalg.norm(np.diff(actions, axis=0), axis=1) / result.segment_durations
    output_velocity = np.linalg.norm(np.diff(result.actions, axis=0), axis=1) / planner.config.dt_ref
    assert np.all(source_velocity <= 1.0 + 1e-8)
    assert np.all(output_velocity <= 1.0 + 1e-8)


def test_time_axis_planner_carries_committed_speed_across_rolling_windows(monkeypatch) -> None:
    planner = TimeAxisPlanner(TimeAxisPlannerConfig(enabled=True, horizon=3, stride=2))
    waypoints = np.arange(8, dtype=np.float64)[:, None]
    calls: list[tuple[np.ndarray | None, float | None]] = []

    def solve_window(
        deltas,
        *,
        reference_inverse_speeds,
        velocity_limits,
        acceleration_limits,
        previous_delta=None,
        previous_inverse_duration=None,
    ):
        del reference_inverse_speeds, velocity_limits, acceleration_limits
        calls.append((previous_delta, previous_inverse_duration))
        return np.arange(len(deltas), dtype=np.float64) + 10.0 * len(calls)

    monkeypatch.setattr(planner, "_solve_qp", solve_window)
    speeds = planner._solve_rolling(
        waypoints,
        reference_durations=np.full(len(waypoints) - 1, planner.config.dt_ref),
        velocity_limits=None,
        acceleration_limits=None,
    )

    assert calls[0] == (None, None)
    np.testing.assert_array_equal(calls[1][0], np.array([1.0]))
    assert calls[1][1] == speeds[1] == 11.0
    np.testing.assert_array_equal(calls[2][0], np.array([1.0]))
    assert calls[2][1] == speeds[3] == 21.0


def test_time_axis_qp_constrains_acceleration_at_rolling_window_boundary(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class RecordingOSQP:
        def setup(self, **kwargs) -> None:
            captured.update(kwargs)

        def solve(self):
            return SimpleNamespace(
                x=np.array([10.0, 10.0]),
                info=SimpleNamespace(status="solved"),
            )

    monkeypatch.setattr(
        time_axis_module,
        "_load_osqp_modules",
        lambda: (SimpleNamespace(OSQP=RecordingOSQP), sparse),
    )
    planner = TimeAxisPlanner(
        TimeAxisPlannerConfig(
            enabled=True,
            dt_ref=0.05,
            dt_min=0.01,
            dt_max=0.2,
            max_acceleration=[2.0],
        )
    )

    planner._solve_qp(
        np.ones((2, 1), dtype=np.float64),
        reference_inverse_speeds=np.full(2, 10.0),
        velocity_limits=None,
        acceleration_limits=np.array([2.0]),
        previous_delta=np.array([1.0]),
        previous_inverse_duration=10.0,
    )

    constraint_matrix = captured["A"].toarray()
    lower = captured["l"]
    upper = captured["u"]
    boundary_rows = [
        index
        for index, row in enumerate(constraint_matrix)
        if np.allclose(row, [1.0, 0.0]) and np.isclose(lower[index], 9.9) and np.isclose(upper[index], 10.1)
    ]
    assert boundary_rows


def test_time_axis_planner_enforces_per_joint_velocity_limits(fake_osqp) -> None:
    actions = np.stack(
        [np.arange(8, dtype=np.float64) * 0.2, np.arange(8, dtype=np.float64) * 0.2],
        axis=1,
    )
    limits = np.array([1.0, 4.0])
    planner = TimeAxisPlanner(
        TimeAxisPlannerConfig(
            enabled=True,
            dt_ref=0.05,
            dt_min=0.02,
            dt_max=0.4,
            lambda_acc=0.0,
            max_velocity=limits.tolist(),
            horizon=4,
        )
    )

    result = planner.plan(actions)

    assert not result.used_fallback
    source_velocity = np.abs(np.diff(actions, axis=0)) / result.segment_durations[:, None]
    output_velocity = np.abs(np.diff(result.actions, axis=0)) / planner.config.dt_ref
    assert np.all(source_velocity <= limits[None, :] + 1e-8)
    assert np.all(output_velocity <= limits[None, :] + 1e-8)


def test_time_axis_planner_selects_full_action_limits_for_optimized_dims(fake_osqp) -> None:
    actions = np.stack(
        [
            np.arange(8, dtype=np.float64) * 10.0,
            np.arange(8, dtype=np.float64) * 0.2,
            np.arange(8, dtype=np.float64) * 20.0,
        ],
        axis=1,
    )
    planner = TimeAxisPlanner(
        TimeAxisPlannerConfig(
            enabled=True,
            dt_ref=0.05,
            dt_min=0.02,
            dt_max=0.4,
            lambda_acc=0.0,
            max_velocity=[1000.0, 1.0, 1000.0],
            optimization_dims=[1],
            horizon=4,
        )
    )

    result = planner.plan(actions)

    assert not result.used_fallback
    velocity = np.abs(np.diff(result.actions[:, 1])) / planner.config.dt_ref
    assert np.all(velocity <= 1.0 + 1e-8)
    np.testing.assert_array_equal(result.actions[:, 0], actions[:, 0])
    np.testing.assert_array_equal(result.actions[:, 2], actions[:, 2])


def test_time_axis_planner_falls_back_when_osqp_is_unavailable(monkeypatch) -> None:
    def unavailable():
        raise RuntimeError("osqp unavailable")

    monkeypatch.setattr(time_axis_module, "_load_osqp_modules", unavailable)
    actions = np.arange(12, dtype=np.float32).reshape(6, 2)
    planner = TimeAxisPlanner(TimeAxisPlannerConfig(enabled=True, horizon=4))

    result = planner.plan(actions, start_index=2)

    assert result.used_fallback
    assert result.reason == "osqp unavailable"
    assert result.actions.tobytes() == actions.tobytes()
    np.testing.assert_allclose(result.segment_durations, planner.config.dt_ref)


def test_delay_aligned_trajectory_interpolates_numeric_state() -> None:
    history = DelayAlignedTrajectory()
    history.append_state({"timestamp": 10.0, "shoulder_pan": 0.0})
    history.append_state({"timestamp": 11.0, "shoulder_pan": 2.0})
    estimate = history.estimate_state(10.25)
    assert estimate is not None
    assert estimate["shoulder_pan"] == 0.5


def test_delay_aligned_trajectory_builds_zero_order_hold_future_commands() -> None:
    history = DelayAlignedTrajectory()
    history.append_action({"timestamp": 10.0, "shoulder_pan": 1.0})
    history.append_action({"timestamp": 10.2, "shoulder_pan": 2.0})
    future = history.future_action_trajectory(10.1, 0.1, 3)
    assert [row["shoulder_pan"] for row in future] == [1.0, 2.0, 2.0]


def test_trace_writer_is_jsonl_and_json_safe(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    trace = RealtimeTraceWriter(path)
    trace.write("inference_chunk", raw_model_chunk=np.zeros((2, 3)), finite=1.0)
    trace.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "session_start"
    row = rows[1]
    assert row["event"] == "inference_chunk"
    assert row["raw_model_chunk"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert rows[2]["event"] == "session_end"
    assert rows[2]["status"] == "completed"
    assert rows[2]["records_before_end"] == 2


def test_trace_writer_keeps_full_pi05_action_chunk_and_records_config(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    chunk = np.arange(300, dtype=np.float32).reshape(50, 6)
    trace = RealtimeTraceWriter(path, session_id="test-session", config_snapshot={"fps": 30})
    trace.write("inference_chunk", raw_model_chunk=chunk)
    trace.close()

    session, event, terminal = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert session["session_id"] == "test-session"
    assert session["config_snapshot"] == {"fps": 30}
    assert event["raw_model_chunk"] == chunk.tolist()
    assert terminal["event"] == "session_end"
    assert terminal["status"] == "completed"
