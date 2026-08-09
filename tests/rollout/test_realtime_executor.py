from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from lerobot.rollout.realtime_executor import (
    FirstOrderReplayEstimator,
    ForwardTracker,
    RealtimeExecutor,
    RealtimeExecutorConfig,
)


def make_config(**overrides) -> RealtimeExecutorConfig:
    values = {
        "action_dim": 2,
        "heartbeat_dt_s": 0.1,
        "max_velocity": np.array([1_000.0, 1_000.0]),
        "max_acceleration": np.array([10_000.0, 10_000.0]),
        "actuator_tau_s": 0.15,
        "enable_forward_tracking": False,
    }
    values.update(overrides)
    return RealtimeExecutorConfig(**values)


def test_fixed_heartbeat_linearly_samples_monotonic_waypoints() -> None:
    executor = RealtimeExecutor(make_config())
    executor.reset(np.zeros(2), timestamp=10.0)
    executor.enqueue_waypoints(
        np.array([10.0, 10.2]),
        np.array([[0.0, 0.0], [2.0, -2.0]]),
    )

    commands = np.stack([executor.heartbeat() for _ in range(3)])

    np.testing.assert_allclose(commands, [[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]])
    assert executor.last_heartbeat_timestamp == pytest.approx(10.2)
    assert executor.next_heartbeat_timestamp == pytest.approx(10.3)


def test_heartbeat_schedule_does_not_accumulate_float_drift() -> None:
    executor = RealtimeExecutor(make_config(heartbeat_dt_s=0.03))
    executor.reset(np.zeros(2), timestamp=1_000_000.0)
    executor.enqueue_waypoints(
        np.array([1_000_000.0, 1_000_003.0]),
        np.zeros((2, 2)),
    )

    for _ in range(100):
        executor.heartbeat()

    assert executor.last_heartbeat_timestamp == 1_000_000.0 + 99 * 0.03
    assert executor.next_heartbeat_timestamp == 1_000_000.0 + 100 * 0.03


def test_waypoint_batches_are_validated_atomically_and_strictly_ordered() -> None:
    executor = RealtimeExecutor(make_config())
    executor.enqueue_waypoints(np.array([1.0, 2.0]), np.zeros((2, 2)))

    with pytest.raises(ValueError, match="strictly increasing"):
        executor.enqueue_waypoints(np.array([3.0, 2.5]), np.ones((2, 2)))
    assert executor.pending_waypoint_count == 2

    with pytest.raises(ValueError, match="across appended batches"):
        executor.enqueue_waypoints(np.array([2.0, 3.0]), np.ones((2, 2)))
    assert executor.pending_waypoint_count == 2


def test_replace_waypoints_from_keeps_history_and_replaces_future_atomically() -> None:
    executor = RealtimeExecutor(
        make_config(
            action_dim=1,
            max_velocity=1_000.0,
            max_acceleration=10_000.0,
            savgol_window_length=3,
            savgol_polyorder=1,
        )
    )
    executor.reset(np.zeros(1), timestamp=10.0)
    executor.enqueue_waypoints(
        np.array([10.0, 10.1, 10.2, 10.3]),
        np.array([[0.0], [1.0], [2.0], [3.0]]),
    )
    executor.heartbeat()
    executor.heartbeat()

    executor.replace_waypoints_from(
        10.2,
        np.array([10.2, 10.3, 10.4]),
        np.array([[20.0], [30.0], [40.0]]),
    )

    assert executor.pending_waypoint_count == 3
    executor.heartbeat()
    assert executor.last_reference is not None
    # The centered window includes retained t=10.1 and replacement t=10.2/10.3.
    assert executor.last_reference[0] == pytest.approx(17.0)


def test_queue_underrun_holds_the_last_emitted_command() -> None:
    executor = RealtimeExecutor(make_config())
    executor.reset(np.zeros(2), timestamp=0.0)
    executor.enqueue_waypoints(
        np.array([0.0, 0.1]),
        np.array([[0.0, 0.0], [1.0, -1.0]]),
    )

    executor.heartbeat()
    final_command = executor.heartbeat()
    held_1 = executor.heartbeat()
    held_2 = executor.heartbeat()

    np.testing.assert_array_equal(held_1, final_command)
    np.testing.assert_array_equal(held_2, final_command)
    assert executor.last_tick_was_underrun
    assert executor.underrun_count == 2
    assert executor.pending_waypoint_count == 0


def test_returned_commands_do_not_expose_internal_mutable_state() -> None:
    executor = RealtimeExecutor(make_config())
    executor.reset(np.zeros(2), timestamp=0.0)
    executor.enqueue_waypoints(np.array([0.0, 0.1]), np.ones((2, 2)))

    command = executor.heartbeat()
    expected = executor.last_command
    command[:] = 123.0

    np.testing.assert_array_equal(executor.last_command, expected)


def test_per_joint_velocity_and_acceleration_limits_are_hard_bounds() -> None:
    dt = 0.1
    max_velocity = np.array([1.0, 2.0])
    max_acceleration = np.array([2.0, 4.0])
    executor = RealtimeExecutor(
        make_config(
            heartbeat_dt_s=dt,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )
    )
    executor.reset(np.zeros(2), timestamp=0.0)
    timestamps = np.arange(20, dtype=np.float64) * dt
    actions = np.full((20, 2), 100.0)
    executor.enqueue_waypoints(timestamps, actions)

    commands = np.vstack([np.zeros(2), *[executor.heartbeat() for _ in range(20)]])
    velocities = np.diff(commands, axis=0) / dt
    accelerations = np.diff(np.vstack([np.zeros(2), velocities]), axis=0) / dt

    assert np.all(np.abs(velocities) <= max_velocity + 1e-12)
    assert np.all(np.abs(accelerations) <= max_acceleration + 1e-12)


def test_acceleration_limit_applies_during_direction_reversal() -> None:
    dt = 0.1
    executor = RealtimeExecutor(
        make_config(
            action_dim=1,
            heartbeat_dt_s=dt,
            max_velocity=np.array([2.0]),
            max_acceleration=np.array([1.0]),
        )
    )
    executor.reset(np.zeros(1), timestamp=0.0)
    timestamps = np.arange(30, dtype=np.float64) * dt
    actions = np.concatenate([np.full((15, 1), 100.0), np.full((15, 1), -100.0)])
    executor.enqueue_waypoints(timestamps, actions)

    commands = np.vstack([np.zeros(1), *[executor.heartbeat() for _ in range(30)]])
    velocities = np.diff(commands[:, 0]) / dt

    assert np.max(np.abs(np.diff(np.concatenate([[0.0], velocities])))) <= 1.0 * dt + 1e-12


def test_numpy_savgol_smoothing_softens_a_waypoint_step() -> None:
    raw = RealtimeExecutor(make_config(action_dim=1, max_velocity=1e6, max_acceleration=1e6))
    smooth = RealtimeExecutor(
        make_config(
            action_dim=1,
            max_velocity=1e6,
            max_acceleration=1e6,
            savgol_window_length=5,
            savgol_polyorder=2,
        )
    )
    timestamps = np.arange(5, dtype=np.float64) * 0.1
    actions = np.array([[0.0], [0.0], [10.0], [10.0], [10.0]])
    for executor in (raw, smooth):
        executor.reset(np.zeros(1), timestamp=0.0)
        executor.enqueue_waypoints(timestamps, actions)
        for _ in range(3):
            executor.heartbeat()

    assert raw.last_reference is not None
    assert smooth.last_reference is not None
    assert raw.last_reference[0] == pytest.approx(10.0)
    assert 0.0 < smooth.last_reference[0] < raw.last_reference[0]


def test_first_order_replay_estimator_matches_analytic_response() -> None:
    estimator = FirstOrderReplayEstimator(action_dim=1, tau_s=0.2)
    estimator.reset(np.array([0.0]), timestamp=1.0)
    estimator.push_command(np.array([1.0]), timestamp=1.0)

    at_one_tau = estimator.estimate(1.2)
    np.testing.assert_allclose(at_one_tau, [1.0 - math.exp(-1.0)], rtol=1e-12)

    estimator.push_command(np.array([0.0]), timestamp=1.2)
    at_two_tau = estimator.estimate(1.4)
    np.testing.assert_allclose(
        at_two_tau,
        [(1.0 - math.exp(-1.0)) * math.exp(-1.0)],
        rtol=1e-12,
    )


def test_replay_estimator_uses_per_joint_tau_and_command_delay() -> None:
    estimator = FirstOrderReplayEstimator(
        action_dim=2,
        tau_s=[0.1, 0.2],
        command_delay_s=[0.0, 0.1],
    )
    estimator.reset(np.zeros(2), timestamp=0.0)
    estimator.push_command(np.ones(2), timestamp=0.0)

    estimate = estimator.estimate(0.2)

    np.testing.assert_allclose(
        estimate,
        [1.0 - math.exp(-2.0), 1.0 - math.exp(-0.5)],
        rtol=1e-12,
    )


def test_executor_defaults_forward_lead_to_per_joint_tau_plus_delay() -> None:
    config = make_config(
        actuator_tau_s=[0.1, 0.2],
        command_delay_s=[0.01, 0.03],
        forward_lead_s=None,
    )

    np.testing.assert_allclose(config.forward_lead_s, [0.11, 0.23])


def test_replay_estimator_reanchors_on_delayed_observation() -> None:
    estimator = FirstOrderReplayEstimator(action_dim=1, tau_s=0.1)
    estimator.reset(np.array([0.0]), timestamp=0.0)
    estimator.push_command(np.array([1.0]), timestamp=0.0)
    estimator.push_observation(np.array([0.25]), timestamp=0.1)

    np.testing.assert_allclose(
        estimator.estimate(0.2),
        [1.0 + (0.25 - 1.0) * math.exp(-1.0)],
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="observation timestamps must be monotonic"):
        estimator.push_observation(np.array([0.0]), timestamp=0.05)


def test_executor_reset_accepts_a_delayed_observation_anchor() -> None:
    executor = RealtimeExecutor(make_config(action_dim=1, max_velocity=1000.0, max_acceleration=10000.0))
    executor.reset(
        np.array([0.5]),
        timestamp=10.0,
        observation_timestamp=9.95,
    )
    executor.enqueue_waypoints(np.array([10.0, 10.1]), np.array([[0.5], [0.5]]))

    np.testing.assert_allclose(
        executor.heartbeat(np.array([0.5]), observation_timestamp=9.98),
        [0.5],
    )


def test_forward_tracker_applies_lag_feedforward_and_state_feedback() -> None:
    tracker = ForwardTracker(action_dim=2, lead_s=0.2, feedback_gain=0.5)

    command = tracker.compensate(
        target=np.array([1.0, 2.0]),
        target_velocity=np.array([3.0, -1.0]),
        estimated_position=np.array([0.8, 2.4]),
    )

    np.testing.assert_allclose(command, [1.7, 1.6])


def test_reset_clears_queue_counters_and_reanchors_output() -> None:
    executor = RealtimeExecutor(make_config())
    executor.reset(np.zeros(2), timestamp=0.0)
    executor.enqueue_waypoints(np.array([0.0, 0.1]), np.ones((2, 2)))
    executor.heartbeat()
    executor.heartbeat()
    executor.heartbeat()

    executor.reset(np.array([5.0, -5.0]), timestamp=20.0)

    np.testing.assert_array_equal(executor.heartbeat(), [5.0, -5.0])
    assert executor.underrun_count == 1
    assert executor.pending_waypoint_count == 0
    assert executor.last_heartbeat_timestamp == 20.0


def test_heartbeat_and_properties_are_thread_safe() -> None:
    executor = RealtimeExecutor(make_config())
    executor.reset(np.zeros(2), timestamp=100.0)
    executor.enqueue_waypoints(
        np.array([100.0, 110.0]),
        np.zeros((2, 2)),
    )

    def run_ticks(count: int) -> list[np.ndarray]:
        outputs = []
        for _ in range(count):
            outputs.append(executor.heartbeat())
            assert executor.last_command is not None
            _ = executor.pending_waypoint_count
        return outputs

    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = list(pool.map(run_ticks, [25, 25, 25, 25]))

    assert sum(len(batch) for batch in batches) == 100
    assert executor.last_heartbeat_timestamp == pytest.approx(109.9)
    np.testing.assert_array_equal(executor.last_command, np.zeros(2))


def test_preview_matches_future_smoothed_limited_commands_without_mutation() -> None:
    executor = RealtimeExecutor(
        make_config(
            action_dim=1,
            max_velocity=np.array([2.0]),
            max_acceleration=np.array([4.0]),
            actuator_tau_s=0.2,
            enable_forward_tracking=True,
            savgol_window_length=3,
            savgol_polyorder=1,
        )
    )
    executor.reset(np.zeros(1), timestamp=10.0, observation_timestamp=9.95)
    executor.enqueue_waypoints(
        np.array([10.0, 10.1, 10.2, 10.3]),
        np.array([[0.0], [1.0], [4.0], [4.0]]),
    )

    before_timestamp = executor.next_heartbeat_timestamp
    before_command = executor.last_command
    preview = executor.preview(3)

    assert [item.timestamp for item in preview] == pytest.approx([10.0, 10.1, 10.2])
    assert executor.next_heartbeat_timestamp == before_timestamp
    np.testing.assert_array_equal(executor.last_command, before_command)
    assert executor.underrun_count == 0

    actual = [executor.heartbeat() for _ in range(3)]
    np.testing.assert_allclose(
        np.stack([item.command for item in preview]),
        np.stack(actual),
    )
    assert [item.underrun for item in preview] == [False, False, False]


def test_preview_reports_future_hold_without_consuming_underrun_state() -> None:
    executor = RealtimeExecutor(make_config(action_dim=1, max_velocity=1_000.0, max_acceleration=10_000.0))
    executor.reset(np.array([3.0]), timestamp=4.0)

    preview = executor.preview_through(4.2)

    assert len(preview) == 3
    assert all(item.underrun for item in preview)
    np.testing.assert_allclose([item.command for item in preview], [[3.0], [3.0], [3.0]])
    assert executor.underrun_count == 0
    assert executor.last_heartbeat_timestamp is None


def test_atomic_control_step_cannot_be_previewed_between_replace_and_heartbeat() -> None:
    executor = RealtimeExecutor(make_config(action_dim=1, max_velocity=1_000.0, max_acceleration=10_000.0))
    executor.reset(np.zeros(1), timestamp=1.0)

    command = executor.control_step(
        1.0,
        np.array([1.0, 1.1]),
        np.array([[2.0], [3.0]]),
    )

    np.testing.assert_allclose(command, [2.0])
    preview = executor.preview(1)
    assert preview[0].timestamp == pytest.approx(1.1)
    np.testing.assert_allclose(preview[0].command, [3.0])


def test_rebase_next_heartbeat_skips_elapsed_deadlines_without_command_burst() -> None:
    executor = RealtimeExecutor(make_config(action_dim=1, max_velocity=1_000.0, max_acceleration=10_000.0))
    executor.reset(np.zeros(1), timestamp=1.0)
    executor.heartbeat()

    executor.rebase_next_heartbeat(1.5)

    assert executor.next_heartbeat_timestamp == pytest.approx(1.5)
    executor.heartbeat()
    assert executor.last_heartbeat_timestamp == pytest.approx(1.5)
    assert executor.next_heartbeat_timestamp == pytest.approx(1.6)
