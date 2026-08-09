from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.realtime_executor import RealtimeExecutor, RealtimeExecutorConfig
from lerobot.rollout.stall_guard import StallContactError
from lerobot.rollout.trajectory import RealtimeTraceWriteError


class _IdentityProcessor:
    steps: list = []

    def __call__(self, value):
        return value

    def reset(self) -> None:
        pass


class _FakePolicy:
    config = SimpleNamespace(chunk_size=8, rtc_training_max_delay=0)

    def reset(self) -> None:
        pass


class _FakeRobot:
    robot_type = "cpu_fake"
    action_features = {"joint.pos": object()}

    def __init__(self, *, send_delay_s: float = 0.0) -> None:
        self.send_delay_s = send_delay_s
        self.sent_actions: list[dict] = []

    @property
    def last_observation_timing(self) -> dict:
        return {
            "clock": "monotonic",
            "state_timestamp": 9.95,
            "camera_timestamps": {},
            "observation_timestamp": 9.95,
        }

    def send_action(self, action: dict) -> dict:
        if self.send_delay_s > 0.0:
            time.sleep(self.send_delay_s)
        sent = dict(action)
        self.sent_actions.append(sent)
        return sent


class _RobotActionProcessor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[dict, dict]] = []

    def __call__(self, transition: tuple[dict, dict]) -> dict:
        action, observation = transition
        self.calls.append((dict(action), dict(observation)))
        if self.fail:
            raise RuntimeError("processor failed")
        return dict(action)


class _RecordingTrace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def write(self, event: str, **fields) -> None:
        self.events.append((event, fields))

    def close(self) -> None:
        pass


class _FailingTrace(_RecordingTrace):
    def __init__(self) -> None:
        super().__init__()
        self.write_count = 0
        self.abnormal_reason: object | None = None

    def write(self, event: str, **fields) -> None:
        del event, fields
        self.write_count += 1
        raise RealtimeTraceWriteError("synthetic trace write failure")

    def mark_abnormal(self, reason: object | None = None) -> None:
        self.abnormal_reason = reason


class _FailingFilter:
    def apply(self, action: dict, observation: dict) -> dict:
        raise RuntimeError("filter failed")

    def reset(self) -> None:
        pass


class _FailingStallGuard:
    def observe(self, requested: dict, sent: dict) -> None:
        raise StallContactError("stall failed")

    def reset(self) -> None:
        pass


def _wait_until(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def _make_threaded_engine(
    *,
    heartbeat_dt_s: float = 0.02,
    send_delay_s: float = 0.0,
    processor: _RobotActionProcessor | None = None,
    trace: _RecordingTrace | None = None,
    action_filter=None,
    stall_guard=None,
) -> tuple[RTCInferenceEngine, _FakeRobot]:
    rtc_config = RTCConfig(enabled=True, execution_horizon=6)
    robot = _FakeRobot(send_delay_s=send_delay_s)
    executor = RealtimeExecutor(
        RealtimeExecutorConfig(
            action_dim=1,
            heartbeat_dt_s=heartbeat_dt_s,
            max_velocity=1_000.0,
            max_acceleration=10_000.0,
            enable_forward_tracking=False,
        )
    )
    engine = RTCInferenceEngine(
        policy=_FakePolicy(),
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        robot_wrapper=robot,
        rtc_config=rtc_config,
        hw_features={},
        task="independent smooth executor",
        fps=1.0 / heartbeat_dt_s,
        device="cpu",
        rtc_queue_threshold=4,
        rtc_timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=2,
        realtime_executor=executor,
        ordered_action_keys=["joint.pos"],
        robot_action_processor=processor or _RobotActionProcessor(),
        trace=trace,
        action_filter=action_filter,
        stall_guard=stall_guard,
    )
    return engine, robot


def test_executor_control_schedule_does_not_accumulate_float_drift(monkeypatch) -> None:
    engine, _ = _make_threaded_engine(heartbeat_dt_s=1.0 / 30.0)
    executor = engine._realtime_executor
    assert executor is not None
    heartbeat_dt = executor.config.heartbeat_dt_s
    origin = 15_632_093.487973886
    scheduled_timestamps: list[float] = []

    def run_cycle(
        scheduled: float,
        *,
        heartbeat_dt: float,
        deadline_tolerance: float,
    ) -> int:
        del heartbeat_dt, deadline_tolerance
        scheduled_timestamps.append(scheduled)
        if len(scheduled_timestamps) == 400:
            engine._shutdown_event.set()
        return 0

    engine._policy_active.set()
    monkeypatch.setattr(
        engine,
        "_monotonic_seconds",
        lambda: origin + len(scheduled_timestamps) * heartbeat_dt,
    )
    monkeypatch.setattr(engine, "_run_executor_control_cycle", run_cycle)

    engine._executor_control_loop()

    assert scheduled_timestamps == [origin + tick * heartbeat_dt for tick in range(400)]


def test_rtc_queue_drives_fixed_heartbeat_smoothing_with_future_lookahead() -> None:
    rtc_config = RTCConfig(enabled=True, execution_horizon=6)
    smooth = RealtimeExecutor(
        RealtimeExecutorConfig(
            action_dim=1,
            heartbeat_dt_s=0.1,
            max_velocity=1_000.0,
            max_acceleration=10_000.0,
            enable_forward_tracking=False,
            savgol_window_length=3,
            savgol_polyorder=1,
        )
    )
    engine = RTCInferenceEngine(
        policy=_FakePolicy(),
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        robot_wrapper=_FakeRobot(),
        rtc_config=rtc_config,
        hw_features={},
        task="smooth executor integration",
        fps=10.0,
        device="cpu",
        rtc_queue_threshold=4,
        rtc_timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=2,
        realtime_executor=smooth,
    )
    engine._action_queue = ActionQueue(rtc_config)
    engine._monotonic_seconds = lambda: 10.0
    queue = engine.action_queue
    assert queue is not None
    actions = torch.tensor([[0.0], [10.0], [10.0], [10.0]])
    queue.merge(actions, actions, real_delay=0)
    engine.notify_observation({"joint.pos": 0.0})

    first = engine.get_action(None)
    second = engine.get_action(None)

    assert first is not None and second is not None
    assert first.item() == pytest.approx(10.0 / 3.0)
    assert second.item() == pytest.approx(20.0 / 3.0)
    assert queue.snapshot().total_consumed == 2
    assert smooth.last_heartbeat_timestamp == pytest.approx(10.1)
    assert smooth.pending_waypoint_count == 2


def test_rtc_executor_fails_closed_when_action_dimension_is_wrong() -> None:
    rtc_config = RTCConfig(enabled=True, execution_horizon=6)
    smooth = RealtimeExecutor(
        RealtimeExecutorConfig(
            action_dim=2,
            heartbeat_dt_s=0.1,
            max_velocity=1_000.0,
            max_acceleration=10_000.0,
        )
    )
    engine = RTCInferenceEngine(
        policy=_FakePolicy(),
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        robot_wrapper=_FakeRobot(),
        rtc_config=rtc_config,
        hw_features={},
        task="smooth executor dimension guard",
        fps=10.0,
        device="cpu",
        realtime_executor=smooth,
    )
    engine._action_queue = ActionQueue(rtc_config)
    queue = engine.action_queue
    assert queue is not None
    actions = torch.tensor([[0.0], [1.0]])
    queue.merge(actions, actions, real_delay=0)

    assert engine.get_action(None) is None
    assert engine.failed is True


def test_direct_smooth_trace_write_failure_enters_fatal_state() -> None:
    trace = _FailingTrace()
    engine, _ = _make_threaded_engine(trace=trace)
    engine._action_queue = ActionQueue(engine._rtc_config)
    engine._monotonic_seconds = lambda: 10.0
    queue = engine.action_queue
    assert queue is not None
    actions = torch.tensor([[0.0], [1.0]])
    queue.merge(actions, actions, real_delay=0)
    engine.notify_observation({"joint.pos": 0.0})

    assert engine.get_action(None) is None
    assert engine.failed is True
    assert engine.fatal_error is not None
    assert "Realtime executor failed" in str(engine.fatal_error)
    assert "synthetic trace write failure" in str(engine.fatal_error)
    assert trace.write_count == 1
    assert trace.abnormal_reason is engine.fatal_error


def test_independent_executor_dispatches_without_get_action_and_holds_on_underrun() -> None:
    trace = _RecordingTrace()
    processor = _RobotActionProcessor()
    engine, robot = _make_threaded_engine(processor=processor, trace=trace)
    engine.start()
    try:
        engine.notify_control_observation({"joint.pos": 0.0})
        queue = engine.action_queue
        assert queue is not None
        actions = torch.tensor([[2.0]])
        queue.merge(actions, actions, real_delay=0)
        engine.resume()

        _wait_until(lambda: len(robot.sent_actions) >= 4)
        engine.pause()
        sent_at_pause = len(robot.sent_actions)
        time.sleep(0.06)

        assert len(robot.sent_actions) == sent_at_pause
        assert queue.snapshot().total_consumed == 1
        assert robot.sent_actions[-1] == robot.sent_actions[-2]
        snapshot = engine.executor_control_snapshot()
        assert snapshot.dispatch_count >= 4
        assert snapshot.hold_count >= 2
        assert snapshot.queue_empty_count >= 3
        assert snapshot.last_applied_command == robot.sent_actions[-1]
        assert processor.calls
        assert any(
            event == "executor_control" and fields["underrun"] is True for event, fields in trace.events
        )
    finally:
        engine.stop()

    sent_at_stop = len(robot.sent_actions)
    time.sleep(0.05)
    assert len(robot.sent_actions) == sent_at_stop


def test_owned_dispatch_get_action_only_reports_latest_command_without_resending() -> None:
    from lerobot.rollout.strategies import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    engine, robot = _make_threaded_engine()
    engine.start()
    try:
        engine.notify_control_observation({"joint.pos": 0.0})
        queue = engine.action_queue
        assert queue is not None
        actions = torch.tensor([[1.0], [1.0]])
        queue.merge(actions, actions, real_delay=0)
        engine.resume()
        _wait_until(lambda: engine.executor_control_snapshot().dispatch_count >= 1)
        engine.pause()
        sent_before_poll = len(robot.sent_actions)
        ctx = SimpleNamespace(
            policy=SimpleNamespace(inference=engine),
            data=SimpleNamespace(dataset_features={}, ordered_action_keys=["joint.pos"]),
        )

        reported = send_next_action(
            {},
            {"joint.pos": 0.0},
            ctx,
            ActionInterpolator(),
        )

        assert reported is not None
        assert len(robot.sent_actions) == sent_before_poll
    finally:
        engine.stop()


def test_slow_hardware_dispatch_records_deadline_miss_and_skips_burst_ticks() -> None:
    engine, robot = _make_threaded_engine(heartbeat_dt_s=0.01, send_delay_s=0.025)
    engine.start()
    try:
        engine.notify_control_observation({"joint.pos": 0.0})
        queue = engine.action_queue
        assert queue is not None
        actions = torch.ones((4, 1))
        queue.merge(actions, actions, real_delay=0)
        engine.resume()
        _wait_until(lambda: len(robot.sent_actions) >= 2)
        engine.pause()

        snapshot = engine.executor_control_snapshot()
        assert snapshot.deadline_miss_count >= 1
        assert snapshot.missed_periods >= 1
        assert snapshot.last_execution_s >= 0.02
    finally:
        engine.stop()


def test_processor_exception_fails_executor_before_hardware_send() -> None:
    processor = _RobotActionProcessor(fail=True)
    engine, robot = _make_threaded_engine(processor=processor)
    engine.start()
    try:
        engine.notify_control_observation({"joint.pos": 0.0})
        queue = engine.action_queue
        assert queue is not None
        actions = torch.ones((2, 1))
        queue.merge(actions, actions, real_delay=0)
        engine.resume()
        _wait_until(lambda: engine.failed)

        assert robot.sent_actions == []
        assert isinstance(engine.fatal_error, RuntimeError)
    finally:
        engine.stop()


def test_filter_exception_fails_executor_before_hardware_send() -> None:
    engine, robot = _make_threaded_engine(action_filter=_FailingFilter())
    engine.start()
    try:
        engine.notify_control_observation({"joint.pos": 0.0})
        queue = engine.action_queue
        assert queue is not None
        actions = torch.ones((2, 1))
        queue.merge(actions, actions, real_delay=0)
        engine.resume()
        _wait_until(lambda: engine.failed)

        assert robot.sent_actions == []
        assert isinstance(engine.fatal_error, RuntimeError)
    finally:
        engine.stop()


def test_stall_exception_sends_present_position_hold_then_enters_fatal_state() -> None:
    engine, robot = _make_threaded_engine(stall_guard=_FailingStallGuard())
    engine.start()
    try:
        engine.notify_control_observation({"joint.pos": 0.0})
        queue = engine.action_queue
        assert queue is not None
        actions = torch.ones((2, 1))
        queue.merge(actions, actions, real_delay=0)
        engine.resume()
        _wait_until(lambda: engine.failed)

        assert len(robot.sent_actions) == 2
        assert robot.sent_actions[-1] == {"joint.pos": 0.0}
        sent_at_fatal = len(robot.sent_actions)
        time.sleep(0.05)
        assert len(robot.sent_actions) == sent_at_fatal
    finally:
        engine.stop()


def test_prefix_health_and_fatal_state_block_all_future_executor_sends() -> None:
    engine, robot = _make_threaded_engine()
    engine.start()
    try:
        engine.notify_control_observation({"joint.pos": 0.0})
        queue = engine.action_queue
        assert queue is not None
        actions = torch.ones((6, 1))
        queue.merge(actions, actions, real_delay=0)
        engine.resume()
        _wait_until(lambda: len(robot.sent_actions) >= 1)

        engine._prefix_health_replan_event.set()
        sent_at_replan = len(robot.sent_actions)
        time.sleep(0.06)
        assert len(robot.sent_actions) == sent_at_replan

        engine._prefix_health_replan_event.clear()
        _wait_until(lambda: len(robot.sent_actions) > sent_at_replan)
        engine._enter_fatal_state(RuntimeError("test fatal"))
        sent_at_fatal = len(robot.sent_actions)
        time.sleep(0.06)
        assert len(robot.sent_actions) == sent_at_fatal
    finally:
        engine.stop()


def test_executor_heartbeat_barrier_holds_until_dynamic_prefix_anchor() -> None:
    engine, robot = _make_threaded_engine(heartbeat_dt_s=0.015)
    engine.start()
    try:
        engine.notify_control_observation({"joint.pos": 0.0})
        queue = engine.action_queue
        assert queue is not None
        actions = torch.tensor([[2.0], [2.0]])
        queue.merge(actions, actions, real_delay=0)
        engine._executor_postfix_not_before_heartbeat = 3
        engine.resume()
        _wait_until(lambda: len(robot.sent_actions) >= 4)
        engine.pause()

        assert [item["joint.pos"] for item in robot.sent_actions[:3]] == [0.0, 0.0, 0.0]
        assert robot.sent_actions[3]["joint.pos"] == pytest.approx(2.0)
        assert queue.snapshot().total_consumed >= 1
    finally:
        engine.stop()
