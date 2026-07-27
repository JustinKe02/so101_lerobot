#!/usr/bin/env python

"""Tests for the stall/contact guard and abnormal-shutdown torque hold."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from lerobot.rollout.stall_guard import StallContactError, StallContactGuard
from lerobot.rollout.strategies.core import RolloutStrategy, send_next_action

REQUESTED = {"shoulder_lift.pos": -93.9, "elbow_flex.pos": 85.3}
CLAMPED = {"shoulder_lift.pos": -94.2, "elbow_flex.pos": 86.6}


def test_guard_counts_and_trips_at_threshold() -> None:
    guard = StallContactGuard(max_consecutive_clamped=3)

    guard.observe(REQUESTED, CLAMPED)
    guard.observe(REQUESTED, CLAMPED)
    assert guard.consecutive_clamped == 2

    with pytest.raises(StallContactError, match="elbow_flex.pos"):
        guard.observe(REQUESTED, CLAMPED)


def test_guard_resets_on_clean_dispatch() -> None:
    guard = StallContactGuard(max_consecutive_clamped=3)

    guard.observe(REQUESTED, CLAMPED)
    guard.observe(REQUESTED, CLAMPED)
    guard.observe(REQUESTED, dict(REQUESTED))
    assert guard.consecutive_clamped == 0

    guard.observe(REQUESTED, CLAMPED)
    guard.observe(REQUESTED, CLAMPED)
    assert guard.consecutive_clamped == 2


def test_guard_ignores_deviations_within_tolerance_and_non_dict_results() -> None:
    guard = StallContactGuard(max_consecutive_clamped=1, tolerance=1.0)
    guard.observe(REQUESTED, {key: value + 0.5 for key, value in REQUESTED.items()})
    assert guard.consecutive_clamped == 0

    strict = StallContactGuard(max_consecutive_clamped=2)
    strict.observe(REQUESTED, CLAMPED)
    strict.observe(REQUESTED, None)
    assert strict.consecutive_clamped == 0


def test_guard_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError):
        StallContactGuard(max_consecutive_clamped=0)
    with pytest.raises(ValueError):
        StallContactGuard(max_consecutive_clamped=1, tolerance=-1.0)


class _RecordingWrapper:
    def __init__(self, applied: dict) -> None:
        self.applied = applied
        self.sent_actions: list[dict] = []

    def send_action(self, action: dict) -> dict:
        self.sent_actions.append(dict(action))
        return self.applied


class _FakeEngine:
    failed = False

    @contextmanager
    def action_dispatch_guard(self):
        yield True

    def get_action(self, obs_frame):
        raise AssertionError("get_action should not be called in this test")

    def notify_action_result(self, processed, sent, obs_raw) -> None:
        pass


def _make_ctx(wrapper: _RecordingWrapper, stall_guard: StallContactGuard | None):
    ordered_keys = list(REQUESTED)
    return SimpleNamespace(
        policy=SimpleNamespace(inference=_FakeEngine(), action_filter=None, stall_guard=stall_guard),
        data=SimpleNamespace(dataset_features={}, ordered_action_keys=ordered_keys),
        hardware=SimpleNamespace(robot_wrapper=wrapper),
        processors=SimpleNamespace(robot_action_processor=lambda pair: dict(pair[0])),
    )


def _make_interpolator():
    return SimpleNamespace(
        needs_new_action=lambda: False,
        get=lambda: torch.tensor([REQUESTED[key] for key in REQUESTED]),
    )


def test_send_next_action_holds_present_position_and_reraises_on_trip() -> None:
    wrapper = _RecordingWrapper(applied=CLAMPED)
    ctx = _make_ctx(wrapper, StallContactGuard(max_consecutive_clamped=1))
    obs_raw = {"shoulder_lift.pos": -99.0, "elbow_flex.pos": 91.6, "top": "image"}

    with pytest.raises(StallContactError):
        send_next_action(obs_raw, obs_raw, ctx, _make_interpolator())

    assert len(wrapper.sent_actions) == 2
    assert wrapper.sent_actions[1] == {"shoulder_lift.pos": -99.0, "elbow_flex.pos": 91.6}


def test_send_next_action_passes_clean_dispatch_through() -> None:
    wrapper = _RecordingWrapper(applied=dict(REQUESTED))
    ctx = _make_ctx(wrapper, StallContactGuard(max_consecutive_clamped=1))

    action = send_next_action(dict(REQUESTED), dict(REQUESTED), ctx, _make_interpolator())

    assert action is not None
    assert len(wrapper.sent_actions) == 1


class _TeardownStrategy(RolloutStrategy):
    def setup(self, ctx) -> None:  # pragma: no cover - unused stub
        pass

    def run(self, ctx) -> None:  # pragma: no cover - unused stub
        pass

    def teardown(self, ctx) -> None:  # pragma: no cover - unused stub
        pass


class _FakeRobot:
    def __init__(self) -> None:
        self.is_connected = True
        self.config = SimpleNamespace(disable_torque_on_disconnect=True)
        self.disable_torque_at_disconnect: bool | None = None

    def disconnect(self) -> None:
        self.disable_torque_at_disconnect = self.config.disable_torque_on_disconnect
        self.is_connected = False


def _make_hw(robot: _FakeRobot, abnormal_shutdown: bool):
    return SimpleNamespace(
        robot_wrapper=SimpleNamespace(inner=robot),
        teleop=None,
        initial_position={"shoulder_lift.pos": 0.0},
        abnormal_shutdown=abnormal_shutdown,
    )


def test_abnormal_teardown_holds_torque_and_skips_return_to_initial() -> None:
    strategy = _TeardownStrategy(config=SimpleNamespace())
    robot = _FakeRobot()

    strategy._teardown_hardware(_make_hw(robot, abnormal_shutdown=True), return_to_initial_position=True)

    assert robot.disable_torque_at_disconnect is False
    assert robot.is_connected is False


def test_normal_teardown_keeps_torque_release_default() -> None:
    strategy = _TeardownStrategy(config=SimpleNamespace())
    robot = _FakeRobot()

    strategy._teardown_hardware(_make_hw(robot, abnormal_shutdown=False), return_to_initial_position=False)

    assert robot.disable_torque_at_disconnect is True
    assert robot.is_connected is False
