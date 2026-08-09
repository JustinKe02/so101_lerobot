#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""CPU integration coverage for RTC dynamic committed-action prefill."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference import rtc as rtc_module
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.realtime_executor import RealtimeExecutor, RealtimeExecutorConfig
from lerobot.rollout.speed_adapter import ROBOT_ACTION_COORDINATE_SPACE
from lerobot.rollout.time_axis import TimeAxisPlan
from lerobot.rollout.trajectory import DelayAlignedTrajectory


class _IdentityProcessor:
    def __init__(self) -> None:
        self.steps: list = []

    def __call__(self, value):
        return value

    def reset(self) -> None:
        pass


class _OffsetProcessor(_IdentityProcessor):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def __call__(self, value):
        return value + self.offset


class _RecordingRobotSpacePlanner:
    feature_coordinate_space = ROBOT_ACTION_COORDINATE_SPACE

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, int]] = []

    def plan(self, actions, *, committed_prefix_steps: int) -> TimeAxisPlan:
        actions = actions.detach().cpu().clone()
        self.calls.append((actions, committed_prefix_steps))
        segment_count = len(actions) - 1
        return TimeAxisPlan(
            actions=actions.numpy(),
            segment_durations=np.full(segment_count, 0.1),
            used_fallback=False,
            reference_segment_durations=np.full(segment_count, 0.1),
            speed_factors=np.ones(segment_count),
        )


class _FakeRobot:
    robot_type = "cpu_fake"
    action_features = {"joint.pos": object()}

    @property
    def last_observation_timing(self) -> dict:
        return {
            "clock": "monotonic",
            "state_timestamp": 10.0,
            "camera_timestamps": {"camera": 10.0},
            "observation_timestamp": 10.0,
        }


class _FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(chunk_size=12, rtc_training_max_delay=6)
        self.calls: list[dict] = []
        self.on_predict = lambda: None
        self.pause_event = None

    def predict_action_chunk(
        self,
        observation: dict,
        *,
        inference_delay: int,
        prev_chunk_left_over: torch.Tensor | None,
        rtc_mode: str,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "inference_delay": inference_delay,
                "prev_chunk_left_over": (
                    None if prev_chunk_left_over is None else prev_chunk_left_over.clone()
                ),
                "rtc_mode": rtc_mode,
            }
        )
        self.on_predict()
        self.pause_event.clear()
        return torch.arange(1000.0, 1012.0).reshape(1, 12, 1)

    def reset(self) -> None:
        pass


class _FakeTime:
    """Two-read inference clock that stops the loop on its next idle sleep."""

    def __init__(self, engine: RTCInferenceEngine) -> None:
        self._engine = engine
        self._values = iter((10.0, 10.2))

    def perf_counter(self) -> float:
        return next(self._values, 10.2)

    def sleep(self, _seconds: float) -> None:
        self._engine._shutdown_event.set()


def _make_dynamic_engine(
    policy: _FakePolicy,
    *,
    trajectory: DelayAlignedTrajectory,
    planner: _RecordingRobotSpacePlanner,
) -> RTCInferenceEngine:
    rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=_IdentityProcessor(),
        postprocessor=_OffsetProcessor(10_000.0),
        robot_wrapper=_FakeRobot(),
        rtc_config=rtc_config,
        hw_features={},
        task="cpu robot-space planning",
        fps=10.0,
        device="cpu",
        rtc_queue_threshold=8,
        rtc_inference_mode="trained_prefix",
        rtc_timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=5,
        dynamic_prefill_enabled=True,
        max_prefill_steps=6,
        trajectory=trajectory,
        time_axis_planner=planner,
    )
    engine._action_queue = ActionQueue(rtc_config)
    return engine


def _run_one_inference(
    monkeypatch: pytest.MonkeyPatch,
    engine: RTCInferenceEngine,
    policy: _FakePolicy,
) -> None:
    monkeypatch.setattr(rtc_module, "time", _FakeTime(engine))
    engine.notify_observation({"joint.pos": -1.0})
    engine.resume()
    policy.pause_event = engine._policy_active
    engine._rtc_loop()
    assert engine.failed is False


def test_first_inference_plans_postprocessed_robot_space_without_a_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _FakePolicy()
    planner = _RecordingRobotSpacePlanner()
    engine = _make_dynamic_engine(policy, trajectory=DelayAlignedTrajectory(), planner=planner)

    _run_one_inference(monkeypatch, engine, policy)

    assert policy.calls[0]["prev_chunk_left_over"] is None
    assert len(planner.calls) == 1
    planned_actions, committed_prefix_steps = planner.calls[0]
    torch.testing.assert_close(
        planned_actions.flatten(),
        torch.arange(11_000.0, 11_012.0),
    )
    assert committed_prefix_steps == 0


def test_prefix_health_suppressed_empty_prefill_still_plans_robot_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _FakePolicy()
    planner = _RecordingRobotSpacePlanner()
    engine = _make_dynamic_engine(policy, trajectory=DelayAlignedTrajectory(), planner=planner)
    queue = engine.action_queue
    assert queue is not None
    old_actions = torch.arange(8.0).reshape(8, 1)
    queue.merge(old_actions, old_actions, real_delay=0)
    monkeypatch.setattr(engine, "_begin_prefix_health_inference", lambda _sequence: (1, True, False))

    _run_one_inference(monkeypatch, engine, policy)

    assert policy.calls[0]["prev_chunk_left_over"] is None
    planned_actions, committed_prefix_steps = planner.calls[0]
    torch.testing.assert_close(
        planned_actions.flatten(),
        torch.arange(11_000.0, 11_012.0),
    )
    assert committed_prefix_steps == 0


def test_normal_dynamic_prefill_plans_postprocessed_robot_space_postfix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _FakePolicy()
    planner = _RecordingRobotSpacePlanner()
    trajectory = DelayAlignedTrajectory()
    trajectory.append_action({"joint.pos": -1.0}, timestamp=10.0)
    engine = _make_dynamic_engine(policy, trajectory=trajectory, planner=planner)
    queue = engine.action_queue
    assert queue is not None
    old_actions = torch.arange(8.0).reshape(8, 1)
    queue.merge(old_actions, old_actions, real_delay=0)

    _run_one_inference(monkeypatch, engine, policy)

    prefix = policy.calls[0]["prev_chunk_left_over"]
    assert prefix is not None
    assert policy.calls[0]["inference_delay"] == 6
    planned_actions, committed_prefix_steps = planner.calls[0]
    torch.testing.assert_close(
        planned_actions.flatten(),
        torch.arange(11_006.0, 11_012.0),
    )
    assert committed_prefix_steps == 0


def test_dynamic_prefill_preserves_committed_queue_through_inclusive_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _FakePolicy()
    trajectory = DelayAlignedTrajectory()
    trajectory.append_action({"joint.pos": -1.0}, timestamp=10.0)
    rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        robot_wrapper=_FakeRobot(),
        rtc_config=rtc_config,
        hw_features={},
        task="cpu dynamic prefill",
        fps=10.0,
        device="cpu",
        rtc_queue_threshold=8,
        rtc_inference_mode="trained_prefix",
        rtc_timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=5,
        dynamic_prefill_enabled=True,
        max_prefill_steps=6,
        trajectory=trajectory,
    )
    engine._action_queue = ActionQueue(rtc_config)
    queue = engine.action_queue
    assert queue is not None
    old_actions = torch.arange(8.0).reshape(8, 1)
    queue.merge(old_actions, old_actions, real_delay=0)

    fake_time = _FakeTime(engine)
    monkeypatch.setattr(rtc_module, "time", fake_time)
    engine.notify_observation({"joint.pos": -1.0})
    engine.resume()
    policy.pause_event = engine._policy_active

    consumed: list[torch.Tensor] = []

    def consume_two_actions() -> None:
        for _ in range(2):
            action = engine.get_action(None)
            assert action is not None
            consumed.append(action)

    policy.on_predict = consume_two_actions
    engine._rtc_loop()

    assert engine.failed is False
    assert torch.equal(torch.stack(consumed).flatten(), torch.tensor([0.0, 1.0]))
    assert len(policy.calls) == 1
    assert policy.calls[0]["rtc_mode"] == "trained_prefix"
    assert policy.calls[0]["inference_delay"] == 6
    prefix_buffer = policy.calls[0]["prev_chunk_left_over"]
    assert prefix_buffer is not None
    assert torch.equal(prefix_buffer[:6].flatten(), torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]))

    # The current action is prefix token 0, while queue actions 0..4 end at
    # the inclusive completion anchor. After two dequeues, queue actions 2..4
    # remain committed and model action 6 is the first generated postfix.
    expected = torch.tensor([2.0, 3.0, 4.0, 1006.0, 1007.0, 1008.0, 1009.0, 1010.0, 1011.0])
    snapshot = queue.snapshot()
    assert snapshot.processed_leftover is not None
    assert torch.equal(snapshot.processed_leftover.flatten(), expected)


def test_dynamic_prefill_uses_limited_executor_preview_not_raw_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _FakePolicy()
    trajectory = DelayAlignedTrajectory()
    executor = RealtimeExecutor(
        RealtimeExecutorConfig(
            action_dim=1,
            heartbeat_dt_s=0.1,
            max_velocity=np.array([1.0]),
            max_acceleration=np.array([10.0]),
            enable_forward_tracking=False,
        )
    )
    executor.reset(np.zeros(1), timestamp=10.0)
    raw_targets = np.full((8, 1), 10.0)
    timestamps = 10.0 + np.arange(8) * 0.1
    first_applied = executor.control_step(10.0, timestamps, raw_targets)
    trajectory.append_action({"joint.pos": float(first_applied[0])}, timestamp=10.0)

    rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        robot_wrapper=_FakeRobot(),
        rtc_config=rtc_config,
        hw_features={},
        task="executor preview prefill",
        fps=10.0,
        device="cpu",
        rtc_queue_threshold=8,
        rtc_inference_mode="trained_prefix",
        rtc_timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=5,
        dynamic_prefill_enabled=True,
        max_prefill_steps=6,
        trajectory=trajectory,
        realtime_executor=executor,
        ordered_action_keys=["joint.pos"],
    )
    engine._action_queue = ActionQueue(rtc_config)
    queue = engine.action_queue
    assert queue is not None
    raw_queue = torch.full((8, 1), 10.0)
    queue.merge(raw_queue, raw_queue, real_delay=0)

    _run_one_inference(monkeypatch, engine, policy)

    prefix = policy.calls[0]["prev_chunk_left_over"]
    assert prefix is not None
    torch.testing.assert_close(
        prefix[:6, 0],
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert not torch.equal(prefix[:6, 0], raw_queue[:6, 0])
    assert executor.last_heartbeat_timestamp == pytest.approx(10.0)
    assert executor.next_heartbeat_timestamp == pytest.approx(10.1)


def test_dynamic_prefill_uses_executor_hold_preview_when_action_queue_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _FakePolicy()
    trajectory = DelayAlignedTrajectory()
    executor = RealtimeExecutor(
        RealtimeExecutorConfig(
            action_dim=1,
            heartbeat_dt_s=0.1,
            max_velocity=np.array([1.0]),
            max_acceleration=np.array([10.0]),
            enable_forward_tracking=False,
        )
    )
    executor.reset(np.array([2.0]), timestamp=10.0)
    held = executor.heartbeat()
    trajectory.append_action({"joint.pos": float(held[0])}, timestamp=10.0)

    rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        robot_wrapper=_FakeRobot(),
        rtc_config=rtc_config,
        hw_features={},
        task="executor hold preview prefill",
        fps=10.0,
        device="cpu",
        rtc_queue_threshold=8,
        rtc_inference_mode="trained_prefix",
        rtc_timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=5,
        dynamic_prefill_enabled=True,
        max_prefill_steps=6,
        trajectory=trajectory,
        realtime_executor=executor,
        ordered_action_keys=["joint.pos"],
    )
    engine._action_queue = ActionQueue(rtc_config)

    _run_one_inference(monkeypatch, engine, policy)

    prefix = policy.calls[0]["prev_chunk_left_over"]
    assert prefix is not None
    torch.testing.assert_close(prefix[:6, 0], torch.full((6,), 2.0))
    assert engine._executor_postfix_not_before_heartbeat == 5
    queue = engine.action_queue
    assert queue is not None
    assert queue.qsize() == 6
