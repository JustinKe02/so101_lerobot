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

"""CPU tests for the optional RTC prefix-health gate."""

from __future__ import annotations

from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference import rtc as rtc_module
from lerobot.rollout.inference.factory import RTCInferenceConfig, create_inference_engine
from lerobot.rollout.inference.rtc import RTCInferenceEngine


class _IdentityProcessor:
    steps: list = []

    def __call__(self, value):
        return value

    def reset(self) -> None:
        pass


class _FakeRobot:
    robot_type = "cpu_fake"
    action_features: dict = {}


class _FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace()
        self.calls: list[dict] = []
        self.on_predict = lambda: None
        self.pause_event = None

    def predict_action_chunk(
        self,
        observation: dict,
        *,
        inference_delay: int,
        prev_chunk_left_over: torch.Tensor | None,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "inference_delay": inference_delay,
                "prev_chunk_left_over": (
                    None if prev_chunk_left_over is None else prev_chunk_left_over.clone()
                ),
            }
        )
        self.on_predict()
        self.pause_event.clear()
        return torch.arange(50, dtype=torch.float32).reshape(1, 50, 1)

    def reset(self) -> None:
        pass


class _FakeTime:
    def __init__(self, engine: RTCInferenceEngine, latency_s: float = 0.15) -> None:
        self._engine = engine
        self._values = iter((10.0, 10.0 + latency_s))
        self._last = 10.0 + latency_s

    def perf_counter(self) -> float:
        return next(self._values, self._last)

    def sleep(self, _seconds: float) -> None:
        self._engine._shutdown_event.set()


class _StopOnSleepTime:
    def __init__(self, engine: RTCInferenceEngine) -> None:
        self._engine = engine

    @staticmethod
    def perf_counter() -> float:
        raise AssertionError("policy inference started before a fresh observation")

    def sleep(self, _seconds: float) -> None:
        self._engine._shutdown_event.set()


def _make_engine(
    *,
    enabled: bool = True,
    queue_threshold: int = 1,
    safety_stop_replans: int = 0,
) -> tuple[RTCInferenceEngine, _FakePolicy]:
    policy = _FakePolicy()
    rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        robot_wrapper=_FakeRobot(),
        rtc_config=rtc_config,
        hw_features={},
        task="cpu prefix health",
        fps=30.0,
        device="cpu",
        rtc_queue_threshold=queue_threshold,
        rtc_timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=5,
        prefix_health_enabled=enabled,
        prefix_health_severe_residual_threshold=5.0,
        prefix_health_consecutive_severe=3,
        prefix_health_safety_stop_replans=safety_stop_replans,
    )
    engine._action_queue = ActionQueue(rtc_config)
    engine.notify_observation({})
    engine.resume()
    policy.pause_event = engine._policy_active
    return engine, policy


def _request_replan(engine: RTCInferenceEngine) -> None:
    for _ in range(3):
        engine.notify_action_result({"joint.pos": 20.0}, {"joint.pos": 0.0}, {})


def _run_one_inference(monkeypatch: pytest.MonkeyPatch, engine: RTCInferenceEngine) -> None:
    engine._shutdown_event.clear()
    engine.resume()
    monkeypatch.setattr(rtc_module, "time", _FakeTime(engine))
    engine._rtc_loop()
    assert engine.failed is False


def test_mild_and_uncomparable_results_do_not_request_replan() -> None:
    engine, _ = _make_engine()

    engine.notify_action_result({"joint.pos": 4.9}, {"joint.pos": 0.0}, {})
    engine.notify_action_result({"joint.pos": 20.0}, None, {})

    snapshot = engine.prefix_health_snapshot()
    assert snapshot.total_action_results == 2
    assert snapshot.comparable_action_results == 1
    assert snapshot.uncomparable_action_results == 1
    assert snapshot.severe_action_results == 0
    assert snapshot.feedback_revision == 0
    assert snapshot.replan_requested is False


def test_consecutive_severe_results_request_replan_without_changing_queue_generation() -> None:
    engine, _ = _make_engine()
    queue = engine.action_queue
    assert queue is not None
    actions = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    queue.merge(actions, actions, real_delay=0)
    generation_before = queue.snapshot().generation

    engine.notify_action_result({"joint.pos": 20.0}, {"joint.pos": 0.0}, {})
    engine.notify_action_result({"joint.pos": 20.0}, {"joint.pos": 0.0}, {})
    assert engine.prefix_health_snapshot().feedback_revision == 0
    engine.notify_action_result({"joint.pos": 20.0}, {"joint.pos": 0.0}, {})

    snapshot = engine.prefix_health_snapshot()
    assert snapshot.severe_action_results == 3
    assert snapshot.feedback_revision == 1
    assert snapshot.acknowledged_revision == 0
    assert snapshot.replan_requested is True
    assert snapshot.replan_requests == 1
    assert snapshot.safety_stop_requested is False
    assert queue.snapshot().generation == generation_before


def test_pending_replan_bypasses_queue_threshold_then_guidance_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, policy = _make_engine(queue_threshold=1)
    queue = engine.action_queue
    assert queue is not None
    old_actions = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    queue.merge(old_actions, old_actions, real_delay=0)
    assert queue.qsize() > engine._rtc_queue_threshold
    _request_replan(engine)
    engine.notify_observation({"after_trigger": True})

    _run_one_inference(monkeypatch, engine)

    assert len(policy.calls) == 1
    assert policy.calls[0]["prev_chunk_left_over"] is None
    snapshot = engine.prefix_health_snapshot()
    assert snapshot.feedback_revision == 1
    assert snapshot.acknowledged_revision == 1
    assert snapshot.replan_requested is False
    assert snapshot.unguided_inferences == 1
    assert snapshot.unguided_merges == 1

    engine._shutdown_event.clear()
    engine._rtc_queue_threshold = 100
    _run_one_inference(monkeypatch, engine)

    assert len(policy.calls) == 2
    assert policy.calls[1]["prev_chunk_left_over"] is not None
    recovered = engine.prefix_health_snapshot()
    assert recovered.unguided_inferences == 1
    assert recovered.unguided_merges == 1


def test_revision_arriving_during_inference_remains_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, policy = _make_engine(queue_threshold=1)
    queue = engine.action_queue
    assert queue is not None
    old_actions = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    queue.merge(old_actions, old_actions, real_delay=0)
    queue_before = queue.snapshot()
    _request_replan(engine)
    engine.notify_observation({"after_trigger": True})
    policy.on_predict = lambda: _request_replan(engine)

    _run_one_inference(monkeypatch, engine)

    snapshot = engine.prefix_health_snapshot()
    assert snapshot.feedback_revision == 2
    assert snapshot.acknowledged_revision == 0
    assert snapshot.replan_requested is True
    assert snapshot.unguided_inferences == 1
    assert snapshot.unguided_merges == 0
    queue_after = queue.snapshot()
    assert queue_after.generation == queue_before.generation
    assert torch.equal(queue_after.processed_leftover, queue_before.processed_leftover)
    assert engine.get_action(None) is None


def test_pending_replan_waits_for_observation_after_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, policy = _make_engine(queue_threshold=1)
    queue = engine.action_queue
    assert queue is not None
    old_actions = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    queue.merge(old_actions, old_actions, real_delay=0)
    _request_replan(engine)
    monkeypatch.setattr(rtc_module, "time", _StopOnSleepTime(engine))

    engine._rtc_loop()

    assert policy.calls == []
    snapshot = engine.prefix_health_snapshot()
    assert snapshot.replan_requested is True
    assert snapshot.unguided_inferences == 0

    engine._shutdown_event.clear()
    engine.notify_observation({"after_trigger": True})
    _run_one_inference(monkeypatch, engine)

    assert len(policy.calls) == 1
    assert policy.calls[0]["prev_chunk_left_over"] is None
    assert engine.prefix_health_snapshot().replan_requested is False


def test_disabled_prefix_health_is_a_noop() -> None:
    engine, _ = _make_engine(enabled=False)

    for _ in range(10):
        engine.notify_action_result({"joint.pos": 20.0}, {"joint.pos": 0.0}, {})

    snapshot = engine.prefix_health_snapshot()
    assert snapshot.enabled is False
    assert snapshot.total_action_results == 0
    assert snapshot.feedback_revision == 0
    assert snapshot.replan_requested is False


def test_safety_stop_immediately_blocks_actions_and_clears_queue() -> None:
    engine, _ = _make_engine(safety_stop_replans=1)
    queue = engine.action_queue
    assert queue is not None
    actions = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    queue.merge(actions, actions, real_delay=0)

    _request_replan(engine)

    snapshot = engine.prefix_health_snapshot()
    assert snapshot.safety_stop_requested is True
    assert engine.failed is True
    assert engine._shutdown_event.is_set()
    assert engine.get_action(None) is None
    assert queue.empty() is True


def test_any_fatal_state_blocks_actions_and_clears_queue() -> None:
    engine, _ = _make_engine()
    queue = engine.action_queue
    assert queue is not None
    actions = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    queue.merge(actions, actions, real_delay=0)
    fatal_error = RuntimeError("fatal RTC invariant")

    engine._enter_fatal_state(fatal_error)

    assert engine.failed is True
    assert engine.fatal_error is fatal_error
    assert engine.get_action(None) is None
    assert queue.empty() is True


def test_background_inference_fatal_clears_queue_and_blocks_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, policy = _make_engine(queue_threshold=30)
    queue = engine.action_queue
    assert queue is not None
    actions = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    queue.merge(actions, actions, real_delay=0)
    inference_error = RuntimeError("inference failed")

    def fail_inference() -> None:
        raise inference_error

    policy.on_predict = fail_inference
    monkeypatch.setattr(rtc_module, "_RTC_MAX_CONSECUTIVE_ERRORS", 1)
    monkeypatch.setattr(rtc_module, "time", _FakeTime(engine))

    engine._rtc_loop()

    assert engine.failed is True
    assert engine.fatal_error is inference_error
    assert engine.get_action(None) is None
    assert queue.empty() is True


def test_fatal_transition_waits_for_inflight_action_dispatch() -> None:
    engine, _ = _make_engine()
    worker_started = Event()
    fatal_complete = Event()

    def request_fatal() -> None:
        worker_started.set()
        engine._enter_fatal_state(RuntimeError("concurrent fatal"))
        fatal_complete.set()

    with engine.action_dispatch_guard() as dispatch_allowed:
        assert dispatch_allowed is True
        worker = Thread(target=request_fatal)
        worker.start()
        assert worker_started.wait(timeout=1.0)
        assert fatal_complete.wait(timeout=0.05) is False
        assert engine.failed is False

    worker.join(timeout=1.0)
    assert worker.is_alive() is False
    assert fatal_complete.is_set()
    assert engine.failed is True


def test_stop_join_timeout_enters_fatal_state_and_keeps_thread_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _ = _make_engine()
    queue = engine.action_queue
    assert queue is not None
    actions = torch.arange(20, dtype=torch.float32).reshape(20, 1)
    queue.merge(actions, actions, real_delay=0)
    stuck_thread = MagicMock()
    stuck_thread.is_alive.return_value = True
    engine._rtc_thread = stuck_thread
    monkeypatch.setattr(rtc_module, "_RTC_JOIN_TIMEOUT_S", 0.01)

    engine.stop()

    stuck_thread.join.assert_called_once_with(timeout=0.01)
    assert engine._rtc_thread is stuck_thread
    assert engine.failed is True
    assert isinstance(engine.fatal_error, TimeoutError)
    assert queue.empty() is True


def test_prefix_health_factory_configuration_is_actual_consumed_only() -> None:
    with pytest.raises(ValueError, match="requires timing_mode='actual_consumed'"):
        RTCInferenceConfig(prefix_health_enabled=True)

    config = RTCInferenceConfig(
        timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        prefix_health_enabled=True,
        prefix_health_severe_residual_threshold=7.0,
        prefix_health_consecutive_severe=4,
        prefix_health_safety_stop_replans=0,
    )
    with patch("lerobot.rollout.inference.factory.RTCInferenceEngine") as engine_cls:
        create_inference_engine(
            config,
            policy=MagicMock(),
            preprocessor=MagicMock(),
            postprocessor=MagicMock(),
            robot_wrapper=MagicMock(robot_type="mock"),
            hw_features={},
            dataset_features={},
            ordered_action_keys=["joint.pos"],
            task="test",
            fps=30.0,
            device="cpu",
        )

    kwargs = engine_cls.call_args.kwargs
    assert kwargs["prefix_health_enabled"] is True
    assert kwargs["prefix_health_severe_residual_threshold"] == 7.0
    assert kwargs["prefix_health_consecutive_severe"] == 4
    assert kwargs["prefix_health_safety_stop_replans"] == 0


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_prefix_health_threshold_must_be_finite_and_positive(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        RTCInferenceConfig(prefix_health_severe_residual_threshold=threshold)

    with pytest.raises(ValueError, match="finite and positive"):
        RTCInferenceEngine(
            policy=_FakePolicy(),
            preprocessor=_IdentityProcessor(),
            postprocessor=_IdentityProcessor(),
            robot_wrapper=_FakeRobot(),
            rtc_config=RTCConfig(enabled=True, execution_horizon=10),
            hw_features={},
            task="invalid prefix threshold",
            fps=30.0,
            device="cpu",
            prefix_health_severe_residual_threshold=threshold,
        )
