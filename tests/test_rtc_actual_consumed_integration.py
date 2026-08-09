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

"""CPU integration tests for the RTC actual-consumed inference loop."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference import rtc as rtc_module
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.time_axis import TimeAxisPlan
from lerobot.rollout.trajectory import RealtimeTraceWriteError, RealtimeTraceWriter


class _IdentityProcessor:
    def __init__(self) -> None:
        self.steps: list = []
        self.reset_count = 0

    def __call__(self, value):
        return value

    def reset(self) -> None:
        self.reset_count += 1


class _FakeRobot:
    robot_type = "cpu_fake"
    action_features: dict = {}


class _FakePolicy:
    def __init__(self, chunk: torch.Tensor) -> None:
        self.config = SimpleNamespace(chunk_size=chunk.shape[1])
        self._chunk = chunk
        self._pause_event = None
        self.on_predict: Callable[[], None] = lambda: None
        self.calls: list[dict] = []
        self.reset_count = 0

    def predict_action_chunk(
        self,
        observation: dict,
        *,
        inference_delay: int,
        prev_chunk_left_over: torch.Tensor | None,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "observation": observation,
                "inference_delay": inference_delay,
                "prev_chunk_left_over": (
                    None if prev_chunk_left_over is None else prev_chunk_left_over.clone()
                ),
            }
        )
        self.on_predict()
        self._pause_event.clear()
        return self._chunk.clone()

    def reset(self) -> None:
        self.reset_count += 1


class _FakeTime:
    """Isolated two-read clock so pytest's own clock is not patched."""

    def __init__(self, engine: RTCInferenceEngine, inference_latency_s: float) -> None:
        self._engine = engine
        self._values = iter((10.0, 10.0 + inference_latency_s))
        self._last = 10.0 + inference_latency_s

    def perf_counter(self) -> float:
        return next(self._values, self._last)

    def sleep(self, _seconds: float) -> None:
        self._engine._shutdown_event.set()


class _FailingTrace(RealtimeTraceWriter):
    def __init__(self) -> None:
        self.write_count = 0
        self.abnormal_reason: object | None = None

    def write(self, event: str, **fields) -> None:
        del event, fields
        self.write_count += 1
        raise RealtimeTraceWriteError("synthetic trace write failure")

    def mark_abnormal(self, reason: object | None = None) -> None:
        self.abnormal_reason = reason

    def close(self, **_kwargs) -> None:
        pass


def _chunk(*, start: float, steps: int = 50) -> torch.Tensor:
    return torch.arange(start, start + steps, dtype=torch.float32).reshape(1, steps, 1)


def _make_engine(
    policy: _FakePolicy,
    *,
    timing_diagnostics: bool = True,
    enforce_guided_execution_window: bool = False,
    time_axis_planner=None,
    trace: RealtimeTraceWriter | None = None,
) -> tuple[RTCInferenceEngine, _IdentityProcessor, _IdentityProcessor]:
    preprocessor = _IdentityProcessor()
    postprocessor = _IdentityProcessor()
    rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=_FakeRobot(),
        rtc_config=rtc_config,
        hw_features={},
        task="cpu integration",
        fps=30.0,
        device="cpu",
        rtc_queue_threshold=30,
        rtc_timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=5,
        timing_diagnostics=timing_diagnostics,
        enforce_guided_execution_window=enforce_guided_execution_window,
        time_axis_planner=time_axis_planner,
        trace=trace,
    )
    engine._action_queue = ActionQueue(rtc_config)
    engine.notify_observation({})
    engine.resume()
    policy._pause_event = engine._policy_active
    return engine, preprocessor, postprocessor


def _run_one_inference(
    monkeypatch: pytest.MonkeyPatch,
    engine: RTCInferenceEngine,
    *,
    latency_s: float,
) -> None:
    engine._shutdown_event.clear()
    engine.resume()
    monkeypatch.setattr(rtc_module, "time", _FakeTime(engine, latency_s))
    engine._rtc_loop()
    assert engine.failed is False


def test_engine_reports_configured_replan_cadence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger=rtc_module.__name__)
    policy = _FakePolicy(_chunk(start=0.0))

    engine, _, _ = _make_engine(policy)

    assert engine._policy_chunk_size == 50
    assert "nominal_replan_interval_steps=20 nominal_replan_hz=1.500" in caplog.text


def test_empty_queue_ignores_thirteen_wall_steps_and_uses_fixed_guidance(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = _FakePolicy(_chunk(start=0.0))
    engine, _, _ = _make_engine(policy)
    caplog.set_level(logging.DEBUG, logger=rtc_module.__name__)

    # 420 ms at 30 Hz is ceil(12.6) == 13 wall-latency steps.
    _run_one_inference(monkeypatch, engine, latency_s=0.420)

    assert len(policy.calls) == 1
    assert policy.calls[0]["inference_delay"] == 5
    assert policy.calls[0]["prev_chunk_left_over"] is None
    assert torch.equal(engine.get_action(None), torch.tensor([0.0]))
    assert "wall_latency_steps=13" in caplog.text
    assert "actual_consumed_steps=0 merge_skip=0" in caplog.text
    assert "source_chunk_generation=1 next_model_action_index=0" in caplog.text
    assert "RTC action dequeue: source_chunk_generation=1 model_action_index=0" in caplog.text


def test_inference_trace_records_speed_adapter_reference_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _DiagnosticPlanner:
        @staticmethod
        def plan(actions, *, committed_prefix_steps: int) -> TimeAxisPlan:
            del committed_prefix_steps
            array = np.asarray(actions.detach().cpu())
            segments = len(array) - 1
            return TimeAxisPlan(
                actions=array,
                segment_durations=np.full(segments, 0.025),
                used_fallback=False,
                reference_segment_durations=np.full(segments, 0.025),
                speed_factors=np.full(segments, 2.0),
            )

    trace_path = tmp_path / "rtc_trace.jsonl"
    trace = RealtimeTraceWriter(trace_path)
    policy = _FakePolicy(_chunk(start=0.0))
    engine, _, _ = _make_engine(policy, time_axis_planner=_DiagnosticPlanner(), trace=trace)

    _run_one_inference(monkeypatch, engine, latency_s=0.1)
    trace.close()

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    event = next(row for row in rows if row["event"] == "inference_chunk")
    assert event["planner_speed_factors"] == [2.0] * 49
    assert event["planner_reference_segment_durations"] == [0.025] * 49
    assert event["planner_feature_coordinate_space"] is None


def test_inference_trace_write_failure_immediately_enters_fatal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _FailingTrace()
    policy = _FakePolicy(_chunk(start=0.0))
    engine, _, _ = _make_engine(policy, trace=trace)
    monkeypatch.setattr(rtc_module, "time", _FakeTime(engine, 0.1))

    engine._rtc_loop()

    assert engine.failed is True
    assert engine.fatal_error is not None
    assert str(engine.fatal_error) == "RTC deployment trace write failed"
    assert trace.write_count == 1
    assert trace.abnormal_reason is engine.fatal_error
    assert len(policy.calls) == 1


def test_existing_queue_skips_exactly_four_actions_consumed_during_inference(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = _FakePolicy(_chunk(start=1000.0))
    engine, _, _ = _make_engine(policy)
    queue = engine.action_queue
    assert queue is not None
    old_actions = _chunk(start=0.0, steps=20).squeeze(0)
    queue.merge(old_actions, old_actions, real_delay=0)
    consumed: list[torch.Tensor] = []

    def consume_four() -> None:
        for _ in range(4):
            action = engine.get_action(None)
            assert action is not None
            consumed.append(action)

    policy.on_predict = consume_four
    caplog.set_level(logging.DEBUG, logger=rtc_module.__name__)

    _run_one_inference(monkeypatch, engine, latency_s=0.150)

    assert torch.equal(torch.stack(consumed).flatten(), torch.arange(4, dtype=torch.float32))
    assert policy.calls[0]["inference_delay"] == 5
    assert torch.equal(engine.get_action(None), torch.tensor([1004.0]))
    assert queue.qsize() == 45
    assert "actual_consumed_steps=4 merge_skip=4" in caplog.text
    assert "source_chunk_generation=2 next_model_action_index=4" in caplog.text
    assert "RTC action dequeue: source_chunk_generation=2 model_action_index=4" in caplog.text


@pytest.mark.parametrize("timing_diagnostics", [False, True])
def test_guided_execution_window_stops_before_dispatching_tail_action(
    timing_diagnostics: bool,
) -> None:
    policy = _FakePolicy(_chunk(start=0.0))
    engine, _, _ = _make_engine(
        policy,
        timing_diagnostics=timing_diagnostics,
        enforce_guided_execution_window=True,
    )
    queue = engine.action_queue
    assert queue is not None
    actions = _chunk(start=0.0).squeeze(0)
    queue.merge(actions, actions, real_delay=0)

    dispatched = [engine.get_action(None) for _ in range(10)]

    assert all(action is not None for action in dispatched)
    assert torch.equal(torch.stack(dispatched).flatten(), torch.arange(10, dtype=torch.float32))
    assert engine.failed is False
    assert engine.get_action(None) is None
    assert engine.failed is True
    assert engine.fatal_error is not None
    assert "outside the guided execution window" in str(engine.fatal_error)
    assert queue.empty() is True


def test_consuming_execution_horizon_during_inference_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = _FakePolicy(_chunk(start=1000.0))
    engine, _, _ = _make_engine(policy)
    queue = engine.action_queue
    assert queue is not None
    old_actions = _chunk(start=0.0, steps=20).squeeze(0)
    queue.merge(old_actions, old_actions, real_delay=0)

    def consume_horizon() -> None:
        for _ in range(10):
            assert engine.get_action(None) is not None

    policy.on_predict = consume_horizon
    caplog.set_level(logging.INFO, logger=rtc_module.__name__)
    monkeypatch.setattr(rtc_module, "time", _FakeTime(engine, 0.350))

    engine._rtc_loop()

    assert engine.failed is True
    assert engine.fatal_error is not None
    assert "reached execution_horizon" in str(engine.fatal_error)
    assert queue.empty() is True
    assert engine.get_action(None) is None
    assert "refusing merge" in caplog.text


def test_reset_during_inference_discards_stale_result(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = _FakePolicy(_chunk(start=2000.0))
    engine, preprocessor, postprocessor = _make_engine(policy)
    queue = engine.action_queue
    assert queue is not None
    old_actions = _chunk(start=0.0, steps=20).squeeze(0)
    queue.merge(old_actions, old_actions, real_delay=0)
    inference_start_generation = queue.snapshot().generation
    policy.on_predict = engine.reset
    caplog.set_level(logging.INFO)

    _run_one_inference(monkeypatch, engine, latency_s=0.150)

    snapshot = queue.snapshot()
    assert snapshot.generation == inference_start_generation + 1
    assert snapshot.queue_size == 0
    assert engine.get_action(None) is None
    assert policy.reset_count == 1
    assert preprocessor.reset_count == 1
    assert postprocessor.reset_count == 1
    assert "RTC discarded stale inference result" in caplog.text


def test_shutdown_during_inference_discards_completed_result(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = _FakePolicy(_chunk(start=2500.0))
    engine, _, _ = _make_engine(policy)
    queue = engine.action_queue
    assert queue is not None
    policy.on_predict = engine._shutdown_event.set
    caplog.set_level(logging.INFO, logger=rtc_module.__name__)

    _run_one_inference(monkeypatch, engine, latency_s=0.150)

    assert queue.empty() is True
    assert engine.get_action(None) is None
    assert "RTC discarded inference result requested during shutdown" in caplog.text


def test_reset_after_observation_read_cannot_enqueue_old_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _FakePolicy(_chunk(start=3000.0))
    engine, _, _ = _make_engine(policy)
    queue = engine.action_queue
    assert queue is not None
    original_qsize = queue.qsize
    reset_done = False

    def reset_between_observation_and_snapshot() -> int:
        nonlocal reset_done
        if not reset_done:
            reset_done = True
            engine.reset()
            engine._shutdown_event.set()
        return original_qsize()

    monkeypatch.setattr(queue, "qsize", reset_between_observation_and_snapshot)
    monkeypatch.setattr(rtc_module, "time", _FakeTime(engine, 0.150))

    engine._rtc_loop()

    assert policy.calls == []
    assert queue.empty() is True
