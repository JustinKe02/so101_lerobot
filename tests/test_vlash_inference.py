#!/usr/bin/env python

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

from __future__ import annotations

import time
from collections import deque
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import draccus
import numpy as np
import pytest
import torch

from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.inference.factory import VLASHInferenceConfig, create_inference_engine
from lerobot.rollout.inference.vlash import VLASHInferenceEngine


class _Resettable:
    def reset(self) -> None:
        pass


class _Policy(_Resettable):
    def __init__(self, *, chunk_size: int = 4, offset_steps: int = 3, state_cond: bool = True):
        self.config = SimpleNamespace(
            type="pi05",
            chunk_size=chunk_size,
            temporal_offset_max_steps=offset_steps,
            state_cond=state_cond,
        )


def _make_engine(**overrides) -> VLASHInferenceEngine:
    values = {
        "policy": _Policy(),
        "preprocessor": _Resettable(),
        "postprocessor": _Resettable(),
        "robot_wrapper": SimpleNamespace(robot_type="mock"),
        "hw_features": {},
        "ordered_action_keys": ["joint.pos"],
        "task": "test",
        "fps": 30.0,
        "device": "cpu",
        "execution_horizon": 4,
        "inference_overlap_steps": 2,
    }
    values.update(overrides)
    return VLASHInferenceEngine(**values)


def _wait_until(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for VLASH background inference")
        time.sleep(0.005)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"execution_horizon": 0}, "positive integer"),
        ({"execution_horizon": 2, "inference_overlap_steps": 3}, "cannot exceed"),
        ({"max_future_state_delta": float("inf")}, "finite and positive"),
        ({"deadline_miss_limit": True}, "positive integer"),
    ],
)
def test_vlash_config_rejects_invalid_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VLASHInferenceConfig(**kwargs)


def test_vlash_config_validates_checkpoint_contract() -> None:
    config = VLASHInferenceConfig(execution_horizon=4, inference_overlap_steps=2)

    config.validate_policy(_Policy(chunk_size=4, offset_steps=3).config)
    with pytest.raises(ValueError, match="state-conditioned"):
        config.validate_policy(_Policy(state_cond=False).config)
    with pytest.raises(ValueError, match="trained_offset"):
        config.validate_policy(_Policy(offset_steps=1).config)
    with pytest.raises(ValueError, match="cannot exceed"):
        config.validate_policy(_Policy(chunk_size=3).config)


def test_rollout_config_decodes_vlash_choice() -> None:
    with patch.object(RolloutConfig, "__post_init__", lambda self: None):
        config = draccus.decode(
            RolloutConfig,
            {
                "inference": {
                    "type": "vlash",
                    "execution_horizon": 8,
                    "inference_overlap_steps": 4,
                }
            },
        )

    assert isinstance(config.inference, VLASHInferenceConfig)
    assert config.inference.execution_horizon == 8
    assert config.inference.inference_overlap_steps == 4


def test_rollout_config_rejects_vlash_interpolation() -> None:
    with (
        patch("lerobot.rollout.configs.parser.get_path_arg", return_value=None),
        pytest.raises(ValueError, match="requires --interpolation_multiplier=1"),
    ):
        RolloutConfig(
            robot=SimpleNamespace(type="mock"),
            policy=_Policy().config,
            inference=VLASHInferenceConfig(execution_horizon=4, inference_overlap_steps=2),
            interpolation_multiplier=2,
            device="cpu",
        )


def test_vlash_ablation_flags_are_explicit() -> None:
    config = VLASHInferenceConfig(
        require_state_conditioning=False,
        require_offset_training=False,
    )
    policy = _Policy(chunk_size=10, offset_steps=0, state_cond=False)

    config.validate_policy(policy.config)


def test_future_state_estimation_uses_absolute_targets_and_optional_clamp() -> None:
    remaining = torch.tensor([[10.0, -10.0], [20.0, -20.0]])
    current = np.array([0.0, 0.0], dtype=np.float32)

    unconstrained = _make_engine()._estimate_future_state(current, remaining)
    constrained = _make_engine(max_future_state_delta=5.0)._estimate_future_state(current, remaining)
    tracking_aware = _make_engine(max_future_state_delta=5.0)._estimate_future_state(
        current,
        remaining,
        tracking_gain=torch.tensor([0.2, 0.5]),
    )

    np.testing.assert_array_equal(unconstrained, np.array([20.0, -20.0], dtype=np.float32))
    np.testing.assert_array_equal(constrained, np.array([10.0, -10.0], dtype=np.float32))
    np.testing.assert_array_equal(tracking_aware, np.array([2.0, -5.0], dtype=np.float32))


def test_action_feedback_projects_from_applied_target_and_observed_tracking() -> None:
    engine = _make_engine(max_future_state_delta=5.0)
    engine._initialized = True
    engine._active_actions.extend([torch.tensor([20.0])])
    engine._awaiting_action_feedback = torch.tensor([10.0])
    engine._previous_feedback_state = torch.tensor([0.0])
    engine._previous_applied_action = torch.tensor([5.0])
    engine.notify_observation({"joint.pos": 1.0})

    engine.notify_action_result(
        requested={"joint.pos": 10.0},
        sent={"joint.pos": 6.0},
        observation={"joint.pos": 1.0},
    )

    request = engine._take_request()
    assert request is not None
    assert request.projected_steps == 2
    np.testing.assert_allclose(request.future_state, np.array([3.0], dtype=np.float32))
    stats = engine.stats_snapshot()
    assert stats.action_feedback_count == 1
    assert stats.hardware_clamp_count == 1
    assert stats.action_filter_rewrite_count == 0
    assert stats.tracking_gain_mean == pytest.approx(0.2)


def test_clear_latency_window_preserves_cumulative_counters() -> None:
    engine = _make_engine()
    engine._inference_count = 3
    engine._deadline_misses = 1
    engine._latencies.extend([0.1, 0.2])

    engine.clear_latency_window()

    stats = engine.stats_snapshot()
    assert stats.inference_count == 3
    assert stats.deadline_misses == 1
    assert stats.latency_p50_ms is None
    assert stats.latency_p95_ms is None
    assert stats.latency_max_ms is None


def test_async_chunk_switch_keeps_full_future_tail() -> None:
    engine = _make_engine()
    chunks = deque(
        [
            torch.arange(10.0, 14.0).unsqueeze(-1),
            torch.arange(20.0, 24.0).unsqueeze(-1),
        ]
    )
    requests: list[np.ndarray | None] = []

    def run_policy(observation: dict, future_state: np.ndarray | None) -> torch.Tensor:
        del observation
        requests.append(None if future_state is None else future_state.copy())
        return chunks.popleft()

    engine._run_policy = run_policy
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"joint.pos": 0.0})
        _wait_until(lambda: engine.stats_snapshot().active_actions == 4)

        engine.notify_observation({"joint.pos": 0.0})
        assert engine.get_action(None).item() == 10.0
        engine.notify_action_result({"joint.pos": 10.0}, {"joint.pos": 10.0}, {"joint.pos": 0.0})
        engine.notify_observation({"joint.pos": 10.0})
        assert engine.get_action(None).item() == 11.0
        engine.notify_action_result({"joint.pos": 11.0}, {"joint.pos": 11.0}, {"joint.pos": 10.0})

        _wait_until(lambda: engine.stats_snapshot().pending_actions == 4)
        np.testing.assert_array_equal(requests[1], np.array([13.0], dtype=np.float32))

        assert engine.get_action(None).item() == 12.0
        assert engine.get_action(None).item() == 13.0
        assert engine.get_action(None).item() == 20.0
    finally:
        engine.stop()


def test_compile_ready_waits_for_first_non_warmup_chunk() -> None:
    second_inference_started = Event()
    release_second_inference = Event()
    calls = 0
    engine = _make_engine(use_torch_compile=True, compile_warmup_inferences=1)

    def run_policy(observation: dict, future_state: np.ndarray | None) -> torch.Tensor:
        nonlocal calls
        del observation, future_state
        calls += 1
        if calls == 2:
            second_inference_started.set()
            release_second_inference.wait(timeout=2.0)
        return torch.arange(4.0).unsqueeze(-1)

    engine._run_policy = run_policy
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"joint.pos": 0.0})
        assert second_inference_started.wait(timeout=2.0)
        assert engine.ready is False

        release_second_inference.set()
        _wait_until(lambda: engine.ready)
        assert engine.stats_snapshot().active_actions == 4
    finally:
        release_second_inference.set()
        engine.stop()


def test_deadline_miss_fails_closed_and_propagates_shutdown() -> None:
    second_inference_started = Event()
    release_second_inference = Event()
    global_shutdown = Event()
    calls = 0
    engine = _make_engine(
        policy=_Policy(chunk_size=2, offset_steps=1),
        execution_horizon=2,
        inference_overlap_steps=1,
        shutdown_event=global_shutdown,
    )

    def run_policy(observation: dict, future_state: np.ndarray | None) -> torch.Tensor:
        nonlocal calls
        del observation, future_state
        calls += 1
        if calls == 2:
            second_inference_started.set()
            release_second_inference.wait(timeout=2.0)
        return torch.tensor([[1.0], [2.0]])

    engine._run_policy = run_policy
    engine.start()
    engine.resume()
    try:
        engine.notify_observation({"joint.pos": 0.0})
        _wait_until(lambda: engine.stats_snapshot().active_actions == 2)
        engine.notify_observation({"joint.pos": 0.0})
        first_action = engine.get_action(None)
        assert first_action is not None
        engine.notify_action_result(
            {"joint.pos": first_action.item()},
            {"joint.pos": first_action.item()},
            {"joint.pos": 0.0},
        )
        engine.notify_observation({"joint.pos": 1.0})
        assert engine.get_action(None) is not None
        assert second_inference_started.wait(timeout=2.0)

        assert engine.get_action(None) is None
        assert engine.failed is True
        assert global_shutdown.is_set()
        assert isinstance(engine.fatal_error, RuntimeError)
    finally:
        release_second_inference.set()
        engine.stop()


def test_factory_returns_vlash_engine() -> None:
    config = VLASHInferenceConfig(execution_horizon=4, inference_overlap_steps=2)
    engine = create_inference_engine(
        config,
        policy=_Policy(),
        preprocessor=_Resettable(),
        postprocessor=_Resettable(),
        robot_wrapper=SimpleNamespace(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["joint.pos"],
        task="test",
        fps=30.0,
        device="cpu",
    )

    assert isinstance(engine, VLASHInferenceEngine)
