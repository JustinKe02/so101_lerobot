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

"""CPU-only compatibility gates for rollout Phase 0-2."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import draccus
import pytest
from draccus.utils import ParsingError

from lerobot.rollout.configs import (
    PI05ActionBackend,
    PI05PrefixBackend,
    RolloutConfig,
)
from lerobot.rollout.inference.factory import (
    RTCGuidanceDelayMode,
    RTCInferenceConfig,
    RTCTimingMode,
    create_inference_engine,
)
from lerobot.rollout.inference.rtc import RTCInferenceEngine


def _make_rollout_config(**overrides) -> RolloutConfig:
    values = {
        "robot": SimpleNamespace(type="mock"),
        "policy": SimpleNamespace(type="pi05", device="cpu"),
        "device": "cpu",
    }
    values.update(overrides)
    with patch("lerobot.rollout.configs.parser.get_path_arg", return_value=None):
        return RolloutConfig(**values)


def test_rtc_phase1_defaults_preserve_legacy_behavior() -> None:
    cfg = RTCInferenceConfig()

    assert cfg.timing_mode is RTCTimingMode.LEGACY
    assert cfg.guidance_delay_mode is RTCGuidanceDelayMode.LEGACY_MAX
    assert cfg.fixed_guidance_delay_steps == 5
    assert cfg.latency_warmup_inferences == 5
    assert cfg.latency_window_size == 32
    assert cfg.latency_percentile == 0.95
    assert cfg.delay_hysteresis_steps == 0.25
    assert cfg.delay_change_confirmations == 3
    assert cfg.timing_diagnostics is False


def test_negative_queue_threshold_is_rejected_by_config() -> None:
    with pytest.raises(ValueError, match="queue_threshold must be non-negative"):
        RTCInferenceConfig(queue_threshold=-1)


def test_rollout_rejects_queue_threshold_at_policy_chunk_size() -> None:
    inference = RTCInferenceConfig(
        queue_threshold=50,
        timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
    )
    policy = SimpleNamespace(type="pi05", device="cpu", chunk_size=50)

    with pytest.raises(ValueError, match="smaller than policy chunk_size"):
        _make_rollout_config(inference=inference, policy=policy)


def test_old_rollout_json_decodes_to_legacy_pytorch_defaults() -> None:
    old_json = {"inference": {"type": "rtc", "queue_threshold": 20}}
    explicit_legacy_json = {
        "inference": {
            "type": "rtc",
            "queue_threshold": 20,
            "timing_mode": "legacy",
            "guidance_delay_mode": "legacy_max",
        },
        "pi05_prefix_backend": "auto",
        "pi05_action_backend": "pytorch",
    }

    # Decode only: RolloutConfig.__post_init__ performs runtime policy/device
    # resolution, which is intentionally outside this serialized-config gate.
    with patch.object(RolloutConfig, "__post_init__", lambda self: None):
        old_cfg = draccus.decode(RolloutConfig, old_json)
        explicit_cfg = draccus.decode(RolloutConfig, explicit_legacy_json)

    assert old_cfg.inference == explicit_cfg.inference
    assert old_cfg.inference.timing_mode is RTCTimingMode.LEGACY
    assert old_cfg.inference.guidance_delay_mode is RTCGuidanceDelayMode.LEGACY_MAX
    assert old_cfg.pi05_prefix_backend is PI05PrefixBackend.AUTO
    assert old_cfg.pi05_action_backend is PI05ActionBackend.PYTORCH
    assert old_cfg.pi05_tensorrt_prefix_engine is None
    assert old_cfg.pi05_tensorrt_action_engine is None
    assert old_cfg.seed is None


@pytest.mark.parametrize("guidance_mode", ["fixed", "rolling_p95"])
def test_actual_consumed_guidance_modes_decode(guidance_mode: str) -> None:
    cfg = draccus.decode(
        RTCInferenceConfig,
        {"timing_mode": "actual_consumed", "guidance_delay_mode": guidance_mode},
    )

    assert cfg.timing_mode is RTCTimingMode.ACTUAL_CONSUMED
    assert cfg.guidance_delay_mode.value == guidance_mode


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"guidance_delay_mode": "fixed"}, "legacy timing requires"),
        ({"guidance_delay_mode": "rolling_p95"}, "legacy timing requires"),
        (
            {"enforce_guided_execution_window": True},
            "guided execution window enforcement requires",
        ),
        ({"timing_mode": "actual_consumed"}, "requires guidance_delay_mode"),
        (
            {
                "timing_mode": "actual_consumed",
                "guidance_delay_mode": "fixed",
                "fixed_guidance_delay_steps": 10,
            },
            "smaller than execution_horizon",
        ),
        (
            {
                "timing_mode": "actual_consumed",
                "guidance_delay_mode": "rolling_p95",
                "latency_window_size": 4,
            },
            "latency_window_size must be at least 5",
        ),
    ],
)
def test_invalid_rtc_timing_combinations_fail_fast(raw: dict[str, str], message: str) -> None:
    with pytest.raises(ParsingError) as exc_info:
        draccus.decode(RTCInferenceConfig, raw)

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert message in str(exc_info.value.__cause__)


def test_direct_rtc_engine_construction_rejects_unknown_timing() -> None:
    with pytest.raises(ValueError, match="Unsupported RTC timing mode"):
        RTCInferenceEngine(
            policy=None,
            preprocessor=None,
            postprocessor=None,
            robot_wrapper=None,
            rtc_config=SimpleNamespace(execution_horizon=10),
            hw_features={},
            task="",
            fps=30.0,
            device="cpu",
            rtc_timing_mode="unknown",
        )


def test_direct_rtc_engine_construction_accepts_actual_consumed_fixed() -> None:
    engine = RTCInferenceEngine(
        policy=SimpleNamespace(config=SimpleNamespace()),
        preprocessor=SimpleNamespace(steps=[]),
        postprocessor=SimpleNamespace(),
        robot_wrapper=SimpleNamespace(action_features={}),
        rtc_config=SimpleNamespace(execution_horizon=10),
        hw_features={},
        task="",
        fps=30.0,
        device="cpu",
        rtc_timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=5,
    )

    assert engine._rtc_timing_mode == "actual_consumed"
    assert engine._guidance_delay_mode == "fixed"


def test_direct_rtc_engine_rejects_queue_threshold_at_policy_chunk_size() -> None:
    with pytest.raises(ValueError, match="smaller than policy chunk_size"):
        RTCInferenceEngine(
            policy=SimpleNamespace(config=SimpleNamespace(chunk_size=50)),
            preprocessor=SimpleNamespace(steps=[]),
            postprocessor=SimpleNamespace(),
            robot_wrapper=SimpleNamespace(action_features={}),
            rtc_config=SimpleNamespace(execution_horizon=10),
            hw_features={},
            task="",
            fps=30.0,
            device="cpu",
            rtc_queue_threshold=50,
            rtc_timing_mode="actual_consumed",
            guidance_delay_mode="fixed",
        )


def test_legacy_engine_preserves_queue_threshold_at_policy_chunk_size() -> None:
    engine = RTCInferenceEngine(
        policy=SimpleNamespace(config=SimpleNamespace(chunk_size=8)),
        preprocessor=SimpleNamespace(steps=[]),
        postprocessor=SimpleNamespace(),
        robot_wrapper=SimpleNamespace(action_features={}),
        rtc_config=SimpleNamespace(execution_horizon=10),
        hw_features={},
        task="",
        fps=30.0,
        device="cpu",
        rtc_queue_threshold=30,
    )

    assert engine._rtc_timing_mode == "legacy"
    assert engine._rtc_queue_threshold == 30


def test_rollout_rejects_actual_consumed_with_interpolation() -> None:
    inference = RTCInferenceConfig(
        timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
    )

    with pytest.raises(ValueError, match="requires --interpolation_multiplier=1"):
        _make_rollout_config(inference=inference, interpolation_multiplier=2)


def test_factory_passes_actual_consumed_guidance_configuration() -> None:
    config = RTCInferenceConfig(
        timing_mode="actual_consumed",
        guidance_delay_mode="rolling_p95",
        fixed_guidance_delay_steps=4,
        latency_warmup_inferences=3,
        latency_window_size=16,
        latency_percentile=0.9,
        delay_hysteresis_steps=0.5,
        delay_change_confirmations=2,
        timing_diagnostics=True,
        enforce_guided_execution_window=True,
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
            ordered_action_keys=["motor.pos"],
            task="test",
            fps=30.0,
            device="cpu",
        )

    kwargs = engine_cls.call_args.kwargs
    assert kwargs["rtc_timing_mode"] == "actual_consumed"
    assert kwargs["guidance_delay_mode"] == "rolling_p95"
    assert kwargs["fixed_guidance_delay_steps"] == 4
    assert kwargs["latency_warmup_inferences"] == 3
    assert kwargs["latency_window_size"] == 16
    assert kwargs["latency_percentile"] == 0.9
    assert kwargs["delay_hysteresis_steps"] == 0.5
    assert kwargs["delay_change_confirmations"] == 2
    assert kwargs["timing_diagnostics"] is True
    assert kwargs["enforce_guided_execution_window"] is True


def test_default_pi05_backends_select_only_pytorch() -> None:
    cfg = _make_rollout_config()

    assert cfg.pi05_prefix_backend is PI05PrefixBackend.AUTO
    assert cfg.pi05_action_backend is PI05ActionBackend.PYTORCH
    assert cfg.use_pi05_tensorrt_prefix is False


def test_explicit_prefix_pytorch_ignores_stale_engine(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _make_rollout_config(
        pi05_prefix_backend="pytorch",
        pi05_tensorrt_prefix_engine="stale-prefix.plan",
    )

    assert cfg.pi05_prefix_backend is PI05PrefixBackend.PYTORCH
    assert cfg.use_pi05_tensorrt_prefix is False
    assert "Ignoring PI0.5 prefix engine" in caplog.text


def test_explicit_action_pytorch_ignores_stale_engine(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _make_rollout_config(
        pi05_action_backend="pytorch",
        pi05_tensorrt_action_engine="stale-action.plan",
    )

    assert cfg.pi05_action_backend is PI05ActionBackend.PYTORCH
    assert "Ignoring PI0.5 action engine" in caplog.text


def test_explicit_prefix_tensorrt_requires_engine() -> None:
    with pytest.raises(ValueError, match="requires --pi05_tensorrt_prefix_engine"):
        _make_rollout_config(pi05_prefix_backend="tensorrt")


def test_explicit_action_tensorrt_requires_engine() -> None:
    with pytest.raises(ValueError, match="requires --pi05_tensorrt_action_engine"):
        _make_rollout_config(pi05_action_backend="tensorrt")


def test_action_tensorrt_placeholder_fails_closed() -> None:
    with pytest.raises(ValueError, match="action backend is not implemented yet"):
        _make_rollout_config(
            pi05_action_backend="tensorrt",
            pi05_tensorrt_action_engine="action.plan",
        )


def test_default_rollout_import_does_not_import_tensorrt_runtime() -> None:
    code = """
import sys
from lerobot.rollout.configs import PI05ActionBackend, PI05PrefixBackend
from lerobot.rollout.inference.factory import RTCInferenceConfig

cfg = RTCInferenceConfig()
assert cfg.timing_mode.value == "legacy"
assert PI05PrefixBackend.AUTO.value == "auto"
assert PI05ActionBackend.PYTORCH.value == "pytorch"
assert "tensorrt" not in sys.modules
assert "lerobot.policies.pi05.tensorrt_prefix" not in sys.modules
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
