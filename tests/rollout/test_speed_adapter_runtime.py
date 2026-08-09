from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import draccus
import pytest

import lerobot.rollout.context as context_module
from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.inference.factory import RTCInferenceConfig, SyncInferenceConfig
from lerobot.rollout.speed_adapter import SpeedAdapterRuntimeConfig
from lerobot.rollout.time_axis import TimeAxisPlannerConfig


def _rollout_config(**overrides) -> RolloutConfig:
    values = {
        "robot": SimpleNamespace(type="mock"),
        "policy": SimpleNamespace(
            type="pi05",
            device="cpu",
            chunk_size=50,
            action_feature_names=["joint_0.pos", "joint_1.pos"],
            rtc_training_max_delay=6,
        ),
        "device": "cpu",
        "fps": 20.0,
        "inference": RTCInferenceConfig(
            mode="trained_prefix",
            timing_mode="actual_consumed",
            guidance_delay_mode="fixed",
            dynamic_prefill_enabled=True,
            max_prefill_steps=6,
        ),
        "time_axis_planner": TimeAxisPlannerConfig(enabled=True, dt_ref=0.05),
        "speed_adapter": SpeedAdapterRuntimeConfig(
            enabled=True,
            checkpoint="/checkpoint/speed_adapter",
        ),
    }
    values.update(overrides)
    with patch("lerobot.rollout.configs.parser.get_path_arg", return_value=None):
        return RolloutConfig(**values)


def test_draccus_decodes_speed_adapter_runtime_config() -> None:
    raw = {
        "speed_adapter": {
            "enabled": True,
            "checkpoint": "/checkpoint/speed_adapter",
            "device": "cpu",
            "verify_checksum": True,
            "require_trained_checkpoint": True,
        }
    }

    with patch.object(RolloutConfig, "__post_init__", lambda self: None):
        cfg = draccus.decode(RolloutConfig, raw)

    assert cfg.speed_adapter == SpeedAdapterRuntimeConfig(
        enabled=True,
        checkpoint="/checkpoint/speed_adapter",
    )


def test_speed_adapter_requires_rtc_and_enabled_time_axis_planner() -> None:
    with pytest.raises(ValueError, match="inference.type=rtc"):
        _rollout_config(inference=SyncInferenceConfig())

    with pytest.raises(ValueError, match="time_axis_planner.enabled=true"):
        _rollout_config(time_axis_planner=TimeAxisPlannerConfig(enabled=False))


def test_speed_adapter_reference_period_must_match_control_period() -> None:
    with pytest.raises(ValueError, match="dt_ref must equal 1/fps"):
        _rollout_config(time_axis_planner=TimeAxisPlannerConfig(enabled=True, dt_ref=0.04))

    cfg = _rollout_config()
    assert cfg.speed_adapter.enabled


def test_invalid_enabled_checkpoint_fails_before_policy_or_hardware_loading() -> None:
    cfg = _rollout_config(
        speed_adapter=SpeedAdapterRuntimeConfig(
            enabled=True,
            checkpoint="/definitely/missing/speed_adapter",
        )
    )
    robot_factory = Mock(side_effect=AssertionError("hardware must not be opened"))
    policy_factory = Mock(side_effect=AssertionError("policy need not load after invalid adapter"))

    with (
        patch.object(context_module, "make_robot_from_config", robot_factory),
        patch.object(context_module, "get_policy_class", policy_factory),
        pytest.raises(RuntimeError, match="failed to load enabled speed adapter checkpoint"),
    ):
        context_module.build_rollout_context(cfg, Event())

    robot_factory.assert_not_called()
    policy_factory.assert_not_called()
