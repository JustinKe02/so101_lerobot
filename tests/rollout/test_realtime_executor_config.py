from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lerobot.rollout.configs import (
    ActionOutputFilterConfig,
    RolloutConfig,
    SmoothExecutorConfig,
)
from lerobot.rollout.inference.factory import RTCInferenceConfig, SyncInferenceConfig
from lerobot.rollout.time_axis import TimeAxisPlannerConfig


def _rollout_config(**overrides) -> RolloutConfig:
    values = {
        "robot": SimpleNamespace(type="mock"),
        "policy": SimpleNamespace(type="pi05", device="cpu", chunk_size=50),
        "device": "cpu",
        "inference": RTCInferenceConfig(
            timing_mode="actual_consumed",
            guidance_delay_mode="fixed",
        ),
        "realtime_executor": SmoothExecutorConfig(enabled=True),
    }
    values.update(overrides)
    with patch("lerobot.rollout.configs.parser.get_path_arg", return_value=None):
        return RolloutConfig(**values)


def test_realtime_executor_config_accepts_complete_rtc_path() -> None:
    cfg = _rollout_config(time_axis_planner=TimeAxisPlannerConfig(enabled=True, dt_ref=1.0 / 30.0))

    assert cfg.realtime_executor.enabled is True


def test_realtime_executor_rejects_double_filtering() -> None:
    with pytest.raises(ValueError, match="avoid double filtering"):
        _rollout_config(action_filter=ActionOutputFilterConfig(enabled=True))


def test_realtime_executor_requires_rtc() -> None:
    with pytest.raises(ValueError, match="inference.type=rtc"):
        _rollout_config(inference=SyncInferenceConfig())


def test_realtime_executor_requires_matching_planner_period() -> None:
    with pytest.raises(ValueError, match="dt_ref must equal 1/fps"):
        _rollout_config(time_axis_planner=TimeAxisPlannerConfig(enabled=True, dt_ref=0.01))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"savgol_window_length": 2}, "odd and positive"),
        ({"savgol_window_length": 3, "savgol_polyorder": 3}, "smaller than the window"),
        ({"actuator_tau_s": 0.0}, "finite and positive"),
        ({"actuator_tau_s": [0.1, 0.0]}, "finite and positive"),
        ({"command_delay_s": [0.01, -0.01]}, "finite and non-negative"),
        ({"forward_feedback_gain": -1.0}, "finite and non-negative"),
    ],
)
def test_invalid_realtime_executor_parameters_fail_early(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SmoothExecutorConfig(**overrides)
