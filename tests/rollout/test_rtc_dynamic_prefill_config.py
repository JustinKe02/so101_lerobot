from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import draccus
import pytest
from draccus.utils import ParsingError

from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.inference.factory import (
    RTCInferenceConfig,
    RTCInferenceMode,
    RTCPrefillOverflowMode,
    RTCTimingMode,
)


def _dynamic_prefill_raw(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "mode": "trained_prefix",
        "timing_mode": "actual_consumed",
        "guidance_delay_mode": "fixed",
        "dynamic_prefill_enabled": True,
        "max_prefill_steps": 6,
    }
    values.update(overrides)
    return values


def _pi05_config(*, capacity: object = 6, chunk_size: int = 50) -> SimpleNamespace:
    return SimpleNamespace(
        type="pi05",
        chunk_size=chunk_size,
        rtc_training_max_delay=capacity,
    )


def test_rollout_draccus_decodes_dynamic_prefill_fields() -> None:
    raw = {
        "inference": {
            "type": "rtc",
            **_dynamic_prefill_raw(
                image_capture_delay_s=0.02,
                state_observation_delay_s=0.01,
                max_camera_skew_s=0.04,
                prefill_overflow="error",
            ),
        }
    }

    with patch.object(RolloutConfig, "__post_init__", lambda self: None):
        cfg = draccus.decode(RolloutConfig, raw)

    assert isinstance(cfg.inference, RTCInferenceConfig)
    assert cfg.inference.mode is RTCInferenceMode.TRAINED_PREFIX
    assert cfg.inference.timing_mode is RTCTimingMode.ACTUAL_CONSUMED
    assert cfg.inference.dynamic_prefill_enabled is True
    assert cfg.inference.max_prefill_steps == 6
    assert cfg.inference.image_capture_delay_s == pytest.approx(0.02)
    assert cfg.inference.state_observation_delay_s == pytest.approx(0.01)
    assert cfg.inference.max_camera_skew_s == pytest.approx(0.04)
    assert cfg.inference.prefill_overflow is RTCPrefillOverflowMode.ERROR


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mode": "guided"}, "requires mode='trained_prefix'"),
        ({"timing_mode": "legacy", "guidance_delay_mode": "legacy_max"}, "requires timing_mode"),
        ({"image_capture_delay_s": -0.01}, "image_capture_delay_s"),
        ({"state_observation_delay_s": float("inf")}, "state_observation_delay_s"),
        ({"max_camera_skew_s": float("nan")}, "max_camera_skew_s"),
        ({"max_prefill_steps": -1}, "max_prefill_steps"),
        ({"rtc": {"enabled": False}}, "requires rtc.enabled=true"),
        ({"prefill_overflow": "truncate_oldest"}, "truncation drops the image anchor"),
        ({"rtc": {"execution_horizon": 5}}, "must not exceed rtc.execution_horizon"),
        ({"max_prefill_steps": 5}, "include the completion anchor"),
    ],
)
def test_invalid_dynamic_prefill_combinations_fail_during_decode(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ParsingError) as exc_info:
        draccus.decode(RTCInferenceConfig, _dynamic_prefill_raw(**overrides))

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert message in str(exc_info.value.__cause__)


def test_direct_config_rejects_non_integer_max_prefill_steps() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        RTCInferenceConfig(**_dynamic_prefill_raw(max_prefill_steps=6.5))


@pytest.mark.parametrize("capacity", [0, -1, 6.5, True])
def test_trained_prefix_rejects_invalid_checkpoint_capacity(capacity: object) -> None:
    cfg = RTCInferenceConfig(**_dynamic_prefill_raw())

    with pytest.raises(ValueError, match="positive integer policy.rtc_training_max_delay"):
        cfg.validate_policy_config(_pi05_config(capacity=capacity))


def test_dynamic_prefill_accepts_exact_checkpoint_capacity() -> None:
    cfg = RTCInferenceConfig(**_dynamic_prefill_raw(max_prefill_steps=6))

    cfg.validate_policy_config(_pi05_config(capacity=6))


def test_zero_max_prefill_uses_checkpoint_capacity_sentinel() -> None:
    cfg = RTCInferenceConfig(**_dynamic_prefill_raw(max_prefill_steps=0))

    cfg.validate_policy_config(_pi05_config(capacity=6))


def test_zero_max_prefill_rejects_checkpoint_capacity_above_execution_horizon() -> None:
    cfg = RTCInferenceConfig(**_dynamic_prefill_raw(max_prefill_steps=0))

    with pytest.raises(ValueError, match="effective=12, execution_horizon=10"):
        cfg.validate_policy_config(_pi05_config(capacity=12))


def test_dynamic_prefill_rejects_capacity_plus_one() -> None:
    cfg = RTCInferenceConfig(**_dynamic_prefill_raw(max_prefill_steps=7))

    with pytest.raises(ValueError, match="requested=7, capacity=6"):
        cfg.validate_policy_config(_pi05_config(capacity=6))


def test_trained_prefix_rejects_non_pi05_policy() -> None:
    cfg = RTCInferenceConfig(**_dynamic_prefill_raw())
    policy_config = SimpleNamespace(type="pi0", chunk_size=50, rtc_training_max_delay=6)

    with pytest.raises(ValueError, match="requires a pi05 policy"):
        cfg.validate_policy_config(policy_config)


def test_fixed_trained_prefix_delay_cannot_exceed_checkpoint_capacity() -> None:
    cfg = RTCInferenceConfig(
        mode="trained_prefix",
        timing_mode="actual_consumed",
        guidance_delay_mode="fixed",
        fixed_guidance_delay_steps=6,
    )

    with pytest.raises(ValueError, match="requested=6, capacity=5"):
        cfg.validate_policy_config(_pi05_config(capacity=5))


def test_rollout_rejects_capacity_that_cannot_cover_image_delay() -> None:
    inference = RTCInferenceConfig(
        **_dynamic_prefill_raw(
            fixed_guidance_delay_steps=5,
            image_capture_delay_s=0.055,
            max_prefill_steps=6,
        )
    )
    values = {
        "robot": SimpleNamespace(type="mock"),
        "policy": SimpleNamespace(
            type="pi05",
            device="cpu",
            chunk_size=50,
            rtc_training_max_delay=6,
        ),
        "device": "cpu",
        "fps": 30.0,
        "inference": inference,
    }

    with (
        patch("lerobot.rollout.configs.parser.get_path_arg", return_value=None),
        pytest.raises(ValueError, match="minimum=7, capacity=6"),
    ):
        RolloutConfig(**values)
