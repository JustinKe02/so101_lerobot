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

"""Tests for the second-order rollout action output filter (strategy doc §6)."""

import math

import pytest
import torch

from lerobot.rollout.action_filter import (
    DEFAULT_MAX_ACCELERATION,
    DEFAULT_MAX_VELOCITY,
    ActionDictOutputFilter,
    SecondOrderActionFilter,
    build_limit_tensor,
    resolve_joint_limit,
)
from lerobot.rollout.configs import ActionOutputFilterConfig

FPS = 30.0
DT = 1.0 / FPS
JOINTS = list(DEFAULT_MAX_VELOCITY)
ACTION_KEYS = [f"{joint}.pos" for joint in JOINTS]
# Hardware rate limiter on the SO-101 follower (max_relative_target) that the
# filter must keep un-exercised during nominal motion.
HARDWARE_CLAMP = 5.0


def make_filter(joints=JOINTS, dt=DT) -> SecondOrderActionFilter:
    keys = [f"{j}.pos" for j in joints]
    return SecondOrderActionFilter(
        build_limit_tensor(keys, DEFAULT_MAX_VELOCITY),
        build_limit_tensor(keys, DEFAULT_MAX_ACCELERATION),
        dt,
    )


def make_single(joint: str, dt=DT) -> SecondOrderActionFilter:
    return make_filter([joint], dt=dt)


# ---------------------------------------------------------------------------
# Limit resolution / validation
# ---------------------------------------------------------------------------


def test_resolve_joint_limit_exact_match_wins():
    limits = {"shoulder_pan": 10.0, "shoulder_pan.pos": 20.0}
    assert resolve_joint_limit("shoulder_pan.pos", limits) == 20.0


def test_resolve_joint_limit_strips_field_suffix():
    assert resolve_joint_limit("elbow_flex.pos", DEFAULT_MAX_VELOCITY) == 95.0


def test_resolve_joint_limit_missing_fails_closed():
    with pytest.raises(KeyError, match="unknown_joint.pos"):
        resolve_joint_limit("unknown_joint.pos", DEFAULT_MAX_VELOCITY)


def test_build_limit_tensor_preserves_key_order():
    tensor = build_limit_tensor(ACTION_KEYS, DEFAULT_MAX_VELOCITY)
    assert tensor.dtype == torch.float64
    assert tensor.tolist() == [DEFAULT_MAX_VELOCITY[j] for j in JOINTS]


@pytest.mark.parametrize("bad_value", [0.0, -5.0, float("inf"), float("nan")])
def test_build_limit_tensor_rejects_nonpositive_or_nonfinite(bad_value):
    limits = dict(DEFAULT_MAX_VELOCITY, elbow_flex=bad_value)
    with pytest.raises(ValueError, match="elbow_flex"):
        build_limit_tensor(ACTION_KEYS, limits)


def test_filter_constructor_rejects_bad_dt():
    v = build_limit_tensor(ACTION_KEYS, DEFAULT_MAX_VELOCITY)
    a = build_limit_tensor(ACTION_KEYS, DEFAULT_MAX_ACCELERATION)
    for bad_dt in (0.0, -0.01, float("nan")):
        with pytest.raises(ValueError):
            SecondOrderActionFilter(v, a, bad_dt)


# ---------------------------------------------------------------------------
# Core dynamics
# ---------------------------------------------------------------------------


def test_step_before_seed_raises():
    filt = make_filter()
    with pytest.raises(RuntimeError, match="seed"):
        filt.step(torch.zeros(len(JOINTS)))


def test_seed_validation():
    filt = make_filter()
    with pytest.raises(ValueError, match="dim"):
        filt.seed(torch.zeros(len(JOINTS) + 1))
    with pytest.raises(ValueError, match="finite"):
        filt.seed(torch.full((len(JOINTS),), float("nan")))


def test_passthrough_below_caps():
    """Targets moving well below every cap must pass through unchanged."""
    filt = make_filter()
    q = torch.zeros(len(JOINTS), dtype=torch.float64)
    filt.seed(q)
    # 0.05 units/tick = 1.5 units/s, far below every v_max; the required
    # acceleration (45 units/s^2 on the first tick) is below every a_max.
    for tick in range(1, 50):
        target = q + 0.05 * tick
        out = filt.step(target)
        assert torch.allclose(out.to(torch.float64), target, atol=1e-9)
    assert filt.intervention_count == 0
    assert filt.max_abs_intervention == 0.0


def test_acceleration_ramp_from_rest():
    """From rest toward a far target the velocity builds by exactly a_max*dt per tick."""
    joint = "shoulder_pan"
    a_max = DEFAULT_MAX_ACCELERATION[joint]
    v_max = DEFAULT_MAX_VELOCITY[joint]
    filt = make_single(joint)
    filt.seed(torch.zeros(1))
    target = torch.tensor([1000.0], dtype=torch.float64)
    prev_q = 0.0
    for tick in range(1, 5):
        out = float(filt.step(target)[0])
        expected_v = min(tick * a_max * DT, v_max)
        assert math.isclose(out - prev_q, expected_v * DT, rel_tol=1e-9)
        prev_q = out


def test_velocity_cap_at_cruise():
    """Once at cruise the per-tick advance equals exactly v_max*dt."""
    joint = "wrist_roll"
    v_max = DEFAULT_MAX_VELOCITY[joint]
    a_max = DEFAULT_MAX_ACCELERATION[joint]
    filt = make_single(joint)
    filt.seed(torch.zeros(1))
    target = torch.tensor([1000.0], dtype=torch.float64)
    ramp_ticks = math.ceil(v_max / (a_max * DT))
    prev_q = 0.0
    for tick in range(1, ramp_ticks + 10):
        out = float(filt.step(target)[0])
        if tick > ramp_ticks:
            assert math.isclose(out - prev_q, v_max * DT, rel_tol=1e-9)
        prev_q = out
    assert filt.intervention_count > 0


def test_per_tick_delta_never_exceeds_hardware_clamp():
    """Worst-case step demand: every per-tick move stays <= v_max*dt < 5.0 units.

    This is the property that closes the zero-clamp T0a gate: the doc caps give
    at most 102/30 = 3.4 units per tick, under the follower's
    max_relative_target = 5.0, so the hardware rate limiter never fires.
    """
    for joint in JOINTS:
        v_max = DEFAULT_MAX_VELOCITY[joint]
        assert v_max * DT < HARDWARE_CLAMP  # doc caps must clear the limiter
        for direction in (1.0, -1.0):
            filt = make_single(joint)
            filt.seed(torch.zeros(1))
            target = torch.tensor([direction * 500.0], dtype=torch.float64)
            prev_q = 0.0
            for _ in range(300):
                out = float(filt.step(target)[0])
                assert abs(out - prev_q) <= v_max * DT + 1e-9
                assert abs(out - prev_q) < HARDWARE_CLAMP
                prev_q = out


def test_step_response_converges_with_bounded_overshoot():
    """Step response settles exactly on the target with overshoot below v_max^2/(2*a_max) + v_max*dt.

    The doc §6 equations brake proportionally (no anticipatory braking), so a
    pure step overshoots by up to ~v_max^2/(2*a_max) before converging.  The
    per-tick bound above still holds throughout, so this is a tracking
    property, not a safety one.
    """
    for joint in JOINTS:
        v_max = DEFAULT_MAX_VELOCITY[joint]
        a_max = DEFAULT_MAX_ACCELERATION[joint]
        overshoot_bound = v_max**2 / (2.0 * a_max) + v_max * DT
        filt = make_single(joint)
        filt.seed(torch.zeros(1))
        target_value = 50.0
        target = torch.tensor([target_value], dtype=torch.float64)
        trajectory = [float(filt.step(target)[0]) for _ in range(600)]
        assert max(trajectory) <= target_value + overshoot_bound + 1e-6, joint
        assert math.isclose(trajectory[-1], target_value, abs_tol=1e-6), joint


def test_direction_reversal_respects_acceleration_limit():
    joint = "elbow_flex"
    a_max = DEFAULT_MAX_ACCELERATION[joint]
    filt = make_single(joint)
    filt.seed(torch.zeros(1))
    forward = torch.tensor([1000.0], dtype=torch.float64)
    backward = torch.tensor([-1000.0], dtype=torch.float64)
    prev_q, prev_v = 0.0, 0.0
    for tick in range(60):
        target = forward if tick < 30 else backward
        out = float(filt.step(target)[0])
        v = (out - prev_q) / DT
        assert abs(v - prev_v) <= a_max * DT + 1e-9
        prev_q, prev_v = out, v


def test_filter_sequence_matches_manual_steps_and_is_deterministic():
    filt = make_filter()
    torch.manual_seed(0)
    targets = torch.cumsum(torch.randn(20, len(JOINTS)) * 3.0, dim=0)
    initial = torch.zeros(len(JOINTS))

    first = filt.filter_sequence(targets, initial)
    second = filt.filter_sequence(targets, initial)
    assert torch.equal(first, second)  # filter_sequence resets state each call

    manual = make_filter()
    manual.seed(initial)
    rows = torch.stack([manual.step(row) for row in targets])
    assert torch.allclose(first, rows, atol=1e-9)


# ---------------------------------------------------------------------------
# Dict wrapper (dispatch-path interface)
# ---------------------------------------------------------------------------


def obs_at(value: float) -> dict:
    obs = dict.fromkeys(ACTION_KEYS, value)
    obs["extra_sensor"] = 42.0
    return obs


def test_dict_filter_lazy_seeds_from_observation_and_passes_through():
    filt = ActionDictOutputFilter(ACTION_KEYS, DEFAULT_MAX_VELOCITY, DEFAULT_MAX_ACCELERATION, DT)
    action = dict.fromkeys(ACTION_KEYS, 10.0)
    action["unrelated"] = "keep-me"
    out = filt.apply(action, obs_at(10.0))
    # Seeded at the measured pose == target: identity, non-filter keys preserved.
    for key in ACTION_KEYS:
        assert out[key] == pytest.approx(10.0, abs=1e-9)
    assert out["unrelated"] == "keep-me"
    assert action[ACTION_KEYS[0]] == 10.0  # input dict not mutated


def test_dict_filter_limits_large_jump_from_seed():
    filt = ActionDictOutputFilter(ACTION_KEYS, DEFAULT_MAX_VELOCITY, DEFAULT_MAX_ACCELERATION, DT)
    action = dict.fromkeys(ACTION_KEYS, 100.0)
    out = filt.apply(action, obs_at(0.0))
    for key in ACTION_KEYS:
        joint = key.split(".", 1)[0]
        # From rest the first tick is acceleration-limited: a_max * dt^2.
        expected = DEFAULT_MAX_ACCELERATION[joint] * DT * DT
        assert out[key] == pytest.approx(expected, rel=1e-9)
        assert out[key] < HARDWARE_CLAMP


def test_dict_filter_missing_action_key_raises():
    filt = ActionDictOutputFilter(ACTION_KEYS, DEFAULT_MAX_VELOCITY, DEFAULT_MAX_ACCELERATION, DT)
    action = dict.fromkeys(ACTION_KEYS[:-1], 0.0)
    with pytest.raises(KeyError, match="missing from action"):
        filt.apply(action, obs_at(0.0))


def test_dict_filter_missing_seed_key_raises():
    filt = ActionDictOutputFilter(ACTION_KEYS, DEFAULT_MAX_VELOCITY, DEFAULT_MAX_ACCELERATION, DT)
    action = dict.fromkeys(ACTION_KEYS, 0.0)
    obs = obs_at(0.0)
    del obs[ACTION_KEYS[0]]
    with pytest.raises(KeyError, match="missing from observation"):
        filt.apply(action, obs)


def test_dict_filter_reset_reseeds_from_next_observation():
    filt = ActionDictOutputFilter(ACTION_KEYS, DEFAULT_MAX_VELOCITY, DEFAULT_MAX_ACCELERATION, DT)
    filt.apply(dict.fromkeys(ACTION_KEYS, 0.0), obs_at(0.0))
    filt.reset()
    # After reset the integrator must restart at the *new* measured pose, not
    # continue from the pre-reset state at 0.
    out = filt.apply(dict.fromkeys(ACTION_KEYS, 50.0), obs_at(50.0))
    for key in ACTION_KEYS:
        assert out[key] == pytest.approx(50.0, abs=1e-9)


def test_dict_filter_stats_accumulate():
    filt = ActionDictOutputFilter(ACTION_KEYS, DEFAULT_MAX_VELOCITY, DEFAULT_MAX_ACCELERATION, DT)
    filt.apply(dict.fromkeys(ACTION_KEYS, 100.0), obs_at(0.0))
    filt.apply(dict.fromkeys(ACTION_KEYS, 100.0), obs_at(0.0))
    stats = filt.stats
    assert stats["steps"] == 2
    assert stats["interventions"] == 2
    assert stats["max_abs_intervention"] > 0


def test_dict_filter_requires_action_keys():
    with pytest.raises(ValueError, match="at least one"):
        ActionDictOutputFilter([], DEFAULT_MAX_VELOCITY, DEFAULT_MAX_ACCELERATION, DT)


# ---------------------------------------------------------------------------
# Rollout config surface
# ---------------------------------------------------------------------------


def test_action_filter_config_disabled_by_default():
    cfg = ActionOutputFilterConfig()
    assert cfg.enabled is False
    assert cfg.resolved_max_velocity() == DEFAULT_MAX_VELOCITY
    assert cfg.resolved_max_acceleration() == DEFAULT_MAX_ACCELERATION


def test_action_filter_config_overrides_merge_over_defaults():
    cfg = ActionOutputFilterConfig(enabled=True, max_velocity={"elbow_flex": 40.0})
    resolved = cfg.resolved_max_velocity()
    assert resolved["elbow_flex"] == 40.0
    assert resolved["shoulder_pan"] == DEFAULT_MAX_VELOCITY["shoulder_pan"]
    # Defaults dictionary must not be mutated by the merge.
    assert DEFAULT_MAX_VELOCITY["elbow_flex"] == 95.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_velocity": {"elbow_flex": 0.0}},
        {"max_velocity": {"elbow_flex": -3.0}},
        {"max_acceleration": {"gripper": float("inf")}},
        {"max_acceleration": {"gripper": float("nan")}},
        {"max_velocity": {"elbow_flex": True}},
    ],
)
def test_action_filter_config_rejects_invalid_overrides(kwargs):
    with pytest.raises(ValueError):
        ActionOutputFilterConfig(enabled=True, **kwargs)


def test_action_filter_config_draccus_decode_round_trip():
    draccus = pytest.importorskip("draccus")
    cfg = draccus.decode(
        ActionOutputFilterConfig,
        {"enabled": True, "max_velocity": {"elbow_flex": 50.0}},
    )
    assert cfg.enabled is True
    assert cfg.resolved_max_velocity()["elbow_flex"] == 50.0
