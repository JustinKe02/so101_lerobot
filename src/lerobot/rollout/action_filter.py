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

"""Second-order (velocity + acceleration) limiter for the rollout action stream.

Deployment spec (PI05_SO101_TRAINING_STRATEGY.md §6): the limiter sits after
the inference engine output (RTC queue pop or sync call) and before
``robot.send_action``.  Per joint, with ``dt`` the command period::

    v_target = clamp((q_target - q_command) / dt, -v_max, v_max)
    v_command = clamp(v_target, v_previous - a_max * dt, v_previous + a_max * dt)
    q_command = q_command + v_command * dt

``robot.max_relative_target`` stays as the final hardware safety limit; the
limiter only shapes the requested trajectory so that limit is not exercised
during nominal motion.  Caps are expressed in dataset action units per second
(the P95-of-demonstrations table from the strategy doc).
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)

# P95-derived initial caps from PI05_SO101_TRAINING_STRATEGY.md §6 (units/s and
# units/s^2 in the recorded action space).
DEFAULT_MAX_VELOCITY: dict[str, float] = {
    "shoulder_pan": 66.0,
    "shoulder_lift": 95.0,
    "elbow_flex": 95.0,
    "wrist_flex": 53.0,
    "wrist_roll": 40.0,
    "gripper": 102.0,
}
DEFAULT_MAX_ACCELERATION: dict[str, float] = {
    "shoulder_pan": 317.0,
    "shoulder_lift": 317.0,
    "elbow_flex": 396.0,
    "wrist_flex": 475.0,
    "wrist_roll": 396.0,
    "gripper": 924.0,
}


def resolve_joint_limit(action_key: str, limits: dict[str, float]) -> float:
    """Resolve a per-joint limit for an action key like ``"shoulder_pan.pos"``.

    Exact key match wins; otherwise the key is stripped at the first ``.``
    (motor-bus convention ``<joint>.<field>``).  Raises ``KeyError`` when no
    limit is configured — the filter fails closed rather than silently passing
    an unlimited joint through.
    """
    if action_key in limits:
        return limits[action_key]
    joint = action_key.split(".", 1)[0]
    if joint in limits:
        return limits[joint]
    raise KeyError(f"No output-filter limit configured for action key '{action_key}'")


def build_limit_tensor(action_keys: list[str], limits: dict[str, float]) -> torch.Tensor:
    """Vectorize per-joint limits into the action-key order, validating values."""
    values = []
    for key in action_keys:
        value = resolve_joint_limit(key, limits)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Output-filter limit for '{key}' must be finite and positive, got {value}")
        values.append(float(value))
    return torch.tensor(values, dtype=torch.float64)


class SecondOrderActionFilter:
    """Stateful per-joint velocity/acceleration limiter over action vectors."""

    def __init__(self, max_velocity: torch.Tensor, max_acceleration: torch.Tensor, dt: float) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"Filter dt must be finite and positive, got {dt}")
        if max_velocity.shape != max_acceleration.shape or max_velocity.ndim != 1:
            raise ValueError("max_velocity and max_acceleration must be 1-D tensors of equal length")
        self._v_max = max_velocity.to(torch.float64)
        self._a_max = max_acceleration.to(torch.float64)
        self._dt = float(dt)
        self._q: torch.Tensor | None = None
        self._v = torch.zeros_like(self._v_max)
        self.intervention_count = 0
        self.max_abs_intervention = 0.0
        self.step_count = 0

    @property
    def action_dim(self) -> int:
        return int(self._v_max.shape[0])

    @property
    def seeded(self) -> bool:
        return self._q is not None

    def reset(self) -> None:
        """Clear integrator state; the next step must re-seed from a position."""
        self._q = None
        self._v = torch.zeros_like(self._v_max)

    def seed(self, position: torch.Tensor) -> None:
        position = position.detach().to(torch.float64).reshape(-1)
        if position.shape != self._v_max.shape:
            raise ValueError(f"Seed position dim {tuple(position.shape)} != filter dim {self.action_dim}")
        if not torch.isfinite(position).all():
            raise ValueError("Seed position contains non-finite values")
        self._q = position.clone()
        self._v = torch.zeros_like(self._v_max)

    def step(self, target: torch.Tensor) -> torch.Tensor:
        """Advance one command period toward ``target``; returns the limited command."""
        if self._q is None:
            raise RuntimeError("SecondOrderActionFilter.step called before seed()")
        target64 = target.detach().to(torch.float64).reshape(-1)
        if target64.shape != self._v_max.shape:
            raise ValueError(f"Target dim {tuple(target64.shape)} != filter dim {self.action_dim}")
        v_target = ((target64 - self._q) / self._dt).clamp(-self._v_max, self._v_max)
        v_command = v_target.clamp(self._v - self._a_max * self._dt, self._v + self._a_max * self._dt)
        self._q = self._q + v_command * self._dt
        self._v = v_command
        delta = float((target64 - self._q).abs().max())
        self.step_count += 1
        if delta > 1e-9:
            self.intervention_count += 1
            self.max_abs_intervention = max(self.max_abs_intervention, delta)
        return self._q.to(target.dtype if target.is_floating_point() else torch.float32)

    def filter_sequence(self, targets: torch.Tensor, initial_position: torch.Tensor) -> torch.Tensor:
        """Filter a ``(T, dim)`` target sequence seeded at ``initial_position``.

        Resets the integrator first (velocity starts at zero) — offline replay
        approximates each deployment window independently.
        """
        if targets.ndim != 2 or targets.shape[1] != self.action_dim:
            raise ValueError(f"targets must be (T, {self.action_dim}), got {tuple(targets.shape)}")
        self.reset()
        self.seed(initial_position)
        return torch.stack([self.step(row) for row in targets])


class ActionDictOutputFilter:
    """Dict-interface wrapper used on the rollout dispatch path.

    Seeds lazily from the first robot observation so the command integrator
    starts at the measured pose (no startup jump), then limits every
    policy-issued action dict in place of the raw targets.
    """

    def __init__(
        self,
        action_keys: list[str],
        max_velocity: dict[str, float],
        max_acceleration: dict[str, float],
        dt: float,
    ) -> None:
        if not action_keys:
            raise ValueError("Action output filter requires at least one action key")
        self._keys = list(action_keys)
        self._filter = SecondOrderActionFilter(
            build_limit_tensor(self._keys, max_velocity),
            build_limit_tensor(self._keys, max_acceleration),
            dt,
        )

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    @property
    def stats(self) -> dict[str, float | int]:
        return {
            "steps": self._filter.step_count,
            "interventions": self._filter.intervention_count,
            "max_abs_intervention": round(self._filter.max_abs_intervention, 6),
        }

    def reset(self) -> None:
        self._filter.reset()

    def apply(self, action: dict, observation: dict) -> dict:
        """Return a copy of ``action`` with filtered targets for all filter keys.

        Every configured key must be present in ``action``; on the first call
        (or after ``reset``) every key must also be present in ``observation``
        to seed the integrator from the measured robot pose.
        """
        missing = [key for key in self._keys if key not in action]
        if missing:
            raise KeyError(f"Action output filter keys missing from action dict: {missing}")
        if not self._filter.seeded:
            missing_obs = [key for key in self._keys if key not in observation]
            if missing_obs:
                raise KeyError(f"Action output filter seed keys missing from observation: {missing_obs}")
            self._filter.seed(
                torch.tensor([float(observation[key]) for key in self._keys], dtype=torch.float64)
            )
            logger.info("Action output filter seeded from robot observation (%d joints)", len(self._keys))
        target = torch.tensor([float(action[key]) for key in self._keys], dtype=torch.float64)
        command = self._filter.step(target)
        filtered = dict(action)
        for index, key in enumerate(self._keys):
            filtered[key] = float(command[index])
        return filtered
