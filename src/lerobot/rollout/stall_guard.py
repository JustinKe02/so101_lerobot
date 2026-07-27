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

"""Stall/contact guard for the dispatched policy action stream.

The robot's relative-motion safety clamp (``max_relative_target``) rewrites a
commanded goal whenever it is further than one safe step from the present
position.  A single rewrite is a benign transient, but a *run* of consecutive
rewrites means the arm persistently fails to track its targets — the signature
of an obstruction or unintended contact, where continuing to dispatch actions
grinds the motors against whatever is blocking them.
"""

from __future__ import annotations

from typing import Any


class StallContactError(RuntimeError):
    """Raised when the robot persistently fails to track dispatched actions."""


class StallContactGuard:
    """Abort the rollout after ``max_consecutive_clamped`` clamped dispatches.

    ``observe`` compares each dispatched action against the value the robot
    reports actually applying.  Joints whose applied value deviates from the
    request by more than ``tolerance`` count as clamped; any dispatch with no
    clamped joint resets the run length.
    """

    def __init__(self, max_consecutive_clamped: int, tolerance: float = 1e-3) -> None:
        if max_consecutive_clamped < 1:
            raise ValueError("max_consecutive_clamped must be >= 1")
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        self.max_consecutive_clamped = max_consecutive_clamped
        self.tolerance = tolerance
        self._consecutive = 0
        self._last_deviations: dict[str, tuple[float, float]] = {}

    @property
    def consecutive_clamped(self) -> int:
        return self._consecutive

    def reset(self) -> None:
        self._consecutive = 0
        self._last_deviations = {}

    def observe(self, requested: dict[str, Any], applied: Any) -> None:
        """Record one dispatch; raise :class:`StallContactError` on a sustained stall.

        ``applied`` is whatever the robot's ``send_action`` returned.  When it
        is not a comparable mapping (some robots return ``None``), the guard
        cannot see clamping and resets rather than accumulating stale state.
        """
        if not isinstance(applied, dict):
            self.reset()
            return
        deviations: dict[str, tuple[float, float]] = {}
        for key, requested_value in requested.items():
            applied_value = applied.get(key)
            if not isinstance(requested_value, (int, float)) or not isinstance(applied_value, (int, float)):
                continue
            if abs(float(requested_value) - float(applied_value)) > self.tolerance:
                deviations[key] = (float(requested_value), float(applied_value))
        if not deviations:
            self.reset()
            return
        self._consecutive += 1
        self._last_deviations = deviations
        if self._consecutive >= self.max_consecutive_clamped:
            details = ", ".join(
                f"{key}: requested={req:.3f} applied={app:.3f}"
                for key, (req, app) in sorted(deviations.items())
            )
            raise StallContactError(
                f"safety clamp rewrote the commanded goal on {self._consecutive} consecutive "
                f"dispatches ({details}) — the arm is not tracking its targets "
                "(obstruction or unintended contact)"
            )
