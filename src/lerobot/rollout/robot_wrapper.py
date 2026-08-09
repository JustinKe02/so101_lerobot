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

"""Thread-safe robot wrapper for concurrent observation/action access."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

from lerobot.robots import Robot


class ThreadSafeRobot:
    """Lock-protected wrapper around a :class:`Robot` for use with background threads.

    When RTC inference runs in a background thread while the main loop
    executes actions, both threads may access the robot concurrently.
    This wrapper serialises ``get_observation`` and ``send_action`` calls.

    Read-only properties are proxied without the lock since they don't
    mutate hardware state.
    """

    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self._lock = Lock()
        self._last_observation_timing: dict[str, Any] | None = None

    # -- Lock-protected I/O --------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            started_at = time.perf_counter()
            observation = self._robot.get_observation()
            finished_at = time.perf_counter()
            timing = getattr(self._robot, "last_observation_timing", None)
            if callable(timing):
                timing = timing()
            if not isinstance(timing, dict):
                timing = {
                    "clock": "monotonic",
                    "state_timestamp": (started_at + finished_at) / 2.0,
                    "camera_timestamps": {},
                    "observation_timestamp": finished_at,
                }
            self._last_observation_timing = dict(timing)
            return observation

    def send_action(self, action: dict[str, Any] | Any) -> Any:
        with self._lock:
            return self._robot.send_action(action)

    @property
    def last_observation_timing(self) -> dict[str, Any] | None:
        """Return a copy of the most recent monotonic observation metadata."""

        with self._lock:
            return None if self._last_observation_timing is None else dict(self._last_observation_timing)

    # -- Read-only proxies (no lock needed) -----------------------------------

    @property
    def observation_features(self) -> dict:
        return self._robot.observation_features

    @property
    def action_features(self) -> dict:
        return self._robot.action_features

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def cameras(self):
        return getattr(self._robot, "cameras", {})

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    @property
    def inner(self) -> Robot:
        """Access the underlying robot (e.g. for connect/disconnect)."""
        return self._robot
