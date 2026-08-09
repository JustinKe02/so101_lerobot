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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.so_follower import (
    SO100Follower,
    SO100FollowerConfig,
)


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the robot."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


def _make_camera_mock(name: str, cleanup_order: list[str]) -> MagicMock:
    camera = MagicMock(name=name)
    camera.is_connected = False

    def _connect():
        camera.is_connected = True

    def _disconnect():
        cleanup_order.append(name)
        camera.is_connected = False

    camera.connect.side_effect = _connect
    camera.disconnect.side_effect = _disconnect
    return camera


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)

        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO100Follower, "configure", lambda self: None),
    ):
        cfg = SO100FollowerConfig(port="/dev/null")
        robot = SO100Follower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(follower):
    assert not follower.is_connected

    follower.connect()
    assert follower.is_connected

    follower.disconnect()
    assert not follower.is_connected


def test_connect_rolls_back_bus_and_partial_cameras_when_second_camera_fails(follower):
    cleanup_order: list[str] = []
    first = _make_camera_mock("first", cleanup_order)
    second = _make_camera_mock("second", cleanup_order)
    unattempted = _make_camera_mock("unattempted", cleanup_order)

    def _fail_second_connect():
        second.is_connected = True
        raise RuntimeError("second camera failed")

    second.connect.side_effect = _fail_second_connect
    follower.cameras = {"first": first, "second": second, "unattempted": unattempted}

    with pytest.raises(RuntimeError, match="second camera failed"):
        follower.connect()

    assert cleanup_order == ["second", "first"]
    first.disconnect.assert_called_once_with()
    second.disconnect.assert_called_once_with()
    unattempted.connect.assert_not_called()
    unattempted.disconnect.assert_not_called()
    follower.bus.disable_torque.assert_called_once_with(num_retry=3)
    follower.bus.disconnect.assert_called_once_with(False)
    assert not follower.bus.is_connected


def test_connect_rolls_back_all_resources_when_configure_fails(follower):
    cleanup_order: list[str] = []
    first = _make_camera_mock("first", cleanup_order)
    second = _make_camera_mock("second", cleanup_order)
    follower.cameras = {"first": first, "second": second}
    follower.configure = MagicMock(side_effect=RuntimeError("configure failed"))

    with pytest.raises(RuntimeError, match="configure failed"):
        follower.connect()

    assert cleanup_order == ["second", "first"]
    first.disconnect.assert_called_once_with()
    second.disconnect.assert_called_once_with()
    follower.bus.disable_torque.assert_called_once_with(num_retry=3)
    follower.bus.disconnect.assert_called_once_with(False)
    assert not follower.bus.is_connected


def test_disconnect_attempts_every_resource_and_aggregates_failures(follower):
    cleanup_order: list[str] = []
    first = _make_camera_mock("first", cleanup_order)
    second = _make_camera_mock("second", cleanup_order)
    follower.cameras = {"first": first, "second": second}
    follower.connect()

    def _fail_bus_disconnect(_disable=True):
        follower.bus.is_connected = False
        raise RuntimeError("bus disconnect failed")

    def _fail_first_disconnect():
        cleanup_order.append("first")
        first.is_connected = False
        raise RuntimeError("first camera disconnect failed")

    follower.bus.disconnect.side_effect = _fail_bus_disconnect
    first.disconnect.side_effect = _fail_first_disconnect

    with pytest.raises(BaseExceptionGroup) as exc_info:
        follower.disconnect()

    assert [str(error) for error in exc_info.value.exceptions] == [
        "bus disconnect failed",
        "first camera disconnect failed",
    ]
    assert cleanup_order == ["first", "second"]
    second.disconnect.assert_called_once_with()
    assert not second.is_connected


def test_get_observation(follower):
    follower.connect()
    obs = follower.get_observation()

    expected_keys = {f"{m}.pos" for m in follower.bus.motors}
    assert set(obs.keys()) == expected_keys

    for idx, motor in enumerate(follower.bus.motors, 1):
        assert obs[f"{motor}.pos"] == idx

    follower.bus.sync_read.assert_called_once_with("Present_Position", num_retry=3)


def test_send_action(follower):
    follower.connect()

    action = {f"{m}.pos": i * 10 for i, m in enumerate(follower.bus.motors, 1)}
    returned = follower.send_action(action)

    assert returned == action

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
    follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos, num_retry=3)


def test_send_action_retries_safety_position_read(follower):
    follower.connect()
    follower.config.max_relative_target = 5.0
    follower.bus.sync_read.return_value = dict.fromkeys(follower.bus.motors, 0.0)

    follower.send_action({f"{motor}.pos": 1.0 for motor in follower.bus.motors})

    follower.bus.sync_read.assert_called_once_with("Present_Position", num_retry=3)
