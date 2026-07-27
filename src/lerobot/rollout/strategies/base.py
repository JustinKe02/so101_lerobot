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

"""Base rollout strategy: autonomous policy execution with no data recording."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any

from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class _TeardownErrors:
    """Run every teardown step while preserving the first failure."""

    def __init__(self, strategy_logger: logging.Logger) -> None:
        self._logger = strategy_logger
        self._first_error: tuple[BaseException, TracebackType | None] | None = None

    def run(self, description: str, operation: Callable[[], Any]) -> tuple[bool, Any]:
        try:
            return True, operation()
        except BaseException as error:
            self._logger.exception("%s failed during teardown", description)
            if self._first_error is None:
                self._first_error = (error, error.__traceback__)
            return False, None

    def raise_if_any(self) -> None:
        if self._first_error is None:
            return
        error, traceback = self._first_error
        raise error.with_traceback(traceback)


class BaseStrategy(RolloutStrategy):
    """Autonomous policy rollout with no data recording.

    All actions flow through the ``robot_action_processor`` pipeline
    before reaching the robot.
    """

    def setup(self, ctx: RolloutContext) -> None:
        """Initialise the inference engine."""
        self._init_engine(ctx)
        logger.info("Base strategy ready")

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop until shutdown or duration expires."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Base strategy control loop started")

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    f"Record loop is running slower ({1 / dt:.1f} Hz) than the target FPS ({cfg.fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
                )

    def teardown(self, ctx: RolloutContext) -> None:
        """Disconnect hardware and stop inference."""
        errors = _TeardownErrors(logger)
        errors.run(
            "hardware cleanup",
            lambda: self._teardown_hardware(
                ctx.hardware,
                return_to_initial_position=ctx.runtime.cfg.return_to_initial_position,
            ),
        )
        errors.raise_if_any()
        logger.info("Base strategy teardown complete")
