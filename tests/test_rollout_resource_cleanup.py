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

"""Regression tests for rollout context and strategy resource cleanup."""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")


def _context_config(*, teleop=None):
    return SimpleNamespace(
        policy=SimpleNamespace(
            pretrained_path="test-checkpoint",
            type="mock",
            use_peft=False,
        ),
        inference=SimpleNamespace(type="sync"),
        device="cpu",
        use_torch_compile=False,
        use_pi05_tensorrt_prefix=False,
        robot=SimpleNamespace(type="mock"),
        teleop=teleop,
        dataset=None,
    )


def _patch_policy_load(monkeypatch, context_module) -> None:
    policy = MagicMock()
    policy.to.return_value = policy
    policy.type = "mock"
    policy_class = MagicMock()
    policy_class.from_pretrained.return_value = policy
    monkeypatch.setattr(context_module, "get_policy_class", lambda _policy_type: policy_class)


def _processors():
    return (MagicMock(), MagicMock(), MagicMock())


def test_context_disconnects_robot_when_initial_observation_fails(monkeypatch):
    from lerobot.rollout import context as context_module

    _patch_policy_load(monkeypatch, context_module)
    failure = RuntimeError("initial observation failed")
    robot = MagicMock()
    robot.name = "mock_robot"
    robot.is_connected = True
    robot.get_observation.side_effect = failure
    monkeypatch.setattr(context_module, "make_robot_from_config", lambda _cfg: robot)
    teleop_processor, action_processor, observation_processor = _processors()

    with pytest.raises(RuntimeError, match="initial observation failed") as exc_info:
        context_module.build_rollout_context(
            _context_config(),
            Event(),
            teleop_action_processor=teleop_processor,
            robot_action_processor=action_processor,
            robot_observation_processor=observation_processor,
        )

    assert exc_info.value is failure
    robot.disconnect.assert_called_once_with()


def test_context_disconnects_teleop_then_robot_when_later_build_step_fails(monkeypatch):
    from lerobot.rollout import context as context_module

    _patch_policy_load(monkeypatch, context_module)
    disconnect_order = []
    robot = MagicMock()
    robot.name = "mock_robot"
    robot.is_connected = True
    robot.get_observation.return_value = {"joint.pos": 0.0}
    robot.observation_features = {"joint.pos": float}
    robot.action_features = {"joint.pos": float}
    robot.disconnect.side_effect = lambda: disconnect_order.append("robot")

    teleop = MagicMock()
    teleop.is_connected = True
    teleop.disconnect.side_effect = lambda: disconnect_order.append("teleop")

    failure = RuntimeError("feature aggregation failed")
    monkeypatch.setattr(context_module, "make_robot_from_config", lambda _cfg: robot)
    monkeypatch.setattr(context_module, "make_teleoperator_from_config", lambda _cfg: teleop)
    monkeypatch.setattr(context_module, "create_initial_features", lambda **_kwargs: {})
    monkeypatch.setattr(
        context_module,
        "aggregate_pipeline_dataset_features",
        MagicMock(side_effect=failure),
    )
    teleop_processor, action_processor, observation_processor = _processors()

    with pytest.raises(RuntimeError, match="feature aggregation failed") as exc_info:
        context_module.build_rollout_context(
            _context_config(teleop=SimpleNamespace(type="mock_teleop")),
            Event(),
            teleop_action_processor=teleop_processor,
            robot_action_processor=action_processor,
            robot_observation_processor=observation_processor,
        )

    assert exc_info.value is failure
    assert disconnect_order == ["teleop", "robot"]


def _make_strategy(kind: str):
    if kind == "sentry":
        from lerobot.rollout.strategies.sentry import SentryStrategy, SentryStrategyConfig

        strategy = SentryStrategy(SentryStrategyConfig())
        strategy._needs_push.set()
        return strategy
    if kind == "highlight":
        from lerobot.rollout.strategies.highlight import HighlightStrategy, HighlightStrategyConfig

        return HighlightStrategy(HighlightStrategyConfig())
    if kind == "dagger":
        from lerobot.rollout.strategies.dagger import DAggerStrategy, DAggerStrategyConfig

        strategy = DAggerStrategy(DAggerStrategyConfig())
        strategy._needs_push.set()
        return strategy
    if kind == "episodic":
        from lerobot.rollout.strategies.episodic import EpisodicStrategy, EpisodicStrategyConfig

        return EpisodicStrategy(EpisodicStrategyConfig())
    raise AssertionError(f"Unknown strategy: {kind}")


def _strategy_context(dataset):
    dataset_cfg = SimpleNamespace(push_to_hub=True, tags=["test"], private=True)
    cfg = SimpleNamespace(
        play_sounds=False,
        return_to_initial_position=False,
        dataset=dataset_cfg,
    )
    return SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg),
        data=SimpleNamespace(dataset=dataset),
        hardware=object(),
    )


@pytest.mark.parametrize("kind", ["sentry", "highlight", "dagger", "episodic"])
def test_finalize_failure_does_not_skip_hardware_cleanup(monkeypatch, kind):
    module = __import__(f"lerobot.rollout.strategies.{kind}", fromlist=["safe_push_to_hub"])
    monkeypatch.setattr(module, "log_say", MagicMock())
    safe_push = MagicMock(return_value=True)
    monkeypatch.setattr(module, "safe_push_to_hub", safe_push)

    failure = RuntimeError(f"{kind} finalize failed")
    dataset = MagicMock()
    dataset.finalize.side_effect = failure
    strategy = _make_strategy(kind)
    strategy._teardown_hardware = MagicMock()

    with pytest.raises(RuntimeError, match="finalize failed") as exc_info:
        strategy.teardown(_strategy_context(dataset))

    assert exc_info.value is failure
    strategy._teardown_hardware.assert_called_once()
    safe_push.assert_not_called()


@pytest.mark.parametrize("kind", ["sentry", "highlight", "dagger", "episodic"])
def test_final_upload_runs_after_hardware_cleanup(monkeypatch, kind):
    module = __import__(f"lerobot.rollout.strategies.{kind}", fromlist=["safe_push_to_hub"])
    monkeypatch.setattr(module, "log_say", MagicMock())
    order = []
    safe_push = MagicMock(side_effect=lambda *_args, **_kwargs: order.append("upload") or True)
    monkeypatch.setattr(module, "safe_push_to_hub", safe_push)

    dataset = MagicMock()
    dataset.finalize.side_effect = lambda: order.append("finalize")
    strategy = _make_strategy(kind)
    strategy._teardown_hardware = MagicMock(side_effect=lambda *_args, **_kwargs: order.append("hardware"))

    strategy.teardown(_strategy_context(dataset))

    assert order.index("hardware") < order.index("upload")


@pytest.mark.parametrize("kind", ["sentry", "highlight", "dagger", "episodic"])
def test_upload_failure_is_reported_after_hardware_cleanup(monkeypatch, kind):
    module = __import__(f"lerobot.rollout.strategies.{kind}", fromlist=["safe_push_to_hub"])
    monkeypatch.setattr(module, "log_say", MagicMock())
    order = []
    failure = RuntimeError(f"{kind} upload failed")

    def fail_upload(*_args, **_kwargs):
        order.append("upload")
        raise failure

    monkeypatch.setattr(module, "safe_push_to_hub", fail_upload)
    strategy = _make_strategy(kind)
    strategy._teardown_hardware = MagicMock(side_effect=lambda *_args, **_kwargs: order.append("hardware"))

    with pytest.raises(RuntimeError, match="upload failed") as exc_info:
        strategy.teardown(_strategy_context(MagicMock()))

    assert exc_info.value is failure
    assert order.index("hardware") < order.index("upload")


@pytest.mark.parametrize("kind", ["sentry", "highlight", "dagger"])
def test_executor_shutdown_failure_does_not_skip_dataset_finalize(monkeypatch, kind):
    module = __import__(f"lerobot.rollout.strategies.{kind}", fromlist=["safe_push_to_hub"])
    monkeypatch.setattr(module, "log_say", MagicMock())
    safe_push = MagicMock(return_value=True)
    monkeypatch.setattr(module, "safe_push_to_hub", safe_push)

    failure = RuntimeError(f"{kind} executor failed")
    executor = MagicMock()
    executor.shutdown.side_effect = failure
    dataset = MagicMock()
    strategy = _make_strategy(kind)
    strategy._push_executor = executor
    strategy._teardown_hardware = MagicMock()

    with pytest.raises(RuntimeError, match="executor failed") as exc_info:
        strategy.teardown(_strategy_context(dataset))

    assert exc_info.value is failure
    strategy._teardown_hardware.assert_called_once()
    dataset.finalize.assert_called_once_with()
    safe_push.assert_not_called()
