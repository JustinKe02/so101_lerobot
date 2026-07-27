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

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_rollout_top_level_imports():
    import lerobot.rollout

    for name in lerobot.rollout.__all__:
        assert hasattr(lerobot.rollout, name), f"Missing export: {name}"


def test_inference_submodule_imports():
    import lerobot.rollout.inference

    for name in lerobot.rollout.inference.__all__:
        assert hasattr(lerobot.rollout.inference, name), f"Missing export: {name}"


def test_strategies_submodule_imports():
    import lerobot.rollout.strategies

    for name in lerobot.rollout.strategies.__all__:
        assert hasattr(lerobot.rollout.strategies, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        BaseStrategyConfig,
        DAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    assert BaseStrategyConfig().type == "base"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"
    assert EpisodicStrategyConfig().type == "episodic"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard' or 'pedal'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.input_device == "keyboard"


def test_inference_config_types():
    from lerobot.rollout import RTCInferenceConfig, SyncInferenceConfig

    sync = SyncInferenceConfig()
    assert sync.type == "sync"
    assert sync.max_actions_per_chunk is None
    assert sync.replan_on_clamp is False
    assert sync.clamp_replan_threshold == 5.0

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        BaseStrategy,
        BaseStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        EpisodicStrategy,
        EpisodicStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)
    assert isinstance(create_strategy(EpisodicStrategyConfig()), EpisodicStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


def test_sync_inference_action_result_hook_is_noop():
    from lerobot.rollout import SyncInferenceEngine

    engine = SyncInferenceEngine(
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        dataset_features={},
        ordered_action_keys=["joint.pos"],
        task="test",
        device="cpu",
        robot_type="mock",
    )

    assert engine.notify_action_result({"joint.pos": 1.0}, None, {"joint.pos": 0.0}) is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_actions_per_chunk": 0}, "max_actions_per_chunk"),
        ({"max_actions_per_chunk": True}, "max_actions_per_chunk"),
        ({"clamp_replan_threshold": -1.0}, "clamp_replan_threshold"),
        ({"clamp_replan_threshold": float("inf")}, "clamp_replan_threshold"),
    ],
)
def test_sync_inference_config_rejects_invalid_guard_values(kwargs, message):
    from lerobot.rollout import SyncInferenceConfig

    with pytest.raises(ValueError, match=message):
        SyncInferenceConfig(**kwargs)


def _make_sync_engine_for_guard_tests(**kwargs):
    from lerobot.rollout import SyncInferenceEngine

    policy = MagicMock()
    policy.config.use_amp = False
    policy.select_action.return_value = torch.tensor([[1.0, 2.0]])
    preprocessor = MagicMock(side_effect=lambda observation: observation)
    postprocessor = MagicMock(side_effect=lambda action: action)
    engine = SyncInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset_features={"action": {"names": ["joint.pos", "gripper.pos"]}},
        ordered_action_keys=["joint.pos", "gripper.pos"],
        task="test",
        device="cpu",
        robot_type="mock",
        **kwargs,
    )
    return engine, policy


def test_sync_guard_replans_after_configured_action_prefix():
    engine, policy = _make_sync_engine_for_guard_tests(max_actions_per_chunk=2)

    with patch(
        "lerobot.rollout.inference.sync.prepare_observation_for_inference",
        side_effect=lambda observation, *_args: observation,
    ):
        engine.get_action({"observation.state": torch.tensor([0.0])})
        engine.get_action({"observation.state": torch.tensor([0.0])})
        policy.reset.assert_not_called()
        engine.get_action({"observation.state": torch.tensor([0.0])})

    policy.reset.assert_called_once_with()
    assert engine._horizon_replans == 1


def test_sync_guard_replans_after_material_robot_clamp():
    engine, policy = _make_sync_engine_for_guard_tests(
        replan_on_clamp=True,
        clamp_replan_threshold=2.0,
    )

    engine.notify_action_result(
        {"joint.pos": 10.0},
        {"joint.pos": 5.0},
        {"joint.pos": 0.0},
    )
    policy.reset.assert_not_called()

    with patch(
        "lerobot.rollout.inference.sync.prepare_observation_for_inference",
        side_effect=lambda observation, *_args: observation,
    ):
        engine.get_action({"observation.state": torch.tensor([0.0])})

    policy.reset.assert_called_once_with()
    assert engine._clamp_replans == 1


def test_sync_guard_ignores_clamp_below_threshold():
    engine, policy = _make_sync_engine_for_guard_tests(
        replan_on_clamp=True,
        clamp_replan_threshold=2.0,
    )

    engine.notify_action_result(
        {"joint.pos": 6.5},
        {"joint.pos": 5.0},
        {"joint.pos": 0.0},
    )

    assert engine._pending_replan_reason is None
    assert engine._clamp_replans == 0
    policy.reset.assert_not_called()


def _make_action_dispatch_context(*, action, processed, sent):
    engine = MagicMock()
    engine.get_action.return_value = torch.tensor(action)
    engine.failed = False

    robot_action_processor = MagicMock(return_value=processed)
    robot_wrapper = MagicMock()
    robot_wrapper.send_action.return_value = sent

    ctx = SimpleNamespace(
        policy=SimpleNamespace(inference=engine, action_filter=None),
        data=SimpleNamespace(dataset_features={}, ordered_action_keys=["joint.pos"]),
        processors=SimpleNamespace(robot_action_processor=robot_action_processor),
        hardware=SimpleNamespace(robot_wrapper=robot_wrapper),
    )
    return ctx, engine, robot_action_processor, robot_wrapper


def test_send_next_action_notifies_action_returned_by_robot():
    from lerobot.rollout.strategies import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    observation = {"joint.pos": 2.0}
    requested = {"joint.pos": 6.0}
    sent = {"joint.pos": 5.0}
    ctx, engine, robot_action_processor, robot_wrapper = _make_action_dispatch_context(
        action=[7.0], processed=requested, sent=sent
    )

    result = send_next_action({}, observation, ctx, ActionInterpolator())

    assert result == {"joint.pos": 7.0}
    robot_action_processor.assert_called_once_with(({"joint.pos": 7.0}, observation))
    robot_wrapper.send_action.assert_called_once_with(requested)
    engine.notify_action_result.assert_called_once_with(requested, sent, observation)


def test_send_next_action_preserves_none_robot_result_for_hook():
    from lerobot.rollout.strategies import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    observation = {"joint.pos": 2.0}
    requested = {"joint.pos": 6.0}
    ctx, engine, _, _ = _make_action_dispatch_context(action=[7.0], processed=requested, sent=None)

    result = send_next_action({}, observation, ctx, ActionInterpolator())

    assert result == {"joint.pos": 7.0}
    engine.notify_action_result.assert_called_once_with(requested, None, observation)


def test_send_next_action_does_not_dispatch_buffered_action_after_inference_failure():
    from lerobot.rollout.strategies import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    requested = {"joint.pos": 6.0}
    ctx, engine, robot_action_processor, robot_wrapper = _make_action_dispatch_context(
        action=[7.0], processed=requested, sent=requested
    )
    engine.failed = True

    result = send_next_action({}, {"joint.pos": 2.0}, ctx, ActionInterpolator())

    assert result is None
    robot_action_processor.assert_not_called()
    robot_wrapper.send_action.assert_not_called()
    engine.notify_action_result.assert_not_called()


def test_send_next_action_honors_atomic_dispatch_guard() -> None:
    from lerobot.rollout.strategies import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    requested = {"joint.pos": 6.0}
    ctx, engine, robot_action_processor, robot_wrapper = _make_action_dispatch_context(
        action=[7.0], processed=requested, sent=requested
    )
    engine.action_dispatch_guard.return_value.__enter__.return_value = False

    result = send_next_action({}, {"joint.pos": 2.0}, ctx, ActionInterpolator())

    assert result is None
    robot_action_processor.assert_called_once()
    engine.action_dispatch_guard.assert_called_once_with()
    robot_wrapper.send_action.assert_not_called()
    engine.notify_action_result.assert_not_called()


def test_inference_failure_skips_return_to_initial_position() -> None:
    from lerobot.rollout import BaseStrategy, BaseStrategyConfig

    strategy = BaseStrategy(BaseStrategyConfig())
    engine = MagicMock()
    engine.failed = True
    strategy._engine = engine

    robot = MagicMock()
    robot.is_connected = True
    wrapper = MagicMock()
    wrapper.inner = robot
    hardware = SimpleNamespace(
        robot_wrapper=wrapper,
        teleop=None,
        initial_position={"joint.pos": 10.0},
    )

    strategy._teardown_hardware(hardware, return_to_initial_position=True)

    engine.stop.assert_called_once_with()
    wrapper.get_observation.assert_not_called()
    wrapper.send_action.assert_not_called()
    robot.disconnect.assert_called_once_with()


def test_rollout_raises_after_inference_engine_failure():
    from threading import Event

    from lerobot.scripts import lerobot_rollout as rollout_module

    fatal_error = ValueError("fatal inference")
    engine = MagicMock()
    engine.failed = True
    engine.fatal_error = fatal_error
    ctx = SimpleNamespace(policy=SimpleNamespace(inference=engine))
    strategy = MagicMock()
    cfg = SimpleNamespace(
        seed=None,
        display_data=False,
        strategy=SimpleNamespace(type="base"),
        robot=SimpleNamespace(type="mock"),
        fps=30.0,
        duration=1.0,
    )

    with (
        patch.object(rollout_module, "init_logging"),
        patch.object(
            rollout_module,
            "ProcessSignalHandler",
            return_value=SimpleNamespace(shutdown_event=Event()),
        ),
        patch.object(rollout_module, "build_rollout_context", return_value=ctx),
        patch.object(rollout_module, "create_strategy", return_value=strategy),
        pytest.raises(RuntimeError, match="Inference engine failed during rollout") as exc_info,
    ):
        rollout_module.rollout.__wrapped__(cfg)

    assert exc_info.value.__cause__ is fatal_error
    strategy.setup.assert_called_once_with(ctx)
    strategy.run.assert_called_once_with(ctx)
    strategy.teardown.assert_called_once_with(ctx)


def test_rollout_raises_when_inference_fails_during_teardown():
    from threading import Event

    from lerobot.scripts import lerobot_rollout as rollout_module

    fatal_error = TimeoutError("RTC join timed out")
    engine = MagicMock()
    engine.failed = False
    engine.fatal_error = None
    ctx = SimpleNamespace(policy=SimpleNamespace(inference=engine))
    strategy = MagicMock()

    def fail_during_teardown(_ctx) -> None:
        engine.failed = True
        engine.fatal_error = fatal_error

    strategy.teardown.side_effect = fail_during_teardown
    cfg = SimpleNamespace(
        seed=None,
        display_data=False,
        strategy=SimpleNamespace(type="base"),
        robot=SimpleNamespace(type="mock"),
        fps=30.0,
        duration=1.0,
    )

    with (
        patch.object(rollout_module, "init_logging"),
        patch.object(
            rollout_module,
            "ProcessSignalHandler",
            return_value=SimpleNamespace(shutdown_event=Event()),
        ),
        patch.object(rollout_module, "build_rollout_context", return_value=ctx),
        patch.object(rollout_module, "create_strategy", return_value=strategy),
        pytest.raises(RuntimeError, match="Inference engine failed during rollout") as exc_info,
    ):
        rollout_module.rollout.__wrapped__(cfg)

    assert exc_info.value.__cause__ is fatal_error
    strategy.run.assert_called_once_with(ctx)
    strategy.teardown.assert_called_once_with(ctx)


def test_rollout_shuts_down_visualization_when_context_build_fails():
    from threading import Event

    from lerobot.scripts import lerobot_rollout as rollout_module

    failure = RuntimeError("context build failed")
    strategy = MagicMock()
    cfg = SimpleNamespace(
        seed=None,
        display_data=True,
        display_mode="rerun",
        display_ip="127.0.0.1",
        display_port=9876,
        strategy=SimpleNamespace(type="base"),
        robot=SimpleNamespace(type="mock"),
        fps=30.0,
        duration=1.0,
    )

    with (
        patch.object(rollout_module, "init_logging"),
        patch.object(rollout_module, "init_visualization") as init_visualization,
        patch.object(rollout_module, "shutdown_visualization") as shutdown_visualization,
        patch.object(
            rollout_module,
            "ProcessSignalHandler",
            return_value=SimpleNamespace(shutdown_event=Event()),
        ),
        patch.object(rollout_module, "create_strategy", return_value=strategy),
        patch.object(rollout_module, "build_rollout_context", side_effect=failure),
        pytest.raises(RuntimeError, match="context build failed") as exc_info,
    ):
        rollout_module.rollout.__wrapped__(cfg)

    assert exc_info.value is failure
    init_visualization.assert_called_once()
    shutdown_visualization.assert_called_once_with("rerun")
    strategy.setup.assert_not_called()
    strategy.teardown.assert_not_called()


@pytest.mark.parametrize("seed", [None, 42])
def test_rollout_applies_optional_seed_before_context_build(seed):
    from unittest.mock import patch

    from lerobot.scripts.lerobot_rollout import rollout

    cfg = SimpleNamespace(
        seed=seed,
        display_data=False,
        strategy=SimpleNamespace(type="base"),
        robot=SimpleNamespace(type="mock"),
        fps=30.0,
        duration=0.0,
    )
    context = SimpleNamespace(
        policy=SimpleNamespace(inference=SimpleNamespace(failed=False)),
    )
    strategy = MagicMock()

    with (
        patch("lerobot.scripts.lerobot_rollout.set_seed") as set_seed,
        patch("lerobot.scripts.lerobot_rollout.ProcessSignalHandler"),
        patch("lerobot.scripts.lerobot_rollout.build_rollout_context", return_value=context),
        patch("lerobot.scripts.lerobot_rollout.create_strategy", return_value=strategy),
    ):
        rollout.__wrapped__(cfg)

    if seed is None:
        set_seed.assert_not_called()
    else:
        set_seed.assert_called_once_with(seed)
    strategy.setup.assert_called_once_with(context)
    strategy.run.assert_called_once_with(context)
    strategy.teardown.assert_called_once_with(context)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> PAUSED
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.PAUSED)

    # PAUSED -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}
