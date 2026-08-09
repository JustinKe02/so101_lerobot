#!/usr/bin/env python

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, prepare_rtc_training_inputs
from lerobot.policies.pi_gemma import PiGemmaRMSNorm


def test_pi05_config_defaults_preserve_original_training() -> None:
    config = PI05Config(chunk_size=8, n_action_steps=8, max_action_dim=4, max_state_dim=4, dtype="float32")

    assert config.rtc_training_max_delay == 0


@pytest.mark.parametrize("value", [-1, 8])
def test_pi05_config_rejects_invalid_training_delay(value: int) -> None:
    with pytest.raises(ValueError, match="rtc_training_max_delay"):
        PI05Config(
            chunk_size=8,
            n_action_steps=8,
            max_action_dim=4,
            max_state_dim=4,
            dtype="float32",
            rtc_training_max_delay=value,
        )


def test_prepare_rtc_training_inputs_uses_clean_prefix_and_postfix_only_weights() -> None:
    actions = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
    noise = actions + 10.0
    time = torch.tensor([0.25, 0.75])
    prefix_lengths = torch.tensor([0, 2])

    x_t, u_t, weights, token_times = prepare_rtc_training_inputs(actions, noise, time, prefix_lengths)

    assert torch.equal(x_t[1, :2], actions[1, :2])
    assert torch.equal(weights[1, :2], torch.zeros_like(weights[1, :2]))
    assert torch.all(weights[1, 2:] > 0)
    assert weights[1, :, 0].sum().item() == pytest.approx(5.0)
    assert torch.equal(token_times[1, :2], torch.zeros(2))
    assert torch.allclose(token_times[1, 2:], torch.full((3,), 0.75))
    assert torch.equal(u_t, noise - actions)


def test_adarms_accepts_independent_condition_per_action_token() -> None:
    layer = PiGemmaRMSNorm(dim=4, cond_dim=4)
    hidden = torch.randn(2, 5, 4)
    condition = torch.randn(2, 5, 4)

    output, gate = layer(hidden, cond=condition)

    assert output.shape == hidden.shape
    assert gate is not None
    assert gate.shape == hidden.shape


def test_trained_prefix_is_hard_inpainted_after_every_denoise_step() -> None:
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_inference_steps=3,
        chunk_size=4,
        max_action_dim=3,
        rtc_training_max_delay=2,
        rtc_config=SimpleNamespace(enabled=True),
    )
    model.rtc_processor = None
    model._compute_prefix_cache = MethodType(lambda self, *args: (None, None), model)
    denoise_timesteps = []

    def denoise_step(self, **kwargs):
        denoise_timesteps.append(kwargs["timestep"].clone())
        return torch.ones_like(kwargs["x_t"])

    model.denoise_step = MethodType(denoise_step, model)

    noise = torch.zeros(1, 4, 3)
    prefix = torch.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    tokens = torch.zeros(1, 1, dtype=torch.long)
    result = model.sample_actions(
        images=[],
        img_masks=[],
        tokens=tokens,
        masks=torch.ones_like(tokens),
        noise=noise,
        num_steps=3,
        rtc_mode="trained_prefix",
        prev_chunk_left_over=prefix,
        inference_delay=2,
    )

    assert torch.equal(result[0, :2], prefix)
    assert torch.all(result[0, 2:] < 0)
    assert len(denoise_timesteps) == 3
    expected_postfix_times = [1.0, 2 / 3, 1 / 3]
    for timestep, postfix_time in zip(denoise_timesteps, expected_postfix_times, strict=True):
        assert timestep.shape == (1, 4)
        assert torch.equal(timestep[:, :2], torch.zeros(1, 2))
        assert torch.allclose(timestep[:, 2:], torch.full((1, 2), postfix_time))


def test_trained_prefix_without_available_prefix_preserves_scalar_timestep() -> None:
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_inference_steps=2,
        chunk_size=4,
        max_action_dim=3,
        rtc_training_max_delay=2,
        rtc_config=SimpleNamespace(enabled=True),
    )
    model.rtc_processor = None
    model._compute_prefix_cache = MethodType(lambda self, *args: (None, None), model)
    denoise_timesteps = []

    def denoise_step(self, **kwargs):
        denoise_timesteps.append(kwargs["timestep"].clone())
        return torch.zeros_like(kwargs["x_t"])

    model.denoise_step = MethodType(denoise_step, model)

    noise = torch.zeros(2, 4, 3)
    tokens = torch.zeros(2, 1, dtype=torch.long)
    model.sample_actions(
        images=[],
        img_masks=[],
        tokens=tokens,
        masks=torch.ones_like(tokens),
        noise=noise,
        num_steps=2,
        rtc_mode="trained_prefix",
        prev_chunk_left_over=None,
        inference_delay=0,
    )

    assert len(denoise_timesteps) == 2
    assert denoise_timesteps[0].shape == (2,)
    assert torch.equal(denoise_timesteps[0], torch.ones(2))
    assert torch.equal(denoise_timesteps[1], torch.full((2,), 0.5))
