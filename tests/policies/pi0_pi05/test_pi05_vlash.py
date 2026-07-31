#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as functional
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaMLP,
    GemmaRotaryEmbedding,
)

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    FusedGemmaAttention,
    FusedGemmaMLP,
    PI05Pytorch,
    _disallowed_checkpoint_keys,
)
from lerobot.policies.pi05.processor_pi05 import (
    Pi05TemporalOffsetProcessorStep,
    reconcile_pi05_processors,
)
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    PolicyProcessorPipeline,
    TransitionKey,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_temporal_offset_processor_aligns_state_action_and_padding() -> None:
    action = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            [[100.0], [101.0], [102.0], [103.0], [104.0], [105.0]],
        ]
    )
    state = torch.tensor(
        [
            [[10.0], [11.0], [12.0]],
            [[110.0], [111.0], [112.0]],
        ]
    )
    image = torch.randn(2, 3, 8, 8)
    action_pad = torch.tensor(
        [[False, False, False, False, True, True], [False, False, False, True, True, True]]
    )
    state_pad = torch.tensor([[False, False, True], [False, False, True]])
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state, "observation.images.top": image},
        TransitionKey.ACTION: action,
        TransitionKey.COMPLEMENTARY_DATA: {
            "action_is_pad": action_pad,
            f"{OBS_STATE}_is_pad": state_pad,
        },
    }
    step = Pi05TemporalOffsetProcessorStep(max_offset_steps=2, chunk_size=4)

    with patch(
        "lerobot.policies.pi05.processor_pi05.torch.randint",
        return_value=torch.tensor([0, 2]),
    ):
        result = step(transition)

    torch.testing.assert_close(result[TransitionKey.ACTION], torch.stack((action[0, :4], action[1, 2:6])))
    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[10.0], [112.0]]))
    assert result[TransitionKey.OBSERVATION]["observation.images.top"] is image
    torch.testing.assert_close(
        result[TransitionKey.COMPLEMENTARY_DATA]["action_is_pad"],
        torch.stack((action_pad[0, :4], action_pad[1, 2:6])),
    )
    torch.testing.assert_close(
        result[TransitionKey.COMPLEMENTARY_DATA][f"{OBS_STATE}_is_pad"],
        torch.tensor([False, True]),
    )
    torch.testing.assert_close(result[TransitionKey.COMPLEMENTARY_DATA]["vlash_offset"], torch.tensor([0, 2]))


def test_temporal_offset_processor_is_noop_during_inference() -> None:
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.ones(1, 3)},
        TransitionKey.ACTION: None,
    }

    result = Pi05TemporalOffsetProcessorStep(max_offset_steps=2, chunk_size=4)(transition)

    assert result is transition


def test_delta_timestamps_keep_images_at_configured_observation_offsets() -> None:
    config = SimpleNamespace(
        reward_delta_indices=None,
        action_delta_indices=list(range(6)),
        observation_delta_indices=[0],
        temporal_offset_max_steps=2,
    )
    metadata = SimpleNamespace(
        fps=20,
        features={ACTION: {}, OBS_STATE: {}, "observation.images.top": {}},
    )

    delta_timestamps = resolve_delta_timestamps(config, metadata)

    assert delta_timestamps[ACTION] == pytest.approx([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    assert delta_timestamps[OBS_STATE] == pytest.approx([0.0, 0.05, 0.10])
    assert delta_timestamps["observation.images.top"] == [0.0]


@pytest.mark.parametrize("value", [-1, 1.5, True, "2"])
def test_pi05_rejects_invalid_temporal_offset(value) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        PI05Config(temporal_offset_max_steps=value)


def test_pi05_action_window_includes_temporal_lookahead() -> None:
    config = PI05Config(chunk_size=4, n_action_steps=4, temporal_offset_max_steps=2)

    assert config.action_delta_indices == [0, 1, 2, 3, 4, 5]


def test_checkpoint_processor_reconcile_inserts_and_updates_temporal_step() -> None:
    preprocessor = PolicyProcessorPipeline(steps=[AddBatchDimensionProcessorStep()])
    postprocessor = PolicyProcessorPipeline(steps=[])
    config = PI05Config(chunk_size=4, n_action_steps=4, temporal_offset_max_steps=2)

    reconcile_pi05_processors(config, preprocessor, postprocessor)

    assert len(preprocessor.steps) == 2
    assert isinstance(preprocessor.steps[1], Pi05TemporalOffsetProcessorStep)
    assert preprocessor.steps[1].chunk_size == 4
    assert preprocessor.steps[1].max_offset_steps == 2

    config.temporal_offset_max_steps = 3
    reconcile_pi05_processors(config, preprocessor, postprocessor)
    temporal_steps = [
        step for step in preprocessor.steps if isinstance(step, Pi05TemporalOffsetProcessorStep)
    ]
    assert len(temporal_steps) == 1
    assert temporal_steps[0].max_offset_steps == 3


def test_state_conditioner_starts_with_zero_contribution() -> None:
    config = PI05Config(state_cond=True, max_state_dim=4)
    tiny_gemma = SimpleNamespace(width=8)
    with (
        patch("lerobot.policies.pi05.modeling_pi05.get_gemma_config", return_value=tiny_gemma),
        patch(
            "lerobot.policies.pi05.modeling_pi05.PaliGemmaWithExpertModel",
            return_value=torch.nn.Identity(),
        ),
    ):
        model = PI05Pytorch(config)

    state = torch.randn(3, config.max_state_dim)
    contribution = model.state_mlp_out(functional.silu(model.state_mlp_in(model.state_proj(state))))

    torch.testing.assert_close(contribution, torch.zeros_like(contribution), rtol=0, atol=0)
    torch.testing.assert_close(
        model.state_mlp_out.weight, torch.zeros_like(model.state_mlp_out.weight), rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.state_mlp_out.bias, torch.zeros_like(model.state_mlp_out.bias), rtol=0, atol=0
    )


def test_legacy_checkpoint_only_allows_missing_state_conditioner() -> None:
    state_keys = [
        "model.state_proj.weight",
        "model.state_proj.bias",
        "model.state_mlp_in.weight",
        "model.state_mlp_in.bias",
        "model.state_mlp_out.weight",
        "model.state_mlp_out.bias",
    ]

    assert _disallowed_checkpoint_keys(state_keys, [], state_cond=True) == ([], [])
    assert _disallowed_checkpoint_keys(state_keys, [], state_cond=False) == (state_keys, [])
    assert _disallowed_checkpoint_keys(
        [*state_keys, "model.action_in_proj.weight"],
        ["unexpected.weight"],
        state_cond=True,
    ) == (["model.action_in_proj.weight"], ["unexpected.weight"])


def _tiny_gemma_config() -> GemmaConfig:
    config = GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        attention_bias=False,
    )
    config._attn_implementation = "eager"
    return config


def test_fused_gemma_attention_matches_unfused() -> None:
    torch.manual_seed(0)
    config = _tiny_gemma_config()
    hidden_states = torch.randn(2, 5, config.hidden_size)
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
    position_embeddings = GemmaRotaryEmbedding(config)(hidden_states, position_ids)
    attention = GemmaAttention(config, layer_idx=0).eval()
    fused = FusedGemmaAttention(attention).eval()

    expected, _ = attention(hidden_states, position_embeddings=position_embeddings)
    actual, _ = fused(hidden_states, position_embeddings=position_embeddings)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_fused_gemma_mlp_matches_unfused() -> None:
    torch.manual_seed(0)
    config = _tiny_gemma_config()
    hidden_states = torch.randn(2, 5, config.hidden_size)
    mlp = GemmaMLP(config).eval()
    fused = FusedGemmaMLP(mlp).eval()

    torch.testing.assert_close(fused(hidden_states), mlp(hidden_states), rtol=0, atol=0)
