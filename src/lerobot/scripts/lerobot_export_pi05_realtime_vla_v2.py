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

"""Export a LeRobot PI0.5 checkpoint for dexmal/realtime-vla-v2.

The upstream Triton kernels consume input-major BF16 matrices rather than a
Hugging Face state dict. This exporter reads tensors directly from the
safetensors mmap, one source key at a time, and never instantiates PI05Policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

_MODEL_PREFIX = "model."
_VISION = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
_ENCODER = "paligemma_with_expert.paligemma.model.language_model"
_PROJECTOR = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
_DECODER = "paligemma_with_expert.gemma_expert.model"

_VISION_DEPTH = 27
_ENCODER_DEPTH = 18
_DECODER_DEPTH = 18
_VISION_WIDTH = 1152
_ENCODER_WIDTH = 2048
_DECODER_WIDTH = 1024
_HEAD_DIM = 256
_NUM_HEADS = 8
_ACTION_WIDTH = 32
_VOCAB_SIZE = 257152


@dataclass(frozen=True)
class CheckpointPaths:
    weights: Path
    config: Path


class SafeTensorSource(AbstractContextManager["SafeTensorSource"]):
    """Shape-checked, mmap-backed access to one safetensors checkpoint."""

    def __init__(self, path: Path):
        self.path = path
        self._handle: Any = None
        self._prefix = ""
        self.shapes: dict[str, tuple[int, ...]] = {}

    def __enter__(self) -> SafeTensorSource:
        self._handle = safe_open(str(self.path), framework="pt", device="cpu")
        raw_shapes = {
            key: tuple(self._handle.get_slice(key).get_shape())
            for key in self._handle.keys()  # noqa: SIM118 - safe_open is not iterable.
        }
        if any(key.startswith(_MODEL_PREFIX) for key in raw_shapes):
            self._prefix = _MODEL_PREFIX
        self.shapes = {
            key.removeprefix(self._prefix): shape
            for key, shape in raw_shapes.items()
            if key.startswith(self._prefix)
        }
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._handle = None

    def read(self, key: str) -> torch.Tensor:
        if self._handle is None:
            raise RuntimeError("SafeTensorSource must be used as a context manager")
        return self._handle.get_tensor(f"{self._prefix}{key}")


def _source_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "action_in_proj.weight": (_DECODER_WIDTH, _ACTION_WIDTH),
        "action_in_proj.bias": (_DECODER_WIDTH,),
        "action_out_proj.weight": (_ACTION_WIDTH, _DECODER_WIDTH),
        "action_out_proj.bias": (_ACTION_WIDTH,),
        "time_mlp_in.weight": (_DECODER_WIDTH, _DECODER_WIDTH),
        "time_mlp_in.bias": (_DECODER_WIDTH,),
        "time_mlp_out.weight": (_DECODER_WIDTH, _DECODER_WIDTH),
        "time_mlp_out.bias": (_DECODER_WIDTH,),
        f"{_VISION}.embeddings.patch_embedding.weight": (_VISION_WIDTH, 3, 14, 14),
        f"{_VISION}.embeddings.patch_embedding.bias": (_VISION_WIDTH,),
        f"{_VISION}.embeddings.position_embedding.weight": (256, _VISION_WIDTH),
        f"{_VISION}.post_layernorm.weight": (_VISION_WIDTH,),
        f"{_VISION}.post_layernorm.bias": (_VISION_WIDTH,),
        f"{_PROJECTOR}.weight": (_ENCODER_WIDTH, _VISION_WIDTH),
        f"{_PROJECTOR}.bias": (_ENCODER_WIDTH,),
        f"{_ENCODER}.embed_tokens.weight": (_VOCAB_SIZE, _ENCODER_WIDTH),
        f"{_ENCODER}.norm.weight": (_ENCODER_WIDTH,),
        f"{_DECODER}.norm.dense.weight": (3 * _DECODER_WIDTH, _DECODER_WIDTH),
        f"{_DECODER}.norm.dense.bias": (3 * _DECODER_WIDTH,),
    }
    for layer in range(_VISION_DEPTH):
        root = f"{_VISION}.encoder.layers.{layer}"
        shapes.update(
            {
                f"{root}.self_attn.q_proj.weight": (_VISION_WIDTH, _VISION_WIDTH),
                f"{root}.self_attn.q_proj.bias": (_VISION_WIDTH,),
                f"{root}.self_attn.k_proj.weight": (_VISION_WIDTH, _VISION_WIDTH),
                f"{root}.self_attn.k_proj.bias": (_VISION_WIDTH,),
                f"{root}.self_attn.v_proj.weight": (_VISION_WIDTH, _VISION_WIDTH),
                f"{root}.self_attn.v_proj.bias": (_VISION_WIDTH,),
                f"{root}.self_attn.out_proj.weight": (_VISION_WIDTH, _VISION_WIDTH),
                f"{root}.self_attn.out_proj.bias": (_VISION_WIDTH,),
                f"{root}.mlp.fc1.weight": (4304, _VISION_WIDTH),
                f"{root}.mlp.fc1.bias": (4304,),
                f"{root}.mlp.fc2.weight": (_VISION_WIDTH, 4304),
                f"{root}.mlp.fc2.bias": (_VISION_WIDTH,),
                f"{root}.layer_norm1.weight": (_VISION_WIDTH,),
                f"{root}.layer_norm1.bias": (_VISION_WIDTH,),
                f"{root}.layer_norm2.weight": (_VISION_WIDTH,),
                f"{root}.layer_norm2.bias": (_VISION_WIDTH,),
            }
        )
    for layer in range(_ENCODER_DEPTH):
        root = f"{_ENCODER}.layers.{layer}"
        shapes.update(
            {
                f"{root}.self_attn.q_proj.weight": (_NUM_HEADS * _HEAD_DIM, _ENCODER_WIDTH),
                f"{root}.self_attn.k_proj.weight": (_HEAD_DIM, _ENCODER_WIDTH),
                f"{root}.self_attn.v_proj.weight": (_HEAD_DIM, _ENCODER_WIDTH),
                f"{root}.self_attn.o_proj.weight": (_ENCODER_WIDTH, _NUM_HEADS * _HEAD_DIM),
                f"{root}.mlp.gate_proj.weight": (16384, _ENCODER_WIDTH),
                f"{root}.mlp.up_proj.weight": (16384, _ENCODER_WIDTH),
                f"{root}.mlp.down_proj.weight": (_ENCODER_WIDTH, 16384),
                f"{root}.input_layernorm.weight": (_ENCODER_WIDTH,),
                f"{root}.post_attention_layernorm.weight": (_ENCODER_WIDTH,),
            }
        )
    for layer in range(_DECODER_DEPTH):
        root = f"{_DECODER}.layers.{layer}"
        shapes.update(
            {
                f"{root}.self_attn.q_proj.weight": (_NUM_HEADS * _HEAD_DIM, _DECODER_WIDTH),
                f"{root}.self_attn.k_proj.weight": (_HEAD_DIM, _DECODER_WIDTH),
                f"{root}.self_attn.v_proj.weight": (_HEAD_DIM, _DECODER_WIDTH),
                f"{root}.self_attn.o_proj.weight": (_DECODER_WIDTH, _NUM_HEADS * _HEAD_DIM),
                f"{root}.mlp.gate_proj.weight": (4096, _DECODER_WIDTH),
                f"{root}.mlp.up_proj.weight": (4096, _DECODER_WIDTH),
                f"{root}.mlp.down_proj.weight": (_DECODER_WIDTH, 4096),
                f"{root}.input_layernorm.dense.weight": (3 * _DECODER_WIDTH, _DECODER_WIDTH),
                f"{root}.input_layernorm.dense.bias": (3 * _DECODER_WIDTH,),
                f"{root}.post_attention_layernorm.dense.weight": (
                    3 * _DECODER_WIDTH,
                    _DECODER_WIDTH,
                ),
                f"{root}.post_attention_layernorm.dense.bias": (3 * _DECODER_WIDTH,),
            }
        )
    return shapes


def _optional_source_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "paligemma_with_expert.paligemma.lm_head.weight": (_VOCAB_SIZE, _ENCODER_WIDTH),
        "paligemma_with_expert.gemma_expert.lm_head.weight": (_VOCAB_SIZE, _DECODER_WIDTH),
    }


def validate_source_shapes(source: SafeTensorSource) -> None:
    expected = _source_shapes()
    optional = _optional_source_shapes()
    missing = sorted(expected.keys() - source.shapes.keys())
    unexpected = sorted(source.shapes.keys() - expected.keys() - optional.keys())
    mismatched = [
        (key, source.shapes[key], shape)
        for key, shape in {**expected, **optional}.items()
        if key in source.shapes and source.shapes[key] != shape
    ]
    problems: list[str] = []
    if missing:
        problems.append(f"missing {len(missing)} keys (first: {missing[:5]})")
    if unexpected:
        problems.append(f"unexpected {len(unexpected)} keys (first: {unexpected[:5]})")
    if mismatched:
        rendered = [f"{key}: got {got}, expected {want}" for key, got, want in mismatched[:5]]
        problems.append(f"mismatched {len(mismatched)} shapes (first: {rendered})")
    if problems:
        raise ValueError("Checkpoint is not the supported PI0.5 architecture: " + "; ".join(problems))


def validate_pi05_config(config: dict[str, Any]) -> None:
    expected_scalars = {
        "type": "pi05",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "n_obs_steps": 1,
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": _ACTION_WIDTH,
        "max_action_dim": _ACTION_WIDTH,
        "num_inference_steps": 10,
        "use_relative_actions": False,
    }
    errors = [
        f"{key}={config.get(key)!r}, expected {expected!r}"
        for key, expected in expected_scalars.items()
        if config.get(key) != expected
    ]
    if tuple(config.get("image_resolution", ())) != (224, 224):
        errors.append(f"image_resolution={config.get('image_resolution')!r}, expected [224, 224]")
    if config.get("train_expert_only") is not False:
        errors.append("train_expert_only must be false for a full-model checkpoint")
    if config.get("freeze_vision_encoder") is not False:
        errors.append("freeze_vision_encoder must be false for a full-model checkpoint")

    input_features = config.get("input_features")
    output_features = config.get("output_features")
    if not isinstance(input_features, dict):
        input_features = {}
        errors.append("input_features must be an object")
    if not isinstance(output_features, dict):
        output_features = {}
        errors.append("output_features must be an object")
    malformed_features = [
        name
        for name, feature in {**input_features, **output_features}.items()
        if not isinstance(feature, dict)
    ]
    if malformed_features:
        errors.append(f"feature definitions must be objects: {malformed_features}")
    valid_inputs = [feature for feature in input_features.values() if isinstance(feature, dict)]
    valid_outputs = [feature for feature in output_features.values() if isinstance(feature, dict)]
    visual_features = [feature for feature in valid_inputs if feature.get("type") == "VISUAL"]
    state_features = [feature for feature in valid_inputs if feature.get("type") == "STATE"]
    action_features = [feature for feature in valid_outputs if feature.get("type") == "ACTION"]
    if len(visual_features) != 2:
        errors.append(f"input_features must contain exactly 2 cameras, got {len(visual_features)}")
    if len(state_features) != 1 or tuple(state_features[0].get("shape", ())) != (6,):
        errors.append("input_features must contain one 6-dimensional state")
    if len(action_features) != 1 or tuple(action_features[0].get("shape", ())) != (6,):
        errors.append("output_features must contain one 6-dimensional action")

    rtc_delay = config.get("rtc_training_max_delay")
    if isinstance(rtc_delay, bool) or not isinstance(rtc_delay, int) or rtc_delay <= 0:
        errors.append("rtc_training_max_delay must be a positive integer for trained-prefix RTC")
    elif isinstance(config.get("chunk_size"), int) and rtc_delay >= config["chunk_size"]:
        errors.append("rtc_training_max_delay must be smaller than chunk_size")
    if errors:
        raise ValueError("Unsupported PI0.5 config: " + "; ".join(errors))


def _linear_kernel(weight: torch.Tensor, input_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Convert HF ``[out, in]`` weight to Triton ``[in, out]`` layout."""
    if weight.ndim != 2:
        raise ValueError(f"Expected a rank-2 linear weight, got shape {tuple(weight.shape)}")
    if input_scale is None:
        return weight.transpose(0, 1).contiguous()
    if input_scale.ndim != 1 or input_scale.shape[0] != weight.shape[1]:
        raise ValueError(
            f"RMS scale shape {tuple(input_scale.shape)} does not match linear input {weight.shape[1]}"
        )
    # PiGemmaRMSNorm applies (1 + weight). Fold that input scaling into
    # the following projection, as the upstream encoder kernels expect.
    folded = weight.float() * (1.0 + input_scale.float()).unsqueeze(0)
    return folded.transpose(0, 1).contiguous()


def _interleave_rope_columns(kernel: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    """Change split-half RoPE columns to the pairwise layout used by Triton."""
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    expected = num_heads * head_dim
    if kernel.ndim != 2 or kernel.shape[1] != expected:
        raise ValueError(f"Expected kernel shape [in, {expected}], got {tuple(kernel.shape)}")
    input_width = kernel.shape[0]
    return (
        kernel.reshape(input_width, num_heads, 2, head_dim // 2)
        .transpose(-2, -1)
        .reshape(input_width, expected)
        .contiguous()
    )


def _fused_qkv_kernel(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    *,
    input_scale: torch.Tensor | None = None,
    num_heads: int = _NUM_HEADS,
    head_dim: int = _HEAD_DIM,
) -> torch.Tensor:
    q = _interleave_rope_columns(
        _linear_kernel(q_weight, input_scale), num_heads=num_heads, head_dim=head_dim
    )
    k = _interleave_rope_columns(_linear_kernel(k_weight, input_scale), num_heads=1, head_dim=head_dim)
    v = _linear_kernel(v_weight, input_scale)
    return torch.cat((q, k, v), dim=1).contiguous()


def _vision_patch_kernel(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 4:
        raise ValueError(f"Expected [out, in, height, width], got {tuple(weight.shape)}")
    return weight.permute(2, 3, 1, 0).contiguous()


def _bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device="cpu", dtype=torch.bfloat16).contiguous()


def _stack_layers(
    count: int,
    element_shape: tuple[int, ...],
    build: Callable[[int], torch.Tensor],
) -> torch.Tensor:
    result = torch.empty((count, *element_shape), dtype=torch.bfloat16, device="cpu")
    for layer in range(count):
        value = _bf16(build(layer))
        if tuple(value.shape) != element_shape:
            raise ValueError(
                f"Converted layer {layer} has shape {tuple(value.shape)}, expected {element_shape}"
            )
        result[layer].copy_(value)
    return result


def prepare_time_embeddings(
    *,
    num_steps: int = 10,
    dimension: int = _DECODER_WIDTH,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    if num_steps <= 0 or dimension % 2:
        raise ValueError("num_steps must be positive and dimension must be even")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    dt = -1.0 / num_steps
    times = torch.tensor([1.0 + step * dt for step in range(num_steps)], dtype=torch.float32)
    scaling_factor = 1.0 / period * 2 * math.pi
    sinusoid = scaling_factor[None, :] * times[:, None]
    return _bf16(torch.cat((torch.sin(sinusoid), torch.cos(sinusoid)), dim=1))


def _prompt_token_ids(prompt: str, tokenizer_path: str, max_length: int) -> torch.Tensor:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    normalized_prompt = prompt.strip().replace("_", " ") + "\n"
    token_ids = tokenizer([normalized_prompt], max_length=max_length, return_tensors="pt")["input_ids"]
    return token_ids.squeeze(0).to(device="cpu", dtype=torch.long)


def _target_shapes(prompt_len: int) -> dict[str, tuple[int, ...]]:
    return {
        "embedding_weight": (_VOCAB_SIZE, _ENCODER_WIDTH),
        "vision_patch_embedding_w": (14, 14, 3, _VISION_WIDTH),
        "vision_patch_embedding_b": (_VISION_WIDTH,),
        "vision_position_embedding": (256, _VISION_WIDTH),
        "vision_attn_qkv_w": (_VISION_DEPTH, _VISION_WIDTH, 3 * _VISION_WIDTH),
        "vision_attn_qkv_b": (_VISION_DEPTH, 3 * _VISION_WIDTH),
        "vision_attn_o_w": (_VISION_DEPTH, _VISION_WIDTH, _VISION_WIDTH),
        "vision_attn_o_b": (_VISION_DEPTH, _VISION_WIDTH),
        "vision_ffn_up_w": (_VISION_DEPTH, _VISION_WIDTH, 4304),
        "vision_ffn_up_b": (_VISION_DEPTH, 4304),
        "vision_ffn_down_w": (_VISION_DEPTH, 4304, _VISION_WIDTH),
        "vision_ffn_down_b": (_VISION_DEPTH, _VISION_WIDTH),
        "vision_pre_attn_norm_w": (_VISION_DEPTH, _VISION_WIDTH),
        "vision_pre_attn_norm_b": (_VISION_DEPTH, _VISION_WIDTH),
        "vision_pre_ffn_norm_w": (_VISION_DEPTH, _VISION_WIDTH),
        "vision_pre_ffn_norm_b": (_VISION_DEPTH, _VISION_WIDTH),
        "vision_final_norm_w": (_VISION_WIDTH,),
        "vision_final_norm_b": (_VISION_WIDTH,),
        "encoder_multi_modal_projector_w": (_VISION_WIDTH, _ENCODER_WIDTH),
        "encoder_multi_modal_projector_b": (_ENCODER_WIDTH,),
        "encoder_attn_qkv_w": (_ENCODER_DEPTH, _ENCODER_WIDTH, 2560),
        "encoder_attn_o_w": (_ENCODER_DEPTH, _ENCODER_WIDTH, _ENCODER_WIDTH),
        "encoder_ffn_gate_w": (_ENCODER_DEPTH, _ENCODER_WIDTH, 16384),
        "encoder_ffn_up_w": (_ENCODER_DEPTH, _ENCODER_WIDTH, 16384),
        "encoder_ffn_down_w": (_ENCODER_DEPTH, 16384, _ENCODER_WIDTH),
        "decoder_time_embeds": (10, _DECODER_WIDTH),
        "decoder_time_mlp_in_w": (_DECODER_WIDTH, _DECODER_WIDTH),
        "decoder_time_mlp_in_b": (_DECODER_WIDTH,),
        "decoder_time_mlp_out_w": (_DECODER_WIDTH, _DECODER_WIDTH),
        "decoder_time_mlp_out_b": (_DECODER_WIDTH,),
        "decoder_pre_attn_norm_mod_w": (_DECODER_DEPTH, _DECODER_WIDTH, 3 * _DECODER_WIDTH),
        "decoder_pre_attn_norm_mod_b": (_DECODER_DEPTH, 3 * _DECODER_WIDTH),
        "decoder_pre_ffn_norm_mod_w": (_DECODER_DEPTH, _DECODER_WIDTH, 3 * _DECODER_WIDTH),
        "decoder_pre_ffn_norm_mod_b": (_DECODER_DEPTH, 3 * _DECODER_WIDTH),
        "decoder_final_norm_mod_w": (_DECODER_WIDTH, 3 * _DECODER_WIDTH),
        "decoder_final_norm_mod_b": (3 * _DECODER_WIDTH,),
        "decoder_attn_qkv_w": (_DECODER_DEPTH, _DECODER_WIDTH, 2560),
        "decoder_attn_o_w": (_DECODER_DEPTH, 2048, _DECODER_WIDTH),
        "decoder_ffn_gate_w": (_DECODER_DEPTH, _DECODER_WIDTH, 4096),
        "decoder_ffn_up_w": (_DECODER_DEPTH, _DECODER_WIDTH, 4096),
        "decoder_ffn_down_w": (_DECODER_DEPTH, 4096, _DECODER_WIDTH),
        "decoder_action_in_proj_w": (_ACTION_WIDTH, _DECODER_WIDTH),
        "decoder_action_in_proj_b": (_DECODER_WIDTH,),
        "decoder_action_out_proj_w": (_DECODER_WIDTH, _ACTION_WIDTH),
        "decoder_action_out_proj_b": (_ACTION_WIDTH,),
        "language_embeds": (prompt_len, _ENCODER_WIDTH),
    }


def convert_weights(source: SafeTensorSource, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}

    embedding = _bf16(source.read(f"{_ENCODER}.embed_tokens.weight"))
    weights["embedding_weight"] = embedding
    weights["language_embeds"] = _bf16(embedding.index_select(0, token_ids) * math.sqrt(_ENCODER_WIDTH))
    weights["decoder_time_embeds"] = prepare_time_embeddings()

    weights["vision_patch_embedding_w"] = _bf16(
        _vision_patch_kernel(source.read(f"{_VISION}.embeddings.patch_embedding.weight"))
    )
    weights["vision_patch_embedding_b"] = _bf16(source.read(f"{_VISION}.embeddings.patch_embedding.bias"))
    weights["vision_position_embedding"] = _bf16(
        source.read(f"{_VISION}.embeddings.position_embedding.weight")
    )

    def vision_key(layer: int, suffix: str) -> str:
        return f"{_VISION}.encoder.layers.{layer}.{suffix}"

    weights["vision_attn_qkv_w"] = _stack_layers(
        _VISION_DEPTH,
        (_VISION_WIDTH, 3 * _VISION_WIDTH),
        lambda layer: torch.cat(
            tuple(
                _linear_kernel(source.read(vision_key(layer, f"self_attn.{name}_proj.weight")))
                for name in ("q", "k", "v")
            ),
            dim=1,
        ),
    )
    weights["vision_attn_qkv_b"] = _stack_layers(
        _VISION_DEPTH,
        (3 * _VISION_WIDTH,),
        lambda layer: torch.cat(
            tuple(source.read(vision_key(layer, f"self_attn.{name}_proj.bias")) for name in ("q", "k", "v"))
        ),
    )
    vision_specs = {
        "vision_attn_o_w": ("self_attn.out_proj.weight", (_VISION_WIDTH, _VISION_WIDTH), True),
        "vision_attn_o_b": ("self_attn.out_proj.bias", (_VISION_WIDTH,), False),
        "vision_ffn_up_w": ("mlp.fc1.weight", (_VISION_WIDTH, 4304), True),
        "vision_ffn_up_b": ("mlp.fc1.bias", (4304,), False),
        "vision_ffn_down_w": ("mlp.fc2.weight", (4304, _VISION_WIDTH), True),
        "vision_ffn_down_b": ("mlp.fc2.bias", (_VISION_WIDTH,), False),
        "vision_pre_attn_norm_w": ("layer_norm1.weight", (_VISION_WIDTH,), False),
        "vision_pre_attn_norm_b": ("layer_norm1.bias", (_VISION_WIDTH,), False),
        "vision_pre_ffn_norm_w": ("layer_norm2.weight", (_VISION_WIDTH,), False),
        "vision_pre_ffn_norm_b": ("layer_norm2.bias", (_VISION_WIDTH,), False),
    }
    for target, (suffix, shape, transpose) in vision_specs.items():
        weights[target] = _stack_layers(
            _VISION_DEPTH,
            shape,
            lambda layer, suffix=suffix, transpose=transpose: (
                _linear_kernel(source.read(vision_key(layer, suffix)))
                if transpose
                else source.read(vision_key(layer, suffix))
            ),
        )
    weights["vision_final_norm_w"] = _bf16(source.read(f"{_VISION}.post_layernorm.weight"))
    weights["vision_final_norm_b"] = _bf16(source.read(f"{_VISION}.post_layernorm.bias"))
    weights["encoder_multi_modal_projector_w"] = _bf16(_linear_kernel(source.read(f"{_PROJECTOR}.weight")))
    weights["encoder_multi_modal_projector_b"] = _bf16(source.read(f"{_PROJECTOR}.bias"))

    def encoder_key(layer: int, suffix: str) -> str:
        return f"{_ENCODER}.layers.{layer}.{suffix}"

    weights["encoder_attn_qkv_w"] = _stack_layers(
        _ENCODER_DEPTH,
        (_ENCODER_WIDTH, 2560),
        lambda layer: _fused_qkv_kernel(
            source.read(encoder_key(layer, "self_attn.q_proj.weight")),
            source.read(encoder_key(layer, "self_attn.k_proj.weight")),
            source.read(encoder_key(layer, "self_attn.v_proj.weight")),
            input_scale=source.read(encoder_key(layer, "input_layernorm.weight")),
        ),
    )
    weights["encoder_attn_o_w"] = _stack_layers(
        _ENCODER_DEPTH,
        (_ENCODER_WIDTH, _ENCODER_WIDTH),
        lambda layer: _linear_kernel(source.read(encoder_key(layer, "self_attn.o_proj.weight"))),
    )
    for target, suffix in (
        ("encoder_ffn_gate_w", "mlp.gate_proj.weight"),
        ("encoder_ffn_up_w", "mlp.up_proj.weight"),
    ):
        weights[target] = _stack_layers(
            _ENCODER_DEPTH,
            (_ENCODER_WIDTH, 16384),
            lambda layer, suffix=suffix: _linear_kernel(
                source.read(encoder_key(layer, suffix)),
                source.read(encoder_key(layer, "post_attention_layernorm.weight")),
            ),
        )
    weights["encoder_ffn_down_w"] = _stack_layers(
        _ENCODER_DEPTH,
        (16384, _ENCODER_WIDTH),
        lambda layer: _linear_kernel(source.read(encoder_key(layer, "mlp.down_proj.weight"))),
    )

    def decoder_key(layer: int, suffix: str) -> str:
        return f"{_DECODER}.layers.{layer}.{suffix}"

    weights["decoder_attn_qkv_w"] = _stack_layers(
        _DECODER_DEPTH,
        (_DECODER_WIDTH, 2560),
        lambda layer: _fused_qkv_kernel(
            source.read(decoder_key(layer, "self_attn.q_proj.weight")),
            source.read(decoder_key(layer, "self_attn.k_proj.weight")),
            source.read(decoder_key(layer, "self_attn.v_proj.weight")),
        ),
    )
    decoder_specs = {
        "decoder_attn_o_w": ("self_attn.o_proj.weight", (2048, _DECODER_WIDTH)),
        "decoder_ffn_gate_w": ("mlp.gate_proj.weight", (_DECODER_WIDTH, 4096)),
        "decoder_ffn_up_w": ("mlp.up_proj.weight", (_DECODER_WIDTH, 4096)),
        "decoder_ffn_down_w": ("mlp.down_proj.weight", (4096, _DECODER_WIDTH)),
        "decoder_pre_attn_norm_mod_w": (
            "input_layernorm.dense.weight",
            (_DECODER_WIDTH, 3 * _DECODER_WIDTH),
        ),
        "decoder_pre_ffn_norm_mod_w": (
            "post_attention_layernorm.dense.weight",
            (_DECODER_WIDTH, 3 * _DECODER_WIDTH),
        ),
    }
    for target, (suffix, shape) in decoder_specs.items():
        weights[target] = _stack_layers(
            _DECODER_DEPTH,
            shape,
            lambda layer, suffix=suffix: _linear_kernel(source.read(decoder_key(layer, suffix))),
        )
    weights["decoder_pre_attn_norm_mod_b"] = _stack_layers(
        _DECODER_DEPTH,
        (3 * _DECODER_WIDTH,),
        lambda layer: source.read(decoder_key(layer, "input_layernorm.dense.bias")),
    )
    weights["decoder_pre_ffn_norm_mod_b"] = _stack_layers(
        _DECODER_DEPTH,
        (3 * _DECODER_WIDTH,),
        lambda layer: source.read(decoder_key(layer, "post_attention_layernorm.dense.bias")),
    )
    weights["decoder_final_norm_mod_w"] = _bf16(_linear_kernel(source.read(f"{_DECODER}.norm.dense.weight")))
    weights["decoder_final_norm_mod_b"] = _bf16(source.read(f"{_DECODER}.norm.dense.bias"))

    for target, source_key in (
        ("decoder_time_mlp_in_w", "time_mlp_in.weight"),
        ("decoder_time_mlp_out_w", "time_mlp_out.weight"),
        ("decoder_action_in_proj_w", "action_in_proj.weight"),
        ("decoder_action_out_proj_w", "action_out_proj.weight"),
    ):
        weights[target] = _bf16(_linear_kernel(source.read(source_key)))
    for target, source_key in (
        ("decoder_time_mlp_in_b", "time_mlp_in.bias"),
        ("decoder_time_mlp_out_b", "time_mlp_out.bias"),
        ("decoder_action_in_proj_b", "action_in_proj.bias"),
        ("decoder_action_out_proj_b", "action_out_proj.bias"),
    ):
        weights[target] = _bf16(source.read(source_key))

    expected = _target_shapes(int(token_ids.numel()))
    actual = {key: tuple(value.shape) for key, value in weights.items()}
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        mismatched = {
            key: (actual[key], expected[key])
            for key in actual.keys() & expected.keys()
            if actual[key] != expected[key]
        }
        raise RuntimeError(
            f"Internal target-shape validation failed: missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}"
        )
    if any(value.dtype != torch.bfloat16 or value.device.type != "cpu" for value in weights.values()):
        raise RuntimeError("Internal dtype validation failed: every exported tensor must be CPU BF16")
    return weights


def resolve_checkpoint(checkpoint: Path, config_override: Path | None = None) -> CheckpointPaths:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_dir() and not (checkpoint / "model.safetensors").is_file():
        nested = checkpoint / "pretrained_model"
        if (nested / "model.safetensors").is_file():
            checkpoint = nested
    weights = checkpoint if checkpoint.is_file() else checkpoint / "model.safetensors"
    config = config_override.expanduser().resolve() if config_override else weights.parent / "config.json"
    if not weights.is_file():
        raise FileNotFoundError(f"PI0.5 weights not found: {weights}")
    if not config.is_file():
        raise FileNotFoundError(f"PI0.5 config not found: {config}")
    return CheckpointPaths(weights=weights, config=config)


def _load_and_validate(paths: CheckpointPaths) -> dict[str, Any]:
    with paths.config.open(encoding="utf-8") as file:
        config = json.load(file)
    validate_pi05_config(config)
    with SafeTensorSource(paths.weights) as source:
        validate_source_shapes(source)
    return config


def export_checkpoint(
    paths: CheckpointPaths,
    output: Path,
    *,
    prompt: str,
    tokenizer_path: str,
    tokenizer_max_length: int = 48,
    overwrite: bool = False,
) -> None:
    if tokenizer_max_length <= 0:
        raise ValueError("tokenizer_max_length must be positive")
    _load_and_validate(paths)
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite to replace it): {output}")
    if output == paths.weights:
        raise ValueError("Output path must differ from the source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)

    token_ids = _prompt_token_ids(prompt, tokenizer_path, tokenizer_max_length)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with SafeTensorSource(paths.weights) as source:
            validate_source_shapes(source)
            weights = convert_weights(source, token_ids)
        with temporary.open("wb") as file:
            pickle.dump(weights, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _estimated_target_bytes(prompt_len: int = 48) -> int:
    return sum(math.prod(shape) * 2 for shape in _target_shapes(prompt_len).values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a full-trained LeRobot PI0.5 checkpoint to realtime-vla-v2 Pi05RTC pickle."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Checkpoint directory or model.safetensors"
    )
    parser.add_argument("--config", type=Path, help="Override the adjacent config.json")
    parser.add_argument("--output", type=Path, help="Destination .pkl used by realtime-vla-v2/server")
    parser.add_argument("--prompt", help="Static fallback prompt embedded in the pickle")
    parser.add_argument("--tokenizer-path", help="Local path or Hub ID for the PaliGemma tokenizer")
    parser.add_argument("--tokenizer-max-length", type=int, default=48)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and safetensors metadata without loading or writing model tensors",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.validate_only:
        missing = [name for name in ("output", "prompt", "tokenizer_path") if getattr(args, name) is None]
        if missing:
            parser.error(
                "the following arguments are required unless --validate-only is set: " + ", ".join(missing)
            )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_checkpoint(args.checkpoint, args.config)
    estimate_gib = _estimated_target_bytes() / 1024**3
    if args.validate_only:
        config = _load_and_validate(paths)
        print(
            f"Validated full-trained PI0.5 RTC checkpoint: {paths.weights} "
            f"(cameras=2, state/action=6, padded_action_width=32, "
            f"rtc_training_max_delay={config['rtc_training_max_delay']}, estimated_export={estimate_gib:.2f} GiB)"
        )
        return 0

    export_checkpoint(
        paths,
        args.output,
        prompt=args.prompt,
        tokenizer_path=args.tokenizer_path,
        tokenizer_max_length=args.tokenizer_max_length,
        overwrite=args.overwrite,
    )
    print(f"Exported realtime-vla-v2 Pi05RTC weights to {args.output} ({estimate_gib:.2f} GiB estimated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
