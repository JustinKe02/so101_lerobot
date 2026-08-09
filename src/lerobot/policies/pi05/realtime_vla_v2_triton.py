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

"""Full-model PI0.5 RTC inference through realtime-vla-v2 Triton kernels.

This module deliberately does not import Triton or the vendored kernels at
module import time. ``PI05RealtimeVLATritonBackend.from_export`` performs all
artifact and environment checks before importing the CUDA implementation or
allocating model memory.

The public boundary is policy space: two named camera images, a normalized
six-dimensional state, and an optional normalized six-dimensional committed
action prefix. The Triton model remains the original 32-wide, 50-token action
expert with ten Euler steps. Physical-unit normalization and relative-action
conversion belong to the LeRobot processor pipeline, not this backend.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import torch

NUM_CAMERAS = 2
STATE_DIM = 6
ACTION_DIM = 6
PADDED_ACTION_DIM = 32
ACTION_CHUNK_SIZE = 50
EULER_STEPS = 10
IMAGE_SIZE = (224, 224)


class PI05RealtimeVLATritonError(RuntimeError):
    """Base error for the full-model Triton backend."""


class PI05RealtimeVLATritonUnavailableError(PI05RealtimeVLATritonError):
    """Raised when the host cannot execute the CUDA/Triton runtime."""


@runtime_checkable
class PI05RTCTritonRuntime(Protocol):
    """Small interface implemented by the vendored CUDA graph runtime."""

    def forward(
        self,
        observation_images_normalized: torch.Tensor,
        diffusion_noise: torch.Tensor,
        task_prompt: str | None = None,
        state_tokens: np.ndarray | None = None,
        action_prefill_len: int | None = None,
        prefill_actions: np.ndarray | None = None,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class PI05RealtimeVLATritonConfig:
    """Static runtime and artifact configuration for SO-101 PI0.5 RTC."""

    prompt: str
    tokenizer_path: str
    camera_keys: tuple[str, str]
    weights_path: str | Path | None = None
    model_config_path: str | Path | None = None
    weights_sha256: str | None = None
    device: str = "cuda"
    image_value_range: Literal["uint8", "zero_one", "minus_one_one"] = "zero_one"
    tokenizer_max_length: int = 200
    prompt_capacity: int = 64
    warmup_prefill_lengths: tuple[int, ...] | None = None
    noise_seed: int | None = None
    min_free_cuda_gib: float = 18.0
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    padded_action_dim: int = PADDED_ACTION_DIM
    chunk_size: int = ACTION_CHUNK_SIZE
    num_inference_steps: int = EULER_STEPS
    image_size: tuple[int, int] = IMAGE_SIZE

    def __post_init__(self) -> None:
        validate_runtime_config(self)


@dataclass(frozen=True)
class PI05RealtimeVLATritonEnvironment:
    torch_version: str
    triton_version: str
    device_name: str
    compute_capability: tuple[int, int]


@dataclass(frozen=True)
class PI05RealtimeVLATritonOutput:
    """One full aligned action chunk returned in normalized policy space."""

    actions: np.ndarray
    prefill_length: int

    @property
    def postfix_actions(self) -> np.ndarray:
        return self.actions[self.prefill_length :]


@dataclass(frozen=True)
class FixedNoiseParityReport:
    passed: bool
    max_abs_error: float
    mean_abs_error: float
    atol: float
    rtol: float
    shape: tuple[int, ...]


def validate_runtime_config(config: PI05RealtimeVLATritonConfig) -> None:
    """Reject runtime shapes that the converted PI0.5 kernels cannot execute."""

    errors: list[str] = []
    if not config.prompt.strip():
        errors.append("prompt must be non-empty")
    if not config.tokenizer_path.strip():
        errors.append("tokenizer_path must be non-empty")
    if len(config.camera_keys) != NUM_CAMERAS or len(set(config.camera_keys)) != NUM_CAMERAS:
        errors.append(f"camera_keys must contain {NUM_CAMERAS} unique names")
    if any(not isinstance(key, str) or not key for key in config.camera_keys):
        errors.append("camera_keys must contain non-empty strings")

    fixed_values = (
        ("state_dim", config.state_dim, STATE_DIM),
        ("action_dim", config.action_dim, ACTION_DIM),
        ("padded_action_dim", config.padded_action_dim, PADDED_ACTION_DIM),
        ("chunk_size", config.chunk_size, ACTION_CHUNK_SIZE),
        ("num_inference_steps", config.num_inference_steps, EULER_STEPS),
        ("image_size", tuple(config.image_size), IMAGE_SIZE),
    )
    errors.extend(
        f"{name}={actual!r}, expected {expected!r}"
        for name, actual, expected in fixed_values
        if actual != expected
    )

    if config.image_value_range not in {"uint8", "zero_one", "minus_one_one"}:
        errors.append(f"unsupported image_value_range={config.image_value_range!r}")
    if config.tokenizer_max_length <= 0:
        errors.append("tokenizer_max_length must be positive")
    if not 0 < config.prompt_capacity <= config.tokenizer_max_length:
        errors.append("prompt_capacity must be in [1, tokenizer_max_length]")
    if not re.fullmatch(r"(?:cpu|cuda(?::\d+)?)", config.device):
        errors.append("device must be 'cpu', 'cuda', or 'cuda:<index>'")
    if config.min_free_cuda_gib < 0:
        errors.append("min_free_cuda_gib must be non-negative")
    if config.weights_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", config.weights_sha256):
        errors.append("weights_sha256 must contain exactly 64 hexadecimal characters")

    if config.warmup_prefill_lengths is not None:
        lengths = config.warmup_prefill_lengths
        if tuple(sorted(set(lengths))) != lengths:
            errors.append("warmup_prefill_lengths must be sorted and unique")
        if any(isinstance(length, bool) or not isinstance(length, int) for length in lengths):
            errors.append("warmup_prefill_lengths must contain integers")
        elif any(length < 0 or length >= ACTION_CHUNK_SIZE for length in lengths):
            errors.append(f"warmup_prefill_lengths must be in [0, {ACTION_CHUNK_SIZE})")
    if errors:
        raise ValueError("Invalid PI0.5 realtime-vla-v2 Triton config: " + "; ".join(errors))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_runtime_environment(device: str = "cuda") -> PI05RealtimeVLATritonEnvironment:
    """Import optional dependencies lazily and verify a BF16-capable CUDA GPU."""

    if not device.startswith("cuda"):
        raise PI05RealtimeVLATritonUnavailableError(
            f"The full Triton runtime requires CUDA, got device={device!r}"
        )
    try:
        import triton
    except ImportError as exc:
        raise PI05RealtimeVLATritonUnavailableError(
            "Triton is not installed. Install a CUDA PyTorch build that provides a compatible Triton package."
        ) from exc

    if not torch.cuda.is_available():
        raise PI05RealtimeVLATritonUnavailableError("torch.cuda.is_available() is false")
    try:
        torch_device = torch.device(device)
        index = torch_device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
    except (AssertionError, RuntimeError, ValueError) as exc:
        raise PI05RealtimeVLATritonUnavailableError(
            f"Cannot select Triton CUDA device {device!r}: {exc}"
        ) from exc
    capability = (int(properties.major), int(properties.minor))
    if capability < (8, 0):
        raise PI05RealtimeVLATritonUnavailableError(
            f"PI0.5 BF16 Triton kernels require compute capability >= 8.0, got {capability}"
        )
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise PI05RealtimeVLATritonUnavailableError(
            f"CUDA device {properties.name!r} does not provide native BF16"
        )
    return PI05RealtimeVLATritonEnvironment(
        torch_version=torch.__version__,
        triton_version=str(getattr(triton, "__version__", "unknown")),
        device_name=str(properties.name),
        compute_capability=capability,
    )


def require_cuda_free_memory(device: str, minimum_gib: float) -> tuple[int, int]:
    """Fail before model loading when another process leaves too little CUDA memory."""

    if minimum_gib < 0:
        raise ValueError("minimum_gib must be non-negative")
    if not device.startswith("cuda"):
        raise PI05RealtimeVLATritonUnavailableError(
            f"CUDA memory preflight requires a CUDA device, got {device!r}"
        )
    torch_device = torch.device(device)
    index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    required_bytes = int(minimum_gib * 1024**3)
    if free_bytes < required_bytes:
        raise PI05RealtimeVLATritonUnavailableError(
            f"Insufficient free CUDA memory on {device}: {free_bytes / 1024**3:.2f} GiB free, "
            f"{minimum_gib:.2f} GiB required. Stop or finish other GPU jobs before loading PI0.5."
        )
    return int(free_bytes), int(total_bytes)


def load_and_validate_model_config(path: str | Path) -> dict[str, Any]:
    """Load the original LeRobot config and enforce the exporter contract."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PI0.5 model config not found: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"PI0.5 model config must be a JSON object: {path}")
    from lerobot.scripts.lerobot_export_pi05_realtime_vla_v2 import validate_pi05_config

    validate_pi05_config(value)
    return value


def validate_exported_weights(weights: Mapping[str, Any]) -> None:
    """Validate every exported tensor without copying or moving it to CUDA."""

    language = weights.get("language_embeds")
    prompt_len = int(language.shape[0]) if isinstance(language, torch.Tensor) and language.ndim == 2 else 0
    from lerobot.scripts.lerobot_export_pi05_realtime_vla_v2 import _target_shapes

    expected = _target_shapes(prompt_len)
    actual_keys = set(weights)
    problems: list[str] = []
    missing = sorted(expected.keys() - actual_keys)
    unexpected = sorted(actual_keys - expected.keys())
    if missing:
        problems.append(f"missing {len(missing)} tensors (first: {missing[:5]})")
    if unexpected:
        problems.append(f"unexpected {len(unexpected)} tensors (first: {unexpected[:5]})")

    mismatched: list[str] = []
    invalid_storage: list[str] = []
    for name, shape in expected.items():
        value = weights.get(name)
        if not isinstance(value, torch.Tensor):
            if name in actual_keys:
                invalid_storage.append(f"{name}: expected torch.Tensor, got {type(value).__name__}")
            continue
        if tuple(value.shape) != shape:
            mismatched.append(f"{name}: got {tuple(value.shape)}, expected {shape}")
        if value.dtype != torch.bfloat16 or value.device.type != "cpu" or not value.is_contiguous():
            invalid_storage.append(
                f"{name}: expected contiguous CPU bfloat16, got {value.device}/{value.dtype}/"
                f"contiguous={value.is_contiguous()}"
            )
    if mismatched:
        problems.append(f"mismatched {len(mismatched)} shapes (first: {mismatched[:5]})")
    if invalid_storage:
        problems.append(f"invalid {len(invalid_storage)} tensor layouts (first: {invalid_storage[:5]})")
    if problems:
        raise ValueError("Invalid realtime-vla-v2 PI0.5 export: " + "; ".join(problems))


def load_exported_weights(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a trusted exporter pickle, then validate its complete tensor contract.

    Pickle can execute code while loading. Only pass files created locally by
    ``lerobot-export-pi05-realtime-vla-v2`` or whose SHA-256 was authenticated.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Exported PI0.5 Triton weights not found: {path}")
    with path.open("rb") as stream:
        value = pickle.load(stream)  # noqa: S301 - the trust boundary is explicit above.
    if not isinstance(value, dict):
        raise ValueError(f"Exported PI0.5 weights must contain a dict, got {type(value).__name__}")
    validate_exported_weights(value)
    return value


def digitize_normalized_state(state: np.ndarray | Sequence[float], state_dim: int = STATE_DIM) -> np.ndarray:
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.shape != (state_dim,):
        raise ValueError(f"normalized_state must have shape ({state_dim},), got {state_array.shape}")
    if not np.isfinite(state_array).all():
        raise ValueError("normalized_state contains NaN or infinity")
    bins = np.linspace(-1.0, 1.0, 257, dtype=np.float32)[:-1]
    return (np.digitize(state_array, bins=bins) - 1).astype(np.int32)


def normalize_task_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
    normalized = prompt.strip().replace("_", " ").replace("\n", " ")
    if not normalized:
        raise ValueError("prompt must be non-empty")
    return normalized


def pad_prefill_actions(
    actions: np.ndarray | Sequence[Sequence[float]] | None,
    *,
    action_dim: int = ACTION_DIM,
    padded_dim: int = PADDED_ACTION_DIM,
) -> tuple[np.ndarray | None, int]:
    if actions is None:
        return None, 0
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != action_dim:
        raise ValueError(f"normalized_prefill_actions must have shape [T,{action_dim}], got {value.shape}")
    if value.shape[0] >= ACTION_CHUNK_SIZE:
        raise ValueError(f"prefill length must be smaller than {ACTION_CHUNK_SIZE}")
    if not np.isfinite(value).all():
        raise ValueError("normalized_prefill_actions contains NaN or infinity")
    padded = np.zeros((value.shape[0], padded_dim), dtype=np.float32)
    padded[:, :action_dim] = value
    return padded, int(value.shape[0])


def make_fixed_diffusion_noise(
    seed: int, shape: tuple[int, int] = (ACTION_CHUNK_SIZE, PADDED_ACTION_DIM)
) -> np.ndarray:
    """Generate the NumPy PCG64 Gaussian input used by the upstream adapter."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("noise seed must be an integer")
    if shape != (ACTION_CHUNK_SIZE, PADDED_ACTION_DIM):
        raise ValueError(f"fixed PI0.5 noise shape must be {(ACTION_CHUNK_SIZE, PADDED_ACTION_DIM)}")
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _normalize_image(image: Any, value_range: str) -> torch.Tensor:
    value = torch.as_tensor(image)
    if value.ndim != 3:
        raise ValueError(f"camera image must be rank 3, got {tuple(value.shape)}")
    if value.shape[-1] == 3:
        value = value.permute(2, 0, 1)
    elif value.shape[0] != 3:
        raise ValueError(f"camera image must be HWC or CHW with 3 channels, got {tuple(value.shape)}")

    if value_range == "uint8":
        if value.dtype != torch.uint8:
            raise ValueError(f"image_value_range='uint8' requires uint8 images, got {value.dtype}")
        value = value.to(torch.float32).div_(255.0)
    else:
        if not value.is_floating_point():
            raise ValueError(f"image_value_range={value_range!r} requires floating-point images")
        value = value.to(torch.float32)
        minimum = float(value.min())
        maximum = float(value.max())
        if value_range == "zero_one":
            if minimum < -1e-5 or maximum > 1.0 + 1e-5:
                raise ValueError(f"zero_one image values must be in [0,1], got [{minimum},{maximum}]")
        else:
            if minimum < -1.0 - 1e-5 or maximum > 1.0 + 1e-5:
                raise ValueError(f"minus_one_one image values must be in [-1,1], got [{minimum},{maximum}]")
            value = value.add(1.0).div(2.0)
    return value.contiguous()


def prepare_camera_images(
    images: Mapping[str, Any],
    camera_keys: Sequence[str],
    *,
    value_range: str,
    image_size: tuple[int, int] = IMAGE_SIZE,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Map two named images to fixed-order NHWC BF16 input with aspect padding."""

    missing = [key for key in camera_keys if key not in images]
    if missing:
        raise KeyError(f"Missing PI0.5 camera images: {missing}")
    if len(camera_keys) != NUM_CAMERAS or len(set(camera_keys)) != NUM_CAMERAS:
        raise ValueError(f"camera_keys must contain {NUM_CAMERAS} unique names")

    from .modeling_pi05 import resize_with_pad_torch

    target_h, target_w = image_size
    prepared: list[torch.Tensor] = []
    for key in camera_keys:
        value = _normalize_image(images[key], value_range)
        resized = resize_with_pad_torch(value.unsqueeze(0), target_h, target_w).squeeze(0)
        prepared.append(resized.mul(2.0).sub(1.0).permute(1, 2, 0))
    return torch.stack(prepared).to(device=device, dtype=torch.bfloat16)


class PI05RealtimeVLATritonBackend:
    """Thread-safe LeRobot wrapper around one static-buffer CUDA graph runtime."""

    def __init__(
        self,
        runtime: PI05RTCTritonRuntime,
        config: PI05RealtimeVLATritonConfig,
        *,
        trained_prefix_max: int,
    ) -> None:
        if isinstance(trained_prefix_max, bool) or not isinstance(trained_prefix_max, int):
            raise ValueError("trained_prefix_max must be an integer")
        if not 0 < trained_prefix_max < config.chunk_size:
            raise ValueError(f"trained_prefix_max must be in [1, {config.chunk_size})")
        lengths = config.warmup_prefill_lengths
        if lengths is not None and any(length > trained_prefix_max for length in lengths):
            raise ValueError(f"warmup_prefill_lengths cannot exceed trained_prefix_max={trained_prefix_max}")
        self.runtime = runtime
        self.config = config
        self.trained_prefix_max = trained_prefix_max
        self._rng = np.random.default_rng(config.noise_seed)
        self._lock = threading.Lock()
        self.warmed_prefill_lengths: tuple[int, ...] = ()

    @classmethod
    def from_export(cls, config: PI05RealtimeVLATritonConfig) -> PI05RealtimeVLATritonBackend:
        """Validate artifacts and environment, then build and warm the CUDA graph."""

        if config.weights_path is None or config.model_config_path is None:
            raise ValueError("weights_path and model_config_path are required by from_export")
        weights_path = Path(config.weights_path).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Exported PI0.5 Triton weights not found: {weights_path}")
        model_config = load_and_validate_model_config(config.model_config_path)
        trained_prefix_max = int(model_config["rtc_training_max_delay"])
        supported_lengths = tuple(range(trained_prefix_max + 1))
        warmup_lengths = config.warmup_prefill_lengths or supported_lengths
        if warmup_lengths != supported_lengths:
            raise ValueError(
                "warmup_prefill_lengths must cover every trained prefix length exactly: "
                f"got {warmup_lengths}, expected {supported_lengths}"
            )

        validate_runtime_environment(config.device)
        require_cuda_free_memory(config.device, config.min_free_cuda_gib)
        try:
            from ._triton.pi05rtc_infer import Pi05RTCInference
        except ImportError as exc:
            raise PI05RealtimeVLATritonUnavailableError(
                f"Cannot import the vendored PI0.5 Triton runtime: {exc}"
            ) from exc

        if config.weights_sha256 is not None:
            actual_sha = file_sha256(weights_path)
            if actual_sha.lower() != config.weights_sha256.lower():
                raise ValueError(
                    f"PI0.5 Triton weight SHA-256 mismatch: got {actual_sha}, expected {config.weights_sha256}"
                )

        weights = load_exported_weights(weights_path)
        try:
            device_context = torch.cuda.device(torch.device(config.device))
            with device_context:
                runtime = Pi05RTCInference(
                    checkpoint=weights,
                    num_views=NUM_CAMERAS,
                    chunk_size=ACTION_CHUNK_SIZE,
                    tokenizer_path=config.tokenizer_path,
                    max_tokenize_len=config.tokenizer_max_length,
                    discrete_state_input=True,
                    max_prompt_len_override=config.prompt_capacity,
                    num_steps=EULER_STEPS,
                )
            runtime.checkpoint = None
        except Exception:
            del weights
            raise
        del weights
        backend = cls(runtime, config, trained_prefix_max=trained_prefix_max)
        backend.warmup(warmup_lengths)
        return backend

    def _build_noise(self, noise: np.ndarray | torch.Tensor | None) -> torch.Tensor:
        if noise is None:
            value = self._rng.standard_normal((ACTION_CHUNK_SIZE, PADDED_ACTION_DIM)).astype(np.float32)
        elif isinstance(noise, torch.Tensor):
            value = noise.detach()
            if tuple(value.shape) != (ACTION_CHUNK_SIZE, PADDED_ACTION_DIM):
                raise ValueError(
                    f"diffusion_noise must have shape {(ACTION_CHUNK_SIZE, PADDED_ACTION_DIM)}, got {tuple(value.shape)}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError("diffusion_noise contains NaN or infinity")
            return value.to(device=self.config.device, dtype=torch.bfloat16)
        else:
            value = np.asarray(noise, dtype=np.float32)
        if value.shape != (ACTION_CHUNK_SIZE, PADDED_ACTION_DIM):
            raise ValueError(
                f"diffusion_noise must have shape {(ACTION_CHUNK_SIZE, PADDED_ACTION_DIM)}, got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError("diffusion_noise contains NaN or infinity")
        return torch.as_tensor(value, device=self.config.device, dtype=torch.bfloat16)

    def warmup(self, prefill_lengths: Sequence[int] | None = None) -> None:
        """Exercise the captured graph for every supported dynamic prefix length."""

        if prefill_lengths is None:
            prefill_lengths = range(self.trained_prefix_max + 1)
        lengths = tuple(prefill_lengths)
        if tuple(sorted(set(lengths))) != lengths:
            raise ValueError("warmup prefix lengths must be sorted and unique")
        if any(length < 0 or length > self.trained_prefix_max for length in lengths):
            raise ValueError(f"warmup prefix lengths must be in [0, {self.trained_prefix_max}]")
        images = torch.zeros((NUM_CAMERAS, *IMAGE_SIZE, 3), dtype=torch.bfloat16, device=self.config.device)
        noise = torch.zeros(
            (ACTION_CHUNK_SIZE, PADDED_ACTION_DIM), dtype=torch.bfloat16, device=self.config.device
        )
        state_tokens = np.zeros((STATE_DIM,), dtype=np.int32)
        prefix = np.zeros((self.trained_prefix_max, PADDED_ACTION_DIM), dtype=np.float32)
        device_context = (
            torch.cuda.device(torch.device(self.config.device))
            if self.config.device.startswith("cuda")
            else nullcontext()
        )
        with self._lock, torch.no_grad(), device_context:
            for length in lengths:
                self.runtime.forward(
                    images,
                    noise,
                    task_prompt=normalize_task_prompt(self.config.prompt),
                    state_tokens=state_tokens,
                    action_prefill_len=length or None,
                    prefill_actions=prefix if length else None,
                )
            if self.config.device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(self.config.device))
        self.warmed_prefill_lengths = lengths

    def infer(
        self,
        *,
        images: Mapping[str, Any],
        normalized_state: np.ndarray | Sequence[float],
        normalized_prefill_actions: np.ndarray | Sequence[Sequence[float]] | None = None,
        noise: np.ndarray | torch.Tensor | None = None,
        prompt: str | None = None,
    ) -> PI05RealtimeVLATritonOutput:
        """Run the complete image encoder, VLM, expert, and trained-prefix denoiser."""

        state_tokens = digitize_normalized_state(normalized_state, self.config.state_dim)
        padded_prefix, prefill_len = pad_prefill_actions(
            normalized_prefill_actions,
            action_dim=self.config.action_dim,
            padded_dim=self.config.padded_action_dim,
        )
        if prefill_len > self.trained_prefix_max:
            raise ValueError(
                f"prefill length {prefill_len} exceeds checkpoint training capacity {self.trained_prefix_max}"
            )
        image_tensor = prepare_camera_images(
            images,
            self.config.camera_keys,
            value_range=self.config.image_value_range,
            image_size=self.config.image_size,
            device=self.config.device,
        )
        noise_tensor = self._build_noise(noise)
        task_prompt = normalize_task_prompt(prompt if prompt is not None else self.config.prompt)

        device_context = (
            torch.cuda.device(torch.device(self.config.device))
            if self.config.device.startswith("cuda")
            else nullcontext()
        )
        with self._lock, torch.no_grad(), device_context:
            result = self.runtime.forward(
                image_tensor,
                noise_tensor,
                task_prompt=task_prompt,
                state_tokens=state_tokens,
                action_prefill_len=prefill_len or None,
                prefill_actions=padded_prefix,
            )
            if not isinstance(result, torch.Tensor):
                raise PI05RealtimeVLATritonError(
                    f"Triton runtime returned {type(result).__name__}, expected torch.Tensor"
                )
            if tuple(result.shape) != (ACTION_CHUNK_SIZE, PADDED_ACTION_DIM):
                raise PI05RealtimeVLATritonError(
                    "Triton runtime returned shape "
                    f"{tuple(result.shape)}, expected {(ACTION_CHUNK_SIZE, PADDED_ACTION_DIM)}"
                )
            actions = (
                result[:, : self.config.action_dim]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
                .copy()
            )
        if not np.isfinite(actions).all():
            raise PI05RealtimeVLATritonError("Triton runtime returned NaN or infinity")
        if prefill_len and not np.allclose(
            actions[:prefill_len], padded_prefix[:prefill_len, : self.config.action_dim], atol=1e-2, rtol=1e-2
        ):
            max_error = float(
                np.max(np.abs(actions[:prefill_len] - padded_prefix[:prefill_len, : self.config.action_dim]))
            )
            raise PI05RealtimeVLATritonError(
                f"Trained-prefix invariant failed after CUDA graph replay (max_abs_error={max_error:.6g})"
            )
        return PI05RealtimeVLATritonOutput(actions=actions, prefill_length=prefill_len)


class PI05RealtimeVLATritonPolicyAdapter(torch.nn.Module):
    """Minimal policy surface consumed by ``RTCInferenceEngine``.

    The existing PI0.5 preprocessor owns quantile normalization. This adapter
    reads its normalized state and image tensors, forwards only the committed
    prefix selected by RTC, and never constructs ``PI05Policy`` or its PyTorch
    model weights.
    """

    def __init__(
        self,
        backend: PI05RealtimeVLATritonBackend,
        policy_config: Any,
        *,
        camera_keys: Sequence[str],
        task: str,
    ) -> None:
        super().__init__()
        if getattr(policy_config, "type", None) != "pi05":
            raise ValueError("The realtime-vla-v2 Triton adapter requires a pi05 policy config")
        if tuple(camera_keys) != tuple(backend.config.camera_keys):
            raise ValueError(
                "Policy adapter camera order must match the captured Triton backend: "
                f"adapter={tuple(camera_keys)}, backend={tuple(backend.config.camera_keys)}"
            )
        if int(getattr(policy_config, "rtc_training_max_delay", 0)) != backend.trained_prefix_max:
            raise ValueError(
                "Policy config and Triton backend disagree on trained-prefix capacity: "
                f"policy={getattr(policy_config, 'rtc_training_max_delay', None)}, "
                f"backend={backend.trained_prefix_max}"
            )
        self.backend = backend
        self.config = policy_config
        self.camera_keys = tuple(camera_keys)
        self.task = normalize_task_prompt(task)

    @property
    def type(self) -> str:
        return "pi05"

    def reset(self) -> None:
        """The captured runtime has no episode-scoped action queue."""

    @staticmethod
    def _single_batch_item(value: Any, *, name: str, trailing_shape: tuple[int, ...]) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.ndim == len(trailing_shape):
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != len(trailing_shape) + 1 or tensor.shape[0] != 1:
            raise ValueError(f"{name} must contain exactly one batch item, got {tuple(tensor.shape)}")
        if tuple(tensor.shape[-len(trailing_shape) :]) != trailing_shape:
            raise ValueError(f"{name} must end in shape {trailing_shape}, got {tuple(tensor.shape)}")
        return tensor[0]

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: Mapping[str, Any],
        *,
        inference_delay: int,
        prev_chunk_left_over: torch.Tensor | None,
        rtc_mode: str,
    ) -> torch.Tensor:
        if rtc_mode != "trained_prefix":
            raise ValueError(
                "The full realtime-vla-v2 Triton action backend only supports rtc_mode='trained_prefix'"
            )
        if isinstance(inference_delay, bool) or not isinstance(inference_delay, int):
            raise ValueError("inference_delay must be an integer")
        if not 0 <= inference_delay <= self.backend.trained_prefix_max:
            raise ValueError(
                f"inference_delay must be in [0,{self.backend.trained_prefix_max}], got {inference_delay}"
            )

        from lerobot.utils.constants import OBS_STATE

        if OBS_STATE not in batch:
            raise KeyError(f"Preprocessed PI0.5 batch is missing {OBS_STATE!r}")
        normalized_state = self._single_batch_item(
            batch[OBS_STATE], name=OBS_STATE, trailing_shape=(STATE_DIM,)
        )
        images: dict[str, torch.Tensor] = {}
        for key in self.camera_keys:
            if key not in batch:
                raise KeyError(f"Preprocessed PI0.5 batch is missing camera {key!r}")
            image = torch.as_tensor(batch[key])
            if image.ndim == 4 and image.shape[0] == 1:
                image = image[0]
            if image.ndim != 3:
                raise ValueError(
                    f"Preprocessed camera {key!r} must be rank 3 or [1,...], got {tuple(image.shape)}"
                )
            images[key] = image

        normalized_prefix = None
        if prev_chunk_left_over is not None:
            prefix = torch.as_tensor(prev_chunk_left_over)
            if prefix.ndim == 3 and prefix.shape[0] == 1:
                prefix = prefix[0]
            if prefix.ndim != 2 or prefix.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"RTC committed prefix must have shape [T,{ACTION_DIM}], got {tuple(prefix.shape)}"
                )
            if prefix.shape[0] < inference_delay:
                raise ValueError(
                    "RTC committed prefix is shorter than inference_delay: "
                    f"rows={prefix.shape[0]}, delay={inference_delay}"
                )
            if inference_delay:
                normalized_prefix = (
                    prefix[:inference_delay].detach().to(device="cpu", dtype=torch.float32).numpy()
                )

        output = self.backend.infer(
            images=images,
            normalized_state=normalized_state.detach().to(device="cpu", dtype=torch.float32).numpy(),
            normalized_prefill_actions=normalized_prefix,
            prompt=self.task,
        )
        expected_prefill = inference_delay if normalized_prefix is not None else 0
        if output.prefill_length != expected_prefill:
            raise PI05RealtimeVLATritonError(
                "Triton backend returned an unexpected committed-prefix length: "
                f"got={output.prefill_length}, expected={expected_prefill}"
            )
        return torch.from_numpy(output.actions).unsqueeze(0).to(device=normalized_state.device)


class PI05PolicyFixedNoiseReference:
    """Run a loaded LeRobot ``PI05Policy`` on the exact Triton request boundary.

    This adapter intentionally accepts a preloaded policy. The serial parity
    CLI runs it in a process separate from the Triton backend so both full
    models never need to reside on the GPU at the same time.
    """

    def __init__(
        self,
        policy: Any,
        config: PI05RealtimeVLATritonConfig,
        *,
        tokenizer: Any | None = None,
    ) -> None:
        model = getattr(policy, "model", None)
        if model is None or not callable(getattr(model, "sample_actions", None)):
            raise TypeError("policy must expose model.sample_actions")
        model_config = getattr(model, "config", None)
        if model_config is None:
            raise TypeError("policy.model must expose config")
        expected = {
            "chunk_size": ACTION_CHUNK_SIZE,
            "max_action_dim": PADDED_ACTION_DIM,
            "num_inference_steps": EULER_STEPS,
        }
        mismatches = [
            f"{name}={getattr(model_config, name, None)!r}, expected {value!r}"
            for name, value in expected.items()
            if getattr(model_config, name, None) != value
        ]
        if int(getattr(model_config, "rtc_training_max_delay", 0)) <= 0:
            mismatches.append("rtc_training_max_delay must be positive")
        if mismatches:
            raise ValueError("Incompatible PI05Policy reference: " + "; ".join(mismatches))
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.policy = policy
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.trained_prefix_max = int(model_config.rtc_training_max_delay)
        self._lock = threading.Lock()

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (StopIteration, TypeError):
            return torch.device(self.config.device)

    def infer(
        self,
        *,
        images: Mapping[str, Any],
        normalized_state: np.ndarray | Sequence[float],
        normalized_prefill_actions: np.ndarray | Sequence[Sequence[float]] | None = None,
        noise: np.ndarray | torch.Tensor,
        prompt: str | None = None,
    ) -> PI05RealtimeVLATritonOutput:
        digitize_normalized_state(normalized_state, self.config.state_dim)
        padded_prefix, prefill_len = pad_prefill_actions(
            normalized_prefill_actions,
            action_dim=self.config.action_dim,
            padded_dim=self.config.padded_action_dim,
        )
        if prefill_len > self.trained_prefix_max:
            raise ValueError(
                f"prefill length {prefill_len} exceeds checkpoint training capacity {self.trained_prefix_max}"
            )
        task_prompt = prompt if prompt is not None else self.config.prompt
        from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
        from lerobot.processor.tokenizer_processor import TokenizerProcessorStep
        from lerobot.types import TransitionKey
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            OBS_STATE,
        )

        transition = {
            TransitionKey.OBSERVATION: {
                OBS_STATE: torch.as_tensor(normalized_state, dtype=torch.float32).unsqueeze(0)
            },
            TransitionKey.COMPLEMENTARY_DATA: {"task": [task_prompt]},
        }
        transition = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=PADDED_ACTION_DIM)(transition)
        transition = TokenizerProcessorStep(
            tokenizer=self.tokenizer,
            max_length=self.config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            truncation=True,
        )(transition)
        encoded_observation = transition[TransitionKey.OBSERVATION]
        if encoded_observation is None:
            raise PI05RealtimeVLATritonError("PI0.5 tokenizer processor returned no observation")
        device = self._device()
        tokens = encoded_observation[OBS_LANGUAGE_TOKENS].to(device=device)
        masks = encoded_observation[OBS_LANGUAGE_ATTENTION_MASK].to(device=device, dtype=torch.bool)

        image_tensor = prepare_camera_images(
            images,
            self.config.camera_keys,
            value_range=self.config.image_value_range,
            image_size=self.config.image_size,
            device=device,
        )
        model_images = [image_tensor[index].permute(2, 0, 1).unsqueeze(0) for index in range(NUM_CAMERAS)]
        image_masks = [torch.ones(1, dtype=torch.bool, device=device) for _ in range(NUM_CAMERAS)]
        noise_tensor = torch.as_tensor(noise, dtype=torch.float32, device=device)
        if tuple(noise_tensor.shape) != (ACTION_CHUNK_SIZE, PADDED_ACTION_DIM):
            raise ValueError(
                f"diffusion_noise must have shape {(ACTION_CHUNK_SIZE, PADDED_ACTION_DIM)}, "
                f"got {tuple(noise_tensor.shape)}"
            )
        if not bool(torch.isfinite(noise_tensor).all()):
            raise ValueError("diffusion_noise contains NaN or infinity")
        prefix_tensor = (
            torch.as_tensor(padded_prefix[:, : self.config.action_dim], dtype=torch.float32, device=device)
            if prefill_len
            else None
        )

        previous_rtc = getattr(self.model.config, "rtc_config", None)
        device_context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with self._lock, torch.no_grad(), device_context:
            # The trained-prefix branch only needs the enabled flag. It does
            # not call the guidance RTC processor.
            self.model.config.rtc_config = SimpleNamespace(enabled=True)
            try:
                result = self.model.sample_actions(
                    model_images,
                    image_masks,
                    tokens,
                    masks,
                    noise=noise_tensor.unsqueeze(0),
                    num_steps=EULER_STEPS,
                    rtc_mode="trained_prefix",
                    inference_delay=prefill_len,
                    prev_chunk_left_over=prefix_tensor,
                )
            finally:
                self.model.config.rtc_config = previous_rtc
        if not isinstance(result, torch.Tensor) or tuple(result.shape) != (
            1,
            ACTION_CHUNK_SIZE,
            PADDED_ACTION_DIM,
        ):
            shape = tuple(result.shape) if isinstance(result, torch.Tensor) else type(result).__name__
            raise PI05RealtimeVLATritonError(
                f"PI05Policy reference returned {shape}, expected {(1, ACTION_CHUNK_SIZE, PADDED_ACTION_DIM)}"
            )
        actions = (
            result[0, :, : self.config.action_dim]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .copy()
        )
        return PI05RealtimeVLATritonOutput(actions=actions, prefill_length=prefill_len)


class FixedNoiseParityHarness:
    """Compare this backend and a reference implementation on identical noise."""

    def __init__(self, *, seed: int = 0, atol: float = 8e-2, rtol: float = 2e-2) -> None:
        if atol < 0 or rtol < 0:
            raise ValueError("parity tolerances must be non-negative")
        self.seed = seed
        self.atol = float(atol)
        self.rtol = float(rtol)

    def noise(self) -> np.ndarray:
        return make_fixed_diffusion_noise(self.seed)

    def compare(self, actual: np.ndarray, reference: np.ndarray) -> FixedNoiseParityReport:
        actual = np.asarray(actual, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        if actual.shape != reference.shape:
            raise ValueError(f"parity output shapes differ: {actual.shape} != {reference.shape}")
        if not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError("parity outputs contain NaN or infinity")
        error = np.abs(actual - reference)
        return FixedNoiseParityReport(
            passed=bool(np.allclose(actual, reference, atol=self.atol, rtol=self.rtol)),
            max_abs_error=float(error.max(initial=0.0)),
            mean_abs_error=float(error.mean()) if error.size else 0.0,
            atol=self.atol,
            rtol=self.rtol,
            shape=actual.shape,
        )

    def run(
        self,
        backend: PI05RealtimeVLATritonBackend,
        reference: Callable[[np.ndarray], np.ndarray],
        **request: Any,
    ) -> FixedNoiseParityReport:
        noise = self.noise()
        actual = backend.infer(noise=noise, **request).actions
        expected = np.asarray(reference(noise.copy()), dtype=np.float32)
        return self.compare(actual, expected)


__all__ = [
    "ACTION_CHUNK_SIZE",
    "ACTION_DIM",
    "EULER_STEPS",
    "FixedNoiseParityHarness",
    "FixedNoiseParityReport",
    "NUM_CAMERAS",
    "PADDED_ACTION_DIM",
    "PI05RTCTritonRuntime",
    "PI05PolicyFixedNoiseReference",
    "PI05RealtimeVLATritonBackend",
    "PI05RealtimeVLATritonConfig",
    "PI05RealtimeVLATritonEnvironment",
    "PI05RealtimeVLATritonError",
    "PI05RealtimeVLATritonOutput",
    "PI05RealtimeVLATritonPolicyAdapter",
    "PI05RealtimeVLATritonUnavailableError",
    "STATE_DIM",
    "digitize_normalized_state",
    "load_and_validate_model_config",
    "load_exported_weights",
    "make_fixed_diffusion_noise",
    "pad_prefill_actions",
    "prepare_camera_images",
    "require_cuda_free_memory",
    "validate_exported_weights",
    "validate_runtime_config",
    "validate_runtime_environment",
]
