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

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.policies.pi05.realtime_vla_v2_triton import (
    FixedNoiseParityHarness,
    PI05PolicyFixedNoiseReference,
    PI05RealtimeVLATritonBackend,
    PI05RealtimeVLATritonConfig,
    PI05RealtimeVLATritonError,
    PI05RealtimeVLATritonPolicyAdapter,
    PI05RealtimeVLATritonUnavailableError,
    digitize_normalized_state,
    make_fixed_diffusion_noise,
    normalize_task_prompt,
    pad_prefill_actions,
    prepare_camera_images,
    require_cuda_free_memory,
    validate_exported_weights,
)
from lerobot.scripts.lerobot_pi05_realtime_vla_v2_parity import (
    ParityRequest,
    load_request,
    load_result,
    main as parity_main,
    request_fingerprint,
    save_request,
    save_result,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE


def _config(**overrides) -> PI05RealtimeVLATritonConfig:
    values = {
        "prompt": "pick_up_cube",
        "tokenizer_path": "unused-tokenizer",
        "camera_keys": ("top", "wrist"),
        "device": "cpu",
        "image_value_range": "zero_one",
        "min_free_cuda_gib": 0.0,
    }
    values.update(overrides)
    return PI05RealtimeVLATritonConfig(**values)


class _FakeRuntime:
    def __init__(self, *, corrupt_prefix: bool = False) -> None:
        self.calls: list[dict] = []
        self.corrupt_prefix = corrupt_prefix

    def forward(
        self,
        observation_images_normalized,
        diffusion_noise,
        task_prompt=None,
        state_tokens=None,
        action_prefill_len=None,
        prefill_actions=None,
    ):
        length = int(action_prefill_len or 0)
        result = diffusion_noise.clone()
        if length:
            prefix = torch.as_tensor(prefill_actions[:length], device=result.device, dtype=result.dtype)
            result[:length].copy_(prefix)
            if self.corrupt_prefix:
                result[0, 0] += 1
        self.calls.append(
            {
                "images": observation_images_normalized.clone(),
                "noise": diffusion_noise.clone(),
                "prompt": task_prompt,
                "state_tokens": np.asarray(state_tokens).copy(),
                "prefill_len": length,
                "prefill_actions": None if prefill_actions is None else np.asarray(prefill_actions).copy(),
            }
        )
        return result


class _FakeTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text, **kwargs):
        self.texts.extend(text)
        length = int(kwargs["max_length"])
        input_ids = torch.zeros((len(text), length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        input_ids[:, :3] = torch.tensor([1, 2, 3])
        attention_mask[:, :3] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakePolicyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            chunk_size=50,
            max_action_dim=32,
            num_inference_steps=10,
            rtc_training_max_delay=6,
            rtc_config=None,
        )
        self.last_call = None

    def sample_actions(self, images, image_masks, tokens, masks, noise, **kwargs):
        assert self.config.rtc_config.enabled
        result = noise.clone()
        length = int(kwargs["inference_delay"])
        prefix = kwargs["prev_chunk_left_over"]
        if length:
            if prefix.ndim == 2:
                prefix = prefix.unsqueeze(0)
            result[:, :length, :6] = prefix[:, :length]
        self.last_call = {
            "images": images,
            "image_masks": image_masks,
            "tokens": tokens,
            "masks": masks,
            "kwargs": kwargs,
        }
        return result


class _FakePolicy:
    def __init__(self) -> None:
        self.model = _FakePolicyModel()


def _images() -> dict[str, np.ndarray]:
    return {
        "top": np.zeros((3, 224, 224), dtype=np.float32),
        "wrist": np.ones((224, 224, 3), dtype=np.float32),
    }


def test_default_runtime_import_does_not_import_vendored_triton_kernels() -> None:
    code = (
        "import sys; import lerobot.policies.pi05.realtime_vla_v2_triton; "
        "assert 'lerobot.policies.pi05._triton.pi05rtc_infer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_vendored_rtc_decoder_rope_starts_after_encoder_prefix() -> None:
    pytest.importorskip("triton")
    from lerobot.policies.pi05._triton.pi05rtc_infer import Pi05RTCInference

    runtime = object.__new__(Pi05RTCInference)
    runtime.num_views = 2
    runtime.chunk_size = 3
    runtime._rope_table = torch.arange(600).unsqueeze(1)

    weights = runtime.get_decoder_rope_weights(prompt_len=4)

    torch.testing.assert_close(weights, torch.tensor([[516], [517], [518]]))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"camera_keys": ("top", "top")}, "2 unique"),
        ({"action_dim": 14}, "action_dim"),
        ({"chunk_size": 49}, "chunk_size"),
        ({"num_inference_steps": 5}, "num_inference_steps"),
        ({"warmup_prefill_lengths": (0, 2, 1)}, "sorted and unique"),
    ],
)
def test_runtime_config_rejects_incompatible_shapes(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**override)


def test_state_digitization_matches_existing_pi05_prompt_processor_at_boundaries() -> None:
    state = np.array([-1.0, -0.5, 0.0, 0.5, np.nextafter(1.0, 0.0), 1.0], dtype=np.float32)
    tokens = digitize_normalized_state(state)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.from_numpy(state).unsqueeze(0)},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick_up\nblock"]},
    }

    result = Pi05PrepareStateTokenizerProcessorStep()(transition)
    prompt = result[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert prompt == f"Task: pick up block, State: {' '.join(map(str, tokens.tolist()))};\nAction: "
    assert tokens.tolist() == [0, 64, 128, 192, 255, 255]


def test_state_digitization_matches_pi05_for_quantile_outliers() -> None:
    state = np.array([-1.01, -1.0, 0.0, 1.0, 1.01, 0.5], dtype=np.float32)
    tokens = digitize_normalized_state(state)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.from_numpy(state).unsqueeze(0)},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick"]},
    }

    result = Pi05PrepareStateTokenizerProcessorStep()(transition)
    prompt = result[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert tokens.tolist() == [-1, 0, 128, 255, 255, 192]
    assert prompt == f"Task: pick, State: {' '.join(map(str, tokens.tolist()))};\nAction: "


def test_task_normalization_matches_pi05_processor() -> None:
    assert normalize_task_prompt("  pick_up\nblock  ") == "pick up block"


def test_camera_mapping_matches_existing_resize_pad_and_preserves_order() -> None:
    top = torch.linspace(0, 1, 3 * 120 * 320, dtype=torch.float32).reshape(3, 120, 320)
    wrist = torch.ones(240, 100, 3, dtype=torch.float32)

    actual = prepare_camera_images(
        {"wrist": wrist, "top": top},
        ("top", "wrist"),
        value_range="zero_one",
        device="cpu",
    )
    expected_top = resize_with_pad_torch(top.unsqueeze(0), 224, 224).squeeze(0)
    expected_wrist = resize_with_pad_torch(wrist.permute(2, 0, 1).unsqueeze(0), 224, 224).squeeze(0)
    expected = torch.stack((expected_top, expected_wrist)).mul(2).sub(1).permute(0, 2, 3, 1)

    torch.testing.assert_close(actual.float(), expected.to(torch.bfloat16).float(), rtol=0, atol=0)
    assert actual.shape == (2, 224, 224, 3)


def test_prefill_padding_is_six_to_thirty_two_without_robot_specific_rules() -> None:
    actions = np.arange(18, dtype=np.float32).reshape(3, 6)
    padded, length = pad_prefill_actions(actions)

    assert length == 3
    np.testing.assert_array_equal(padded[:, :6], actions)
    np.testing.assert_array_equal(padded[:, 6:], np.zeros((3, 26), dtype=np.float32))


def test_backend_interface_maps_two_cameras_six_actions_and_preserves_prefix() -> None:
    runtime = _FakeRuntime()
    backend = PI05RealtimeVLATritonBackend(runtime, _config(), trained_prefix_max=6)
    prefix = np.full((2, 6), 0.25, dtype=np.float32)
    noise = make_fixed_diffusion_noise(17)

    output = backend.infer(
        images=_images(),
        normalized_state=np.zeros(6, dtype=np.float32),
        normalized_prefill_actions=prefix,
        noise=noise,
        prompt=" pick_up\ncube ",
    )

    assert output.actions.shape == (50, 6)
    assert output.prefill_length == 2
    assert output.postfix_actions.shape == (48, 6)
    np.testing.assert_allclose(output.actions[:2], prefix, atol=1e-2, rtol=1e-2)
    call = runtime.calls[0]
    assert call["images"].shape == (2, 224, 224, 3)
    assert call["prompt"] == "pick up cube"
    assert call["state_tokens"].shape == (6,)
    assert call["prefill_actions"].shape == (2, 32)


def test_lightweight_policy_adapter_consumes_preprocessed_batch_and_rtc_prefix() -> None:
    runtime = _FakeRuntime()
    backend = PI05RealtimeVLATritonBackend(runtime, _config(), trained_prefix_max=6)
    adapter = PI05RealtimeVLATritonPolicyAdapter(
        backend,
        SimpleNamespace(type="pi05", rtc_training_max_delay=6),
        camera_keys=("top", "wrist"),
        task="pick_up_cube",
    )
    committed = torch.arange(20 * 6, dtype=torch.float32).reshape(20, 6) / 100
    batch = {
        "observation.state": torch.zeros(1, 6),
        "top": torch.zeros(1, 3, 224, 224),
        "wrist": torch.ones(1, 224, 224, 3),
    }

    actions = adapter.predict_action_chunk(
        batch,
        inference_delay=3,
        prev_chunk_left_over=committed,
        rtc_mode="trained_prefix",
    )

    assert actions.shape == (1, 50, 6)
    assert actions.device.type == "cpu"
    np.testing.assert_allclose(actions[0, :3].numpy(), committed[:3].numpy(), atol=1e-2, rtol=1e-2)
    assert runtime.calls[0]["prefill_len"] == 3
    assert runtime.calls[0]["prefill_actions"].shape == (3, 32)


def test_lightweight_policy_adapter_allows_first_inference_without_committed_prefix() -> None:
    runtime = _FakeRuntime()
    backend = PI05RealtimeVLATritonBackend(runtime, _config(), trained_prefix_max=6)
    adapter = PI05RealtimeVLATritonPolicyAdapter(
        backend,
        SimpleNamespace(type="pi05", rtc_training_max_delay=6),
        camera_keys=("top", "wrist"),
        task="pick",
    )
    batch = {
        "observation.state": torch.zeros(1, 6),
        "top": torch.zeros(1, 3, 224, 224),
        "wrist": torch.ones(1, 3, 224, 224),
    }

    actions = adapter.predict_action_chunk(
        batch,
        inference_delay=5,
        prev_chunk_left_over=None,
        rtc_mode="trained_prefix",
    )

    assert actions.shape == (1, 50, 6)
    assert runtime.calls[0]["prefill_len"] == 0
    with pytest.raises(ValueError, match="only supports rtc_mode='trained_prefix'"):
        adapter.predict_action_chunk(
            batch,
            inference_delay=0,
            prev_chunk_left_over=None,
            rtc_mode="guided",
        )


def test_backend_fails_closed_when_runtime_breaks_hard_prefix_invariant() -> None:
    backend = PI05RealtimeVLATritonBackend(_FakeRuntime(corrupt_prefix=True), _config(), trained_prefix_max=6)
    with pytest.raises(PI05RealtimeVLATritonError, match="Trained-prefix invariant"):
        backend.infer(
            images=_images(),
            normalized_state=np.zeros(6, dtype=np.float32),
            normalized_prefill_actions=np.zeros((1, 6), dtype=np.float32),
            noise=make_fixed_diffusion_noise(0),
        )


def test_warmup_exercises_each_dynamic_prefill_length() -> None:
    runtime = _FakeRuntime()
    backend = PI05RealtimeVLATritonBackend(runtime, _config(), trained_prefix_max=3)

    backend.warmup((0, 1, 2, 3))

    assert backend.warmed_prefill_lengths == (0, 1, 2, 3)
    assert [call["prefill_len"] for call in runtime.calls] == [0, 1, 2, 3]


def test_fixed_noise_harness_is_reproducible_and_reports_error() -> None:
    first = make_fixed_diffusion_noise(123)
    second = make_fixed_diffusion_noise(123)
    np.testing.assert_array_equal(first, second)
    harness = FixedNoiseParityHarness(seed=123, atol=0.01, rtol=0)

    report = harness.compare(first[:, :6], first[:, :6] + 0.005)

    assert report.passed
    assert report.shape == (50, 6)
    assert report.max_abs_error == pytest.approx(0.005, abs=1e-6)


def test_weight_validation_checks_complete_shape_dtype_and_layout(monkeypatch) -> None:
    import lerobot.scripts.lerobot_export_pi05_realtime_vla_v2 as exporter

    monkeypatch.setattr(
        exporter,
        "_target_shapes",
        lambda prompt_len: {"language_embeds": (prompt_len, 3), "weight": (2, 4)},
    )
    valid = {
        "language_embeds": torch.zeros(2, 3, dtype=torch.bfloat16),
        "weight": torch.zeros(2, 4, dtype=torch.bfloat16),
    }
    validate_exported_weights(valid)

    invalid = dict(valid)
    invalid["weight"] = torch.zeros(4, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="mismatched.+invalid"):
        validate_exported_weights(invalid)


def test_non_cuda_memory_preflight_fails_closed() -> None:
    with pytest.raises(PI05RealtimeVLATritonUnavailableError, match="requires a CUDA device"):
        require_cuda_free_memory("cpu", 0)


def test_real_policy_reference_adapter_uses_existing_prompt_tokenizer_and_resize_paths() -> None:
    tokenizer = _FakeTokenizer()
    policy = _FakePolicy()
    reference = PI05PolicyFixedNoiseReference(policy, _config(), tokenizer=tokenizer)
    prefix = np.full((2, 6), -0.25, dtype=np.float32)
    noise = make_fixed_diffusion_noise(7)

    output = reference.infer(
        images=_images(),
        normalized_state=np.zeros(6, dtype=np.float32),
        normalized_prefill_actions=prefix,
        noise=noise,
        prompt=" pick_up\ncube ",
    )

    expected_prompt = "Task: pick up cube, State: 128 128 128 128 128 128;\nAction: "
    assert tokenizer.texts == [expected_prompt]
    assert output.prefill_length == 2
    np.testing.assert_array_equal(output.actions[:2], prefix)
    call = policy.model.last_call
    assert len(call["images"]) == 2
    assert call["images"][0].shape == (1, 3, 224, 224)
    assert call["tokens"].shape == (1, 200)
    assert call["masks"].dtype == torch.bool
    assert call["kwargs"]["rtc_mode"] == "trained_prefix"
    assert policy.model.config.rtc_config is None


def test_serial_parity_artifacts_round_trip_and_compare_on_cpu(tmp_path, capsys) -> None:
    images = _images()
    state = np.zeros(6, dtype=np.float32)
    prefix = np.zeros((2, 6), dtype=np.float32)
    fingerprint = request_fingerprint(
        images=images,
        normalized_state=state,
        normalized_prefill_actions=prefix,
        prompt="pick",
        camera_keys=("top", "wrist"),
        image_value_range="zero_one",
    )
    request = ParityRequest(
        images=images,
        normalized_state=state,
        normalized_prefill_actions=prefix,
        prompt="pick",
        camera_keys=("top", "wrist"),
        image_value_range="zero_one",
        fingerprint=fingerprint,
    )
    request_path = tmp_path / "request.npz"
    save_request(request_path, request)
    loaded_request = load_request(request_path)
    assert loaded_request.fingerprint == fingerprint

    noise = make_fixed_diffusion_noise(4)
    output = PI05RealtimeVLATritonBackend(_FakeRuntime(), _config(), trained_prefix_max=6).infer(
        images=images,
        normalized_state=state,
        normalized_prefill_actions=prefix,
        noise=noise,
    )
    pytorch_path = tmp_path / "pytorch.npz"
    triton_path = tmp_path / "triton.npz"
    save_result(
        pytorch_path,
        output,
        noise,
        implementation="lerobot-pytorch",
        request=request,
        seed=4,
    )
    save_result(
        triton_path,
        output,
        noise,
        implementation="realtime-vla-v2-triton",
        request=request,
        seed=4,
    )

    assert load_result(pytorch_path).actions.shape == (50, 6)
    assert (
        parity_main(
            [
                "compare",
                "--pytorch-output",
                str(pytorch_path),
                "--triton-output",
                str(triton_path),
            ]
        )
        == 0
    )
    assert '"passed": true' in capsys.readouterr().out
