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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from transformers import DynamicCache  # noqa: E402

import lerobot.policies.pi05.tensorrt_prefix as tensorrt_prefix  # noqa: E402
from lerobot.policies.pi05.tensorrt_prefix import (  # noqa: E402
    PI05PrefixCacheExport,
    PI05TensorRTError,
    compute_model_architecture_fingerprint,
    compute_prefix_weight_fingerprint,
    dynamic_cache_from_flat,
    engine_verification_path,
    file_sha256,
    flatten_dynamic_cache,
    prefix_cache_input_names,
    prefix_cache_output_names,
    validate_prefix_engine_verification,
)
from lerobot.scripts.lerobot_export_pi05_tensorrt import patch_onnx_for_tensorrt  # noqa: E402


def test_prefix_cache_tensor_names():
    assert prefix_cache_input_names(2) == [
        "image_0",
        "image_1",
        "img_mask_0",
        "img_mask_1",
        "tokens",
        "token_masks",
    ]
    assert prefix_cache_output_names(2) == [
        "prefix_pad_masks",
        "key_0",
        "value_0",
        "key_1",
        "value_1",
    ]


def test_dynamic_cache_flatten_roundtrip():
    original = DynamicCache()
    expected: list[torch.Tensor] = []
    for layer_idx in range(2):
        keys = torch.full((1, 1, 4, 3), float(layer_idx + 1))
        values = keys + 0.5
        original.update(keys, values, layer_idx)
        expected.extend((keys, values))

    flat = flatten_dynamic_cache(original)
    rebuilt = dynamic_cache_from_flat(flat)
    rebuilt_flat = flatten_dynamic_cache(rebuilt)

    assert len(rebuilt_flat) == len(expected)
    for actual, reference in zip(rebuilt_flat, expected, strict=True):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("flat_cache", [[], [torch.zeros(1)]])
def test_dynamic_cache_from_flat_rejects_invalid_input(flat_cache):
    with pytest.raises(ValueError, match="key/value"):
        dynamic_cache_from_flat(flat_cache)


class _FakePaliGemmaWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace()))
        )

    def forward(self, *, inputs_embeds, **_kwargs):
        prefix_embs = inputs_embeds[0]
        cache = DynamicCache()
        for layer_idx in range(2):
            keys = prefix_embs[..., :2].unsqueeze(1) + layer_idx
            cache.update(keys, keys + 0.5, layer_idx)
        return [prefix_embs, None], cache


class _FakePI05Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _FakePaliGemmaWithExpert()

    def embed_prefix(self, images, img_masks, tokens, token_masks):
        del tokens
        batch_size = token_masks.shape[0]
        image_tokens = [image.mean(dim=(-2, -1)).unsqueeze(1) for image in images]
        language_tokens = torch.ones(batch_size, token_masks.shape[1], 3)
        prefix_embs = torch.cat([*image_tokens, language_tokens], dim=1)
        prefix_pad_masks = torch.cat(
            [*[mask[:, None] for mask in img_masks], token_masks],
            dim=1,
        )
        prefix_att_masks = torch.zeros_like(prefix_pad_masks)
        return prefix_embs, prefix_pad_masks, prefix_att_masks


def test_prefix_cache_export_returns_explicit_kv_tensors():
    wrapper = PI05PrefixCacheExport(_FakePI05Model(), num_cameras=2)
    images = [torch.ones(1, 3, 4, 4), torch.full((1, 3, 4, 4), 2.0)]
    img_masks = [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)]
    tokens = torch.ones(1, 3, dtype=torch.int32)
    token_masks = torch.ones(1, 3, dtype=torch.bool)

    outputs = wrapper(*images, *img_masks, tokens, token_masks)

    assert len(outputs) == len(prefix_cache_output_names(2))
    assert outputs[0].shape == (1, 5)
    assert all(tensor.shape == (1, 1, 5, 2) for tensor in outputs[1:])


class _TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(2)])


class _TinyPaligemmaInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _TinyLanguageModel()


class _TinyPaligemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyPaligemmaInner()
        self.vision_tower = nn.Linear(3, 3)


class _TinyPaligemmaWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = _TinyPaligemma()


class _TinyPI05Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _TinyPaligemmaWithExpert()


def _tiny_policy():
    torch.manual_seed(0)
    return SimpleNamespace(
        model=_TinyPI05Model(),
        config=SimpleNamespace(
            type="pi05",
            image_features={"observation.images.top": None, "observation.images.wrist": None},
            image_resolution=(224, 224),
            tokenizer_max_length=48,
            chunk_size=50,
            max_action_dim=32,
        ),
    )


def test_prefix_weight_fingerprint_is_deterministic_and_weight_sensitive():
    policy_a = _tiny_policy()
    policy_b = _tiny_policy()
    fingerprint_a = compute_prefix_weight_fingerprint(policy_a.model)
    assert fingerprint_a == compute_prefix_weight_fingerprint(policy_a.model)
    assert fingerprint_a == compute_prefix_weight_fingerprint(policy_b.model)  # same seed, same weights

    with torch.no_grad():
        policy_b.model.paligemma_with_expert.paligemma.vision_tower.weight.add_(1e-3)
    assert fingerprint_a != compute_prefix_weight_fingerprint(policy_b.model)


def test_prefix_weight_fingerprint_requires_resident_prefix():
    model = _TinyPI05Model()
    model.paligemma_with_expert.paligemma = None
    with pytest.raises(PI05TensorRTError, match="release_torch_prefix_modules"):
        compute_prefix_weight_fingerprint(model)


def test_architecture_fingerprint_tracks_config_changes():
    policy = _tiny_policy()
    baseline = compute_model_architecture_fingerprint(policy)
    assert baseline == compute_model_architecture_fingerprint(_tiny_policy())
    policy.config.chunk_size = 25
    assert baseline != compute_model_architecture_fingerprint(policy)


_FAKE_ENVIRONMENT = {
    "tensorrt_version": "10.8.0",
    "cuda_version": "12.4",
    "gpu_name": "Fake RTX",
    "gpu_compute_capability": "8.9",
}


def _write_verified_engine(tmp_path, policy, **overrides):
    engine_path = tmp_path / "prefix_cache_bf16.plan"
    engine_path.write_bytes(b"fake-engine-bytes")
    report = {
        "passed": True,
        "engine_sha256": file_sha256(engine_path),
        "prefix_weight_fingerprint": compute_prefix_weight_fingerprint(policy.model),
        "model_architecture_fingerprint": compute_model_architecture_fingerprint(policy),
        "num_cameras": len(policy.config.image_features),
        **_FAKE_ENVIRONMENT,
    }
    report.update(overrides)
    import json

    engine_verification_path(engine_path).write_text(json.dumps(report), encoding="utf-8")
    return engine_path


@pytest.fixture
def fake_runtime_environment(monkeypatch):
    monkeypatch.setattr(tensorrt_prefix, "runtime_environment_info", lambda device: dict(_FAKE_ENVIRONMENT))


def test_validate_verification_happy_path(tmp_path, fake_runtime_environment):
    policy = _tiny_policy()
    engine_path = _write_verified_engine(tmp_path, policy)
    report = validate_prefix_engine_verification(engine_path, policy, "cuda")
    assert report["passed"] is True


def test_validate_verification_missing_report_fails_closed(tmp_path):
    policy = _tiny_policy()
    engine_path = tmp_path / "prefix_cache_bf16.plan"
    engine_path.write_bytes(b"fake-engine-bytes")
    with pytest.raises(PI05TensorRTError, match="no verification report"):
        validate_prefix_engine_verification(engine_path, policy, "cuda")


def test_validate_verification_rejects_legacy_report_schema(tmp_path):
    policy = _tiny_policy()
    engine_path = _write_verified_engine(tmp_path, policy)
    import json

    report_path = engine_verification_path(engine_path)
    legacy = json.loads(report_path.read_text(encoding="utf-8"))
    del legacy["prefix_weight_fingerprint"]
    del legacy["engine_sha256"]
    report_path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(PI05TensorRTError, match="lacks required fields"):
        validate_prefix_engine_verification(engine_path, policy, "cuda")


def test_validate_verification_rejects_failed_report(tmp_path):
    policy = _tiny_policy()
    engine_path = _write_verified_engine(tmp_path, policy, passed=False)
    with pytest.raises(PI05TensorRTError, match="failed verification"):
        validate_prefix_engine_verification(engine_path, policy, "cuda")


def test_validate_verification_detects_engine_tampering(tmp_path):
    policy = _tiny_policy()
    engine_path = _write_verified_engine(tmp_path, policy)
    engine_path.write_bytes(b"tampered-engine-bytes")
    with pytest.raises(PI05TensorRTError, match="changed since verification"):
        validate_prefix_engine_verification(engine_path, policy, "cuda")


def test_validate_verification_detects_weight_mismatch(tmp_path):
    policy = _tiny_policy()
    engine_path = _write_verified_engine(tmp_path, policy)
    with torch.no_grad():
        policy.model.paligemma_with_expert.paligemma.vision_tower.weight.add_(1.0)
    with pytest.raises(PI05TensorRTError, match="different prefix weights"):
        validate_prefix_engine_verification(engine_path, policy, "cuda")


def test_validate_verification_detects_architecture_mismatch(tmp_path):
    policy = _tiny_policy()
    engine_path = _write_verified_engine(tmp_path, policy)
    policy.config.chunk_size = 25
    with pytest.raises(PI05TensorRTError, match="different model architecture"):
        validate_prefix_engine_verification(engine_path, policy, "cuda")


def test_validate_verification_detects_camera_count_mismatch(tmp_path):
    policy = _tiny_policy()
    engine_path = _write_verified_engine(tmp_path, policy, num_cameras=3)
    with pytest.raises(PI05TensorRTError, match="cameras"):
        validate_prefix_engine_verification(engine_path, policy, "cuda")


def test_validate_verification_detects_environment_mismatch(tmp_path, fake_runtime_environment):
    policy = _tiny_policy()
    engine_path = _write_verified_engine(tmp_path, policy, tensorrt_version="9.0.0")
    with pytest.raises(PI05TensorRTError, match="TensorRT version mismatch"):
        validate_prefix_engine_verification(engine_path, policy, "cuda")

    engine_path = _write_verified_engine(tmp_path, policy, gpu_compute_capability="7.5")
    with pytest.raises(PI05TensorRTError, match="compute capability mismatch"):
        validate_prefix_engine_verification(engine_path, policy, "cuda")


def test_patch_onnx_cumsum_for_tensorrt_is_idempotent(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    axis = helper.make_tensor("axis", TensorProto.INT64, (), [0])
    graph = helper.make_graph(
        [helper.make_node("CumSum", ["mask", "axis"], ["result"], name="CumSum")],
        "bool_cumsum",
        [helper.make_tensor_value_info("mask", TensorProto.BOOL, (3,))],
        [helper.make_tensor_value_info("result", TensorProto.INT32, (3,))],
        [axis],
    )
    path = tmp_path / "cumsum.onnx"
    onnx.save_model(helper.make_model(graph), path)

    assert patch_onnx_for_tensorrt(path) == 1
    assert patch_onnx_for_tensorrt(path) == 0

    patched = onnx.load_model(path)
    assert [node.op_type for node in patched.graph.node] == ["Cast", "CumSum"]
    assert patched.graph.node[1].input[0] == patched.graph.node[0].output[0]
