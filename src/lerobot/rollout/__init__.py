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

"""Policy deployment engine with pluggable rollout strategies."""

from lerobot.utils.import_utils import require_package

require_package("datasets", extra="dataset")

from .configs import (
    BaseStrategyConfig,
    DAggerKeyboardConfig,
    DAggerPedalConfig,
    DAggerStrategyConfig,
    EpisodicStrategyConfig,
    HighlightStrategyConfig,
    RealtimeTraceConfig,
    RolloutConfig,
    RolloutStrategyConfig,
    SentryStrategyConfig,
    SmoothExecutorConfig,
)
from .context import (
    DatasetContext,
    HardwareContext,
    PolicyContext,
    ProcessorContext,
    RolloutContext,
    RolloutPreflightResult,
    RuntimeContext,
    build_rollout_context,
)
from .inference import (
    InferenceEngine,
    InferenceEngineConfig,
    RTCExecutorControlSnapshot,
    RTCInferenceConfig,
    RTCInferenceEngine,
    RTCInferenceMode,
    SyncInferenceConfig,
    SyncInferenceEngine,
    create_inference_engine,
)
from .realtime_executor import ExecutorCommandPreview, RealtimeExecutor, RealtimeExecutorConfig
from .strategies import (
    BaseStrategy,
    DAggerStrategy,
    EpisodicStrategy,
    HighlightStrategy,
    RolloutStrategy,
    SentryStrategy,
    create_strategy,
)
from .time_axis import TimeAxisPlan, TimeAxisPlanner, TimeAxisPlannerConfig
from .trajectory import DelayAlignedTrajectory, RealtimeTraceWriter

__all__ = [
    "BaseStrategy",
    "BaseStrategyConfig",
    "DAggerKeyboardConfig",
    "DAggerPedalConfig",
    "DAggerStrategy",
    "DAggerStrategyConfig",
    "DatasetContext",
    "HardwareContext",
    "HighlightStrategy",
    "HighlightStrategyConfig",
    "EpisodicStrategy",
    "EpisodicStrategyConfig",
    "InferenceEngine",
    "InferenceEngineConfig",
    "PolicyContext",
    "ProcessorContext",
    "RTCInferenceConfig",
    "RTCExecutorControlSnapshot",
    "RTCInferenceEngine",
    "RTCInferenceMode",
    "RolloutConfig",
    "RolloutContext",
    "RolloutPreflightResult",
    "RolloutStrategy",
    "RolloutStrategyConfig",
    "RealtimeTraceConfig",
    "RealtimeExecutor",
    "RealtimeExecutorConfig",
    "ExecutorCommandPreview",
    "RuntimeContext",
    "SentryStrategy",
    "SentryStrategyConfig",
    "SmoothExecutorConfig",
    "SyncInferenceConfig",
    "SyncInferenceEngine",
    "build_rollout_context",
    "create_inference_engine",
    "create_strategy",
    "DelayAlignedTrajectory",
    "RealtimeTraceWriter",
    "TimeAxisPlan",
    "TimeAxisPlanner",
    "TimeAxisPlannerConfig",
]
