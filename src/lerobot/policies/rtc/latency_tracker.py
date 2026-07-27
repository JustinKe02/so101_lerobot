#!/usr/bin/env python

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

"""Latency tracking utilities for Real-Time Chunking (RTC)."""

import math
from collections import deque

import numpy as np


class LatencyTracker:
    """Tracks recent latencies and provides max/percentile queries.

    Args:
        maxlen (int | None): Optional sliding window size. If provided, only the
            most recent ``maxlen`` latencies are kept. If ``None``, keeps all.
    """

    def __init__(self, maxlen: int = 100):
        self._values = deque(maxlen=maxlen)
        self.reset()

    def reset(self) -> None:
        """Clear all recorded latencies."""
        self._values.clear()
        self.max_latency = 0.0

    def add(self, latency: float) -> None:
        """Add a latency sample (seconds)."""
        # Ensure numeric and non-negative
        val = float(latency)

        if val < 0:
            return
        self._values.append(val)
        self.max_latency = max(self.max_latency, val)

    def __len__(self) -> int:
        return len(self._values)

    def max(self) -> float | None:
        """Return the maximum latency or None if empty."""
        return self.max_latency

    def percentile(self, q: float) -> float | None:
        """Return the q-quantile (q in [0,1]) of recorded latencies or None if empty."""
        if not self._values:
            return 0.0
        q = float(q)
        if q <= 0.0:
            return min(self._values)
        if q >= 1.0:
            # Percentiles describe the active sliding window. ``max()`` keeps
            # its legacy lifetime-maximum behavior for the legacy RTC path.
            return max(self._values)
        vals = np.array(list(self._values), dtype=np.float32)
        return float(np.quantile(vals, q))

    def p95(self) -> float | None:
        """Return the 95th percentile latency or None if empty."""
        return self.percentile(0.95)


class GuidanceDelayEstimator:
    """Estimate an integer RTC guidance delay from inference latency.

    ``fixed`` mode always returns ``fixed_delay_steps``. ``rolling_p95``
    excludes an initial warmup, then derives a delay from a configurable
    latency percentile over a bounded window. Delay bucket changes must clear
    the configured hysteresis and remain stable for a number of observations.

    The caller reads :meth:`estimate` before inference and calls
    :meth:`observe` with the completed inference latency. Any horizon-specific
    limit remains the caller's responsibility.
    """

    FIXED = "fixed"
    ROLLING_P95 = "rolling_p95"
    _SUPPORTED_MODES = frozenset((FIXED, ROLLING_P95))

    def __init__(
        self,
        *,
        fps: float,
        mode: str = FIXED,
        fixed_delay_steps: int = 5,
        warmup_inferences: int = 5,
        window_size: int = 32,
        min_samples: int = 5,
        percentile: float = 0.95,
        hysteresis_steps: float = 0.25,
        change_confirmations: int = 3,
    ) -> None:
        mode = getattr(mode, "value", mode)
        self.mode = str(mode)
        self.fps = float(fps)
        self.fixed_delay_steps = int(fixed_delay_steps)
        self.warmup_inferences = int(warmup_inferences)
        self.window_size = int(window_size)
        self.min_samples = int(min_samples)
        self.percentile = float(percentile)
        self.hysteresis_steps = float(hysteresis_steps)
        self.change_confirmations = int(change_confirmations)

        self._validate()
        self._latencies = LatencyTracker(maxlen=self.window_size)
        self.reset()

    def _validate(self) -> None:
        if self.mode not in self._SUPPORTED_MODES:
            supported = ", ".join(sorted(self._SUPPORTED_MODES))
            raise ValueError(f"Unsupported guidance delay mode {self.mode!r}; expected one of: {supported}")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        if self.fixed_delay_steps < 0:
            raise ValueError("fixed_delay_steps must be non-negative")
        if self.warmup_inferences < 0:
            raise ValueError("warmup_inferences must be non-negative")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.min_samples < 1 or self.min_samples > self.window_size:
            raise ValueError("min_samples must be in [1, window_size]")
        if not 0.0 < self.percentile <= 1.0:
            raise ValueError("percentile must be in (0, 1]")
        if not math.isfinite(self.hysteresis_steps) or self.hysteresis_steps < 0.0:
            raise ValueError("hysteresis_steps must be finite and non-negative")
        if self.change_confirmations < 1:
            raise ValueError("change_confirmations must be positive")

    def reset(self) -> None:
        """Reset warmup, latency samples, and the active delay bucket."""
        self._latencies.reset()
        self._warmup_seen = 0
        self._current_delay_steps = self.fixed_delay_steps
        self._pending_delay_steps: int | None = None
        self._pending_confirmations = 0
        self._last_percentile_steps: float | None = None

    def estimate(self) -> int:
        """Return the guidance delay to use for the next inference."""
        return self._current_delay_steps

    @property
    def sample_count(self) -> int:
        """Number of post-warmup latency samples currently in the window."""
        return len(self._latencies)

    @property
    def warmup_remaining(self) -> int:
        """Number of rolling-mode observations still excluded as warmup."""
        if self.mode == self.FIXED:
            return 0
        return max(0, self.warmup_inferences - self._warmup_seen)

    @property
    def last_percentile_steps(self) -> float | None:
        """Most recently evaluated latency percentile expressed in control steps."""
        return self._last_percentile_steps

    def observe(self, latency: float) -> int:
        """Observe one completed inference latency and return the active delay."""
        value = float(latency)
        if not math.isfinite(value) or value < 0.0 or self.mode == self.FIXED:
            return self.estimate()

        if self._warmup_seen < self.warmup_inferences:
            self._warmup_seen += 1
            return self.estimate()

        self._latencies.add(value)
        if self.sample_count < self.min_samples:
            return self.estimate()

        percentile_latency = self._latencies.percentile(self.percentile)
        percentile_steps = max(0.0, float(percentile_latency) * self.fps)
        self._last_percentile_steps = percentile_steps
        candidate = int(math.ceil(percentile_steps))
        self._consider_candidate(candidate, percentile_steps)
        return self.estimate()

    def _consider_candidate(self, candidate: int, percentile_steps: float) -> None:
        if candidate == self._current_delay_steps:
            self._clear_pending()
            return

        if candidate > self._current_delay_steps:
            bucket_boundary = float(self._current_delay_steps)
            clears_hysteresis = percentile_steps >= bucket_boundary + self.hysteresis_steps
        else:
            bucket_boundary = float(self._current_delay_steps - 1)
            clears_hysteresis = percentile_steps <= bucket_boundary - self.hysteresis_steps

        if not clears_hysteresis:
            self._clear_pending()
            return

        if self._pending_delay_steps == candidate:
            self._pending_confirmations += 1
        else:
            self._pending_delay_steps = candidate
            self._pending_confirmations = 1

        if self._pending_confirmations >= self.change_confirmations:
            self._current_delay_steps = candidate
            self._clear_pending()

    def _clear_pending(self) -> None:
        self._pending_delay_steps = None
        self._pending_confirmations = 0
