#!/usr/bin/env python

"""Analyze one supervised PI0.5 T0a rollout log and emit a safety report."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TIMING_PATTERN = re.compile(
    r"RTC timing diagnostics:.*?latency_ms=(?P<latency>[0-9.]+).*?"
    r"actual_consumed_steps=(?P<actual>\d+) merge_skip=(?P<merge>\d+).*?"
    r"replan_interval_steps=(?P<interval>\d+) replan_hz=(?P<replan_hz>[0-9.]+).*?"
    r"merge_skip_clamped=(?P<clamped>[01])"
)
FATAL_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "fatal_error": re.compile(r"Fatal error|FATAL", re.IGNORECASE),
    "connection_error": re.compile(r"ConnectionError|PermissionError|Incorrect status packet"),
    "guided_window_stop": re.compile(r"outside the guided execution window"),
    "queue_failure": re.compile(r"queue underflow|merge skip exceeded", re.IGNORECASE),
    "core_dump": re.compile(r"core dumped|核心已转储", re.IGNORECASE),
}
TENSORRT_RUNTIME_TOKEN = "TensorRT PI0.5 prefix enabled"
ACTION_FILTER_RUNTIME_TOKEN = "Action output filter enabled"
STALL_GUARD_RUNTIME_TOKEN = "Stall/contact guard enabled"


def build_required_tokens(
    *, duration_s: float, prefix_backend: str, action_filter: bool, stall_guard: bool = False
) -> dict[str, str]:
    tokens = {
        "gate_verified": "RTC replay gate verified:",
        "policy_full_checkpoint": "policy_kind=old_full_005613",
        "timing_actual_consumed": "rtc_timing=actual_consumed",
        "prefix_backend": f"prefix_backend={prefix_backend}",
        "action_pytorch": "action_backend=pytorch",
        "queue_q45": "queue_threshold=45",
        "cadence_6hz": "nominal_replan_hz=6.000",
        "guided_window_enforced": "enforce_guided_execution_window=true",
        "policy_loaded": "Policy loaded: type=pi05, device=cuda",
        "robot_connected": "Robot connected:",
        # The rollout logs the duration with %.0f formatting.
        "duration_reached": f"Duration limit reached ({duration_s:.0f}s)",
        "rollout_finished": "Rollout finished",
    }
    if prefix_backend == "tensorrt":
        tokens["tensorrt_prefix_enabled"] = TENSORRT_RUNTIME_TOKEN
    if action_filter:
        tokens["action_filter_config"] = "action_filter_enabled=true"
        tokens["action_filter_runtime"] = ACTION_FILTER_RUNTIME_TOKEN
    if stall_guard:
        tokens["stall_guard_runtime"] = STALL_GUARD_RUNTIME_TOKEN
    return tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--rollout-exit-code", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-actual-consumed", type=int, default=5)
    parser.add_argument("--max-latency-ms", type=float, default=166.667)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--expect-prefix-backend", choices=("pytorch", "tensorrt"), default="pytorch")
    parser.add_argument("--expect-action-filter", choices=("true", "false"), default="false")
    parser.add_argument("--expect-stall-guard", choices=("true", "false"), default="false")
    parser.add_argument("--run-kind", default="supervised_real_robot_t0a")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_actual_consumed < 0:
        raise ValueError("max-actual-consumed must be non-negative")
    if not math.isfinite(args.max_latency_ms) or args.max_latency_ms <= 0:
        raise ValueError("max-latency-ms must be finite and positive")
    if not math.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("duration must be finite and positive")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[rank]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def analyze_log(
    text: str,
    *,
    rollout_exit_code: int,
    max_actual_consumed: int,
    max_latency_ms: float,
    duration_s: float = 5.0,
    prefix_backend: str = "pytorch",
    action_filter: bool = False,
    stall_guard: bool = False,
    run_kind: str = "supervised_real_robot_t0a",
) -> dict[str, Any]:
    required_tokens = build_required_tokens(
        duration_s=duration_s,
        prefix_backend=prefix_backend,
        action_filter=action_filter,
        stall_guard=stall_guard,
    )
    required = {name: {"passed": token in text, "token": token} for name, token in required_tokens.items()}
    fatal_matches = {name: len(pattern.findall(text)) for name, pattern in FATAL_PATTERNS.items()}
    timing = [
        {
            "latency_ms": float(match.group("latency")),
            "actual_consumed_steps": int(match.group("actual")),
            "merge_skip": int(match.group("merge")),
            "replan_interval_steps": int(match.group("interval")),
            "replan_hz": float(match.group("replan_hz")),
            "merge_skip_clamped": int(match.group("clamped")),
        }
        for match in TIMING_PATTERN.finditer(text)
    ]
    latencies = [entry["latency_ms"] for entry in timing]
    steady_timing = [entry for entry in timing if entry["replan_interval_steps"] > 0]
    steady_latencies = [entry["latency_ms"] for entry in steady_timing]
    steady_actual = [entry["actual_consumed_steps"] for entry in steady_timing]
    clamp_warning_count = text.count("Relative goal position magnitude had to be clamped to be safe")
    tensorrt_enabled_count = text.count(TENSORRT_RUNTIME_TOKEN)
    action_filter_runtime_count = text.count(ACTION_FILTER_RUNTIME_TOKEN)
    timing_checks = {
        "records_present": {"passed": bool(timing), "value": len(timing), "minimum": 1},
        "steady_records_present": {
            "passed": bool(steady_actual),
            "value": len(steady_actual),
            "minimum": 1,
        },
        "actual_consumed_max": {
            "passed": bool(steady_actual) and max(steady_actual) <= max_actual_consumed,
            "value": max(steady_actual) if steady_actual else None,
            "maximum": max_actual_consumed,
        },
        "merge_matches_actual": {
            "passed": bool(timing)
            and all(entry["merge_skip"] == entry["actual_consumed_steps"] for entry in timing),
            "mismatch_count": sum(entry["merge_skip"] != entry["actual_consumed_steps"] for entry in timing),
            "maximum": 0,
        },
        "merge_skip_clamped": {
            "passed": bool(timing) and all(entry["merge_skip_clamped"] == 0 for entry in timing),
            "value": sum(entry["merge_skip_clamped"] for entry in timing),
            "maximum": 0,
        },
        "latency_max_ms": {
            "passed": bool(steady_latencies) and max(steady_latencies) <= max_latency_ms,
            "value": max(steady_latencies) if steady_latencies else None,
            "maximum": max_latency_ms,
        },
    }
    hard_checks = {
        "rollout_exit_code": {
            "passed": rollout_exit_code == 0,
            "value": rollout_exit_code,
            "expected": 0,
        },
        "fatal_patterns": {
            "passed": not any(fatal_matches.values()),
            "matches": fatal_matches,
        },
        "safety_clamp_warnings": {
            "passed": clamp_warning_count == 0,
            "value": clamp_warning_count,
            "maximum": 0,
        },
    }
    if prefix_backend == "tensorrt":
        hard_checks["tensorrt_prefix_enabled"] = {
            "passed": tensorrt_enabled_count >= 1,
            "value": tensorrt_enabled_count,
            "minimum": 1,
        }
    else:
        hard_checks["tensorrt_not_enabled"] = {
            "passed": tensorrt_enabled_count == 0,
            "value": tensorrt_enabled_count,
            "maximum": 0,
        }
    if action_filter:
        hard_checks["action_filter_enabled"] = {
            "passed": action_filter_runtime_count >= 1,
            "value": action_filter_runtime_count,
            "minimum": 1,
        }
    else:
        hard_checks["action_filter_not_enabled"] = {
            "passed": action_filter_runtime_count == 0,
            "value": action_filter_runtime_count,
            "maximum": 0,
        }
    passed = (
        all(check["passed"] for check in required.values())
        and all(check["passed"] for check in timing_checks.values())
        and all(check["passed"] for check in hard_checks.values())
    )
    return {
        "schema_version": 1,
        "report_status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "run_kind": run_kind,
        "hardware_run": True,
        "expectations": {
            "duration_s": duration_s,
            "prefix_backend": prefix_backend,
            "action_filter": action_filter,
            "stall_guard": stall_guard,
            "max_actual_consumed": max_actual_consumed,
            "max_latency_ms": max_latency_ms,
        },
        "passed": passed,
        "required_log_contract": required,
        "timing_checks": timing_checks,
        "hard_checks": hard_checks,
        "timing_summary": {
            "record_count": len(timing),
            "steady_record_count": len(steady_timing),
            "latency_ms_p50": _percentile(steady_latencies, 0.50),
            "latency_ms_p95": _percentile(steady_latencies, 0.95),
            "latency_ms_max": max(steady_latencies) if steady_latencies else None,
            "initial_latency_ms": latencies[0] if latencies else None,
            "actual_consumed_histogram": {
                str(value): steady_actual.count(value) for value in sorted(set(steady_actual))
            },
            "replan_hz_p50": _percentile(
                [entry["replan_hz"] for entry in timing if entry["replan_hz"] > 0],
                0.50,
            ),
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    log_path = args.log.expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"rollout log not found: {log_path}")
    report = analyze_log(
        log_path.read_text(encoding="utf-8", errors="replace"),
        rollout_exit_code=args.rollout_exit_code,
        max_actual_consumed=args.max_actual_consumed,
        max_latency_ms=args.max_latency_ms,
        duration_s=args.duration,
        prefix_backend=args.expect_prefix_backend,
        action_filter=args.expect_action_filter == "true",
        stall_guard=args.expect_stall_guard == "true",
        run_kind=args.run_kind,
    )
    output_path = args.output_json.expanduser().resolve()
    report["log"] = str(log_path)
    _write_json(output_path, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "output_json": str(output_path),
                "log": str(log_path),
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
