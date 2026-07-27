#!/usr/bin/env python

from examples.inference.analyze_pi05_t0a_log import analyze_log

BASE_LOG = """
RTC replay gate verified: /tmp/gate.json
policy_kind=old_full_005613
rtc_timing=actual_consumed
prefix_backend=pytorch
action_backend=pytorch
queue_threshold=45
nominal_replan_hz=6.000
enforce_guided_execution_window=true
INFO Policy loaded: type=pi05, device=cuda
INFO Robot connected: so_follower
INFO RTC timing diagnostics: mode=actual_consumed latency_ms=140.000 guidance_delay_estimate=5 wall_latency_steps=5 actual_consumed_steps=0 merge_skip=0 generation=0->0->1 index=0->0->0 queue=0->0->50 measured_control_fps=0.00 replan_interval_steps=0 replan_hz=0.000 source_chunk_generation=1 next_model_action_index=0 merge_skip_clamped=0 warmup_state=steady
INFO RTC timing diagnostics: mode=actual_consumed latency_ms=145.000 guidance_delay_estimate=5 wall_latency_steps=5 actual_consumed_steps=5 merge_skip=5 generation=1->1->2 index=5->10->0 queue=45->40->45 measured_control_fps=34.48 replan_interval_steps=5 replan_hz=6.000 source_chunk_generation=2 next_model_action_index=5 merge_skip_clamped=0 warmup_state=steady
INFO Duration limit reached (5s)
INFO Rollout finished
"""


def test_valid_t0a_log_passes_hard_contract() -> None:
    report = analyze_log(
        BASE_LOG,
        rollout_exit_code=0,
        max_actual_consumed=5,
        max_latency_ms=166.667,
    )

    assert report["passed"] is True
    assert report["hardware_run"] is True
    assert report["timing_summary"]["record_count"] == 2
    assert report["timing_summary"]["actual_consumed_histogram"] == {"5": 1}
    assert report["timing_summary"]["latency_ms_p95"] == 145.0


def test_empty_queue_initial_latency_is_diagnostic_only() -> None:
    cold_start = BASE_LOG.replace("latency_ms=140.000", "latency_ms=900.000", 1)

    report = analyze_log(
        cold_start,
        rollout_exit_code=0,
        max_actual_consumed=5,
        max_latency_ms=166.667,
    )

    assert report["passed"] is True
    assert report["timing_summary"]["initial_latency_ms"] == 900.0
    assert report["timing_summary"]["latency_ms_max"] == 145.0


def test_tail_cutoff_fatal_and_tensorrt_are_all_rejected() -> None:
    log = (
        BASE_LOG
        + """
TensorRT PI0.5 prefix enabled: engine=/tmp/prefix.plan
Relative goal position magnitude had to be clamped to be safe
Fatal error in RTC thread: RTC refused to dispatch an action outside the guided execution window
Traceback (most recent call last):
"""
    )

    report = analyze_log(
        log,
        rollout_exit_code=1,
        max_actual_consumed=5,
        max_latency_ms=166.667,
    )

    assert report["passed"] is False
    assert report["hard_checks"]["rollout_exit_code"]["passed"] is False
    assert report["hard_checks"]["fatal_patterns"]["passed"] is False
    assert report["hard_checks"]["safety_clamp_warnings"]["passed"] is False
    assert report["hard_checks"]["tensorrt_not_enabled"]["passed"] is False


def test_actual_consumed_six_or_merge_mismatch_fails_timing_gate() -> None:
    unsafe = BASE_LOG.replace(
        "actual_consumed_steps=5 merge_skip=5",
        "actual_consumed_steps=6 merge_skip=4",
    )

    report = analyze_log(
        unsafe,
        rollout_exit_code=0,
        max_actual_consumed=5,
        max_latency_ms=166.667,
    )

    assert report["passed"] is False
    assert report["timing_checks"]["actual_consumed_max"]["passed"] is False
    assert report["timing_checks"]["merge_matches_actual"]["passed"] is False


def test_missing_runtime_completion_tokens_does_not_pass() -> None:
    report = analyze_log(
        "queue_threshold=45\n",
        rollout_exit_code=0,
        max_actual_consumed=5,
        max_latency_ms=166.667,
    )

    assert report["passed"] is False
    assert report["required_log_contract"]["rollout_finished"]["passed"] is False
    assert report["timing_checks"]["records_present"]["passed"] is False


T1_LOG = (
    BASE_LOG.replace("prefix_backend=pytorch", "prefix_backend=tensorrt").replace(
        "Duration limit reached (5s)", "Duration limit reached (45s)"
    )
    + """
action_filter_enabled=true
TensorRT PI0.5 prefix enabled: engine=/tmp/prefix_cache_bf16.plan, cameras=2, cache_layers=18
Action output filter enabled: joints=6, dt=0.033333
Stall/contact guard enabled: ticks=5, tolerance=0.0010
"""
)


def test_valid_t1_log_passes_under_t1_expectations() -> None:
    report = analyze_log(
        T1_LOG,
        rollout_exit_code=0,
        max_actual_consumed=5,
        max_latency_ms=166.667,
        duration_s=45.0,
        prefix_backend="tensorrt",
        action_filter=True,
        stall_guard=True,
        run_kind="supervised_real_robot_t1",
    )

    assert report["passed"] is True
    assert report["run_kind"] == "supervised_real_robot_t1"
    assert report["hard_checks"]["tensorrt_prefix_enabled"]["passed"] is True
    assert report["hard_checks"]["action_filter_enabled"]["passed"] is True
    assert report["required_log_contract"]["stall_guard_runtime"]["passed"] is True
    assert report["required_log_contract"]["duration_reached"]["token"] == "Duration limit reached (45s)"


def test_t1_log_is_rejected_under_t0a_expectations() -> None:
    report = analyze_log(
        T1_LOG,
        rollout_exit_code=0,
        max_actual_consumed=5,
        max_latency_ms=166.667,
    )

    assert report["passed"] is False
    assert report["required_log_contract"]["prefix_backend"]["passed"] is False
    assert report["hard_checks"]["tensorrt_not_enabled"]["passed"] is False
    assert report["hard_checks"]["action_filter_not_enabled"]["passed"] is False


def test_t1_expectations_reject_missing_tensorrt_or_filter_runtime() -> None:
    silent_fallback = BASE_LOG.replace("prefix_backend=pytorch", "prefix_backend=tensorrt").replace(
        "Duration limit reached (5s)", "Duration limit reached (45s)"
    )

    report = analyze_log(
        silent_fallback,
        rollout_exit_code=0,
        max_actual_consumed=5,
        max_latency_ms=166.667,
        duration_s=45.0,
        prefix_backend="tensorrt",
        action_filter=True,
        stall_guard=True,
    )

    assert report["passed"] is False
    assert report["hard_checks"]["tensorrt_prefix_enabled"]["passed"] is False
    assert report["hard_checks"]["action_filter_enabled"]["passed"] is False
    assert report["required_log_contract"]["tensorrt_prefix_enabled"]["passed"] is False
    assert report["required_log_contract"]["action_filter_runtime"]["passed"] is False
    assert report["required_log_contract"]["stall_guard_runtime"]["passed"] is False
