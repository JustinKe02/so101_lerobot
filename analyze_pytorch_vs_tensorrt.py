#!/usr/bin/env python3
"""
PyTorch vs TensorRT 完整对比分析
分析 5秒和 30秒运行的性能差异
"""

import re
import json
from pathlib import Path
from statistics import mean, median, stdev
from dataclasses import dataclass
from typing import List

@dataclass
class RunResult:
    """单次运行结果"""
    log_path: Path
    backend: str  # pytorch or tensorrt
    duration: int  # 5 or 30
    seed: int
    success: bool
    exit_code: int

    # 延迟统计
    initial_latency_ms: float
    latencies_ms: List[float]
    mean_latency_ms: float
    median_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float

    # 运行统计
    clamp_count: int
    consecutive_clamp_count: int
    actual_runtime_s: float

def extract_latencies(log_path):
    """提取延迟数据"""
    pattern = r"RTC timing diagnostics:.*?latency_ms=([\d.]+)"
    latencies = []

    with open(log_path) as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                latencies.append(float(match.group(1)))

    return latencies

def count_clamps(log_path):
    """统计钳制次数"""
    clamp_pattern = r"Relative goal position magnitude had to be clamped"
    consecutive_pattern = r"safety clamp rewrote the commanded goal on (\d+) consecutive"

    clamp_count = 0
    consecutive_count = 0

    with open(log_path) as f:
        content = f.read()
        clamp_count = len(re.findall(clamp_pattern, content))

        match = re.search(consecutive_pattern, content)
        if match:
            consecutive_count = int(match.group(1))

    return clamp_count, consecutive_count

def check_success(log_path):
    """检查运行是否成功"""
    with open(log_path) as f:
        content = f.read()

    # 成功标志
    success_markers = [
        "Rollout finished",
        "Duration limit reached"
    ]

    # 失败标志
    failure_markers = [
        "Traceback",
        "StallContactError",
        "Abnormal shutdown"
    ]

    has_success = any(marker in content for marker in success_markers)
    has_failure = any(marker in content for marker in failure_markers)

    if has_failure:
        return False, 1
    elif has_success:
        return True, 0
    else:
        return False, -1  # 未知状态

def parse_log(log_path: Path, backend: str, duration: int, seed: int) -> RunResult:
    """解析单个日志文件"""
    latencies = extract_latencies(log_path)
    clamp_count, consecutive_count = count_clamps(log_path)
    success, exit_code = check_success(log_path)

    if not latencies:
        print(f"  ⚠️  {log_path.name}: 无延迟数据")
        return None

    initial = latencies[0]
    steady = latencies[1:] if len(latencies) > 1 else latencies

    return RunResult(
        log_path=log_path,
        backend=backend,
        duration=duration,
        seed=seed,
        success=success,
        exit_code=exit_code,
        initial_latency_ms=initial,
        latencies_ms=steady,
        mean_latency_ms=mean(steady),
        median_latency_ms=median(steady),
        max_latency_ms=max(steady),
        p95_latency_ms=sorted(steady)[int(len(steady) * 0.95)] if len(steady) > 1 else max(steady),
        clamp_count=clamp_count,
        consecutive_clamp_count=consecutive_count,
        actual_runtime_s=len(steady) * (1/6)  # 假设 6Hz replan
    )

def print_summary(results: List[RunResult], title: str):
    """打印汇总统计"""
    if not results:
        print(f"\n{title}: ❌ 无数据")
        return

    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r.success)
    print(f"运行次数: {len(results)}")
    print(f"成功次数: {success_count} / {len(results)} ({success_count/len(results)*100:.1f}%)")
    print()

    # 延迟统计
    all_latencies = []
    for r in results:
        all_latencies.extend(r.latencies_ms)

    if all_latencies:
        print(f"延迟统计 (合并 {len(all_latencies)} 个样本):")
        print(f"  均值:   {mean(all_latencies):.2f} ms")
        print(f"  中位数: {median(all_latencies):.2f} ms")
        print(f"  最大值: {max(all_latencies):.2f} ms")
        print(f"  最小值: {min(all_latencies):.2f} ms")
        print(f"  标准差: {stdev(all_latencies) if len(all_latencies) > 1 else 0:.2f} ms")
        print(f"  p95:    {sorted(all_latencies)[int(len(all_latencies)*0.95)]:.2f} ms")
        print()

    # 钳制统计
    total_clamps = sum(r.clamp_count for r in results)
    print(f"钳制统计:")
    print(f"  总钳制次数: {total_clamps}")
    print(f"  平均每次:   {total_clamps/len(results):.1f}")
    print()

    # 每次运行详情
    print("详细记录:")
    for i, r in enumerate(results, 1):
        status = "✅" if r.success else "❌"
        print(f"  运行{i} (seed={r.seed}): {status} | "
              f"延迟={r.mean_latency_ms:.1f}ms (max={r.max_latency_ms:.1f}ms) | "
              f"钳制={r.clamp_count}次")

def compare_backends(pytorch_results: List[RunResult], trt_results: List[RunResult], duration: int):
    """对比两个后端"""
    print(f"\n{'='*80}")
    print(f"【对比分析 - {duration}秒运行】")
    print(f"{'='*80}")

    if not pytorch_results or not trt_results:
        print("⚠️  数据不足，无法对比")
        return

    # 合并延迟数据
    pt_latencies = []
    for r in pytorch_results:
        pt_latencies.extend(r.latencies_ms)

    trt_latencies = []
    for r in trt_results:
        trt_latencies.extend(r.latencies_ms)

    pt_mean = mean(pt_latencies)
    trt_mean = mean(trt_latencies)
    speedup = (pt_mean - trt_mean) / pt_mean * 100

    print(f"\n延迟对比:")
    print(f"  PyTorch 均值:  {pt_mean:.2f} ms")
    print(f"  TensorRT 均值: {trt_mean:.2f} ms")
    print(f"  加速幅度:      {speedup:+.1f}%")
    print()

    pt_max = max(pt_latencies)
    trt_max = max(trt_latencies)
    max_improve = (pt_max - trt_max) / pt_max * 100

    print(f"最大延迟对比:")
    print(f"  PyTorch:  {pt_max:.2f} ms")
    print(f"  TensorRT: {trt_max:.2f} ms")
    print(f"  改善:     {max_improve:+.1f}%")
    print()

    target = 166.667
    pt_margin = target - pt_max
    trt_margin = target - trt_max

    print(f"相对目标 ({target:.1f} ms) 余量:")
    print(f"  PyTorch:  {pt_margin:+.2f} ms {'✅' if pt_margin > 0 else '❌'}")
    print(f"  TensorRT: {trt_margin:+.2f} ms {'✅' if trt_margin > 0 else '❌'}")
    print()

    # 成功率对比
    pt_success = sum(1 for r in pytorch_results if r.success)
    trt_success = sum(1 for r in trt_results if r.success)

    print(f"成功率对比:")
    print(f"  PyTorch:  {pt_success}/{len(pytorch_results)} ({pt_success/len(pytorch_results)*100:.1f}%)")
    print(f"  TensorRT: {trt_success}/{len(trt_results)} ({trt_success/len(trt_results)*100:.1f}%)")

def main():
    """主函数 - 需要手动填入日志路径"""

    print("="*80)
    print("PyTorch vs TensorRT 完整对比分析")
    print("="*80)
    print("\n⚠️  请在运行完实验后，手动填入日志路径\n")

    # ========== 手动填入日志路径 ==========

    # PyTorch 5秒运行
    pytorch_5s_logs = [
        # Path("/path/to/pytorch_5s_seed1000.log"),
        # Path("/path/to/pytorch_5s_seed1001.log"),
        # Path("/path/to/pytorch_5s_seed1002.log"),
    ]

    # TensorRT 5秒运行
    trt_5s_logs = [
        # Path("/path/to/trt_5s_seed1000.log"),
        # Path("/path/to/trt_5s_seed1001.log"),
        # Path("/path/to/trt_5s_seed1002.log"),
    ]

    # PyTorch 30秒运行
    pytorch_30s_logs = [
        # Path("/path/to/pytorch_30s_seed1000.log"),
        # Path("/path/to/pytorch_30s_seed1001.log"),
    ]

    # TensorRT 30秒运行
    trt_30s_logs = [
        # Path("/path/to/trt_30s_seed1000.log"),
        # Path("/path/to/trt_30s_seed1001.log"),
    ]

    # ======================================

    # 解析日志
    pytorch_5s_results = []
    for log in pytorch_5s_logs:
        if log.exists():
            result = parse_log(log, "pytorch", 5, 1000)
            if result:
                pytorch_5s_results.append(result)

    trt_5s_results = []
    for log in trt_5s_logs:
        if log.exists():
            result = parse_log(log, "tensorrt", 5, 1000)
            if result:
                trt_5s_results.append(result)

    pytorch_30s_results = []
    for log in pytorch_30s_logs:
        if log.exists():
            result = parse_log(log, "pytorch", 30, 1000)
            if result:
                pytorch_30s_results.append(result)

    trt_30s_results = []
    for log in trt_30s_logs:
        if log.exists():
            result = parse_log(log, "tensorrt", 30, 1000)
            if result:
                trt_30s_results.append(result)

    # 打印汇总
    print_summary(pytorch_5s_results, "【PyTorch - 5秒快速抓取】")
    print_summary(trt_5s_results, "【TensorRT - 5秒快速抓取】")
    print_summary(pytorch_30s_results, "【PyTorch - 30秒完整流程】")
    print_summary(trt_30s_results, "【TensorRT - 30秒完整流程】")

    # 对比分析
    if pytorch_5s_results and trt_5s_results:
        compare_backends(pytorch_5s_results, trt_5s_results, 5)

    if pytorch_30s_results and trt_30s_results:
        compare_backends(pytorch_30s_results, trt_30s_results, 30)

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)

if __name__ == "__main__":
    main()
