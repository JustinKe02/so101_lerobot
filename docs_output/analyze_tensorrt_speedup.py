#!/usr/bin/env python3
"""
量化 TensorRT 加速效果的分析脚本
对比 T0a (PyTorch) vs T1 (TensorRT) 的推理延迟
"""

import re
import sys
from pathlib import Path
from statistics import mean, median, stdev

def extract_latencies(log_path):
    """从日志中提取 RTC 延迟数据"""
    pattern = r"RTC timing diagnostics:.*?latency_ms=([\d.]+)"
    latencies = []

    with open(log_path) as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                latencies.append(float(match.group(1)))

    return latencies

def analyze_latencies(latencies, label):
    """统计分析延迟数据"""
    if not latencies:
        return None

    # 去掉首次推理（冷启动）
    steady = latencies[1:] if len(latencies) > 1 else latencies

    if not steady:
        return None

    return {
        "label": label,
        "total_records": len(latencies),
        "steady_records": len(steady),
        "initial_ms": latencies[0],
        "steady_mean_ms": mean(steady),
        "steady_median_ms": median(steady),
        "steady_min_ms": min(steady),
        "steady_max_ms": max(steady),
        "steady_std_ms": stdev(steady) if len(steady) > 1 else 0,
    }

def main():
    root = Path("/data/cqy_workspace/tk/lerobot_src/outputs/rollout")

    # T0a: PyTorch prefix (成功的 5s 抓取)
    t0a_log = root / "t0a_q45/t0a_q45_20260726_185544.log"

    # T1: TensorRT prefix (两次运行，虽然守护触发但延迟数据有效)
    t1_logs = [
        root / "t1_q45/t1_q45_20260727_091710.log",
        root / "t1_q45/t1_q45_20260727_094933.log",
    ]

    print("=" * 80)
    print("TensorRT 加速效果量化分析")
    print("=" * 80)
    print()

    # 分析 T0a (PyTorch)
    print("【PyTorch Prefix - T0a】")
    print(f"日志: {t0a_log.name}")
    t0a_latencies = extract_latencies(t0a_log)
    t0a_stats = analyze_latencies(t0a_latencies, "PyTorch")

    if t0a_stats:
        print(f"  总记录数: {t0a_stats['total_records']}")
        print(f"  稳态记录数: {t0a_stats['steady_records']}")
        print(f"  初始延迟: {t0a_stats['initial_ms']:.2f} ms")
        print(f"  稳态延迟: 均值={t0a_stats['steady_mean_ms']:.2f} ms, "
              f"中位数={t0a_stats['steady_median_ms']:.2f} ms, "
              f"标准差={t0a_stats['steady_std_ms']:.2f} ms")
        print(f"  稳态范围: [{t0a_stats['steady_min_ms']:.2f}, {t0a_stats['steady_max_ms']:.2f}] ms")
    else:
        print("  ❌ 未找到有效数据")
    print()

    # 分析 T1 (TensorRT)
    print("【TensorRT Prefix - T1】")
    all_t1_latencies = []
    for i, log_path in enumerate(t1_logs, 1):
        print(f"\n运行 {i}: {log_path.name}")
        latencies = extract_latencies(log_path)
        stats = analyze_latencies(latencies, f"TensorRT-{i}")

        if stats:
            print(f"  总记录数: {stats['total_records']}")
            print(f"  稳态记录数: {stats['steady_records']}")
            print(f"  初始延迟: {stats['initial_ms']:.2f} ms")
            print(f"  稳态延迟: 均值={stats['steady_mean_ms']:.2f} ms, "
                  f"中位数={stats['steady_median_ms']:.2f} ms, "
                  f"标准差={stats['steady_std_ms']:.2f} ms")
            print(f"  稳态范围: [{stats['steady_min_ms']:.2f}, {stats['steady_max_ms']:.2f}] ms")

            # 收集所有稳态数据
            all_t1_latencies.extend(latencies[1:])
        else:
            print("  ❌ 未找到有效数据")

    # T1 汇总统计
    if all_t1_latencies:
        print("\n【TensorRT 汇总统计】")
        t1_stats = {
            "steady_mean_ms": mean(all_t1_latencies),
            "steady_median_ms": median(all_t1_latencies),
            "steady_min_ms": min(all_t1_latencies),
            "steady_max_ms": max(all_t1_latencies),
            "steady_std_ms": stdev(all_t1_latencies) if len(all_t1_latencies) > 1 else 0,
        }
        print(f"  合并稳态记录数: {len(all_t1_latencies)}")
        print(f"  稳态延迟: 均值={t1_stats['steady_mean_ms']:.2f} ms, "
              f"中位数={t1_stats['steady_median_ms']:.2f} ms, "
              f"标准差={t1_stats['steady_std_ms']:.2f} ms")
        print(f"  稳态范围: [{t1_stats['steady_min_ms']:.2f}, {t1_stats['steady_max_ms']:.2f}] ms")

    print()
    print("=" * 80)
    print("【加速效果对比】")
    print("=" * 80)

    if t0a_stats and all_t1_latencies:
        t0a_mean = t0a_stats['steady_mean_ms']
        t1_mean = t1_stats['steady_mean_ms']
        speedup_abs = t0a_mean - t1_mean
        speedup_pct = (speedup_abs / t0a_mean) * 100

        print(f"\n稳态延迟对比 (均值):")
        print(f"  PyTorch:  {t0a_mean:.2f} ms")
        print(f"  TensorRT: {t1_mean:.2f} ms")
        print(f"  降低:     {speedup_abs:.2f} ms ({speedup_pct:.1f}%)")
        print()

        t0a_median = t0a_stats['steady_median_ms']
        t1_median = t1_stats['steady_median_ms']
        speedup_median_abs = t0a_median - t1_median
        speedup_median_pct = (speedup_median_abs / t0a_median) * 100

        print(f"稳态延迟对比 (中位数):")
        print(f"  PyTorch:  {t0a_median:.2f} ms")
        print(f"  TensorRT: {t1_median:.2f} ms")
        print(f"  降低:     {speedup_median_abs:.2f} ms ({speedup_median_pct:.1f}%)")
        print()

        t0a_max = t0a_stats['steady_max_ms']
        t1_max = t1_stats['steady_max_ms']
        print(f"稳态延迟对比 (最大值):")
        print(f"  PyTorch:  {t0a_max:.2f} ms")
        print(f"  TensorRT: {t1_max:.2f} ms")
        print(f"  降低:     {t0a_max - t1_max:.2f} ms ({(t0a_max - t1_max)/t0a_max*100:.1f}%)")
        print()

        # 吞吐量提升（6Hz replan 的理论上限是 166.667ms）
        target_latency = 166.667
        t0a_margin = target_latency - t0a_max
        t1_margin = target_latency - t1_max

        print(f"相对目标延迟 (166.667 ms) 的余量:")
        print(f"  PyTorch:  {t0a_margin:.2f} ms 余量")
        print(f"  TensorRT: {t1_margin:.2f} ms 余量")
        print(f"  额外获得: {t1_margin - t0a_margin:.2f} ms 余量")
    else:
        print("\n❌ 数据不足，无法对比")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
