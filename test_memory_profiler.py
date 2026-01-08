#!/usr/bin/env python3
"""
Test script to demonstrate the memory profiler with periodic logging
"""
import time
from memory_profiler import MemoryProfiler


def simulate_long_job():
    """Simulate a long-running job with varying memory usage"""
    print("Starting long job simulation...")

    # Phase 1: Initial allocation
    data = []
    for i in range(5):
        data.append([0] * 1_000_000)  # Allocate ~8MB per iteration
        print(f"  Phase 1: Allocated {(i+1) * 8} MB")
        time.sleep(2)  # Sleep to allow monitoring logs to appear

    # Phase 2: Peak usage
    print("  Phase 2: Peak memory usage")
    time.sleep(3)

    # Phase 3: Cleanup
    print("  Phase 3: Cleaning up")
    data.clear()
    time.sleep(2)

    print("Job completed!")
    return len(data)


if __name__ == "__main__":
    print("=" * 80)
    print("Memory Profiler Test - Long Running Job with Periodic Logging")
    print("=" * 80)
    print()

    # Configure profiler for demonstration
    profiler = MemoryProfiler(
        enable_continuous_monitoring=True,
        monitoring_interval=1.0,  # Check every second
        log_interval=3.0          # Log every 3 seconds (or when peak increases)
    )

    print("Configuration:")
    print(f"  - Monitoring interval: {profiler.monitoring_interval}s (how often to check memory)")
    print(f"  - Log interval: {profiler.log_interval}s (how often to write logs)")
    print()
    print("Watch for [MEMORY_MONITOR] log entries appearing every 3 seconds...")
    print("=" * 80)
    print()

    # Run the simulated job
    profiler.run_function_job("long_running_simulation", simulate_long_job)

    # Print final report
    print()
    print("=" * 80)
    profiler.print_report()
