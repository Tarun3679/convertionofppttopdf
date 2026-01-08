"""
Memory Profiling Utility for OpenShift/RHEL Job Monitoring

This module provides memory tracking for Python jobs to identify OOM culprits.
Works with both subprocess-based and function-based job execution patterns.

Requirements:
    pip install psutil

Usage:
    See examples at the bottom of this file
"""

import psutil
import os
import time
import functools
import subprocess
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time"""
    timestamp: float
    rss_mb: float  # Resident Set Size (actual physical memory)
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float  # System available memory

    def __str__(self):
        return f"RSS: {self.rss_mb:.2f}MB, VMS: {self.vms_mb:.2f}MB, Used: {self.percent:.1f}%"


@dataclass
class JobMemoryStats:
    """Memory statistics for a single job execution"""
    job_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Memory snapshots
    before: Optional[MemorySnapshot] = None
    after: Optional[MemorySnapshot] = None
    peak: Optional[MemorySnapshot] = None

    # Deltas
    rss_delta_mb: float = 0.0
    vms_delta_mb: float = 0.0

    # Status
    success: bool = True
    error: Optional[str] = None

    def calculate_deltas(self):
        """Calculate memory deltas after job completion"""
        if self.before and self.after:
            self.rss_delta_mb = self.after.rss_mb - self.before.rss_mb
            self.vms_delta_mb = self.after.vms_mb - self.before.vms_mb

    def __str__(self):
        status = "[SUCCESS]" if self.success else "[FAILED]"
        peak_info = f" | Peak RSS: {self.peak.rss_mb:.2f}MB" if self.peak else ""
        return (
            f"{status} {self.job_name} | "
            f"Duration: {self.duration_seconds:.2f}s | "
            f"RSS Delta: {self.rss_delta_mb:+.2f}MB | "
            f"VMS Delta: {self.vms_delta_mb:+.2f}MB"
            f"{peak_info}"
        )


class MemoryProfiler:
    """
    Central memory profiler for tracking all jobs in a leader process
    """

    def __init__(self, enable_continuous_monitoring: bool = False,
                 monitoring_interval: float = 1.0,
                 log_interval: float = 30.0):
        self.process = psutil.Process(os.getpid())
        self.stats: List[JobMemoryStats] = []
        self.enable_continuous_monitoring = enable_continuous_monitoring
        self.monitoring_interval = monitoring_interval
        self.log_interval = log_interval
        self._monitoring_thread = None
        self._current_job_stats: Optional[JobMemoryStats] = None

    def get_memory_snapshot(self) -> MemorySnapshot:
        """Capture current memory usage"""
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        system_mem = psutil.virtual_memory()

        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=mem_info.rss / (1024 * 1024),
            vms_mb=mem_info.vms / (1024 * 1024),
            percent=mem_percent,
            available_mb=system_mem.available / (1024 * 1024)
        )

    def _monitor_peak_memory(self, job_stats: JobMemoryStats):
        """Background monitoring thread for peak memory tracking with periodic logging"""
        import threading

        def monitor():
            last_log_time = 0

            while self._current_job_stats == job_stats:
                snapshot = self.get_memory_snapshot()
                current_time = time.time()

                # Update peak if current memory is higher
                peak_updated = False
                if not job_stats.peak or snapshot.rss_mb > job_stats.peak.rss_mb:
                    job_stats.peak = snapshot
                    peak_updated = True

                # Log if: peak was updated OR periodic log interval reached
                if peak_updated or (current_time - last_log_time >= self.log_interval):
                    elapsed = current_time - job_stats.start_time.timestamp()
                    logger.info(
                        f"[MEMORY_MONITOR] job='{job_stats.job_name}' "
                        f"elapsed={elapsed:.1f}s "
                        f"current_rss_mb={snapshot.rss_mb:.2f} "
                        f"peak_rss_mb={job_stats.peak.rss_mb:.2f} "
                        f"current_vms_mb={snapshot.vms_mb:.2f} "
                        f"system_available_mb={snapshot.available_mb:.2f} "
                        f"memory_percent={snapshot.percent:.2f}"
                    )
                    last_log_time = current_time

                time.sleep(self.monitoring_interval)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        return thread

    def start_job(self, job_name: str) -> JobMemoryStats:
        """Start tracking a new job"""
        job_stats = JobMemoryStats(
            job_name=job_name,
            start_time=datetime.now(),
            before=self.get_memory_snapshot()
        )

        self._current_job_stats = job_stats

        # Start peak monitoring if enabled
        if self.enable_continuous_monitoring:
            job_stats.peak = job_stats.before
            self._monitoring_thread = self._monitor_peak_memory(job_stats)
            logger.info(
                f"[MEMORY_TRACK_START] job='{job_name}' "
                f"rss_mb={job_stats.before.rss_mb:.2f} "
                f"vms_mb={job_stats.before.vms_mb:.2f} "
                f"memory_percent={job_stats.before.percent:.2f} "
                f"system_available_mb={job_stats.before.available_mb:.2f} "
                f"continuous_monitoring=enabled interval={self.monitoring_interval}s"
            )
        else:
            logger.info(
                f"[MEMORY_TRACK_START] job='{job_name}' "
                f"rss_mb={job_stats.before.rss_mb:.2f} "
                f"vms_mb={job_stats.before.vms_mb:.2f} "
                f"memory_percent={job_stats.before.percent:.2f} "
                f"system_available_mb={job_stats.before.available_mb:.2f}"
            )

        return job_stats

    def end_job(self, job_stats: JobMemoryStats, success: bool = True,
                error: Optional[str] = None):
        """End tracking for a job"""
        job_stats.end_time = datetime.now()
        job_stats.duration_seconds = (job_stats.end_time - job_stats.start_time).total_seconds()
        job_stats.after = self.get_memory_snapshot()
        job_stats.success = success
        job_stats.error = error

        if not self.enable_continuous_monitoring:
            job_stats.peak = job_stats.after

        job_stats.calculate_deltas()

        self._current_job_stats = None
        self.stats.append(job_stats)

        # Log completion with detailed metrics
        status_tag = "MEMORY_TRACK_END" if success else "MEMORY_TRACK_FAILED"
        log_msg = (
            f"[{status_tag}] job='{job_stats.job_name}' "
            f"duration={job_stats.duration_seconds:.2f}s "
            f"rss_before={job_stats.before.rss_mb:.2f}MB "
            f"rss_after={job_stats.after.rss_mb:.2f}MB "
            f"rss_delta={job_stats.rss_delta_mb:+.2f}MB "
            f"vms_delta={job_stats.vms_delta_mb:+.2f}MB "
            f"peak_rss={job_stats.peak.rss_mb:.2f}MB " if job_stats.peak else ""
            f"system_available_mb={job_stats.after.available_mb:.2f}"
        )

        if error:
            log_msg += f" error='{error}'"

        if success:
            logger.info(log_msg)
        else:
            logger.error(log_msg)

        return job_stats

    def run_subprocess_job(self, job_name: str, command: List[str],
                          **subprocess_kwargs) -> subprocess.CompletedProcess:
        """
        Execute a subprocess job with memory tracking

        Args:
            job_name: Name of the job for logging
            command: Command to execute as list (e.g., ['python', 'script.py'])
            **subprocess_kwargs: Additional arguments for subprocess.run()

        Returns:
            CompletedProcess object
        """
        job_stats = self.start_job(job_name)

        try:
            logger.debug(f"[SUBPROCESS_START] job='{job_name}' command={command}")
            result = subprocess.run(command, **subprocess_kwargs)
            success = result.returncode == 0
            error = None if success else f"Exit code: {result.returncode}"

            if not success:
                logger.warning(
                    f"[SUBPROCESS_NONZERO_EXIT] job='{job_name}' "
                    f"exit_code={result.returncode}"
                )

        except Exception as e:
            logger.error(
                f"[SUBPROCESS_EXCEPTION] job='{job_name}' "
                f"exception_type={type(e).__name__} exception_message='{str(e)}'"
            )
            result = None
            success = False
            error = str(e)

        finally:
            self.end_job(job_stats, success=success, error=error)

        return result

    def run_function_job(self, job_name: str, func: Callable,
                        *args, **kwargs) -> Any:
        """
        Execute a function job with memory tracking

        Args:
            job_name: Name of the job for logging
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Function return value
        """
        job_stats = self.start_job(job_name)

        try:
            logger.debug(f"[FUNCTION_START] job='{job_name}' function={func.__name__}")
            result = func(*args, **kwargs)
            success = True
            error = None

        except Exception as e:
            logger.error(
                f"[FUNCTION_EXCEPTION] job='{job_name}' function={func.__name__} "
                f"exception_type={type(e).__name__} exception_message='{str(e)}'"
            )
            result = None
            success = False
            error = str(e)
            raise  # Re-raise to maintain original behavior

        finally:
            self.end_job(job_stats, success=success, error=error)

        return result

    def get_report(self) -> str:
        """Generate a comprehensive memory report"""
        if not self.stats:
            return "No jobs tracked yet."

        lines = [
            "=" * 80,
            "MEMORY PROFILING REPORT",
            "=" * 80,
            ""
        ]

        # Individual job stats
        lines.append("Job-by-Job Memory Usage:")
        lines.append("-" * 80)

        for i, stat in enumerate(self.stats, 1):
            lines.append(f"{i}. {stat}")
            if stat.error:
                lines.append(f"   Error: {stat.error}")

        lines.append("")
        lines.append("-" * 80)

        # Summary statistics
        total_rss_delta = sum(s.rss_delta_mb for s in self.stats)
        max_rss_delta = max(self.stats, key=lambda s: s.rss_delta_mb)
        max_peak = max(self.stats, key=lambda s: s.peak.rss_mb if s.peak else 0)

        lines.extend([
            "SUMMARY:",
            f"  Total jobs tracked: {len(self.stats)}",
            f"  Total RSS delta: {total_rss_delta:+.2f}MB",
            f"  Largest RSS increase: {max_rss_delta.job_name} ({max_rss_delta.rss_delta_mb:+.2f}MB)",
            f"  Highest peak RSS: {max_peak.job_name} ({max_peak.peak.rss_mb:.2f}MB)" if max_peak.peak else "",
            ""
        ])

        # Identify potential culprits
        lines.append("POTENTIAL OOM CULPRITS (RSS delta > 50MB or RSS > 500MB):")
        culprits = [
            s for s in self.stats
            if s.rss_delta_mb > 50 or (s.peak and s.peak.rss_mb > 500)
        ]

        if culprits:
            for culprit in sorted(culprits, key=lambda s: s.rss_delta_mb, reverse=True):
                lines.append(f"  [WARNING] {culprit.job_name}:")
                lines.append(f"      RSS Delta: {culprit.rss_delta_mb:+.2f}MB")
                if culprit.peak:
                    lines.append(f"      Peak RSS: {culprit.peak.rss_mb:.2f}MB")
        else:
            lines.append("  No obvious culprits found.")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def print_report(self):
        """Print the memory report to console"""
        print(self.get_report())

    def save_report(self, filepath: str):
        """Save the memory report to a file"""
        try:
            with open(filepath, 'w') as f:
                f.write(self.get_report())
            logger.info(f"[REPORT_SAVED] filepath='{filepath}' size_bytes={os.path.getsize(filepath)}")
        except Exception as e:
            logger.error(
                f"[REPORT_SAVE_FAILED] filepath='{filepath}' "
                f"exception_type={type(e).__name__} exception_message='{str(e)}'"
            )


def profile_memory(job_name: str, profiler: Optional[MemoryProfiler] = None):
    """
    Decorator for profiling function-based jobs

    Usage:
        profiler = MemoryProfiler()

        @profile_memory("my_job", profiler)
        def my_job():
            # job code
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal profiler
            if profiler is None:
                # Create a local profiler if none provided
                profiler = MemoryProfiler()

            return profiler.run_function_job(job_name, func, *args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Memory Profiler - Example Usage\n")

    # Create a profiler instance
    profiler = MemoryProfiler(
        enable_continuous_monitoring=True,  # Track peak memory during execution
        monitoring_interval=0.5,  # Check memory every 0.5 seconds
        log_interval=5.0  # Log every 5 seconds (for demo; use 30-60s for long jobs)
    )

    # Example 1: Track subprocess jobs
    print("\n--- Example 1: Subprocess Jobs ---")

    profiler.run_subprocess_job(
        job_name="libreoffice_conversion",
        command=["python", "convert_to_markdown.py"],
        capture_output=True,
        text=True
    )

    profiler.run_subprocess_job(
        job_name="excel_generation",
        command=["python", "create_sample_excel.py"],
        capture_output=True,
        text=True
    )

    # Example 2: Track function jobs
    print("\n--- Example 2: Function Jobs ---")

    def memory_intensive_job():
        """Simulate a memory-intensive job"""
        data = [i for i in range(1000000)]
        return len(data)

    def another_job():
        """Another simulated job"""
        result = sum(range(1000000))
        return result

    profiler.run_function_job("memory_intensive", memory_intensive_job)
    profiler.run_function_job("calculation_job", another_job)

    # Example 3: Using decorator
    print("\n--- Example 3: Using Decorator ---")

    @profile_memory("decorated_job", profiler)
    def decorated_job(n):
        return [i**2 for i in range(n)]

    decorated_job(100000)

    # Print final report
    print("\n")
    profiler.print_report()

    # Optionally save to file
    # profiler.save_report("/tmp/memory_report.txt")
