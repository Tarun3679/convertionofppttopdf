#!/usr/bin/env python3
"""
Memory Leak Analyzer for memory_profiler.py Logs

Parses logs from memory_profiler.py and applies industrial-standard metrics
to detect memory leaks and identify culprits.

Based on industry best practices from:
- Datadog Memory Leak Workflow
- Stripe's per-request memory tracking
- psleak framework (giampaolo/psleak)
- Academic research (Precog algorithm)

Usage:
    python analyze_memory_logs.py <logfile>
    python analyze_memory_logs.py logfile.txt --format csv
    cat logfile.txt | python analyze_memory_logs.py
"""

import re
import sys
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import statistics


# Log entry data structure
@dataclass
class LogEntry:
    """Parsed log entry from memory_profiler"""
    timestamp: datetime
    log_type: str  # START, END, MONITOR
    job_name: str

    # Memory metrics (all in MB)
    rss_mb: Optional[float] = None
    vms_mb: Optional[float] = None
    rss_before: Optional[float] = None
    rss_after: Optional[float] = None
    rss_delta: Optional[float] = None
    vms_delta: Optional[float] = None
    peak_rss: Optional[float] = None
    current_rss_mb: Optional[float] = None
    peak_rss_mb: Optional[float] = None

    # Other metrics
    duration: Optional[float] = None
    memory_percent: Optional[float] = None
    system_available_mb: Optional[float] = None
    elapsed: Optional[float] = None


@dataclass
class OrphanedJob:
    """Job that started but never completed - primary OOM suspect"""
    job_name: str
    start_time: datetime
    rss_at_start: float
    vms_at_start: float
    memory_percent_at_start: float
    system_available_at_start: float


@dataclass
class JobStats:
    """Aggregated statistics for a job across multiple runs"""
    job_name: str
    run_count: int = 0
    orphaned_count: int = 0  # Number of times job started but didn't finish

    # RSS tracking
    rss_deltas: List[float] = field(default_factory=list)
    rss_absolutes: List[Tuple[datetime, float]] = field(default_factory=list)  # (time, rss)
    peak_rss_values: List[float] = field(default_factory=list)

    # VMS tracking
    vms_deltas: List[float] = field(default_factory=list)

    # Duration tracking
    durations: List[float] = field(default_factory=list)

    # Calculated metrics
    avg_rss_delta: float = 0.0
    total_rss_delta: float = 0.0
    avg_vms_delta: float = 0.0
    avg_peak_rss: float = 0.0
    max_peak_rss: float = 0.0

    # Leak detection metrics
    rss_growth_rate_mb_per_hour: float = 0.0  # Linear regression slope
    baseline_shift_mb: float = 0.0  # First run vs last run
    cumulative_impact: float = 0.0  # frequency × avg_delta

    # Container-aware metrics
    time_to_oom_hours: Optional[float] = None  # Predicted hours until OOM
    headroom_consumed_percent: float = 0.0  # % of available headroom consumed

    # Statistical metrics
    stddev_rss_delta: float = 0.0
    is_statistical_outlier: bool = False

    # Scoring
    leak_score: float = 0.0


@dataclass
class MemoryAnalysisReport:
    """Complete analysis report"""
    total_entries: int
    total_jobs_tracked: int
    time_range: Tuple[datetime, datetime]

    overall_rss_growth: float  # Total RSS change from start to end
    overall_growth_rate_mb_per_hour: float

    # CRITICAL: Orphaned jobs (started but never finished)
    orphaned_jobs: List[OrphanedJob]

    job_stats: Dict[str, JobStats]
    top_culprits: List[Tuple[str, JobStats]]  # Sorted by leak_score

    # Container context
    container_limit_mb: Optional[float] = None
    current_rss_mb: Optional[float] = None
    available_headroom_mb: Optional[float] = None
    headroom_percent: Optional[float] = None

    # Global statistics
    global_avg_delta: float = 0.0
    global_stddev_delta: float = 0.0


class MemoryLogParser:
    """Parser for memory_profiler.py log format"""

    # Regex patterns for different log types
    TRACK_START_PATTERN = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?\[MEMORY_TRACK_START\] '
        r"job='([^']+)' "
        r'rss_mb=([\d.]+) '
        r'vms_mb=([\d.]+) '
        r'memory_percent=([\d.]+) '
        r'system_available_mb=([\d.]+)'
    )

    TRACK_END_PATTERN = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?\[MEMORY_TRACK_END\] '
        r"job='([^']+)' "
        r'duration=([\d.]+)s '
        r'rss_before=([\d.]+)MB '
        r'rss_after=([\d.]+)MB '
        r'rss_delta=([+-]?[\d.]+)MB '
        r'vms_delta=([+-]?[\d.]+)MB '
        r'peak_rss=([\d.]+)MB'
    )

    MONITOR_PATTERN = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?\[MEMORY_MONITOR\] '
        r"job='([^']+)' "
        r'elapsed=([\d.]+)s '
        r'current_rss_mb=([\d.]+) '
        r'peak_rss_mb=([\d.]+) '
        r'current_vms_mb=([\d.]+) '
        r'system_available_mb=([\d.]+) '
        r'memory_percent=([\d.]+)'
    )

    @staticmethod
    def parse_timestamp(ts_str: str) -> datetime:
        """Parse timestamp string to datetime"""
        return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')

    @classmethod
    def parse_line(cls, line: str) -> Optional[LogEntry]:
        """Parse a single log line"""
        # Try TRACK_END pattern (most useful for leak detection)
        match = cls.TRACK_END_PATTERN.search(line)
        if match:
            return LogEntry(
                timestamp=cls.parse_timestamp(match.group(1)),
                log_type='END',
                job_name=match.group(2),
                duration=float(match.group(3)),
                rss_before=float(match.group(4)),
                rss_after=float(match.group(5)),
                rss_delta=float(match.group(6)),
                vms_delta=float(match.group(7)),
                peak_rss=float(match.group(8))
            )

        # Try TRACK_START pattern
        match = cls.TRACK_START_PATTERN.search(line)
        if match:
            return LogEntry(
                timestamp=cls.parse_timestamp(match.group(1)),
                log_type='START',
                job_name=match.group(2),
                rss_mb=float(match.group(3)),
                vms_mb=float(match.group(4)),
                memory_percent=float(match.group(5)),
                system_available_mb=float(match.group(6))
            )

        # Try MONITOR pattern
        match = cls.MONITOR_PATTERN.search(line)
        if match:
            return LogEntry(
                timestamp=cls.parse_timestamp(match.group(1)),
                log_type='MONITOR',
                job_name=match.group(2),
                elapsed=float(match.group(3)),
                current_rss_mb=float(match.group(4)),
                peak_rss_mb=float(match.group(5)),
                memory_percent=float(match.group(8))
            )

        return None


class MemoryLeakAnalyzer:
    """
    Analyzes memory logs using industrial-standard leak detection metrics

    Implements (in priority order):
    1. Orphaned job detection (START without END) - PRIMARY OOM indicator
    2. Container-aware analysis (time-to-OOM, headroom)
    3. Linear regression for RSS growth rate (industry standard)
    4. Per-job average delta (Stripe approach)
    5. Baseline shift detection (long-running patterns)
    6. Statistical outlier detection
    7. Cumulative impact scoring (frequency × delta)
    """

    def __init__(self,
                 container_limit_mb: Optional[float] = None,
                 growth_rate_threshold_mb_per_hour: float = 1.0,
                 baseline_shift_threshold_mb: float = 10.0,
                 avg_delta_threshold_mb: float = 0.5):
        """
        Initialize analyzer with configurable thresholds

        Args:
            container_limit_mb: Container memory limit in MB (e.g., 2048 for 2GiB)
            growth_rate_threshold_mb_per_hour: Alert if RSS grows faster than this (default: 1.0 MB/hr)
            baseline_shift_threshold_mb: Alert if baseline shifts by this much (default: 10 MB)
            avg_delta_threshold_mb: Alert if average delta exceeds this (default: 0.5 MB)
        """
        self.container_limit_mb = container_limit_mb
        self.growth_rate_threshold = growth_rate_threshold_mb_per_hour
        self.baseline_shift_threshold = baseline_shift_threshold_mb
        self.avg_delta_threshold = avg_delta_threshold_mb

        self.entries: List[LogEntry] = []
        self.job_stats: Dict[str, JobStats] = defaultdict(lambda: JobStats(job_name=""))
        self.orphaned_jobs: List[OrphanedJob] = []

    def parse_logs(self, log_lines: List[str]) -> int:
        """Parse log lines and extract entries"""
        parser = MemoryLogParser()

        for line in log_lines:
            entry = parser.parse_line(line)
            if entry:
                self.entries.append(entry)

        return len(self.entries)

    def detect_orphaned_jobs(self):
        """
        Detect jobs that started but never finished - PRIMARY OOM suspects

        This is the most reliable indicator of which job caused a container crash.
        Works regardless of log order (forward or reverse chronological).
        """
        # Sort entries by timestamp to handle both forward and reverse chronological logs
        sorted_entries = sorted(self.entries, key=lambda e: e.timestamp)

        # Track START entries by job name and timestamp
        started_jobs: Dict[Tuple[str, datetime], LogEntry] = {}
        completed_job_keys: set = set()

        for entry in sorted_entries:
            if entry.log_type == 'START':
                key = (entry.job_name, entry.timestamp)
                started_jobs[key] = entry
            elif entry.log_type == 'END':
                # Find matching START entry (closest timestamp before END)
                matching_start = None
                min_time_diff = None

                for (job_name, start_time) in started_jobs.keys():
                    if job_name == entry.job_name and start_time <= entry.timestamp:
                        time_diff = (entry.timestamp - start_time).total_seconds()
                        if min_time_diff is None or time_diff < min_time_diff:
                            min_time_diff = time_diff
                            matching_start = (job_name, start_time)

                if matching_start:
                    completed_job_keys.add(matching_start)

        # Jobs with START but no matching END
        orphaned_keys = set(started_jobs.keys()) - completed_job_keys

        for key in orphaned_keys:
            entry = started_jobs[key]
            orphaned = OrphanedJob(
                job_name=entry.job_name,
                start_time=entry.timestamp,
                rss_at_start=entry.rss_mb or 0.0,
                vms_at_start=entry.vms_mb or 0.0,
                memory_percent_at_start=entry.memory_percent or 0.0,
                system_available_at_start=entry.system_available_mb or 0.0
            )
            self.orphaned_jobs.append(orphaned)

            # Track orphaned count in job stats
            stats = self.job_stats[entry.job_name]
            stats.job_name = entry.job_name
            stats.orphaned_count += 1

    def aggregate_job_stats(self):
        """Aggregate statistics per job (for completed jobs)"""
        for entry in self.entries:
            if entry.log_type == 'END':
                stats = self.job_stats[entry.job_name]
                stats.job_name = entry.job_name
                stats.run_count += 1

                if entry.rss_delta is not None:
                    stats.rss_deltas.append(entry.rss_delta)

                if entry.vms_delta is not None:
                    stats.vms_deltas.append(entry.vms_delta)

                if entry.peak_rss is not None:
                    stats.peak_rss_values.append(entry.peak_rss)

                if entry.duration is not None:
                    stats.durations.append(entry.duration)

                # Track absolute RSS over time for trend analysis
                if entry.rss_after is not None:
                    stats.rss_absolutes.append((entry.timestamp, entry.rss_after))

    def calculate_statistics(self):
        """Calculate statistical metrics for each job"""
        all_deltas = []

        for job_name, stats in self.job_stats.items():
            # Basic aggregates
            if stats.rss_deltas:
                stats.avg_rss_delta = statistics.mean(stats.rss_deltas)
                stats.total_rss_delta = sum(stats.rss_deltas)
                stats.stddev_rss_delta = statistics.stdev(stats.rss_deltas) if len(stats.rss_deltas) > 1 else 0.0
                all_deltas.extend(stats.rss_deltas)

            if stats.vms_deltas:
                stats.avg_vms_delta = statistics.mean(stats.vms_deltas)

            if stats.peak_rss_values:
                stats.avg_peak_rss = statistics.mean(stats.peak_rss_values)
                stats.max_peak_rss = max(stats.peak_rss_values)

            # Industry Metric 1: Linear Regression for Growth Rate (MB/hour)
            if len(stats.rss_absolutes) >= 2:
                stats.rss_growth_rate_mb_per_hour = self._calculate_growth_rate(stats.rss_absolutes)

            # Industry Metric 2: Baseline Shift (first vs last occurrence)
            if len(stats.rss_absolutes) >= 2:
                first_rss = stats.rss_absolutes[0][1]
                last_rss = stats.rss_absolutes[-1][1]
                stats.baseline_shift_mb = last_rss - first_rss

            # Industry Metric 3: Cumulative Impact (Stripe approach)
            stats.cumulative_impact = stats.run_count * stats.avg_rss_delta

            # Container-aware metrics
            if self.container_limit_mb and stats.avg_peak_rss > 0:
                available_headroom = self.container_limit_mb - stats.avg_peak_rss
                stats.headroom_consumed_percent = (stats.avg_peak_rss / self.container_limit_mb) * 100

                # Time to OOM prediction
                if stats.rss_growth_rate_mb_per_hour > 0 and available_headroom > 0:
                    stats.time_to_oom_hours = available_headroom / stats.rss_growth_rate_mb_per_hour

        # Global statistics for outlier detection
        if all_deltas:
            global_avg = statistics.mean(all_deltas)
            global_stddev = statistics.stdev(all_deltas) if len(all_deltas) > 1 else 0.0

            # Mark statistical outliers (> mean + 2*stddev)
            for stats in self.job_stats.values():
                if stats.avg_rss_delta > global_avg + (2 * global_stddev):
                    stats.is_statistical_outlier = True

    def _calculate_growth_rate(self, rss_time_series: List[Tuple[datetime, float]]) -> float:
        """
        Calculate RSS growth rate using linear regression

        Returns: MB per hour
        """
        if len(rss_time_series) < 2:
            return 0.0

        # Convert to seconds since first timestamp
        first_time = rss_time_series[0][0]
        x_values = [(ts - first_time).total_seconds() for ts, _ in rss_time_series]
        y_values = [rss for _, rss in rss_time_series]

        # Simple linear regression: y = mx + b
        n = len(x_values)
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator  # MB per second
        slope_per_hour = slope * 3600  # Convert to MB per hour

        return slope_per_hour

    def calculate_leak_scores(self):
        """
        Calculate leak score for each job using multi-metric approach

        Score components (industry best practices):
        - Orphaned jobs × 10000 (CRITICAL - job caused crash)
        - RSS growth rate × 40 (primary leak indicator)
        - Average RSS delta × 20 (per-execution leak)
        - Baseline shift × 15 (long-term drift)
        - VMS delta × 10 (virtual memory issues)
        - Cumulative impact × 10 (frequency-based)
        - Statistical outlier × 5 (bonus for outliers)
        """
        for stats in self.job_stats.values():
            score = 0.0

            # Weight 10000: Orphaned jobs (MOST CRITICAL - job likely caused OOM)
            if stats.orphaned_count > 0:
                score += stats.orphaned_count * 10000

            # Weight 40: Growth rate (most important for completed jobs)
            if stats.rss_growth_rate_mb_per_hour > self.growth_rate_threshold:
                score += stats.rss_growth_rate_mb_per_hour * 40

            # Weight 20: Average delta per execution
            if stats.avg_rss_delta > self.avg_delta_threshold:
                score += stats.avg_rss_delta * 20

            # Weight 15: Baseline shift
            if stats.baseline_shift_mb > self.baseline_shift_threshold:
                score += stats.baseline_shift_mb * 15

            # Weight 10: VMS delta (virtual memory issues)
            score += abs(stats.avg_vms_delta) * 10

            # Weight 10: Cumulative impact
            score += stats.cumulative_impact * 10

            # Weight 5: Statistical outlier bonus
            if stats.is_statistical_outlier:
                score += 5

            stats.leak_score = score

    def analyze(self) -> MemoryAnalysisReport:
        """Run complete analysis pipeline"""
        if not self.entries:
            raise ValueError("No log entries to analyze")

        # Step 0: CRITICAL - Detect orphaned jobs (START without END)
        self.detect_orphaned_jobs()

        # Step 1: Aggregate per-job statistics
        self.aggregate_job_stats()

        # Step 2: Calculate statistical metrics
        self.calculate_statistics()

        # Step 3: Calculate leak scores
        self.calculate_leak_scores()

        # Step 4: Sort jobs by leak score (orphaned jobs will be at top)
        sorted_jobs = sorted(
            self.job_stats.items(),
            key=lambda x: x[1].leak_score,
            reverse=True
        )

        # Step 5: Calculate overall metrics
        end_entries = [e for e in self.entries if e.log_type == 'END']

        if end_entries:
            first_rss = end_entries[0].rss_before
            last_rss = end_entries[-1].rss_after
            overall_growth = last_rss - first_rss if first_rss and last_rss else 0.0

            time_range = (self.entries[0].timestamp, self.entries[-1].timestamp)
            duration_hours = (time_range[1] - time_range[0]).total_seconds() / 3600
            overall_growth_rate = overall_growth / duration_hours if duration_hours > 0 else 0.0
        else:
            overall_growth = 0.0
            overall_growth_rate = 0.0
            time_range = (self.entries[0].timestamp, self.entries[-1].timestamp)

        # Calculate global statistics
        all_deltas = [d for stats in self.job_stats.values() for d in stats.rss_deltas]
        global_avg = statistics.mean(all_deltas) if all_deltas else 0.0
        global_stddev = statistics.stdev(all_deltas) if len(all_deltas) > 1 else 0.0

        # Container context
        current_rss = None
        if end_entries:
            current_rss = end_entries[-1].rss_after

        available_headroom = None
        headroom_percent = None
        if self.container_limit_mb and current_rss:
            available_headroom = self.container_limit_mb - current_rss
            headroom_percent = (available_headroom / self.container_limit_mb) * 100

        return MemoryAnalysisReport(
            total_entries=len(self.entries),
            total_jobs_tracked=len(self.job_stats),
            time_range=time_range,
            overall_rss_growth=overall_growth,
            overall_growth_rate_mb_per_hour=overall_growth_rate,
            orphaned_jobs=self.orphaned_jobs,
            job_stats=dict(self.job_stats),
            top_culprits=sorted_jobs[:10],  # Top 10 culprits
            container_limit_mb=self.container_limit_mb,
            current_rss_mb=current_rss,
            available_headroom_mb=available_headroom,
            headroom_percent=headroom_percent,
            global_avg_delta=global_avg,
            global_stddev_delta=global_stddev
        )


class ReportGenerator:
    """Generate human-readable reports"""

    @staticmethod
    def generate_text_report(report: MemoryAnalysisReport) -> str:
        """Generate comprehensive text report"""
        lines = [
            "=" * 100,
            "MEMORY LEAK ANALYSIS REPORT",
            "=" * 100,
            "",
            "OVERVIEW:",
            f"  Log entries analyzed: {report.total_entries}",
            f"  Unique jobs tracked: {report.total_jobs_tracked}",
            f"  Time range: {report.time_range[0]} to {report.time_range[1]}",
            f"  Duration: {(report.time_range[1] - report.time_range[0]).total_seconds() / 60:.1f} minutes",
            "",
            f"  Overall RSS growth: {report.overall_rss_growth:+.2f} MB",
            f"  Overall growth rate: {report.overall_growth_rate_mb_per_hour:+.2f} MB/hour",
            ""
        ]

        # Container context
        if report.container_limit_mb:
            lines.extend([
                "CONTAINER CONTEXT:",
                f"  Container memory limit: {report.container_limit_mb:.2f} MB ({report.container_limit_mb / 1024:.2f} GiB)",
                f"  Current RSS usage: {report.current_rss_mb:.2f} MB ({(report.current_rss_mb / report.container_limit_mb * 100):.1f}% of limit)" if report.current_rss_mb else "  Current RSS usage: N/A",
                f"  Available headroom: {report.available_headroom_mb:.2f} MB ({report.headroom_percent:.1f}%)" if report.available_headroom_mb else "  Available headroom: N/A",
                ""
            ])

        # CRITICAL SECTION: Orphaned jobs
        if report.orphaned_jobs:
            lines.extend([
                "=" * 100,
                "!!! CRITICAL: JOBS THAT STARTED BUT NEVER COMPLETED !!!",
                "=" * 100,
                "",
                "These jobs are PRIMARY OOM SUSPECTS - they started but have no completion log.",
                "This typically indicates the container was killed (OOM) during job execution.",
                ""
            ])

            for idx, orphaned in enumerate(report.orphaned_jobs, 1):
                lines.extend([
                    f"{idx}. {orphaned.job_name}",
                    f"   {'─' * 95}",
                    f"   [STATUS]: JOB STARTED BUT NEVER FINISHED - LIKELY CAUSED CONTAINER CRASH",
                    f"   Start Time: {orphaned.start_time}",
                    f"   RSS at Start: {orphaned.rss_at_start:.2f} MB",
                    f"   VMS at Start: {orphaned.vms_at_start:.2f} MB",
                    f"   Memory % at Start: {orphaned.memory_percent_at_start:.2f}%",
                    f"   System Available at Start: {orphaned.system_available_at_start:.2f} MB",
                    ""
                ])

            lines.extend([
                "RECOMMENDATION: Investigate the jobs listed above. They are the most likely",
                "                culprits for OOM issues. Check what these jobs do and why",
                "                they might consume excessive memory.",
                ""
            ])
        else:
            lines.extend([
                "=" * 100,
                "ORPHANED JOBS CHECK: PASSED",
                "=" * 100,
                "",
                "All jobs that started also completed successfully.",
                "No evidence of container crash during job execution in this log sample.",
                ""
            ])

        lines.extend([
            "",
            "=" * 100,
            "TOP MEMORY LEAK CULPRITS (ranked by leak score)",
            "=" * 100,
            ""
        ])

        if not report.top_culprits:
            lines.append("  No culprits identified.")
        else:
            for rank, (job_name, stats) in enumerate(report.top_culprits, 1):
                if stats.leak_score < 1:  # Skip jobs with negligible scores
                    continue

                leak_indicators = []
                if stats.orphaned_count > 0:
                    leak_indicators.append("ORPHANED_JOB")
                if stats.rss_growth_rate_mb_per_hour > 1.0:
                    leak_indicators.append("GROWTH_RATE")
                if stats.avg_rss_delta > 0.5:
                    leak_indicators.append("AVG_DELTA")
                if stats.baseline_shift_mb > 10:
                    leak_indicators.append("BASELINE_SHIFT")

                # Determine severity
                if stats.orphaned_count > 0:
                    severity = "CRITICAL - CAUSED CRASH"
                elif stats.leak_score > 50:
                    severity = "HIGH PRIORITY"
                else:
                    severity = ""

                indicators = f"[{', '.join(leak_indicators)}]" if leak_indicators else ""

                lines.extend([
                    f"{rank}. {job_name}",
                    f"   {'─' * 95}",
                    f"   Leak Score: {stats.leak_score:.2f} {severity} {indicators}",
                    f"   Completed Runs: {stats.run_count}",
                ])

                if stats.orphaned_count > 0:
                    lines.append(f"   Orphaned Runs: {stats.orphaned_count} [CRITICAL - Job did not complete]")

                lines.extend([
                    "",
                    f"   RSS Metrics:",
                    f"     • Growth Rate: {stats.rss_growth_rate_mb_per_hour:+.2f} MB/hour {'[LEAK]' if stats.rss_growth_rate_mb_per_hour > 1.0 else '[OK]'}",
                    f"     • Avg Delta per Run: {stats.avg_rss_delta:+.2f} MB {'[LEAK]' if stats.avg_rss_delta > 0.5 else '[OK]'}",
                    f"     • Total Delta: {stats.total_rss_delta:+.2f} MB",
                    f"     • Baseline Shift: {stats.baseline_shift_mb:+.2f} MB {'[LEAK]' if stats.baseline_shift_mb > 10 else '[OK]'}",
                    f"     • Cumulative Impact: {stats.cumulative_impact:.2f} MB",
                    "",
                    f"   Peak RSS:",
                    f"     • Average: {stats.avg_peak_rss:.2f} MB",
                    f"     • Maximum: {stats.max_peak_rss:.2f} MB",
                ])

                # Container-aware metrics
                if stats.headroom_consumed_percent > 0:
                    lines.append(f"     • Headroom Consumed: {stats.headroom_consumed_percent:.1f}% of container limit")

                if stats.time_to_oom_hours is not None:
                    if stats.time_to_oom_hours < 24:
                        time_str = f"{stats.time_to_oom_hours:.1f} hours [CRITICAL]"
                    elif stats.time_to_oom_hours < 168:  # 7 days
                        time_str = f"{stats.time_to_oom_hours / 24:.1f} days [WARNING]"
                    else:
                        time_str = f"{stats.time_to_oom_hours / 24:.1f} days"
                    lines.append(f"     • Predicted Time to OOM: {time_str}")

                lines.extend([
                    "",
                    f"   VMS Metrics:",
                    f"     • Avg VMS Delta: {stats.avg_vms_delta:+.2f} MB",
                    "",
                    f"   Statistical:",
                    f"     • Outlier: {'YES [WARNING]' if stats.is_statistical_outlier else 'No'}",
                    f"     • StdDev: {stats.stddev_rss_delta:.2f} MB",
                    ""
                ])

        lines.extend([
            "=" * 100,
            "GLOBAL STATISTICS",
            "=" * 100,
            "",
            f"  Global average RSS delta: {report.global_avg_delta:.2f} MB",
            f"  Global standard deviation: {report.global_stddev_delta:.2f} MB",
            f"  Outlier threshold (mean + 2*stddev): {report.global_avg_delta + 2 * report.global_stddev_delta:.2f} MB",
            "",
            "=" * 100,
            "INTERPRETATION GUIDE",
            "=" * 100,
            "",
            "Priority Levels:",
            "  1. CRITICAL - Orphaned Jobs: Jobs with START but no END log",
            "     → Most reliable indicator that this job caused container crash/OOM",
            "     → Investigate immediately",
            "",
            "  2. HIGH PRIORITY - Growth Rate: RSS growing > 1 MB/hour",
            "     → Will cause future OOM if not addressed",
            "     → Time-to-OOM prediction shows urgency",
            "",
            "  3. WARNING - Statistical Anomalies: Outliers, high cumulative impact",
            "     → Monitor these jobs for trends",
            "",
            "Leak Indicators:",
            "  • Orphaned Job: Job started but never finished (PRIMARY OOM indicator)",
            "  • Growth Rate > 1 MB/hour: Sustained memory growth over time",
            "  • Avg Delta > 0.5 MB: Per-execution memory leak (Stripe approach)",
            "  • Baseline Shift > 10 MB: Long-running baseline drift",
            "  • High Cumulative Impact: Frequent small leaks accumulating",
            "  • Statistical Outlier: Unusual memory behavior compared to other jobs",
            "",
            "Container-Aware Metrics:",
            "  • Headroom: Available memory before hitting container limit",
            "  • Time to OOM: Predicted hours/days until container runs out of memory",
            "  • Critical: < 24 hours to OOM",
            "  • Warning: < 7 days to OOM",
            "",
            "Metrics Explained:",
            "  • RSS (Resident Set Size): Actual physical memory used",
            "  • VMS (Virtual Memory Size): Total virtual memory (includes swapped)",
            "  • Growth Rate: Linear regression slope (industry standard)",
            "  • Baseline Shift: Memory at last run - memory at first run",
            "  • Cumulative Impact: frequency × avg_delta (total impact on system)",
            "",
            "Sources:",
            "  • Datadog Memory Leak Workflow",
            "  • Stripe per-request memory tracking",
            "  • psleak framework (giampaolo/psleak)",
            "  • Academic research: Precog algorithm",
            "",
            "=" * 100
        ])

        return "\n".join(lines)

    @staticmethod
    def generate_csv_report(report: MemoryAnalysisReport) -> str:
        """Generate CSV report for spreadsheet analysis"""
        lines = [
            "job_name,orphaned_count,run_count,leak_score,growth_rate_mb_per_hour,avg_rss_delta_mb,"
            "total_rss_delta_mb,baseline_shift_mb,cumulative_impact_mb,avg_vms_delta_mb,"
            "avg_peak_rss_mb,max_peak_rss_mb,time_to_oom_hours,headroom_consumed_pct,is_outlier"
        ]

        for job_name, stats in report.top_culprits:
            time_to_oom = stats.time_to_oom_hours if stats.time_to_oom_hours else ""
            headroom_pct = f"{stats.headroom_consumed_percent:.2f}" if stats.headroom_consumed_percent > 0 else ""

            lines.append(
                f'"{job_name}",{stats.orphaned_count},{stats.run_count},{stats.leak_score:.2f},'
                f'{stats.rss_growth_rate_mb_per_hour:.2f},{stats.avg_rss_delta:.2f},'
                f'{stats.total_rss_delta:.2f},{stats.baseline_shift_mb:.2f},'
                f'{stats.cumulative_impact:.2f},{stats.avg_vms_delta:.2f},'
                f'{stats.avg_peak_rss:.2f},{stats.max_peak_rss:.2f},'
                f'{time_to_oom},{headroom_pct},'
                f'{1 if stats.is_statistical_outlier else 0}'
            )

        return "\n".join(lines)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze memory_profiler.py logs for memory leaks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_memory_logs.py logfile.txt
  python analyze_memory_logs.py logfile.txt --container-limit 2048
  python analyze_memory_logs.py logfile.txt --format csv
  cat logfile.txt | python analyze_memory_logs.py
  python analyze_memory_logs.py logfile.txt --growth-rate-threshold 2.0

Container-aware analysis (recommended for production):
  python analyze_memory_logs.py logfile.txt --container-limit 2048

Thresholds (configurable):
  --container-limit: Container memory limit in MB (e.g., 2048 for 2GiB)
  --growth-rate-threshold: Alert if RSS grows faster than X MB/hour (default: 1.0)
  --baseline-shift-threshold: Alert if baseline shifts by X MB (default: 10.0)
  --avg-delta-threshold: Alert if average delta exceeds X MB (default: 0.5)
        """
    )

    parser.add_argument(
        'logfile',
        nargs='?',
        help='Path to log file (or use stdin)'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'csv'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--container-limit',
        type=float,
        help='Container memory limit in MB (e.g., 2048 for 2GiB container)'
    )
    parser.add_argument(
        '--growth-rate-threshold',
        type=float,
        default=1.0,
        help='Growth rate threshold in MB/hour (default: 1.0)'
    )
    parser.add_argument(
        '--baseline-shift-threshold',
        type=float,
        default=10.0,
        help='Baseline shift threshold in MB (default: 10.0)'
    )
    parser.add_argument(
        '--avg-delta-threshold',
        type=float,
        default=0.5,
        help='Average delta threshold in MB (default: 0.5)'
    )

    args = parser.parse_args()

    # Read log lines
    if args.logfile:
        try:
            with open(args.logfile, 'r') as f:
                log_lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: File '{args.logfile}' not found", file=sys.stderr)
            sys.exit(1)
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin
        log_lines = sys.stdin.readlines()

    if not log_lines:
        print("Error: No log data provided", file=sys.stderr)
        sys.exit(1)

    # Analyze
    analyzer = MemoryLeakAnalyzer(
        container_limit_mb=args.container_limit,
        growth_rate_threshold_mb_per_hour=args.growth_rate_threshold,
        baseline_shift_threshold_mb=args.baseline_shift_threshold,
        avg_delta_threshold_mb=args.avg_delta_threshold
    )

    parsed_count = analyzer.parse_logs(log_lines)

    if parsed_count == 0:
        print("Error: No memory profiler log entries found in input", file=sys.stderr)
        print("Expected format: [YYYY-MM-DD HH:MM:SS] ... [MEMORY_TRACK_END] job='...' ...", file=sys.stderr)
        sys.exit(1)

    try:
        report = analyzer.analyze()
    except ValueError as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate report
    if args.format == 'csv':
        output = ReportGenerator.generate_csv_report(report)
    else:
        output = ReportGenerator.generate_text_report(report)

    print(output)


if __name__ == '__main__':
    main()
