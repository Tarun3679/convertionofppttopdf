# Memory Profiling Guide for OpenShift/RHEL Jobs

This guide explains how to integrate the memory profiler into your leader job to identify OOM culprits.

## Installation

Add psutil to your Docker container:

```dockerfile
# In your Dockerfile
RUN pip install psutil
```

Or in your requirements.txt:
```
psutil>=5.8.0
```

## Quick Start

### Option 1: Wrap Your Existing Leader Job

If you have a leader job that executes other jobs, wrap them with the profiler:

```python
from memory_profiler import MemoryProfiler

# Create profiler instance
profiler = MemoryProfiler(
    enable_continuous_monitoring=True,  # Track peak memory
    monitoring_interval=1.0,             # Check every second
    log_interval=30.0                    # Log every 30 seconds
)

# Wrap your existing job executions
profiler.run_subprocess_job(
    job_name="libreoffice_conversion",
    command=["python", "convert_to_markdown.py", "--input", "file.pptx"]
)

profiler.run_subprocess_job(
    job_name="data_processing",
    command=["python", "process_data.py"]
)

# At the end, print the report
profiler.print_report()

# Or save to file for later analysis
profiler.save_report("/tmp/memory_report.txt")
```

### Option 2: Minimal Integration Pattern

```python
from memory_profiler import MemoryProfiler
import subprocess

profiler = MemoryProfiler()

# Before: your old code
# subprocess.run(["python", "job1.py"])
# subprocess.run(["python", "job2.py"])

# After: wrapped with profiler
profiler.run_subprocess_job("job1", ["python", "job1.py"])
profiler.run_subprocess_job("job2", ["python", "job2.py"])

profiler.print_report()
```

## Integration Examples

### Example 1: Sequential Job Runner (Most Common)

```python
#!/usr/bin/env python3
"""
Leader job that runs multiple Python jobs sequentially
"""
from memory_profiler import MemoryProfiler
import sys

def main():
    profiler = MemoryProfiler(
        enable_continuous_monitoring=True,
        monitoring_interval=1.0,
        log_interval=30.0
    )

    jobs = [
        ("libreoffice_conversion", ["python", "convert_to_markdown.py"]),
        ("excel_processing", ["python", "process_excel.py"]),
        ("data_validation", ["python", "validate_data.py"]),
        ("report_generation", ["python", "generate_report.py"]),
    ]

    print("Starting job execution with memory profiling...")

    for job_name, command in jobs:
        print(f"\n{'='*60}")
        print(f"Executing: {job_name}")
        print(f"{'='*60}")

        result = profiler.run_subprocess_job(
            job_name=job_name,
            command=command,
            capture_output=False,  # Show output in real-time
            check=False            # Don't raise on non-zero exit
        )

        if result.returncode != 0:
            print(f"WARNING: {job_name} failed with code {result.returncode}")
            # Continue with other jobs or break here if needed

    # Print final report
    print("\n\n")
    profiler.print_report()

    # Save to file in container
    profiler.save_report("/tmp/memory_report.txt")

    # Exit with error if any job showed high memory usage
    culprits = [s for s in profiler.stats if s.rss_delta_mb > 100]
    if culprits:
        print(f"\n[WARNING] {len(culprits)} jobs used >100MB")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Example 2: Function-Based Jobs

```python
from memory_profiler import MemoryProfiler

profiler = MemoryProfiler()

def libreoffice_job(input_file):
    # Your conversion logic here
    import subprocess
    subprocess.run(["libreoffice", "--convert-to", "pdf", input_file])

def processing_job(data):
    # Your processing logic
    return process(data)

# Wrap function calls
profiler.run_function_job("libreoffice", libreoffice_job, "presentation.pptx")
profiler.run_function_job("processing", processing_job, some_data)

profiler.print_report()
```

### Example 3: Mixed Subprocess + Functions

```python
from memory_profiler import MemoryProfiler

profiler = MemoryProfiler(
    enable_continuous_monitoring=True,
    monitoring_interval=2.0,
    log_interval=30.0
)

# Track subprocess
profiler.run_subprocess_job(
    "external_script",
    ["python", "external.py"]
)

# Track function
def internal_processing():
    # Your code
    pass

profiler.run_function_job("internal_task", internal_processing)

profiler.print_report()
```

### Example 4: Using Decorator for Functions

```python
from memory_profiler import MemoryProfiler, profile_memory

profiler = MemoryProfiler()

@profile_memory("conversion_job", profiler)
def convert_document(filepath):
    # Conversion logic
    pass

@profile_memory("validation_job", profiler)
def validate_output(filepath):
    # Validation logic
    pass

# Just call your functions normally
convert_document("input.pptx")
validate_output("output.md")

# Report is automatically tracked
profiler.print_report()
```

## Understanding the Logs

### Real-Time Monitoring Logs

When continuous monitoring is enabled, you'll see periodic log entries:

```
2026-01-08 10:15:23 - __main__ - INFO - [MEMORY_TRACK_START] job='convert_large_file.pptx' rss_mb=145.23 vms_mb=450.12 memory_percent=2.3 system_available_mb=5234.56 continuous_monitoring=enabled interval=1.0s
2026-01-08 10:15:33 - __main__ - INFO - [MEMORY_MONITOR] job='convert_large_file.pptx' elapsed=10.0s current_rss_mb=456.78 peak_rss_mb=456.78 current_vms_mb=678.90 system_available_mb=4923.45 memory_percent=7.2
2026-01-08 10:15:48 - __main__ - INFO - [MEMORY_MONITOR] job='convert_large_file.pptx' elapsed=25.1s current_rss_mb=823.45 peak_rss_mb=823.45 current_vms_mb=1045.23 system_available_mb=4356.78 memory_percent=12.8
2026-01-08 10:16:03 - __main__ - INFO - [MEMORY_MONITOR] job='convert_large_file.pptx' elapsed=40.3s current_rss_mb=1245.67 peak_rss_mb=1245.67 current_vms_mb=1523.12 system_available_mb=3890.45 memory_percent=19.4
2026-01-08 10:16:35 - __main__ - INFO - [MEMORY_TRACK_END] job='convert_large_file.pptx' duration=72.34s rss_before=145.23MB rss_after=890.12MB rss_delta=+744.89MB vms_delta=+856.34MB peak_rss=1245.67MB system_available_mb=4123.56
```

**Key log tags:**
- `[MEMORY_TRACK_START]`: Job begins, shows initial memory state
- `[MEMORY_MONITOR]`: Periodic updates during job execution (every `log_interval` seconds OR when peak increases)
- `[MEMORY_TRACK_END]`: Job completes, shows final memory state and deltas

**IMPORTANT for OOM debugging**: If your container OOMs (exit code 137), the `MEMORY_TRACK_END` will NOT be logged. Look for the last `MEMORY_MONITOR` entry to see memory usage just before the crash.

### Final Summary Report

When you run `profiler.print_report()`, you'll see:

```
================================================================================
MEMORY PROFILING REPORT
================================================================================

Job-by-Job Memory Usage:
--------------------------------------------------------------------------------
1. [SUCCESS] libreoffice_conversion | Duration: 12.34s | RSS Delta: +245.67MB | VMS Delta: +256.12MB | Peak RSS: 512.34MB
2. [SUCCESS] data_processing | Duration: 5.23s | RSS Delta: +15.23MB | VMS Delta: +18.45MB | Peak RSS: 234.56MB
3. [SUCCESS] report_generation | Duration: 3.45s | RSS Delta: +8.12MB | VMS Delta: +10.23MB | Peak RSS: 145.67MB

--------------------------------------------------------------------------------
SUMMARY:
  Total jobs tracked: 3
  Total RSS delta: +268.02MB
  Largest RSS increase: libreoffice_conversion (+245.67MB)
  Highest peak RSS: libreoffice_conversion (512.34MB)

POTENTIAL OOM CULPRITS (RSS delta > 50MB or RSS > 500MB):
  [WARNING] libreoffice_conversion:
      RSS Delta: +245.67MB
      Peak RSS: 512.34MB

================================================================================
```

### Configuration Parameters Explained

- **`enable_continuous_monitoring`**: Enable background thread to track peak memory during job execution
  - `True`: Tracks actual peak memory usage (recommended for debugging OOM)
  - `False`: Only captures memory at start/end of job

- **`monitoring_interval`**: How often (in seconds) to check memory usage
  - Lower values (1-2s): More accurate peak detection, slightly higher CPU overhead
  - Higher values (5-10s): Less accurate but minimal overhead
  - Recommended: **1-2s for debugging, 5s for production**

- **`log_interval`**: How often (in seconds) to write memory logs during job execution
  - Lower values (5-10s): More detailed logs, higher log volume
  - Higher values (30-60s): Less log spam, still catch issues
  - Recommended: **30-60s for long-running jobs (8+ hours), 10-30s for short jobs**
  - **Note**: Logs are always written when peak memory increases, regardless of interval

### Key Metrics Explained

- **RSS (Resident Set Size)**: Actual physical memory used (most important)
- **VMS (Virtual Memory Size)**: Total virtual memory (includes swap)
- **RSS Δ**: Memory increase/decrease after job completion
- **Peak RSS**: Maximum memory used during job execution
- **Duration**: How long the job took

### What to Look For

1. **High RSS Delta** (>50MB): Job is retaining memory after completion (potential leak)
2. **High Peak RSS** (>500MB): Job uses lots of memory during execution (potential OOM cause)
3. **Negative Delta**: Job cleaned up memory (good!)
4. **Positive Delta**: Job left memory allocated (investigate if large)

## OpenShift/RHEL Specific Considerations

### 1. Container Memory Limits

Check your pod's memory limits:

```bash
# In OpenShift
oc describe pod <pod-name> | grep -i memory
```

If your container has a 2GB limit and libreoffice peaks at 1.8GB, that's your culprit.

### 2. Accessing Reports in OpenShift

```python
# Save report to a mounted volume
profiler.save_report("/mnt/shared/memory_report.txt")

# Or log to stdout (captured by OpenShift logs)
profiler.print_report()
```

Then view logs:
```bash
oc logs <pod-name>
```

### 3. RHEL 8.10 Compatibility

The profiler uses only psutil, which is fully compatible with RHEL 8.10:

```bash
# In your Dockerfile or container
pip install psutil>=5.8.0
```

## Troubleshooting

### Issue: "No module named 'psutil'"

**Solution**: Add to your Dockerfile:
```dockerfile
RUN pip install psutil
```

### Issue: "Permission denied" when saving report

**Solution**: Use `/tmp` directory or a mounted volume:
```python
profiler.save_report("/tmp/memory_report.txt")
```

### Issue: Report shows all jobs with minimal memory delta

**Possible causes**:
1. Memory leak is in the leader process itself (not individual jobs)
2. OOM happens after all jobs complete (check container metrics)
3. Enable continuous monitoring to track peak usage:
   ```python
   profiler = MemoryProfiler(
    enable_continuous_monitoring=True,
    monitoring_interval=2.0,
    log_interval=30.0
)
   ```

### Issue: Want to track child processes

**Note**: The profiler tracks the leader process. If jobs spawn subprocesses (like libreoffice does), memory will be attributed to the parent job. This is usually what you want.

## Advanced Usage

### Custom Thresholds for Culprits

```python
# After running jobs
culprits = [
    s for s in profiler.stats
    if s.rss_delta_mb > 100 or (s.peak and s.peak.rss_mb > 1024)
]

if culprits:
    print("HIGH MEMORY JOBS DETECTED:")
    for c in culprits:
        print(f"  - {c.job_name}: {c.rss_delta_mb:.2f}MB delta, {c.peak.rss_mb:.2f}MB peak")
```

### Export Report as JSON

```python
import json

# Convert stats to dict
report_data = {
    "jobs": [
        {
            "name": s.job_name,
            "duration": s.duration_seconds,
            "rss_delta_mb": s.rss_delta_mb,
            "peak_rss_mb": s.peak.rss_mb if s.peak else 0,
            "success": s.success
        }
        for s in profiler.stats
    ]
}

with open("/tmp/memory_report.json", "w") as f:
    json.dump(report_data, f, indent=2)
```

### Monitor System Memory

```python
import psutil

before_jobs = psutil.virtual_memory()
print(f"System memory before jobs: {before_jobs.percent}% used")

# Run your jobs...

after_jobs = psutil.virtual_memory()
print(f"System memory after jobs: {after_jobs.percent}% used")
```

## Next Steps

1. **Integrate**: Add the profiler to your leader job
2. **Run**: Execute your normal job sequence in OpenShift
3. **Analyze**: Check the report for high RSS delta or peak RSS
4. **Fix**: Optimize the culprit job(s)
5. **Verify**: Re-run with profiler to confirm improvement

## Example Dockerfile Integration

```dockerfile
FROM registry.access.redhat.com/ubi8/python-38

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy profiler and jobs
COPY memory_profiler.py .
COPY leader_job.py .
COPY convert_to_markdown.py .
# ... other job scripts

# Run with profiling
CMD ["python", "leader_job.py"]
```

## Questions?

The profiler is designed to be drop-in compatible with your existing code. If you need help integrating it, share your current leader job structure and I can provide specific integration code.
