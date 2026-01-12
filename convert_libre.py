#!/usr/bin/env python3
"""
Memory-Optimized LibreOffice Document Converter
Uses memory_profiler for accurate memory tracking and OOM prevention.
"""

import os
import subprocess
import tempfile
import shutil
import signal
import logging
import time
import threading
import gc
import resource
from pathlib import Path
from typing import Optional, Tuple, List
from contextlib import contextmanager
import atexit
import hashlib
import psutil

# Import the memory profiler from the repo
from memory_profiler import MemoryProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessTracker:
    """
    Track LibreOffice processes for cleanup.
    Enhanced with memory monitoring.
    """

    def __init__(self):
        self.active_processes = {}  # {pid: (process_obj, profile_dir, start_time)}
        self.lock = threading.Lock()
        atexit.register(self.cleanup_all)

    def register(self, process, profile_dir):
        """Register a process for tracking."""
        with self.lock:
            self.active_processes[process.pid] = {
                'process': process,
                'profile_dir': profile_dir,
                'start_time': time.time()
            }

    def unregister(self, pid):
        """Unregister a process."""
        with self.lock:
            if pid in self.active_processes:
                del self.active_processes[pid]

    def cleanup_process(self, pid, timeout=5):
        """
        Cleanup a specific process and all its children.
        """
        with self.lock:
            if pid not in self.active_processes:
                return

            info = self.active_processes[pid]
            process = info['process']
            profile_dir = info['profile_dir']

        # Kill all child processes first
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Wait for children to terminate
            gone, alive = psutil.wait_procs(children, timeout=timeout)

            # Force kill remaining children
            for child in alive:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Try graceful termination of parent
        if process.poll() is None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=timeout)
                    logger.debug(f"Process {pid} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if still alive
                    process.kill()
                    process.wait(timeout=2)
                    logger.warning(f"Process {pid} force killed")
            except Exception as e:
                logger.error(f"Error terminating process {pid}: {e}")

        # Kill related processes by profile path
        self._kill_by_profile_path(profile_dir)

        self.unregister(pid)

    def _kill_by_profile_path(self, profile_dir):
        """Find and kill LibreOffice processes using the profile directory."""
        try:
            profile_basename = os.path.basename(profile_dir)
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline')
                    if cmdline and any(profile_basename in arg for arg in cmdline):
                        proc.terminate()
                        logger.debug(f"Killed related process: {proc.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.debug(f"Could not search for related processes: {e}")

    def cleanup_stale_processes(self, max_age_seconds=300):
        """Cleanup processes that have been running too long."""
        current_time = time.time()
        stale_pids = []

        with self.lock:
            for pid, info in self.active_processes.items():
                if current_time - info['start_time'] > max_age_seconds:
                    stale_pids.append(pid)

        for pid in stale_pids:
            logger.warning(f"Cleaning up stale process: {pid}")
            self.cleanup_process(pid)

    def cleanup_all(self):
        """Emergency cleanup of all tracked processes."""
        logger.info("Cleaning up all tracked LibreOffice processes")

        with self.lock:
            pids = list(self.active_processes.keys())

        for pid in pids:
            try:
                self.cleanup_process(pid, timeout=2)
            except Exception as e:
                logger.error(f"Error cleaning up process {pid}: {e}")


class LibreOfficeConverter:
    """
    Memory-optimized converter with memory_profiler integration.
    """

    def __init__(
        self,
        soffice_path: str = "soffice",
        timeout: int = 120,
        max_retries: int = 2,
        cleanup_interval: int = 300,
        max_memory_mb: int = 2048,
        memory_check_interval: float = 1.0,
        enable_continuous_monitoring: bool = True
    ):
        """
        Initialize converter.

        Args:
            soffice_path: Path to LibreOffice binary
            timeout: Conversion timeout in seconds
            max_retries: Number of retry attempts
            cleanup_interval: Cleanup stale processes after this many seconds
            max_memory_mb: Maximum memory allowed per conversion process
            memory_check_interval: How often to check memory usage
            enable_continuous_monitoring: Enable continuous memory monitoring
        """
        self.soffice_path = soffice_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.cleanup_interval = cleanup_interval
        self.max_memory_mb = max_memory_mb
        self.memory_check_interval = memory_check_interval
        self.process_tracker = ProcessTracker()

        # Initialize memory profiler with continuous monitoring
        self.memory_profiler = MemoryProfiler(
            enable_continuous_monitoring=enable_continuous_monitoring,
            monitoring_interval=memory_check_interval,
            log_interval=30.0  # Log every 30 seconds
        )

        # Verify LibreOffice
        self._verify_libreoffice()

        # Start background cleanup thread
        self._start_cleanup_thread()

    def _verify_libreoffice(self):
        """Verify LibreOffice is available."""
        try:
            result = subprocess.run(
                [self.soffice_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"LibreOffice found: {version}")
            else:
                raise RuntimeError("LibreOffice not responding")
        except FileNotFoundError:
            raise RuntimeError(
                f"LibreOffice not found at '{self.soffice_path}'. "
                "Please install LibreOffice or provide the correct path."
            )
        except Exception as e:
            logger.warning(f"Could not verify LibreOffice version: {e}")

    def _start_cleanup_thread(self):
        """Start background thread for cleaning up stale processes."""
        def cleanup_worker():
            while True:
                time.sleep(60)  # Check every minute
                try:
                    self.process_tracker.cleanup_stale_processes(
                        self.cleanup_interval
                    )
                    # Force garbage collection
                    gc.collect()
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")

        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
        logger.debug("Started background cleanup thread")

    @contextmanager
    def isolated_user_profile(self):
        """
        Create isolated LibreOffice user profile.
        Uses disk-based temp directory to avoid RAM pressure.
        """
        # ALWAYS use disk-based temp directory, never /dev/shm
        temp_base = tempfile.gettempdir()

        # Create unique profile directory
        profile_dir = tempfile.mkdtemp(
            prefix="lo_profile_",
            suffix=f"_{os.getpid()}",
            dir=temp_base
        )

        logger.debug(f"Created profile: {profile_dir}")

        try:
            yield profile_dir
        finally:
            # Cleanup profile directory with multiple attempts
            self._cleanup_directory(profile_dir)
            # Force garbage collection after profile cleanup
            gc.collect()

    def _cleanup_directory(self, directory, max_attempts=3):
        """Cleanup directory with retry logic."""
        for attempt in range(max_attempts):
            try:
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                    logger.debug(f"Cleaned up directory: {directory}")
                    return
            except PermissionError as e:
                if attempt < max_attempts - 1:
                    logger.debug(f"Cleanup attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.5)
                else:
                    logger.warning(f"Could not cleanup {directory}: {e}")
            except Exception as e:
                logger.warning(f"Error cleaning up {directory}: {e}")
                break

    @contextmanager
    def conversion_lock(self, file_path: str):
        """Simple file-based lock to prevent concurrent conversions of same file."""
        lock_name = hashlib.md5(file_path.encode()).hexdigest()
        lock_file = Path(tempfile.gettempdir()) / f".lo_lock_{lock_name}"

        # Wait for lock with timeout
        start_time = time.time()
        timeout = 30

        while lock_file.exists():
            if time.time() - start_time > timeout:
                try:
                    lock_file.unlink()
                    logger.warning(f"Removed stale lock: {lock_file}")
                except:
                    pass
                break
            time.sleep(0.5)

        # Create lock
        try:
            lock_file.touch()
            logger.debug(f"Acquired lock: {lock_file}")
        except Exception as e:
            logger.warning(f"Could not create lock file: {e}")

        try:
            yield
        finally:
            # Remove lock
            try:
                if lock_file.exists():
                    lock_file.unlink()
                    logger.debug(f"Released lock: {lock_file}")
            except Exception as e:
                logger.warning(f"Could not remove lock file: {e}")

    def _set_resource_limits(self):
        """
        Set resource limits for child process.
        This runs in the child process before exec.
        """
        try:
            # Limit virtual memory to max_memory_mb
            max_memory_bytes = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(
                resource.RLIMIT_AS,
                (max_memory_bytes, max_memory_bytes)
            )

            # Limit CPU time (soft limit: timeout, hard limit: timeout + 30s)
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.timeout, self.timeout + 30)
            )

            logger.debug(f"Set resource limits: memory={self.max_memory_mb}MB, cpu={self.timeout}s")
        except Exception as e:
            logger.warning(f"Could not set resource limits: {e}")

    def _build_conversion_command(
        self,
        input_file: str,
        output_dir: str,
        profile_dir: str
    ) -> List[str]:
        """Build LibreOffice conversion command with memory optimization flags."""
        return [
            self.soffice_path,
            "--headless",
            "--invisible",
            "--nocrashreport",
            "--nodefault",
            "--nofirststartwizard",
            "--nolockcheck",
            "--nologo",
            "--norestore",
            "--safe-mode",
            "--norestore",
            f"-env:UserInstallation=file://{profile_dir}",
            # Memory optimization flags
            "--writer",  # Use specific module to reduce memory
            "--convert-to", "pdf",
            "--outdir", output_dir,
            str(Path(input_file).absolute())
        ]

    def _monitor_conversion(
        self,
        process: subprocess.Popen,
        timeout: int,
        log_file_path: str
    ) -> Tuple[bool, str, int]:
        """
        Monitor conversion process with timeout and memory checks using psutil.
        Returns (success, error_message, returncode).
        """
        start_time = time.time()
        last_memory_mb = 0.0

        # Poll process with timeout and memory monitoring
        while process.poll() is None:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > timeout:
                logger.error(f"Process timeout after {timeout}s")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

                return False, f"Timeout after {timeout}s", -1

            # Check memory usage using psutil
            try:
                proc = psutil.Process(process.pid)
                mem_info = proc.memory_info()
                memory_mb = mem_info.rss / (1024 * 1024)

                # Log memory spikes
                if memory_mb > last_memory_mb + 100:
                    logger.debug(f"Memory usage: {memory_mb:.1f}MB")
                    last_memory_mb = memory_mb

                # Kill if exceeding memory limit
                if memory_mb > self.max_memory_mb:
                    logger.error(
                        f"Process exceeded memory limit: {memory_mb:.1f}MB > {self.max_memory_mb}MB"
                    )
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)

                    return False, f"Memory limit exceeded: {memory_mb:.1f}MB", -2

                # Check system memory
                system_mem = psutil.virtual_memory()
                available_mb = system_mem.available / (1024 * 1024)

                if available_mb < 500:
                    logger.error("System low on memory, terminating conversion")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)

                    return False, "System low on memory", -3

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            time.sleep(self.memory_check_interval)

        # Process completed
        returncode = process.returncode

        # Read error log if conversion failed
        error_msg = ""
        if returncode != 0:
            try:
                with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    error_output = f.read()
                    if error_output:
                        # Check for memory errors
                        error_lower = error_output.lower()
                        if "out of memory" in error_lower or "cannot allocate" in error_lower:
                            error_msg = "OUT OF MEMORY detected in LibreOffice logs"
                        else:
                            # Include first 300 chars
                            error_msg = error_output[:300]
            except Exception as e:
                logger.debug(f"Could not read error log: {e}")

        return True, error_msg, returncode

    def convert_to_pdf(
        self,
        input_file: str,
        output_dir: Optional[str] = None,
        attempt: int = 1
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Convert document to PDF with memory profiling.

        Args:
            input_file: Path to input file
            output_dir: Output directory (default: same as input)
            attempt: Current attempt number

        Returns:
            (success, message, output_path)
        """
        input_path = Path(input_file)

        # Validate input
        if not input_path.exists():
            return False, f"File not found: {input_file}", None

        if not input_path.is_file():
            return False, f"Not a file: {input_file}", None

        # Validate file type
        valid_extensions = {
            '.pptx', '.ppt', '.odp',    # PowerPoint
            '.xlsx', '.xls', '.ods',    # Excel
            '.docx', '.doc', '.odt',    # Word
        }

        if input_path.suffix.lower() not in valid_extensions:
            return False, f"Unsupported file type: {input_path.suffix}", None

        # Check file size
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            logger.warning(
                f"Large file: {file_size_mb:.1f}MB - "
                "conversion may take longer and use more memory"
            )

        # Check system memory before starting using psutil
        system_mem = psutil.virtual_memory()
        available_mb = system_mem.available / (1024 * 1024)
        if available_mb < 1000:
            logger.warning(
                f"Low system memory: {available_mb:.1f}MB available. "
                "Conversion may fail."
            )

        # Setup output directory
        if output_dir is None:
            output_dir = str(input_path.parent)
        else:
            os.makedirs(output_dir, exist_ok=True)

        output_filename = input_path.stem + ".pdf"
        output_path = Path(output_dir) / output_filename

        # Remove existing output
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception as e:
                logger.warning(f"Could not remove existing output: {e}")

        # Create temp file for subprocess output
        log_fd, log_file_path = tempfile.mkstemp(prefix="lo_log_", suffix=".txt")
        os.close(log_fd)

        # Wrap conversion with memory profiler
        def do_conversion():
            # Prevent concurrent conversions of same file
            with self.conversion_lock(str(input_path)):
                # Use isolated profile
                with self.isolated_user_profile() as profile_dir:
                    process = None

                    try:
                        # Build command
                        command = self._build_conversion_command(
                            str(input_path),
                            output_dir,
                            profile_dir
                        )

                        logger.info(
                            f"[Attempt {attempt}/{self.max_retries}] "
                            f"Converting {input_path.name} ({file_size_mb:.1f}MB)..."
                        )
                        logger.debug(f"Command: {' '.join(command)}")

                        # Set environment for the subprocess
                        env = os.environ.copy()
                        env['HOME'] = profile_dir

                        # Open log file for writing
                        log_file = open(log_file_path, 'w', encoding='utf-8')

                        # Start conversion with resource limits
                        start_time = time.time()

                        # On Unix systems, set resource limits via preexec_fn
                        preexec_fn = None
                        if os.name != 'nt':  # Not Windows
                            preexec_fn = self._set_resource_limits

                        process = subprocess.Popen(
                            command,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            env=env,
                            preexec_fn=preexec_fn
                        )

                        log_file.close()

                        # Register for tracking
                        self.process_tracker.register(process, profile_dir)

                        # Monitor with timeout and memory checks
                        success, error_msg, returncode = self._monitor_conversion(
                            process,
                            self.timeout,
                            log_file_path
                        )

                        elapsed = time.time() - start_time

                        # Cleanup process immediately
                        if process and process.pid:
                            self.process_tracker.cleanup_process(process.pid)

                        # Force garbage collection
                        gc.collect()

                        # Check success
                        if success and returncode == 0 and output_path.exists():
                            output_size_mb = output_path.stat().st_size / (1024 * 1024)
                            logger.info(
                                f"✓ Converted {input_path.name} to PDF "
                                f"({output_size_mb:.1f}MB) in {elapsed:.1f}s"
                            )

                            # Cleanup log file
                            try:
                                os.unlink(log_file_path)
                            except:
                                pass

                            return True, "Success", str(output_path)

                        # Handle failures
                        if not success:
                            error_message = error_msg or f"Conversion failed (code: {returncode})"
                        else:
                            error_message = f"Conversion failed (exit code: {returncode})"
                            if error_msg:
                                error_message += f"\n{error_msg}"

                        logger.error(error_message)

                        # Cleanup log file
                        try:
                            os.unlink(log_file_path)
                        except:
                            pass

                        return False, error_message, None

                    except Exception as e:
                        error_msg = f"Unexpected error: {str(e)}"
                        logger.error(error_msg, exc_info=True)

                        # Cleanup
                        if process and process.pid:
                            self.process_tracker.cleanup_process(process.pid)

                        gc.collect()

                        # Cleanup log file
                        try:
                            os.unlink(log_file_path)
                        except:
                            pass

                        raise  # Re-raise to be caught by profiler

        # Use memory profiler to track the conversion
        job_name = f"convert_{input_path.name}"

        try:
            result = self.memory_profiler.run_function_job(
                job_name=job_name,
                func=do_conversion
            )

            # If successful, return the result
            if result and result[0]:
                return result

            # If failed but not from exception, retry
            if attempt < self.max_retries and result and result[1] != "Memory limit exceeded":
                logger.info(f"Retrying (attempt {attempt + 1})...")
                gc.collect()
                time.sleep(3)
                return self.convert_to_pdf(input_file, output_dir, attempt + 1)

            return result if result else (False, "Unknown error", None)

        except Exception as e:
            # Exception during conversion
            if attempt < self.max_retries:
                logger.info(f"Retrying after error (attempt {attempt + 1})...")
                gc.collect()
                time.sleep(3)
                return self.convert_to_pdf(input_file, output_dir, attempt + 1)

            return False, f"Conversion failed: {str(e)}", None

    def convert_powerpoint_to_pdf(
        self,
        pptx_file: str,
        output_dir: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """Convert PowerPoint file to PDF."""
        return self.convert_to_pdf(pptx_file, output_dir)

    def convert_excel_to_pdf(
        self,
        xlsx_file: str,
        output_dir: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """Convert Excel file to PDF."""
        return self.convert_to_pdf(xlsx_file, output_dir)

    def batch_convert(
        self,
        input_files: List[str],
        output_dir: Optional[str] = None
    ) -> dict:
        """
        Convert multiple files sequentially with memory profiling.

        Args:
            input_files: List of file paths
            output_dir: Output directory

        Returns:
            Dictionary with results for each file
        """
        results = {}

        logger.info(f"Starting batch conversion of {len(input_files)} files")

        for idx, input_file in enumerate(input_files, 1):
            logger.info(f"Processing file {idx}/{len(input_files)}: {Path(input_file).name}")

            # Check system memory before each conversion
            system_mem = psutil.virtual_memory()
            available_mb = system_mem.available / (1024 * 1024)
            logger.debug(f"Available memory: {available_mb:.1f}MB")

            if available_mb < 500:
                logger.error("Insufficient memory for conversion, stopping batch")
                results[input_file] = {
                    'success': False,
                    'message': 'Insufficient system memory',
                    'output_path': None
                }
                break

            success, message, output_path = self.convert_to_pdf(
                input_file,
                output_dir
            )

            results[input_file] = {
                'success': success,
                'message': message,
                'output_path': output_path
            }

            # Aggressive cleanup between conversions
            gc.collect()

            # Longer delay between conversions to let system recover
            if idx < len(input_files):
                time.sleep(2)

        # Summary
        successful = sum(1 for r in results.values() if r['success'])
        logger.info(
            f"Batch conversion complete: {successful}/{len(input_files)} successful"
        )

        return results


# Configuration
if __name__ == "__main__":
    # ============================================================================
    # CONFIGURATION - Modify these values as needed
    # ============================================================================

    # Memory limit: 1000MB (1GB)
    MAX_MEMORY_MB = 1000

    # Timeout per conversion: 120 seconds (2 minutes)
    TIMEOUT_SECONDS = 120

    # Max retries per file: 2
    MAX_RETRIES = 2

    # Input files to convert (add your files here)
    INPUT_FILES = [
        "sample.pptx",
        "document.docx",
        "spreadsheet.xlsx"
    ]

    # Output directory (None = same as input file location)
    OUTPUT_DIR = None  # or set to "output/" for a specific directory

    # ============================================================================
    # END CONFIGURATION
    # ============================================================================

    # Check psutil availability
    try:
        import psutil
    except ImportError:
        logger.error("psutil module required. Install with: pip install psutil")
        exit(1)

    # Initialize converter with fixed configuration
    converter = LibreOfficeConverter(
        timeout=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        max_memory_mb=MAX_MEMORY_MB,
        enable_continuous_monitoring=True
    )

    logger.info(f"Configuration: Memory Limit={MAX_MEMORY_MB}MB, Timeout={TIMEOUT_SECONDS}s, Retries={MAX_RETRIES}")

    # Convert files
    if len(INPUT_FILES) == 1:
        # Single file
        success, message, output_path = converter.convert_to_pdf(
            INPUT_FILES[0],
            OUTPUT_DIR
        )

        print(f"\n{'='*60}")
        print(f"Status: {'SUCCESS' if success else 'FAILED'}")
        print(f"Message: {message}")
        if output_path:
            print(f"Output: {output_path}")
        print('='*60)

        exit(0 if success else 1)
    else:
        # Batch conversion
        results = converter.batch_convert(
            INPUT_FILES,
            OUTPUT_DIR
        )

        # Print results
        print(f"\n{'='*60}")
        print("BATCH CONVERSION RESULTS")
        print('='*60)

        for file_path, result in results.items():
            status = "✓" if result['success'] else "✗"
            print(f"\n{status} {Path(file_path).name}")
            print(f"  {result['message']}")
            if result['output_path']:
                print(f"  → {result['output_path']}")

        successful = sum(1 for r in results.values() if r['success'])
        print(f"\n{'='*60}")
        print(f"Total: {len(results)} | Success: {successful} | Failed: {len(results) - successful}")
        print('='*60)

        exit(0 if successful == len(results) else 1)
