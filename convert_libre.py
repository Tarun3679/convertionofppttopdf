#!/usr/bin/env python3
"""
Production-Grade LibreOffice Document Converter
Works with standard user permissions, no root/proc access required.
Addresses OOM, memory leaks, and process management issues.
"""

import os
import subprocess
import tempfile
import shutil
import signal
import logging
import time
import threading
from pathlib import Path
from typing import Optional, Tuple, List
from contextlib import contextmanager
import atexit
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessTracker:
    """
    Track LibreOffice processes for cleanup.
    Works without proc access by tracking PIDs and profile paths.
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
        Cleanup a specific process.
        Uses standard signals that work with user permissions.
        """
        with self.lock:
            if pid not in self.active_processes:
                return
            
            info = self.active_processes[pid]
            process = info['process']
            profile_dir = info['profile_dir']
        
        # Try graceful termination
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
        """
        Find and kill LibreOffice processes using the profile directory.
        Uses pgrep which works with user permissions.
        """
        try:
            # Search for soffice processes with this profile in command line
            result = subprocess.run(
                ["pgrep", "-f", f"UserInstallation.*{os.path.basename(profile_dir)}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid_str in pids:
                    try:
                        pid = int(pid_str)
                        os.kill(pid, signal.SIGTERM)
                        logger.debug(f"Killed related process: {pid}")
                    except (ValueError, ProcessLookupError, PermissionError):
                        pass
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # pgrep not available or timeout
            pass
        except Exception as e:
            logger.debug(f"Could not search for related processes: {e}")
    
    def cleanup_stale_processes(self, max_age_seconds=300):
        """
        Cleanup processes that have been running too long.
        Safety mechanism for stuck conversions.
        """
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
    Production-grade converter optimized for standard user permissions.
    """
    
    def __init__(
        self,
        soffice_path: str = "soffice",
        timeout: int = 120,
        max_retries: int = 2,
        cleanup_interval: int = 300
    ):
        """
        Initialize converter.
        
        Args:
            soffice_path: Path to LibreOffice binary
            timeout: Conversion timeout in seconds
            max_retries: Number of retry attempts
            cleanup_interval: Cleanup stale processes after this many seconds
        """
        self.soffice_path = soffice_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.cleanup_interval = cleanup_interval
        self.process_tracker = ProcessTracker()
        
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
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
        logger.debug("Started background cleanup thread")
    
    @contextmanager
    def isolated_user_profile(self):
        """
        Create isolated LibreOffice user profile.
        Uses temp directory with unique naming to prevent conflicts.
        """
        # Try to use /dev/shm if available (faster, RAM-based)
        temp_base = None
        for candidate in ["/dev/shm", "/tmp", tempfile.gettempdir()]:
            if os.path.exists(candidate) and os.access(candidate, os.W_OK):
                temp_base = candidate
                break
        
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
    
    def _cleanup_directory(self, directory, max_attempts=3):
        """
        Cleanup directory with retry logic.
        Sometimes files are locked by LibreOffice temporarily.
        """
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
        """
        Simple file-based lock to prevent concurrent conversions of same file.
        Uses standard file operations available to all users.
        """
        # Create lock file with hash of original path
        lock_name = hashlib.md5(file_path.encode()).hexdigest()
        lock_file = Path(tempfile.gettempdir()) / f".lo_lock_{lock_name}"
        
        # Wait for lock with timeout
        start_time = time.time()
        timeout = 30
        
        while lock_file.exists():
            if time.time() - start_time > timeout:
                # Force remove stale lock
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
    
    def _build_conversion_command(
        self,
        input_file: str,
        output_dir: str,
        profile_dir: str
    ) -> List[str]:
        """
        Build LibreOffice conversion command with all critical flags.
        """
        return [
            self.soffice_path,
            "--headless",
            "--invisible",
            "--nocrashreport",
            "--nodefault",
            "--nofirststartwizard",
            "--nolockcheck",           # Critical: prevents lock file issues
            "--nologo",
            "--norestore",
            "--safe-mode",             # Minimal extensions
            f"-env:UserInstallation=file://{profile_dir}",
            "--convert-to", "pdf",
            "--outdir", output_dir,
            str(Path(input_file).absolute())
        ]
    
    def _monitor_conversion(
        self,
        process: subprocess.Popen,
        timeout: int
    ) -> Tuple[Optional[str], Optional[str], int]:
        """
        Monitor conversion process with timeout.
        Returns (stdout, stderr, returncode).
        """
        start_time = time.time()
        
        # Poll process with timeout
        while process.poll() is None:
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                logger.error(f"Process timeout after {timeout}s")
                # Kill process
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                
                return None, f"Timeout after {timeout}s", -1
            
            time.sleep(0.5)
        
        # Get output
        try:
            stdout, stderr = process.communicate(timeout=5)
            return stdout, stderr, process.returncode
        except subprocess.TimeoutExpired:
            return None, "Could not get process output", process.returncode
    
    def convert_to_pdf(
        self,
        input_file: str,
        output_dir: Optional[str] = None,
        attempt: int = 1
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Convert document to PDF with comprehensive error handling.
        
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
                "conversion may take longer"
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
                    env['HOME'] = profile_dir  # Some versions need this
                    
                    # Start conversion
                    start_time = time.time()
                    
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env
                    )
                    
                    # Register for tracking
                    self.process_tracker.register(process, profile_dir)
                    
                    # Monitor with timeout
                    stdout, stderr, returncode = self._monitor_conversion(
                        process,
                        self.timeout
                    )
                    
                    elapsed = time.time() - start_time
                    
                    # Check success
                    if returncode == 0 and output_path.exists():
                        output_size_mb = output_path.stat().st_size / (1024 * 1024)
                        logger.info(
                            f"✓ Converted {input_path.name} to PDF "
                            f"({output_size_mb:.1f}MB) in {elapsed:.1f}s"
                        )
                        return True, "Success", str(output_path)
                    
                    # Handle failures
                    if returncode == -1:
                        error_msg = "Conversion timeout"
                    else:
                        error_msg = f"Conversion failed (exit code: {returncode})"
                        
                        if stderr:
                            # Check for specific errors
                            stderr_lower = stderr.lower()
                            if "out of memory" in stderr_lower:
                                error_msg += " - OUT OF MEMORY"
                            elif "xml" in stderr_lower and "memory" in stderr_lower:
                                error_msg += " - XML parsing memory error"
                            
                            # Include first 300 chars of error
                            error_msg += f"\n{stderr[:300]}"
                    
                    logger.error(error_msg)
                    
                    # Retry logic
                    if attempt < self.max_retries:
                        logger.info(f"Retrying (attempt {attempt + 1})...")
                        time.sleep(2)  # Brief delay before retry
                        return self.convert_to_pdf(
                            input_file,
                            output_dir,
                            attempt + 1
                        )
                    
                    return False, error_msg, None
                
                except Exception as e:
                    error_msg = f"Unexpected error: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    
                    # Retry on exceptions
                    if attempt < self.max_retries:
                        logger.info(f"Retrying after error (attempt {attempt + 1})...")
                        time.sleep(2)
                        return self.convert_to_pdf(
                            input_file,
                            output_dir,
                            attempt + 1
                        )
                    
                    return False, error_msg, None
                
                finally:
                    # Cleanup process
                    if process and process.pid:
                        self.process_tracker.cleanup_process(process.pid)
    
    def convert_powerpoint_to_pdf(
        self,
        pptx_file: str,
        output_dir: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Convert PowerPoint file to PDF.
        
        Args:
            pptx_file: Path to PowerPoint file (.pptx, .ppt, .odp)
            output_dir: Output directory
            
        Returns:
            (success, message, output_path)
        """
        return self.convert_to_pdf(pptx_file, output_dir)
    
    def convert_excel_to_pdf(
        self,
        xlsx_file: str,
        output_dir: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Convert Excel file to PDF.
        
        Args:
            xlsx_file: Path to Excel file (.xlsx, .xls, .ods)
            output_dir: Output directory
            
        Returns:
            (success, message, output_path)
        """
        return self.convert_to_pdf(xlsx_file, output_dir)
    
    def batch_convert(
        self,
        input_files: List[str],
        output_dir: Optional[str] = None
    ) -> dict:
        """
        Convert multiple files sequentially.
        Sequential processing is safer than parallel for LibreOffice.
        
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
            
            success, message, output_path = self.convert_to_pdf(
                input_file,
                output_dir
            )
            
            results[input_file] = {
                'success': success,
                'message': message,
                'output_path': output_path
            }
            
            # Small delay between conversions to let system cleanup
            if idx < len(input_files):
                time.sleep(1)
        
        # Summary
        successful = sum(1 for r in results.values() if r['success'])
        logger.info(
            f"Batch conversion complete: {successful}/{len(input_files)} successful"
        )
        
        return results


# Example usage and CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert Office documents to PDF using LibreOffice"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input file(s) to convert"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=120,
        help="Conversion timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retry attempts (default: 2)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize converter
    converter = LibreOfficeConverter(
        timeout=args.timeout,
        max_retries=args.max_retries
    )
    
    # Convert
    if len(args.input_files) == 1:
        # Single file
        success, message, output_path = converter.convert_to_pdf(
            args.input_files[0],
            args.output_dir
        )
        
        print(f"\n{'='*60}")
        print(f"Status: {'SUCCESS' if success else 'FAILED'}")
        print(f"Message: {message}")
        if output_path:
            print(f"Output: {output_path}")
        print('='*60)
        
        exit(0 if success else 1)
    else:
        # Batch
        results = converter.batch_convert(
            args.input_files,
            args.output_dir
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
