#!/usr/bin/env python3
"""
Memory-Optimized LibreOffice Document Converter
Uses memory_profiler for accurate memory tracking and OOM prevention.
Includes external link detection and sanitization to prevent OOM from problematic PowerPoints.
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
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
import atexit
import hashlib
import psutil

from memory_profiler import MemoryProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskLevel(IntEnum):
    """Risk levels for files with external links."""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class LinkScanResult:
    """Result of scanning a file for external links."""
    risk_level: RiskLevel
    external_links: List[str]
    link_types: List[str]
    is_sanitizable: bool
    file_size_mb: float


@dataclass
class SanitizationResult:
    """Result of sanitizing a file."""
    success: bool
    sanitized_path: Optional[str]
    operations_performed: List[str]
    size_before_mb: float
    size_after_mb: float
    error_message: Optional[str] = None


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


class ExternalLinkDetector:
    """
    Detects external links in PowerPoint files by inspecting PPTX XML structure.
    Fast and memory-efficient - no LibreOffice loading required.
    """

    def scan_pptx_file(self, file_path: str) -> LinkScanResult:
        """Scan a PPTX file for external links and assess risk level."""
        try:
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

            if not zipfile.is_zipfile(file_path):
                return LinkScanResult(
                    risk_level=RiskLevel.SAFE,
                    external_links=[],
                    link_types=[],
                    is_sanitizable=True,
                    file_size_mb=file_size_mb
                )

            external_links = []
            link_types = []

            with zipfile.ZipFile(file_path, 'r') as pptx_zip:
                ole_objects = self._find_ole_objects(pptx_zip)
                chart_links = self._find_chart_external_links(pptx_zip)
                external_rels = self._find_external_relationships(pptx_zip)

                if ole_objects:
                    external_links.extend(ole_objects)
                    link_types.append('ole')

                if chart_links:
                    external_links.extend(chart_links)
                    link_types.append('chart')

                if external_rels:
                    external_links.extend(external_rels)
                    link_types.append('external_rel')

            risk_level = self._assess_risk_level(link_types, external_links)
            is_sanitizable = risk_level != RiskLevel.CRITICAL

            return LinkScanResult(
                risk_level=risk_level,
                external_links=external_links,
                link_types=link_types,
                is_sanitizable=is_sanitizable,
                file_size_mb=file_size_mb
            )

        except Exception as e:
            logger.warning(f"Error scanning {file_path}: {e}")
            return LinkScanResult(
                risk_level=RiskLevel.SAFE,
                external_links=[],
                link_types=[],
                is_sanitizable=True,
                file_size_mb=0.0
            )

    def _find_ole_objects(self, pptx_zip: zipfile.ZipFile) -> List[str]:
        """Find OLE objects embedded in slides."""
        ole_objects = []

        for file_name in pptx_zip.namelist():
            if file_name.startswith('ppt/slides/slide') and file_name.endswith('.xml'):
                try:
                    content = pptx_zip.read(file_name).decode('utf-8')
                    if '<p:oleObj' in content or 'oleObject' in content:
                        ole_objects.append(f"OLE in {file_name}")
                except Exception:
                    pass

        return ole_objects

    def _find_chart_external_links(self, pptx_zip: zipfile.ZipFile) -> List[str]:
        """Find charts with external data links."""
        chart_links = []

        for file_name in pptx_zip.namelist():
            if file_name.startswith('ppt/charts/') and file_name.endswith('.xml'):
                try:
                    content = pptx_zip.read(file_name).decode('utf-8')
                    if '<c:externalData' in content:
                        chart_links.append(f"External data in {file_name}")
                except Exception:
                    pass

        return chart_links

    def _find_external_relationships(self, pptx_zip: zipfile.ZipFile) -> List[str]:
        """Find external relationships in .rels files."""
        external_rels = []

        for file_name in pptx_zip.namelist():
            if file_name.endswith('.rels'):
                try:
                    content = pptx_zip.read(file_name).decode('utf-8')

                    if 'TargetMode="External"' in content:
                        if 'Target="http://' in content or 'Target="https://' in content:
                            external_rels.append(f"Network link in {file_name}")
                        elif 'Target="file://' in content:
                            external_rels.append(f"File link in {file_name}")
                except Exception:
                    pass

        return external_rels

    def _assess_risk_level(self, link_types: List[str], external_links: List[str]) -> RiskLevel:
        """Assess risk level based on detected link types."""
        if not link_types:
            return RiskLevel.SAFE

        if 'ole' in link_types:
            return RiskLevel.HIGH

        if 'chart' in link_types and len(external_links) > 3:
            return RiskLevel.HIGH

        if 'chart' in link_types:
            return RiskLevel.MEDIUM

        if 'external_rel' in link_types:
            network_links = [link for link in external_links if 'Network link' in link]
            if network_links:
                return RiskLevel.MEDIUM

        return RiskLevel.LOW


class LinkSanitizer:
    """
    Sanitizes PowerPoint files by removing or disabling external links.
    Works by modifying PPTX XML structure directly.
    """

    def sanitize_pptx(self, input_path: str, output_path: str) -> SanitizationResult:
        """Create a sanitized copy of the PPTX file."""
        size_before_mb = Path(input_path).stat().st_size / (1024 * 1024)
        operations_performed = []

        try:
            with zipfile.ZipFile(input_path, 'r') as input_zip:
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as output_zip:
                    for item in input_zip.namelist():
                        content = input_zip.read(item)

                        if item.endswith('.xml'):
                            modified_content = self._sanitize_xml_content(item, content, operations_performed)
                            output_zip.writestr(item, modified_content)
                        else:
                            output_zip.writestr(item, content)

            size_after_mb = Path(output_path).stat().st_size / (1024 * 1024)

            return SanitizationResult(
                success=True,
                sanitized_path=output_path,
                operations_performed=operations_performed,
                size_before_mb=size_before_mb,
                size_after_mb=size_after_mb
            )

        except Exception as e:
            logger.error(f"Sanitization failed: {e}")
            return SanitizationResult(
                success=False,
                sanitized_path=None,
                operations_performed=operations_performed,
                size_before_mb=size_before_mb,
                size_after_mb=0.0,
                error_message=str(e)
            )

    def _sanitize_xml_content(self, file_name: str, content: bytes, operations: List[str]) -> bytes:
        """Sanitize XML content by removing problematic elements."""
        try:
            content_str = content.decode('utf-8')

            if file_name.startswith('ppt/slides/') and '<p:oleObj' in content_str:
                content_str = self._remove_ole_objects(content_str)
                operations.append(f"Removed OLE from {file_name}")

            if file_name.startswith('ppt/charts/') and '<c:externalData' in content_str:
                content_str = self._break_chart_links(content_str)
                operations.append(f"Broke chart links in {file_name}")

            if file_name.endswith('.rels') and 'TargetMode="External"' in content_str:
                content_str = self._remove_external_relationships(content_str)
                operations.append(f"Removed external rels in {file_name}")

            return content_str.encode('utf-8')

        except Exception as e:
            logger.debug(f"Could not sanitize {file_name}: {e}")
            return content

    def _remove_ole_objects(self, xml_content: str) -> str:
        """Remove OLE object elements from XML."""
        try:
            root = ET.fromstring(xml_content)
            namespaces = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}

            for ole_obj in root.findall('.//p:oleObj', namespaces):
                parent = root.find('.//*[p:oleObj]', namespaces)
                if parent is not None:
                    parent.remove(ole_obj)

            return ET.tostring(root, encoding='unicode')
        except Exception:
            pattern_start = '<p:oleObj'
            pattern_end = '</p:oleObj>'

            while pattern_start in xml_content:
                start_idx = xml_content.find(pattern_start)
                end_idx = xml_content.find(pattern_end, start_idx)

                if end_idx == -1:
                    break

                xml_content = xml_content[:start_idx] + xml_content[end_idx + len(pattern_end):]

            return xml_content

    def _break_chart_links(self, xml_content: str) -> str:
        """Remove external data references from charts."""
        patterns_to_remove = [
            ('<c:externalData', '</c:externalData>'),
            ('<c:autoUpdate', '/>'),
        ]

        for start_pattern, end_pattern in patterns_to_remove:
            while start_pattern in xml_content:
                start_idx = xml_content.find(start_pattern)
                end_idx = xml_content.find(end_pattern, start_idx)

                if end_idx == -1:
                    break

                xml_content = xml_content[:start_idx] + xml_content[end_idx + len(end_pattern):]

        return xml_content

    def _remove_external_relationships(self, xml_content: str) -> str:
        """Remove external relationships from .rels files."""
        try:
            root = ET.fromstring(xml_content)

            for relationship in root.findall('.//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'):
                target_mode = relationship.get('TargetMode')
                if target_mode == 'External':
                    root.remove(relationship)

            return ET.tostring(root, encoding='unicode')
        except Exception:
            return xml_content


def _find_libreoffice_path() -> str:
    """
    Find LibreOffice installation path based on operating system.

    Returns:
        Path to LibreOffice executable, or 'soffice' as fallback
    """
    import platform

    system = platform.system()

    if system == "Windows":
        possible_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            r"C:\Program Files\LibreOffice 7\program\soffice.exe",
            r"C:\Program Files\LibreOffice 6\program\soffice.exe",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found LibreOffice at: {path}")
                return path

    elif system == "Linux":
        possible_paths = [
            "/usr/bin/soffice",
            "/usr/bin/libreoffice",
            "/usr/local/bin/soffice",
            "/opt/libreoffice/program/soffice",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found LibreOffice at: {path}")
                return path

    elif system == "Darwin":
        possible_paths = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found LibreOffice at: {path}")
                return path

    logger.warning("LibreOffice path not found, using fallback 'soffice'")
    return "soffice"


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
        enable_continuous_monitoring: bool = True,
        enable_link_detection: bool = True,
        enable_auto_sanitization: bool = True
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
            enable_link_detection: Enable external link detection
            enable_auto_sanitization: Enable automatic sanitization of risky files
        """
        self.soffice_path = soffice_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.cleanup_interval = cleanup_interval
        self.max_memory_mb = max_memory_mb
        self.memory_check_interval = memory_check_interval
        self.enable_link_detection = enable_link_detection
        self.enable_auto_sanitization = enable_auto_sanitization
        self.process_tracker = ProcessTracker()

        # Initialize memory profiler with continuous monitoring
        self.memory_profiler = MemoryProfiler(
            enable_continuous_monitoring=enable_continuous_monitoring,
            monitoring_interval=memory_check_interval,
            log_interval=30.0
        )

        # Initialize link detector and sanitizer
        self.link_detector = ExternalLinkDetector()
        self.link_sanitizer = LinkSanitizer()

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

        # Scan for external links (PowerPoint files only)
        sanitized_file_path = None
        file_to_convert = str(input_path)

        if self.enable_link_detection and input_path.suffix.lower() in ['.pptx', '.ppt']:
            scan_result = self.link_detector.scan_pptx_file(str(input_path))

            logger.info(
                f"[LINK_SCAN] file='{input_path.name}' risk={scan_result.risk_level.name} "
                f"links_found={len(scan_result.external_links)} types={scan_result.link_types}"
            )

            if scan_result.risk_level >= RiskLevel.MEDIUM:
                if self.enable_auto_sanitization and scan_result.is_sanitizable:
                    logger.info(f"[SANITIZE] Attempting to sanitize '{input_path.name}'")

                    sanitized_fd, sanitized_file_path = tempfile.mkstemp(
                        prefix="sanitized_",
                        suffix=".pptx"
                    )
                    os.close(sanitized_fd)

                    sanitization_result = self.link_sanitizer.sanitize_pptx(
                        str(input_path),
                        sanitized_file_path
                    )

                    if sanitization_result.success:
                        logger.info(
                            f"[SANITIZE] Success - operations: {sanitization_result.operations_performed}, "
                            f"size: {sanitization_result.size_before_mb:.1f}MB -> "
                            f"{sanitization_result.size_after_mb:.1f}MB"
                        )
                        file_to_convert = sanitized_file_path
                    else:
                        logger.warning(
                            f"[SANITIZE] Failed: {sanitization_result.error_message}"
                        )

                        if scan_result.risk_level == RiskLevel.CRITICAL:
                            if sanitized_file_path:
                                try:
                                    os.unlink(sanitized_file_path)
                                except:
                                    pass
                            return False, f"Critical risk file - sanitization failed: {sanitization_result.error_message}", None
                else:
                    if scan_result.risk_level == RiskLevel.CRITICAL:
                        return False, f"Critical risk file with external links - skipping to prevent OOM", None

                    logger.warning(
                        f"[RISK] File has {scan_result.risk_level.name} risk but proceeding without sanitization"
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
                        # Build command using the sanitized file if available
                        command = self._build_conversion_command(
                            file_to_convert,
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

            # Cleanup sanitized file if it was created
            if sanitized_file_path:
                try:
                    os.unlink(sanitized_file_path)
                    logger.debug(f"Cleaned up sanitized file: {sanitized_file_path}")
                except Exception as e:
                    logger.debug(f"Could not cleanup sanitized file: {e}")

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
            # Cleanup sanitized file if it was created
            if sanitized_file_path:
                try:
                    os.unlink(sanitized_file_path)
                except:
                    pass

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

    def _convert_powerpoint_to_pdf_bytes(
        self,
        document_bytes: bytes,
        temp_input_path: str,
        temp_output_dir: str
    ) -> bytes:
        """
        Convert PowerPoint bytes to PDF bytes.

        Args:
            document_bytes: PowerPoint file content as bytes
            temp_input_path: Temporary path for input file
            temp_output_dir: Temporary directory for output

        Returns:
            PDF content as bytes

        Raises:
            Exception: If conversion fails
        """
        try:
            with open(temp_input_path, 'wb') as f:
                f.write(document_bytes)

            success, message, output_path = self.convert_to_pdf(
                temp_input_path,
                temp_output_dir
            )

            if not success or not output_path:
                raise RuntimeError(f"PowerPoint conversion failed: {message}")

            with open(output_path, 'rb') as f:
                pdf_bytes = f.read()

            return pdf_bytes

        finally:
            if os.path.exists(temp_input_path):
                try:
                    os.unlink(temp_input_path)
                except:
                    pass

    def _convert_excel_to_pdf_bytes(
        self,
        document_bytes: bytes,
        temp_input_path: str,
        temp_output_dir: str
    ) -> bytes:
        """
        Convert Excel bytes to PDF bytes.

        Args:
            document_bytes: Excel file content as bytes
            temp_input_path: Temporary path for input file
            temp_output_dir: Temporary directory for output

        Returns:
            PDF content as bytes

        Raises:
            Exception: If conversion fails
        """
        try:
            with open(temp_input_path, 'wb') as f:
                f.write(document_bytes)

            success, message, output_path = self.convert_to_pdf(
                temp_input_path,
                temp_output_dir
            )

            if not success or not output_path:
                raise RuntimeError(f"Excel conversion failed: {message}")

            with open(output_path, 'rb') as f:
                pdf_bytes = f.read()

            return pdf_bytes

        finally:
            if os.path.exists(temp_input_path):
                try:
                    os.unlink(temp_input_path)
                except:
                    pass

    def convert_office_to_pdf_bytes(
        self,
        document_bytes: bytes,
        office_type: str
    ) -> bytes:
        """
        Convert Office document bytes to PDF bytes.

        Args:
            document_bytes: Office document content as bytes
            office_type: Type of office document ('powerpoint', 'excel', 'word')

        Returns:
            PDF content as bytes

        Raises:
            ValueError: If office_type is not supported
            RuntimeError: If conversion fails
        """
        office_type_lower = office_type.lower()

        extension_map = {
            'powerpoint': '.pptx',
            'ppt': '.pptx',
            'pptx': '.pptx',
            'excel': '.xlsx',
            'xls': '.xlsx',
            'xlsx': '.xlsx',
            'word': '.docx',
            'doc': '.docx',
            'docx': '.docx',
        }

        if office_type_lower not in extension_map:
            raise ValueError(
                f"Unsupported office type: {office_type}. "
                f"Supported types: {list(extension_map.keys())}"
            )

        extension = extension_map[office_type_lower]
        temp_input_fd, temp_input_path = tempfile.mkstemp(suffix=extension)
        os.close(temp_input_fd)

        temp_output_dir = tempfile.mkdtemp(prefix="pdf_output_")

        try:
            if office_type_lower in ['powerpoint', 'ppt', 'pptx']:
                pdf_bytes = self._convert_powerpoint_to_pdf_bytes(
                    document_bytes,
                    temp_input_path,
                    temp_output_dir
                )

            elif office_type_lower in ['excel', 'xls', 'xlsx']:
                pdf_bytes = self._convert_excel_to_pdf_bytes(
                    document_bytes,
                    temp_input_path,
                    temp_output_dir
                )

            elif office_type_lower in ['word', 'doc', 'docx']:
                with open(temp_input_path, 'wb') as f:
                    f.write(document_bytes)

                success, message, output_path = self.convert_to_pdf(
                    temp_input_path,
                    temp_output_dir
                )

                if not success or not output_path:
                    raise RuntimeError(f"Word conversion failed: {message}")

                with open(output_path, 'rb') as f:
                    pdf_bytes = f.read()
            else:
                raise ValueError(f"Unsupported office type: {office_type}")

            logger.info(f"Successfully converted {office_type} bytes to PDF ({len(pdf_bytes)} bytes)")
            return pdf_bytes

        finally:
            if os.path.exists(temp_input_path):
                try:
                    os.unlink(temp_input_path)
                except:
                    pass

            if os.path.exists(temp_output_dir):
                try:
                    shutil.rmtree(temp_output_dir)
                except:
                    pass

            gc.collect()


# Example usage
if __name__ == "__main__":
    # Example: Convert a file
    converter = LibreOfficeConverter(
        soffice_path=_find_libreoffice_path(),
        timeout=120,
        max_retries=2,
        max_memory_mb=1000,
        enable_link_detection=True,
        enable_auto_sanitization=True
    )

    # Example 1: Convert file to PDF
    success, message, output_path = converter.convert_to_pdf("sample.pptx")
    print(f"Conversion: {message}")

    # Example 2: Convert bytes to PDF
    with open("sample.pptx", "rb") as f:
        pptx_bytes = f.read()

    pdf_bytes = converter.convert_office_to_pdf_bytes(pptx_bytes, "powerpoint")
    print(f"Converted to PDF: {len(pdf_bytes)} bytes")
