"""
Log Parser for Document Conversion Logs

Parses conversion logs to extract:
- Document input filename
- Whether links were detected
- Number of links found
- Link types detected
- Conversion success/failure status
- PDF size (if successful)
- Conversion time
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class DocumentConversion:
    """Represents a single document conversion attempt."""
    filename: str
    file_type: str  # 'excel' or 'powerpoint'
    has_links: bool = False
    links_found: int = 0
    link_types: list = field(default_factory=list)
    conversion_successful: bool = False
    pdf_size_mb: Optional[float] = None
    conversion_time_seconds: Optional[float] = None
    attempt_number: int = 1
    max_retries: int = 2
    timestamp: str = ""

    def __repr__(self):
        status = "SUCCESS" if self.conversion_successful else "FAILED"
        links_info = f"Links: {self.links_found}" if self.has_links else "No links"
        return f"{self.filename} | {self.file_type} | {status} | {links_info}"


def parse_log_line(line: str) -> dict:
    """Parse a single log line and extract relevant information."""
    result = {
        'timestamp': None,
        'event_type': None,
        'filename': None,
        'file_type': None,
        'links_found': None,
        'link_types': None,
        'success': None,
        'pdf_size': None,
        'conversion_time': None,
        'attempt': None,
    }

    # Extract timestamp
    timestamp_match = re.match(r'\[?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]?', line)
    if timestamp_match:
        result['timestamp'] = timestamp_match.group(1)

    # Check for ENTRY (new conversion starting)
    entry_match = re.search(r'\[ENTRY\]\s+convert_office_to_pdf\s+called:\s+type=(\w+),\s+size=([\d.]+)MB,\s+max_retries=(\d+)', line)
    if entry_match:
        result['event_type'] = 'ENTRY'
        result['file_type'] = entry_match.group(1)
        result['max_retries'] = int(entry_match.group(3))

    # Check for conversion attempt with filename
    attempt_match = re.search(r'\[Attempt\s+(\d+)/(\d+)\]\s+Converting\s+(\w+)_input_([^\s]+)', line)
    if attempt_match:
        result['event_type'] = 'ATTEMPT'
        result['attempt'] = int(attempt_match.group(1))
        result['file_type'] = attempt_match.group(3)
        result['filename'] = f"{attempt_match.group(3)}_input_{attempt_match.group(4)}"

    # Check for LINK_SCAN results
    link_scan_match = re.search(r'\[LINK_SCAN\]\s+file=[\'"]?([^\'"]+)[\'"]?\s+external_links_found=(\d+)\s+link_types=\[([^\]]*)\]', line)
    if link_scan_match:
        result['event_type'] = 'LINK_SCAN'
        result['filename'] = link_scan_match.group(1)
        result['links_found'] = int(link_scan_match.group(2))
        link_types_str = link_scan_match.group(3)
        result['link_types'] = [lt.strip().strip("'\"") for lt in link_types_str.split(',') if lt.strip()]

    # Check for SUCCESS
    success_match = re.search(r'\[SUCCESS\]\s+(\w+)\s+conversion\s+completed,\s+PDF\s+size:\s+([\d.]+)MB', line)
    if success_match:
        result['event_type'] = 'SUCCESS'
        result['file_type'] = success_match.group(1)
        result['pdf_size'] = float(success_match.group(2))
        result['success'] = True

    # Check for conversion time in "Converted X to PDF ... in X.Xs"
    converted_match = re.search(r'Converted\s+(\w+_input_[^\s]+)\s+to\s+PDF\s+\(([\d.]+)MB\)\s+in\s+([\d.]+)s', line)
    if converted_match:
        result['event_type'] = 'CONVERTED'
        result['filename'] = converted_match.group(1)
        result['pdf_size'] = float(converted_match.group(2))
        result['conversion_time'] = float(converted_match.group(3))
        result['success'] = True

    return result


def parse_logs(log_content: str) -> list[DocumentConversion]:
    """Parse the entire log content and return a list of DocumentConversion objects."""
    lines = log_content.strip().split('\n')

    # Track conversions by filename
    conversions = {}
    current_file = None
    current_type = None

    for line in lines:
        parsed = parse_log_line(line)

        # Handle ENTRY - marks start of a new conversion
        if parsed['event_type'] == 'ENTRY':
            current_type = parsed['file_type']

        # Handle ATTEMPT - get the filename
        if parsed['event_type'] == 'ATTEMPT':
            current_file = parsed['filename']
            if current_file not in conversions:
                conversions[current_file] = DocumentConversion(
                    filename=current_file,
                    file_type=parsed['file_type'],
                    attempt_number=parsed['attempt'],
                    timestamp=parsed['timestamp'] or ""
                )
            else:
                conversions[current_file].attempt_number = parsed['attempt']

        # Handle LINK_SCAN - update link information
        if parsed['event_type'] == 'LINK_SCAN':
            filename = parsed['filename']
            # Try to match with existing conversion
            matching_file = None
            for key in conversions:
                if filename in key or key in filename:
                    matching_file = key
                    break

            if matching_file:
                conversions[matching_file].has_links = parsed['links_found'] > 0
                conversions[matching_file].links_found = parsed['links_found']
                conversions[matching_file].link_types = parsed['link_types']
            elif current_file and current_file in conversions:
                conversions[current_file].has_links = parsed['links_found'] > 0
                conversions[current_file].links_found = parsed['links_found']
                conversions[current_file].link_types = parsed['link_types']

        # Handle SUCCESS
        if parsed['event_type'] == 'SUCCESS':
            if current_file and current_file in conversions:
                conversions[current_file].conversion_successful = True
                conversions[current_file].pdf_size_mb = parsed['pdf_size']

        # Handle CONVERTED - contains time and confirms success
        if parsed['event_type'] == 'CONVERTED':
            filename = parsed['filename']
            if filename in conversions:
                conversions[filename].conversion_successful = True
                conversions[filename].pdf_size_mb = parsed['pdf_size']
                conversions[filename].conversion_time_seconds = parsed['conversion_time']

    return list(conversions.values())


def generate_report(conversions: list[DocumentConversion]) -> str:
    """Generate a formatted report from the parsed conversions."""
    report_lines = []

    report_lines.append("=" * 100)
    report_lines.append("DOCUMENT CONVERSION LOG ANALYSIS REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")

    # Summary statistics
    total = len(conversions)
    successful = sum(1 for c in conversions if c.conversion_successful)
    with_links = sum(1 for c in conversions if c.has_links)
    successful_with_links = sum(1 for c in conversions if c.has_links and c.conversion_successful)

    report_lines.append("SUMMARY")
    report_lines.append("-" * 50)
    report_lines.append(f"Total documents processed: {total}")
    report_lines.append(f"Successful conversions: {successful} ({100*successful/total:.1f}%)" if total > 0 else "No documents")
    report_lines.append(f"Documents with links: {with_links}")
    report_lines.append(f"Documents with links - successful: {successful_with_links}")
    if with_links > 0:
        report_lines.append(f"Success rate for docs with links: {100*successful_with_links/with_links:.1f}%")
    report_lines.append("")

    # By file type
    excel_docs = [c for c in conversions if c.file_type == 'excel']
    ppt_docs = [c for c in conversions if c.file_type == 'powerpoint']

    report_lines.append("BY FILE TYPE")
    report_lines.append("-" * 50)
    if excel_docs:
        excel_success = sum(1 for c in excel_docs if c.conversion_successful)
        report_lines.append(f"Excel: {len(excel_docs)} total, {excel_success} successful")
    if ppt_docs:
        ppt_success = sum(1 for c in ppt_docs if c.conversion_successful)
        report_lines.append(f"PowerPoint: {len(ppt_docs)} total, {ppt_success} successful")
    report_lines.append("")

    # Detailed table
    report_lines.append("DETAILED CONVERSION RESULTS")
    report_lines.append("-" * 100)
    report_lines.append(f"{'Filename':<45} {'Type':<12} {'Links':<8} {'Link Types':<20} {'Status':<10} {'PDF Size':<10}")
    report_lines.append("-" * 100)

    for conv in sorted(conversions, key=lambda x: (not x.has_links, x.filename)):
        links_str = str(conv.links_found) if conv.has_links else "0"
        link_types_str = ", ".join(conv.link_types) if conv.link_types else "-"
        status = "SUCCESS" if conv.conversion_successful else "FAILED"
        pdf_size = f"{conv.pdf_size_mb:.2f}MB" if conv.pdf_size_mb else "-"

        # Truncate filename if too long
        filename = conv.filename[:42] + "..." if len(conv.filename) > 45 else conv.filename
        link_types_str = link_types_str[:17] + "..." if len(link_types_str) > 20 else link_types_str

        report_lines.append(f"{filename:<45} {conv.file_type:<12} {links_str:<8} {link_types_str:<20} {status:<10} {pdf_size:<10}")

    report_lines.append("")
    report_lines.append("=" * 100)

    # Documents with links detail
    docs_with_links = [c for c in conversions if c.has_links]
    if docs_with_links:
        report_lines.append("")
        report_lines.append("DOCUMENTS WITH LINKS - DETAILED")
        report_lines.append("-" * 100)
        for conv in docs_with_links:
            status = "SUCCESS" if conv.conversion_successful else "FAILED"
            report_lines.append(f"File: {conv.filename}")
            report_lines.append(f"  Type: {conv.file_type}")
            report_lines.append(f"  Links found: {conv.links_found}")
            report_lines.append(f"  Link types: {conv.link_types}")
            report_lines.append(f"  Conversion status: {status}")
            if conv.pdf_size_mb:
                report_lines.append(f"  PDF size: {conv.pdf_size_mb}MB")
            if conv.conversion_time_seconds:
                report_lines.append(f"  Conversion time: {conv.conversion_time_seconds}s")
            report_lines.append("")

    return "\n".join(report_lines)


def generate_csv(conversions: list[DocumentConversion]) -> str:
    """Generate CSV output from the parsed conversions."""
    lines = ["filename,file_type,has_links,links_found,link_types,conversion_successful,pdf_size_mb,conversion_time_seconds"]

    for conv in conversions:
        link_types_str = "|".join(conv.link_types) if conv.link_types else ""
        lines.append(
            f'"{conv.filename}",{conv.file_type},{conv.has_links},{conv.links_found},'
            f'"{link_types_str}",{conv.conversion_successful},'
            f'{conv.pdf_size_mb or ""},{conv.conversion_time_seconds or ""}'
        )

    return "\n".join(lines)


def main():
    """Main function to parse logs from file or stdin."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Parse document conversion logs')
    parser.add_argument('logfile', nargs='?', help='Log file to parse (or use stdin)')
    parser.add_argument('--csv', action='store_true', help='Output as CSV')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')

    args = parser.parse_args()

    # Read log content
    if args.logfile:
        with open(args.logfile, 'r', encoding='utf-8') as f:
            log_content = f.read()
    else:
        log_content = sys.stdin.read()

    # Parse logs
    conversions = parse_logs(log_content)

    # Generate output
    if args.csv:
        output = generate_csv(conversions)
    else:
        output = generate_report(conversions)

    # Write output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Output written to {args.output}")
    else:
        print(output)





if __name__ == '__main__':
    import sys

    # If no arguments provided, run with sample data for demonstration
    if len(sys.argv) == 1:
        print("Running with sample log data from screenshots...")
        print("(Use 'python parse_conversion_logs.py <logfile>' to parse actual logs)")
        print("")
        conversions = parse_logs(SAMPLE_LOG)
        print(generate_report(conversions))
    else:
        main()
