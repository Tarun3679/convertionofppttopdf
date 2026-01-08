#!/usr/bin/env python3
"""
Script to convert PowerPoint (PPTX/PPT) and Excel (XLSX/XLS) files to Markdown using docling.

Usage:
    python convert_to_markdown.py <input_file_or_folder>
    python convert_to_markdown.py samples/powerpoint
    python convert_to_markdown.py samples/powerpoint/SamplePPTX-All.pptx
"""

import os
import sys
from pathlib import Path
from docling.document_converter import DocumentConverter

# Supported file extensions (docling only supports modern Office formats)
# Note: Old formats (.ppt, .xls, .doc) are NOT supported by docling
SUPPORTED_EXTENSIONS = {'.pptx', '.xlsx'}


def convert_file_to_markdown(input_path: Path, output_folder: Path, converter: DocumentConverter):
    """
    Convert a single file to Markdown format.

    Args:
        input_path: Path to the input file
        output_folder: Path to the output folder for markdown files
        converter: DocumentConverter instance
    """
    try:
        print(f"\n{'='*60}")
        print(f"Converting: {input_path.name}")
        print(f"Size: {input_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"{'='*60}")

        # Convert the document
        result = converter.convert(str(input_path))

        # Create output filename (replace extension with .md)
        output_filename = input_path.stem + ".md"
        output_path = output_folder / output_filename

        # Export to Markdown
        markdown_content = result.document.export_to_markdown()

        # Write to file
        output_path.write_text(markdown_content, encoding='utf-8')

        print(f"[SUCCESS] Converted to: {output_path}")
        print(f"  Output size: {output_path.stat().st_size / 1024:.2f} KB")

        return True

    except Exception as e:
        print(f"[ERROR] Error converting {input_path.name}: {str(e)}")
        return False


def process_path(input_path: Path, output_folder: Path):
    """
    Process a file or folder for conversion.

    Args:
        input_path: Path to file or folder
        output_folder: Path to output folder
    """
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Initialize the document converter
    print("Initializing Docling DocumentConverter...")
    converter = DocumentConverter()

    files_to_convert = []

    if input_path.is_file():
        # Single file
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files_to_convert.append(input_path)
        else:
            print(f"Error: Unsupported file type '{input_path.suffix}'")
            print(f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}")
            return
    elif input_path.is_dir():
        # Folder - find all supported files
        for ext in SUPPORTED_EXTENSIONS:
            files_to_convert.extend(input_path.glob(f"*{ext}"))

        if not files_to_convert:
            print(f"No supported files found in {input_path}")
            return
    else:
        print(f"Error: Path not found: {input_path}")
        return

    # Process all files
    print(f"\nFound {len(files_to_convert)} file(s) to convert")

    success_count = 0
    fail_count = 0

    for file_path in files_to_convert:
        if convert_file_to_markdown(file_path, output_folder, converter):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(files_to_convert)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output folder: {output_folder.absolute()}")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_to_markdown.py <input_file_or_folder>")
        print("\nExamples:")
        print("  python convert_to_markdown.py samples/powerpoint")
        print("  python convert_to_markdown.py samples/powerpoint/SamplePPTX-All.pptx")
        print("  python convert_to_markdown.py samples/excel/sample.xlsx")
        sys.exit(1)

    input_arg = sys.argv[1]
    input_path = Path(input_arg)

    # Create output folder next to the input
    if input_path.is_file():
        output_folder = input_path.parent / "markdown_output"
    else:
        output_folder = input_path / "markdown_output"

    process_path(input_path, output_folder)


if __name__ == "__main__":
    main()
