#!/usr/bin/env python3
"""
Simple script to convert PPTX or PDF to Markdown/JSON using Docling Serve API.

Usage:
    python convert_with_docling_api.py <file_path> [output_format]

Examples:
    python convert_with_docling_api.py presentation.pptx
    python convert_with_docling_api.py document.pdf json
    python convert_with_docling_api.py report.pptx md output.md
"""

import sys
import json
import requests
from pathlib import Path


# Configuration - change this to your Docling Serve endpoint
DOCLING_SERVER = "http://localhost:5001"


def convert_file(file_path: str, output_format: str = "md") -> str:
    """
    Convert a file using Docling Serve API.

    Args:
        file_path: Path to the PPTX or PDF file
        output_format: "md" for markdown, "json" for JSON

    Returns:
        Converted content as string
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    url = f"{DOCLING_SERVER}/v1/convert/file"

    with open(path, 'rb') as f:
        # Multipart form data with file and parameters
        files = {'files': (path.name, f)}
        data = {
            'to_formats': output_format,  # md, json, html, text
        }

        print(f"Uploading {path.name} to {DOCLING_SERVER}...")
        response = requests.post(url, files=files, data=data, timeout=300)

    if response.status_code != 200:
        raise Exception(f"Conversion failed: {response.status_code} - {response.text}")

    # Parse response JSON
    result = response.json()

    # Check for errors
    if result.get('status') == 'failure':
        errors = result.get('errors', [])
        raise Exception(f"Conversion failed: {errors}")

    # Extract content based on format
    document = result.get('document', {})

    if output_format == 'json':
        json_content = document.get('json_content')
        if json_content:
            if isinstance(json_content, str):
                return json_content
            return json.dumps(json_content, indent=2, ensure_ascii=False)
    elif output_format == 'md':
        return document.get('md_content', '')
    elif output_format == 'html':
        return document.get('html_content', '')
    elif output_format == 'text':
        return document.get('text_content', '')

    # Fallback: return markdown content
    return document.get('md_content', '')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    file_path = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else "md"
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    # Normalize format
    if output_format.lower() == 'markdown':
        output_format = 'md'

    try:
        print(f"Converting: {file_path}")
        print(f"Format: {output_format}")
        print("-" * 50)

        content = convert_file(file_path, output_format)

        if output_file:
            Path(output_file).write_text(content, encoding='utf-8')
            print(f"Saved to: {output_file}")
        else:
            # Auto-generate output filename
            input_path = Path(file_path)
            ext = '.json' if output_format == 'json' else '.md'
            output_path = input_path.with_suffix(ext)
            output_path.write_text(content, encoding='utf-8')
            print(f"Saved to: {output_path}")

        print("Done!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to Docling server at {DOCLING_SERVER}")
        print("Make sure the server is running.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
