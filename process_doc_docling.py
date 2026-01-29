#!/usr/bin/env python3
"""
Comprehensive Document Processing Script using Docling

This script processes various document formats (PDF, Office files, images, HTML, Markdown)
with maximum OCR accuracy, table extraction in multiple formats, and batch processing capabilities.

Features:
- Supports PDF (including scanned), DOCX, PPTX, XLSX, images (PNG, JPEG, TIFF, etc.), HTML, Markdown
- Maximum accuracy OCR with full page OCR support
- Table extraction to CSV, Markdown, JSON, and HTML
- Image extraction at high resolution
- Page image generation
- Multi-language OCR support
- Batch processing with detailed progress and summary statistics

Usage:
    python process_documents_with_docling.py <input_path> [options]
    python process_documents_with_docling.py samples/
    python process_documents_with_docling.py document.pdf --languages en,es
    python process_documents_with_docling.py folder/ --output-dir ./output --fast-mode
    python process_documents_with_docling.py document.pdf --vlm
    python process_documents_with_docling.py document.pdf --vlm --vlm-model granite-docling-mlx
"""

import os
import sys
import json
import time
import argparse
import tempfile
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)

try:
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
        WordFormatOption,
        ImageFormatOption,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
        EasyOcrOptions,
        VlmPipelineOptions,
    )
    from docling.datamodel import vlm_model_specs
    from docling.pipeline.vlm_pipeline import VlmPipeline
except ImportError:
    print("Error: docling is required. Install with: pip install docling")
    sys.exit(1)


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# File extension to InputFormat mapping
FILE_FORMAT_MAP = {
    # PDF
    '.pdf': InputFormat.PDF,
    # Office formats
    '.docx': InputFormat.DOCX,
    '.pptx': InputFormat.PPTX,
    '.xlsx': InputFormat.XLSX,
    # Images
    '.png': InputFormat.IMAGE,
    '.jpg': InputFormat.IMAGE,
    '.jpeg': InputFormat.IMAGE,
    '.tiff': InputFormat.IMAGE,
    '.tif': InputFormat.IMAGE,
    '.bmp': InputFormat.IMAGE,
    '.webp': InputFormat.IMAGE,
    # Web formats
    '.html': InputFormat.HTML,
    '.htm': InputFormat.HTML,
    '.md': InputFormat.MD,
    '.markdown': InputFormat.MD,
}

SUPPORTED_EXTENSIONS = set(FILE_FORMAT_MAP.keys())

# Default configuration
DEFAULT_OCR_LANGUAGES = ["en"]
DEFAULT_IMAGE_SCALE = 2.0
DEFAULT_OUTPUT_DIR = "docling_output"


# ============================================================================
# URL HANDLING FUNCTIONS
# ============================================================================

def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    return path.startswith(('http://', 'https://'))


def get_extension_from_url(url: str) -> str:
    """Extract file extension from URL."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    # Get the extension from the path
    ext = os.path.splitext(path)[1].lower()
    return ext if ext else '.html'  # Default to .html if no extension


def download_url_to_temp(url: str) -> Tuple[Path, str]:
    """
    Download a URL to a temporary file.

    Args:
        url: The URL to download

    Returns:
        Tuple of (temp_file_path, original_filename)
    """
    print(f"Downloading URL: {url}")

    # Get extension from URL
    ext = get_extension_from_url(url)

    # Extract filename from URL for display purposes
    parsed = urllib.parse.urlparse(url)
    original_filename = os.path.basename(parsed.path) or "downloaded_content"
    if not original_filename.endswith(ext):
        original_filename += ext

    # Create a temporary file with the correct extension
    temp_dir = tempfile.mkdtemp(prefix="docling_url_")
    temp_file = Path(temp_dir) / original_filename

    try:
        # Download the content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        request = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
            temp_file.write_bytes(content)

        print(f"  Downloaded to: {temp_file}")
        print(f"  Size: {temp_file.stat().st_size / 1024:.1f} KB")

        return temp_file, original_filename

    except Exception as e:
        # Clean up on error
        if temp_file.exists():
            temp_file.unlink()
        if Path(temp_dir).exists():
            Path(temp_dir).rmdir()
        raise Exception(f"Failed to download URL: {e}")


# ============================================================================
# DOCUMENTCONVERTER CONFIGURATION
# ============================================================================

def create_document_converter(
    enable_ocr: bool = True,
    enable_tables: bool = True,
    enable_images: bool = True,
    enable_page_images: bool = True,
    ocr_languages: List[str] = None,
    fast_mode: bool = False,
    image_scale: float = DEFAULT_IMAGE_SCALE,
    artifacts_path: Optional[str] = None,
    use_vlm: bool = False,
    vlm_model: str = "granite-docling"
) -> DocumentConverter:
    """
    Create and configure a DocumentConverter with all pipeline options.

    Args:
        enable_ocr: Enable OCR for text extraction
        enable_tables: Enable table structure extraction
        enable_images: Enable image extraction
        enable_page_images: Enable page image generation
        ocr_languages: List of language codes for OCR (e.g., ["en", "es"])
        artifacts_path: Optional path to store downloaded models (for offline use)
        fast_mode: Use fast mode (TableFormerMode.FAST) instead of accurate mode
        image_scale: Resolution scale for images (default: 2.0)
        use_vlm: Use VLM (Vision Language Model) pipeline instead of standard pipeline
        vlm_model: VLM model to use: "granite-docling", "granite-docling-mlx", "smoldocling"

    Returns:
        Configured DocumentConverter instance
    """
    if ocr_languages is None:
        ocr_languages = DEFAULT_OCR_LANGUAGES

    # Use VLM pipeline if requested
    if use_vlm:
        # Select VLM model specification
        vlm_model_map = {
            "granite-docling": vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
            "granite-docling-mlx": vlm_model_specs.GRANITEDOCLING_MLX,
            "smoldocling": vlm_model_specs.SMOLDOCLING_TRANSFORMERS,
        }

        model_spec = vlm_model_map.get(vlm_model.lower())
        if model_spec is None:
            print(f"Warning: Unknown VLM model '{vlm_model}', using granite-docling")
            model_spec = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS

        # Configure VLM pipeline options
        vlm_options = VlmPipelineOptions(vlm_options=model_spec)

        # Set artifacts path if provided
        if artifacts_path:
            vlm_options.artifacts_path = artifacts_path

        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=vlm_options,
            ),
        }

        converter = DocumentConverter(format_options=format_options)
        return converter

    # Standard pipeline configuration
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = enable_ocr
    pdf_pipeline_options.do_table_structure = enable_tables
    pdf_pipeline_options.generate_page_images = enable_page_images
    pdf_pipeline_options.images_scale = image_scale

    # Set artifacts path if provided
    if artifacts_path:
        pdf_pipeline_options.artifacts_path = artifacts_path

    if enable_ocr:
        pdf_pipeline_options.ocr_options = EasyOcrOptions(
            lang=ocr_languages,
            force_full_page_ocr=True
        )

    if enable_tables:
        table_mode = TableFormerMode.FAST if fast_mode else TableFormerMode.ACCURATE
        pdf_pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode=table_mode
        )

    # Configure format options
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
    }

    # Note: Other formats (DOCX, PPTX, XLSX, HTML, Image) use default pipelines
    # as they don't have the same extensive configuration options as PDF

    # Create converter (artifacts_path is set in pipeline options above)
    converter = DocumentConverter(format_options=format_options)

    return converter


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def determine_file_format(file_path: Path) -> Optional[InputFormat]:
    """
    Determine the InputFormat based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        InputFormat enum value or None if unsupported
    """
    ext = file_path.suffix.lower()
    return FILE_FORMAT_MAP.get(ext)


def find_supported_files(input_path: Path) -> List[Path]:
    """
    Find all supported files in a directory or return single file.

    Args:
        input_path: Path to file or directory

    Returns:
        List of supported file paths
    """
    files = []

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(input_path)
    elif input_path.is_dir():
        for ext in SUPPORTED_EXTENSIONS:
            # Recursively find files with this extension
            files.extend(input_path.rglob(f"*{ext}"))
    else:
        print(f"Warning: Path not found: {input_path}")

    return sorted(files)


def create_output_structure(output_dir: Path, base_filename: str) -> Dict[str, Path]:
    """
    Create organized output directory structure.

    Args:
        output_dir: Base output directory
        base_filename: Base filename for organization

    Returns:
        Dictionary with paths for different output types
    """
    file_dir = output_dir / base_filename
    file_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = file_dir / "tables"
    images_dir = file_dir / "images"
    pages_dir = file_dir / "pages"

    tables_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    pages_dir.mkdir(exist_ok=True)

    return {
        'base': file_dir,
        'tables': tables_dir,
        'images': images_dir,
        'pages': pages_dir
    }


# ============================================================================
# TABLE EXTRACTION FUNCTIONS
# ============================================================================

def extract_table_metadata(table, table_df) -> Dict:
    """
    Extract metadata from a table including headers, rows, columns count.

    Args:
        table: Table object from docling
        table_df: DataFrame representation of the table

    Returns:
        Dictionary with table metadata
    """
    metadata = {
        'num_rows': len(table_df),
        'num_columns': len(table_df.columns),
        'headers': list(table_df.columns),
        'has_headers': True,  # DataFrame always has column names
        'total_cells': len(table_df) * len(table_df.columns),
    }

    # Try to get additional metadata from table object
    try:
        if hasattr(table, 'data') and hasattr(table.data, 'grid'):
            grid = table.data.grid
            metadata['grid_rows'] = len(grid) if grid else 0
            metadata['grid_cols'] = len(grid[0]) if grid and len(grid) > 0 else 0
    except Exception:
        pass

    # Try to get bbox/position information
    try:
        if hasattr(table, 'prov') and table.prov:
            metadata['bbox'] = table.prov[0].bbox.as_tuple() if hasattr(table.prov[0], 'bbox') else None
            metadata['page_number'] = table.prov[0].page_no if hasattr(table.prov[0], 'page_no') else None
    except Exception:
        pass

    return metadata


def export_tables_to_csv(tables, doc, output_dir: Path, base_filename: str) -> int:
    """
    Export tables to CSV format.

    Args:
        tables: List of tables from document
        doc: Document object for context
        output_dir: Output directory for CSV files
        base_filename: Base filename for output files

    Returns:
        Number of successfully exported tables
    """
    count = 0
    for table_ix, table in enumerate(tables):
        try:
            table_df = table.export_to_dataframe(doc=doc)
            csv_path = output_dir / f"{base_filename}_table_{table_ix}.csv"
            table_df.to_csv(csv_path, index=False)

            # Export table metadata
            metadata = extract_table_metadata(table, table_df)
            metadata_path = output_dir / f"{base_filename}_table_{table_ix}_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

            count += 1
        except Exception as e:
            print(f"  Warning: Failed to export table {table_ix} to CSV: {e}")

    return count


def export_tables_to_markdown(tables, doc, output_dir: Path, base_filename: str) -> int:
    """
    Export tables to Markdown format.

    Args:
        tables: List of tables from document
        doc: Document object for context
        output_dir: Output directory for Markdown files
        base_filename: Base filename for output files

    Returns:
        Number of successfully exported tables
    """
    count = 0
    for table_ix, table in enumerate(tables):
        try:
            table_df = table.export_to_dataframe(doc=doc)
            md_path = output_dir / f"{base_filename}_table_{table_ix}.md"

            # Try to use tabulate if available, otherwise use basic markdown
            try:
                md_content = table_df.to_markdown(index=False)
            except Exception:
                # Fallback: create simple markdown table
                md_content = f"# Table {table_ix}\n\n"
                md_content += table_df.to_string(index=False)

            md_path.write_text(md_content, encoding='utf-8')
            count += 1
        except Exception as e:
            print(f"  Warning: Failed to export table {table_ix} to Markdown: {e}")

    return count


def export_tables_to_json(tables, doc, output_dir: Path, base_filename: str) -> int:
    """
    Export tables to JSON format.

    Args:
        tables: List of tables from document
        doc: Document object for context
        output_dir: Output directory for JSON files
        base_filename: Base filename for output files

    Returns:
        Number of successfully exported tables
    """
    count = 0
    for table_ix, table in enumerate(tables):
        try:
            table_df = table.export_to_dataframe(doc=doc)
            json_path = output_dir / f"{base_filename}_table_{table_ix}.json"

            # Convert to JSON with proper formatting
            table_json = table_df.to_json(orient='records', indent=2)
            json_path.write_text(table_json, encoding='utf-8')
            count += 1
        except Exception as e:
            print(f"  Warning: Failed to export table {table_ix} to JSON: {e}")

    return count


def export_tables_to_html(tables, doc, output_dir: Path, base_filename: str) -> int:
    """
    Export tables to HTML format.

    Args:
        tables: List of tables from document
        doc: Document object for context
        output_dir: Output directory for HTML files
        base_filename: Base filename for output files

    Returns:
        Number of successfully exported tables
    """
    count = 0
    for table_ix, table in enumerate(tables):
        try:
            # Try using table's export_to_html method if available
            try:
                html_content = table.export_to_html()
            except AttributeError:
                # Fallback: use pandas to_html
                table_df = table.export_to_dataframe(doc=doc)
                html_content = table_df.to_html(index=False)

            html_path = output_dir / f"{base_filename}_table_{table_ix}.html"
            html_path.write_text(html_content, encoding='utf-8')
            count += 1
        except Exception as e:
            print(f"  Warning: Failed to export table {table_ix} to HTML: {e}")

    return count


# ============================================================================
# IMAGE EXTRACTION FUNCTIONS
# ============================================================================

def extract_images(conv_result, output_dir: Path, base_filename: str) -> int:
    """
    Extract embedded images from document.

    Args:
        conv_result: Conversion result from DocumentConverter
        output_dir: Output directory for images
        base_filename: Base filename for output files

    Returns:
        Number of successfully extracted images
    """
    count = 0
    try:
        # Access pictures from the document
        if hasattr(conv_result.document, 'pictures') and conv_result.document.pictures:
            for img_ix, picture in enumerate(conv_result.document.pictures):
                try:
                    # Get image data
                    if hasattr(picture, 'image') and picture.image:
                        img_path = output_dir / f"{base_filename}_image_{img_ix}.png"

                        # Save image
                        if hasattr(picture.image, 'pil_image'):
                            picture.image.pil_image.save(img_path)
                            count += 1
                        elif hasattr(picture, 'get_image'):
                            img = picture.get_image()
                            if img:
                                img.save(img_path)
                                count += 1
                except Exception as e:
                    print(f"  Warning: Failed to extract image {img_ix}: {e}")
    except Exception as e:
        print(f"  Warning: Failed to access images: {e}")

    return count


def extract_page_images(conv_result, output_dir: Path, base_filename: str) -> int:
    """
    Extract generated page images from document.

    Args:
        conv_result: Conversion result from DocumentConverter
        output_dir: Output directory for page images
        base_filename: Base filename for output files

    Returns:
        Number of successfully extracted page images
    """
    count = 0
    try:
        # Access page images if they were generated
        if hasattr(conv_result.document, 'pages'):
            for page_ix, page in enumerate(conv_result.document.pages):
                try:
                    if hasattr(page, 'image') and page.image:
                        img_path = output_dir / f"{base_filename}_page_{page_ix}.png"

                        # Save page image
                        if hasattr(page.image, 'pil_image'):
                            page.image.pil_image.save(img_path)
                            count += 1
                        elif hasattr(page, 'get_image'):
                            img = page.get_image()
                            if img:
                                img.save(img_path)
                                count += 1
                except Exception as e:
                    print(f"  Warning: Failed to extract page image {page_ix}: {e}")
    except Exception as e:
        print(f"  Warning: Failed to access page images: {e}")

    return count


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_single_document(
    file_path: Path,
    output_dir: Path,
    converter: DocumentConverter,
    enable_tables: bool = True,
    enable_images: bool = True,
    enable_page_images: bool = True
) -> Dict:
    """
    Process a single document and extract all content.

    Args:
        file_path: Path to the document
        output_dir: Base output directory
        converter: Configured DocumentConverter
        enable_tables: Export tables
        enable_images: Extract images
        enable_page_images: Extract page images

    Returns:
        Dictionary with processing statistics
    """
    start_time = time.time()
    base_filename = file_path.stem

    stats = {
        'filename': file_path.name,
        'file_path': str(file_path),
        'file_size_mb': file_path.stat().st_size / (1024 * 1024),
        'format': determine_file_format(file_path),
        'success': False,
        'tables_found': 0,
        'tables_exported': 0,
        'images_extracted': 0,
        'page_images_extracted': 0,
        'processing_time_seconds': 0,
        'error': None
    }

    print(f"\n{'='*70}")
    print(f"Processing: {file_path.name}")
    print(f"Size: {stats['file_size_mb']:.2f} MB | Format: {stats['format']}")
    print(f"{'='*70}")

    try:
        # Create output structure
        output_paths = create_output_structure(output_dir, base_filename)

        # Convert document
        print(f"Converting document...")
        conv_result = converter.convert(str(file_path))

        # Export main markdown content
        markdown_path = output_paths['base'] / f"{base_filename}.md"
        markdown_content = conv_result.document.export_to_markdown()
        markdown_path.write_text(markdown_content, encoding='utf-8')
        print(f"  [OK] Markdown exported: {markdown_path.name}")

        # Export tables in all formats
        if enable_tables and hasattr(conv_result.document, 'tables'):
            tables = list(conv_result.document.tables)
            stats['tables_found'] = len(tables)

            if tables:
                print(f"  Exporting {len(tables)} table(s)...")

                # Print table metadata for each table
                for table_ix, table in enumerate(tables):
                    try:
                        table_df = table.export_to_dataframe(doc=conv_result.document)
                        metadata = extract_table_metadata(table, table_df)
                        print(f"    Table {table_ix}: {metadata['num_rows']} rows x {metadata['num_columns']} columns")
                        print(f"      Headers: {', '.join([str(h) for h in metadata['headers'][:5]])}{'...' if len(metadata['headers']) > 5 else ''}")
                    except Exception:
                        pass

                # Export to CSV
                csv_count = export_tables_to_csv(
                    tables, conv_result.document,
                    output_paths['tables'], base_filename
                )
                print(f"    [OK] CSV: {csv_count} tables (with metadata)")

                # Export to Markdown
                md_count = export_tables_to_markdown(
                    tables, conv_result.document,
                    output_paths['tables'], base_filename
                )
                print(f"    [OK] Markdown: {md_count} tables")

                # Export to JSON
                json_count = export_tables_to_json(
                    tables, conv_result.document,
                    output_paths['tables'], base_filename
                )
                print(f"    [OK] JSON: {json_count} tables")

                # Export to HTML
                html_count = export_tables_to_html(
                    tables, conv_result.document,
                    output_paths['tables'], base_filename
                )
                print(f"    [OK] HTML: {html_count} tables")

                stats['tables_exported'] = max(csv_count, md_count, json_count, html_count)

        # Extract embedded images
        if enable_images:
            img_count = extract_images(
                conv_result,
                output_paths['images'],
                base_filename
            )
            stats['images_extracted'] = img_count
            if img_count > 0:
                print(f"  [OK] Extracted {img_count} embedded image(s)")

        # Extract page images
        if enable_page_images:
            page_count = extract_page_images(
                conv_result,
                output_paths['pages'],
                base_filename
            )
            stats['page_images_extracted'] = page_count
            if page_count > 0:
                print(f"  [OK] Extracted {page_count} page image(s)")

        # Save metadata
        metadata_path = output_paths['base'] / f"{base_filename}_metadata.json"
        processing_time = time.time() - start_time
        stats['processing_time_seconds'] = round(processing_time, 2)
        stats['success'] = True

        metadata = {
            'processing_date': datetime.now().isoformat(),
            'statistics': stats,
            'output_location': str(output_paths['base'])
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

        print(f"  [OK] Processing completed in {processing_time:.2f}s")
        print(f"  Output: {output_paths['base']}")

    except Exception as e:
        stats['error'] = str(e)
        stats['processing_time_seconds'] = round(time.time() - start_time, 2)
        print(f"  [ERROR] Error: {e}")

    return stats


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_batch(
    files: List[Path],
    output_dir: Path,
    converter: DocumentConverter,
    enable_tables: bool = True,
    enable_images: bool = True,
    enable_page_images: bool = True
) -> Dict:
    """
    Process multiple files with progress tracking.

    Args:
        files: List of file paths to process
        output_dir: Base output directory
        converter: Configured DocumentConverter
        enable_tables: Export tables
        enable_images: Extract images
        enable_page_images: Extract page images

    Returns:
        Dictionary with aggregate statistics
    """
    total_files = len(files)
    all_stats = []
    start_time = time.time()

    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {total_files} file(s)")
    print(f"{'='*70}")

    for idx, file_path in enumerate(files, 1):
        print(f"\n[{idx}/{total_files}] ", end='')

        stats = process_single_document(
            file_path, output_dir, converter,
            enable_tables, enable_images, enable_page_images
        )
        all_stats.append(stats)

    # Calculate aggregate statistics
    total_time = time.time() - start_time
    success_count = sum(1 for s in all_stats if s['success'])
    fail_count = total_files - success_count
    total_tables = sum(s['tables_found'] for s in all_stats)
    total_images = sum(s['images_extracted'] for s in all_stats)
    total_page_images = sum(s['page_images_extracted'] for s in all_stats)

    # Group by file type
    format_stats = {}
    for stat in all_stats:
        fmt = str(stat['format'])
        if fmt not in format_stats:
            format_stats[fmt] = {'count': 0, 'success': 0}
        format_stats[fmt]['count'] += 1
        if stat['success']:
            format_stats[fmt]['success'] += 1

    summary = {
        'total_files': total_files,
        'successful': success_count,
        'failed': fail_count,
        'total_tables_extracted': total_tables,
        'total_images_extracted': total_images,
        'total_page_images_extracted': total_page_images,
        'total_processing_time_seconds': round(total_time, 2),
        'output_directory': str(output_dir.absolute()),
        'format_breakdown': format_stats,
        'individual_results': all_stats
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total files:        {total_files}")
    print(f"Successful:         {success_count}")
    print(f"Failed:             {fail_count}")
    print(f"Tables extracted:   {total_tables}")
    print(f"Images extracted:   {total_images}")
    print(f"Page images:        {total_page_images}")
    print(f"Total time:         {total_time:.2f}s")
    print(f"Output directory:   {output_dir.absolute()}")
    print(f"\nFormat breakdown:")
    for fmt, data in format_stats.items():
        print(f"  {fmt}: {data['success']}/{data['count']} successful")
    print(f"{'='*70}\n")

    # Save summary to JSON
    summary_path = output_dir / "processing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Summary saved to: {summary_path}")

    return summary


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Comprehensive document processing with Docling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf
  %(prog)s samples/ --output-dir ./results
  %(prog)s folder/ --languages en,es,de --fast-mode
  %(prog)s document.pdf --no-ocr --no-page-images
  %(prog)s https://example.com/page.html --output-dir ./results
  %(prog)s document.pdf --vlm
  %(prog)s document.pdf --vlm --vlm-model granite-docling-mlx
        """
    )

    parser.add_argument(
        'input_path',
        type=str,
        help='Input file, directory path, or URL (http:// or https://)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--languages',
        type=str,
        default='en',
        help='OCR languages as comma-separated list (default: en)'
    )

    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR processing'
    )

    parser.add_argument(
        '--no-tables',
        action='store_true',
        help='Skip table extraction'
    )

    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Skip image extraction'
    )

    parser.add_argument(
        '--no-page-images',
        action='store_true',
        help='Skip page image generation'
    )

    parser.add_argument(
        '--fast-mode',
        action='store_true',
        help='Use fast mode (TableFormerMode.FAST) instead of accurate mode'
    )

    parser.add_argument(
        '--artifacts-path',
        type=str,
        default=None,
        help='Path to store/load model artifacts (helps with offline use and network issues)'
    )

    parser.add_argument(
        '--vlm',
        action='store_true',
        help='Use VLM (Vision Language Model) pipeline for document conversion'
    )

    parser.add_argument(
        '--vlm-model',
        type=str,
        default='granite-docling',
        choices=['granite-docling', 'granite-docling-mlx', 'smoldocling'],
        help='VLM model to use (default: granite-docling). Use granite-docling-mlx for Mac with MPS'
    )

    args = parser.parse_args()

    # Track if we downloaded from URL (for cleanup)
    temp_file_path = None
    is_url_input = is_url(args.input_path)

    # Handle URL input
    if is_url_input:
        try:
            temp_file_path, original_filename = download_url_to_temp(args.input_path)
            input_path = temp_file_path
            print(f"Processing URL as: {original_filename}")
        except Exception as e:
            print(f"Error downloading URL: {e}")
            sys.exit(1)
    else:
        # Validate local input path
        input_path = Path(args.input_path)
        if not input_path.exists():
            print(f"Error: Input path not found: {input_path}")
            sys.exit(1)

    # Parse languages
    ocr_languages = [lang.strip() for lang in args.languages.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find files to process
    files = find_supported_files(input_path)
    if not files:
        print(f"No supported files found in: {input_path}")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)

    # Configure converter
    print("Initializing Docling DocumentConverter...")
    if args.vlm:
        print(f"  Pipeline: VLM (Vision Language Model)")
        print(f"  VLM Model: {args.vlm_model}")
    else:
        print(f"  Pipeline: Standard")
        print(f"  OCR: {'Disabled' if args.no_ocr else 'Enabled'}")
        print(f"  OCR Languages: {', '.join(ocr_languages)}")
        print(f"  Table Extraction: {'Disabled' if args.no_tables else 'Enabled'}")
        print(f"  Table Mode: {'Fast' if args.fast_mode else 'Accurate'}")
    print(f"  Image Extraction: {'Disabled' if args.no_images else 'Enabled'}")
    print(f"  Page Images: {'Disabled' if args.no_page_images else 'Enabled'}")
    if args.artifacts_path:
        print(f"  Artifacts Path: {args.artifacts_path}")

    converter = create_document_converter(
        enable_ocr=not args.no_ocr,
        enable_tables=not args.no_tables,
        enable_images=not args.no_images,
        enable_page_images=not args.no_page_images,
        ocr_languages=ocr_languages,
        fast_mode=args.fast_mode,
        artifacts_path=args.artifacts_path,
        use_vlm=args.vlm,
        vlm_model=args.vlm_model
    )

    # Process files
    try:
        if len(files) == 1:
            # Single file processing
            stats = process_single_document(
                files[0], output_dir, converter,
                enable_tables=not args.no_tables,
                enable_images=not args.no_images,
                enable_page_images=not args.no_page_images
            )

            if not stats['success']:
                sys.exit(1)
        else:
            # Batch processing
            summary = process_batch(
                files, output_dir, converter,
                enable_tables=not args.no_tables,
                enable_images=not args.no_images,
                enable_page_images=not args.no_page_images
            )

            if summary['failed'] > 0:
                sys.exit(1)
    finally:
        # Clean up temporary files from URL downloads
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                temp_file_path.parent.rmdir()
                print(f"\nCleaned up temporary file: {temp_file_path}")
            except Exception:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
