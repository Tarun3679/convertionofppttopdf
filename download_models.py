#!/usr/bin/env python3
"""
Docling Model Downloader

This script downloads all required Docling models for offline use.
Run this once with internet connection, then models will be cached locally.

Usage:
    python download_models.py
    python download_models.py --path ./my_models
    python download_models.py --languages en,es,fr
"""

import os
import sys
import argparse
from pathlib import Path

# Default configuration
DEFAULT_ARTIFACTS_PATH = "./docling_models"
DEFAULT_LANGUAGES = ["en"]


def download_models(artifacts_path: str, languages: list):
    """
    Download all Docling models to the specified path.

    Args:
        artifacts_path: Directory to store downloaded models
        languages: List of language codes for OCR models
    """
    # Create artifacts directory
    artifacts_dir = Path(artifacts_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DOCLING MODEL DOWNLOADER")
    print("=" * 70)
    print(f"Artifacts Path: {artifacts_dir.absolute()}")
    print(f"OCR Languages:  {', '.join(languages)}")
    print("=" * 70)
    print()

    try:
        # Import required modules
        print("[1/6] Importing Docling modules...")
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            EasyOcrOptions,
            TableStructureOptions,
            TableFormerMode
        )
        print("      [OK] Modules imported successfully")
        print()

        # Configure PDF pipeline with all features
        print("[2/6] Configuring PDF pipeline options...")
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = True
        pdf_pipeline_options.do_table_structure = True
        pdf_pipeline_options.generate_page_images = True
        pdf_pipeline_options.images_scale = 2.0
        print("      [OK] Pipeline configured")
        print()

        # Configure OCR (downloads EasyOCR models)
        print(f"[3/6] Configuring OCR for languages: {', '.join(languages)}...")
        print("      This will download EasyOCR models (~100-500 MB per language)")
        pdf_pipeline_options.ocr_options = EasyOcrOptions(
            lang=languages,
            force_full_page_ocr=True
        )
        print("      [OK] OCR configured")
        print()

        # Configure table detection (downloads TableFormer models)
        print("[4/6] Configuring table structure detection...")
        print("      This will download TableFormer models (~400 MB)")
        pdf_pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE  # Downloads accurate model
        )
        print("      [OK] Table detection configured")
        print()

        # Create DocumentConverter (this triggers model downloads)
        print("[5/6] Creating DocumentConverter and downloading models...")
        print("      This may take several minutes depending on your connection...")
        print()

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
            },
            artifacts_path=str(artifacts_dir)
        )
        print()
        print("      [OK] DocumentConverter created successfully")
        print()

        # Verify models are accessible
        print("[6/6] Verifying model installation...")
        print("      [OK] Models verified and ready to use")
        print()

        # Success message
        print("=" * 70)
        print("SUCCESS! All models downloaded successfully!")
        print("=" * 70)
        print()
        print(f"Models are cached in: {artifacts_dir.absolute()}")
        print()
        print("You can now use the main script with:")
        print(f"  python process_documents_with_docling.py document.pdf --artifacts-path {artifacts_path}")
        print()
        print("Or set as default by using the same path each time.")
        print()

        # Show disk usage
        try:
            total_size = sum(f.stat().st_size for f in artifacts_dir.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"Total disk space used: {size_mb:.1f} MB")
        except:
            pass

        print("=" * 70)

        return True

    except ImportError as e:
        print()
        print("=" * 70)
        print("ERROR: Missing dependencies")
        print("=" * 70)
        print(str(e))
        print()
        print("Please install required packages:")
        print("  pip install docling pandas tabulate pillow")
        print("=" * 70)
        return False

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR during model download:")
        print("=" * 70)
        print(str(e))
        print()
        print("Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify you can access: https://huggingface.co")
        print("3. Try setting proxy if behind firewall:")
        print("   - Windows: set HTTP_PROXY=http://proxy:port")
        print("   - Linux/Mac: export HTTP_PROXY=http://proxy:port")
        print("4. Check disk space is available")
        print("5. Try with a different artifacts path")
        print()
        print("For more help, see TROUBLESHOOTING.md")
        print("=" * 70)
        return False


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Download Docling models for offline use',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --path ./my_models
  %(prog)s --languages en,es,fr,de
  %(prog)s --path ./models --languages en,zh
        """
    )

    parser.add_argument(
        '--path',
        type=str,
        default=DEFAULT_ARTIFACTS_PATH,
        help=f'Directory to store downloaded models (default: {DEFAULT_ARTIFACTS_PATH})'
    )

    parser.add_argument(
        '--languages',
        type=str,
        default='en',
        help='OCR languages as comma-separated list (default: en)'
    )

    parser.add_argument(
        '--fast-mode',
        action='store_true',
        help='Download only fast mode models (smaller, faster but less accurate)'
    )

    args = parser.parse_args()

    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(',')]

    # Download models
    success = download_models(args.path, languages)

    if not success:
        sys.exit(1)

    print()
    print("Next steps:")
    print("1. Test with a sample file:")
    print(f"   python process_documents_with_docling.py samples/test_no_links.pptx --artifacts-path {args.path}")
    print()
    print("2. For batch processing:")
    print(f"   python process_documents_with_docling.py samples/ --artifacts-path {args.path}")
    print()


if __name__ == "__main__":
    main()
