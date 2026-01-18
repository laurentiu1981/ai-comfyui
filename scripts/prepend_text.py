#!/usr/bin/env python3
"""
CLI script to prepend text to all .txt files in a folder.
Useful for adding trigger words to training caption files.

Example usage:
    python3 prepend_text.py --folder=./input/dataset --text="[trigger], "
    python3 prepend_text.py --folder=./captions --text="photo of sks person, " --dry-run
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepend text to all .txt files in a folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --folder=./input/dataset --text="[trigger], "
  %(prog)s --folder=./captions --text="photo of sks person, " --dry-run
  %(prog)s -f ./data -t "prefix: " --skip-empty
        """
    )
    parser.add_argument(
        "--folder", "-f",
        required=True,
        help="Folder containing .txt files (relative or absolute path)"
    )
    parser.add_argument(
        "--text", "-t",
        required=True,
        help="Text to prepend to each file (e.g., '[trigger], ')"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Preview changes without modifying files"
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip empty .txt files"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already start with the prepend text"
    )
    return parser.parse_args()


def resolve_folder(folder_path: str) -> Path:
    """Resolve folder path (handles relative and absolute paths)."""
    path = Path(folder_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def prepend_to_txt_files(folder: Path, text: str, dry_run: bool = False,
                          skip_empty: bool = False, skip_existing: bool = False):
    """
    Prepend text to all .txt files in folder.
    Returns statistics dict.
    """
    stats = {
        "total_found": 0,
        "updated": 0,
        "skipped_empty": 0,
        "skipped_existing": 0,
        "errors": [],
    }

    txt_files = sorted(folder.glob("*.txt"))
    stats["total_found"] = len(txt_files)

    for file_path in txt_files:
        try:
            # Read existing content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Skip empty files if requested
            if skip_empty and not original_content.strip():
                print(f"  [SKIP-EMPTY] {file_path.name}")
                stats["skipped_empty"] += 1
                continue

            # Skip if already has the prepend text
            if skip_existing and original_content.startswith(text):
                print(f"  [SKIP-EXISTS] {file_path.name}")
                stats["skipped_existing"] += 1
                continue

            # Prepend the text
            new_content = text + original_content

            if dry_run:
                # Show preview
                preview_orig = original_content[:50].replace('\n', '\\n')
                preview_new = new_content[:50].replace('\n', '\\n')
                print(f"  {file_path.name}")
                print(f"    Before: \"{preview_orig}{'...' if len(original_content) > 50 else ''}\"")
                print(f"    After:  \"{preview_new}{'...' if len(new_content) > 50 else ''}\"")
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"  [UPDATED] {file_path.name}")

            stats["updated"] += 1

        except PermissionError as e:
            stats["errors"].append({
                "file": str(file_path),
                "error": f"Permission denied: {e}",
                "suggestion": "Check file permissions with 'ls -la'. You may need 'chmod u+w <file>'."
            })
        except UnicodeDecodeError as e:
            stats["errors"].append({
                "file": str(file_path),
                "error": f"Encoding error: {e}",
                "suggestion": "File may not be UTF-8 encoded. Check the file encoding."
            })
        except OSError as e:
            stats["errors"].append({
                "file": str(file_path),
                "error": str(e),
                "suggestion": "Check if disk is full or file is in use."
            })

    return stats


def print_statistics(stats: dict, dry_run: bool):
    """Print summary statistics."""
    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)

    if dry_run:
        print("  Mode: DRY RUN (no files were modified)")

    print(f"  Total .txt files found: {stats['total_found']}")
    print(f"  Files {'would be ' if dry_run else ''}updated: {stats['updated']}")

    if stats['skipped_empty'] > 0:
        print(f"  Files skipped (empty): {stats['skipped_empty']}")

    if stats['skipped_existing'] > 0:
        print(f"  Files skipped (already has text): {stats['skipped_existing']}")

    if stats['errors']:
        print(f"  Files with errors: {len(stats['errors'])}")
        print("\n" + "-" * 50)
        print("ERRORS:")
        print("-" * 50)
        for error in stats['errors']:
            print(f"\n  File: {error['file']}")
            print(f"  Error: {error['error']}")
            print(f"  Suggestion: {error['suggestion']}")

    print("=" * 50)


def main():
    args = parse_args()

    # Resolve folder path
    folder = resolve_folder(args.folder)

    # Validate folder exists
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder}", file=sys.stderr)
        sys.exit(1)

    if not folder.is_dir():
        print(f"Error: Path is not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning folder: {folder}")
    print(f"Text to prepend: \"{args.text}\"")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***\n")

    # Process files
    stats = prepend_to_txt_files(
        folder=folder,
        text=args.text,
        dry_run=args.dry_run,
        skip_empty=args.skip_empty,
        skip_existing=args.skip_existing
    )

    if stats["total_found"] == 0:
        print("\nNo .txt files found in the folder.")
        sys.exit(0)

    # Print statistics
    print_statistics(stats, args.dry_run)

    # Exit with error code if there were errors
    if stats['errors']:
        sys.exit(2)


if __name__ == "__main__":
    main()
