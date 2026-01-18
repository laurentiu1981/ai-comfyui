#!/usr/bin/env python3
"""
CLI script to rename files with sequential counters starting from a given index.

Example usage:
    python rename_files.py --folder=./output --prefix=ComfyUI_ --index=2300
    python rename_files.py --folder=../images --prefix=image --index=100 --suffix=_
    python rename_files.py --folder=/full/path --prefix=photo_ --index=1 --dry-run
"""

import argparse
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename files with sequential counters starting from a given index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --folder=./output --prefix=ComfyUI_ --index=2300
  %(prog)s --folder=../images --prefix=image --index=100 --suffix=_
  %(prog)s --folder=/full/path --prefix=photo_ --index=1 --dry-run
        """
    )
    parser.add_argument(
        "--folder", "-f",
        required=True,
        help="Folder containing files to rename (relative or absolute path)"
    )
    parser.add_argument(
        "--prefix", "-p",
        required=True,
        help="File prefix to match (e.g., 'ComfyUI_')"
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        required=True,
        help="Starting index for renaming"
    )
    parser.add_argument(
        "--suffix", "-s",
        default=None,
        help="Suffix after the counter (e.g., '_'). If not provided, will be auto-detected."
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Show what would be renamed without actually renaming"
    )
    return parser.parse_args()


def resolve_folder(folder_path: str) -> Path:
    """Resolve folder path (handles relative and absolute paths)."""
    path = Path(folder_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def find_matching_files(folder: Path, prefix: str, suffix: str = None):
    """
    Find files matching the pattern: {prefix}{counter}{suffix}.{ext}
    Returns list of tuples: (original_path, counter, suffix, extension, counter_width)
    """
    # Build regex pattern
    # Counter is one or more digits
    # Suffix can be empty or any non-digit characters before the extension
    if suffix is not None:
        # Escape special regex characters in suffix
        escaped_suffix = re.escape(suffix)
        pattern = rf"^{re.escape(prefix)}(\d+){escaped_suffix}(\.[^.]+)$"
    else:
        # Auto-detect suffix: match digits followed by optional non-digit chars before extension
        pattern = rf"^{re.escape(prefix)}(\d+)([^.\d]*)(\.[^.]+)$"

    regex = re.compile(pattern)
    matches = []

    for entry in folder.iterdir():
        if not entry.is_file():
            continue

        match = regex.match(entry.name)
        if match:
            if suffix is not None:
                counter_str = match.group(1)
                ext = match.group(2)
                detected_suffix = suffix
            else:
                counter_str = match.group(1)
                detected_suffix = match.group(2)
                ext = match.group(3)

            counter = int(counter_str)
            counter_width = len(counter_str)
            matches.append((entry, counter, detected_suffix, ext, counter_width))

    # Sort by counter value
    matches.sort(key=lambda x: x[1])
    return matches


def rename_files(matches, folder: Path, prefix: str, start_index: int, dry_run: bool = False):
    """
    Rename files starting from start_index.
    Returns statistics dict.
    """
    stats = {
        "total_found": len(matches),
        "renamed": 0,
        "skipped": 0,
        "errors": [],
    }

    if not matches:
        return stats

    # Determine counter width from the first file
    counter_width = matches[0][4]

    # First pass: rename to temporary names to avoid conflicts
    temp_names = []
    for idx, (file_path, old_counter, suffix, ext, _) in enumerate(matches):
        temp_name = f"__temp_rename_{idx}_{file_path.name}"
        temp_path = folder / temp_name
        temp_names.append((file_path, temp_path, suffix, ext))

        if not dry_run:
            try:
                file_path.rename(temp_path)
            except PermissionError as e:
                stats["errors"].append({
                    "file": str(file_path),
                    "error": f"Permission denied: {e}",
                    "suggestion": "Check file permissions with 'ls -la'. You may need to run 'chmod u+w <file>' or run the script with appropriate permissions."
                })
                stats["skipped"] += 1
                temp_names[-1] = (file_path, None, suffix, ext)  # Mark as failed
            except OSError as e:
                stats["errors"].append({
                    "file": str(file_path),
                    "error": str(e),
                    "suggestion": "Check if the file is in use by another process or if the filesystem is read-only."
                })
                stats["skipped"] += 1
                temp_names[-1] = (file_path, None, suffix, ext)

    # Second pass: rename from temp to final names
    for idx, (original_path, temp_path, suffix, ext) in enumerate(temp_names):
        if temp_path is None:
            continue  # Skip files that failed in first pass

        new_counter = start_index + idx
        new_name = f"{prefix}{new_counter:0{counter_width}d}{suffix}{ext}"
        new_path = folder / new_name

        if dry_run:
            print(f"  {original_path.name} -> {new_name}")
            stats["renamed"] += 1
        else:
            try:
                temp_path.rename(new_path)
                print(f"  {original_path.name} -> {new_name}")
                stats["renamed"] += 1
            except PermissionError as e:
                # Try to restore original name
                try:
                    temp_path.rename(original_path)
                except:
                    pass
                stats["errors"].append({
                    "file": str(original_path),
                    "error": f"Permission denied: {e}",
                    "suggestion": "Check folder write permissions with 'ls -la'. You may need 'chmod u+w <folder>'."
                })
                stats["skipped"] += 1
            except OSError as e:
                try:
                    temp_path.rename(original_path)
                except:
                    pass
                stats["errors"].append({
                    "file": str(original_path),
                    "error": str(e),
                    "suggestion": "Check if disk is full or filesystem has issues."
                })
                stats["skipped"] += 1

    return stats


def print_statistics(stats: dict, dry_run: bool):
    """Print summary statistics."""
    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)

    if dry_run:
        print("  Mode: DRY RUN (no files were actually renamed)")

    print(f"  Files found matching pattern: {stats['total_found']}")
    print(f"  Files {'would be ' if dry_run else ''}renamed: {stats['renamed']}")

    if stats['skipped'] > 0:
        print(f"  Files skipped due to errors: {stats['skipped']}")

    if stats['errors']:
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
    print(f"Looking for files with prefix: '{args.prefix}'")
    if args.suffix is not None:
        print(f"Using suffix: '{args.suffix}'")
    else:
        print("Suffix: auto-detect")
    print(f"Starting index: {args.index}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be renamed ***\n")

    # Find matching files
    matches = find_matching_files(folder, args.prefix, args.suffix)

    if not matches:
        print(f"\nNo files found matching pattern: {args.prefix}<counter>{args.suffix or '<suffix>'}.<ext>")
        sys.exit(0)

    # Show detected suffix if auto-detected
    if args.suffix is None and matches:
        detected_suffix = matches[0][2]
        print(f"Auto-detected suffix: '{detected_suffix}'")

    print(f"\nFound {len(matches)} matching file(s). Renaming...\n")

    # Rename files
    stats = rename_files(matches, folder, args.prefix, args.index, args.dry_run)

    # Print statistics
    print_statistics(stats, args.dry_run)

    # Exit with error code if there were errors
    if stats['errors']:
        sys.exit(2)


if __name__ == "__main__":
    main()
