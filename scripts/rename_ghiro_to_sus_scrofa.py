#!/usr/bin/env python3
"""
Smart renaming script for Ghiro → SusScrofa migration.

Handles different naming conventions:
- Python modules/packages: ghiro → sus_scrofa (underscore)
- Python classes/display: Ghiro → SusScrofa
- Python constants: GHIRO → SUS_SCROFA
- Exception classes: GhiroException → SusScrofaException
- CSS classes/IDs: ghiro- → sus-scrofa- (hyphen prefix)
- Database names: ghirodb → sus_scrofa_db
- Container names: ghiro-mongodb → sus-scrofa-mongodb
- HTML text content: Ghiro → SusScrofa (title case)
- File paths: ghiro/ → sus_scrofa/
- Output folders: ghiro_output → sus_scrofa_output
"""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ─── Self-protection ────────────────────────────────────────────────
# This script must skip itself to avoid corrupting its own regex patterns.
SELF_NAME = "rename_ghiro_to_sus_scrofa.py"

# ─── Patterns and their replacements by context ────────────────────
RENAME_RULES = {
    'python_code': [
        # Exception classes (must come before generic Ghiro→SusScrofa)
        (r'\bGhiroValidationException\b', 'SusScrofaValidationException'),
        (r'\bGhiroPluginException\b', 'SusScrofaPluginException'),
        (r'\bGhiroException\b', 'SusScrofaException'),
        # Constants
        (r'\bGHIRO_VERSION\b', 'SUS_SCROFA_VERSION'),
        (r'\bGHIRO\b', 'SUS_SCROFA'),
        # Context processor and template variable
        (r'\bghiro_release\b', 'sus_scrofa_release'),
        # Database name
        (r'\bghirodb\b', 'sus_scrofa_db'),
        # Output folder
        (r'\bghiro_output\b', 'sus_scrofa_output'),
        # Python identifiers with ghiro_ prefix (ghiro_path, ghiro_dir, etc.)
        (r'\bghiro_', 'sus_scrofa_'),
        (r'\bGhiro_', 'SusScrofa_'),
        # Python imports, module names (generic catch-all, last)
        (r'\bghiro\b', 'sus_scrofa'),
        (r'\bGhiro\b', 'SusScrofa'),
        # Fixup: ensure already-correct imports stay correct
        (r'from sus_scrofa', 'from sus_scrofa'),
        (r'import sus_scrofa', 'import sus_scrofa'),
        (r"'sus_scrofa\.", "'sus_scrofa."),
        (r'"sus_scrofa\.', '"sus_scrofa.'),
    ],
    'makefile': [
        # Container and network names (use hyphens)
        (r'\bghiro-mongodb\b', 'sus-scrofa-mongodb'),
        (r'\bghiro-opencv\b', 'sus-scrofa-opencv'),
        (r'\bghiro-net\b', 'sus-scrofa-net'),
        # Database name in connection strings
        (r'\bghirodb\b', 'sus_scrofa_db'),
        # Echo/display text
        (r'\bGhiro\b', 'SusScrofa'),
        (r'\bghiro\b', 'sus_scrofa'),
        (r'\bGHIRO\b', 'SUS_SCROFA'),
    ],
    'css': [
        # CSS classes and IDs - keep hyphens for multi-word
        (r'\.ghiro-', '.sus-scrofa-'),
        (r'#ghiro-', '#sus-scrofa-'),
        # Generic in CSS
        (r'\bghiro\b', 'sus-scrofa'),
        (r'\bGhiro\b', 'SusScrofa'),
    ],
    'html_classes': [
        # HTML class attributes - only rename custom ghiro-prefixed classes
        (r'class="([^"]*?)ghiro-', r'class="\1sus-scrofa-'),
        (r"class='([^']*?)ghiro-", r"class='\1sus-scrofa-"),
        (r'class="ghiro-', 'class="sus-scrofa-'),
        (r"class='ghiro-", "class='sus-scrofa-"),
    ],
    'html_text': [
        # Display text - title case for display, lowercase for identifiers
        (r'\bGHIRO\b', 'SUS SCROFA'),
        (r'\bGhiro\b', 'SusScrofa'),
        (r'\bghiro\b', 'sus_scrofa'),
    ],
    'urls': [
        # URL paths
        (r'/ghiro/', '/sus-scrofa/'),
        (r'ghiro\.', 'sus-scrofa.'),
    ],
    'rst': [
        # ReStructuredText docs
        (r'\bGhiro\b', 'SusScrofa'),
        (r'\bghiro\b', 'sus_scrofa'),
        (r'\bGHIRO\b', 'SUS_SCROFA'),
    ],
    'migration_comments': [
        # Only touch comments in migration files, not the actual migration code
        (r'# This file is part of Ghiro\.', '# This file is part of SusScrofa.'),
        (r'\bGhiro\b', 'SusScrofa'),
        (r'\bghiro\b', 'sus_scrofa'),
    ],
}

# ─── File type mappings ─────────────────────────────────────────────
FILE_CONTEXTS = {
    '.py': ['python_code'],
    '.css': ['css'],
    '.html': ['html_classes', 'html_text'],
    '.js': ['html_text', 'urls'],
    '.md': ['html_text'],
    '.txt': ['html_text'],
    '.json': ['python_code'],
    '.yml': ['python_code'],
    '.yaml': ['python_code'],
    '.rst': ['rst'],
}

# Files without extensions that need processing
EXTENSIONLESS_FILES = {
    'Makefile': ['makefile'],
}

# ─── Skip these files/dirs ──────────────────────────────────────────
SKIP_PATTERNS = [
    '__pycache__',
    '.git',
    '.venv',
    'node_modules',
    'static/css/bootstrap',      # Don't touch Bootstrap CSS
    'static/css/font-awesome',
    'static/js/lib',             # Third-party JS
    '.pyc',
    'migrations/0001_initial.py',  # Keep initial migration as-is
    SELF_NAME,                     # Don't process ourselves!
    # ─── Attribution / upstream project docs ─────────────────────────
    # These reference "Ghiro" as the original project name intentionally.
    'docs/AUTHORS.txt',            # Credits to original Ghiro developers
    'docs/Changelog.txt',          # Original Ghiro release history
    'docs/src/',                    # Original Ghiro user manual (149 refs)
    'README.md',                    # "fork of ... Ghiro" attribution
]


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    path_str = str(path)
    return any(skip in path_str for skip in SKIP_PATTERNS)


def apply_renames(content: str, contexts: List[str], filepath: str) -> Tuple[str, int]:
    """Apply rename rules for given contexts."""
    modified = content
    changes = 0

    for context in contexts:
        if context not in RENAME_RULES:
            continue

        for pattern, replacement in RENAME_RULES[context]:
            new_modified = re.sub(pattern, replacement, modified)
            if new_modified != modified:
                change_count = len(re.findall(pattern, modified))
                changes += change_count
                print(f"  [{context}] {change_count} x {pattern} -> {replacement}")
                modified = new_modified

    return modified, changes


def get_file_contexts(filepath: Path) -> List[str]:
    """Determine which rename contexts apply to a file."""
    # Check extensionless files by name
    if filepath.name in EXTENSIONLESS_FILES:
        return EXTENSIONLESS_FILES[filepath.name]

    # Check by extension
    suffix = filepath.suffix
    if suffix in FILE_CONTEXTS:
        return FILE_CONTEXTS[suffix]

    return []


def process_file(filepath: Path, dry_run: bool = True) -> bool:
    """Process a single file."""
    if should_skip(filepath):
        return False

    contexts = get_file_contexts(filepath)
    if not contexts:
        return False

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            original = f.read()
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")
        return False

    modified, changes = apply_renames(original, contexts, str(filepath))

    if changes > 0:
        print(f"\n>> {filepath} ({changes} changes)")

        if not dry_run:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(modified)
                print(f"   Written OK")
            except Exception as e:
                print(f"Warning: Error writing {filepath}: {e}")
                return False
        else:
            print(f"   (dry run - no changes written)")

        return True

    return False


def rename_files(root_dir: Path, dry_run: bool = True) -> Dict[str, int]:
    """Rename actual files/dirs that have 'ghiro' in their name."""
    stats = {
        'files_renamed': 0,
        'files_failed': 0,
    }

    # Patterns for files to rename
    file_patterns = [
        '*ghiro*',
        '*Ghiro*',
    ]

    files_to_rename = []
    for pattern in file_patterns:
        files_to_rename.extend(root_dir.rglob(pattern))

    # Sort longest paths first so we rename leaf nodes before parents
    for filepath in sorted(files_to_rename, key=lambda p: len(str(p)), reverse=True):
        if not filepath.is_file():
            continue

        if should_skip(filepath):
            continue

        # Generate new filename
        old_name = filepath.name
        new_name = (old_name
                    .replace('ghiro', 'sus_scrofa')
                    .replace('Ghiro', 'SusScrofa')
                    .replace('GHIRO', 'SUS_SCROFA'))

        if old_name == new_name:
            continue

        new_path = filepath.parent / new_name

        print(f"\n>> Rename file:")
        print(f"   From: {filepath}")
        print(f"   To:   {new_path}")

        if not dry_run:
            try:
                filepath.rename(new_path)
                stats['files_renamed'] += 1
                print(f"   Renamed OK")
            except Exception as e:
                print(f"Warning: Error: {e}")
                stats['files_failed'] += 1
        else:
            stats['files_renamed'] += 1
            print(f"   (dry run - no changes made)")

    return stats


def remove_dead_ghiro_dir(root_dir: Path, dry_run: bool = True) -> bool:
    """Remove the dead ghiro/ settings directory if sus_scrofa/ exists."""
    ghiro_dir = root_dir / 'ghiro'
    sus_scrofa_dir = root_dir / 'sus_scrofa'

    if not ghiro_dir.is_dir():
        print("\n   No ghiro/ directory found (already removed)")
        return False

    if not sus_scrofa_dir.is_dir():
        print("\nWarning: ghiro/ exists but sus_scrofa/ doesn't -- skipping removal for safety")
        return False

    print(f"\n   Remove dead directory: {ghiro_dir}/")

    # List contents for confirmation
    contents = list(ghiro_dir.rglob('*'))
    file_count = sum(1 for c in contents if c.is_file())
    print(f"   Contains {file_count} file(s)")

    if not dry_run:
        try:
            shutil.rmtree(ghiro_dir)
            print(f"   Removed OK")
            return True
        except Exception as e:
            print(f"Warning: Error: {e}")
            return False
    else:
        print(f"   (dry run - directory not removed)")
        return True


def scan_repository(root_dir: Path, dry_run: bool = True) -> Dict[str, int]:
    """Scan and optionally rename files in repository."""
    stats = {
        'files_scanned': 0,
        'files_modified': 0,
        'files_skipped': 0,
    }

    print(f"Scanning {root_dir}")
    print(f"Mode: {'DRY RUN (no files will be changed)' if dry_run else 'LIVE (files will be modified)'}\n")

    for filepath in sorted(root_dir.rglob('*')):
        if not filepath.is_file():
            continue

        if should_skip(filepath):
            stats['files_skipped'] += 1
            continue

        stats['files_scanned'] += 1

        if process_file(filepath, dry_run):
            stats['files_modified'] += 1

    return stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Smart renaming: Ghiro -> SusScrofa with context-aware rules'
    )
    parser.add_argument(
        'directory',
        type=Path,
        nargs='?',
        default=Path.cwd(),
        help='Root directory to scan (default: current directory)'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually apply changes (default is dry run)'
    )
    parser.add_argument(
        '--file',
        type=Path,
        help='Process a single file instead of scanning'
    )
    parser.add_argument(
        '--no-delete-ghiro-dir',
        action='store_true',
        help='Do not remove the dead ghiro/ directory'
    )

    args = parser.parse_args()

    if args.file:
        # Single file mode
        if not args.file.exists():
            print(f"File not found: {args.file}")
            return 1

        print(f"Processing single file: {args.file}")
        process_file(args.file, dry_run=not args.apply)
        return 0

    # Directory scan mode
    if not args.directory.exists():
        print(f"Directory not found: {args.directory}")
        return 1

    # Step 1: Rename file contents
    print("=" * 60)
    print("Step 1: Renaming file contents...")
    print("=" * 60)
    stats = scan_repository(args.directory, dry_run=not args.apply)

    # Step 2: Rename actual files
    print("\n" + "=" * 60)
    print("Step 2: Renaming files with 'ghiro' in the name...")
    print("=" * 60)
    file_stats = rename_files(args.directory, dry_run=not args.apply)

    # Step 3: Remove dead ghiro/ directory
    if not args.no_delete_ghiro_dir:
        print("\n" + "=" * 60)
        print("Step 3: Removing dead ghiro/ directory...")
        print("=" * 60)
        remove_dead_ghiro_dir(args.directory, dry_run=not args.apply)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"   Files scanned:  {stats['files_scanned']}")
    print(f"   Files modified: {stats['files_modified']}")
    print(f"   Files skipped:  {stats['files_skipped']}")
    print(f"   Files renamed:  {file_stats['files_renamed']}")
    if file_stats['files_failed'] > 0:
        print(f"   Rename failed:  {file_stats['files_failed']}")

    if not args.apply and (stats['files_modified'] > 0 or file_stats['files_renamed'] > 0):
        print("\n** This was a DRY RUN. No files were changed.")
        print("   Run with --apply to actually modify files.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
