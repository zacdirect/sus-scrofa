#!/usr/bin/env python3
"""
Smart renaming script for SusScrofa → SusScrofa migration.

Handles different naming conventions:
- Python modules/packages: sus_scrofa → sus_scrofa (underscore)
- CSS classes/IDs: sus_scrofa- → sus-scrofa- (hyphen prefix)
- Database migrations: keep existing, add new as sus_scrofa
- HTML text content: SusScrofa → SusScrofa (spaces, title case)
- File paths: sus_scrofa/ → sus_scrofa/
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Patterns and their replacements by context
RENAME_RULES = {
    'python_code': [
        # Python imports, module names
        (r'\bghiro\b', 'sus_scrofa'),
        (r'\bGhiro\b', 'SusScrofa'),
        (r'from sus_scrofa', 'from sus_scrofa'),
        (r'import sus_scrofa', 'import sus_scrofa'),
        (r"'sus_scrofa\.", "'sus_scrofa."),
        (r'"sus_scrofa\.', '"sus_scrofa.'),
    ],
    'css': [
        # CSS classes and IDs - keep hyphens for multi-word
        (r'\.sus_scrofa-', '.sus-scrofa-'),
        (r'#sus_scrofa-', '#sus-scrofa-'),
        # Don't touch Bootstrap standard classes like .thumbnails
    ],
    'html_classes': [
        # HTML class attributes - critical to not break standard Bootstrap classes
        # Only rename custom sus_scrofa-prefixed classes
        (r'class="([^"]*?)sus_scrofa-', r'class="\1sus-scrofa-'),
        (r"class='([^']*?)sus_scrofa-", r"class='\1sus-scrofa-"),
        # Fix invalid space-separated class names (like "sus_scrofa-thumbnails")
        (r'class="sus_scrofa-', 'class="sus-scrofa-'),
        (r"class='sus_scrofa-", "class='sus-scrofa-"),
    ],
    'html_text': [
        # Display text - keep spaces and title case
        (r'\bGhiro\b', 'SusScrofa'),
        (r'\bghiro\b', 'sus_scrofa'),
    ],
    'urls': [
        # URL paths
        (r'/sus_scrofa/', '/sus-scrofa/'),
        (r'sus_scrofa\.', 'sus-scrofa.'),
    ],
}

# File type mappings
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
}

# Skip these files/dirs
SKIP_PATTERNS = [
    '__pycache__',
    '.git',
    '.venv',
    'node_modules',
    'static/css/bootstrap',  # Don't touch Bootstrap CSS
    'static/css/font-awesome',
    'static/js/lib',  # Third-party JS
    '.pyc',
    'migrations/0001_initial.py',  # Keep initial migrations as-is
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
                print(f"  [{context}] {change_count} × {pattern} → {replacement}")
                modified = new_modified
    
    return modified, changes


def process_file(filepath: Path, dry_run: bool = True) -> bool:
    """Process a single file."""
    if should_skip(filepath):
        return False
    
    suffix = filepath.suffix
    if suffix not in FILE_CONTEXTS:
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            original = f.read()
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
        return False
    
    contexts = FILE_CONTEXTS[suffix]
    modified, changes = apply_renames(original, contexts, str(filepath))
    
    if changes > 0:
        print(f"\n📝 {filepath} ({changes} changes)")
        
        if not dry_run:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(modified)
                print(f"✅ Written")
            except Exception as e:
                print(f"⚠️  Error writing {filepath}: {e}")
                return False
        else:
            print(f"   (dry run - no changes written)")
        
        return True
    
    return False


def rename_files(root_dir: Path, dry_run: bool = True) -> Dict[str, int]:
    """Rename actual files (like images) from sus_scrofa to sus-scrofa."""
    stats = {
        'files_renamed': 0,
        'files_failed': 0,
    }
    
    # Patterns for files to rename
    file_patterns = [
        '*sus_scrofa*',
        '*SusScrofa*',
    ]
    
    files_to_rename = []
    for pattern in file_patterns:
        files_to_rename.extend(root_dir.rglob(pattern))
    
    for filepath in files_to_rename:
        if not filepath.is_file():
            continue
        
        if should_skip(filepath):
            continue
        
        # Generate new filename
        old_name = filepath.name
        new_name = old_name.replace('sus_scrofa', 'sus-scrofa').replace('SusScrofa', 'Sus-Scrofa')
        
        if old_name == new_name:
            continue
        
        new_path = filepath.parent / new_name
        
        print(f"\n📝 Rename file:")
        print(f"   From: {filepath}")
        print(f"   To:   {new_path}")
        
        if not dry_run:
            try:
                filepath.rename(new_path)
                stats['files_renamed'] += 1
                print(f"✅ Renamed")
            except Exception as e:
                print(f"⚠️  Error: {e}")
                stats['files_failed'] += 1
        else:
            stats['files_renamed'] += 1
            print(f"   (dry run - no changes made)")
    
    return stats


def scan_repository(root_dir: Path, dry_run: bool = True) -> Dict[str, int]:
    """Scan and optionally rename files in repository."""
    stats = {
        'files_scanned': 0,
        'files_modified': 0,
        'files_skipped': 0,
    }
    
    print(f"🔍 Scanning {root_dir}")
    print(f"Mode: {'DRY RUN (no files will be changed)' if dry_run else 'LIVE (files will be modified)'}\n")
    
    for filepath in root_dir.rglob('*'):
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
        description='Smart renaming: SusScrofa → SusScrofa with context-aware rules'
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
    
    args = parser.parse_args()
    
    if args.file:
        # Single file mode
        if not args.file.exists():
            print(f"❌ File not found: {args.file}")
            return 1
        
        print(f"Processing single file: {args.file}")
        process_file(args.file, dry_run=not args.apply)
        return 0
    
    # Directory scan mode
    if not args.directory.exists():
        print(f"❌ Directory not found: {args.directory}")
        return 1
    
    # First: Rename file contents
    stats = scan_repository(args.directory, dry_run=not args.apply)
    
    # Second: Rename actual files
    print("\n" + "="*60)
    print("🔄 Renaming files...")
    file_stats = rename_files(args.directory, dry_run=not args.apply)
    
    print("\n" + "="*60)
    print("📊 Summary:")
    print(f"   Files scanned:  {stats['files_scanned']}")
    print(f"   Files modified: {stats['files_modified']}")
    print(f"   Files skipped:  {stats['files_skipped']}")
    print(f"   Files renamed:  {file_stats['files_renamed']}")
    if file_stats['files_failed'] > 0:
        print(f"   Rename failed:  {file_stats['files_failed']}")
    
    if not args.apply and (stats['files_modified'] > 0 or file_stats['files_renamed'] > 0):
        print("\n⚠️  This was a DRY RUN. No files were changed.")
        print("   Run with --apply to actually modify files.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
