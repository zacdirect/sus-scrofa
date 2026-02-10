# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import os
import re
from pathlib import Path
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Auto-fix simple legacy template patterns'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be changed without making changes',
        )

    def handle(self, *args, **options):
        templates_dir = Path(os.getcwd()) / 'templates'
        dry_run = options['dry_run']
        
        # Define replacement patterns
        replacements = [
            # Bootstrap 2 grid to Bootstrap 5
            {
                'pattern': r'\bspan12\b',
                'replacement': 'col-12',
                'description': 'Bootstrap grid: span12 → col-12',
            },
            {
                'pattern': r'\bspan11\b',
                'replacement': 'col-11',
                'description': 'Bootstrap grid: span11 → col-11',
            },
            {
                'pattern': r'\bspan10\b',
                'replacement': 'col-10',
                'description': 'Bootstrap grid: span10 → col-10',
            },
            {
                'pattern': r'\bspan9\b',
                'replacement': 'col-9',
                'description': 'Bootstrap grid: span9 → col-9',
            },
            {
                'pattern': r'\bspan8\b',
                'replacement': 'col-8',
                'description': 'Bootstrap grid: span8 → col-8',
            },
            {
                'pattern': r'\bspan6\b',
                'replacement': 'col-6',
                'description': 'Bootstrap grid: span6 → col-6',
            },
            {
                'pattern': r'\bspan4\b',
                'replacement': 'col-4',
                'description': 'Bootstrap grid: span4 → col-4',
            },
            {
                'pattern': r'\bspan3\b',
                'replacement': 'col-3',
                'description': 'Bootstrap grid: span3 → col-3',
            },
            {
                'pattern': r'\brow-fluid\b',
                'replacement': 'row',
                'description': 'Bootstrap grid: row-fluid → row',
            },
            # Icon replacements (common ones)
            {
                'pattern': r'\bicon-home\b',
                'replacement': 'fa-solid fa-house',
                'description': 'Icon: icon-home → fa-solid fa-house',
            },
            {
                'pattern': r'\bicon-search\b',
                'replacement': 'fa-solid fa-magnifying-glass',
                'description': 'Icon: icon-search → fa-solid fa-magnifying-glass',
            },
            {
                'pattern': r'\bicon-lock\b',
                'replacement': 'fa-solid fa-lock',
                'description': 'Icon: icon-lock → fa-solid fa-lock',
            },
            {
                'pattern': r'\bicon-user\b',
                'replacement': 'fa-solid fa-user',
                'description': 'Icon: icon-user → fa-solid fa-user',
            },
            {
                'pattern': r'\bicon-folder-open\b',
                'replacement': 'fa-solid fa-folder-open',
                'description': 'Icon: icon-folder-open → fa-solid fa-folder-open',
            },
            {
                'pattern': r'\bicon-folder-close\b',
                'replacement': 'fa-solid fa-folder',
                'description': 'Icon: icon-folder-close → fa-solid fa-folder',
            },
            {
                'pattern': r'\bicon-picture\b',
                'replacement': 'fa-solid fa-image',
                'description': 'Icon: icon-picture → fa-solid fa-image',
            },
            {
                'pattern': r'\bicon-plus\b',
                'replacement': 'fa-solid fa-plus',
                'description': 'Icon: icon-plus → fa-solid fa-plus',
            },
            {
                'pattern': r'\bicon-trash\b',
                'replacement': 'fa-solid fa-trash',
                'description': 'Icon: icon-trash → fa-solid fa-trash',
            },
            {
                'pattern': r'\bicon-edit\b',
                'replacement': 'fa-solid fa-pen-to-square',
                'description': 'Icon: icon-edit → fa-solid fa-pen-to-square',
            },
            {
                'pattern': r'\bicon-remove\b',
                'replacement': 'fa-solid fa-xmark',
                'description': 'Icon: icon-remove → fa-solid fa-xmark',
            },
            {
                'pattern': r'\bicon-ok\b',
                'replacement': 'fa-solid fa-check',
                'description': 'Icon: icon-ok → fa-solid fa-check',
            },
            {
                'pattern': r'\bicon-download\b',
                'replacement': 'fa-solid fa-download',
                'description': 'Icon: icon-download → fa-solid fa-download',
            },
            {
                'pattern': r'\bicon-upload\b',
                'replacement': 'fa-solid fa-upload',
                'description': 'Icon: icon-upload → fa-solid fa-upload',
            },
            {
                'pattern': r'\bicon-list\b',
                'replacement': 'fa-solid fa-list',
                'description': 'Icon: icon-list → fa-solid fa-list',
            },
            {
                'pattern': r'\bicon-th\b',
                'replacement': 'fa-solid fa-table-cells',
                'description': 'Icon: icon-th → fa-solid fa-table-cells',
            },
            {
                'pattern': r'\bicon-globe\b',
                'replacement': 'fa-solid fa-globe',
                'description': 'Icon: icon-globe → fa-solid fa-globe',
            },
            {
                'pattern': r'\bicon-cog\b',
                'replacement': 'fa-solid fa-gear',
                'description': 'Icon: icon-cog → fa-solid fa-gear',
            },
            {
                'pattern': r'\bicon-refresh\b',
                'replacement': 'fa-solid fa-rotate',
                'description': 'Icon: icon-refresh → fa-solid fa-rotate',
            },
            {
                'pattern': r'\bicon-eye-open\b',
                'replacement': 'fa-solid fa-eye',
                'description': 'Icon: icon-eye-open → fa-solid fa-eye',
            },
            {
                'pattern': r'\bicon-eye-close\b',
                'replacement': 'fa-solid fa-eye-slash',
                'description': 'Icon: icon-eye-close → fa-solid fa-eye-slash',
            },
            {
                'pattern': r'\bicon-file\b',
                'replacement': 'fa-solid fa-file',
                'description': 'Icon: icon-file → fa-solid fa-file',
            },
            {
                'pattern': r'\bicon-time\b',
                'replacement': 'fa-solid fa-clock',
                'description': 'Icon: icon-time → fa-solid fa-clock',
            },
            {
                'pattern': r'\bicon-calendar\b',
                'replacement': 'fa-solid fa-calendar',
                'description': 'Icon: icon-calendar → fa-solid fa-calendar',
            },
            {
                'pattern': r'\bicon-comment\b',
                'replacement': 'fa-solid fa-comment',
                'description': 'Icon: icon-comment → fa-solid fa-comment',
            },
            {
                'pattern': r'\bicon-tag\b',
                'replacement': 'fa-solid fa-tag',
                'description': 'Icon: icon-tag → fa-solid fa-tag',
            },
            {
                'pattern': r'\bicon-tags\b',
                'replacement': 'fa-solid fa-tags',
                'description': 'Icon: icon-tags → fa-solid fa-tags',
            },
            {
                'pattern': r'\bicon-info-sign\b',
                'replacement': 'fa-solid fa-circle-info',
                'description': 'Icon: icon-info-sign → fa-solid fa-circle-info',
            },
            {
                'pattern': r'\bicon-warning-sign\b',
                'replacement': 'fa-solid fa-triangle-exclamation',
                'description': 'Icon: icon-warning-sign → fa-solid fa-triangle-exclamation',
            },
            {
                'pattern': r'\bicon-question-sign\b',
                'replacement': 'fa-solid fa-circle-question',
                'description': 'Icon: icon-question-sign → fa-solid fa-circle-question',
            },
            # Breadcrumb dividers
            {
                'pattern': r'<span class="divider">&rsaquo;</span>',
                'replacement': '',
                'description': 'Remove old breadcrumb dividers',
            },
            {
                'pattern': r'<span class="divider">›</span>',
                'replacement': '',
                'description': 'Remove old breadcrumb dividers',
            },
        ]
        
        files_modified = 0
        total_changes = 0
        changes_by_file = {}
        
        # Process all HTML templates
        for template_file in templates_dir.rglob('*.html'):
            relative_path = template_file.relative_to(templates_dir)
            
            try:
                original_content = template_file.read_text()
                modified_content = original_content
                file_changes = []
                
                # Apply all replacements
                for replacement in replacements:
                    pattern = replacement['pattern']
                    new_text = replacement['replacement']
                    
                    # Count matches before replacement
                    matches = len(re.findall(pattern, modified_content))
                    
                    if matches > 0:
                        modified_content = re.sub(pattern, new_text, modified_content)
                        file_changes.append({
                            'description': replacement['description'],
                            'count': matches,
                        })
                        total_changes += matches
                
                # Write changes if content was modified
                if modified_content != original_content:
                    files_modified += 1
                    changes_by_file[str(relative_path)] = file_changes
                    
                    if not dry_run:
                        template_file.write_text(modified_content)
                        
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing {relative_path}: {e}'))
        
        # Report results
        self.stdout.write('\n' + '='*80)
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - No files were modified'))
        else:
            self.stdout.write(self.style.SUCCESS('Auto-Fix Results'))
        self.stdout.write('='*80 + '\n')
        
        if files_modified == 0:
            self.stdout.write(self.style.SUCCESS('✓ No changes needed!'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Modified {files_modified} files with {total_changes} total changes\n'))
        
        # Show details for each file
        for file_path, changes in sorted(changes_by_file.items()):
            self.stdout.write(self.style.HTTP_INFO(f'\n{file_path}:'))
            for change in changes:
                self.stdout.write(f"  • {change['description']} ({change['count']} occurrences)")
        
        self.stdout.write('\n' + '='*80)
        if dry_run:
            self.stdout.write(self.style.WARNING('\nRun without --dry-run to apply these changes'))
        else:
            self.stdout.write(self.style.SUCCESS('\n✓ Changes applied successfully!'))
            self.stdout.write('\nRun "python manage.py lint_templates" to check for remaining issues')
        self.stdout.write('='*80 + '\n')
