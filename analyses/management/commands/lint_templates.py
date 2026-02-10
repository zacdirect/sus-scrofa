# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import os
import re
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings


class Command(BaseCommand):
    help = 'Lint templates for legacy Bootstrap 2/3 patterns and missing modern elements'

    def add_arguments(self, parser):
        parser.add_argument(
            '--fix',
            action='store_true',
            help='Attempt to auto-fix simple issues',
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed output',
        )

    def handle(self, *args, **options):
        # Use current working directory since BASE_DIR isn't defined in settings
        templates_dir = Path(os.getcwd()) / 'templates'
        
        # Patterns to detect legacy code
        legacy_patterns = {
            'bootstrap2_grid': {
                'pattern': r'\b(span\d+|row-fluid)\b',
                'message': 'Legacy Bootstrap 2 grid classes (span*, row-fluid)',
                'severity': 'ERROR',
            },
            'bootstrap2_icons': {
                'pattern': r'\bicon-\w+',
                'message': 'Legacy Bootstrap 2 icon classes (icon-*)',
                'severity': 'ERROR',
            },
            'old_breadcrumb': {
                'pattern': r'<span class="divider">',
                'message': 'Old breadcrumb divider pattern',
                'severity': 'WARNING',
            },
            'old_nav_pills': {
                'pattern': r'<ul class="nav nav-pills"[^>]*>\s*<li\s+class="active">',
                'message': 'Old nav-pills pattern (missing nav-item/nav-link)',
                'severity': 'WARNING',
            },
            'old_tabs': {
                'pattern': r'<ul class="nav nav-tabs"[^>]*>\s*<li\s+class="active">',
                'message': 'Old nav-tabs pattern (missing nav-item/nav-link)',
                'severity': 'WARNING',
            },
            'content_wrapper': {
                'pattern': r'<div class="content content-large">',
                'message': 'Legacy content wrapper (should use Bootstrap 5 grid)',
                'severity': 'ERROR',
            },
            'old_button_classes': {
                'pattern': r'\bbtn-(primary|success|danger|warning|info)\b(?!["\'])',
                'message': 'Check if buttons need btn-sm, btn-lg, or other modern classes',
                'severity': 'INFO',
            },
            'jquery_selectors': {
                'pattern': r'\$\(["\']#',
                'message': 'jQuery selector detected (consider vanilla JS)',
                'severity': 'INFO',
            },
            'has_key_method': {
                'pattern': r'\.has_key\(',
                'message': 'Python 2 has_key() method (use "in" operator)',
                'severity': 'ERROR',
            },
            'pull_classes': {
                'pattern': r'\b(pull-left|pull-right)\b',
                'message': 'Legacy pull classes (use float-start/float-end)',
                'severity': 'WARNING',
            },
            'old_badge_classes': {
                'pattern': r'\bbadge-(important|success|warning|info|inverse)\b',
                'message': 'Legacy badge classes (use bg-danger, bg-success, etc.)',
                'severity': 'WARNING',
            },
            'old_text_classes': {
                'pattern': r'\btext-error\b',
                'message': 'Legacy text-error class (use text-danger)',
                'severity': 'WARNING',
            },
            'dl_horizontal': {
                'pattern': r'\bdl-horizontal\b',
                'message': 'Legacy dl-horizontal class (use row/col grid)',
                'severity': 'INFO',
            },
        }
        
        # Patterns for missing modern elements
        missing_patterns = {
            'no_breadcrumb': {
                'check': lambda content: 'breadcrumb' not in content and '{% block content %}' in content,
                'message': 'Missing breadcrumb navigation',
                'severity': 'WARNING',
            },
            'no_fontawesome': {
                'check': lambda content: 'fa-solid' not in content and 'fa-regular' not in content and '<i class="icon-' in content,
                'message': 'Using old icons instead of Font Awesome 6',
                'severity': 'WARNING',
            },
        }
        
        issues_found = []
        files_checked = 0
        fixed_count = 0
        
        # Files to exclude from linting (base templates, layouts, partials)
        # However, for fixing lightbox, we might want to check partials if they contain image links.
        # But let's stick to the current logic for exclude_patterns for now, unless 'fix' is on?
        # Actually partials like _list_image.html definitely need fixing.
        
        # Let's reduce the strictness of excludes for the fix pass or generally.
        # The user's request implies _list_image.html needs to be fixed.
        # So I will remove '_' from excludes, but maybe keep 'base.html'.
        
        exclude_patterns = [
            'base.html',
            'base_index.html',
            'error.html',
            'layout/',
             # '_',  Removed to allow fixing partials
        ]
        
        # Scan all HTML templates
        for template_file in templates_dir.rglob('*.html'):
            relative_path = template_file.relative_to(templates_dir)
            relative_str = str(relative_path)
            
            # Skip excluded files
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in relative_str:
                    should_exclude = True
                    break
            
            if should_exclude:
                continue
                
            files_checked += 1
            
            try:
                content = template_file.read_text()
                original_content = content
                file_issues = []
                
                # Check legacy patterns
                for pattern_name, pattern_info in legacy_patterns.items():
                    matches = re.finditer(pattern_info['pattern'], content, re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        file_issues.append({
                            'file': str(relative_path),
                            'line': line_num,
                            'severity': pattern_info['severity'],
                            'message': pattern_info['message'],
                            'pattern': pattern_name,
                            'match': match.group(0),
                        })
                
                # Check for missing modern elements
                for check_name, check_info in missing_patterns.items():
                    if check_info['check'](content):
                        file_issues.append({
                            'file': str(relative_path),
                            'line': 0,
                            'severity': check_info['severity'],
                            'message': check_info['message'],
                            'pattern': check_name,
                            'match': '',
                        })
                
                # Check for missing lightbox class
                # Find <a> tags pointing to 'image' URL
                lightbox_pattern = r'(<a\s+[^>]*href="{% url "image" [^"]+"[^>]*>)'
                lightbox_matches = re.finditer(lightbox_pattern, content)
                for match in lightbox_matches:
                    tag = match.group(0)
                    if 'glightbox' not in tag:
                        line_num = content[:match.start()].count('\n') + 1
                        file_issues.append({
                            'file': str(relative_path),
                            'line': line_num,
                            'severity': 'WARNING',
                            'message': 'Image link missing lightbox class',
                            'pattern': 'missing_lightbox',
                            'match': tag,
                        })

                if file_issues:
                    issues_found.extend(file_issues)
                
                # Apply fixes if requested
                if options['fix']:
                    modified = False
                    
                    # Fix lightbox
                    def lightbox_replacer(match):
                        tag = match.group(0)
                        if 'glightbox' in tag:
                            return tag
                        if 'class="' in tag:
                            return re.sub(r'class="([^"]*)"', r'class="\1 glightbox"', tag)
                        elif "class='" in tag:
                            return re.sub(r"class='([^']*)'", r"class='\1 glightbox'", tag)
                        else:
                            return tag.replace('<a ', '<a class="glightbox" ')
                            
                    new_content = re.sub(lightbox_pattern, lightbox_replacer, content)
                    if new_content != content:
                        content = new_content
                        modified = True
                        fixed_count += 1
                    
                    if modified:
                        template_file.write_text(content)
                        self.stdout.write(self.style.SUCCESS(f'Fixed {relative_path}'))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error reading {relative_path}: {e}'))
        
        # Report results
        self.stdout.write('\n' + '='*80)
        self.stdout.write(self.style.SUCCESS(f'Template Linter Results'))
        self.stdout.write('='*80 + '\n')
        
        if options['fix']:
             self.stdout.write(self.style.SUCCESS(f'Fixed {fixed_count} issues.'))
        
        if not issues_found:
            self.stdout.write(self.style.SUCCESS(f'✓ No issues found in {files_checked} templates!'))
            return
        
        # Group by severity
        errors = [i for i in issues_found if i['severity'] == 'ERROR']
        warnings = [i for i in issues_found if i['severity'] == 'WARNING']
        info = [i for i in issues_found if i['severity'] == 'INFO']
        
        # Display errors
        if errors:
            self.stdout.write(self.style.ERROR(f'\n🔴 ERRORS ({len(errors)}):'))
            for issue in sorted(errors, key=lambda x: (x['file'], x['line'])):
                self.stdout.write(f"  {issue['file']}:{issue['line']} - {issue['message']}")
                if options['verbose'] and issue['match']:
                    self.stdout.write(f"    Match: {issue['match'][:80]}")
        
        # Display warnings
        if warnings:
            self.stdout.write(self.style.WARNING(f'\n⚠️  WARNINGS ({len(warnings)}):'))
            for issue in sorted(warnings, key=lambda x: (x['file'], x['line'])):
                self.stdout.write(f"  {issue['file']}:{issue['line']} - {issue['message']}")
                if options['verbose'] and issue['match']:
                    self.stdout.write(f"    Match: {issue['match'][:80]}")
        
        # Display info
        if info and options['verbose']:
            self.stdout.write(self.style.HTTP_INFO(f'\nℹ️  INFO ({len(info)}):'))
            for issue in sorted(info, key=lambda x: (x['file'], x['line'])):
                self.stdout.write(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        
        # Summary
        self.stdout.write('\n' + '='*80)
        self.stdout.write(f'Checked {files_checked} templates')
        self.stdout.write(f'Found {len(errors)} errors, {len(warnings)} warnings, {len(info)} info items')
        
        # Files needing attention
        files_with_issues = set(i['file'] for i in errors + warnings)
        if files_with_issues:
            self.stdout.write(self.style.WARNING(f'\n📝 Files needing attention ({len(files_with_issues)}):'))
            for file in sorted(files_with_issues):
                file_errors = len([i for i in errors if i['file'] == file])
                file_warnings = len([i for i in warnings if i['file'] == file])
                self.stdout.write(f"  {file} ({file_errors} errors, {file_warnings} warnings)")
        
        self.stdout.write('='*80 + '\n')
