#!/usr/bin/env python3
"""Fix RequestContext usage in view files - Django 1.8+ compatibility."""
import re
import sys

def fix_file(filepath):
    """Remove context_instance=RequestContext(request) from render calls."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Pattern to match context_instance=RequestContext(request) with various whitespace
    # This handles cases like:
    # context_instance=RequestContext(request))
    # context_instance = RequestContext(request))
    pattern = r',\s*context_instance\s*=\s*RequestContext\s*\(\s*request\s*\)'
    
    content = re.sub(pattern, '', content)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
        return True
    return False

if __name__ == '__main__':
    files_to_fix = [
        'analyses/views.py',
        'hashes/views.py',
        'users/views.py',
        'manage/views.py',
    ]
    
    fixed_count = 0
    for filepath in files_to_fix:
        try:
            if fix_file(filepath):
                fixed_count += 1
        except FileNotFoundError:
            print(f"Not found: {filepath}")
    
    print(f"\nFixed {fixed_count} files")
