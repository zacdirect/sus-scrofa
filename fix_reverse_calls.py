#!/usr/bin/env python3
"""Fix reverse() calls in Python view files - Django 2.0+ compatibility."""
import re

# Mapping of old-style view names to new URL names (same as template mapping)
URL_MAPPING = {
    # users app
    'users.views.profile': 'profile',
    'users.views.admin_list_users': 'admin_list_users',
    'users.views.admin_list_activity': 'admin_list_activity',
    'users.views.admin_new_user': 'admin_new_user',
    'users.views.admin_show_user': 'admin_show_user',
    'users.views.admin_show_activity': 'admin_show_activity',
    'users.views.admin_edit_user': 'admin_edit_user',
    'users.views.admin_disable_user': 'admin_disable_user',
    # analyses app
    'analyses.views.list_cases': 'list_cases',
    'analyses.views.new_case': 'new_case',
    'analyses.views.edit_case': 'edit_case',
    'analyses.views.close_case': 'close_case',
    'analyses.views.delete_case': 'delete_case',
    'analyses.views.show_case': 'show_case',
    'analyses.views.new_image': 'new_image',
    'analyses.views.new_folder': 'new_folder',
    'analyses.views.new_url': 'new_url',
    'analyses.views.show_analysis': 'show_analysis',
    'analyses.views.delete_analysis': 'delete_analysis',
    'analyses.views.list_images': 'list_images',
    'analyses.views.image': 'image',
    'analyses.views.favorite': 'favorite',
    'analyses.views.add_comment': 'add_comment',
    'analyses.views.delete_comment': 'delete_comment',
    'analyses.views.search': 'search',
    'analyses.views.add_tag': 'add_tag',
    'analyses.views.delete_tag': 'delete_tag',
    'analyses.views.count_new_analysis': 'count_new_analysis',
    'analyses.views.hex_dump': 'hex_dump',
    'analyses.views.static_report': 'static_report',
    'analyses.views.export_json': 'export_json',
    'analyses.views.show': 'show_analysis',
    # hashes app
    'hashes.views.list_hashes': 'list_hashes',
    'hashes.views.new_hashes': 'new_hashes',
    'hashes.views.show_hashes': 'show_hashes',
    'hashes.views.delete_hashes': 'delete_hashes',
    # manage app
    'manage.views.dependencies_list': 'dependencies_list',
    # auth
    'django.contrib.auth.views.logout': 'logout',
    'django.contrib.auth.views.login': 'login',
}

def fix_file(filepath):
    """Fix reverse() calls in a Python file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Replace each old-style reverse() call with new name
    for old_name, new_name in URL_MAPPING.items():
        # Match reverse("old.name", ...) and reverse('old.name', ...)
        content = re.sub(
            r'reverse\(["\']' + re.escape(old_name) + r'["\']',
            f'reverse("{new_name}"',
            content
        )
    
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
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")
    
    print(f"\nFixed {fixed_count} files")
