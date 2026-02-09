#!/usr/bin/env python3
"""Add URL names to URL patterns and fix templates to use them."""
import re

# Mapping of old-style view names to new URL names
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
    # auth (already fixed)
    'django.contrib.auth.views.logout': 'logout',
    'django.contrib.auth.views.login': 'login',
}

def fix_templates():
    """Fix all template files."""
    import os
    import glob
    
    template_files = []
    for root, dirs, files in os.walk('templates'):
        for file in files:
            # Skip Mac resource fork files
            if file.startswith('._') or file.startswith('.'):
                continue
            if file.endswith('.html'):
                template_files.append(os.path.join(root, file))
    
    fixed_count = 0
    for filepath in template_files:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            original = content
            
            # Replace each old-style URL with new name
            for old_name, new_name in URL_MAPPING.items():
                # Match {% url "old.name" %} and {% url 'old.name' %}
                content = re.sub(
                    rf'{{% url ["\']' + re.escape(old_name) + r'["\']',
                    f'{{% url "{new_name}"',
                    content
                )
            
            if content != original:
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"Fixed: {filepath}")
                fixed_count += 1
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")
    
    print(f"\nFixed {fixed_count} template files")

if __name__ == '__main__':
    fix_templates()
