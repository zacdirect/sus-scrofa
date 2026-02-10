import os
import re

def scan_templates(template_dir):
    legacy_patterns = {
        'Bootstrap 2 Grid (span*)': r'span\d+',
        'Bootstrap 2/3 Grid (row-fluid)': r'row-fluid',
        'Legacy Icons (icon-*)': r'icon-[a-zA-Z0-9-]+',
        'Legacy Buttons (btn-inverse, etc.)': r'btn-(inverse|large|mini|small)',
        'Legacy Layout (hero-unit, well, etc.)': r'(hero-unit|well|page-header|nav-header|content-large|box-header|corner-top|corner-all)',
        'Legacy Float (pull-right/left)': r'pull-(right|left)',
        'Legacy Forms (form-horizontal, etc.)': r'(form-horizontal|control-group|control-label)',
        'Legacy Attributes (align=, width=)': r'\s(align|width)=[\"\']',
    }

    results = {}

    for root, _, files in os.walk(template_dir):
        for file in files:
            if not file.endswith('.html'):
                continue
            
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, template_dir)
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            file_issues = []
            for issue_name, pattern in legacy_patterns.items():
                matches = re.findall(pattern, content)
                if matches:
                    file_issues.append(f"{issue_name}: {len(matches)} matches")
            
            if file_issues:
                results[relative_path] = file_issues

    return results

if __name__ == "__main__":
    template_dir = "/home/zac/repos/temp/sus-scrofa/templates"
    print(f"Scanning {template_dir} for legacy styles...\n")
    
    issues = scan_templates(template_dir)
    
    if issues:
        print(f"Found legacy styles in {len(issues)} files:\n")
        for file, problems in sorted(issues.items()):
            print(f"File: {file}")
            for problem in problems:
                print(f"  - {problem}")
            print("-" * 40)
    else:
        print("No legacy styles found! Great job!")
