import os

def resolve_file(path, prefer_head=False):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    new_lines = []
    in_head = False
    in_origin = False
    
    for line in lines:
        if line.startswith('<<<<<<<'):
            in_head = True
            in_origin = False
            continue
        if line.startswith('======='):
            in_head = False
            in_origin = True
            continue
        if line.startswith('>>>>>>>'):
            in_head = False
            in_origin = False
            continue
            
        if prefer_head:
            if in_head:
                new_lines.append(line)
            elif not in_origin:
                new_lines.append(line)
        else:
            if in_origin:
                new_lines.append(line)
            elif not in_head:
                new_lines.append(line)
            
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f'Resolved {path} (prefer_head={prefer_head})')

# Files to resolve
resolve_file("SPEC.md", prefer_head=True)
resolve_file(".github/workflows/release.yml", prefer_head=True)
resolve_file("tests/unit/dbc/test_dbc_runtime_signal_toolkit.py", prefer_head=False)
