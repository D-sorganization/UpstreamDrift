import os

workflow_dir = '.github/workflows'
for filename in os.listdir(workflow_dir):
    if filename.endswith('.yml') or filename.endswith('.yaml'):
        path = os.path.join(workflow_dir, filename)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Replace non-ASCII with -
        clean_content = ''.join([c if ord(c) < 128 else '-' for c in content])
        
        if clean_content != content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(clean_content)
            print(f'Cleaned {filename}')
