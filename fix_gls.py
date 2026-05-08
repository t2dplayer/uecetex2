import os
import re

dir_path = '/home/sergiosvieira/Documentos/uecetex2'
pattern = re.compile(r'(([A-Z][a-zA-Zçãõíéáóúâêô]*\s*)+(?:de\s+|por\s+|da\s+|do\s+|dos\s+|das\s+|para\s+|a\s+|e\s+)*([A-Z][a-zA-Zçãõíéáóúâêô]*\s*)*)\(\\(?:gls|glspl)\{([^}]+)\}\)')
pattern2 = re.compile(r'((?:[A-Z][a-z0-9\-]+\s*)+)\(\\(?:gls|glspl)\{([^}]+)\}\)', re.IGNORECASE)

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # We will just print matches for now
    matches = re.finditer(r'([a-zA-ZÀ-ÿ0-9\-\s]+)\(\\(gls|glspl)\{([^}]+)\}\)', content)
    for m in matches:
        text_before = m.group(1)[-50:] # last 50 chars before (\gls{})
        # check if it looks like an expansion
        if any(c.isupper() for c in text_before):
            print(f"{filepath}: {text_before.strip()} (\\{m.group(2)}{{{m.group(3)}}})")

for root, dirs, files in os.walk(dir_path):
    if '.git' in root or 'build' in root: continue
    for file in files:
        if file.endswith('.tex'):
            process_file(os.path.join(root, file))
