import re
import os

gls_file = '/home/sergiosvieira/Documentos/uecetex2/elementos-pre-textuais/lista-de-abreviaturas-e-siglas.tex'
with open(gls_file, 'r', encoding='utf-8') as f:
    gls_content = f.read()

defs = {}
for m in re.finditer(r'\\newacronym\{([^}]+)\}\{([^}]+)\}\{(.+)\}', gls_content):
    acr_key = m.group(1)
    definition = m.group(3)
    # Extract italic text which is usually English
    match = re.search(r'\\textit\{([^}]+)\}', definition)
    if match:
        defs[acr_key] = match.group(1)
    else:
        # Maybe the definition itself is English if no textit
        defs[acr_key] = definition

dir_path = '/home/sergiosvieira/Documentos/uecetex2'
for root, dirs, files in os.walk(dir_path):
    if '.git' in root or 'build' in root: continue
    for file in files:
        if file.endswith('.tex'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for m in re.finditer(r'([a-zA-ZÀ-ÿ0-9\-\s,{}]+)\(\\(gls|glspl)\{([^}]+)\}\)', content):
                text_before = m.group(1).strip()
                acr = m.group(3)
                if acr in defs:
                    expected_def = defs[acr]
                    t_b_clean = re.sub(r'[^\w\s]', '', text_before.lower())
                    e_d_clean = re.sub(r'[^\w\s]', '', expected_def.lower())
                    
                    if e_d_clean in t_b_clean and len(e_d_clean) > 3:
                        print(f"File: {filepath}")
                        print(f"Match: '{m.group(0)}'")
                        print(f"Acr: {acr}, Expected: {expected_def}")
                        print("-" * 40)
