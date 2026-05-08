import re
import os

gls_file = '/home/sergiosvieira/Documentos/uecetex2/elementos-pre-textuais/lista-de-abreviaturas-e-siglas.tex'
with open(gls_file, 'r', encoding='utf-8') as f:
    gls_content = f.read()

defs = {}
for m in re.finditer(r'\\newacronym\{([^}]+)\}\{([^}]+)\}\{(.+)\}', gls_content):
    acr_key = m.group(1)
    definition = m.group(3)
    # Extract the main Portuguese text (ignoring \textit{...} inside)
    # e.g. "Veículo-para-Rede (\textit{Vehicle-to-Network})" -> "Veículo-para-Rede"
    main_def = re.sub(r'\s*\(\\textit\{[^}]+\}\)', '', definition).strip()
    main_def = re.sub(r'\s*\([^)]+\)', '', main_def).strip()
    defs[acr_key] = main_def

# print(defs)

dir_path = '/home/sergiosvieira/Documentos/uecetex2'
for root, dirs, files in os.walk(dir_path):
    if '.git' in root or 'build' in root: continue
    for file in files:
        if file.endswith('.tex'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for m in re.finditer(r'([a-zA-ZÀ-ÿ0-9\-\s,]+)\(\\(gls|glspl)\{([^}]+)\}\)', content):
                text_before = m.group(1).strip()
                acr = m.group(3)
                if acr in defs:
                    expected_def = defs[acr]
                    # Check if the text before ends with the expected definition (or something very similar)
                    # We compare lowercase, ignoring punctuation
                    t_b_clean = re.sub(r'[^\w\s]', '', text_before.lower())
                    e_d_clean = re.sub(r'[^\w\s]', '', expected_def.lower())
                    
                    if e_d_clean in t_b_clean and len(e_d_clean) > 3:
                        print(f"File: {filepath}")
                        print(f"Match: '{m.group(0)}'")
                        print(f"Acr: {acr}, Expected: {expected_def}")
                        print("-" * 40)
