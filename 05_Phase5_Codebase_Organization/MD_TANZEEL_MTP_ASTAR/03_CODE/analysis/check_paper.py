#!/usr/bin/env python3
"""Full paper completeness audit."""
import re, os

BASE = os.path.dirname(os.path.abspath(__file__))
TEX  = os.path.join(BASE, 'latex', 'main.tex')
BIB1 = os.path.join(BASE, 'latex', 'custom.bib')
BIB2 = os.path.join(BASE, 'latex', 'custom1.bib')

with open(TEX, encoding='utf-8') as f:
    content = f.read()
lines = content.split('\n')

sep = '=' * 60

# ── 1. Placeholders ──────────────────────────────────────────────────────────
print(sep)
print('1. PLACEHOLDERS')
print(sep)
KNOWN_TMLR_PLACEHOLDERS = {
    r'\def\month{MM}',
    r'\def\year{YYYY}',
    r'\def\openreview',
}
issues = []
for i, line in enumerate(lines, 1):
    ls = line.strip()
    if ls.startswith('%'):
        continue
    suspects = ['TODO', 'TBA', 'FIXME', 'XXX', 'PLACEHOLDER', '???']
    for s in suspects:
        if s in line:
            issues.append(f'  L{i:4d} [{s}]: {ls[:90]}')
    # TMLR-required placeholders (OK to have)
    if any(ls.startswith(p) for p in KNOWN_TMLR_PLACEHOLDERS):
        print(f'  L{i:4d} TMLR-required (fill before submission): {ls[:80]}')

if issues:
    print('  PROBLEMS FOUND:')
    for x in issues: print(x)
else:
    print('  No TODO/TBA/FIXME found.')

# ── 2. Citation keys ─────────────────────────────────────────────────────────
print()
print(sep)
print('2. CITATION KEYS')
print(sep)

# Load all bib keys
all_keys = set()
for bf in [BIB1, BIB2]:
    with open(bf, encoding='utf-8', errors='ignore') as f:
        bib = f.read()
    for m in re.finditer(r'@\w+\{([^,\s]+)', bib):
        all_keys.add(m.group(1).strip())

# Find all cited keys
cited_raw = re.findall(r'\\cite[pt]?\{([^}]+)\}', content)
cited_keys = set()
for block in cited_raw:
    for k in block.split(','):
        cited_keys.add(k.strip())

missing = cited_keys - all_keys
if missing:
    print(f'  MISSING ({len(missing)}):')
    for k in sorted(missing):
        for i, line in enumerate(lines, 1):
            if k in line and 'cite' in line.lower():
                print(f'    [{k}]  L{i}: {line.strip()[:80]}')
                break
else:
    print(f'  All {len(cited_keys)} citation keys found in bib files. OK.')

# ── 3. Cross-references ───────────────────────────────────────────────────────
print()
print(sep)
print('3. CROSS-REFERENCES (\\ref, \\autoref, \\cref)')
print(sep)

labels  = set(re.findall(r'\\label\{([^}]+)\}', content))
refs    = set(re.findall(r'\\(?:auto)?ref\{([^}]+)\}', content))
missing_refs = refs - labels
if missing_refs:
    print(f'  MISSING labels ({len(missing_refs)}):')
    for r in sorted(missing_refs):
        for i, line in enumerate(lines, 1):
            if r in line and 'ref{' in line:
                print(f'    [{r}]  L{i}: {line.strip()[:80]}')
                break
else:
    print(f'  All {len(refs)} \\ref targets have matching \\label. OK.')

# ── 4. Anonymous repo ────────────────────────────────────────────────────────
print()
print(sep)
print('4. ANONYMOUS REPO / URLs')
print(sep)
for i, line in enumerate(lines, 1):
    ls = line.strip()
    if ls.startswith('%'): continue
    if 'anonymous.4open.science' in line:
        print(f'  L{i:4d} Anonymous repo (replace with real URL before submission):')
        print(f'         {ls[:90]}')
    if 'openreview.net/forum?id=XXXX' in line:
        print(f'  L{i:4d} OpenReview placeholder (fill after submission):')
        print(f'         {ls[:90]}')

# ── 5. Table/figure labels ────────────────────────────────────────────────────
print()
print(sep)
print('5. TABLES AND FIGURES')
print(sep)
tables  = re.findall(r'\\caption\{([^}]{1,60})', content)
tlabels = re.findall(r'\\label\{tab:([^}]+)\}', content)
flabels = re.findall(r'\\label\{fig:([^}]+)\}', content)
print(f'  Tables defined:  {len(tlabels)} labels: {tlabels}')
print(f'  Figures defined: {len(flabels)} labels: {flabels}')
print(f'  Captions total:  {len(tables)}')

# ── 6. Sections check ─────────────────────────────────────────────────────────
print()
print(sep)
print('6. SECTION STRUCTURE')
print(sep)
for i, line in enumerate(lines, 1):
    ls = line.strip()
    if ls.startswith(r'\section') or ls.startswith(r'\subsection'):
        indent = '    ' if 'subsection' in ls else ''
        print(f'  L{i:4d} {indent}{ls[:80]}')

# ── 7. Math / equation integrity ──────────────────────────────────────────────
print()
print(sep)
print('7. EQUATION / MATH CHECK')
print(sep)
eq_open  = content.count(r'\begin{equation}')
eq_close = content.count(r'\end{equation}')
print(f'  equation environments: {eq_open} open, {eq_close} close — {"OK" if eq_open==eq_close else "MISMATCH"}')
# Check for unmatched $ signs (rough check on a line-by-line basis)
inline_issues = []
for i, line in enumerate(lines, 1):
    if line.strip().startswith('%'): continue
    # Remove verbatim content
    stripped = re.sub(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', '', line, flags=re.DOTALL)
    dollar_count = stripped.count('$') - stripped.count(r'\$')
    if dollar_count % 2 != 0:
        inline_issues.append(f'  L{i:4d} Odd $ count: {line.strip()[:80]}')
if inline_issues:
    print('  Potential unmatched $ signs:')
    for x in inline_issues[:10]: print(x)
else:
    print('  No unmatched inline math detected.')

# ── 8. Verb env balance ──────────────────────────────────────────────────────
print()
print(sep)
print('8. VERBATIM ENVIRONMENTS')
print(sep)
verb_open  = content.count(r'\begin{verbatim}')
verb_close = content.count(r'\end{verbatim}')
print(f'  verbatim: {verb_open} open, {verb_close} close — {"OK" if verb_open==verb_close else "MISMATCH"}')

# ── 9. Line count and size ────────────────────────────────────────────────────
print()
print(sep)
print('9. PAPER STATS')
print(sep)
print(f'  Lines in main.tex: {len(lines)}')
pdf_size = os.path.getsize(os.path.join(BASE, 'latex', 'main.pdf'))
print(f'  PDF size: {pdf_size/1024:.1f} KB')
print(f'  Bib keys available: {len(all_keys)}')
print(f'  Bib keys cited:     {len(cited_keys)}')

print()
print(sep)
print('AUDIT COMPLETE')
print(sep)
