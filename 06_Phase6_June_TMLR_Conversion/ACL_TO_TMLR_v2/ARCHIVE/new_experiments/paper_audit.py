"""Final comprehensive paper audit."""
import re, sys, os
sys.stdout.reconfigure(encoding="utf-8")

tex_path = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\TMLR_Version\main.tex"
bib_path = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\TMLR_Version\custom.bib"
pdf_path = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\TMLR_Version\main.pdf"

with open(tex_path, "r", encoding="utf-8") as f:
    content = f.read()
with open(bib_path, "r", encoding="utf-8") as f:
    bib = f.read()

# Get active (non-commented) lines
lines = content.split("\n")
active_lines = [l for l in lines if not l.strip().startswith("%")]
active = "\n".join(active_lines)

print("FINAL PAPER REVIEW AUDIT")
print("=" * 60)

# 1. UNDEFINED REFERENCES
print("\n1. REFERENCES CHECK")
# Find \ref{...} and \eqref{...}
refs_used = set(re.findall(r"\\ref\{([^}]+)\}", active))
refs_used.update(re.findall(r"\\eqref\{([^}]+)\}", active))
labels_defined = set(re.findall(r"\\label\{([^}]+)\}", active))
undefined = refs_used - labels_defined
print(f"   Labels defined: {len(labels_defined)}")
print(f"   Refs used: {len(refs_used)}")
if undefined:
    print(f"   *** UNDEFINED REFS: {undefined}")
else:
    print(f"   All resolved. OK")

# 2. CITATIONS
print("\n2. CITATIONS CHECK")
cite_keys = set()
for m in re.finditer(r"\\cite[tp]?\{([^}]+)\}", active):
    for k in m.group(1).split(","):
        cite_keys.add(k.strip())
bib_keys = set(re.findall(r"@\w+\{(\S+?),", bib))
missing = cite_keys - bib_keys
print(f"   Cited: {len(cite_keys)}, Bib entries: {len(bib_keys)}")
if missing:
    print(f"   *** MISSING FROM BIB: {missing}")
else:
    print(f"   All present. OK")

# 3. TABLES
print("\n3. TABLES")
tables = re.findall(r"\\label\{(tab:[^}]+)\}", active)
for t in tables:
    # Check if referenced
    is_ref = t in active.replace("\\label{" + t + "}", "")
    print(f"   {t} {'(referenced)' if is_ref else '*** NOT REFERENCED'}")

# 4. FIGURES
print("\n4. FIGURES")
figures = re.findall(r"\\label\{(fig:[^}]+)\}", active)
for f_label in figures:
    is_ref = f_label in active.replace("\\label{" + f_label + "}", "")
    print(f"   {f_label} {'(referenced)' if is_ref else '*** NOT REFERENCED'}")

# 5. STALE / TODO markers
print("\n5. STALE MARKERS")
for keyword in ["TBA", "TODO", "FIXME", "XXX", "five benchmarks"]:
    count = active.count(keyword)
    if count > 0:
        print(f"   *** '{keyword}' found {count} times!")
        for i, line in enumerate(active_lines):
            if keyword in line:
                print(f"       Line ~{i}: {line.strip()[:70]}")
    else:
        print(f"   '{keyword}': 0 (clean)")

# 6. RED TEXT (new additions)
print("\n6. RED TEXT MARKERS (new content)")
red1 = active.count("\\textcolor{red}")
red2 = active.count("{\\color{red}")
print(f"   \\textcolor{{red}}: {red1}")
print(f"   {{\\color{{red}}: {red2}")
print(f"   Total red markers: {red1 + red2}")

# 7. KEY NUMBERS
print("\n7. KEY NUMBERS CONSISTENCY")
numbers_to_check = {
    "72.50 (HE 8B topo)": "72.50",
    "79.75 (HE 14B topo)": "79.75",
    "75.20 (CC 0.5B topo)": "75.20",
    "67.8 (gap HE)": "67.8",
    "89.7 (gap CC)": "89.7",
    "29.2 (overall gap)": "29.2",
    "18.4 (CC 0.5B A* adv)": "18.4",
    "eight benchmarks": "eight benchmarks",
}
for desc, val in numbers_to_check.items():
    count = active.count(val)
    print(f"   {desc}: {count} occurrences")

# 8. PDF CHECK
print("\n8. PDF STATUS")
if os.path.exists(pdf_path):
    import PyPDF2
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        print(f"   Pages: {len(reader.pages)}")
        print(f"   Size: {os.path.getsize(pdf_path)//1024}KB")
else:
    print("   PDF NOT FOUND")

# 9. Check for -- (empty cells) in router table
print("\n9. EMPTY CELLS (--) in active text")
dash_lines = [(i, l.strip()[:80]) for i, l in enumerate(active_lines) if "& --" in l or "& \\textcolor{red}{--}" in l]
print(f"   Lines with '--': {len(dash_lines)}")
for linenum, text in dash_lines[:10]:
    print(f"     Line {linenum}: {text}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
