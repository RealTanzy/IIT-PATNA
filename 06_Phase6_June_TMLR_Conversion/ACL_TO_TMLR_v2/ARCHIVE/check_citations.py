"""Check all citations in main.tex against custom.bib"""
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

tex_path = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\TMLR_Version\main.tex"
bib_path = r"C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL_TO_TMLR_v2\TMLR_Version\custom.bib"

with open(tex_path, "r", encoding="utf-8") as f:
    tex = f.read()
with open(bib_path, "r", encoding="utf-8") as f:
    bib = f.read()

# Get citation keys from NON-COMMENTED lines only
lines = tex.split("\n")
active_lines = [l for l in lines if not l.strip().startswith("%")]
active_tex = "\n".join(active_lines)

# Find all \cite{...}, \citep{...}, \citet{...}
cite_keys = set()
for m in re.finditer(r"\\cite[tp]?\{([^}]+)\}", active_tex):
    keys = m.group(1).split(",")
    for k in keys:
        cite_keys.add(k.strip())

# Find all bib entry keys
bib_keys = set()
for m in re.finditer(r"@\w+\{(\S+?),", bib):
    bib_keys.add(m.group(1))

print(f"Citation keys used in paper (active lines): {len(cite_keys)}")
print("=" * 50)
for k in sorted(cite_keys):
    status = "OK" if k in bib_keys else "*** MISSING ***"
    print(f"  {k:45s} [{status}]")

print(f"\nBib entries available: {len(bib_keys)}")

missing = cite_keys - bib_keys
if missing:
    print(f"\n{'!'*50}")
    print(f"MISSING FROM BIB ({len(missing)} entries):")
    print(f"{'!'*50}")
    for k in sorted(missing):
        print(f"  {k}")
else:
    print("\nAll citations have matching bib entries.")

# Check new code citations
print("\n" + "=" * 50)
print("CODE BENCHMARK CITATIONS:")
print("=" * 50)
for k in ["chen2021humaneval", "austin2021mbpp", "li2022codecontests"]:
    in_tex = k in active_tex
    in_bib = k in bib_keys
    print(f"  {k}: cited_in_paper={in_tex}, in_bib={in_bib}")

# Also check GSM8K and MATH citations
print("\nMATH BENCHMARK CITATIONS:")
for k in ["cobbe2021trainingverifierssolvemath", "hendrycks2021measuringmathematicalproblemsolving"]:
    in_tex = k in active_tex
    in_bib = k in bib_keys
    print(f"  {k}: cited_in_paper={in_tex}, in_bib={in_bib}")
