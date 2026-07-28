"""Cross-check V3 paper: citations, references, reviewer coverage."""
import re, os

path = r'C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL Submission V3 July 26\main.tex'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

bib_path = r'C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL Submission V3 July 26\custom.bib'
with open(bib_path, 'r', encoding='utf-8') as f:
    bib_content = f.read()

print("=" * 60)
print("ACL V3 CROSS-CHECK REPORT")
print("=" * 60)

# 1. Reviewer tags
print("\n1. REVIEWER CONCERN COVERAGE")
tags = re.findall(r'\[R[123]-[WC]\d+\]', content)
found_tags = sorted(set(tags))
print(f"   Tags in paper: {found_tags}")

# Expected tags from rebuttal
expected = ['[R1-W1]', '[R1-W2]', '[R2-W1]', '[R2-W2]', '[R2-W3]', '[R2-W4]',
            '[R3-W1]', '[R3-W2]', '[R3-W3]', '[R3-W4]', '[R3-W5]', '[R3-C1]', '[R3-C2]', '[R3-C3]', '[R3-C4]']
missing_tags = [t for t in expected if t not in found_tags]
if missing_tags:
    print(f"   MISSING: {missing_tags}")
else:
    print("   All expected tags present!")

# 2. Citations
print("\n2. CITATIONS")
cite_matches = re.findall(r'\\cite[pt]?\{([^}]+)\}', content)
all_cited = set()
for c in cite_matches:
    for key in c.split(','):
        all_cited.add(key.strip())
print(f"   Cited keys: {len(all_cited)}")

bib_keys = set(re.findall(r'@\w+\{([^,\s]+)', bib_content))
print(f"   Bib entries: {len(bib_keys)}")

missing_bib = all_cited - bib_keys
if missing_bib:
    print(f"   CITED BUT MISSING FROM BIB: {sorted(missing_bib)}")
else:
    print("   All citations resolved!")

unused_bib = bib_keys - all_cited
if unused_bib:
    print(f"   Unused bib entries: {len(unused_bib)}")

# 3. Cross-references
print("\n3. CROSS-REFERENCES (\\ref vs \\label)")
labels = set(re.findall(r'\\label\{([^}]+)\}', content))
refs = set(re.findall(r'\\ref\{([^}]+)\}', content))
missing_refs = refs - labels
if missing_refs:
    print(f"   UNDEFINED \\ref (no matching \\label): {sorted(missing_refs)}")
else:
    print("   All references have labels!")

# 4. Key content checks
print("\n4. CONTENT VERIFICATION")
checks = {
    "Branching coefficient equation": r'\\label\{eq:branching_coefficient\}' in content,
    "Cost comparison table": 'tab:cost_comparison' in content,
    "Feature ablation appendix": 'app:feature_ablation' in content,
    "Leave-one-out appendix": 'app:leave_one_out' in content,
    "Component ablation appendix": 'app:component_ablation' in content,
    "Router protocol paragraph": 'Router evaluation protocol' in content or '5-fold' in content,
    "Terminology clarification": 'progress scorer' in content,
    "CoT-A* router naming": 'CoT--A* router' in content,
    "Topology defined operationally": 'reasoning topology' in content.lower(),
    "Confidence drawback analysis": 'premature commitment' in content,
    "Human eval scope clarified": 'supplementary diagnostic' in content,
    "R2-W1 Table corrected": '54.73' in content,
}
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"   [{status}] {check}")

# 5. Potential issues
print("\n5. POTENTIAL ISSUES")
# Check for unresolved red text from old version
old_red_count = content.count(r'\textcolor{red}')
if old_red_count > 0:
    print(f"   WARNING: {old_red_count} instances of \\textcolor{{red}} (old markers still present)")
else:
    print("   No leftover red markers from old version.")

# Check for ?? references
if '??' in content:
    lines_with_qq = [i+1 for i, l in enumerate(content.split('\n')) if '??' in l]
    print(f"   WARNING: '??' found (unresolved refs) at lines: {lines_with_qq[:5]}")
else:
    print("   No unresolved '??' references.")

print("\n" + "=" * 60)
