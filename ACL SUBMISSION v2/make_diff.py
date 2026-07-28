"""
Mark new content in main_diff.tex using line-level \textcolor{red}{}.
Safe approach: only wrap text-bearing lines that are not inside special environments.
"""
import difflib, re, os

with open(r'C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL SUBMISSION v2\main.tex', 'r', encoding='utf-8') as f:
    new_lines = f.readlines()
with open(r'C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL SUBMISSION v2\old_acl\old_ACL.tex', 'r', encoding='utf-8') as f:
    old_lines = f.readlines()

# Find which line numbers in new are additions/changes
sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
added_line_nums = set()
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag in ('insert', 'replace'):
        for j in range(j1, j2):
            added_line_nums.add(j)

# Track which environments are "unsafe" for inline color wrapping
UNSAFE_ENVS = {'tabular', 'tabularx', 'table', 'figure', 'algorithm',
               'algorithmic', 'algorithm2e', 'align', 'align*', 'equation',
               'equation*', 'math', 'gather', 'gather*', 'multline',
               'tikzpicture', 'pgfplots', 'examplebox', 'tcolorbox',
               'proof', 'theorem', 'assumption',
               'itemize', 'enumerate', 'description'}

def is_colorable(line, in_unsafe):
    """Can we safely wrap this line in \textcolor{red}{}?"""
    s = line.strip()
    if not s or s.startswith('%'):
        return False
    if in_unsafe:
        return False
    if len(s) < 8:
        return False
    # Skip pure LaTeX command lines
    no_go_starts = [
        r'\documentclass', r'\usepackage', r'\PassOptions',
        r'\def\\', r'\let\\', r'\newtheorem', r'\bibliographystyle',
        r'\bibliography', r'\begin{', r'\end{', r'\maketitle',
        r'\author', r'\date{', r'\title{', r'\label{', r'\ref{',
        r'\cite', r'\section', r'\subsection', r'\paragraph',
        r'\caption', r'\midrule', r'\toprule', r'\bottomrule',
        r'\hline', r'\\', r'\hspace', r'\vspace', r'\newpage',
        r'\clearpage', r'\appendix', r'\tcbuselibrary', r'\newtcolorbox',
        r'\pgfplotsset', r'\setlength', r'\addtolength',
        r'\let', r'\renewcommand', r'\newcommand', r'\DeclareMath',
        r'\item', r'\bibitem', r'\footnote', r'\footnotetext',
    ]
    for ng in no_go_starts:
        if s.startswith(ng):
            return False
    # Table rows (contain & or \\)
    if '&' in s or s.endswith(r'\\'):
        return False
    # Math display lines
    if s.startswith(r'\[') or s.startswith(r'\]') or s.startswith(r'\frac') or s.startswith(r'\sum') or s.startswith(r'\prod'):
        return False
    # Lines inside math (contain only math symbols)
    if re.match(r'^[=\+\-\*\/\|\{\}\[\]\^\_\s\\\.0-9,]+$', s):
        return False
    # Lines that are closing braces or brackets only
    if re.match(r'^[}\]\s]+$', s):
        return False
    return True

# Build output
output_lines = []
in_unsafe = 0  # depth counter
env_stack = []

for i, line in enumerate(new_lines):
    s = line.strip()

    # Track \[ \] display math
    if s.strip() == r'\[':
        in_unsafe += 1
    elif s.strip() == r'\]':
        if in_unsafe > 0:
            in_unsafe -= 1

    # Track environment depth
    begins = re.findall(r'\\begin\{(\w+)', s)
    ends = re.findall(r'\\end\{(\w+)', s)
    for env in begins:
        env_stack.append(env)
        if env in UNSAFE_ENVS:
            in_unsafe += 1
    for env in ends:
        if env_stack and env_stack[-1] == env:
            env_stack.pop()
        if env in UNSAFE_ENVS and in_unsafe > 0:
            in_unsafe -= 1

    # Should we color this line?
    if i in added_line_nums and is_colorable(line, in_unsafe > 0):
        stripped = line.rstrip('\n')
        output_lines.append(r'\textcolor{red}{' + stripped + '}\n')
    else:
        output_lines.append(line)

out_path = r'C:\Users\madamkh\Desktop\Research Work\tanzeel_server_backup\tanzeel server backup\ACL SUBMISSION v2\main_diff.tex'
with open(out_path, 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

colored = sum(1 for i, line in enumerate(new_lines)
              if i in added_line_nums and is_colorable(line, False))
print(f'Lines colored red: {colored}')
print(f'Written: {out_path}')
