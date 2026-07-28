"""Compile all book chapters into a single PDF using fpdf2 with unicode support."""
import os
import re
from fpdf import FPDF

BOOK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BOOK_DIR, "The_Experience_That_Never_Counted.pdf")

CHAPTER_ORDER = [
    "00_preface.md",
    "01_ai_landscape.md",
    "02_ml_fundamentals.md",
    "03_deep_learning.md",
    "04_large_language_models.md",
    "05_information_retrieval.md",
    "06_rag_systems.md",
    "07_ai_agents.md",
    "08_graph_algorithms.md",
    "09_compliance_ai.md",
    "10_search_algorithms.md",
    "11_routing_method_selection.md",
    "12_experimental_design.md",
    "13_python_engineering.md",
    "14_data_pipelines.md",
    "15_frontend_deployment.md",
    "16a_questions_ml_dl.md",
    "16b_questions_llm.md",
    "16b_questions_rag.md",
    "16c_questions_agents_systems.md",
    "16d_questions_python_behavioral.md",
]

UNICODE_MAP = {
    '—': '--', '–': '-', '‘': "'", '’': "'",
    '“': '"', '”': '"', '…': '...', '→': '->',
    '←': '<-', '↔': '<->', '≥': '>=', '≤': '<=',
    '≠': '!=', '×': 'x', '≈': '~=', '∞': 'inf',
    '∑': 'sum', '∏': 'prod', '√': 'sqrt',
    'Δ': 'Delta', 'α': 'alpha', 'β': 'beta',
    'θ': 'theta', 'λ': 'lambda', 'σ': 'sigma',
    '•': '*', '▶': '>', '▸': '>', '✓': '[v]',
    '✗': '[x]', '★': '*', '☐': '[ ]', '☑': '[v]',
    '☒': '[x]', '\xd7': 'x', '≡': '===',
    '➤': '->', '●': '*', '○': 'o',
}


def sanitize(text):
    for char, replacement in UNICODE_MAP.items():
        text = text.replace(char, replacement)
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text


class BookPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 5, sanitize('The Experience That Never Counted -- Md Tanzeel Adam Khan'), align='C')
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'{self.page_no()}', align='C')

    def title_page(self):
        self.add_page()
        self.ln(60)
        self.set_font('Helvetica', 'B', 28)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 12, 'The Experience\nThat Never Counted', align='C')
        self.ln(10)
        self.set_font('Helvetica', 'I', 16)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, 'A Book for AI and ML Enthusiasts', align='C')
        self.ln(30)
        self.set_font('Helvetica', '', 14)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, 'Md Tanzeel Adam Khan', align='C')
        self.ln(8)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'M.Tech AI, Indian Institute of Technology Patna', align='C')
        self.ln(40)
        self.set_font('Helvetica', '', 10)
        self.cell(0, 8, 'July 2026', align='C')

    def add_chapter_from_md(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        self.add_page()
        in_code_block = False
        code_buffer = []

        for line in content.split('\n'):
          try:
            if line.startswith('```'):
                if in_code_block:
                    self._render_code_block('\n'.join(code_buffer))
                    code_buffer = []
                    in_code_block = False
                else:
                    in_code_block = True
                continue

            if in_code_block:
                code_buffer.append(line)
                continue

            line_clean = sanitize(self._strip_md(line))

            if line.startswith('# '):
                self.ln(5)
                self.set_font('Helvetica', 'B', 16)
                self.set_text_color(0, 0, 0)
                self.multi_cell(0, 8, line_clean[2:])
                self.ln(2)
                self.set_draw_color(60, 60, 60)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(4)
            elif line.startswith('## '):
                self.ln(3)
                self.set_font('Helvetica', 'B', 13)
                self.set_text_color(30, 30, 30)
                self.multi_cell(0, 7, line_clean[3:])
                self.ln(2)
            elif line.startswith('### '):
                self.ln(2)
                self.set_font('Helvetica', 'B', 11)
                self.set_text_color(50, 50, 50)
                self.multi_cell(0, 6, line_clean[4:])
                self.ln(1)
            elif line.startswith('| ') and '|' in line[1:]:
                self._render_table_row(line)
            elif line.startswith('---'):
                self.ln(2)
                self.set_draw_color(200, 200, 200)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(2)
            elif line.strip().startswith(('- ', '* ')):
                text = sanitize(self._strip_md(line.strip()[2:]))
                self.set_font('Helvetica', '', 9.5)
                self.set_text_color(0, 0, 0)
                self.set_x(14)
                self.multi_cell(0, 5, '* ' + text)
            elif re.match(r'^\s*\d+\.', line.strip()):
                text = re.sub(r'^\s*\d+\.\s*', '', line.strip())
                text = sanitize(self._strip_md(text))
                num = re.match(r'\s*(\d+\.)', line.strip()).group(1)
                self.set_font('Helvetica', '', 9.5)
                self.set_text_color(0, 0, 0)
                self.set_x(14)
                self.multi_cell(0, 5, num + ' ' + text)
            elif line.strip() == '':
                self.ln(2)
            else:
                self.set_font('Helvetica', '', 9.5)
                self.set_text_color(0, 0, 0)
                self.multi_cell(0, 5, line_clean)
          except Exception:
                pass

    def _strip_md(self, text):
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        text = re.sub(r'\$\$(.+?)\$\$', r'\1', text)
        text = re.sub(r'\$(.+?)\$', r'\1', text)
        return text

    def _render_code_block(self, code):
        self.ln(1)
        self.set_font('Courier', '', 7.5)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(30, 30, 30)
        lines = code.split('\n')[:35]
        for code_line in lines:
            code_line = sanitize(code_line)[:100]
            self.set_x(12)
            self.cell(186, 3.8, code_line, fill=True)
            self.ln(3.8)
        if len(code.split('\n')) > 35:
            self.set_x(12)
            self.cell(186, 3.8, '  ... (truncated)', fill=True)
            self.ln(3.8)
        self.ln(2)

    def _render_table_row(self, line):
        if re.match(r'\|[\s\-:]+\|', line):
            return
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if not cells:
            return
        self.set_font('Helvetica', '', 8)
        self.set_text_color(0, 0, 0)
        n_cols = len(cells)
        col_width = min(180 / max(n_cols, 1), 45)
        total_w = col_width * n_cols
        if total_w > 186:
            col_width = 186 / n_cols
        self.set_x(12)
        for cell in cells:
            cell = sanitize(self._strip_md(cell))
            cell = cell[:int(col_width / 2)]
            self.cell(col_width, 4.5, cell, border=1)
        self.ln(4.5)


if __name__ == "__main__":
    print("Compiling 'The Experience That Never Counted'...")
    pdf = BookPDF()
    pdf.title_page()

    for chapter_file in CHAPTER_ORDER:
        filepath = os.path.join(BOOK_DIR, chapter_file)
        if not os.path.exists(filepath):
            print(f"  SKIP: {chapter_file}")
            continue
        print(f"  Adding: {chapter_file}")
        try:
            pdf.add_chapter_from_md(filepath)
        except Exception as e:
            print(f"    ERROR in {chapter_file}: {e}")
            continue

    pdf.output(OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"\nDone! PDF: {OUTPUT}")
    print(f"Size: {size_mb:.1f} MB | Pages: {pdf.page_no()}")
