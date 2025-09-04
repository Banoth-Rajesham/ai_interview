from fpdf import FPDF
from datetime import datetime

# This file handles the creation of the PDF report.

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI Interview Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(name, role, summary, questions, answers):
    pdf = PDF()
    pdf.add_page()
    def write_text(text):
        pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.set_font('Arial', 'B', 16); write_text(f"Candidate: {name}")
    pdf.set_font('Arial', '', 12); write_text(f"Role: {role}\nOverall Score: {summary.get('overall_score', 'N/A')}/10\nRecommendation: {summary.get('recommendation', 'N/A')}\nDate: {datetime.now().strftime('%Y-%m-%d')}")
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14); write_text("Detailed Question & Answer Analysis")
    for i, (q, a) in enumerate(zip(questions, answers)):
        pdf.set_font('Arial', 'B', 12); write_text(f"Q{i+1}: {q['text']}")
        pdf.set_font('Arial', '', 12); write_text(f"Answer: {a['answer']}")
        pdf.set_font('Arial', 'I', 12); write_text(f"Feedback: {a['feedback']} (Score: {a['score']}/10)"); pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')
