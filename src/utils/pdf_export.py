from fpdf import FPDF


def create_pdf_report(report_text: str) -> bytes:
    """
    Converts a Markdown-like report text into a PDF and returns its bytes.
    Uses fpdf2.
    """
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", style="B", size=18)
    pdf.cell(0, 12, txt="Assessment Quality Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    lines = report_text.split("\n")

    for line in lines:
        if line.startswith("# "):
            # H1 — skip, already rendered as title
            pass
        elif line.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", style="B", size=13)
            pdf.multi_cell(0, 8, txt=line.replace("## ", "").strip())
        elif line.startswith("- "):
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, txt="  \u2022 " + line[2:].strip())
        elif line.startswith("*") and line.endswith("*"):
            pdf.set_font("Helvetica", style="I", size=10)
            pdf.multi_cell(0, 6, txt=line.strip("*").strip())
        elif line.strip() == "":
            pdf.ln(2)
        else:
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, txt=line.strip())

    return bytes(pdf.output())
