import re
from fpdf import FPDF

# PDF Layout Constants
LEFT_MARGIN = 15
TOP_MARGIN  = 15
RIGHT_MARGIN = 15
SECTION_TITLE_SIZE = 13
BODY_TEXT_SIZE = 11


def _sanitize(text: str) -> str:
    """
    Remove or replace characters that can't be encoded in latin-1.
    FPDF (the default font) requires latin-1 encoding. This function 
    ensures that emojis or higher-order unicode characters don't crash 
    the PDF generation.
    """
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _strip_markdown(text: str) -> str:
    """
    Remove basic Markdown syntax from a string for plain-text rendering.
    Currently handles:
    - **Bold** (double asterisks)
    - *Italic* (single asterisks)
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # Handle bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)        # Handle italic
    return text


def create_pdf_report(report_text: str, state: dict = None) -> bytes:
    """
    Converts a Markdown-like report text into a PDF and returns its bytes.
    Uses fpdf2.

    Args:
        report_text: Markdown-formatted report string from the reporter agent.
        state:       Optional pipeline state dict (passed for compatibility;
                     all data is already embedded in report_text).
    """
    pdf = FPDF()
    pdf.set_margins(left=LEFT_MARGIN, top=TOP_MARGIN, right=RIGHT_MARGIN)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", style="B", size=18)
    pdf.cell(w=0, h=12, text="Assessment Quality Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    lines = report_text.split("\n")

    for line in lines:
        raw = line.strip()
        if not raw:
            pdf.ln(3)
            continue

        text = _strip_markdown(_sanitize(raw))

        try:
            # Reset X to left margin before every line
            pdf.set_x(pdf.l_margin)

            if raw.startswith("# "):
                # H1 — already rendered as title above, skip
                continue
            elif raw.startswith("## "):
                pdf.ln(4)
                pdf.set_font("Helvetica", style="B", size=SECTION_TITLE_SIZE)
                heading = _sanitize(_strip_markdown(raw[3:].strip()))
                pdf.cell(w=0, h=8, text=heading, new_x="LMARGIN", new_y="NEXT")
            elif raw.startswith("- "):
                pdf.set_font("Helvetica", size=BODY_TEXT_SIZE)
                content = _sanitize(_strip_markdown(raw[2:].strip()))
                pdf.multi_cell(w=0, h=6, text="  \u2022  " + content)
            elif re.match(r"^\d+[.)]", raw):
                pdf.set_font("Helvetica", size=BODY_TEXT_SIZE)
                pdf.multi_cell(w=0, h=6, text="  " + text)
            elif raw.startswith("*") and raw.endswith("*") and not raw.startswith("**"):
                pdf.set_font("Helvetica", style="I", size=10)
                italic_text = _sanitize(raw.strip("*").strip())
                pdf.multi_cell(w=0, h=6, text=italic_text)
            else:
                pdf.set_font("Helvetica", size=BODY_TEXT_SIZE)
                pdf.multi_cell(w=0, h=6, text=text)
        except Exception:
            # Skip any line that still can't render
            pass

    return bytes(pdf.output())
