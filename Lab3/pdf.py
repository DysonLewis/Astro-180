from fpdf import FPDF

inp = 'phot_calcs.py'  # input file
out = 'phot_calcs.pdf'  # output pdf file

# Create PDF instance
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Read the Python file and add its content to the PDF
with open(inp, 'r') as f:
    txt = f.read()
pdf.multi_cell(0, 10, txt)

# Save the PDF
pdf.output(out)