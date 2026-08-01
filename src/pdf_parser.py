import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    """
    Extract all text from a PDF using PyMuPDF.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    str
        Extracted text.
    """

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text.strip()