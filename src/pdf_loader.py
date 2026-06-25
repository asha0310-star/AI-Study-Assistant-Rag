from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    """Extract text from one uploaded PDF file.

    Returns a list of dictionaries. Each dictionary contains the file name,
    page number, and text for one page.
    """
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            {
                "file_name": uploaded_file.name,
                "page_number": page_number,
                "text": text,
            }
        )

    return pages


def extract_text_from_pdfs(uploaded_files):
    """Extract text from multiple uploaded PDF files."""
    all_pages = []

    for uploaded_file in uploaded_files:
        pages = extract_text_from_pdf(uploaded_file)
        all_pages.extend(pages)

    combined_text = "\n\n".join(page["text"] for page in all_pages)

    return all_pages, combined_text
