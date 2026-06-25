from pypdf import PdfReader

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            documents.append({
                "text": text,
                "page": page_number,
                "file_name": uploaded_file.name
            })

    return documents