import os
from typing import List, Dict

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def load_pdfs(directory: str) -> List[Dict[str, str]]:
    """Load all PDFs under `directory` and return list of {path, text}.

    This function uses PyPDF2 if available; if not, files are skipped.
    """
    results = []
    if not os.path.isdir(directory):
        return results
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                text = ""
                if PdfReader is None:
                    # PyPDF2 not installed — skip extracting but include path
                    results.append({"path": path, "text": ""})
                    continue
                try:
                    reader = PdfReader(path)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                except Exception:
                    # Ignore files that cannot be read
                    continue
                results.append({"path": path, "text": text})
    return results
