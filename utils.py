import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from werkzeug.datastructures import FileStorage
from docx import Document

def read_text_file(file: FileStorage):
    return file.read().decode('utf-8')

def extract_text_from_pdf(file: FileStorage):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file.stream)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    except Exception:
        pass

    # Fallback for image-based PDFs
    if not text.strip():
        try:
            images = convert_from_path(file.filename)
            for image in images:
                text += pytesseract.image_to_string(image)
        except Exception as e:
            raise ValueError(f"Could not extract from image-based PDF: {e}")
    
    return text

def extract_text_from_docx(file: FileStorage):
    text = ""
    try:
        document = Document(file)
        text = "\n".join([para.text for para in document.paragraphs])
    except Exception as e:
        raise ValueError(f"Could not read DOCX: {e}")
    return text

def extract_text(file: FileStorage):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith('.txt'):
        return read_text_file(file)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
