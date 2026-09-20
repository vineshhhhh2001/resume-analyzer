"""File processing module for extracting text from PDF and DOCX files."""

import io
import re
from typing import Optional

import pdfplumber
from PyPDF2 import PdfReader
from docx import Document


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file using pdfplumber with PyPDF2 fallback.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        Extracted and cleaned text.
    """
    text = ""

    # Primary: pdfplumber (better layout handling)
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        text = ""

    # Fallback: PyPDF2
    if not text.strip():
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")

    if not text.strip():
        raise ValueError("Could not extract any text from the PDF file.")

    return clean_text(text)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX file.

    Args:
        file_bytes: Raw bytes of the DOCX file.

    Returns:
        Extracted and cleaned text.
    """
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = []

        # Extract from paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text.strip())

        # Extract from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    paragraphs.append(row_text)

        text = "\n".join(paragraphs)

        if not text.strip():
            raise ValueError("Could not extract any text from the DOCX file.")

        return clean_text(text)

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {e}")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from a file based on its extension.

    Args:
        file_bytes: Raw bytes of the file.
        filename: Original filename (used to determine type).

    Returns:
        Extracted and cleaned text.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(file_bytes)
    elif ext == "txt":
        return clean_text(file_bytes.decode("utf-8", errors="ignore"))
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Please upload a PDF, DOCX, or TXT file.")


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    # Normalize unicode
    text = text.encode("utf-8", errors="ignore").decode("utf-8")

    # Replace multiple whitespace with single space (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text
