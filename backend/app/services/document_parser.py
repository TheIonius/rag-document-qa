import csv
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pypdf

class ParsedDocument:
    def __init__(self, filename: str, title: str, file_type: str, file_size: int, raw_text: str, pages: List[Tuple[int, str]]):
        self.filename = filename
        self.title = title
        self.file_type = file_type
        self.file_size = file_size
        self.raw_text = raw_text
        self.pages = pages  # List of (page_number, page_text)

def parse_document_content(filename: str, content_bytes: bytes) -> ParsedDocument:
    file_type = Path(filename).suffix.lower().lstrip(".")
    file_size = len(content_bytes)

    if file_type == "pdf":
        return parse_pdf(filename, content_bytes)
    elif file_type in ("md", "markdown"):
        return parse_markdown(filename, content_bytes.decode("utf-8", errors="replace"))
    elif file_type in ("csv", "tsv"):
        return parse_csv(filename, content_bytes.decode("utf-8", errors="replace"), delimiter="\t" if file_type == "tsv" else ",")
    else:  # txt or other plain text
        return parse_text(filename, content_bytes.decode("utf-8", errors="replace"))

def parse_csv(filename: str, text: str, delimiter: str = ",") -> ParsedDocument:
    cleaned = clean_text(text)
    title = infer_title(cleaned, filename)
    reader = csv.reader(io.StringIO(cleaned), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return ParsedDocument(filename, title, "csv" if delimiter == "," else "tsv", len(text.encode("utf-8")), "", [(1, "")])

    headers = [cell.strip().replace("|", "/") for cell in rows[0]]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    table_lines = [f"# {title}\n", header_line, separator_line]

    for row in rows[1:]:
        padded = [cell.strip().replace("|", "/").replace("\n", " ") for cell in row]
        if len(padded) < len(headers):
            padded.extend([""] * (len(headers) - len(padded)))
        else:
            padded = padded[:len(headers)]
        table_lines.append("| " + " | ".join(padded) + " |")

    markdown_table = "\n".join(table_lines)
    return ParsedDocument(
        filename=filename,
        title=title,
        file_type="csv" if delimiter == "," else "tsv",
        file_size=len(text.encode("utf-8")),
        raw_text=markdown_table,
        pages=[(1, markdown_table)]
    )

def parse_pdf(filename: str, content_bytes: bytes) -> ParsedDocument:
    stream = io.BytesIO(content_bytes)
    reader = pypdf.PdfReader(stream)
    pages = []
    full_text_parts = []

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = clean_text(text)
        page_num = idx + 1
        pages.append((page_num, text))
        full_text_parts.append(text)

    raw_text = "\n\n".join(full_text_parts)
    title = infer_title(raw_text, filename)
    return ParsedDocument(filename, title, "pdf", len(content_bytes), raw_text, pages)

def parse_markdown(filename: str, text: str) -> ParsedDocument:
    cleaned = clean_text(text)
    title = infer_title(cleaned, filename)
    # Markdown defaults to page 1, but we retain section headers for chunking
    return ParsedDocument(filename, title, "md", len(text.encode("utf-8")), cleaned, [(1, cleaned)])

def parse_text(filename: str, text: str) -> ParsedDocument:
    cleaned = clean_text(text)
    title = infer_title(cleaned, filename)
    return ParsedDocument(filename, title, "txt", len(text.encode("utf-8")), cleaned, [(1, cleaned)])

def clean_text(text: str) -> str:
    # Normalize carriage returns and excessive whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def infer_title(text: str, filename: str) -> str:
    # Try finding markdown top-level header: # Title
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Try first non-empty line if short
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and len(lines[0]) <= 80:
        return lines[0]

    # Fallback to prettified filename
    base = Path(filename).stem
    return base.replace("_", " ").replace("-", " ").title()
