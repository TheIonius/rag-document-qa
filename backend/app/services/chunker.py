import re
from typing import Dict, List, Optional
from app.config import settings
from app.services.document_parser import ParsedDocument

class Chunk:
    def __init__(
        self,
        chunk_index: int,
        section_title: Optional[str],
        page_number: int,
        text: str,
        word_count: int,
        char_start: int,
        char_end: int
    ):
        self.chunk_index = chunk_index
        self.section_title = section_title
        self.page_number = page_number
        self.text = text
        self.word_count = word_count
        self.char_start = char_start
        self.char_end = char_end

def chunk_document(
    parsed: ParsedDocument,
    target_words: int = settings.target_chunk_words,
    overlap_words: int = settings.overlap_words
) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_counter = 0

    if parsed.file_type == "md":
        # Chunk markdown using heading hierarchy
        sections = split_markdown_sections(parsed.raw_text)
        current_offset = 0

        for sec_title, sec_text in sections:
            sec_words = sec_text.split()
            if len(sec_words) <= target_words + (overlap_words // 2):
                if sec_text.strip():
                    char_len = len(sec_text)
                    chunks.append(Chunk(
                        chunk_index=chunk_counter,
                        section_title=sec_title,
                        page_number=1,
                        text=sec_text.strip(),
                        word_count=len(sec_words),
                        char_start=current_offset,
                        char_end=current_offset + char_len
                    ))
                    chunk_counter += 1
                    current_offset += char_len
            else:
                # Sub-chunk long section with sliding window
                sub_chunks = sliding_window_chunk(
                    sec_text, sec_title, 1, target_words, overlap_words, current_offset, chunk_counter
                )
                chunks.extend(sub_chunks)
                chunk_counter += len(sub_chunks)
                current_offset += len(sec_text)

    else:
        # PDF or plain text: chunk page by page
        current_offset = 0
        for page_num, page_text in parsed.pages:
            if not page_text.strip():
                continue

            # Detect any section headers in page text
            sec_title = extract_header_from_block(page_text)
            page_words = page_text.split()

            if len(page_words) <= target_words + (overlap_words // 2):
                char_len = len(page_text)
                chunks.append(Chunk(
                    chunk_index=chunk_counter,
                    section_title=sec_title or f"Page {page_num}",
                    page_number=page_num,
                    text=page_text.strip(),
                    word_count=len(page_words),
                    char_start=current_offset,
                    char_end=current_offset + char_len
                ))
                chunk_counter += 1
                current_offset += char_len
            else:
                sub_chunks = sliding_window_chunk(
                    page_text, sec_title or f"Page {page_num}", page_num,
                    target_words, overlap_words, current_offset, chunk_counter
                )
                chunks.extend(sub_chunks)
                chunk_counter += len(sub_chunks)
                current_offset += len(page_text)

    return chunks

def split_markdown_sections(text: str) -> List[tuple]:
    """Split markdown by # and ## headers, preserving header title and text."""
    lines = text.split("\n")
    sections = []
    current_title = "Overview"
    current_lines = []

    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$")

    for line in lines:
        match = header_pattern.match(line)
        if match:
            # Save prior section if not empty
            if current_lines:
                sec_content = "\n".join(current_lines).strip()
                if sec_content:
                    sections.append((current_title, sec_content))
                current_lines = []
            current_title = match.group(2).strip()
        current_lines.append(line)

    if current_lines:
        sec_content = "\n".join(current_lines).strip()
        if sec_content:
            sections.append((current_title, sec_content))

    if not sections:
        sections.append(("Document Content", text))

    return sections

def is_markdown_table_block(text: str) -> bool:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) >= 2:
        has_divider = any(re.match(r"^\|?\s*[-:]+\s*\|", l) for l in lines)
        table_lines = sum(1 for l in lines if l.startswith("|") or l.endswith("|") or "|" in l)
        return has_divider and (table_lines / len(lines) >= 0.7)
    return False

def split_table_by_rows(table_text: str, max_words: int) -> List[str]:
    lines = [l for l in table_text.strip().splitlines() if l.strip()]
    if len(lines) <= 2:
        return [table_text]
    header = lines[:2]
    header_str = "\n".join(header)
    sub_tables = []
    current_rows = []
    current_count = len(header_str.split())

    for row in lines[2:]:
        row_words = len(row.split())
        if current_count + row_words > max_words and current_rows:
            sub_tables.append(header_str + "\n" + "\n".join(current_rows))
            current_rows = [row]
            current_count = len(header_str.split()) + row_words
        else:
            current_rows.append(row)
            current_count += row_words

    if current_rows:
        sub_tables.append(header_str + "\n" + "\n".join(current_rows))

    return sub_tables or [table_text]

def sliding_window_chunk(
    text: str,
    section_title: str,
    page_number: int,
    target_words: int,
    overlap_words: int,
    base_offset: int,
    start_index: int
) -> List[Chunk]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_words: List[str] = []
    current_para_start = 0
    chunk_idx = start_index

    for para in paragraphs:
        p_clean = para.strip()
        if not p_clean:
            continue

        p_words = p_clean.split()

        # Preserve markdown tables without cutting across rows
        if is_markdown_table_block(p_clean):
            if current_words:
                chunk_text = " ".join(current_words)
                chunks.append(Chunk(
                    chunk_index=chunk_idx,
                    section_title=section_title,
                    page_number=page_number,
                    text=chunk_text,
                    word_count=len(current_words),
                    char_start=base_offset + current_para_start,
                    char_end=base_offset + current_para_start + len(chunk_text)
                ))
                chunk_idx += 1
                current_words = []

            if len(p_words) <= 400:
                chunks.append(Chunk(
                    chunk_index=chunk_idx,
                    section_title=section_title,
                    page_number=page_number,
                    text=p_clean,
                    word_count=len(p_words),
                    char_start=base_offset + current_para_start,
                    char_end=base_offset + current_para_start + len(p_clean)
                ))
                chunk_idx += 1
            else:
                table_slices = split_table_by_rows(p_clean, max_words=target_words)
                for tbl_slice in table_slices:
                    s_words = tbl_slice.split()
                    chunks.append(Chunk(
                        chunk_index=chunk_idx,
                        section_title=section_title,
                        page_number=page_number,
                        text=tbl_slice,
                        word_count=len(s_words),
                        char_start=base_offset + current_para_start,
                        char_end=base_offset + current_para_start + len(tbl_slice)
                    ))
                    chunk_idx += 1
            continue

        if len(current_words) + len(p_words) <= target_words:
            current_words.extend(p_words)
        else:
            if current_words:
                chunk_text = " ".join(current_words)
                chunks.append(Chunk(
                    chunk_index=chunk_idx,
                    section_title=section_title,
                    page_number=page_number,
                    text=chunk_text,
                    word_count=len(current_words),
                    char_start=base_offset + current_para_start,
                    char_end=base_offset + current_para_start + len(chunk_text)
                ))
                chunk_idx += 1

                # Keep overlap words from the end
                if overlap_words > 0 and len(current_words) > overlap_words:
                    current_words = current_words[-overlap_words:]
                else:
                    current_words = []

            # If single paragraph is itself longer than target_words, chunk by words
            if len(p_words) > target_words:
                step = target_words - overlap_words
                for i in range(0, len(p_words), step):
                    slice_words = p_words[i:i + target_words]
                    if not slice_words:
                        break
                    sub_text = " ".join(slice_words)
                    chunks.append(Chunk(
                        chunk_index=chunk_idx,
                        section_title=section_title,
                        page_number=page_number,
                        text=sub_text,
                        word_count=len(slice_words),
                        char_start=base_offset,
                        char_end=base_offset + len(sub_text)
                    ))
                    chunk_idx += 1
                current_words = []
            else:
                current_words.extend(p_words)

    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append(Chunk(
            chunk_index=chunk_idx,
            section_title=section_title,
            page_number=page_number,
            text=chunk_text,
            word_count=len(current_words),
            char_start=base_offset,
            char_end=base_offset + len(chunk_text)
        ))

    return chunks

def extract_header_from_block(text: str) -> Optional[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        first = lines[0]
        if first.startswith("#"):
            return re.sub(r"^#+\s*", "", first)
        if len(first) < 50 and first.isupper():
            return first
    return None
