from app.services.document_parser import ParsedDocument
from app.services.chunker import chunk_document, split_markdown_sections

def test_split_markdown_sections():
    md = """# Main Header
Intro paragraph here.

## Section 1: Security
Security requirements details.

## Section 2: Benefits
Benefits details.
"""
    sections = split_markdown_sections(md)
    assert len(sections) == 3
    assert sections[0][0] == "Main Header"
    assert sections[1][0] == "Section 1: Security"
    assert sections[2][0] == "Section 2: Benefits"

def test_chunk_document_preserves_metadata():
    parsed = ParsedDocument(
        filename="test_policy.md",
        title="Test Policy",
        file_type="md",
        file_size=500,
        raw_text="# Section Alpha\nThis is paragraph one.\n\n## Section Beta\nThis is paragraph two.",
        pages=[(1, "")]
    )
    chunks = chunk_document(parsed, target_words=100, overlap_words=20)
    assert len(chunks) >= 2
    assert chunks[0].section_title in ("Section Alpha", "Test Policy")
    assert chunks[0].word_count > 0
    assert chunks[1].section_title == "Section Beta"
