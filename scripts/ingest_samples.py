import hashlib
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings
from app.database import init_db, get_db
from app.services.document_parser import parse_document_content
from app.services.chunker import chunk_document
from app.services.vector_engine import IndexedChunk, search_engine

def ingest_all_documents():
    init_db()
    docs_dir = settings.docs_dir
    if not docs_dir.exists():
        print(f"[Ingest] Documents directory {docs_dir} does not exist.")
        return

    doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.txt"))
    print(f"[Ingest] Found {len(doc_files)} documents to process.")

    total_chunks_indexed = 0

    for fpath in doc_files:
        content_bytes = fpath.read_bytes()
        checksum = hashlib.sha256(content_bytes).hexdigest()
        parsed = parse_document_content(fpath.name, content_bytes)

        doc_id = f"doc_{fpath.stem.lower()}"
        now = datetime.now(timezone.utc).isoformat()

        # Generate chunks
        chunks = chunk_document(parsed)
        print(f"[Ingest] '{parsed.title}' ({fpath.name}) -> {len(chunks)} chunks.")

        # Determine enterprise collection
        fname_lower = fpath.name.lower()
        if "security" in fname_lower:
            collection = "Security & Infrastructure"
        elif "benefit" in fname_lower or "handbook" in fname_lower:
            collection = "People Operations & Policies"
        elif "api" in fname_lower or "spec" in fname_lower:
            collection = "Developer & API Specifications"
        else:
            collection = "General Documentation"

        # Compute embeddings for chunks
        texts_to_embed = [c.text for c in chunks]
        embeddings = search_engine._embed_texts(texts_to_embed)

        with get_db() as conn:
            # Check if document already exists
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

            conn.execute("""
                INSERT INTO documents (
                    id, workspace_id, filename, title, file_type, file_size,
                    page_count, chunk_count, created_at, raw_text, collection,
                    version, sha256_checksum, status
                ) VALUES (?, 'ws_default', ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 'READY')
            """, (
                doc_id, parsed.filename, parsed.title, parsed.file_type,
                parsed.file_size, len(parsed.pages), len(chunks), now, parsed.raw_text,
                collection, checksum
            ))

            chunk_records = []
            for idx, c in enumerate(chunks):
                chunk_id = f"chk_{doc_id}_{c.chunk_index}"
                blob = embeddings[idx].tobytes() if idx < len(embeddings) else None
                chunk_records.append((
                    chunk_id, doc_id, c.chunk_index, c.section_title,
                    c.page_number, c.text, c.word_count, c.char_start, c.char_end, blob, now
                ))

            conn.executemany("""
                INSERT INTO chunks (
                    id, document_id, chunk_index, section_title,
                    page_number, text, word_count, char_start, char_end, embedding_blob, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, chunk_records)

            total_chunks_indexed += len(chunks)

    # Re-index search engine from database
    reload_search_index()
    print(f"[Ingest] Ingestion complete. Total indexed chunks: {total_chunks_indexed}.")

def reload_search_index():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT 
                c.id as chunk_id, c.document_id, d.title as document_title,
                c.section_title, c.page_number, c.text, c.word_count,
                c.embedding_blob, d.workspace_id, d.version, d.effective_from, d.effective_until
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            ORDER BY c.document_id, c.chunk_index
        """).fetchall()

        indexed_chunks = [
            IndexedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                document_title=r["document_title"],
                section_title=r["section_title"],
                page_number=r["page_number"],
                text=r["text"],
                word_count=r["word_count"],
                workspace_id=r["workspace_id"] or "ws_default",
                embedding=np.frombuffer(r["embedding_blob"], dtype=np.float32) if r["embedding_blob"] else None,
                effective_from=r["effective_from"],
                effective_until=r["effective_until"],
                version=r["version"] or 1
            )
            for r in rows
        ]
        search_engine.index_chunks(indexed_chunks, persist_to_db=False)
        print(f"[SearchEngine] Successfully built BM25 & Semantic vector index over {len(indexed_chunks)} chunks.")

if __name__ == "__main__":
    ingest_all_documents()
