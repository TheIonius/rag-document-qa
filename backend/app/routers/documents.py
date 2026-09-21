import uuid
from datetime import datetime, timezone
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile, status

from app.config import settings
from app.database import get_db, record_audit_event, utc_now_iso
from app.security.auth import get_optional_user
from app.models.schemas import (
    DocumentMetadata,
    ChunkDetail,
    ChunkAnalyticsResponse,
    ChunkAnalyticsBucket,
    ChunkOverlapPair,
    SectionChunkCount,
    IngestionJobResponse,
    DocumentUploadResponse
)
from app.services.document_parser import parse_document_content
from app.services.chunker import chunk_document
from app.services.storage import storage_service
from app.services.vector_engine import IndexedChunk, search_engine

router = APIRouter(prefix="/documents", tags=["documents"])

@router.get("", response_model=List[DocumentMetadata])
def list_documents(
    workspace_id: Optional[str] = Query(None),
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
    user: Optional[dict] = Depends(get_optional_user)
):
    active_ws = workspace_id or x_workspace_id or "ws_default"
    with get_db() as conn:
        if active_ws != "ws_default":
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to access this workspace.")
            if not user.get("is_superuser"):
                mem = conn.execute(
                    "SELECT id FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], active_ws)
                ).fetchone()
                if not mem:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this workspace.")

        rows = conn.execute("""
            SELECT id, filename, title, file_type, file_size, page_count, chunk_count,
                   created_at, collection, workspace_id, version, sha256_checksum,
                   status, effective_from, effective_until, uploaded_by_id
            FROM documents
            WHERE (workspace_id = ? OR (workspace_id IS NULL AND ? = 'ws_default'))
            ORDER BY created_at DESC
        """, (active_ws, active_ws)).fetchall()
        return [DocumentMetadata.model_validate(dict(r)) for r in rows]

@router.get("/jobs/{job_id}", response_model=IngestionJobResponse)
def get_ingestion_job(job_id: str):
    with get_db() as conn:
        row = conn.execute("""
            SELECT id, workspace_id, document_id, status, progress_pct,
                   error_message, created_at, completed_at
            FROM ingestion_jobs
            WHERE id = ?
        """, (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Ingestion job not found")
        return IngestionJobResponse.model_validate(dict(row))

@router.get("/{document_id}")
def get_document_details(document_id: str, user: Optional[dict] = Depends(get_optional_user)):
    with get_db() as conn:
        doc = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_ws = doc["workspace_id"] or "ws_default"
        if doc_ws != "ws_default":
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to view this document.")
            if not user.get("is_superuser"):
                mem = conn.execute(
                    "SELECT id FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], doc_ws)
                ).fetchone()
                if not mem:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this document.")

        chunks = conn.execute("""
            SELECT id, document_id, chunk_index, section_title, page_number, text, word_count, char_start, char_end
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
        """, (document_id,)).fetchall()

        return {
            "document": DocumentMetadata.model_validate(dict(doc)),
            "chunks": [ChunkDetail.model_validate(dict(c)) for c in chunks]
        }

@router.get("/{document_id}/chunk-analytics", response_model=ChunkAnalyticsResponse)
def get_chunk_analytics(document_id: str, user: Optional[dict] = Depends(get_optional_user)):
    with get_db() as conn:
        doc = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        doc_ws = doc["workspace_id"] or "ws_default"
        if doc_ws != "ws_default":
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to view chunk analytics.")
            if not user.get("is_superuser"):
                mem = conn.execute(
                    "SELECT id FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], doc_ws)
                ).fetchone()
                if not mem:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this document.")

        rows = conn.execute("""
            SELECT id, document_id, chunk_index, section_title, page_number, text, word_count, char_start, char_end
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
        """, (document_id,)).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No chunks found for document")

    chunks = [dict(r) for r in rows]
    total_chunks = len(chunks)
    word_counts = [c["word_count"] for c in chunks]
    total_words = sum(word_counts)
    avg_words = round(total_words / total_chunks, 1) if total_chunks else 0.0
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0

    # Bucket histogram
    buckets = [
        {"range_label": "0-100 words", "min": 0, "max": 100, "count": 0},
        {"range_label": "101-175 words", "min": 101, "max": 175, "count": 0},
        {"range_label": "176-250 words", "min": 176, "max": 250, "count": 0},
        {"range_label": "251-325 words", "min": 251, "max": 325, "count": 0},
        {"range_label": "326+ words", "min": 326, "max": 999999, "count": 0},
    ]
    for wc in word_counts:
        for b in buckets:
            if b["min"] <= wc <= b["max"]:
                b["count"] += 1
                break

    histogram = [ChunkAnalyticsBucket(range_label=b["range_label"], count=b["count"]) for b in buckets]

    # Sections breakdown
    sec_map = {}
    for c in chunks:
        sec = c["section_title"] or "Overview / General"
        if sec not in sec_map:
            sec_map[sec] = {"chunk_count": 0, "total_words": 0}
        sec_map[sec]["chunk_count"] += 1
        sec_map[sec]["total_words"] += c["word_count"]

    sections_breakdown = [
        SectionChunkCount(section_title=s, chunk_count=data["chunk_count"], total_words=data["total_words"])
        for s, data in sec_map.items()
    ]

    # Overlap detection between consecutive chunks
    overlap_pairs = []
    for i in range(len(chunks) - 1):
        c1_words = chunks[i]["text"].split()
        c2_words = chunks[i+1]["text"].split()
        overlap_cnt = 0
        overlap_snip = ""

        max_k = min(40, len(c1_words), len(c2_words))
        for k in range(max_k, 4, -1):
            s1 = [w.lower().strip(".,;:\"'()[]*#`") for w in c1_words[-k:]]
            s2 = [w.lower().strip(".,;:\"'()[]*#`") for w in c2_words[:k]]
            if s1 == s2:
                overlap_cnt = k
                overlap_snip = " ".join(c2_words[:k])
                break

        if overlap_cnt > 0:
            overlap_pairs.append(ChunkOverlapPair(
                chunk_index_a=chunks[i]["chunk_index"],
                chunk_index_b=chunks[i+1]["chunk_index"],
                overlap_word_count=overlap_cnt,
                overlap_snippet=overlap_snip
            ))

    return ChunkAnalyticsResponse(
        document_id=doc["id"],
        document_title=doc["title"],
        total_chunks=total_chunks,
        total_words=total_words,
        avg_chunk_words=avg_words,
        min_chunk_words=min_words,
        max_chunk_words=max_words,
        histogram_buckets=histogram,
        sections_breakdown=sections_breakdown,
        overlap_pairs=overlap_pairs
    )

@router.post("/upload", response_model=DocumentMetadata, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    workspace_id: Optional[str] = Query(None),
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
    user: Optional[dict] = Depends(get_optional_user)
):
    active_ws = workspace_id or x_workspace_id or "ws_default"

    # Role check: viewer cannot upload
    if active_ws != "ws_default":
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to upload documents to this workspace.")
        if not user.get("is_superuser"):
            with get_db() as conn:
                mem = conn.execute(
                    "SELECT role FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], active_ws)
                ).fetchone()
                if not mem or mem["role"] not in ("owner", "admin", "editor"):
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Editor or Admin role required to upload documents.")

    filename = file.filename
    allowed_exts = (".pdf", ".md", ".markdown", ".txt")
    if not any(filename.lower().endswith(ext) for ext in allowed_exts):
        raise HTTPException(status_code=400, detail="Only .pdf, .md, and .txt documents are supported.")

    content = await file.read(settings.max_upload_size_bytes + 1)
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload limit of {settings.max_upload_size_bytes // (1024 * 1024)}MB."
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Storage service & SHA-256 duplicate detection
    storage_path, checksum, existing_dup = storage_service.save_file(
        workspace_id=active_ws,
        filename=filename,
        content=content
    )

    now = utc_now_iso()
    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    job_id = f"job_{uuid.uuid4().hex[:10]}"

    # Parse and chunk document
    parsed = parse_document_content(filename, content)
    chunks = chunk_document(parsed)

    # Ingestion job record
    with get_db() as conn:
        conn.execute("""
            INSERT INTO ingestion_jobs (
                id, workspace_id, document_id, status, progress_pct, created_at
            ) VALUES (?, ?, ?, 'PARSING', 20, ?)
        """, (job_id, active_ws, doc_id, now))

        # Mark prior versions in this workspace as SUPERSEDED
        conn.execute("""
            UPDATE documents
            SET status = 'SUPERSEDED'
            WHERE workspace_id = ? AND filename = ? AND status = 'READY'
        """, (active_ws, parsed.filename))

        # Compute version: check if document with same filename exists in this workspace
        latest_ver_row = conn.execute("""
            SELECT MAX(version) FROM documents WHERE workspace_id = ? AND filename = ?
        """, (active_ws, parsed.filename)).fetchone()
        version = (latest_ver_row[0] or 0) + 1 if latest_ver_row and latest_ver_row[0] else 1

        conn.execute("""
            INSERT INTO documents (
                id, workspace_id, filename, title, file_type, file_size,
                page_count, chunk_count, created_at, raw_text, collection,
                version, sha256_checksum, storage_path, status, uploaded_by_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'General Documentation', ?, ?, ?, 'READY', ?)
        """, (
            doc_id, active_ws, parsed.filename, parsed.title, parsed.file_type,
            parsed.file_size, len(parsed.pages), len(chunks), now, parsed.raw_text,
            version, checksum, storage_path, user["id"] if user else None
        ))

        # Generate embeddings for chunks
        texts_to_embed = [c.text for c in chunks]
        embed_matrix = search_engine._embed_texts(texts_to_embed)

        chunk_records = []
        for idx, c in enumerate(chunks):
            chk_id = f"chk_{doc_id}_{c.chunk_index}"
            blob = embed_matrix[idx].tobytes() if idx < len(embed_matrix) else None
            chunk_records.append((
                chk_id, doc_id, c.chunk_index, c.section_title,
                c.page_number, c.text, c.word_count, c.char_start, c.char_end, blob, now
            ))

        conn.executemany("""
            INSERT INTO chunks (
                id, document_id, chunk_index, section_title,
                page_number, text, word_count, char_start, char_end, embedding_blob, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, chunk_records)

        # Update job to complete
        conn.execute("""
            UPDATE ingestion_jobs
            SET status = 'READY', progress_pct = 100, completed_at = ?
            WHERE id = ?
        """, (utc_now_iso(), job_id))

    # Audit event
    record_audit_event(
        action="document_upload",
        resource_type="document",
        resource_id=doc_id,
        workspace_id=active_ws,
        actor_id=user["id"] if user else None,
        actor_email=user["email"] if user else None,
        details={
            "filename": parsed.filename,
            "chunks_count": len(chunks),
            "version": version,
            "sha256": checksum
        }
    )

    # Refresh search engine index
    refresh_index_from_db()

    return DocumentMetadata(
        id=doc_id,
        filename=parsed.filename,
        title=parsed.title,
        file_type=parsed.file_type,
        file_size=parsed.file_size,
        page_count=len(parsed.pages),
        chunk_count=len(chunks),
        created_at=now,
        workspace_id=active_ws,
        version=version,
        sha256_checksum=checksum,
        status="READY"
    )

@router.delete("/{document_id}")
def delete_document(document_id: str, user: Optional[dict] = Depends(get_optional_user)):
    with get_db() as conn:
        doc = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        storage_path = doc["storage_path"]
        workspace_id = doc["workspace_id"] or "ws_default"

        if workspace_id != "ws_default":
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to delete documents from this workspace.")
            if not user.get("is_superuser"):
                mem = conn.execute(
                    "SELECT role FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], workspace_id)
                ).fetchone()
                if not mem or mem["role"] not in ("owner", "admin", "editor"):
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Editor or Admin role required to delete documents.")

        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    storage_service.delete_file(storage_path)

    record_audit_event(
        action="document_delete",
        resource_type="document",
        resource_id=document_id,
        workspace_id=workspace_id,
        actor_id=user["id"] if user else None,
        actor_email=user["email"] if user else None,
        details={"title": doc["title"], "filename": doc["filename"]}
    )

    refresh_index_from_db()
    return {"status": "success", "deleted_document_id": document_id}

def refresh_index_from_db():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT 
                c.id as chunk_id, c.document_id, d.title as document_title,
                c.section_title, c.page_number, c.text, c.word_count,
                c.embedding_blob, d.workspace_id, d.version, d.effective_from, d.effective_until
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.status = 'READY'
            ORDER BY c.document_id, c.chunk_index
        """).fetchall()

        indexed_chunks = []
        for r in rows:
            emb = None
            if r["embedding_blob"]:
                try:
                    emb = np.frombuffer(r["embedding_blob"], dtype=np.float32)
                except Exception:
                    emb = None

            indexed_chunks.append(IndexedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                document_title=r["document_title"],
                section_title=r["section_title"],
                page_number=r["page_number"],
                text=r["text"],
                word_count=r["word_count"],
                workspace_id=r["workspace_id"] or "ws_default",
                embedding=emb,
                effective_from=r["effective_from"],
                effective_until=r["effective_until"],
                version=r["version"] or 1
            ))

        search_engine.index_chunks(indexed_chunks, persist_to_db=True)
