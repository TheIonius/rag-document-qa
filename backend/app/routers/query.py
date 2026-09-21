import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from app.security.auth import get_current_user, get_optional_user, require_workspace_membership

from app.config import settings
from app.database import get_db, record_audit_event, utc_now_iso
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    CompareRequest,
    CompareResponse,
    DocumentMetadata,
    SubQueryDecomposition,
    FeedbackCreateRequest,
    FeedbackResponse
)
from app.services.vector_engine import search_engine
from app.services.llm_rag_engine import (
    generate_rag_answer,
    decompose_query,
    generate_comparative_synthesis
)

router = APIRouter(prefix="/query", tags=["query"])

@router.post("", response_model=QueryResponse)
async def execute_rag_query(
    req: QueryRequest,
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
    user: Optional[dict] = Depends(get_optional_user),
    persist_thread: bool = True
):
    if not isinstance(user, dict):
        user = None

    if not search_engine.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload or ingest documents first."
        )

    active_ws = req.workspace_id or x_workspace_id or "ws_default"

    if active_ws != "ws_default":
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to query this workspace."
            )
        if not user.get("is_superuser"):
            with get_db() as conn:
                mem = conn.execute(
                    "SELECT id FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], active_ws)
                ).fetchone()
                if not mem:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied to this workspace."
                    )

    # 1. Hybrid Retrieval (BM25 + Dense Semantic Vectors) with configurable weights
    bm25_w = req.bm25_weight if req.bm25_weight is not None else settings.bm25_weight
    dense_w = req.dense_weight if req.dense_weight is not None else settings.vector_weight
    refusal_thresh = req.refusal_threshold if req.refusal_threshold is not None else 0.20

    sub_queries_decomposed = []
    if req.multi_hop:
        sub_qs = decompose_query(req.query)
        if len(sub_qs) > 1:
            all_chunks = []
            seen_chunk_ids = set()
            for idx, sq in enumerate(sub_qs, start=1):
                sq_chunks = search_engine.search(
                    query=sq,
                    top_k=max(2, req.top_k // len(sub_qs)),
                    doc_filter=req.document_filter,
                    workspace_id=active_ws,
                    bm25_weight=bm25_w,
                    vector_weight=dense_w
                )
                top_doc = sq_chunks[0][0].document_title if sq_chunks else "No direct match"
                sub_queries_decomposed.append(SubQueryDecomposition(
                    sub_query_id=idx,
                    sub_query=sq,
                    retrieved_chunks_count=len(sq_chunks),
                    top_source=top_doc
                ))
                for chk, score in sq_chunks:
                    if chk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chk.chunk_id)
                        all_chunks.append((chk, score))
            all_chunks.sort(key=lambda x: x[1], reverse=True)
            retrieved_chunks = all_chunks[:req.top_k]
        else:
            retrieved_chunks = search_engine.search(
                query=req.query,
                top_k=req.top_k,
                doc_filter=req.document_filter,
                workspace_id=active_ws,
                bm25_weight=bm25_w,
                vector_weight=dense_w
            )
    else:
        retrieved_chunks = search_engine.search(
            query=req.query,
            top_k=req.top_k,
            doc_filter=req.document_filter,
            workspace_id=active_ws,
            bm25_weight=bm25_w,
            vector_weight=dense_w
        )

    # 2. Multi-turn conversation context
    conversation_history = []
    if req.thread_id:
        try:
            with get_db() as conn:
                prior_rows = conn.execute("""
                    SELECT role, content FROM thread_messages
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT 4
                """, (req.thread_id,)).fetchall()
                for r in reversed(prior_rows):
                    conversation_history.append({"role": r["role"], "content": r["content"]})
        except Exception:
            pass

    # 3. RAG Generation, Prompt Defense & Citation Extraction
    response = await generate_rag_answer(
        req.query,
        retrieved_chunks,
        refusal_threshold=refusal_thresh,
        sub_queries=sub_queries_decomposed,
        conversation_history=conversation_history
    )

    # 4. Log query execution & Thread Persistence
    log_id = f"qry_{uuid.uuid4().hex[:12]}"
    now = utc_now_iso()
    chunks_meta = [
        {"chunk_id": c[0].chunk_id, "doc": c[0].document_title, "score": c[1]}
        for c in retrieved_chunks
    ]
    citations_data = [c.model_dump() for c in response.citations]

    active_thread_id = req.thread_id
    try:
        with get_db() as conn:
            if persist_thread:
                if active_thread_id:
                    thread_exists = conn.execute("SELECT id FROM threads WHERE id = ?", (active_thread_id,)).fetchone()
                    if not thread_exists:
                        active_thread_id = None

                if not active_thread_id:
                    active_thread_id = f"th_{uuid.uuid4().hex[:12]}"
                    clean_title = req.query.strip().rstrip("?").strip()
                    if len(clean_title) > 55:
                        clean_title = clean_title[:52] + "..."
                    conn.execute("""
                        INSERT INTO threads (id, workspace_id, user_id, title, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (active_thread_id, active_ws, user["id"] if user else None, clean_title, now, now))
                else:
                    conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now, active_thread_id))

                user_msg_id = f"msg_{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO thread_messages (id, thread_id, role, content, citations_json, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_msg_id, active_thread_id, "user", req.query, "[]", "{}", now))

                asst_msg_id = f"msg_{uuid.uuid4().hex[:12]}"
                meta_payload = {
                    "confidence_score": response.confidence_score,
                    "groundedness_score": response.groundedness_score,
                    "processing_time_ms": response.processing_time_ms,
                    "model_used": response.model_used,
                    "is_out_of_scope": response.is_out_of_scope,
                    "query_log_id": log_id
                }
                conn.execute("""
                    INSERT INTO thread_messages (id, thread_id, role, content, citations_json, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    asst_msg_id, active_thread_id, "assistant", response.answer,
                    json.dumps(citations_data), json.dumps(meta_payload), now
                ))

            conn.execute("""
                INSERT INTO query_logs (
                    id, workspace_id, user_id, query, timestamp, retrieved_chunks_json, answer,
                    citations_json, confidence, is_out_of_scope, duration_ms, model_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_id, active_ws, user["id"] if user else None, req.query, now, json.dumps(chunks_meta), response.answer,
                json.dumps(citations_data), response.confidence_score,
                1 if response.is_out_of_scope else 0,
                response.processing_time_ms, response.model_used
            ))
    except Exception as e:
        print(f"[Query Router Warning] Failed to persist thread/query log: {e}")

    # Record audit event
    record_audit_event(
        action="rag_query",
        resource_type="query_log",
        resource_id=log_id,
        workspace_id=active_ws,
        actor_id=user["id"] if user else None,
        actor_email=user["email"] if user else None,
        details={
            "query": req.query[:100],
            "confidence": response.confidence_score,
            "citations_count": len(response.citations),
            "is_out_of_scope": response.is_out_of_scope
        }
    )

    response.thread_id = active_thread_id
    response.workspace_id = active_ws
    response.query_log_id = log_id
    return response

@router.post("/compare", response_model=CompareResponse)
def compare_documents(
    req: CompareRequest,
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
    user: Optional[dict] = Depends(get_optional_user)
):
    if not search_engine.is_indexed:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload or ingest documents first."
        )

    with get_db() as conn:
        row_a = conn.execute("SELECT * FROM documents WHERE id = ?", (req.doc_id_a,)).fetchone()
        row_b = conn.execute("SELECT * FROM documents WHERE id = ?", (req.doc_id_b,)).fetchone()

    if not row_a:
        raise HTTPException(status_code=404, detail=f"Baseline document {req.doc_id_a} not found")
    if not row_b:
        raise HTTPException(status_code=404, detail=f"Comparison document {req.doc_id_b} not found")

    doc_a = DocumentMetadata.model_validate(dict(row_a))
    doc_b = DocumentMetadata.model_validate(dict(row_b))

    active_ws = x_workspace_id or doc_a.workspace_id or "ws_default"

    # Cross-tenant check: both documents must belong to the active workspace
    if (doc_a.workspace_id or "ws_default") != active_ws or (doc_b.workspace_id or "ws_default") != active_ws:
        if (doc_a.workspace_id or "ws_default") != (doc_b.workspace_id or "ws_default"):
            raise HTTPException(
                status_code=400,
                detail="Cannot compare documents across different workspaces. Both documents must belong to the same workspace."
            )
        active_ws = doc_a.workspace_id or "ws_default"

    if user and not user.get("is_superuser") and active_ws != "ws_default":
        with get_db() as conn:
            mem = conn.execute(
                "SELECT id FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                (user["id"], active_ws)
            ).fetchone()
            if not mem:
                raise HTTPException(status_code=403, detail="Access denied to the workspace of these documents.")

    bm25_w = req.bm25_weight if req.bm25_weight is not None else settings.bm25_weight
    dense_w = req.dense_weight if req.dense_weight is not None else settings.vector_weight

    chunks_a = search_engine.search(
        query=req.query,
        top_k=req.top_k_per_doc,
        doc_filter=[req.doc_id_a],
        workspace_id=active_ws,
        bm25_weight=bm25_w,
        vector_weight=dense_w
    )
    if not chunks_a:
        doc_a_chunks = search_engine.get_document_chunks(req.doc_id_a, workspace_id=active_ws)
        chunks_a = [(c, 50.0) for c in doc_a_chunks[:req.top_k_per_doc]]

    chunks_b = search_engine.search(
        query=req.query,
        top_k=req.top_k_per_doc,
        doc_filter=[req.doc_id_b],
        workspace_id=active_ws,
        bm25_weight=bm25_w,
        vector_weight=dense_w
    )
    if not chunks_b:
        doc_b_chunks = search_engine.get_document_chunks(req.doc_id_b, workspace_id=active_ws)
        chunks_b = [(c, 50.0) for c in doc_b_chunks[:req.top_k_per_doc]]

    return generate_comparative_synthesis(
        query=req.query,
        doc_a_chunks=chunks_a,
        doc_b_chunks=chunks_b,
        doc_a=doc_a,
        doc_b=doc_b
    )

@router.get("/history")
def get_query_history(
    limit: int = Query(20, ge=1, le=100),
    workspace_id: Optional[str] = Query(None),
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
    user: Optional[dict] = Depends(get_optional_user)
):
    active_ws = workspace_id or x_workspace_id or "ws_default"
    with get_db() as conn:
        if active_ws != "ws_default":
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required to view query history.")
            if not user.get("is_superuser"):
                mem = conn.execute(
                    "SELECT id FROM workspace_memberships WHERE user_id = ? AND workspace_id = ?",
                    (user["id"], active_ws)
                ).fetchone()
                if not mem:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this workspace.")

        rows = conn.execute("""
            SELECT id, query, timestamp, answer, citations_json, confidence, is_out_of_scope, duration_ms, model_used
            FROM query_logs
            WHERE (workspace_id = ? OR (workspace_id IS NULL AND ? = 'ws_default'))
            ORDER BY timestamp DESC
            LIMIT ?
        """, (active_ws, active_ws, limit)).fetchall()

        history = []
        for r in rows:
            d = dict(r)
            d["is_out_of_scope"] = bool(d["is_out_of_scope"])
            d["citations"] = json.loads(d["citations_json"]) if d["citations_json"] else []
            del d["citations_json"]
            history.append(d)

        return history

@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_query_feedback(
    req: FeedbackCreateRequest,
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
    user: Optional[dict] = Depends(get_optional_user)
):
    feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
    now = utc_now_iso()
    ws_id = x_workspace_id
    with get_db() as conn:
        if not ws_id and req.query_log_id:
            row = conn.execute("SELECT workspace_id FROM query_logs WHERE id = ?", (req.query_log_id,)).fetchone()
            if row and row["workspace_id"]:
                ws_id = row["workspace_id"]
        if not ws_id and user:
            m = conn.execute(
                "SELECT workspace_id FROM workspace_memberships WHERE user_id = ? ORDER BY created_at ASC LIMIT 1",
                (user["id"],)
            ).fetchone()
            if m and m["workspace_id"]:
                ws_id = m["workspace_id"]
        if not ws_id:
            ws_id = "ws_default"

        conn.execute("""
            INSERT INTO feedback (id, query_log_id, is_positive, notes, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (feedback_id, req.query_log_id, 1 if req.is_positive else 0, req.notes, now))

    record_audit_event(
        action="submit_feedback",
        resource_type="feedback",
        resource_id=feedback_id,
        workspace_id=ws_id,
        actor_id=user["id"] if user else None,
        actor_email=user["email"] if user else None,
        details={"is_positive": req.is_positive, "query_log_id": req.query_log_id}
    )

    return FeedbackResponse(
        id=feedback_id,
        query_log_id=req.query_log_id,
        is_positive=req.is_positive,
        notes=req.notes,
        created_at=now
    )
