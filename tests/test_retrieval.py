from app.services.vector_engine import IndexedChunk, HybridSearchEngine

def test_hybrid_search_bm25_and_vector():
    engine = HybridSearchEngine()
    chunks = [
        IndexedChunk(
            chunk_id="c1",
            document_id="doc1",
            document_title="Security Policy",
            section_title="MFA Requirements",
            page_number=1,
            text="Multi-factor authentication (MFA) requires hardware tokens or TOTP. SMS is strictly prohibited.",
            word_count=13
        ),
        IndexedChunk(
            chunk_id="c2",
            document_id="doc2",
            document_title="Benefits Policy",
            section_title="Retirement Plan",
            page_number=1,
            text="The 401(k) retirement plan matches 100% of the first 4% and 50% of the next 2%.",
            word_count=16
        )
    ]
    engine.index_chunks(chunks)

    # Search for acronym "MFA"
    results = engine.search("MFA requirements", top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"

    # Search for "401(k) matching"
    results_ben = engine.search("401(k) match formula", top_k=1)
    assert len(results_ben) == 1
    assert results_ben[0][0].chunk_id == "c2"

def test_hybrid_search_empty_query():
    engine = HybridSearchEngine()
    results = engine.search("", top_k=4)
    assert results == []

def test_sqlite_fts5_native_search():
    from app.database import get_db, init_db
    init_db()
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO documents (id, workspace_id, filename, title, file_type, file_size, raw_text, created_at) VALUES ('doc_test_fts', 'ws_default', 'test.md', 'Test FTS', 'md', 100, 'text', '2026-09-21T00:00:00Z')")
        conn.execute("INSERT OR REPLACE INTO chunks (id, document_id, chunk_index, text, word_count, char_start, char_end, created_at) VALUES ('chk_test_fts_1', 'doc_test_fts', 0, 'Hardware security key FIDO2Quantum token is mandatory for production.', 10, 0, 60, '2026-09-21T00:00:00Z')")
        
        row = conn.execute("SELECT chunk_id, text FROM chunks_fts WHERE chunks_fts MATCH 'FIDO2Quantum'").fetchone()
        assert row is not None
        assert row["chunk_id"] == "chk_test_fts_1"
