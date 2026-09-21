def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_list_and_get_documents(client):
    res = client.get("/api/documents")
    assert res.status_code == 200
    docs = res.json()
    assert len(docs) >= 3

    # Check details of first document
    doc_id = docs[0]["id"]
    detail_res = client.get(f"/api/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert "document" in detail
    assert "chunks" in detail
    assert len(detail["chunks"]) > 0

def test_query_answering_with_citations(client):
    res = client.post("/api/query", json={
        "query": "What is the company 401(k) matching formula?",
        "top_k": 3
    })
    assert res.status_code == 200
    data = res.json()
    assert data["is_out_of_scope"] is False
    assert len(data["citations"]) > 0
    assert data["citations"][0]["verification_status"] in ("verified", "partial_match")
    assert "100%" in data["answer"] or "4%" in data["answer"]

def test_out_of_scope_hallucination_prevention(client):
    res = client.post("/api/query", json={
        "query": "What is the recipe for baking chocolate brownies?",
        "top_k": 3
    })
    assert res.status_code == 200
    data = res.json()
    assert data["is_out_of_scope"] is True
    assert "cannot find information" in data["answer"].lower()

def test_evaluation_benchmark(client):
    res = client.get("/api/evaluation/benchmark")
    assert res.status_code == 200
    benchmarks = res.json()
    assert len(benchmarks) >= 5
    for b in benchmarks:
        assert "PASSED" in b["status"] or "PARTIAL" in b["status"]

def test_retrieval_tuning_weights_and_threshold(client):
    # Test with custom weights: heavy BM25 vs heavy Dense
    res = client.post("/api/query", json={
        "query": "What is the company 401(k) matching formula?",
        "top_k": 2,
        "dense_weight": 0.8,
        "bm25_weight": 0.2,
        "refusal_threshold": 0.15
    })
    assert res.status_code == 200
    data = res.json()
    assert data["is_out_of_scope"] is False
    assert len(data["citations"]) > 0

    # Test that a very strict refusal threshold triggers guardrail refusal
    res_strict = client.post("/api/query", json={
        "query": "What is the company 401(k) matching formula?",
        "top_k": 2,
        "refusal_threshold": 5.0
    })
    assert res_strict.status_code == 200
    assert res_strict.json()["is_out_of_scope"] is True

def test_multi_hop_query_and_sentence_groundedness(client):
    res = client.post("/api/query", json={
        "query": "What is the 401(k) company matching formula and what are the MFA requirements for production?",
        "top_k": 4,
        "multi_hop": True
    })
    assert res.status_code == 200
    data = res.json()
    assert data["is_out_of_scope"] is False
    assert len(data["citations"]) > 0
    assert "sentence_evaluations" in data
    assert len(data["sentence_evaluations"]) > 0
    # Verify groundedness evaluation fields
    first_sent = data["sentence_evaluations"][0]
    assert "status" in first_sent
    assert first_sent["status"] in ("directly_grounded", "synthesized_bridge", "unsupported")
    assert "confidence" in first_sent

def test_chunk_analytics(client):
    # Fetch first doc
    docs_res = client.get("/api/documents")
    assert docs_res.status_code == 200
    docs = docs_res.json()
    assert len(docs) > 0
    doc_id = docs[0]["id"]

    res = client.get(f"/api/documents/{doc_id}/chunk-analytics")
    assert res.status_code == 200
    analytics = res.json()
    assert analytics["document_id"] == doc_id
    assert analytics["total_chunks"] > 0
    assert analytics["total_words"] > 0
    assert len(analytics["histogram_buckets"]) > 0
    assert len(analytics["sections_breakdown"]) > 0

def test_comparative_document_analysis(client):
    docs_res = client.get("/api/documents")
    docs = docs_res.json()
    assert len(docs) >= 2
    doc_a_id = docs[0]["id"]
    doc_b_id = docs[1]["id"]

    res = client.post("/api/query/compare", json={
        "query": "Compare core guidelines and employee compliance requirements",
        "doc_id_a": doc_a_id,
        "doc_id_b": doc_b_id,
        "top_k_per_doc": 2
    })
    assert res.status_code == 200
    data = res.json()
    assert data["doc_a"]["id"] == doc_a_id
    assert data["doc_b"]["id"] == doc_b_id
    assert len(data["dimensions"]) >= 2
    assert len(data["doc_a_citations"]) > 0
    assert len(data["doc_b_citations"]) > 0
    assert len(data["executive_synthesis"]) > 0

def test_research_threads_lifecycle_and_followup(client):
    # 1. Create thread
    th_res = client.post("/api/threads", json={"title": "Cloud Security Protocol Review"})
    assert th_res.status_code == 201
    th_data = th_res.json()
    thread_id = th_data["id"]
    assert th_data["title"] == "Cloud Security Protocol Review"

    # 2. Query 1 within thread
    q1_res = client.post("/api/query", json={
        "query": "What are the MFA requirements and prohibited authentication methods?",
        "top_k": 3,
        "thread_id": thread_id
    })
    assert q1_res.status_code == 200
    q1_data = q1_res.json()
    assert q1_data["thread_id"] == thread_id

    # 3. Follow-up Query 2 within same thread
    q2_res = client.post("/api/query", json={
        "query": "What hardware keys or protocols are approved?",
        "top_k": 2,
        "thread_id": thread_id
    })
    assert q2_res.status_code == 200
    q2_data = q2_res.json()
    assert q2_data["thread_id"] == thread_id

    # 4. Verify thread details and message ledger
    detail_res = client.get(f"/api/threads/{thread_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["thread"]["id"] == thread_id
    assert len(detail["messages"]) == 4  # 2 user queries + 2 assistant responses
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][2]["role"] == "user"
    assert detail["messages"][3]["role"] == "assistant"

    # 5. Delete thread
    del_res = client.delete(f"/api/threads/{thread_id}")
    assert del_res.status_code == 200
    assert client.get(f"/api/threads/{thread_id}").status_code == 404



