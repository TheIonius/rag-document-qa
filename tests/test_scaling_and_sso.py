import asyncio
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.vector_store import ApproximateNearestNeighborAdapter, get_vector_store_adapter
from app.services.event_bus import WorkspaceEventBus, workspace_event_bus


def test_ann_vector_adapter_clustering_and_search():
    adapter = ApproximateNearestNeighborAdapter(n_clusters=4, n_probe=2)
    dim = 64
    rng = np.random.default_rng(42)

    records = []
    for i in range(40):
        v = rng.standard_normal(dim).astype(np.float32)
        records.append({
            "chunk_id": f"chk_{i}",
            "vector": v,
            "workspace_id": "ws_test_ann"
        })

    upserted = adapter.upsert_vectors(records)
    assert upserted == 40
    assert adapter.count("ws_test_ann") == 40
    assert adapter.count() == 40

    # Search with exact match of chunk 5
    target_vec = records[5]["vector"]
    results = adapter.search_vectors(target_vec, top_k=5, workspace_id="ws_test_ann")
    assert len(results) > 0
    # Top result should be chunk 5 with high similarity (~1.0)
    top_id, top_score = results[0]
    assert top_id == "chk_5"
    assert top_score > 0.99

    # Test deletion and re-indexing
    deleted = adapter.delete_vectors(["chk_5"])
    assert deleted == 1
    assert adapter.count("ws_test_ann") == 39

    # Re-search, chunk 5 should no longer be in results
    post_delete_results = adapter.search_vectors(target_vec, top_k=5, workspace_id="ws_test_ann")
    retrieved_ids = [r[0] for r in post_delete_results]
    assert "chk_5" not in retrieved_ids


def test_vector_store_factory_ann():
    adapter = get_vector_store_adapter("ann")
    assert isinstance(adapter, ApproximateNearestNeighborAdapter)


@pytest.mark.asyncio
async def test_workspace_event_bus_pub_sub():
    bus = WorkspaceEventBus()
    ws_id = "ws_test_bus"

    queue = await bus.subscribe(ws_id)
    assert ws_id in bus._subscribers

    test_data = {"document_id": "doc_123", "filename": "policy.docx"}
    await bus.publish(ws_id, "document_indexed", test_data)

    item = await asyncio.wait_for(queue.get(), timeout=2.0)
    assert item["type"] == "document_indexed"
    assert item["workspace_id"] == ws_id
    assert item["data"]["document_id"] == "doc_123"

    await bus.unsubscribe(ws_id, queue)
    assert ws_id not in bus._subscribers


def test_sso_providers_endpoint():
    client = TestClient(app)
    res = client.get("/api/v1/auth/sso/providers")
    assert res.status_code == 200
    providers = res.json()
    assert isinstance(providers, list)
    ids = [p["id"] for p in providers]
    assert "google" in ids
    assert "okta" in ids
    assert "azure_ad" in ids
    assert "mock_oidc" in ids


def test_sso_callback_auto_provisioning_and_login():
    client = TestClient(app)
    sso_email = "alex.rivera@enterprise-partner.corp"

    # 1. New user SSO callback creates account and personal workspace
    payload = {
        "provider": "okta",
        "email": sso_email,
        "full_name": "Alex Rivera",
        "provider_user_id": "okta-sub-98765"
    }
    res = client.post("/api/v1/auth/sso/callback", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == sso_email
    assert data["user"]["full_name"] == "Alex Rivera"
    first_user_id = data["user"]["id"]

    # 2. Existing user logging in again via SSO returns existing profile without conflict
    res2 = client.post("/api/v1/auth/sso/callback", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["user"]["id"] == first_user_id
    assert data2["user"]["email"] == sso_email

    # 3. Verify user can access /auth/me with the issued SSO token
    token = data["access_token"]
    me_res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
from app.routers.workspaces import workspace_events_stream

@pytest.mark.asyncio
async def test_workspace_sse_generator_and_streaming():
    bus = WorkspaceEventBus()
    queue = await bus.subscribe("ws_test_sse")
    gen = bus.sse_generator("ws_test_sse", queue)
    
    # 1. First event is connected handshake
    first_chunk = await anext(gen)
    assert "event: connected" in first_chunk
    assert "ws_test_sse" in first_chunk

    # 2. Publish event and verify stream yields formatted SSE frame
    await bus.publish("ws_test_sse", "document_indexed", {"document_id": "doc_999", "status": "READY"})
    second_chunk = await anext(gen)
    assert "event: document_indexed" in second_chunk
    assert "doc_999" in second_chunk

    # 3. Clean closure and unsubscription
    await gen.aclose()
    assert "ws_test_sse" not in bus._subscribers


@pytest.mark.asyncio
async def test_workspace_events_stream_route_response():
    resp = await workspace_events_stream("ws_default")
    assert resp.media_type == "text/event-stream"
    assert resp.headers["Cache-Control"] == "no-cache, no-transform"
    assert resp.headers["Connection"] == "keep-alive"


