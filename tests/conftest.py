import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from app.config import settings
from app.database import init_db
from app.main import app
from app.services.vector_engine import search_engine

@pytest.fixture
def temp_env(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db_path = Path(f.name)

    monkeypatch.setattr(settings, "db_path", temp_db_path)
    init_db(temp_db_path)

    monkeypatch.setattr(settings, "ollama_url", "http://127.0.0.1:9")
    monkeypatch.setattr(settings, "request_timeout_seconds", 0.1)

    # Ingest samples for testing
    from scripts.ingest_samples import ingest_all_documents
    ingest_all_documents()

    yield temp_db_path

    if temp_db_path.exists():
        temp_db_path.unlink()

@pytest.fixture
def client(temp_env):
    return TestClient(app)
