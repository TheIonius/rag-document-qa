from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "Enterprise RAG Document Intelligence"
    version: str = "1.0.0"
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    docs_dir: Path = data_dir / "documents"
    index_dir: Path = data_dir / "index"
    storage_dir: Path = data_dir / "storage"
    db_path: Path = index_dir / "rag_store.db"

    # Security and Authentication
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # Uploads and limits
    max_upload_size_bytes: int = 25 * 1024 * 1024  # 25 MB

    # Chunking defaults
    target_chunk_words: int = 250
    overlap_words: int = 40

    # Retrieval defaults
    default_top_k: int = 4
    bm25_weight: float = 0.45
    vector_weight: float = 0.55
    confidence_threshold: float = 0.25
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # LLM inference config
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    request_timeout_seconds: float = 15.0

    # Governance and Safety
    enable_prompt_injection_defense: bool = True
    rate_limit_per_minute: int = 120

    def __init__(self, **values):
        super().__init__(**values)
        if not self.secret_key:
            import os, secrets
            env_secret = os.getenv("CORTEX_SECRET_KEY") or os.getenv("JWT_SECRET_KEY")
            if env_secret:
                self.secret_key = env_secret
            else:
                secret_file = self.data_dir / ".secret"
                if secret_file.exists():
                    self.secret_key = secret_file.read_text(encoding="utf-8").strip()
                else:
                    self.data_dir.mkdir(parents=True, exist_ok=True)
                    generated = secrets.token_hex(32)
                    secret_file.write_text(generated, encoding="utf-8")
                    self.secret_key = generated

settings = Settings()
settings.docs_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
settings.storage_dir.mkdir(parents=True, exist_ok=True)
