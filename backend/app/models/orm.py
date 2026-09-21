import enum
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class Base(DeclarativeBase):
    pass

class WorkspaceRole(str, enum.Enum):
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

class IngestionStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    READY = "READY"
    FAILED = "FAILED"

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(String(50), default=utc_now_iso, nullable=False)

    workspaces = relationship("Workspace", back_populates="organization", cascade="all, delete-orphan")

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(String(50), default=utc_now_iso, nullable=False)

    memberships = relationship("WorkspaceMembership", back_populates="user", cascade="all, delete-orphan")

class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(String(36), primary_key=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), nullable=False, index=True)
    created_at = Column(String(50), default=utc_now_iso, nullable=False)

    organization = relationship("Organization", back_populates="workspaces")
    memberships = relationship("WorkspaceMembership", back_populates="workspace", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="workspace", cascade="all, delete-orphan")

class WorkspaceMembership(Base):
    __tablename__ = "workspace_memberships"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(50), default=WorkspaceRole.VIEWER.value, nullable=False)
    created_at = Column(String(50), default=utc_now_iso, nullable=False)

    user = relationship("User", back_populates="memberships")
    workspace = relationship("Workspace", back_populates="memberships")

class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    title = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    page_count = Column(Integer, default=1, nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    created_at = Column(String(50), default=utc_now_iso, nullable=False)
    raw_text = Column(Text, nullable=False)
    collection = Column(String(100), default="General Documentation", nullable=True)

    # Production extensions
    version = Column(Integer, default=1, nullable=False)
    sha256_checksum = Column(String(64), nullable=True, index=True)
    storage_path = Column(String(500), nullable=True)
    status = Column(String(50), default="READY", nullable=False)
    effective_from = Column(String(50), nullable=True)
    effective_until = Column(String(50), nullable=True)
    uploaded_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    workspace = relationship("Workspace", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    section_title = Column(String(255), nullable=True)
    page_number = Column(Integer, default=1, nullable=False)
    text = Column(Text, nullable=False)
    word_count = Column(Integer, nullable=False)
    char_start = Column(Integer, nullable=False)
    char_end = Column(Integer, nullable=False)
    embedding_blob = Column(LargeBinary, nullable=True)
    created_at = Column(String(50), default=utc_now_iso, nullable=False)

    document = relationship("Document", back_populates="chunks")

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(String(36), primary_key=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True, index=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(50), default=IngestionStatus.QUEUED.value, nullable=False)
    progress_pct = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(String(50), default=utc_now_iso, nullable=False)
    completed_at = Column(String(50), nullable=True)

class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(String(36), primary_key=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    query = Column(Text, nullable=False)
    timestamp = Column(String(50), default=utc_now_iso, nullable=False, index=True)
    retrieved_chunks_json = Column(Text, nullable=True)
    answer = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False)
    is_out_of_scope = Column(Integer, default=0, nullable=False)
    duration_ms = Column(Integer, nullable=False)
    model_used = Column(String(100), nullable=False)

class Thread(Base):
    __tablename__ = "threads"

    id = Column(String(36), primary_key=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    title = Column(String(255), nullable=False)
    created_at = Column(String(50), default=utc_now_iso, nullable=False)
    updated_at = Column(String(50), default=utc_now_iso, nullable=False, index=True)

    messages = relationship("ThreadMessage", back_populates="thread", cascade="all, delete-orphan")

class ThreadMessage(Base):
    __tablename__ = "thread_messages"

    id = Column(String(36), primary_key=True)
    thread_id = Column(String(36), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(String(50), default=utc_now_iso, nullable=False)

    thread = relationship("Thread", back_populates="messages")

class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(String(36), primary_key=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True, index=True)
    actor_id = Column(String(36), nullable=True)
    actor_email = Column(String(255), nullable=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(100), nullable=True)
    ip_address = Column(String(50), nullable=True)
    details_json = Column(Text, nullable=True)
    created_at = Column(String(50), default=utc_now_iso, nullable=False, index=True)

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String(36), primary_key=True)
    query_log_id = Column(String(36), ForeignKey("query_logs.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    is_positive = Column(Boolean, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(String(50), default=utc_now_iso, nullable=False)

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(String(36), primary_key=True)
    workspace_id = Column(String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True, index=True)
    run_timestamp = Column(String(50), default=utc_now_iso, nullable=False, index=True)
    total_cases = Column(Integer, nullable=False)
    passed_cases = Column(Integer, nullable=False)
    avg_latency_ms = Column(Float, nullable=False)
    avg_groundedness = Column(Float, nullable=False)
    results_json = Column(Text, nullable=False)
