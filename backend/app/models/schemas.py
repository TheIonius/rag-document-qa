import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

class DocumentMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    title: str
    file_type: str
    file_size: int
    page_count: int
    chunk_count: int
    created_at: str
    collection: Optional[str] = "General Documentation"
    workspace_id: Optional[str] = "ws_default"
    version: int = 1
    sha256_checksum: Optional[str] = None
    status: str = "READY"
    effective_from: Optional[str] = None
    effective_until: Optional[str] = None
    uploaded_by_id: Optional[str] = None
    security_tags: Optional[str] = "[\"public\"]"


class ChunkDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    chunk_index: int
    section_title: Optional[str] = None
    page_number: int = 1
    text: str
    word_count: int
    char_start: int
    char_end: int
    parent_chunk_id: Optional[str] = None
    parent_text: Optional[str] = None

class Citation(BaseModel):
    citation_index: int = Field(..., description="[1], [2] citation reference index")
    document_id: str
    document_title: str
    section_title: Optional[str] = None
    page_number: int = 1
    chunk_id: str
    exact_quote: str = Field(..., description="Exact textual excerpt from source chunk")
    verification_status: str = Field("verified", description="verified, partial_match, or unverified")
    relevance_score: float = Field(..., description="Hybrid retrieval score")
    bm25_rank: Optional[int] = Field(None, description="Rank in BM25 lexical search")
    dense_rank: Optional[int] = Field(None, description="Rank in Dense semantic search")
    matched_terms: Optional[List[str]] = Field(default_factory=list, description="Query keywords matching chunk text")
    effective_date: Optional[str] = Field(None, description="Document effective date if specified")

class RAGTriadResult(BaseModel):
    context_relevance: float = Field(..., ge=0.0, le=1.0, description="Retrieved chunk relevance to inquiry")
    groundedness: float = Field(..., ge=0.0, le=1.0, description="Faithfulness of answer to retrieved context")
    answer_relevance: float = Field(..., ge=0.0, le=1.0, description="Direct relevance of answer to user question")
    composite_score: float = Field(..., ge=0.0, le=1.0, description="RAG Triad harmonic score")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Question to answer from indexed documents")
    top_k: int = Field(4, ge=1, le=10, description="Number of context chunks to retrieve")
    document_filter: Optional[List[str]] = Field(None, description="Optional document IDs to restrict search to")
    user_tags: Optional[List[str]] = Field(None, description="Optional user security tags for document ACL filtering")
    dense_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight for Dense vector retrieval (0.0 - 1.0)")
    bm25_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight for BM25 lexical retrieval (0.0 - 1.0)")
    refusal_threshold: Optional[float] = Field(None, ge=0.0, le=10.0, description="Relevance cutoff threshold below which queries are marked out of scope")
    enable_reranker: Optional[bool] = Field(None, description="Enable 2nd-stage cross-encoder reranker")
    stream: Optional[bool] = Field(False, description="Enable real-time token streaming")
    multi_hop: Optional[bool] = Field(False, description="Enable multi-hop sub-query decomposition for complex multi-part questions")
    thread_id: Optional[str] = Field(None, description="Optional thread ID to link inquiry to an ongoing investigation")
    workspace_id: Optional[str] = Field(None, description="Workspace scoping for multi-tenancy")

class SentenceGroundedness(BaseModel):
    sentence: str
    grounded: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    status: str = Field(..., description="directly_grounded, synthesized_bridge, or unsupported")
    supporting_chunk_id: Optional[str] = None
    supporting_citation_index: Optional[int] = None
    matched_phrase: Optional[str] = None

class SubQueryDecomposition(BaseModel):
    sub_query_id: int
    sub_query: str
    retrieved_chunks_count: int
    top_source: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    is_out_of_scope: bool = Field(False, description="True if question cannot be answered from documents")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    groundedness_score: float = Field(..., ge=0.0, le=1.0)
    citations: List[Citation]
    retrieved_chunks_count: int
    processing_time_ms: int
    model_used: str
    sentence_evaluations: Optional[List[SentenceGroundedness]] = Field(default_factory=list)
    sub_queries: Optional[List[SubQueryDecomposition]] = Field(default_factory=list)
    thread_id: Optional[str] = Field(None, description="Active thread ID for this conversation")
    workspace_id: Optional[str] = Field(None, description="Workspace ID")
    query_log_id: Optional[str] = Field(None, description="Query audit log ID")
    prompt_injection_warning: Optional[str] = Field(None, description="Warning if adversarial prompt injection detected")
    rag_triad: Optional[RAGTriadResult] = Field(None, description="RAG Triad evaluation metrics")

class ThreadMessage(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    citations: Optional[List[Citation]] = None
    metadata: Optional[dict] = None
    created_at: str

class ResearchThread(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    citations_count: int = 0
    workspace_id: Optional[str] = None

class ThreadDetail(BaseModel):
    thread: ResearchThread
    messages: List[ThreadMessage]

class ThreadCreateRequest(BaseModel):
    title: Optional[str] = "New Investigation"
    workspace_id: Optional[str] = None

class ThreadUpdateRequest(BaseModel):
    title: str

class ComparisonDimension(BaseModel):
    dimension: str
    doc_a_finding: str
    doc_b_finding: str
    discrepancy_type: str = Field(..., description="aligned, distinct_scope, diverging_terms, or asymmetric")
    notes: str
    doc_a_citation_index: Optional[int] = None
    doc_b_citation_index: Optional[int] = None

class CompareRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Comparison topic or question")
    doc_id_a: str = Field(..., description="ID of baseline document A")
    doc_id_b: str = Field(..., description="ID of comparison document B")
    top_k_per_doc: int = Field(3, ge=1, le=8, description="Chunks to retrieve per document")
    dense_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    bm25_weight: Optional[float] = Field(None, ge=0.0, le=1.0)

class CompareResponse(BaseModel):
    query: str
    doc_a: DocumentMetadata
    doc_b: DocumentMetadata
    doc_a_chunks_count: int
    doc_b_chunks_count: int
    doc_a_citations: List[Citation]
    doc_b_citations: List[Citation]
    dimensions: List[ComparisonDimension]
    executive_synthesis: str
    processing_time_ms: int

class ChunkAnalyticsBucket(BaseModel):
    range_label: str
    count: int

class ChunkOverlapPair(BaseModel):
    chunk_index_a: int
    chunk_index_b: int
    overlap_word_count: int
    overlap_snippet: str

class SectionChunkCount(BaseModel):
    section_title: str
    chunk_count: int
    total_words: int

class ChunkAnalyticsResponse(BaseModel):
    document_id: str
    document_title: str
    total_chunks: int
    total_words: int
    avg_chunk_words: float
    min_chunk_words: int
    max_chunk_words: int
    histogram_buckets: List[ChunkAnalyticsBucket]
    sections_breakdown: List[SectionChunkCount]
    overlap_pairs: List[ChunkOverlapPair]

class EvaluationBenchmarkResult(BaseModel):
    query: str
    expected_doc: str
    expected_fact: str
    actual_answer: str
    citation_found: bool
    grounded: bool
    latency_ms: int
    status: str
    rag_triad: Optional[RAGTriadResult] = None

# Enterprise and Multi-Tenancy Schemas

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9_.-]+\.[a-zA-Z0-9-.]+$")

class UserRegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = "Enterprise User"

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned or not EMAIL_REGEX.match(cleaned):
            raise ValueError("Please provide a valid corporate or personal email address with domain (e.g. user@enterprise.com).")
        return cleaned

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if len(v) > 128:
            raise ValueError("Password cannot exceed 128 characters.")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter (A-Z).")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter (a-z).")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one numeric digit (0-9).")
        if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?/~`"]', v):
            raise ValueError("Password must contain at least one special character or symbol (!@#$%^&*).")
        return v

class UserLoginRequest(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    is_active: bool
    is_superuser: bool
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

class WorkspaceResponse(BaseModel):
    id: str
    organization_id: str
    name: str
    slug: str
    created_at: str
    role: Optional[str] = None

class WorkspaceCreateRequest(BaseModel):
    name: str
    slug: Optional[str] = None

class WorkspaceMemberAdd(BaseModel):
    email: str
    role: str = "member"

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned or not EMAIL_REGEX.match(cleaned):
            raise ValueError("Please provide a valid corporate or personal email address.")
        return cleaned

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        valid_roles = ("admin", "editor", "member", "viewer")
        if v.lower() not in valid_roles:
            raise ValueError(f"Role must be one of: {', '.join(valid_roles)}")
        return v.lower()

class WorkspaceMemberResponse(BaseModel):
    id: str
    user_id: str
    workspace_id: str
    role: str
    email: str
    full_name: str
    created_at: str

class IngestionJobResponse(BaseModel):
    id: str
    workspace_id: Optional[str]
    document_id: str
    status: str
    progress_pct: int
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    document: DocumentMetadata
    job: Optional[IngestionJobResponse] = None
    is_duplicate: bool = False
    message: str = "Document uploaded successfully."

class AuditEventResponse(BaseModel):
    id: str
    workspace_id: Optional[str]
    actor_id: Optional[str]
    actor_email: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    ip_address: Optional[str]
    details: Optional[Dict[str, Any]] = None
    created_at: str

class FeedbackCreateRequest(BaseModel):
    query_log_id: Optional[str] = None
    is_positive: bool
    notes: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: str
    query_log_id: Optional[str]
    is_positive: bool
    notes: Optional[str]
    created_at: str

class EvaluationRunResponse(BaseModel):
    id: str
    workspace_id: Optional[str]
    run_timestamp: str
    total_cases: int
    passed_cases: int
    pass_rate_pct: float
    avg_latency_ms: float
    avg_groundedness: float
    results: List[EvaluationBenchmarkResult]

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
