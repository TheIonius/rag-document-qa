from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db, get_db
from app.routers import documents, query, evaluation, threads, auth, workspaces, audit
from app.routers.documents import refresh_index_from_db
from app.security.ratelimit import RateLimitMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if count == 0:
            print("[Lifespan] Document store is empty. Auto-ingesting samples from data/documents...")
            from scripts.ingest_samples import ingest_all_documents
            ingest_all_documents()
        else:
            refresh_index_from_db()
    yield

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Enterprise RAG Document Intelligence system with verified hybrid retrieval, RBAC multi-tenancy, and audit governance.",
    lifespan=lifespan
)

# Rate limiting
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_per_minute,
    window_seconds=60
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.exceptions import RequestValidationError

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": msg,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": msg
            }
        },
        headers=exc.headers
    )

@app.exception_handler(RequestValidationError)
async def custom_validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err.get("loc", []) if loc != "body")
        msg = err.get("msg", "Invalid value")
        errors.append(f"{field}: {msg}" if field else msg)
    detail_str = "; ".join(errors) if errors else "Invalid request data."
    return JSONResponse(
        status_code=422,
        content={
            "detail": detail_str,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": detail_str
            }
        }
    )

from fastapi import APIRouter

api_router = APIRouter(prefix="/api")
api_v1_router = APIRouter(prefix="/api/v1")

all_subrouters = [
    documents.router,
    query.router,
    evaluation.router,
    threads.router,
    auth.router,
    workspaces.router,
    audit.router
]

for r in all_subrouters:
    api_router.include_router(r)
    api_v1_router.include_router(r)

app.include_router(api_router)
app.include_router(api_v1_router)

frontend_dir = settings.base_dir / "frontend"
static_dir = frontend_dir / "static"
static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.version,
        "embedding_model": settings.embedding_model,
        "features": {
            "hybrid_retrieval": True,
            "prompt_injection_defense": settings.enable_prompt_injection_defense,
            "rbac_multi_tenancy": True,
            "audit_ledger": True,
            "persistent_vector_store": True
        }
    }

@app.api_route("/", methods=["GET", "HEAD"])
def serve_index():
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Enterprise RAG backend active."}
