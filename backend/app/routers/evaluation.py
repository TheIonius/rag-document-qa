import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Header, Query

from app.database import get_db, record_audit_event, utc_now_iso
from app.models.schemas import (
    EvaluationBenchmarkResult,
    EvaluationRunResponse,
    QueryRequest
)
from app.routers.query import execute_rag_query

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

BENCHMARK_CASES = [
    {
        "query": "What is the 401(k) company matching formula and vesting schedule?",
        "expected_doc": "Employee Benefits & Workplace Policies Handbook",
        "expected_fact": "100%",
        "allow_out_of_scope": False
    },
    {
        "query": "What are the MFA requirements and prohibited authentication methods?",
        "expected_doc": "Enterprise Cloud Security & Compliance Policy",
        "expected_fact": "prohibited",
        "allow_out_of_scope": False
    },
    {
        "query": "What is the 5-attempt webhook retry backoff schedule and Dead Letter Queue policy?",
        "expected_doc": "Enterprise REST API Architecture & Developer Specification",
        "expected_fact": "5 retry attempts",
        "allow_out_of_scope": False
    },
    {
        "query": "How many weeks of paid parental leave are offered to caregivers?",
        "expected_doc": "Employee Benefits & Workplace Policies Handbook",
        "expected_fact": "16 consecutive weeks",
        "allow_out_of_scope": False
    },
    {
        "query": "What ingredients are required to bake a classic sourdough bread?",
        "expected_doc": "None",
        "expected_fact": "cannot find information",
        "allow_out_of_scope": True
    },
    {
        "query": "What are the designated outdoor smoking zones and tobacco rules?",
        "expected_doc": "None",
        "expected_fact": "cannot find information",
        "allow_out_of_scope": True
    },
    {
        "query": "What are the planetary atmospheric conditions and orbital periods of Jupiter and Saturn?",
        "expected_doc": "None",
        "expected_fact": "cannot find information",
        "allow_out_of_scope": True
    }
]

@router.get("/synthetic")
def get_synthetic_qa_cases(
    workspace_id: Optional[str] = Query(None),
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id")
):
    """Retrieves automatically generated synthetic Q&A verification pairs for active workspace."""
    active_ws = workspace_id or x_workspace_id or "ws_default"
    from app.services.synthetic_qa import get_synthetic_benchmarks
    return get_synthetic_benchmarks(active_ws)

@router.get("/benchmark", response_model=List[EvaluationBenchmarkResult])
async def run_evaluation_benchmark(
    workspace_id: Optional[str] = Query(None),
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id")
):
    active_ws = workspace_id or x_workspace_id or "ws_default"
    results: List[EvaluationBenchmarkResult] = []
    total_latency = 0
    total_groundedness = 0.0

    all_cases = list(BENCHMARK_CASES)
    try:
        from app.services.synthetic_qa import get_synthetic_benchmarks
        synth_cases = get_synthetic_benchmarks(active_ws)
        all_cases.extend(synth_cases)
    except Exception as e:
        print(f"[Evaluation Warning] Could not load synthetic benchmarks: {e}")

    for test_case in all_cases:
        req = QueryRequest(query=test_case["query"], top_k=4, workspace_id=active_ws)
        res = await execute_rag_query(req, persist_thread=False)

        answer_lower = res.answer.lower()
        fact_lower = test_case["expected_fact"].lower()

        if test_case["allow_out_of_scope"]:
            passed = res.is_out_of_scope or "cannot find" in answer_lower
            status = "PASSED (Correctly rejected out-of-scope)" if passed else "FAILED (Hallucinated answer)"
        else:
            fact_found = fact_lower in answer_lower
            has_citations = len(res.citations) > 0
            passed = fact_found and has_citations
            status = "PASSED (Fact + Citation Verified)" if passed else ("PARTIAL" if fact_found else "FAILED")

        res_item = EvaluationBenchmarkResult(
            query=test_case["query"],
            expected_doc=test_case["expected_doc"],
            expected_fact=test_case["expected_fact"],
            actual_answer=res.answer[:180] + ("..." if len(res.answer) > 180 else ""),
            citation_found=len(res.citations) > 0,
            grounded=res.groundedness_score >= 0.70,
            latency_ms=res.processing_time_ms,
            status=status,
            rag_triad=res.rag_triad
        )
        results.append(res_item)
        total_latency += res.processing_time_ms
        total_groundedness += res.groundedness_score

    # Persist benchmark run to database
    run_id = f"eval_{uuid.uuid4().hex[:12]}"
    now = utc_now_iso()
    passed_cases = sum(1 for r in results if "PASSED" in r.status)
    avg_lat = round(total_latency / len(results), 1) if results else 0.0
    avg_gnd = round(total_groundedness / len(results), 2) if results else 0.0
    results_json = json.dumps([r.model_dump() for r in results])

    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO evaluation_runs (
                    id, workspace_id, run_timestamp, total_cases, passed_cases,
                    avg_latency_ms, avg_groundedness, results_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, active_ws, now, len(results), passed_cases, avg_lat, avg_gnd, results_json))
    except Exception as e:
        print(f"[Evaluation Warning] Could not save evaluation run: {e}")

    record_audit_event(
        action="run_evaluation_benchmark",
        resource_type="evaluation_run",
        resource_id=run_id,
        workspace_id=active_ws,
        details={"total_cases": len(results), "passed_cases": passed_cases, "avg_latency_ms": avg_lat}
    )

    return results

@router.get("/history", response_model=List[EvaluationRunResponse])
def get_evaluation_history(
    limit: int = Query(10, ge=1, le=50),
    workspace_id: Optional[str] = Query(None),
    x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id")
):
    active_ws = workspace_id or x_workspace_id or "ws_default"
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, workspace_id, run_timestamp, total_cases, passed_cases,
                   avg_latency_ms, avg_groundedness, results_json
            FROM evaluation_runs
            WHERE workspace_id = ? OR workspace_id IS NULL
            ORDER BY run_timestamp DESC
            LIMIT ?
        """, (active_ws, limit)).fetchall()

    history = []
    for r in rows:
        results_data = []
        if r["results_json"]:
            try:
                results_data = [EvaluationBenchmarkResult(**item) for item in json.loads(r["results_json"])]
            except Exception:
                results_data = []

        total = r["total_cases"] or 1
        passed = r["passed_cases"] or 0
        rate = round((passed / total) * 100.0, 1)

        history.append(EvaluationRunResponse(
            id=r["id"],
            workspace_id=r["workspace_id"],
            run_timestamp=r["run_timestamp"],
            total_cases=total,
            passed_cases=passed,
            pass_rate_pct=rate,
            avg_latency_ms=r["avg_latency_ms"],
            avg_groundedness=r["avg_groundedness"],
            results=results_data
        ))

    return history
