import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from app.database import get_db, utc_now_iso

def extract_prominent_facts_from_chunk(text: str, section_title: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Extracts candidate factual query and expected fact pairs from a chunk text.
    Returns list of (query, expected_fact).
    """
    candidates: List[Tuple[str, str]] = []
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    
    clean_sec = section_title if section_title and not section_title.startswith("Page ") else "the document"

    for sentence in sentences:
        # 1. Statements with numeric quantities or percentages
        num_match = re.search(r'\b(\d+(?:\.\d+)?\s*(?:%|percent|minutes|hours|days|weeks|months|years|USD|dollars)?)\b', sentence, re.IGNORECASE)
        if num_match:
            fact = num_match.group(0).strip()
            # Formulate question based on sentence content
            words = sentence.split()
            query_core = " ".join(words[:12]).rstrip(",.:;")
            candidates.append((
                f"What is the specified quantity or timeframe regarding {clean_sec} ({query_core})?",
                fact
            ))
            continue

        # 2. Statements with requirements or prohibitions
        req_match = re.search(r'\b(must|required|prohibited|mandatory|shall)\s+([^,.;]{5,35})', sentence, re.IGNORECASE)
        if req_match:
            fact = req_match.group(0).strip()
            candidates.append((
                f"What is the requirement or policy regarding {clean_sec}?",
                fact
            ))
            continue

        # 3. Definitions or structured statements ("is defined as", "refers to", "applies to")
        def_match = re.search(r'\b(?:is defined as|refers to|applies to|consists of)\s+([^,.;]{5,35})', sentence, re.IGNORECASE)
        if def_match:
            fact = def_match.group(0).strip()
            candidates.append((
                f"How is the policy or scope defined in {clean_sec}?",
                fact
            ))
            continue

    return candidates

def generate_synthetic_qa_pairs(
    document_id: str,
    document_title: str,
    chunks: List[Any],
    workspace_id: Optional[str] = "ws_default"
) -> List[Dict[str, Any]]:
    """
    Generates 3 synthetic question and verification pairs for a newly ingested document
    and persists them to the synthetic_benchmarks table.
    """
    if not chunks:
        return []

    generated_cases: List[Dict[str, Any]] = []
    seen_facts = set()

    # Iterate across chunks to extract varied pairs
    for chunk in chunks:
        chunk_text = getattr(chunk, "text", "")
        sec_title = getattr(chunk, "section_title", None)
        extracted = extract_prominent_facts_from_chunk(chunk_text, sec_title)
        
        for q, fact in extracted:
            if fact.lower() not in seen_facts:
                seen_facts.add(fact.lower())
                generated_cases.append({
                    "id": f"syn_{uuid.uuid4().hex[:12]}",
                    "workspace_id": workspace_id or "ws_default",
                    "document_id": document_id,
                    "query": q,
                    "expected_doc": document_title,
                    "expected_fact": fact,
                    "allow_out_of_scope": False
                })
                if len(generated_cases) >= 3:
                    break
        if len(generated_cases) >= 3:
            break

    # If fewer than 3 generated from patterns, synthesize fallback factual queries from available chunks
    if len(generated_cases) < 3:
        for idx, chunk in enumerate(chunks):
            if len(generated_cases) >= 3:
                break
            chunk_text = getattr(chunk, "text", "")
            words = chunk_text.split()
            if len(words) >= 6:
                fact_snippet = " ".join(words[2:6]).strip(".,;:")
                sec = getattr(chunk, "section_title", None) or f"Section {idx+1}"
                q = f"According to {document_title}, what details are outlined in {sec}?"
                if fact_snippet.lower() not in seen_facts:
                    seen_facts.add(fact_snippet.lower())
                    generated_cases.append({
                        "id": f"syn_{uuid.uuid4().hex[:12]}",
                        "workspace_id": workspace_id or "ws_default",
                        "document_id": document_id,
                        "query": q,
                        "expected_doc": document_title,
                        "expected_fact": fact_snippet,
                        "allow_out_of_scope": False
                    })

    # Persist to database
    if generated_cases:
        now = utc_now_iso()
        try:
            with get_db() as conn:
                records = [
                    (
                        c["id"], c["workspace_id"], c["document_id"],
                        c["query"], c["expected_doc"], c["expected_fact"], now
                    )
                    for c in generated_cases
                ]
                conn.executemany("""
                    INSERT OR REPLACE INTO synthetic_benchmarks (
                        id, workspace_id, document_id, query, expected_doc, expected_fact, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, records)
        except Exception as e:
            print(f"[Synthetic QA Warning] Could not persist synthetic benchmarks: {e}")

    return generated_cases

def get_synthetic_benchmarks(workspace_id: Optional[str] = "ws_default") -> List[Dict[str, Any]]:
    """Retrieves all active synthetic benchmark cases for the workspace."""
    active_ws = workspace_id or "ws_default"
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT id, workspace_id, document_id, query, expected_doc, expected_fact
                FROM synthetic_benchmarks
                WHERE workspace_id = ? OR workspace_id IS NULL
                ORDER BY created_at DESC
            """, (active_ws,)).fetchall()
            return [
                {
                    "id": r["id"],
                    "workspace_id": r["workspace_id"],
                    "document_id": r["document_id"],
                    "query": r["query"],
                    "expected_doc": r["expected_doc"],
                    "expected_fact": r["expected_fact"],
                    "allow_out_of_scope": False
                }
                for r in rows
            ]
    except Exception as e:
        print(f"[Synthetic QA Warning] Failed to load synthetic benchmarks: {e}")
        return []
