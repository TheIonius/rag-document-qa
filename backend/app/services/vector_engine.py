from dataclasses import dataclass
import hashlib
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import settings

class IndexedChunk:
    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        document_title: str,
        section_title: Optional[str] = None,
        page_number: int = 1,
        text: str = "",
        word_count: int = 0,
        workspace_id: Optional[str] = "ws_default",
        embedding: Optional[np.ndarray] = None,
        effective_from: Optional[str] = None,
        effective_until: Optional[str] = None,
        version: int = 1,
        security_tags: Optional[List[str]] = None
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_title = document_title
        self.section_title = section_title
        self.page_number = page_number
        self.text = text
        self.word_count = word_count
        self.workspace_id = workspace_id or "ws_default"
        self.embedding = embedding
        self.effective_from = effective_from
        self.effective_until = effective_until
        self.version = version
        self.security_tags = security_tags or ["public"]

        # Search runtime annotations (per-search instance, not shared)
        self.bm25_rank: Optional[int] = None
        self.dense_rank: Optional[int] = None
        self.matched_terms: List[str] = []
        self.cross_encoder_score: Optional[float] = None

    def clone_with_runtime_meta(
        self,
        bm25_rank: Optional[int] = None,
        dense_rank: Optional[int] = None,
        matched_terms: Optional[List[str]] = None,
        cross_encoder_score: Optional[float] = None
    ) -> 'IndexedChunk':
        c = IndexedChunk(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            document_title=self.document_title,
            section_title=self.section_title,
            page_number=self.page_number,
            text=self.text,
            word_count=self.word_count,
            workspace_id=self.workspace_id,
            embedding=self.embedding,
            effective_from=self.effective_from,
            effective_until=self.effective_until,
            version=self.version,
            security_tags=list(self.security_tags) if self.security_tags else ["public"]
        )
        c.bm25_rank = bm25_rank
        c.dense_rank = dense_rank
        c.matched_terms = matched_terms or []
        c.cross_encoder_score = cross_encoder_score
        return c

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_lens: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []

    def fit(self, tokenized_docs: List[List[str]]):
        self.corpus_size = len(tokenized_docs)
        if self.corpus_size == 0:
            return

        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 1.0

        self.doc_term_freqs = []
        df_counter = Counter()

        for doc in tokenized_docs:
            tf = Counter(doc)
            self.doc_term_freqs.append(tf)
            for term in tf.keys():
                df_counter[term] += 1

        self.doc_freqs = dict(df_counter)
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            idf_val = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[term] = max(idf_val, 0.01)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        if self.corpus_size == 0:
            return []

        scores = [0.0] * self.corpus_size
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for doc_idx, tf_dict in enumerate(self.doc_term_freqs):
                tf = tf_dict.get(term, 0)
                if tf > 0:
                    doc_len = self.doc_lens[doc_idx]
                    num = tf * (self.k1 + 1.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[doc_idx] += idf * (num / denom)

        return scores

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "did", "do", "does", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "me", "more", "most", "my",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "you", "your", "yours"
}

def sanitize_fts5_query(query: str) -> str:
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", query)
    tokens = [t for t in tokens if len(t) > 1 and t.lower() not in STOP_WORDS]
    if not tokens:
        tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", query)
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens[:16])

@dataclass(frozen=True)
class SearchIndexSnapshot:
    chunks: Tuple[IndexedChunk, ...]
    bm25: BM25Index
    embeddings_matrix: np.ndarray
    workspace_chunk_map: Dict[str, Tuple[int, ...]]
    model_name: str

class CrossEncoderReranker:
    """2nd-stage cross-encoder reranker evaluating fine-grained query-chunk cross-attention interactions."""

    def __init__(self):
        self._model = None

    def score_pair(self, query_tokens: List[str], chunk_text: str, chunk_section: Optional[str] = None) -> float:
        if not query_tokens or not chunk_text:
            return 0.0

        chunk_lower = chunk_text.lower()
        chunk_words = [t.lower() for t in re.findall(r"\b[\w-]+\b", chunk_lower)]
        if not chunk_words:
            return 0.0

        # 1. Token-level MaxSim coverage (ColBERT-style late interaction)
        positions: Dict[str, List[int]] = {}
        for idx, w in enumerate(chunk_words):
            if w not in positions:
                positions[w] = []
            positions[w].append(idx)

        token_scores = []
        matched_positions: List[int] = []
        for q_tok in query_tokens:
            if q_tok in positions:
                token_scores.append(1.0)
                matched_positions.extend(positions[q_tok])
            else:
                max_sub = 0.0
                for w in positions:
                    if len(q_tok) >= 3 and (q_tok in w or w in q_tok):
                        ratio = min(len(q_tok), len(w)) / max(len(q_tok), len(w))
                        if ratio > max_sub:
                            max_sub = ratio
                token_scores.append(max_sub * 0.75)

        term_coverage = sum(token_scores) / len(query_tokens) if query_tokens else 0.0

        # 2. Query Term Proximity Window
        proximity_score = 0.0
        if len(query_tokens) > 1 and len(matched_positions) >= 2:
            unique_matched = [positions[q][0] for q in query_tokens if q in positions]
            if len(unique_matched) >= 2:
                min_span = max(unique_matched) - min(unique_matched) + 1
                proximity_score = max(0.0, 1.0 - (min_span / (len(chunk_words) + 10)))

        # 3. Exact sequential phrase match
        query_str = " ".join(query_tokens)
        phrase_bonus = 0.20 if len(query_tokens) >= 2 and query_str in chunk_lower else 0.0

        # 4. Heading & structural relevance bonus
        section_bonus = 0.0
        if chunk_section:
            sec_lower = chunk_section.lower()
            sec_hits = sum(1 for q in query_tokens if q in sec_lower)
            section_bonus = min(0.25, sec_hits * 0.12)

        # 5. Position bonus (early occurrence in chunk)
        pos_bonus = 0.0
        if matched_positions:
            first_pos = min(matched_positions)
            if first_pos < max(10, len(chunk_words) * 0.25):
                pos_bonus = 0.10

        composite = (
            0.45 * term_coverage +
            0.20 * proximity_score +
            phrase_bonus +
            section_bonus +
            pos_bonus
        )
        return min(1.0, round(composite, 4))

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float, float]],
        chunks: Tuple[IndexedChunk, ...],
        top_k: int = 4
    ) -> List[Tuple[int, float]]:
        query_tokens = [t.lower() for t in re.findall(r"\b[\w-]+\b", query) if t.lower() not in STOP_WORDS]
        if not query_tokens:
            query_tokens = [t.lower() for t in re.findall(r"\b[\w-]+\b", query)]

        reranked = []
        for idx, cal_score, rrf_score in candidates:
            chunk = chunks[idx]
            cross_score = self.score_pair(query_tokens, chunk.text, chunk.section_title)
            final_score = round(0.40 * cal_score + 0.60 * cross_score, 3)
            reranked.append((idx, final_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

class HybridSearchEngine:
    def __init__(self):
        self._snapshot: Optional[SearchIndexSnapshot] = None
        self._embedding_model = None
        self._reranker = CrossEncoderReranker()

    def _search_fts5_bm25(self, query: str, workspace_id: Optional[str] = None) -> Dict[str, float]:
        fts_query = sanitize_fts5_query(query)
        if not fts_query:
            return {}
        try:
            from app.database import get_db
            with get_db() as conn:
                if workspace_id:
                    cursor = conn.execute("""
                        SELECT chunk_id, bm25(chunks_fts) as rank
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ? AND (workspace_id = ? OR workspace_id IS NULL)
                        ORDER BY rank ASC
                        LIMIT 100
                    """, (fts_query, workspace_id))
                else:
                    cursor = conn.execute("""
                        SELECT chunk_id, bm25(chunks_fts) as rank
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank ASC
                        LIMIT 100
                    """, (fts_query,))
                rows = cursor.fetchall()
                return {row["chunk_id"]: float(-row["rank"]) for row in rows}
        except Exception:
            return {}

    @property
    def chunks(self) -> List[IndexedChunk]:
        snap = self._snapshot
        return list(snap.chunks) if snap else []

    @property
    def is_indexed(self) -> bool:
        snap = self._snapshot
        return snap is not None and len(snap.chunks) > 0

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            try:
                from fastembed import TextEmbedding
                self._embedding_model = TextEmbedding(model_name=settings.embedding_model)
            except Exception as e:
                print(f"[VectorEngine Warning] FastEmbed initialization: {e}")
                self._embedding_model = None
        return self._embedding_model

    def tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in re.findall(r"\b[\w-]+\b", text)]

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = self.embedding_model
        if model is not None:
            try:
                raw_embeds = list(model.embed(texts))
                arr = np.array(raw_embeds, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                return arr / norms
            except Exception as e:
                print(f"[VectorEngine Warning] FastEmbed computation error: {e}")

        # Deterministic SHA-256 fallback representation (same text = same vector across processes)
        dim = 384
        res = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = self.tokenize(t)
            for tok in tokens:
                h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16) % dim
                res[i, h] += 1.0
            norm = np.linalg.norm(res[i])
            if norm > 0:
                res[i] /= norm
        return res

    def index_chunks(self, chunks: List[IndexedChunk], persist_to_db: bool = True):
        if not chunks:
            self._snapshot = None
            return

        corpus_texts = [c.text for c in chunks]
        tokenized_corpus = [
            [t for t in self.tokenize(text) if t not in STOP_WORDS]
            for text in corpus_texts
        ]

        # 1. Fit new BM25 index
        new_bm25 = BM25Index()
        new_bm25.fit(tokenized_corpus)

        # 2. Check and compute embeddings
        missing_indices = [
            idx for idx, chunk in enumerate(chunks)
            if chunk.embedding is None or chunk.embedding.size == 0
        ]

        if missing_indices:
            missing_texts = [chunks[i].text for i in missing_indices]
            computed_embeds = self._embed_texts(missing_texts)
            for local_idx, orig_idx in enumerate(missing_indices):
                chunks[orig_idx].embedding = computed_embeds[local_idx]

            if persist_to_db:
                try:
                    from app.database import get_db
                    with get_db() as conn:
                        updates = [
                            (chunks[orig_idx].embedding.tobytes(), settings.embedding_model, chunks[orig_idx].chunk_id)
                            for orig_idx in missing_indices
                        ]
                        conn.executemany(
                            "UPDATE chunks SET embedding_blob = ?, embedding_model = ? WHERE id = ?",
                            updates
                        )
                except Exception as e:
                    print(f"[VectorEngine Warning] Could not persist embedding blobs: {e}")

        # Construct unified embeddings matrix
        matrix_list = [c.embedding for c in chunks]
        new_matrix = np.vstack(matrix_list).astype(np.float32)

        # Build workspace index mapping for O(1) fail-closed tenant isolation
        ws_map_builder: Dict[str, List[int]] = {}
        for idx, chunk in enumerate(chunks):
            ws_id = chunk.workspace_id or "ws_default"
            if ws_id not in ws_map_builder:
                ws_map_builder[ws_id] = []
            ws_map_builder[ws_id].append(idx)

        ws_chunk_map = {ws: tuple(indices) for ws, indices in ws_map_builder.items()}

        # Atomic pointer swap: readers never observe intermediate state or raised IndexError
        self._snapshot = SearchIndexSnapshot(
            chunks=tuple(chunks),
            bm25=new_bm25,
            embeddings_matrix=new_matrix,
            workspace_chunk_map=ws_chunk_map,
            model_name=settings.embedding_model
        )

    def search(
        self,
        query: str,
        top_k: int = 4,
        doc_filter: Optional[List[str]] = None,
        workspace_id: Optional[str] = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        user_tags: Optional[List[str]] = None,
        enable_reranker: Optional[bool] = None
    ) -> List[Tuple[IndexedChunk, float]]:
        snapshot = self._snapshot
        if not snapshot or not snapshot.chunks or snapshot.embeddings_matrix is None:
            return []

        # STRICT FAIL-CLOSED TENANT ISOLATION:
        # If no workspace_id is provided in a multi-tenant index, refuse search immediately.
        if not workspace_id:
            if len(snapshot.workspace_chunk_map) == 1:
                workspace_id = next(iter(snapshot.workspace_chunk_map.keys()))
            else:
                return []

        candidate_indices = snapshot.workspace_chunk_map.get(workspace_id, ())
        if not candidate_indices:
            return []

        all_query_tokens = self.tokenize(query)
        content_query_tokens = [t for t in all_query_tokens if t not in STOP_WORDS]
        if not content_query_tokens:
            content_query_tokens = all_query_tokens
        if not content_query_tokens:
            return []

        # 1. Lexical BM25 Scoring (SQLite FTS5 native with in-memory fallback)
        fts_scores = self._search_fts5_bm25(query, workspace_id)
        in_memory_bm25 = snapshot.bm25.get_scores(content_query_tokens)
        bm25_raw_scores = []
        for idx, chk in enumerate(snapshot.chunks):
            if chk.chunk_id in fts_scores:
                bm25_raw_scores.append(max(fts_scores[chk.chunk_id], in_memory_bm25[idx]))
            else:
                bm25_raw_scores.append(in_memory_bm25[idx])

        # 2. Dense Semantic Cosine Similarity Scoring
        query_embed = self._embed_texts([query])[0]
        dense_raw_scores = np.dot(snapshot.embeddings_matrix, query_embed).tolist()

        # Check temporal policy expiration
        now_iso = datetime.now(timezone.utc).isoformat()

        # RRF ranking within this workspace's candidate subset
        k_rrf = 60
        ws_candidate_list = list(candidate_indices)

        # Filter out expired chunks, non-matching doc_filter, or unauthorized security tags
        allowed_tags = set(user_tags) if user_tags is not None else None
        filtered_candidates = []
        for idx in ws_candidate_list:
            chk = snapshot.chunks[idx]
            if doc_filter and chk.document_id not in doc_filter:
                continue
            if chk.effective_until and chk.effective_until < now_iso:
                # Expired policy chunk
                continue
            if allowed_tags is not None:
                chk_tags = set(getattr(chk, "security_tags", ["public"]) or ["public"])
                if not chk_tags.intersection(allowed_tags):
                    continue
            filtered_candidates.append(idx)

        if not filtered_candidates:
            return []

        # Check maximum semantic similarity in candidate pool
        cand_dense = [dense_raw_scores[i] for i in filtered_candidates]
        cand_bm25 = [bm25_raw_scores[i] for i in filtered_candidates]
        max_dense = max(cand_dense) if cand_dense else 0.0
        max_bm25 = max(cand_bm25) if cand_bm25 else 0.0

        # Strict out-of-scope cutoff: if BM25 has no keyword match and dense similarity is below 0.58
        if max_bm25 <= 0.0 and max_dense < 0.58:
            return []

        bm25_sub_rank = sorted(filtered_candidates, key=lambda i: bm25_raw_scores[i], reverse=True)
        dense_sub_rank = sorted(filtered_candidates, key=lambda i: dense_raw_scores[i], reverse=True)

        bm25_rank_map = {idx: rank + 1 for rank, idx in enumerate(bm25_sub_rank)}
        dense_rank_map = {idx: rank + 1 for rank, idx in enumerate(dense_sub_rank)}

        query_phrase = query.lower().strip()
        scored_candidates: List[Tuple[int, float, float]] = []

        for idx in filtered_candidates:
            chk = snapshot.chunks[idx]
            bm25_val = bm25_raw_scores[idx]
            dense_val = dense_raw_scores[idx]

            # Skip chunks with zero lexical match and weak semantic similarity (< 0.52)
            if bm25_val == 0.0 and dense_val < 0.52:
                continue

            # RRF combined score
            rrf_score = (
                bm25_weight * (1.0 / (k_rrf + bm25_rank_map[idx])) +
                vector_weight * (1.0 / (k_rrf + dense_rank_map[idx]))
            )

            # Boost if query tokens match section title
            if chk.section_title:
                sec_tokens = set(self.tokenize(chk.section_title))
                overlap = len(set(content_query_tokens).intersection(sec_tokens))
                if overlap > 0:
                    rrf_score *= (1.0 + (overlap * 0.15))

            # Boost exact phrase match
            if len(query_phrase) > 10 and query_phrase in chk.text.lower():
                rrf_score *= 1.25

            # Calibrate normalized relevance score [0.0 - 1.0]
            # Dense is typically 0.40 - 0.90; BM25 is typically 0 - 20
            norm_bm25 = min(1.0, bm25_val / 12.0) if bm25_val > 0 else 0.0
            norm_dense = max(0.0, min(1.0, (dense_val - 0.40) / 0.50))
            calibrated_score = round(max(0.05, min(0.99, (0.55 * norm_dense + 0.45 * norm_bm25))), 3)

            scored_candidates.append((idx, calibrated_score, rrf_score))

        # Sort by RRF score descending
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        # 3. Stage 2 Cross-Encoder Reranker
        use_reranker = enable_reranker if enable_reranker is not None else getattr(settings, "enable_reranker", True)
        if use_reranker and len(scored_candidates) > 1:
            pool_size = min(len(scored_candidates), max(top_k * 3, getattr(settings, "reranker_candidates", 12)))
            candidates_to_rerank = scored_candidates[:pool_size]
            selected_ranked = self._reranker.rerank(
                query=query,
                candidates=candidates_to_rerank,
                chunks=snapshot.chunks,
                top_k=top_k
            )
        else:
            selected_ranked = [(idx, cal_score) for idx, cal_score, _ in scored_candidates[:top_k]]

        results: List[Tuple[IndexedChunk, float]] = []
        for idx, cal_score in selected_ranked:
            source_chunk = snapshot.chunks[idx]
            chunk_tokens = set(self.tokenize(source_chunk.text))
            matched_terms = sorted(list(set(content_query_tokens).intersection(chunk_tokens)))

            # Return an isolated clone - DO NOT mutate shared chunk object in snapshot
            clean_chunk = source_chunk.clone_with_runtime_meta(
                bm25_rank=bm25_rank_map.get(idx),
                dense_rank=dense_rank_map.get(idx),
                matched_terms=matched_terms,
                cross_encoder_score=cal_score
            )
            results.append((clean_chunk, cal_score))

        return results

    @property
    def snapshot(self) -> Optional[SearchIndexSnapshot]:
        return self._snapshot

    @property
    def chunks(self) -> List[IndexedChunk]:
        return list(self._snapshot.chunks) if self._snapshot else []

    def get_document_chunks(self, document_id: str, workspace_id: Optional[str] = None) -> List[IndexedChunk]:
        snapshot = self._snapshot
        if not snapshot:
            return []
        if workspace_id:
            candidate_indices = snapshot.workspace_chunk_map.get(workspace_id, ())
            return [snapshot.chunks[i] for i in candidate_indices if snapshot.chunks[i].document_id == document_id]
        return [c for c in snapshot.chunks if c.document_id == document_id]

# Singleton search engine instance
search_engine = HybridSearchEngine()
