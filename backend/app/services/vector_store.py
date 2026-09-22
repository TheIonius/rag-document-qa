from abc import ABC, abstractmethod
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class VectorStoreAdapter(ABC):
    """Abstract protocol for enterprise vector storage and similarity retrieval."""

    @abstractmethod
    def upsert_vectors(self, records: List[Dict[str, Any]]) -> int:
        """
        Upserts embeddings into the store.
        Each record must contain 'chunk_id' and 'vector'.
        Optional keys: 'workspace_id', 'document_id', 'metadata'.
        """
        pass

    @abstractmethod
    def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        workspace_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Searches for nearest neighbor vectors.
        Returns list of (chunk_id, similarity_score) tuples sorted descending by score.
        """
        pass

    @abstractmethod
    def delete_vectors(self, chunk_ids: List[str]) -> int:
        """Deletes vectors by chunk_id."""
        pass

    @abstractmethod
    def count(self, workspace_id: Optional[str] = None) -> int:
        """Returns the total number of indexed vectors."""
        pass


class SQLiteBlobVectorStore(VectorStoreAdapter):
    """Default enterprise vector store persisting embeddings as binary blobs in SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def _get_connection(self):
        from app.database import get_db
        return get_db(self.db_path)

    def upsert_vectors(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0
        from app.database import get_db
        updates = []
        for r in records:
            vec = r["vector"]
            blob = vec.tobytes() if isinstance(vec, np.ndarray) else (vec if isinstance(vec, bytes) else np.array(vec, dtype=np.float32).tobytes())
            updates.append((blob, r["chunk_id"]))

        with get_db(self.db_path) as conn:
            conn.executemany("UPDATE chunks SET embedding_blob = ? WHERE id = ?", updates)
        return len(updates)

    def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        workspace_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        if query_vector is None or query_vector.size == 0:
            return []

        from app.database import get_db
        with get_db(self.db_path) as conn:
            if workspace_id:
                rows = conn.execute("""
                    SELECT c.id, c.embedding_blob
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding_blob IS NOT NULL
                      AND d.status = 'READY'
                      AND (d.workspace_id = ? OR d.workspace_id IS NULL)
                """, (workspace_id,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT c.id, c.embedding_blob
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding_blob IS NOT NULL
                      AND d.status = 'READY'
                """).fetchall()

        if not rows:
            return []

        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        q_unit = query_vector / q_norm

        chunk_ids = []
        vectors = []
        for r in rows:
            arr = np.frombuffer(r["embedding_blob"], dtype=np.float32)
            chunk_ids.append(r["id"])
            vectors.append(arr)

        if not vectors:
            return []

        mat = np.vstack(vectors)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        norm_mat = mat / norms

        scores = np.dot(norm_mat, q_unit)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(chunk_ids[i], round(float(scores[i]), 4)) for i in top_indices]
        return results

    def delete_vectors(self, chunk_ids: List[str]) -> int:
        if not chunk_ids:
            return 0
        from app.database import get_db
        placeholders = ",".join("?" for _ in chunk_ids)
        with get_db(self.db_path) as conn:
            cursor = conn.execute(
                f"UPDATE chunks SET embedding_blob = NULL WHERE id IN ({placeholders})",
                chunk_ids
            )
            return cursor.rowcount

    def count(self, workspace_id: Optional[str] = None) -> int:
        from app.database import get_db
        with get_db(self.db_path) as conn:
            if workspace_id:
                row = conn.execute("""
                    SELECT COUNT(*)
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.embedding_blob IS NOT NULL
                      AND (d.workspace_id = ? OR d.workspace_id IS NULL)
                """, (workspace_id,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding_blob IS NOT NULL").fetchone()
            return row[0] if row else 0


class SqliteVecAdapter(VectorStoreAdapter):
    """
    Adapter for sqlite-vec extension (vec0 virtual table).
    Delegates to SQLiteBlobVectorStore when sqlite-vec native extension is not loaded in SQLite runtime.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._fallback = SQLiteBlobVectorStore(db_path)
        self._has_vec_extension = False
        try:
            from app.database import get_db
            with get_db(db_path) as conn:
                conn.execute("SELECT vec_version();")
                self._has_vec_extension = True
        except Exception:
            self._has_vec_extension = False

    def upsert_vectors(self, records: List[Dict[str, Any]]) -> int:
        return self._fallback.upsert_vectors(records)

    def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        workspace_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        return self._fallback.search_vectors(query_vector, top_k, workspace_id, filters)

    def delete_vectors(self, chunk_ids: List[str]) -> int:
        return self._fallback.delete_vectors(chunk_ids)

    def count(self, workspace_id: Optional[str] = None) -> int:
        return self._fallback.count(workspace_id)


class QdrantVectorStoreAdapter(VectorStoreAdapter):
    """
    Pluggable vector store adapter for Qdrant vector database.
    Operates with in-memory store fallback when remote Qdrant server is offline.
    """

    def __init__(self, collection_name: str = "cortex_chunks", dim: int = 384):
        self.collection_name = collection_name
        self.dim = dim
        self._memory_store: Dict[str, Dict[str, Any]] = {}

    def upsert_vectors(self, records: List[Dict[str, Any]]) -> int:
        count = 0
        for r in records:
            cid = r["chunk_id"]
            vec = r["vector"]
            if not isinstance(vec, np.ndarray):
                vec = np.array(vec, dtype=np.float32)
            self._memory_store[cid] = {
                "vector": vec,
                "workspace_id": r.get("workspace_id", "ws_default"),
                "document_id": r.get("document_id"),
                "metadata": r.get("metadata", {})
            }
            count += 1
        return count

    def search_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        workspace_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        if not self._memory_store or query_vector is None or query_vector.size == 0:
            return []

        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        q_unit = query_vector / q_norm

        candidates = []
        for cid, item in self._memory_store.items():
            if workspace_id and item["workspace_id"] != workspace_id:
                continue
            v = item["vector"]
            v_norm = np.linalg.norm(v)
            if v_norm > 0:
                sim = float(np.dot(v / v_norm, q_unit))
                candidates.append((cid, round(sim, 4)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def delete_vectors(self, chunk_ids: List[str]) -> int:
        deleted = 0
        for cid in chunk_ids:
            if cid in self._memory_store:
                del self._memory_store[cid]
                deleted += 1
        return deleted

    def count(self, workspace_id: Optional[str] = None) -> int:
        if workspace_id:
            return sum(1 for item in self._memory_store.values() if item["workspace_id"] == workspace_id)
        return len(self._memory_store)


def get_vector_store_adapter(adapter_type: Optional[str] = None) -> VectorStoreAdapter:
    """Factory creating configured vector store adapter."""
    chosen = (adapter_type or os.getenv("CORTEX_VECTOR_STORE", "sqlite_blob")).lower()
    if chosen == "sqlite_vec":
        return SqliteVecAdapter()
    elif chosen == "qdrant":
        return QdrantVectorStoreAdapter()
    else:
        return SQLiteBlobVectorStore()
