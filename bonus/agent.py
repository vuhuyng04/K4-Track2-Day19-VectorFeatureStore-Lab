"""HybridMemoryAgent — episodic memory (Qdrant) + stable profile (Feast).

POC for BONUS-CHALLENGE.md. Design rationale lives in bonus/ARCHITECTURE.md;
this file only demonstrates the decisions:

  remember(text)  chunk (sentence groups, 1-sentence overlap) -> embed ->
                  upsert into ONE shared collection with a `user_id` payload
  recall(query)   Feast online features (profile + recent activity)
                  + hybrid search over this user's memories only
                    (BM25 with accent folding  +  vector  ->  RRF k=60)
                  -> assembled context string (no LLM call)
"""
from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
import uuid
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct, VectorParams,
)
from rank_bm25 import BM25Okapi  # noqa: E402

from app.embeddings import Embedder  # noqa: E402  -- importing `app` loads .env

COLLECTION = "bonus_memory"
RRF_K = 60
FEATURES = [
    "user_profile_features:topic_affinity",
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]
# Used when Feast/Redis is unreachable so the demo still runs (and says so).
DEFAULT_PROFILE = {"topic_affinity": "unknown", "reading_speed_wpm": 200,
                   "preferred_language": "vi", "queries_last_hour": 0, "distinct_topics_24h": 0}


def fold_accents(text: str) -> str:
    """'tự động mở rộng' -> 'tu dong mo rong' (Vietnamese users often type without diacritics)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    # Whitespace split + accent-folded copy: robust to missing diacritics and to
    # vi/en code-switching ("deploy app lên k8s") without a VN word segmenter.
    words = re.findall(r"\w+", text.lower())
    return words + [fold_accents(w) for w in words]


def chunk(text: str, max_words: int = 60) -> list[str]:
    """Group sentences up to ~max_words, carrying the last sentence over as overlap."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks, current = [], []
    for s in sentences:
        if current and sum(len(x.split()) for x in current) + len(s.split()) > max_words:
            chunks.append(" ".join(current))
            current = current[-1:]          # 1-sentence overlap keeps context across the cut
        current.append(s)
    if current:
        chunks.append(" ".join(current))
    return chunks


class HybridMemoryAgent:
    def __init__(self, feast_repo: Path = ROOT / "app" / "feast_repo") -> None:
        self.embedder = Embedder()
        self.client = self._connect_qdrant()
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION, vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE))
            # Indexed payload field: every read is filtered by user_id (tenant isolation).
            self.client.create_payload_index(COLLECTION, "user_id", PayloadSchemaType.KEYWORD)
        self.store = self._connect_feast(feast_repo)
        self.recent: dict[str, deque[str]] = {}   # in-process stand-in for a streaming push source

    # ── wiring ──────────────────────────────────────────────────────────
    @staticmethod
    def _connect_qdrant() -> QdrantClient:
        if os.getenv("QDRANT_MODE", "memory") == "server":
            try:
                client = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), prefer_grpc=True)
                client.get_collections()
                return client
            except Exception as exc:  # noqa: BLE001
                print(f"[agent] Qdrant server unreachable ({type(exc).__name__}) -> in-memory")
        return QdrantClient(":memory:")

    @staticmethod
    def _connect_feast(repo: Path):
        try:
            from feast import FeatureStore
            return FeatureStore(repo_path=str(repo))
        except Exception as exc:  # noqa: BLE001
            print(f"[agent] Feast unavailable ({type(exc).__name__}) -> default profile")
            return None

    def _user_filter(self, user_id: str) -> Filter:
        return Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

    # ── public API ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        pieces = chunk(text)
        vectors = list(self.embedder.embed(pieces))
        self.client.upsert(COLLECTION, points=[
            PointStruct(id=str(uuid.uuid4()), vector=v.tolist(),
                        payload={"user_id": user_id, "text": p, "ts": time.time()})
            for p, v in zip(pieces, vectors)
        ])

    def profile(self, user_id: str) -> tuple[dict, str]:
        if self.store is not None:
            try:
                row = self.store.get_online_features(FEATURES, entity_rows=[{"user_id": user_id}]).to_dict()
                feats = {k: v[0] for k, v in row.items() if k != "user_id"}
                if any(v is not None for v in feats.values()):
                    return {k: (DEFAULT_PROFILE[k] if v is None else v) for k, v in feats.items()}, "feast"
            except Exception as exc:  # noqa: BLE001
                print(f"[agent] online lookup failed ({type(exc).__name__}) -> default profile")
        return dict(DEFAULT_PROFILE), "default"

    def search(self, query: str, user_id: str, top_k: int = 3) -> list[str]:
        """Hybrid (BM25 + vector, RRF) over this user's memories only."""
        flt = self._user_filter(user_id)
        mine, _ = self.client.scroll(COLLECTION, scroll_filter=flt, limit=10_000, with_payload=True)
        if not mine:
            return []
        texts = {str(p.id): p.payload["text"] for p in mine}
        ids = list(texts)
        bm25 = BM25Okapi([tokenize(texts[i]) for i in ids])
        scores = bm25.get_scores(tokenize(query))
        kw_rank = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i]) if scores[i] > 0]
        q_vec = next(self.embedder.embed([query])).tolist()
        sem = self.client.query_points(COLLECTION, query=q_vec, query_filter=flt, limit=20).points
        sem_rank = [str(p.id) for p in sem]
        rrf: dict[str, float] = {}
        for ranking in (kw_rank, sem_rank):
            for rank, pid in enumerate(ranking, start=1):        # rank is 1-based
                rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (RRF_K + rank)
        return [texts[pid] for pid, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]]

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features -> assembled context."""
        feats, source = self.profile(user_id)
        recent = self.recent.setdefault(user_id, deque(maxlen=5))
        memories = self.search(query, user_id)
        recent_txt = "; ".join(recent) if recent else "(none this session)"
        recent.append(query)
        lines = [
            f"[profile via {source}] User likes '{feats['topic_affinity']}', reads at "
            f"{feats['reading_speed_wpm']} wpm, prefers '{feats['preferred_language']}'.",
            f"[activity] {feats['queries_last_hour']} queries in the last hour, "
            f"{feats['distinct_topics_24h']} distinct topics in 24h. Recent: {recent_txt}",
            "[memories]" if memories else "[memories] (nothing stored for this user)",
        ]
        lines += [f"  {i}. {m}" for i, m in enumerate(memories, 1)]
        return "\n".join(lines)

    def forget_user(self, user_id: str) -> None:
        """Right-to-erasure (Decree 13/2023): drop every memory of one user."""
        self.client.delete(COLLECTION, points_selector=self._user_filter(user_id))
