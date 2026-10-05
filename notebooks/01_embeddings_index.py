# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB1 — Embeddings & Vector Indexing
#
# **Stack:** `fastembed` (ONNX, CPU) + Qdrant in-memory.
# Maps to slide §1 (Embeddings) + §2 (Vector DB Landscape) + deliverable bullet 1.
#
# > Mục tiêu: hiểu cách 1 đoạn text được biến thành vector dày, và cách Qdrant
# > index + query vectors đó. Không cần GPU, không cần Docker.

# %%
import _setup  # noqa: F401  -- adds repo root to sys.path
import json
from pathlib import Path

import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.embeddings import Embedder  # importing `app` also loads .env

DATA = Path(_setup.__file__).resolve().parent.parent / "data"

# %% [markdown]
# ## 1. Load corpus
#
# Corpus được sinh bởi `scripts/seed_corpus.py` (chạy trong `make seed`):
# 1000 docs tiếng Việt, 10 chủ đề × 100 docs/chủ đề. Mỗi doc có `doc_id`,
# `topic`, `title`, `text`.

# %%
docs = []
with (DATA / "corpus_vn.jsonl").open(encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

print(f"Corpus size: {len(docs)} docs")
print(f"First doc:")
print(json.dumps(docs[0], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 2. Embedding model: `BAAI/bge-small-en-v1.5`
#
# `fastembed` chạy ONNX → CPU friendly, không cần GPU. 384-dim vectors.
# Model được chọn qua `EMBEDDING_BACKEND` trong `.env` (`app/embeddings.py`);
# path Docker này giữ `fastembed` vì máy không có GPU — xem `.env`.
#
# > Trong production tiếng Việt 2026, bạn nên dùng `bge-m3` hoặc
# > `text-embedding-3-large` (xem deck §1, bảng *Embedding Models 2026*).
# > Cho lab này dùng `bge-small-en` để mọi laptop chạy được nhanh.

# %%
embedder = Embedder()
print(f"Backend: {embedder.backend} -> {embedder.model_name} ({embedder.dim}d)")
sample = list(embedder.embed(["cloud computing tiếng Việt"]))[0]
print(f"Vector dim: {len(sample)}")
print(f"First 8 values: {sample[:8].tolist()}")

# %% [markdown]
# ## 3. Index vào Qdrant (server mode — Docker path)
#
# `QDRANT_MODE=server` (trong `.env`) → kết nối Qdrant server trong Docker
# (`docker compose up -d`, dashboard http://localhost:6333/dashboard).
# `QDRANT_MODE=memory` → Qdrant in-process, không cần Docker. Cùng một API,
# chỉ khác cách khởi tạo client. Số chiều lấy từ `embedder.dim`, không hard-code.

# %%
QDRANT_MODE = os.getenv("QDRANT_MODE", "memory")
if QDRANT_MODE == "server":
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
else:
    client = QdrantClient(":memory:")

# Server collections persist across runs — drop and rebuild for a clean index.
if client.collection_exists("lab19"):
    client.delete_collection("lab19")
client.create_collection(
    collection_name="lab19",
    vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
)
print(f"Qdrant mode: {QDRANT_MODE}  |  collection lab19 ({embedder.dim}d) ready")

# %% [markdown]
# ## 4. TODO — embed + upsert toàn bộ corpus
#
# Embed `title + " " + text` cho từng doc, batch theo 64 docs/lần (fastembed
# CPU-bound, batch=64 là sweet spot). Upsert vào Qdrant collection `lab19`.
#
# **Hint:** xem `app/search.py` `_build_vector_index()` để tham khảo pattern.

# %%
# TODO: implement the embed + upsert loop here.
# Expected outcome: client.count("lab19") == 1000
# (~30 seconds on first run as fastembed downloads the model.)

BATCH = 64
points: list[PointStruct] = []
for start in range(0, len(docs), BATCH):
    batch = docs[start:start + BATCH]
    texts = [d["title"] + " " + d["text"] for d in batch]
    vectors = list(embedder.embed(texts))
    for i, (d, v) in enumerate(zip(batch, vectors)):
        points.append(PointStruct(
            id=start + i,
            vector=v.tolist(),
            payload={"doc_id": d["doc_id"], "topic": d["topic"], "title": d["title"]},
        ))

client.upsert(collection_name="lab19", points=points)
n_indexed = client.count(collection_name="lab19").count
print(f"Indexed: {n_indexed} vectors")
assert n_indexed == 1000, f"expected 1000 indexed, got {n_indexed}"

# %% [markdown]
# ## 5. First similarity search
#
# Top-5 docs gần nhất với câu query. Chú ý: cùng 1 query có thể trả về docs
# từ nhiều topic — đó là dấu hiệu vector embedding tổng quát. Để filter theo
# topic, dùng Qdrant payload filter.

# %%
query = "cloud computing và tự động mở rộng"
q_vec = next(embedder.embed([query])).tolist()
hits = client.query_points(collection_name="lab19", query=q_vec, limit=5).points

print(f"Query: {query!r}")
print(f"Top-5:")
for i, h in enumerate(hits, 1):
    print(f"  {i}. [{h.payload['topic']:>9}] score={h.score:.3f}  {h.payload['title']}")

# %% [markdown]
# ## 6. Quick sanity — top-5 should be mostly `cloud` topic
#
# Vector embedding cluster theo semantic, không cần keyword "cloud" xuất hiện
# trong doc. Chạy lại với query 100% paraphrase, không có chữ "cloud":

# %%
query2 = "phương pháp tự động mở rộng hạ tầng theo lưu lượng người dùng"
q_vec2 = next(embedder.embed([query2])).tolist()
hits2 = client.query_points(collection_name="lab19", query=q_vec2, limit=5).points

print(f"Query (paraphrase): {query2!r}")
for h in hits2:
    print(f"  [{h.payload['topic']:>9}] score={h.score:.3f}  {h.payload['title']}")

# %% [markdown]
# ## Deliverable evidence (chụp màn hình)
#
# 1. Output cell 4: `Indexed: 1000 vectors`
# 2. Output cell 5: top-5 results với scores
# 3. Output cell 6: paraphrase query vẫn tìm đúng cluster `cloud`
#
# ---
#
# ## Vibe-coding callout
#
# **Delegate freely:** the `for start in range(0, ..., BATCH)` upsert loop —
# pattern is mechanical, AI generates it perfectly. Just give it the spec
# (batch size, payload schema) and review the diff.
#
# **Think hard yourself:** the choice of `BAAI/bge-small-en-v1.5`. Is it
# right for tiếng Việt? (Hint: xem deck §1 bảng *Embedding Models 2026* —
# `bge-m3` hỗ trợ multilingual tốt hơn nhưng nặng 4× hơn.) **Don't ask AI to
# pick the embedding model without first telling it: language(s), corpus
# size, latency budget, and re-index cost.** Đây là 1 quyết định kiến trúc,
# không phải boilerplate.
