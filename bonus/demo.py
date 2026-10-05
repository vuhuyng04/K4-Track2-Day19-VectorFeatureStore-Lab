"""5-query demo for HybridMemoryAgent (BONUS-CHALLENGE.md §3).

Run:  python bonus/demo.py      (Docker stack up + NB4 materialized for real
Feast features; otherwise the agent falls back to in-memory Qdrant / default
profile and says so in the output).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import HybridMemoryAgent  # noqa: E402

MEMORIES_U001 = [
    "Hôm nay mình đọc tài liệu Kubernetes về Horizontal Pod Autoscaler. HPA tăng số pod "
    "khi CPU vượt ngưỡng 70%. Cần đặt resource requests thì HPA mới tính được.",
    "Ghi chú: cluster autoscaler thêm node khi pod bị pending. Kết hợp với HPA để tự động "
    "mở rộng hạ tầng theo lưu lượng người dùng giờ cao điểm.",
    "Đọc bài về cloud security: nguyên tắc least privilege cho IAM role, bật MFA cho tài "
    "khoản root, mã hoá dữ liệu at rest bằng KMS. Không commit secret lên git.",
    "Security checklist cho S3 bucket: chặn public access, bật versioning, log truy cập "
    "vào một bucket riêng. Rà soát security group mở cổng 0.0.0.0/0.",
    "Mình đang học RAG: hybrid search kết hợp BM25 và vector bằng RRF k=60. Hybrid thắng "
    "trên query hỗn hợp nhưng BM25 vẫn tốt cho mã lỗi chính xác.",
    "Note nhanh: deploy app len k8s bang helm chart, rollback bang helm rollback neu "
    "readiness probe fail.",   # typed without diacritics on purpose
]
MEMORIES_U002 = [
    "Bí mật của u_002: kế hoạch migrate database Postgres sang Aurora vào quý 4.",
]

QUERIES = [
    ("chỉ vector hit", "Tôi đã đọc gì về Kubernetes?"),
    ("cần profile", "Recommend đọc gì tiếp"),
    ("cần fresh activity", "Tôi đang quan tâm gì gần đây?"),
    ("paraphrase", "Tài liệu về tự động mở rộng hạ tầng?"),
    ("mixed: hybrid + profile", "Cho tôi summary cloud security"),
]


def main() -> int:
    agent = HybridMemoryAgent()
    for uid in ("u_001", "u_002"):          # idempotent re-runs: start from a clean slate
        agent.forget_user(uid)
    for m in MEMORIES_U001:
        agent.remember(m, user_id="u_001")
    for m in MEMORIES_U002:
        agent.remember(m, user_id="u_002")

    for i, (kind, q) in enumerate(QUERIES, 1):
        print(f"\n=== Q{i} ({kind}): {q}")
        print(agent.recall(q, user_id="u_001"))

    # Isolation check: u_001 must never see u_002's memory, even on a direct hit.
    leaked = any("u_002" in m for m in agent.search("kế hoạch migrate Postgres Aurora", "u_001"))
    print(f"\n=== isolation: u_001 sees u_002 memory? {'YES (BUG)' if leaked else 'no'}")
    return 1 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
