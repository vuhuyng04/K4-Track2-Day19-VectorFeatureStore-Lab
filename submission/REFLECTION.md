# Reflection — Lab 19

**Tên:** Nguyễn Vũ Huy
**Cohort:** IV
**Path đã chạy:** docker (Qdrant server + Redis + Postgres, embedding bge-small; bge-m3 để so sánh)

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Precision@10 với bge-small: hybrid 78.6% > BM25 77.8% > vector 73.2%.
- **exact:** BM25 hoà hybrid (96.7%), vì từ khoá nằm nguyên văn trong doc.
- **mixed:** hybrid thắng tuyệt đối (100%), vì hai retriever sai ở những doc khác nhau và RRF bù cho nhau.
- **paraphrase:** cả hai đều yếu (BM25 33%, vector 24%). bge-small học tiếng Anh nên không hiểu cách diễn đạt lại tiếng Việt.

Đổi sang **bge-m3**, vector đạt 95.2% và **vượt hybrid (89.0%)**: ở paraphrase vector 86.7% còn hybrid 66.7%. RRF cho hai retriever trọng số bằng nhau, nên BM25 kém kéo cả kết quả xuống.

Không nên dùng hybrid khi:
- một retriever yếu hẳn trên phân bố query thật, ví dụ vector đa ngữ mạnh với người dùng hay diễn đạt lại (khi đó dùng vector thuần, hoặc RRF có trọng số);
- tra cứu mã lỗi, SKU, tên hàm, nơi BM25 thuần vừa chính xác vừa rẻ (P99 3ms so với 21ms của hybrid).

Hybrid là mặc định an toàn, không phải lúc nào cũng tối ưu. Phải đo trên query của chính mình.

---

## Điều ngạc nhiên nhất khi làm lab này

Nút thắt latency không nằm ở model. `localhost` trên Windows tốn khoảng 5 s mỗi kết nối, còn REST/JSON chậm gấp 5 lần gRPC. Chỉ sửa hạ tầng mà hybrid P99 giảm từ 340ms xuống 24ms.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: (làm một mình)
