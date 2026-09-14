# Phân cụm cộng đồng lập trình viên trên mạng lưới cộng tác GitHub

Pipeline Python đầy đủ cho đề tài: Graph Clustering / Attributed Graph Clustering trên dataset
**MUSAE GitHub Social Network** (37.700 node, 289.003 cạnh, nhãn web/ML developer).

## Sơ đồ pipeline

```
[1] Download MUSAE data ─┐
                          ├─► [3] Preprocess (build graph, SVD feature) ─► [4] Homophily + Community Detection (Louvain/Leiden)  ──┐
[2] Crawl GitHub bổ sung ─┘                                              └─► [5] GCN Clustering (GAE/VGAE + KMeans) ──────────────┼─► [6] Evaluate (NMI/ARI/Purity/Modularity)
    (cho RQ3, real stars/forks)                                                                                                    │
                                                                          [7] Centrality + Correlation với star/fork ◄──────────────┘
                                                                                          │
                                                                                          ▼
                                                                          [8] Visualize + Tổng hợp báo cáo
```

## Ánh xạ Research Question ↔ Script

| RQ | Câu hỏi | Script trả lời |
|---|---|---|
| RQ1 | Web dev và ML dev có tách cụm riêng biệt không? | `step4_homophily_community.py` |
| RQ2 | Louvain/Leiden vs GCN, phương pháp nào khớp nhãn gốc tốt hơn? | `step4` + `step5` + `step6_evaluate.py` |
| RQ3 | Centrality có tương quan với star/fork không? | `step7_centrality_correlation.py` (cần `step2` crawl bổ sung) |

## Cài đặt môi trường

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Lưu ý cài `torch` và `torch-geometric` đúng bản khớp CUDA/CPU của máy bạn
(xem hướng dẫn chính thức: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Thứ tự chạy

```bash
python src/step1_download_musae.py
python src/step2_crawl_github_supplement.py      # cần GitHub token, xem RQ3 caveat bên dưới
python src/step3_preprocess.py
python src/step4_homophily_community.py
python src/step5_gcn_clustering.py
python src/step6_evaluate.py
python src/step7_centrality_correlation.py
python src/step8_visualize_report.py
```

Kết quả (số liệu, hình, bảng) được lưu vào thư mục `outputs/`.

## ⚠️ Lưu ý quan trọng về RQ3 (bắt buộc đọc trước khi làm)

Dataset MUSAE **ẩn danh hóa hoàn toàn** — mỗi lập trình viên chỉ có một ID số (0, 1, 2, ..., 37699),
KHÔNG có username GitHub thật. Vì vậy **không thể** tra ngược để lấy số star/fork thật của 37.700
người này qua GitHub API.

Hướng xử lý được cài sẵn trong pipeline (2 lựa chọn, có thể làm cả 2 để đối chiếu):

1. **Proxy nội tại (không cần crawl thêm)**: dùng chính cấu trúc mạng đã có —
   in-degree (số follower trong mạng) làm proxy cho "được nhiều người chú ý",
   để kiểm tra correlation giữa các loại centrality khác nhau (degree vs betweenness vs eigenvector)
   — không trả lời được "star/fork" thật, chỉ trả lời "trong nội bộ mạng, ai trung tâm hơn thì
   có được follow nhiều hơn không" (nghe có vẻ hiển nhiên nhưng vẫn là kết quả cần kiểm chứng).

2. **Crawl bổ sung độc lập (khuyến nghị, trả lời đúng RQ3 như đề bài)**: `step2_crawl_github_supplement.py`
   tự thu thập một **mẫu mới, độc lập** (khoảng 500–1000 user thật, có username thật) qua GitHub REST API,
   lấy đúng dữ liệu: follower/following (dựng lại mini-network), và star/fork thật trên các repo họ sở hữu.
   Đây là "case study bổ sung" — không dùng chung ID với 37.700 node MUSAE, nhưng dùng để chứng minh
   giả thuyết trên dữ liệu thật, có kiểm chứng được.

Trong báo cáo/luận văn, bạn nên trình bày rõ ràng phần Limitation này — đây là điểm cộng, không phải
điểm trừ, vì thể hiện bạn hiểu rõ dữ liệu chứ không áp dụng máy móc.

## Khung viết kết luận (Conclusion Template)

Sau khi chạy xong toàn bộ pipeline, điền số liệu thực tế thay cho `[...]` :

- **RQ1**: Edge homophily thực tế = `[...]`% so với baseline ngẫu nhiên = `[...]`%
  → (Có/Không) bằng chứng cho thấy web dev và ML dev hình thành cụm tách biệt.
  Community detection tìm được `[...]` cụm, trong đó `[...]`/`[...]` cụm có tỷ lệ lệch >70% khỏi baseline.

- **RQ2**: Louvain đạt NMI=`[...]`, ARI=`[...]`; Leiden đạt NMI=`[...]`, ARI=`[...]`;
  GCN (GAE/VGAE+KMeans) đạt NMI=`[...]`, ARI=`[...]`
  → Phương pháp `[...]` cho kết quả khớp nhãn gốc tốt nhất, gợi ý rằng
  `[cấu trúc mạng thuần túy / kết hợp thuộc tính cá nhân]` là yếu tố quyết định hơn trong việc
  phân biệt cộng đồng web vs ML.

- **RQ3**: Hệ số tương quan Spearman giữa centrality và star/fork = `[...]` (p-value = `[...]`)
  → (Có/Không) tương quan có ý nghĩa thống kê → centrality (degree/eigenvector/betweenness)
  `[có thể / không thể]` dùng làm chỉ báo nhanh để phát hiện "người ảnh hưởng" trong cộng đồng
  mã nguồn mở mà không cần chờ số liệu star/fork thật.
