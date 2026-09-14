# KẾ HOẠCH RÚT GỌN — HOÀN THÀNH TRONG HÔM NAY + NGÀY MAI

## Đánh giá thực tế trước tiên (đọc kỹ trước khi bắt đầu)

Với deadline 2 ngày, **bắt buộc phải rút gọn phạm vi** ở 2 chỗ tốn thời gian nhất, nếu không cả nhóm
sẽ bị kẹt:

1. **Crawl GitHub API (Nhánh B / RQ3)**: giảm từ 500-1000 user xuống còn **200-300 user**
   (đủ để tính tương quan có ý nghĩa thống kê, nhưng giảm ~70% thời gian chờ crawl).
2. **Train GCN**: giảm EPOCHS từ 200 xuống **80-100**, dùng embedding dim nhỏ hơn (16 thay vì 32).
   Với đồ thị 37.700 node, 80-100 epoch vẫn đủ hội tụ để có kết quả so sánh hợp lý.

Đã chỉnh sẵn 2 thông số này trong code — chỉ cần mở file sửa 2 biến số:
- `src/step2_crawl_github_supplement.py` → `MAX_USERS = 300` (dòng ~24)
- `src/step5_gcn_clustering.py` → `EPOCHS = 100`, `EMBED_DIM = 16` (dòng ~20-21)

## Phân công 3 người chạy SONG SONG ngay từ giờ đầu tiên

| Người | Phụ trách | File chạy | Không phụ thuộc ai |
|---|---|---|---|
| Leader | Setup chung + Preprocess + RQ1 | `run_main_branch.sh` (phần 1-3) | Bắt đầu ngay |
| Thành viên A | Kiểm tra/tinh chỉnh RQ1 + viết phần kết luận RQ1 | Dựa trên output của Leader ở bước 3 | Chờ Leader xong bước preprocess (~15-30 phút) |
| Thành viên B | Crawl GitHub + RQ3 | `run_branch_b.sh` | Bắt đầu ngay, KHÔNG cần chờ ai |

---

## HÔM NAY

### Giờ 0 — Setup (cả 3 người, làm song song, ~30 phút)
- Cả 3 cùng clone/tải project, cài `pip install -r requirements.txt`.
- Thành viên B: tạo GitHub Personal Access Token ngay lập tức (xem hướng dẫn đã gửi trước đó),
  tạo file `.env` từ `.env.example`.

### Giờ 0:30 — Bắt đầu song song
**Leader:**
```bash
bash run_main_branch.sh   # chạy step1 (download) + step3 (preprocess) trước
```
Ước tính: download ~5-10 phút (tùy mạng), preprocess (SVD) ~5-10 phút → xong trong ~30 phút.
Sau khi xong step3, **gửi ngay file `data/processed/` cho Thành viên B** (qua Drive/Zalo) —
để B không cần tự chạy lại bước tải/preprocess, tiết kiệm thời gian máy B.

**Thành viên B:** chạy song song ngay, không chờ Leader:
```bash
bash run_branch_b.sh      # bắt đầu crawl ngay (đây là bước LÂU NHẤT toàn dự án)
```
Ước tính với MAX_USERS=300: khoảng **2-4 giờ** (phụ thuộc rate-limit + tốc độ mạng).
→ Đây là lý do bước này phải bắt đầu SỚM NHẤT trong ngày, chạy nền trong lúc làm việc khác.

### Giờ 1 — Leader tiếp tục, Thành viên A bắt đầu
**Leader** (đã có `data/processed/` từ Giờ 0:30):
```bash
python src/step4_homophily_community.py    # RQ1 — Louvain + Leiden + homophily
```
Ước tính: ~10-20 phút (Leiden trên 37.700 node khá nhanh, Louvain cũng vậy).

**Thành viên A:** trong lúc chờ Leader chạy xong step4, đọc trước code `step4_homophily_community.py`
và phần giải thích RQ1 (edge homophily, tỷ lệ nhãn theo cụm) để khi có kết quả là hiểu ngay,
viết được phần diễn giải luôn — tránh mất thời gian đọc hiểu sau khi đã có số liệu.

### Giờ 1:30 — 4:00 (buổi chiều/tối hôm nay)
**Leader:**
```bash
python src/step5_gcn_clustering.py    # RQ2 phần GCN — với EPOCHS=100 ước tính 20-40 phút trên CPU
python src/step6_evaluate.py          # RQ2 — bảng so sánh, chạy nhanh (<1 phút)
```
Xong bước này, Leader có đủ kết quả RQ1 + RQ2.

**Thành viên A:** dựa vào output `outputs/leiden_cluster_distribution.csv` và
`outputs/rq1_homophily_summary.txt` (đã có từ Giờ 1), viết trước phần "Kết luận RQ1" theo template
đã có trong README, kèm giải thích ý nghĩa.

**Thành viên B:** crawl vẫn đang chạy nền — trong lúc chờ, đọc trước code `step7_centrality_correlation.py`
để hiểu logic Nhánh A (centrality nội tại) và Nhánh B (centrality vs star/fork thật).

### Cuối hôm nay (check-in nhóm, 15-30 phút)
Cả 3 họp nhanh (voice call/nhắn tin):
- Leader báo cáo: đã xong RQ1 + RQ2, gửi file `outputs/` cho cả nhóm.
- Thành viên B báo cáo: crawl xong chưa (nếu network chậm, có thể phải chạy qua đêm — script đã có
  `time.sleep(0.2)` lịch sự với API và tự retry khi rate-limit, an toàn để chạy qua đêm không cần canh).
- Nếu B chưa xong crawl, để chạy tiếp qua đêm (không cần tắt máy), sáng mai chạy tiếp `step7`.

---

## NGÀY MAI

### Sáng — Thành viên B hoàn tất RQ3
```bash
python src/step7_centrality_correlation.py   # nếu crawl đã xong từ đêm qua
```
Nếu crawl chưa đủ 300 user do rate-limit, vẫn có thể chạy step7 với số lượng đã crawl được
(kể cả 150-200 user vẫn đủ ý nghĩa thống kê cho tương quan Spearman) — không cần đợi đủ 300.

### Trưa — Cả 3 ghép kết quả
**Leader** chạy bước tổng hợp cuối cùng (sau khi đã có đủ output từ cả 2 nhánh):
```bash
python src/step8_visualize_report.py
```
File `outputs/SUMMARY_REPORT.md` và các hình trong `outputs/figures/` sẽ tự động được tạo.

### Chiều — Viết báo cáo (chia việc theo đúng RQ đã phụ trách)
- Thành viên A: hoàn thiện phần RQ1 (đã viết nháp từ hôm qua), thêm hình `rq1_cluster_label_distribution.png`.
- Leader: viết phần RQ2 (dựa vào `rq2_comparison_table.csv` + `rq2_method_comparison.png`),
  viết phần Phương pháp luận chung, phần Giới thiệu.
- Thành viên B: viết phần RQ3, **nhớ nêu rõ giới hạn** (mẫu crawl độc lập, không trùng 37.700 node gốc,
  thiên lệch mẫu vì seed ≥500 follower — đã có sẵn trong README).

### Tối — Leader tổng hợp, rà soát, hoàn thiện
- Ghép 3 phần lại, thống nhất văn phong.
- Kiểm tra lại: mỗi RQ trong phần Mở đầu đều có câu trả lời tương ứng trong Kết luận.
- Đọc lại phần Limitation (giới hạn của RQ3 về dữ liệu ẩn danh, giới hạn mẫu crawl) — đây là phần
  giảng viên/hội đồng thường hỏi trước tiên, cần chuẩn bị sẵn câu trả lời khi phản biện.

---

## Rủi ro cần lường trước

| Rủi ro | Cách xử lý |
|---|---|
| Crawl bị rate-limit kéo dài, không đủ 300 user | Giảm xuống MAX_USERS=150-200, hoặc dùng 2-3 token của 3 người crawl song song rồi gộp data |
| Train GCN trên máy yếu quá chậm | Dùng Google Colab (miễn phí, có GPU) — chỉ cần upload folder `data/processed/` lên |
| Leiden/Louvain lỗi thư viện (hay gặp trên Windows do `python-igraph` khó cài) | Cài qua `conda install -c conda-forge python-igraph leidenalg` thay vì pip nếu dùng Windows |
| Không đủ thời gian làm cả 2 nhánh RQ3 | Ưu tiên Nhánh B (dữ liệu thật) vì đây mới là kết quả chính; Nhánh A (centrality nội tại) có thể làm tối giản hoặc bỏ nếu quá gấp |
