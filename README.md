# Phân cụm cộng đồng lập trình viên trên mạng lưới GitHub

Pipeline Python cho đề tài phân tích mạng xã hội trên dataset **MUSAE GitHub Social Network**
(37.700 node, 289.003 cạnh, nhãn web developer / ML developer).

Dự án trả lời ba câu hỏi nghiên cứu:

| RQ | Câu hỏi | Script trả lời |
|---|---|---|
| RQ1 | Web dev và ML dev có xu hướng kết nối đồng loại không? | `step03_rq1_homophily.py` |
| RQ2 | Các phương pháp phân cụm tạo ra cấu trúc gì, và khớp nhãn nghề nghiệp đến đâu? | `step04`, `step05`, `step06` |
| RQ3 | Vị trí trung tâm trong mạng (2019) có liên hệ với star/fork quan sát tại thời điểm thu thập không? | `step07`, `step08`, `step09` |

## Dữ liệu

Tải thủ công từ [SNAP](https://snap.stanford.edu/data/github-social.html), giải nén 3 file vào `data/git_web_ml/`:

```
data/git_web_ml/musae_git_edges.csv       # 289.003 cạnh
data/git_web_ml/musae_git_target.csv      # id, name, ml_target
data/git_web_ml/musae_git_features.json   # 4.005 chiều đặc trưng
```

Chạy `python src/step01_check_data.py` để kiểm tra dữ liệu đã đặt đúng chỗ chưa.

Repository **không chứa** dữ liệu gốc và các kết quả trung gian dung lượng lớn
(`data/`, `outputs/` — xem `.gitignore`). Riêng các bảng LaTeX và hình PDF dùng cho
báo cáo thì được commit trong `report/`, nên có thể đọc kết quả mà không cần chạy lại pipeline.

### Lưu ý về định danh và quyền riêng tư

Cột `name` trong `musae_git_target.csv` là **username GitHub thật**, không phải ID ẩn danh.
`step08` dùng chính cột này để gọi GitHub API cho RQ3.

Username tồn tại trong tệp dữ liệu gốc và được nạp tạm vào bộ nhớ khi gọi API, nhưng
**không script nào trong pipeline ghi username ra đĩa**. Cụ thể:

- `data/processed/labels.csv` — `step02` ghi tường minh chỉ hai cột `id`, `ml_target`
- `outputs/results/rq3_centrality.csv` — chỉ có `id` và các độ đo
- `data/external/github_stars_sample.csv` — chỉ có `id`, `stratum`, `degree` và các số liệu
  công khai (`followers`, `following`, `public_repos`, `stars`, `forks`, `created_at`, `fetched_at`)

`step01` có đọc cột `name`, nhưng chỉ để đếm số username trùng lặp và ghi ra một số nguyên.

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Cài `torch` và `torch-geometric` đúng bản khớp CUDA/CPU của máy bạn — xem
[hướng dẫn chính thức](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Để chạy `step08`, tạo file `.env` ở thư mục gốc dự án (mẫu: `.env.example`):

```
GITHUB_TOKEN=<personal access token của bạn>
```

Token chỉ cần quyền đọc public. Giới hạn 5.000 request/giờ; script tự chờ khi hết lượt
và lưu tiến độ theo từng lô nên có thể dừng giữa chừng rồi chạy tiếp.

## Thứ tự chạy

```bash
python src/step01_check_data.py            # kiểm tra dữ liệu gốc (chỉ đọc)
python src/step02_preprocess.py            # dựng graph, ma trận thưa, SVD-128
python src/step03_rq1_homophily.py         # RQ1: homophily + kiểm định hoán vị
python src/step04_rq2_structure_methods.py # RQ2: Louvain, Leiden, sweep γ, Spectral
python src/step05_rq2_embedding_methods.py # RQ2: SVD, featprop, DeepWalk, GAE
python src/step06_rq2_evaluate.py          # RQ2: NMI/ARI/Purity/Modularity + linear probe
python src/step07_rq3_centrality.py        # RQ3: centrality toàn cục và trong cộng đồng
python src/step08_rq3_fetch_stars.py       # RQ3: star/fork qua GitHub API (cần token)
python src/step09_rq3_correlation.py       # RQ3: Spearman, bootstrap, hồi quy nhị thức âm
python src/step10_figures.py               # hình PDF cho báo cáo

python src/export_gephi.py                 # tuỳ chọn: xuất .gexf cho Gephi
```

Bảng LaTeX xuất ra `report/tables/`, hình PDF xuất ra `report/figures/`.
Kết quả trung gian (graph, embedding, phân hoạch, CSV) nằm trong `data/processed/` và `outputs/`.

Hai bước chạy lâu: **Spectral Clustering** (15–40 phút, đặt `RUN_SPECTRAL = False`
trong `config.py` để bỏ qua) và **GAE** (300 epoch). `step08` mất vài giờ do giới hạn API.

Mọi tham số nằm tập trung trong `src/config.py` để phần Phương pháp của báo cáo luôn khớp
với tham số thật sự đã dùng. Các phương pháp có yếu tố ngẫu nhiên chạy lặp với 5 seed
(`SEEDS = [0, 1, 2, 3, 4]`), kết quả báo cáo dưới dạng trung bình ± độ lệch chuẩn.

## Thiết kế từng phần

### RQ1 — Homophily

Hai tuyến bằng chứng độc lập.

Ở **mức cạnh**: edge homophily so với mức nền ngẫu nhiên `p_web² + p_ml²`; adjusted homophily
(Platonov et al., 2023) hiệu chỉnh cho mất cân bằng nhãn — với nhãn nhị phân chỉ số này trùng
với hệ số đồng loại của Newman; ma trận trộn web–web / web–ML / ML–ML; kiểm định hoán vị
1.000 lần xáo nhãn để lấy p-value.

Ở **mức cộng đồng**: chạy Leiden trên cấu trúc mạng **không dùng nhãn**, rồi so tỷ lệ ML trong
từng cụm với tỷ lệ toàn mạng bằng chi-bình phương và Cramér's V, lặp qua nhiều seed.

### RQ2 — So sánh phương pháp phân cụm

| Nhóm | Phương pháp |
|---|---|
| Chỉ cấu trúc | Louvain (γ=1), Leiden (γ=1), Leiden (γ chọn qua sweep), Spectral, DeepWalk + KMeans |
| Chỉ đặc trưng | KMeans trên SVD-128 |
| Cấu trúc + đặc trưng | Feature propagation + KMeans, Graph Autoencoder + KMeans |

DeepWalk thuộc nhóm "chỉ cấu trúc" vì random walk chỉ dùng thông tin cạnh, không dùng
ma trận đặc trưng node.

Resolution của Leiden được chọn theo **hai cách tách biệt**, và đây là điểm cần đọc kỹ:

- `leiden_selected` (γ = 0.7): chọn **không dùng nhãn** — trong các γ có modularity ≥ 90%
  mức cao nhất, lấy γ cho kết quả ổn định nhất giữa các seed. Đây là cấu hình dùng để
  **so sánh công bằng** với các phương pháp khác.
- `leiden_oracle` (γ = 0.5): chọn **theo ARI cao nhất**, tức có dùng nhãn. Chỉ là cận trên
  tham khảo, **không** dùng để kết luận phương pháp nào tốt hơn.

Bảng "gộp theo nhãn đa số" (`rq2_majority_merge.tex`) cũng dùng nhãn trong bước gộp, nên chỉ
để giải thích *vì sao* ARI gốc thấp, không dùng để xếp hạng phương pháp.

`step06` chạy thêm **linear probe**: Logistic Regression 5-fold trên từng biểu diễn, đo xem
embedding *chứa* bao nhiêu thông tin về nhãn, độc lập với việc KMeans có khai thác được hay
không. Đây là phân tích **có giám sát, phục vụ chẩn đoán** — không xếp chung bảng với các
phương pháp phân cụm không giám sát.

### RQ3 — Centrality và mức ảnh hưởng

`step07` tính centrality **toàn cục** (degree, PageRank, eigenvector, betweenness xấp xỉ,
k-core) và **theo cộng đồng** (bậc trong cụm, tỷ lệ kết nối nội bộ, PageRank trong cụm đã
chuẩn hoá theo kích thước cụm, participation coefficient).

`step08` lấy mẫu **2.200 người** từ chính dataset MUSAE, theo hai nhóm tách biệt:

- **Mẫu chính (2.000 người).** Mạng được chia thành 5 tầng bằng nhau theo bậc
  (`pd.qcut` trên rank, mỗi tầng đúng 7.540 node), rồi lấy 400 người từ mỗi tầng. Vì các
  tầng bằng nhau theo thiết kế, tỷ lệ chọn là như nhau (~5,3%) giữa các tầng — đây là mẫu
  tỷ lệ, đại diện cho toàn mạng trừ nhóm top bên dưới.
- **Nhóm top (200 người).** Toàn bộ 200 người có bậc cao nhất, được tách ra khỏi khung lấy
  mẫu trước khi bốc mẫu chính (`rest = df.drop(top.index)`) và phân tích riêng.

`step09` giữ nguyên sự tách biệt này: Spearman, Spearman riêng phần và hồi quy nhị thức âm
chỉ chạy trên **mẫu chính**; nhóm top 200 chỉ dùng cho bảng `rq3_within_top`.

Với mỗi người, API trả về follower/following hiện tại, số repo công khai, tổng star và fork
trên các repo họ tự tạo (bỏ repo fork), và tuổi tài khoản.

Phân tích gồm: tỷ lệ tài khoản còn tồn tại theo từng tầng; Spearman kèm KTC 95% bootstrap;
Spearman riêng phần kiểm soát nhãn nghề nghiệp, số repo và tuổi tài khoản; hồi quy nhị thức âm
cho số star, so sánh các độ đo bằng AIC; và tỷ lệ trùng top-K giữa danh sách theo centrality
và danh sách theo star/follower.

## Giới hạn cần nêu trong báo cáo

**Lệch thời gian.** Mạng MUSAE là ảnh chụp tháng 6/2019, còn star/fork/follower được lấy qua
API ngày 17/09/2026 — cách nhau khoảng 7 năm 3 tháng (thời điểm chính xác của từng bản ghi
lưu ở cột `fetched_at`). Centrality mô tả vị trí năm 2019, còn mức ảnh hưởng đo ở hiện tại.
Khoảng 5–9% tài khoản trong mẫu đã không còn truy cập được; chi tiết theo tầng ở
`report/tables/rq3_coverage.tex`.

**Tương quan, không phải nhân quả.** Kết quả RQ3 cho thấy centrality đi kèm mức ảnh hưởng cao
hơn, kể cả sau khi kiểm soát nghề nghiệp, số repo và tuổi tài khoản. Nhưng thiết kế quan sát
này không chứng minh vị trí trung tâm *gây ra* nhiều star/fork; quan hệ nhiều khả năng hai chiều.

**Phạm vi của phân tích influencer.** Bảng `rq3_influencer_overlap` so sánh top 50 theo
centrality với top 50 theo star/follower **trong phạm vi mẫu 2.200 người**, không phải trong
toàn bộ 37.700 node. Các con số trùng khớp vì vậy đo khả năng sàng lọc trong mẫu nghiên cứu.

**Các độ đo centrality tương quan rất chặt với nhau** (degree–k-core: 0,99; degree–PageRank:
0,97), nên không nên diễn giải chúng như những yếu tố độc lập với nhau.

**Một số phương pháp không tái tạo được phân chia Web/ML.** KMeans trên SVD, feature
propagation, DeepWalk và GAE đều cho purity đúng bằng mức nền 0,7417, tức cả hai cụm dự đoán
đều có đa số là web developer. Điều này không nhất thiết là thuật toán sai — phân hoạch tự
nhiên trong không gian biểu diễn có thể đơn giản là không tương ứng với hai nhãn nghề nghiệp
(DeepWalk chẳng hạn vẫn đạt modularity 0,275, tức tìm được cấu trúc thật). Riêng Spectral
Clustering thì khác: modularity ≈ 0 và sau khi gộp theo nhãn đa số chỉ còn ~0,01% node được
gán ML, tức phép chia gần như suy biến thành một cụm khổng lồ và một cụm vài node.

**Nhãn không phải ground truth cho cộng đồng.** Web/ML là nhãn nghề nghiệp, không phải nhãn
cộng đồng. ARI thấp nghĩa là cấu trúc cộng đồng không trùng với nghề nghiệp — không phải
bằng chứng thuật toán sai.

## Công nghệ

`NetworkX`, `python-igraph`, `leidenalg` (graph và phát hiện cộng đồng) · `scikit-learn`
(SVD, KMeans, đánh giá, linear probe) · `gensim` (DeepWalk) · `PyTorch` + `PyTorch Geometric`
(Graph Autoencoder) · `SciPy`, `statsmodels` (kiểm định, bootstrap, hồi quy nhị thức âm) ·
`pandas`, `matplotlib`, `seaborn` · GitHub REST API.
