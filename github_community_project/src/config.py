"""Cấu hình chung cho toàn bộ dự án: đường dẫn và tham số.

Mọi script đều import từ đây, để phần Phương pháp trong báo cáo
luôn khớp với tham số thật sự đã dùng.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ===== Đường dẫn =====
RAW_DIR = ROOT / "data" / "git_web_ml"      # dữ liệu gốc, CHỈ ĐỌC
PROC_DIR = ROOT / "data" / "processed"      # sinh bởi step02
EXT_DIR = ROOT / "data" / "external"        # star/fork lấy qua GitHub API (RQ3)

OUT_DIR = ROOT / "outputs"
RES_DIR = OUT_DIR / "results"               # bảng kết quả dạng CSV
PART_DIR = OUT_DIR / "partitions"           # kết quả phân cụm (mảng seed x node)
EMB_DIR = OUT_DIR / "embeddings"            # embedding đã học
LOG_DIR = OUT_DIR / "logs"

REPORT_DIR = ROOT / "report"
TABLE_DIR = REPORT_DIR / "tables"           # bảng LaTeX sinh tự động
FIG_DIR = REPORT_DIR / "figures"
ENV_FILE = ROOT / ".env"

# File dữ liệu gốc
EDGES_FILE = RAW_DIR / "musae_git_edges.csv"
TARGET_FILE = RAW_DIR / "musae_git_target.csv"
FEATURES_FILE = RAW_DIR / "musae_git_features.json"

# File sau tiền xử lý
GRAPH_FILE = PROC_DIR / "graph.gpickle"
EDGE_ARRAY_FILE = PROC_DIR / "edges.npy"            # mảng (2, E), mỗi cạnh 1 lần
LABELS_FILE = PROC_DIR / "labels.csv"               # id, ml_target (không có tên)
FEAT_SPARSE_FILE = PROC_DIR / "features_sparse.npz"
FEAT_SVD_FILE = PROC_DIR / "features_svd128.npy"

# ===== Tham số chung =====
N_NODES = 37700
N_FEATURE_DIMS = 4005
SEEDS = [0, 1, 2, 3, 4]      # chạy lặp cho các phương pháp có yếu tố ngẫu nhiên
N_CLUSTERS = 2               # số nhãn thật (web / ML)
SVD_DIM = 128
SVD_SEED = 42

# ===== RQ1 =====
N_PERMUTATIONS = 1000        # số lần xáo nhãn trong kiểm định hoán vị
SMALL_CLUSTER_SIZE = 50      # cụm nhỏ hơn mức này được gộp vào nhóm "khác"
SKEW_THRESHOLD = 0.20        # cụm "lệch" nếu tỷ lệ ML chênh >= 20 điểm % so với toàn mạng

# ===== RQ2: phương pháp dựa trên cấu trúc =====
LEIDEN_RESOLUTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
MODULARITY_TOLERANCE = 0.90  # chỉ xét resolution có modularity >= 90% mức cao nhất
RUN_SPECTRAL = True          # Spectral chậm (15-40 phút), đặt False để bỏ qua

# ===== RQ2: phương pháp dùng embedding =====
FEATURE_PROP_K = 2           # số bước lan truyền đặc trưng

DW_DIM = 64                  # DeepWalk
DW_WALK_LENGTH = 30
DW_NUM_WALKS = 10
DW_WINDOW = 10
DW_EPOCHS = 5
DW_WORKERS = 4

GAE_HIDDEN = 64              # Graph Autoencoder
GAE_DIM = 32
GAE_EPOCHS = 300
GAE_LR = 0.01
GAE_ALPHA = 1.0              # trọng số của loss tái tạo đặc trưng

# ===== RQ3: thu thập star/fork =====
STAR_SAMPLE_SIZE = 2000
STAR_N_STRATA = 5
STAR_SAMPLE_SEED = 42
STAR_TOP_N = 200
STARS_FILE = EXT_DIR / "github_stars_sample.csv"


def ensure_dirs() -> None:
    """Tạo các thư mục đầu ra nếu chưa có."""
    for d in (PROC_DIR, EXT_DIR, RES_DIR, PART_DIR, EMB_DIR, LOG_DIR, TABLE_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
