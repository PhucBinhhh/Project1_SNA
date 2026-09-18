"""Các hàm dùng chung: đọc dữ liệu, đánh giá phân cụm, lưu kết quả.

Quy ước quan trọng: mọi mảng theo node đều được sắp theo id,
tức phần tử thứ i luôn là của node có id = i.
"""
import itertools
import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import config as C


# ---------- Đọc dữ liệu đã tiền xử lý ----------
def load_graph():
    with open(C.GRAPH_FILE, "rb") as f:
        return pickle.load(f)


def load_edges() -> np.ndarray:
    """Mảng (2, E): mỗi cạnh vô hướng xuất hiện đúng 1 lần."""
    return np.load(C.EDGE_ARRAY_FILE)


def load_labels() -> np.ndarray:
    df = pd.read_csv(C.LABELS_FILE).sort_values("id")
    return df["ml_target"].to_numpy()


def load_svd_features() -> np.ndarray:
    return np.load(C.FEAT_SVD_FILE)


def load_sparse_features():
    return sparse.load_npz(C.FEAT_SPARSE_FILE)


# ---------- Chuyển đổi đồ thị và phân cụm ----------
def to_igraph(edges: np.ndarray, n: int):
    import igraph as ig
    return ig.Graph(n=n, edges=edges.T.tolist())


def run_leiden(g, seed: int, resolution: float = 1.0) -> np.ndarray:
    """Leiden với hàm mục tiêu modularity có tham số resolution.
    resolution = 1 tương đương modularity chuẩn."""
    import leidenalg
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed,
    )
    return np.asarray(part.membership)


def communities_to_array(communities, n: int) -> np.ndarray:
    """Danh sách tập node -> mảng nhãn cụm theo id."""
    labels = np.full(n, -1, dtype=np.int64)
    for c, members in enumerate(communities):
        labels[list(members)] = c
    assert (labels >= 0).all(), "Có node chưa được gán cụm"
    return labels


def pairwise_ari(partitions: np.ndarray) -> float:
    """Độ ổn định: ARI trung bình giữa các cặp lần chạy (không dùng nhãn thật)."""
    pairs = itertools.combinations(range(len(partitions)), 2)
    return float(np.mean([adjusted_rand_score(partitions[i], partitions[j]) for i, j in pairs]))


# ---------- Chỉ số đánh giá ----------
def modularity(labels: np.ndarray, edges: np.ndarray) -> float:
    """Modularity chuẩn (resolution = 1) của đồ thị vô hướng không trọng số."""
    _, lab = np.unique(labels, return_inverse=True)
    n_edges = edges.shape[1]
    deg = np.bincount(edges.ravel(), minlength=len(lab))
    u, v = edges
    same = lab[u] == lab[v]
    internal = same.sum()
    deg_sum = np.bincount(lab, weights=deg)
    return float(internal / n_edges - ((deg_sum / (2 * n_edges)) ** 2).sum())


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mỗi cụm gán nhãn đa số, rồi tính tỷ lệ node đúng."""
    table = pd.crosstab(y_pred, y_true)
    return float(table.max(axis=1).sum() / len(y_true))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, edges: np.ndarray) -> dict:
    return {
        "n_clusters": int(len(np.unique(y_pred))),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "Purity": float(purity(y_true, y_pred)),
        "Modularity": modularity(y_pred, edges),
    }


# ---------- Lưu kết quả ----------
def save_partitions(name: str, partitions: np.ndarray) -> None:
    """partitions: mảng (số seed, số node)."""
    np.save(C.PART_DIR / f"{name}.npy", np.atleast_2d(partitions))


def load_partitions(name: str):
    path = C.PART_DIR / f"{name}.npy"
    return np.load(path) if path.exists() else None


def save_table(df: pd.DataFrame, name: str, caption: str | None = None,
               label: str | None = None, index: bool = False) -> None:
    """Lưu CSV vào outputs/results; nếu có caption thì xuất thêm bảng LaTeX."""
    df.to_csv(C.RES_DIR / f"{name}.csv", index=index, encoding="utf-8-sig")
    if caption:
        tex = df.to_latex(index=index, caption=caption, label=label,
                          float_format="%.4f", escape=True, position="htbp")
        (C.TABLE_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")


def mean_std(values, digits: int = 4) -> str:
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.{digits}f} ± {values.std(ddof=0):.{digits}f}"
