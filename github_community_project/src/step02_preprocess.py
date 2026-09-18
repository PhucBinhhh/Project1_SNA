"""
STEP 02 — Tiền xử lý dữ liệu.

1. Dựng đồ thị NetworkX với node được thêm THEO ĐÚNG THỨ TỰ id (0, 1, 2, ...),
   để mọi mảng theo node đều khớp id (tránh lỗi lệch thứ tự của GCN cũ).
2. Lưu mảng cạnh (2, E) dùng cho tính toán nhanh và cho PyTorch Geometric.
3. Dựng ma trận đặc trưng thưa 37.700 x 4.005 và giảm chiều bằng TruncatedSVD.
4. Lưu nhãn (không kèm tên tài khoản).
5. Tính thống kê mô tả mạng cho báo cáo.

Chạy: python src/step02_preprocess.py
"""
import json
import pickle
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

import config as C
from utils import save_table


def build_graph(edges_df: pd.DataFrame, n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))                      # thêm node theo thứ tự id trước
    G.add_edges_from(edges_df[["id_1", "id_2"]].itertuples(index=False, name=None))
    assert list(G.nodes()) == list(range(n)), "Thứ tự node không khớp id"
    return G


def build_edge_array(G: nx.Graph) -> np.ndarray:
    edges = np.array(G.edges(), dtype=np.int64).T   # (2, E)
    return np.sort(edges, axis=0)                   # hàng 0 luôn là id nhỏ hơn


def build_sparse_features(n: int) -> sparse.csr_matrix:
    with open(C.FEATURES_FILE, encoding="utf-8") as f:
        raw = json.load(f)                          # {"0": [12, 340, ...], ...}
    rows, cols = [], []
    for node_id, idx_list in raw.items():
        rows.extend([int(node_id)] * len(idx_list))
        cols.extend(idx_list)
    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, C.N_FEATURE_DIMS),
    )
    X.sum_duplicates()
    X.data[:] = 1.0                                 # đặc trưng nhị phân
    return X


def fmt(value) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"                         # 37700 -> 37,700
    if abs(value) < 0.01:
        return f"{value:.6f}"                       # số rất nhỏ như mật độ
    return f"{value:.4f}"


def network_statistics(G: nx.Graph, y: np.ndarray) -> pd.DataFrame:
    deg = np.array([d for _, d in G.degree()])
    components = list(nx.connected_components(G))
    largest = max(len(c) for c in components)

    print("  Đang tính hệ số gom cụm (khoảng 1 phút)...")
    rows = [
        ("Số node", G.number_of_nodes()),
        ("Số cạnh", G.number_of_edges()),
        ("Mật độ", nx.density(G)),
        ("Bậc trung bình", deg.mean()),
        ("Bậc trung vị", int(np.median(deg))),
        ("Bậc lớn nhất", int(deg.max())),
        ("Số thành phần liên thông", len(components)),
        ("Tỷ lệ node trong thành phần lớn nhất", largest / G.number_of_nodes()),
        ("Hệ số gom cụm trung bình", nx.average_clustering(G)),
        ("Transitivity", nx.transitivity(G)),
        ("Hệ số đồng loại theo bậc", nx.degree_assortativity_coefficient(G)),
        ("Tỷ lệ ML developer", float(y.mean())),
    ]
    return pd.DataFrame([(k, fmt(v)) for k, v in rows], columns=["Chỉ số", "Giá trị"])


def main():
    start = time.time()
    C.ensure_dirs()

    edges_df = pd.read_csv(C.EDGES_FILE)
    target = pd.read_csv(C.TARGET_FILE).sort_values("id")
    n = len(target)
    y = target["ml_target"].to_numpy()

    # 1-2. Đồ thị và mảng cạnh
    print("Dựng đồ thị...")
    G = build_graph(edges_df, n)
    edges = build_edge_array(G)
    print(f"  {G.number_of_nodes()} node, {G.number_of_edges()} cạnh")

    # 3. Đặc trưng
    print("Dựng ma trận đặc trưng thưa...")
    X = build_sparse_features(n)
    print(f"  Kích thước {X.shape}, tỷ lệ ô khác 0: {X.nnz / (X.shape[0] * X.shape[1]):.5f}")

    print(f"Giảm chiều TruncatedSVD -> {C.SVD_DIM} chiều...")
    svd = TruncatedSVD(n_components=C.SVD_DIM, random_state=C.SVD_SEED)
    X_svd = svd.fit_transform(X).astype(np.float32)
    explained = float(svd.explained_variance_ratio_.sum())
    print(f"  Tỷ lệ phương sai giữ lại: {explained:.3f}")

    # 4. Lưu
    with open(C.GRAPH_FILE, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(C.EDGE_ARRAY_FILE, edges)
    sparse.save_npz(C.FEAT_SPARSE_FILE, X)
    np.save(C.FEAT_SVD_FILE, X_svd)
    target[["id", "ml_target"]].to_csv(C.LABELS_FILE, index=False)
    print(f"Đã lưu dữ liệu tiền xử lý vào: {C.PROC_DIR}")

    # 5. Thống kê mạng
    print("Tính thống kê mô tả mạng...")
    stats = network_statistics(G, y)
    stats.loc[len(stats)] = ["Phương sai giữ lại sau SVD-128", fmt(explained)]
    print(stats.to_string(index=False))
    save_table(stats, "network_stats",
               caption="Thống kê mô tả mạng theo dõi lẫn nhau trên GitHub",
               label="tab:network_stats")

    print(f"\nHOÀN TẤT sau {time.time() - start:.0f} giây. "
          f"Chạy tiếp: python src/step03_rq1_homophily.py")


if __name__ == "__main__":
    main()
