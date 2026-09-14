"""
STEP 3 — Tiền xử lý dữ liệu:
  1. Load edges/features/target vào NetworkX graph
  2. Chuyển feature thưa (4005 chiều, dạng list index) -> ma trận sparse
  3. Giảm chiều feature bằng TruncatedSVD (theo khuyến nghị chính tác giả dataset)
  4. Lưu graph (pickle) + feature đã giảm chiều (npz) + label (csv) để các bước sau dùng chung
"""
import os
import json

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix, save_npz
from sklearn.decomposition import TruncatedSVD

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "git_web_ml")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SVD_DIM = 128   # số chiều sau khi giảm — tương đương bản 128-dim MUSAE cung cấp sẵn


def load_graph(edges_path: str) -> nx.Graph:
    edges_df = pd.read_csv(edges_path)
    G = nx.from_pandas_edgelist(edges_df, source="id_1", target="id_2")
    print(f"Graph: {G.number_of_nodes()} node, {G.number_of_edges()} cạnh")
    return G


def load_sparse_features(features_path: str, n_nodes: int, n_dims: int = 4005):
    with open(features_path) as f:
        raw = json.load(f)  # {"0": [12, 340, ...], "1": [...], ...}

    mat = lil_matrix((n_nodes, n_dims), dtype=np.float32)
    for node_id_str, idx_list in raw.items():
        node_id = int(node_id_str)
        for idx in idx_list:
            if idx < n_dims:
                mat[node_id, idx] = 1.0
    return mat.tocsr()


def load_labels(target_path: str) -> pd.DataFrame:
    df = pd.read_csv(target_path)  # cột: id, name, ml_target (0=web, 1=ml theo README gốc)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    G = load_graph(os.path.join(DATA_DIR, "musae_git_edges.csv"))
    n_nodes = max(G.nodes) + 1  # đảm bảo đủ index kể cả node cô lập không có trong edge list

    labels_df = load_labels(os.path.join(DATA_DIR, "musae_git_target.csv"))
    print(f"Phân bố nhãn:\n{labels_df['ml_target'].value_counts(normalize=True)}")

    features_sparse = load_sparse_features(
        os.path.join(DATA_DIR, "musae_git_features.json"), n_nodes
    )
    print(f"Feature matrix shape: {features_sparse.shape}, "
          f"độ thưa: {features_sparse.nnz / (features_sparse.shape[0]*features_sparse.shape[1]):.5f}")

    print(f"Giảm chiều bằng TruncatedSVD -> {SVD_DIM} chiều ...")
    svd = TruncatedSVD(n_components=SVD_DIM, random_state=42)
    features_reduced = svd.fit_transform(features_sparse)
    print(f"Explained variance ratio (tổng): {svd.explained_variance_ratio_.sum():.3f}")

    # Lưu output cho các bước sau
    nx.write_gpickle(G, os.path.join(OUT_DIR, "graph.gpickle")) \
        if hasattr(nx, "write_gpickle") else \
        __import__("pickle").dump(G, open(os.path.join(OUT_DIR, "graph.gpickle"), "wb"))
    save_npz(os.path.join(OUT_DIR, "features_sparse.npz"), features_sparse)
    np.save(os.path.join(OUT_DIR, "features_svd128.npy"), features_reduced)
    labels_df.to_csv(os.path.join(OUT_DIR, "labels.csv"), index=False)

    print(f"Đã lưu kết quả tiền xử lý vào: {OUT_DIR}")


if __name__ == "__main__":
    main()
