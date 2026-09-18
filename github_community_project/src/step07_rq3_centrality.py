"""
STEP 07 — RQ3 (phần 1): tính các độ đo centrality trên mạng MUSAE.

Độ đo toàn cục (trên cả mạng):
  degree, PageRank, eigenvector, betweenness (xấp xỉ), k-core
Độ đo theo cộng đồng (dùng phân cụm Leiden γ chọn không dùng nhãn, seed đầu tiên):
  - bậc trong cụm (k_in) và tỷ lệ kết nối trong cụm
  - PageRank trong cụm (đã nhân với kích thước cụm để so sánh được giữa các cụm)
  - hệ số tham gia (participation coefficient): gần 0 = chỉ kết nối trong cụm,
    càng lớn = kết nối trải đều nhiều cụm (vai trò "cầu nối")

Đầu ra:
  outputs/results/rq3_centrality.csv   (chỉ có id, không có tên tài khoản)
  bảng tương quan giữa các độ đo, bảng mô tả nhóm người có ảnh hưởng nhất
Chạy: python src/step07_rq3_centrality.py
"""
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

import config as C
from utils import load_edges, load_graph, load_labels, load_partitions, save_table

BETWEENNESS_K = 1000     # số node nguồn lấy mẫu khi xấp xỉ betweenness
TOP_K = 100              # số người đứng đầu dùng để so sánh các cách xếp hạng

CENTRALITY_COLS = ["degree", "pagerank", "eigenvector", "betweenness", "core",
                   "k_in", "pagerank_in", "participation"]


def global_centralities(G: nx.Graph) -> pd.DataFrame:
    n = G.number_of_nodes()
    df = pd.DataFrame(index=pd.RangeIndex(n, name="id"))

    df["degree"] = [d for _, d in G.degree()]

    t = time.time()
    pr = nx.pagerank(G, alpha=0.85)
    df["pagerank"] = [pr[i] for i in range(n)]
    print(f"  PageRank: {time.time() - t:.0f}s")

    t = time.time()
    ev = nx.eigenvector_centrality_numpy(G)
    df["eigenvector"] = [abs(ev[i]) for i in range(n)]
    print(f"  Eigenvector: {time.time() - t:.0f}s")

    t = time.time()
    print(f"  Betweenness xấp xỉ với {BETWEENNESS_K} node nguồn (10-30 phút)...")
    bc = nx.betweenness_centrality(G, k=BETWEENNESS_K, seed=0, normalized=True)
    df["betweenness"] = [bc[i] for i in range(n)]
    print(f"  Betweenness: {time.time() - t:.0f}s")

    core = nx.core_number(G)
    df["core"] = [core[i] for i in range(n)]
    return df


def community_centralities(G: nx.Graph, edges: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    n = len(labels)
    _, lab = np.unique(labels, return_inverse=True)
    n_clusters = lab.max() + 1
    u, v = edges

    # Ma trận (node x cụm): số láng giềng của mỗi node thuộc từng cụm
    rows = np.concatenate([u, v])
    cols = np.concatenate([lab[v], lab[u]])
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n_clusters))
    degree = np.asarray(M.sum(axis=1)).ravel()
    k_in = np.asarray(M[np.arange(n), lab]).ravel()
    share = M.multiply(1.0 / degree[:, None]).tocsr()
    participation = 1.0 - np.asarray(share.multiply(share).sum(axis=1)).ravel()

    # PageRank tính riêng trong từng cụm, nhân với kích thước cụm (trung bình = 1)
    pr_in = np.zeros(n)
    for c in range(n_clusters):
        members = np.flatnonzero(lab == c)
        if len(members) < 3:
            pr_in[members] = 1.0
            continue
        pr = nx.pagerank(G.subgraph(members.tolist()), alpha=0.85)
        pr_in[members] = [pr[i] * len(members) for i in members]

    return pd.DataFrame({
        "cluster": lab,
        "cluster_size": np.bincount(lab)[lab],
        "k_in": k_in.astype(int),
        "share_in": k_in / degree,
        "pagerank_in": pr_in,
        "participation": participation,
    }, index=pd.RangeIndex(n, name="id"))


def describe_top(df: pd.DataFrame, y_mean: float) -> pd.DataFrame:
    """So sánh nhóm top-K theo từng độ đo với toàn mạng."""
    rows = [{
        "Nhóm": "Toàn mạng",
        "Tỷ lệ ML": f"{y_mean:.3f}",
        "Bậc trung vị": f"{df['degree'].median():.0f}",
        "Tỷ lệ kết nối trong cụm": f"{df['share_in'].median():.3f}",
        "Hệ số tham gia (trung vị)": f"{df['participation'].median():.3f}",
        "Số cụm có mặt": df["cluster"].nunique(),
    }]
    for col, name in [("pagerank", "PageRank toàn cục"),
                      ("betweenness", "Betweenness"),
                      ("pagerank_in", "PageRank trong cụm")]:
        top = df.nlargest(TOP_K, col)
        rows.append({
            "Nhóm": f"Top {TOP_K} theo {name}",
            "Tỷ lệ ML": f"{top['ml_target'].mean():.3f}",
            "Bậc trung vị": f"{top['degree'].median():.0f}",
            "Tỷ lệ kết nối trong cụm": f"{top['share_in'].median():.3f}",
            "Hệ số tham gia (trung vị)": f"{top['participation'].median():.3f}",
            "Số cụm có mặt": top["cluster"].nunique(),
        })
    return pd.DataFrame(rows)


def top_k_overlap(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["degree", "pagerank", "betweenness", "pagerank_in", "participation"]
    tops = {c: set(df.nlargest(TOP_K, c).index) for c in cols}
    mat = pd.DataFrame([[len(tops[a] & tops[b]) / TOP_K for b in cols] for a in cols],
                       index=cols, columns=cols)
    mat.index.name = f"Tỷ lệ trùng top {TOP_K}"
    return mat


def main():
    start = time.time()
    C.ensure_dirs()
    G = load_graph()
    edges = load_edges()
    y = load_labels()

    parts = load_partitions("leiden_selected")
    if parts is None:
        raise RuntimeError("Chưa có outputs/partitions/leiden_selected.npy. Hãy chạy step04 trước.")
    labels = parts[0]
    print(f"Dùng phân cụm Leiden (γ chọn không dùng nhãn, seed đầu): {len(np.unique(labels))} cụm")

    print("[1] Độ đo toàn cục")
    glob = global_centralities(G)
    print("[2] Độ đo theo cộng đồng")
    comm = community_centralities(G, edges, labels)

    df = glob.join(comm)
    df.insert(0, "ml_target", y)
    df.to_csv(C.RES_DIR / "rq3_centrality.csv", encoding="utf-8-sig")
    print(f"  Đã lưu: {C.RES_DIR / 'rq3_centrality.csv'}")

    print("\n[3] Tương quan Spearman giữa các độ đo")
    corr = df[CENTRALITY_COLS].corr(method="spearman").round(3)
    print(corr.to_string())
    save_table(corr, "rq3_centrality_correlation", index=True,
               caption="Tương quan Spearman giữa các độ đo centrality",
               label="tab:rq3_centrality_corr")

    print(f"\n[4] Đặc điểm nhóm top {TOP_K}")
    top = describe_top(df, y.mean())
    print(top.to_string(index=False))
    save_table(top, "rq3_top_nodes",
               caption=f"Đặc điểm nhóm {TOP_K} người đứng đầu theo từng độ đo",
               label="tab:rq3_top")

    overlap = top_k_overlap(df).round(2)
    print("\n" + overlap.to_string())
    save_table(overlap, "rq3_topk_overlap", index=True,
               caption=f"Tỷ lệ trùng giữa các danh sách top {TOP_K}",
               label="tab:rq3_overlap")

    print(f"\nHOÀN TẤT sau {(time.time() - start) / 60:.1f} phút. "
          f"Khi step08 xong, chạy: python src/step09_rq3_correlation.py")


if __name__ == "__main__":
    main()