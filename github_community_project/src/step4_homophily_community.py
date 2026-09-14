"""
STEP 4 — Trả lời RQ1: web dev và ML dev có tách cụm riêng biệt không?

  A. Edge Homophily — đo trực tiếp trên cạnh, so với baseline ngẫu nhiên
  B. Community Detection (Louvain + Leiden) — chia cụm thuần cấu trúc,
     sau đó đối chiếu tỷ lệ web/ML trong từng cụm với tỷ lệ tổng thể

Kết quả của bước này (community_louvain.csv, community_leiden.csv) cũng chính là
input cho STEP 6 (so sánh với GCN ở RQ2).
"""
import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain     # python-louvain, import name là `community`
import igraph as ig
import leidenalg

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_graph_and_labels():
    with open(os.path.join(PROC_DIR, "graph.gpickle"), "rb") as f:
        G = pickle.load(f)
    labels_df = pd.read_csv(os.path.join(PROC_DIR, "labels.csv"))
    label_map = dict(zip(labels_df["id"], labels_df["ml_target"]))
    return G, label_map


# ---------- A. Edge Homophily ----------
def edge_homophily(G: nx.Graph, label_map: dict) -> tuple[float, float]:
    same, total = 0, 0
    for u, v in G.edges():
        lu, lv = label_map.get(u), label_map.get(v)
        if lu is None or lv is None:
            continue
        total += 1
        if lu == lv:
            same += 1
    observed = same / total

    # Baseline ngẫu nhiên: p_web^2 + p_ml^2
    labels = np.array(list(label_map.values()))
    p_web = (labels == 0).mean()
    p_ml = (labels == 1).mean()
    baseline = p_web ** 2 + p_ml ** 2

    return observed, baseline


# ---------- B1. Louvain ----------
def run_louvain(G: nx.Graph) -> dict:
    partition = community_louvain.best_partition(G, random_state=42)
    return partition  # {node_id: cluster_id}


# ---------- B2. Leiden ----------
def run_leiden(G: nx.Graph) -> dict:
    # Chuyển NetworkX graph -> igraph để dùng leidenalg
    node_list = list(G.nodes())
    node_index = {n: i for i, n in enumerate(node_list)}
    edges_idx = [(node_index[u], node_index[v]) for u, v in G.edges()]

    g_ig = ig.Graph(n=len(node_list), edges=edges_idx)
    part = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition, seed=42)

    return {node_list[i]: cl for i, cl in enumerate(part.membership)}


# ---------- Phân tích tỷ lệ nhãn theo từng cụm ----------
def cluster_label_distribution(partition: dict, label_map: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "node": list(partition.keys()),
        "cluster": list(partition.values()),
    })
    df["label"] = df["node"].map(label_map)
    df = df.dropna(subset=["label"])

    dist = df.groupby("cluster")["label"].agg(
        n_nodes="count",
        pct_ml=lambda x: (x == 1).mean() * 100,
    ).reset_index()
    dist["pct_web"] = 100 - dist["pct_ml"]
    return dist.sort_values("n_nodes", ascending=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    G, label_map = load_graph_and_labels()

    # --- A. Edge Homophily ---
    observed, baseline = edge_homophily(G, label_map)
    print(f"[RQ1 - Cách A] Edge Homophily quan sát  = {observed:.4f}")
    print(f"[RQ1 - Cách A] Baseline ngẫu nhiên       = {baseline:.4f}")
    print(f"[RQ1 - Cách A] Kết luận: "
          f"{'CÓ' if observed > baseline + 0.05 else 'KHÔNG rõ'} bằng chứng homophily")

    # --- B. Community Detection ---
    print("\nĐang chạy Louvain ...")
    louvain_partition = run_louvain(G)
    print(f"Louvain tìm được {len(set(louvain_partition.values()))} cụm")

    print("Đang chạy Leiden ...")
    leiden_partition = run_leiden(G)
    print(f"Leiden tìm được {len(set(leiden_partition.values()))} cụm")

    louvain_dist = cluster_label_distribution(louvain_partition, label_map)
    leiden_dist = cluster_label_distribution(leiden_partition, label_map)

    print("\nTop 10 cụm lớn nhất (Louvain) theo tỷ lệ web/ML:")
    print(louvain_dist.head(10).to_string(index=False))

    # --- Lưu kết quả cho STEP 6 (so sánh với GCN ở RQ2) ---
    pd.DataFrame(list(louvain_partition.items()), columns=["node", "cluster"]) \
        .to_csv(os.path.join(OUT_DIR, "community_louvain.csv"), index=False)
    pd.DataFrame(list(leiden_partition.items()), columns=["node", "cluster"]) \
        .to_csv(os.path.join(OUT_DIR, "community_leiden.csv"), index=False)
    louvain_dist.to_csv(os.path.join(OUT_DIR, "louvain_cluster_distribution.csv"), index=False)
    leiden_dist.to_csv(os.path.join(OUT_DIR, "leiden_cluster_distribution.csv"), index=False)

    with open(os.path.join(OUT_DIR, "rq1_homophily_summary.txt"), "w") as f:
        f.write(f"Edge Homophily quan sát: {observed:.4f}\n")
        f.write(f"Baseline ngẫu nhiên: {baseline:.4f}\n")
        f.write(f"Chênh lệch: {observed - baseline:.4f}\n")

    print(f"\nĐã lưu toàn bộ kết quả RQ1 vào: {OUT_DIR}")


if __name__ == "__main__":
    main()
