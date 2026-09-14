"""
STEP 7 — Trả lời RQ3: centrality có tương quan với star/fork không?

Chạy 2 nhánh song song (xem README phần "Lưu ý quan trọng về RQ3"):

  Nhánh A (nội tại, luôn chạy được): tính các loại centrality trên đúng 37.700-node
    MUSAE graph, so sánh tương quan GIỮA CÁC CENTRALITY VỚI NHAU + với degree
    (proxy cho "được chú ý trong mạng"). Không cần dữ liệu star/fork thật.

  Nhánh B (cần STEP 2 đã crawl xong): dùng mini-network GitHub thật (có username thật),
    tính centrality trên mini-network đó, tương quan Spearman với total_stars_received /
    total_forks_received THẬT lấy từ GitHub API.
"""
import os
import pickle
import json

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr, pearsonr

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SUPP_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "supplement")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def compute_centralities(G: nx.Graph, approx: bool = True) -> pd.DataFrame:
    print("Tính degree centrality ...")
    degree = nx.degree_centrality(G)

    print("Tính eigenvector centrality ...")
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        eigenvector = nx.eigenvector_centrality_numpy(G)

    print("Tính betweenness centrality (approx bằng sampling k node cho graph lớn) ...")
    k_sample = 500 if approx and G.number_of_nodes() > 5000 else None
    betweenness = nx.betweenness_centrality(G, k=k_sample, seed=42)

    df = pd.DataFrame({
        "node": list(degree.keys()),
        "degree_centrality": list(degree.values()),
        "eigenvector_centrality": [eigenvector.get(n, np.nan) for n in degree.keys()],
        "betweenness_centrality": [betweenness.get(n, np.nan) for n in degree.keys()],
    })
    return df


def branch_a_internal(G: nx.Graph):
    print("\n===== NHÁNH A: Centrality nội tại trên MUSAE graph =====")
    df = compute_centralities(G)
    df.to_csv(os.path.join(OUT_DIR, "centrality_musae.csv"), index=False)

    pairs = [
        ("degree_centrality", "eigenvector_centrality"),
        ("degree_centrality", "betweenness_centrality"),
        ("eigenvector_centrality", "betweenness_centrality"),
    ]
    rows = []
    for a, b in pairs:
        rho, p = spearmanr(df[a], df[b])
        rows.append({"pair": f"{a} vs {b}", "spearman_rho": round(rho, 4), "p_value": p})
        print(f"{a} vs {b}: rho={rho:.4f}, p={p:.2e}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "rq3_internal_correlation.csv"), index=False)


def branch_b_supplement():
    users_path = os.path.join(SUPP_DIR, "supplement_users.json")
    edges_path = os.path.join(SUPP_DIR, "supplement_edges.json")

    if not (os.path.exists(users_path) and os.path.exists(edges_path)):
        print("\n[Bỏ qua NHÁNH B] Chưa có dữ liệu crawl bổ sung — hãy chạy step2 trước "
              "nếu muốn kết quả RQ3 dựa trên star/fork THẬT.")
        return

    print("\n===== NHÁNH B: Centrality vs star/fork THẬT (mini-network GitHub) =====")
    with open(users_path) as f:
        users = pd.DataFrame(json.load(f))
    with open(edges_path) as f:
        edges = json.load(f)

    G_mini = nx.DiGraph()
    G_mini.add_edges_from(edges)

    degree = dict(G_mini.degree())
    try:
        eigenvector = nx.eigenvector_centrality(G_mini.to_undirected(), max_iter=500)
    except nx.PowerIterationFailedConvergence:
        eigenvector = nx.eigenvector_centrality_numpy(G_mini.to_undirected())

    users["degree_centrality"] = users["login"].map(degree)
    users["eigenvector_centrality"] = users["login"].map(eigenvector)
    users = users.dropna(subset=["degree_centrality", "eigenvector_centrality"])

    results = []
    for centrality_col in ["degree_centrality", "eigenvector_centrality"]:
        for target_col in ["total_stars_received", "total_forks_received"]:
            rho, p = spearmanr(users[centrality_col], users[target_col])
            results.append({
                "centrality": centrality_col, "target": target_col,
                "spearman_rho": round(rho, 4), "p_value": p,
            })
            print(f"{centrality_col} vs {target_col}: rho={rho:.4f}, p={p:.2e}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(OUT_DIR, "rq3_real_star_fork_correlation.csv"), index=False)
    users.to_csv(os.path.join(OUT_DIR, "supplement_users_with_centrality.csv"), index=False)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(PROC_DIR, "graph.gpickle"), "rb") as f:
        G = pickle.load(f)

    branch_a_internal(G)
    branch_b_supplement()

    print(f"\nĐã lưu toàn bộ kết quả RQ3 vào: {OUT_DIR}")


if __name__ == "__main__":
    main()
