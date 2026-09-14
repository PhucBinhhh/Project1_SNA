"""
STEP 6 — Trả lời RQ2: phương pháp nào (Louvain / Leiden / GCN) khớp nhãn gốc (web/ML) tốt nhất?

So sánh 2 "cách chia nhóm" cho mỗi phương pháp:
  Cách chia 1 = cụm do thuật toán tự tìm ra (không biết nhãn)
  Cách chia 2 = nhãn thật (web/ML) có sẵn trong dataset

Metric:
  - NMI  (Normalized Mutual Information)
  - ARI  (Adjusted Rand Index)
  - Purity (tỷ lệ node được gán đúng nhãn đa số trong cụm của nó)
  - Modularity (chỉ đánh giá độ tốt cấu trúc của cụm, không cần nhãn — để tham khảo thêm)
"""
import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import community as community_louvain

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Với mỗi cụm dự đoán, gán nhãn = nhãn đa số trong cụm đó, rồi tính accuracy."""
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    majority_map = df.groupby("pred")["true"].agg(lambda x: x.value_counts().idxmax())
    df["mapped"] = df["pred"].map(majority_map)
    return (df["true"] == df["mapped"]).mean()


def evaluate_method(name: str, cluster_csv: str, labels_df: pd.DataFrame,
                     graph: nx.Graph, partition_for_modularity: dict) -> dict:
    clusters_df = pd.read_csv(cluster_csv)
    merged = clusters_df.merge(labels_df, left_on="node", right_on="id", how="inner")

    y_true = merged["ml_target"].values
    y_pred = merged["cluster"].values

    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)
    modularity = community_louvain.modularity(partition_for_modularity, graph)

    return {
        "method": name,
        "n_clusters_found": merged["cluster"].nunique(),
        "NMI": round(nmi, 4),
        "ARI": round(ari, 4),
        "Purity": round(purity, 4),
        "Modularity": round(modularity, 4),
    }


def main():
    labels_df = pd.read_csv(os.path.join(PROC_DIR, "labels.csv"))

    with open(os.path.join(PROC_DIR, "graph.gpickle"), "rb") as f:
        G = pickle.load(f)

    results = []

    for name, csv_file in [
        ("Louvain", "community_louvain.csv"),
        ("Leiden", "community_leiden.csv"),
        ("GCN (VGAE + KMeans)", "community_gcn.csv"),
    ]:
        path = os.path.join(OUT_DIR, csv_file)
        if not os.path.exists(path):
            print(f"[Bỏ qua] Chưa có file {csv_file}, hãy chạy step4/step5 trước.")
            continue

        partition_dict = dict(zip(
            pd.read_csv(path)["node"], pd.read_csv(path)["cluster"]
        ))
        res = evaluate_method(name, path, labels_df, G, partition_dict)
        results.append(res)
        print(f"{name}: {res}")

    result_df = pd.DataFrame(results).sort_values("ARI", ascending=False)
    result_df.to_csv(os.path.join(OUT_DIR, "rq2_comparison_table.csv"), index=False)

    print("\n===== BẢNG SO SÁNH CUỐI CÙNG (RQ2) =====")
    print(result_df.to_string(index=False))
    print(f"\n>>> Phương pháp khớp nhãn gốc tốt nhất (theo ARI): {result_df.iloc[0]['method']}")

    print(f"\nĐã lưu bảng so sánh vào: {os.path.join(OUT_DIR, 'rq2_comparison_table.csv')}")


if __name__ == "__main__":
    main()
