"""
STEP 04 — RQ2 (phần 1): các phương pháp CHỈ DÙNG CẤU TRÚC MẠNG.

  1. Louvain (resolution = 1), chạy với nhiều seed
  2. Leiden (resolution = 1), chạy với nhiều seed
  3. Quét resolution cho Leiden, rồi chọn resolution theo 2 cách:
       - "selected": KHÔNG dùng nhãn — trong các resolution có modularity đủ cao,
         chọn cái cho kết quả ổn định nhất giữa các seed
       - "oracle":   dùng nhãn (ARI cao nhất) — chỉ để tham khảo cận trên
  4. Spectral Clustering (k = 2): tính embedding phổ 1 lần, rồi KMeans với nhiều seed

Đầu ra: outputs/partitions/{louvain, leiden, leiden_selected, leiden_oracle, spectral}.npy
Chạy:   python src/step04_rq2_structure_methods.py
"""
import json
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding

import config as C
from utils import (communities_to_array, evaluate, load_edges, load_graph, load_labels,
                   pairwise_ari, run_leiden, save_partitions, save_table, to_igraph)


def run_louvain(G: nx.Graph, seed: int) -> np.ndarray:
    # Louvain có sẵn trong NetworkX: resolution cao hơn -> nhiều cụm hơn (cùng quy ước với Leiden)
    comms = nx.community.louvain_communities(G, resolution=1.0, seed=seed)
    return communities_to_array(comms, G.number_of_nodes())


def leiden_sweep(g, y: np.ndarray, edges: np.ndarray):
    rows, parts_by_res = [], {}
    for res in C.LEIDEN_RESOLUTIONS:
        parts = np.stack([run_leiden(g, seed=s, resolution=res) for s in C.SEEDS])
        parts_by_res[res] = parts
        scores = pd.DataFrame([evaluate(y, p, edges) for p in parts])
        rows.append({
            "resolution": res,
            "n_clusters": scores["n_clusters"].mean(),
            "Modularity": scores["Modularity"].mean(),
            "Stability": pairwise_ari(parts),
            "NMI": scores["NMI"].mean(),
            "ARI": scores["ARI"].mean(),
            "ARI_std": scores["ARI"].std(ddof=0),
        })
        r = rows[-1]
        print(f"  γ = {res:<4} | {r['n_clusters']:7.1f} cụm | Q = {r['Modularity']:.4f} | "
              f"ổn định = {r['Stability']:.3f} | ARI = {r['ARI']:.4f} ± {r['ARI_std']:.4f}")
    return pd.DataFrame(rows), parts_by_res


def select_resolution(sweep: pd.DataFrame) -> float:
    """Chọn resolution KHÔNG dùng nhãn: modularity >= 90% mức cao nhất, rồi ưu tiên ổn định."""
    candidates = sweep[sweep["Modularity"] >= C.MODULARITY_TOLERANCE * sweep["Modularity"].max()]
    best = candidates.sort_values(["Stability", "Modularity"], ascending=False).iloc[0]
    return float(best["resolution"])


def run_spectral(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    adj = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), dtype=float, format="csr")
    adj.indices = adj.indices.astype(np.int32)    # arpack yêu cầu chỉ số int32
    adj.indptr = adj.indptr.astype(np.int32)
    print("  Tính embedding phổ (bước chậm nhất, có thể 15-40 phút)...")
    emb = spectral_embedding(adj, n_components=C.N_CLUSTERS, eigen_solver="arpack",
                             random_state=0, drop_first=False)
    np.save(C.EMB_DIR / "spectral.npy", emb)
    return np.stack([
        KMeans(n_clusters=C.N_CLUSTERS, n_init=10, random_state=s).fit_predict(emb)
        for s in C.SEEDS
    ])


def main():
    C.ensure_dirs()
    y = load_labels()
    edges = load_edges()
    G = load_graph()
    g = to_igraph(edges, len(y))
    meta = {}

    # 1. Louvain
    t = time.time()
    print(f"[1] Louvain với {len(C.SEEDS)} seed (mỗi lần khoảng 1 phút)...")
    louvain = np.stack([run_louvain(G, s) for s in C.SEEDS])
    save_partitions("louvain", louvain)
    print(f"  Xong sau {time.time() - t:.0f} giây")

    # 2. Leiden mặc định
    print("[2] Leiden (γ = 1)...")
    leiden = np.stack([run_leiden(g, seed=s, resolution=1.0) for s in C.SEEDS])
    save_partitions("leiden", leiden)

    # 3. Quét resolution
    print("[3] Quét resolution cho Leiden...")
    sweep, parts_by_res = leiden_sweep(g, y, edges)
    res_selected = select_resolution(sweep)
    res_oracle = float(sweep.loc[sweep["ARI"].idxmax(), "resolution"])
    save_partitions("leiden_selected", parts_by_res[res_selected])
    save_partitions("leiden_oracle", parts_by_res[res_oracle])
    meta.update({"leiden_selected_resolution": res_selected,
                 "leiden_oracle_resolution": res_oracle})
    print(f"  γ chọn không dùng nhãn = {res_selected} | γ oracle (theo ARI) = {res_oracle}")

    sweep_out = sweep.round(4)
    sweep_out.columns = ["γ", "Số cụm", "Modularity", "Độ ổn định", "NMI", "ARI", "ARI (độ lệch chuẩn)"]
    save_table(sweep_out, "rq2_leiden_resolution_sweep",
               caption=f"Quét tham số resolution của Leiden (trung bình {len(C.SEEDS)} seed)",
               label="tab:rq2_sweep")

    # 4. Spectral
    if C.RUN_SPECTRAL:
        print("[4] Spectral Clustering...")
        t = time.time()
        save_partitions("spectral", run_spectral(G))
        print(f"  Xong sau {time.time() - t:.0f} giây")
    else:
        print("[4] Bỏ qua Spectral (RUN_SPECTRAL = False trong config.py)")

    (C.RES_DIR / "rq2_structure_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nĐã lưu kết quả phân cụm vào: {C.PART_DIR}")
    print("Chạy tiếp: python src/step05_rq2_embedding_methods.py")


if __name__ == "__main__":
    main()
