"""
STEP 03 — RQ1: Web developer và ML developer có hình thành cộng đồng tách biệt không?

Bằng chứng A — ở mức cạnh:
  - Edge homophily và mức nền ngẫu nhiên p_web^2 + p_ml^2
  - Adjusted homophily (Platonov et al., 2023), hiệu chỉnh cho mất cân bằng nhãn;
    với nhãn nhị phân, chỉ số này bằng hệ số đồng loại (assortativity) của Newman
  - Ma trận trộn: số cạnh web-web, web-ML, ML-ML so với kỳ vọng ngẫu nhiên
  - Kiểm định hoán vị: xáo nhãn N lần để tính p-value

Bằng chứng B — ở mức cộng đồng:
  - Leiden (modularity chuẩn) chạy trên cấu trúc mạng, KHÔNG dùng nhãn
  - So sánh tỷ lệ ML trong từng cụm với toàn mạng, kiểm định chi-bình phương, Cramér's V
  - Lặp lại với nhiều seed để kiểm tra độ ổn định

Chạy: python src/step03_rq1_homophily.py
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import config as C
from utils import load_edges, load_labels, mean_std, run_leiden, save_table, to_igraph


# ---------- A. Mức cạnh ----------
def edge_homophily(y: np.ndarray, edges: np.ndarray) -> float:
    return float((y[edges[0]] == y[edges[1]]).mean())


def adjusted_homophily(y: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    """Trả về (adjusted homophily, mức kỳ vọng có trọng số theo bậc)."""
    deg = np.bincount(edges.ravel(), minlength=len(y))
    two_m = 2 * edges.shape[1]
    deg_share = np.bincount(y, weights=deg, minlength=2) / two_m
    expected = float((deg_share ** 2).sum())
    h = edge_homophily(y, edges)
    return (h - expected) / (1 - expected), expected


def mixing_matrix(y: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    a, b = y[edges[0]], y[edges[1]]
    n_edges = edges.shape[1]
    p_ml = y.mean()
    p_web = 1 - p_ml
    rows = [
        ("Web - Web", int(((a == 0) & (b == 0)).sum()), n_edges * p_web ** 2),
        ("Web - ML", int((a != b).sum()), n_edges * 2 * p_web * p_ml),
        ("ML - ML", int(((a == 1) & (b == 1)).sum()), n_edges * p_ml ** 2),
    ]
    df = pd.DataFrame(rows, columns=["Loại cạnh", "Quan sát", "Kỳ vọng ngẫu nhiên"])
    df["Tỷ lệ quan sát"] = df["Quan sát"] / n_edges
    df["Quan sát / Kỳ vọng"] = df["Quan sát"] / df["Kỳ vọng ngẫu nhiên"]
    df["Kỳ vọng ngẫu nhiên"] = df["Kỳ vọng ngẫu nhiên"].round().astype(int)
    return df


def permutation_test(y: np.ndarray, edges: np.ndarray, observed: float):
    rng = np.random.default_rng(C.SEEDS[0])
    u, v = edges
    null = np.empty(C.N_PERMUTATIONS)
    for i in range(C.N_PERMUTATIONS):
        yp = rng.permutation(y)
        null[i] = (yp[u] == yp[v]).mean()
    p_value = (np.sum(null >= observed) + 1) / (C.N_PERMUTATIONS + 1)
    z_score = (observed - null.mean()) / null.std()
    return null, float(p_value), float(z_score)


# ---------- B. Mức cộng đồng ----------
def cluster_composition(labels: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"cluster": labels, "ml": y})
    comp = df.groupby("cluster")["ml"].agg(n_nodes="size", n_ml="sum").reset_index()
    comp["pct_ml"] = comp["n_ml"] / comp["n_nodes"]
    comp["chenh_lech"] = comp["pct_ml"] - y.mean()
    comp["lon"] = comp["n_nodes"] >= C.SMALL_CLUSTER_SIZE
    comp["lech"] = comp["lon"] & (comp["chenh_lech"].abs() >= C.SKEW_THRESHOLD)
    return comp.sort_values("n_nodes", ascending=False).reset_index(drop=True)


def composition_test(comp: pd.DataFrame, n_total: int) -> dict:
    """Chi-bình phương trên bảng (cụm x nhãn); cụm nhỏ được gộp thành 1 nhóm."""
    big = comp[comp["lon"]]
    small = comp[~comp["lon"]]
    table = big[["n_nodes", "n_ml"]].to_numpy()
    if len(small):
        table = np.vstack([table, [small["n_nodes"].sum(), small["n_ml"].sum()]])
    table = np.column_stack([table[:, 0] - table[:, 1], table[:, 1]])   # [n_web, n_ml]
    chi2, p, dof, _ = chi2_contingency(table)
    return {
        "so_cum": len(comp),
        "so_cum_lon": len(big),
        "so_cum_lech": int(comp["lech"].sum()),
        "ty_le_node_trong_cum_lech": float(comp.loc[comp["lech"], "n_nodes"].sum() / n_total),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": float(np.sqrt(chi2 / n_total)),   # bảng k x 2 nên min(r, c) - 1 = 1
    }


def main():
    C.ensure_dirs()
    y = load_labels()
    edges = load_edges()
    n = len(y)

    # ===== A. Mức cạnh =====
    print("[A] Homophily ở mức cạnh")
    h = edge_homophily(y, edges)
    baseline = float(y.mean() ** 2 + (1 - y.mean()) ** 2)
    h_adj, expected_deg = adjusted_homophily(y, edges)
    print(f"  Edge homophily = {h:.4f} | mức nền = {baseline:.4f} | adjusted = {h_adj:.4f}")

    mix = mixing_matrix(y, edges)
    print(mix.to_string(index=False))

    print(f"  Kiểm định hoán vị ({C.N_PERMUTATIONS} lần)...")
    null, p_perm, z = permutation_test(y, edges, h)
    np.save(C.RES_DIR / "rq1_permutation_null.npy", null)
    print(f"  Phân phối null: {null.mean():.4f} ± {null.std():.4f} | z = {z:.1f} | p = {p_perm:.4f}")

    # ===== B. Mức cộng đồng =====
    print("\n[B] Thành phần nhãn trong các cụm Leiden")
    g = to_igraph(edges, n)
    summaries = []
    for seed in C.SEEDS:
        labels = run_leiden(g, seed=seed, resolution=1.0)
        comp = cluster_composition(labels, y)
        s = composition_test(comp, n)
        summaries.append({"seed": seed, **s})
        print(f"  seed {seed}: {s['so_cum']} cụm ({s['so_cum_lon']} cụm lớn, "
              f"{s['so_cum_lech']} cụm lệch) | Cramér's V = {s['cramers_v']:.3f} | "
              f"p = {s['p_value']:.1e}")
        if seed == C.SEEDS[0]:
            comp_main = comp
    summ = pd.DataFrame(summaries)

    # ===== Lưu kết quả =====
    summary = pd.DataFrame([
        ("Edge homophily quan sát", f"{h:.4f}"),
        ("Mức nền ngẫu nhiên (p_web² + p_ml²)", f"{baseline:.4f}"),
        ("Mức kỳ vọng theo bậc", f"{expected_deg:.4f}"),
        ("Adjusted homophily (= assortativity)", f"{h_adj:.4f}"),
        ("Phân phối null (trung bình ± độ lệch chuẩn)", f"{null.mean():.4f} ± {null.std():.4f}"),
        ("z-score", f"{z:.1f}"),
        (f"p-value hoán vị ({C.N_PERMUTATIONS} lần)", f"< {1 / (C.N_PERMUTATIONS + 1):.4f}"
         if p_perm <= 1 / (C.N_PERMUTATIONS + 1) else f"{p_perm:.4f}"),
        ("Số cụm Leiden", mean_std(summ["so_cum"], 1)),
        (f"Số cụm lớn (≥ {C.SMALL_CLUSTER_SIZE} node)", mean_std(summ["so_cum_lon"], 1)),
        (f"Số cụm lệch (chênh ≥ {C.SKEW_THRESHOLD:.0%})", mean_std(summ["so_cum_lech"], 1)),
        ("Tỷ lệ node thuộc cụm lệch", mean_std(summ["ty_le_node_trong_cum_lech"], 3)),
        ("Cramér's V", mean_std(summ["cramers_v"], 3)),
        ("p-value chi-bình phương (lớn nhất qua các seed)",
         "< 1e-10" if summ["p_value"].max() < 1e-10 else f"{summ['p_value'].max():.1e}"),
    ], columns=["Chỉ số", "Giá trị"])
    print("\n" + summary.to_string(index=False))

    save_table(summary, "rq1_summary",
               caption="Kết quả RQ1: homophily và thành phần nhãn theo cộng đồng",
               label="tab:rq1_summary")
    save_table(mix, "rq1_mixing_matrix",
               caption="Ma trận trộn theo loại cạnh so với kỳ vọng ngẫu nhiên",
               label="tab:rq1_mixing")
    save_table(summ, "rq1_leiden_per_seed")

    comp_out = comp_main[comp_main["lon"]].copy()
    comp_out["pct_ml"] = (comp_out["pct_ml"] * 100).round(1)
    comp_out["chenh_lech"] = (comp_out["chenh_lech"] * 100).round(1)
    comp_out = comp_out[["cluster", "n_nodes", "n_ml", "pct_ml", "chenh_lech", "lech"]]
    comp_out.columns = ["Cụm", "Số node", "Số ML", "% ML", "Chênh lệch (điểm %)", "Lệch"]
    save_table(comp_out, "rq1_cluster_composition",
               caption=f"Thành phần nhãn trong các cụm Leiden lớn (seed {C.SEEDS[0]})",
               label="tab:rq1_composition")
    np.save(C.PART_DIR / "rq1_leiden_seed0.npy", run_leiden(g, seed=C.SEEDS[0]))

    print(f"\nĐã lưu kết quả RQ1 vào {C.RES_DIR} và {C.TABLE_DIR}")
    print("Chạy tiếp: python src/step04_rq2_structure_methods.py")


if __name__ == "__main__":
    main()
