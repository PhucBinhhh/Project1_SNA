"""
STEP 06 — RQ2 (phần 3): đánh giá và so sánh tất cả phương pháp.

1. Bảng so sánh chính: NMI, ARI, Purity, Modularity (trung bình ± độ lệch chuẩn qua các seed)
2. Bảng diễn giải "gộp theo nhãn đa số": gộp mỗi cụm về nghề chiếm đa số trong cụm,
   rồi tính lại ARI và F1 cho lớp ML. Bước gộp CÓ DÙNG NHÃN, nên bảng này chỉ để
   giải thích vì sao ARI gốc thấp, KHÔNG dùng để so sánh hay chọn phương pháp.
3. Phân tích chẩn đoán (linear probe): Logistic Regression 5-fold trên từng loại biểu diễn,
   để biết mỗi biểu diễn CHỨA bao nhiêu thông tin về nhãn, độc lập với việc phân cụm.

Chạy: python src/step06_rq2_evaluate.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from utils import (evaluate, load_edges, load_labels, load_partitions, load_svd_features,
                   mean_std, save_table)

# (tên file phân cụm, tên hiển thị, dùng cấu trúc, dùng đặc trưng)
METHODS = [
    ("louvain", "Louvain (γ = 1)", True, False),
    ("leiden", "Leiden (γ = 1)", True, False),
    ("leiden_selected", "Leiden (γ = {sel}, chọn không dùng nhãn)", True, False),
    ("spectral", "Spectral Clustering", True, False),
    ("deepwalk_kmeans", "DeepWalk + KMeans", True, False),
    ("svd_kmeans", "KMeans trên đặc trưng SVD", False, True),
    ("featprop_kmeans", "Lan truyền đặc trưng + KMeans", True, True),
    ("gae_kmeans", "GAE + KMeans", True, True),
]
ORACLE = ("leiden_oracle", "Leiden (γ = {ora}, chọn theo nhãn, tham khảo)", True, False)

# (tên hiển thị, đường dẫn embedding)
REPRESENTATIONS = [
    ("Đặc trưng SVD-128", C.FEAT_SVD_FILE),
    ("Lan truyền đặc trưng", C.EMB_DIR / "featprop.npy"),
    ("DeepWalk", C.EMB_DIR / "deepwalk_seed0.npy"),
    ("GAE", C.EMB_DIR / "gae_seed0.npy"),
    ("Embedding phổ (2 chiều)", C.EMB_DIR / "spectral.npy"),
]


def evaluate_all(y, edges, meta):
    fmt = {"sel": meta.get("leiden_selected_resolution", "?"),
           "ora": meta.get("leiden_oracle_resolution", "?")}
    summary, per_seed = [], []
    for key, name, uses_struct, uses_feat in METHODS + [ORACLE]:
        parts = load_partitions(key)
        if parts is None:
            print(f"  [Bỏ qua] chưa có outputs/partitions/{key}.npy")
            continue
        name = name.format(**fmt)
        scores = pd.DataFrame([evaluate(y, p, edges) for p in parts])
        scores.insert(0, "method", name)
        scores.insert(1, "seed", C.SEEDS[:len(parts)])
        per_seed.append(scores)
        summary.append({
            "Phương pháp": name,
            "Cấu trúc": "x" if uses_struct else "",
            "Đặc trưng": "x" if uses_feat else "",
            "Số cụm": mean_std(scores["n_clusters"], 1),
            "NMI": mean_std(scores["NMI"]),
            "ARI": mean_std(scores["ARI"]),
            "Purity": mean_std(scores["Purity"]),
            "Modularity": mean_std(scores["Modularity"]),
            "_ari": scores["ARI"].mean(),
            "_oracle": key == ORACLE[0],
        })
        print(f"  {name:<50} ARI = {summary[-1]['ARI']}")

    table = (pd.DataFrame(summary)
             .sort_values(["_oracle", "_ari"], ascending=[True, False])
             .drop(columns=["_ari", "_oracle"]))
    table.loc[len(table)] = ["Mức nền: gán tất cả là web", "", "", "1", "0.0000", "0.0000",
                             f"{1 - y.mean():.4f}", "0.0000"]
    return table, pd.concat(per_seed, ignore_index=True)


def merge_by_majority(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Gán mỗi cụm về nghề chiếm đa số trong cụm (1 = ML nếu tỷ lệ ML > 50%)."""
    share_ml = pd.Series(y).groupby(labels).transform("mean").to_numpy()
    return (share_ml > 0.5).astype(int)


def majority_merge_analysis(y, meta):
    fmt = {"sel": meta.get("leiden_selected_resolution", "?"),
           "ora": meta.get("leiden_oracle_resolution", "?")}
    rows = []
    for key, name, _, _ in METHODS + [ORACLE]:
        parts = load_partitions(key)
        if parts is None:
            continue
        ari_raw, ari_merged, f1_ml, share_ml, n_ml_clusters = [], [], [], [], []
        for p in parts:
            merged = merge_by_majority(y, p)
            ari_raw.append(adjusted_rand_score(y, p))
            ari_merged.append(adjusted_rand_score(y, merged))
            f1_ml.append(f1_score(y, merged, zero_division=0))
            share_ml.append(merged.mean())
            ml_share_by_cluster = pd.Series(y).groupby(p).mean()
            n_ml_clusters.append(int((ml_share_by_cluster > 0.5).sum()))
        rows.append({
            "Phương pháp": name.format(**fmt),
            "ARI gốc": mean_std(ari_raw),
            "ARI sau gộp": mean_std(ari_merged),
            "F1 lớp ML": mean_std(f1_ml),
            "Số cụm đa số ML": mean_std(n_ml_clusters, 1),
            "Tỷ lệ gán ML": mean_std(share_ml),
            "_sort": (key == ORACLE[0], -np.mean(ari_merged)),
        })
        print(f"  {rows[-1]['Phương pháp']:<50} ARI sau gộp = {rows[-1]['ARI sau gộp']}")

    table = pd.DataFrame(rows)
    table = (table.assign(_o=table["_sort"].str[0], _a=table["_sort"].str[1])
                  .sort_values(["_o", "_a"]).drop(columns=["_sort", "_o", "_a"]))
    table.loc[len(table)] = ["Nhãn thật", "1.0000", "1.0000", "1.0000", "", f"{y.mean():.4f}"]
    return table


def linear_probe(y):
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, path in REPRESENTATIONS:
        if not path.exists():
            print(f"  [Bỏ qua] chưa có {path.name}")
            continue
        X = load_svd_features() if path == C.FEAT_SVD_FILE else np.load(path)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        rows.append({"Biểu diễn": name, "Số chiều": X.shape[1], "AUC (5-fold)": mean_std(auc)})
        print(f"  {name:<28} AUC = {rows[-1]['AUC (5-fold)']}")
    return pd.DataFrame(rows)


def main():
    C.ensure_dirs()
    y = load_labels()
    edges = load_edges()
    meta_path = C.RES_DIR / "rq2_structure_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    print("[1] Đánh giá các phương pháp phân cụm")
    table, per_seed = evaluate_all(y, edges, meta)
    print("\n" + table.to_string(index=False))
    save_table(table, "rq2_comparison",
               caption=(f"So sánh các phương pháp phân cụm với nhãn web/ML "
                        f"(trung bình ± độ lệch chuẩn qua {len(C.SEEDS)} seed)"),
               label="tab:rq2_comparison")
    save_table(per_seed.round(4), "rq2_per_seed")

    print("\n[2] Diễn giải: gộp mỗi cụm về nghề chiếm đa số (có dùng nhãn)")
    merged = majority_merge_analysis(y, meta)
    print("\n" + merged.to_string(index=False))
    save_table(merged, "rq2_majority_merge",
               caption=("Diễn giải: ARI và F1 sau khi gộp mỗi cụm về nghề chiếm đa số "
                        "(bước gộp có dùng nhãn, không dùng để so sánh phương pháp)"),
               label="tab:rq2_merge")

    print("\n[3] Linear probe: biểu diễn nào chứa thông tin về nhãn?")
    probe = linear_probe(y)
    print("\n" + probe.to_string(index=False))
    save_table(probe, "rq2_linear_probe",
               caption="AUC của Logistic Regression trên từng loại biểu diễn (có giám sát, chỉ để chẩn đoán)",
               label="tab:rq2_probe")

    print(f"\nĐã lưu kết quả RQ2 vào {C.RES_DIR} và {C.TABLE_DIR}")


if __name__ == "__main__":
    main()