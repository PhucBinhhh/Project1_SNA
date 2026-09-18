"""
STEP 01 — Kiểm tra dữ liệu gốc MUSAE GitHub.

Bước này CHỈ ĐỌC, không tải, không sửa, không ghi đè dữ liệu gốc.
Nếu thiếu dữ liệu, hãy tải thủ công từ:
    https://snap.stanford.edu/data/github-social.html
rồi giải nén 3 file vào data/git_web_ml/

Đầu ra: outputs/results/data_check.csv (bảng kết quả kiểm tra, dùng cho báo cáo)
Chạy:   python src/step01_check_data.py
"""
import json
import sys

import pandas as pd

import config as C

N_FEATURE_DIMS = 4005   # theo mô tả dataset


def check_files_exist() -> None:
    missing = [f.name for f in (C.EDGES_FILE, C.TARGET_FILE, C.FEATURES_FILE)
               if not f.exists()]
    if missing:
        sys.exit(
            f"THIẾU FILE: {', '.join(missing)}\n"
            f"Tải tại https://snap.stanford.edu/data/github-social.html "
            f"và giải nén vào: {C.RAW_DIR}"
        )


def main():
    C.ensure_dirs()
    check_files_exist()

    edges = pd.read_csv(C.EDGES_FILE)
    target = pd.read_csv(C.TARGET_FILE)
    with open(C.FEATURES_FILE, encoding="utf-8") as f:
        features = json.load(f)

    checks = []   # mỗi phần tử: (hạng mục, giá trị, đạt hay không)

    def add(item, value, ok=True):
        checks.append({"hang_muc": item, "gia_tri": value, "dat": ok})

    # ---- 1. Bảng nhãn (target) ----
    need_cols = {"id", "name", "ml_target"}
    add("Cột của bảng nhãn", ", ".join(target.columns), need_cols <= set(target.columns))
    n_nodes = len(target)
    add("Số node", n_nodes, n_nodes == 37700)
    add("id liên tục từ 0 đến N-1",
        "có" if set(target["id"]) == set(range(n_nodes)) else "không",
        set(target["id"]) == set(range(n_nodes)))
    add("Nhãn chỉ gồm 0 và 1", sorted(target["ml_target"].unique().tolist()),
        set(target["ml_target"]) <= {0, 1})
    add("Giá trị thiếu trong bảng nhãn", int(target.isna().sum().sum()),
        target.isna().sum().sum() == 0)
    add("Tên tài khoản trùng nhau", int(target["name"].duplicated().sum()),
        target["name"].duplicated().sum() == 0)
    p_ml = target["ml_target"].mean()
    add("Tỷ lệ web / ML", f"{1 - p_ml:.4f} / {p_ml:.4f}")

    # ---- 2. Bảng cạnh (edges) ----
    add("Số cạnh", len(edges), len(edges) == 289003)
    ids = set(target["id"])
    edge_ids = set(edges["id_1"]) | set(edges["id_2"])
    add("Mọi id trong cạnh đều có nhãn", "có" if edge_ids <= ids else "không",
        edge_ids <= ids)
    add("Số node không có cạnh nào", len(ids - edge_ids))
    n_loops = int((edges["id_1"] == edges["id_2"]).sum())
    add("Số khuyên (tự nối)", n_loops, n_loops == 0)
    lo = edges[["id_1", "id_2"]].min(axis=1)
    hi = edges[["id_1", "id_2"]].max(axis=1)
    n_dup = int(pd.DataFrame({"a": lo, "b": hi}).duplicated().sum())
    add("Số cạnh trùng lặp (kể cả đảo chiều)", n_dup, n_dup == 0)

    # ---- 3. Đặc trưng (features) ----
    feat_ids = {int(k) for k in features}
    add("Số node có đặc trưng", len(feat_ids), feat_ids == ids)
    all_idx = [i for v in features.values() for i in v]
    add("Chỉ số đặc trưng nằm trong [0, 4004]",
        f"{min(all_idx)} đến {max(all_idx)}",
        min(all_idx) >= 0 and max(all_idx) < N_FEATURE_DIMS)
    n_feat = pd.Series([len(v) for v in features.values()])
    add("Số đặc trưng bằng 1 mỗi node (min / trung vị / max)",
        f"{n_feat.min()} / {int(n_feat.median())} / {n_feat.max()}")

    # ---- In và lưu ----
    report = pd.DataFrame(checks)
    report["dat"] = report["dat"].map({True: "OK", False: "LỖI"})
    print(report.to_string(index=False))

    out = C.RES_DIR / "data_check.csv"
    report.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu: {out}")

    if (report["dat"] == "LỖI").any():
        sys.exit("\nCÓ LỖI trong dữ liệu. Xem các dòng 'LỖI' ở trên trước khi chạy tiếp.")
    print("Dữ liệu hợp lệ. Chạy tiếp: python src/step02_preprocess.py")


if __name__ == "__main__":
    main()