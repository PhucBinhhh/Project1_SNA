#!/bin/bash
# ============================================================
# NHÁNH CHÍNH — chạy bởi Leader + Thành viên A
# Bao gồm: download MUSAE -> preprocess -> RQ1 (homophily+community) -> RQ2 (GCN+evaluate)
# Chạy: bash run_main_branch.sh
# ============================================================
set -e  # dừng ngay nếu có bước lỗi

echo ">>> [1/6] Tải dataset MUSAE GitHub..."
python src/step1_download_musae.py

echo ">>> [2/6] Tiền xử lý dữ liệu (build graph + SVD feature)..."
python src/step3_preprocess.py

echo ">>> [3/6] RQ1 — Homophily + Louvain/Leiden community detection..."
python src/step4_homophily_community.py

echo ">>> [4/6] RQ2 phần 1 — Train GCN (VGAE) + KMeans..."
python src/step5_gcn_clustering.py

echo ">>> [5/6] RQ2 phần 2 — So sánh Louvain/Leiden/GCN (NMI/ARI/Purity)..."
python src/step6_evaluate.py

echo ">>> [6/6] Hoàn tất nhánh chính. Kết quả nằm trong outputs/"
echo "Bây giờ đợi Nhánh B (RQ3) xong rồi chạy: python src/step8_visualize_report.py"
