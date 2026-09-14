#!/bin/bash
# ============================================================
# NHÁNH B — chạy bởi Thành viên B, ĐỘC LẬP với nhánh chính,
# chỉ cần file data/processed/graph.gpickle đã có (do Leader tạo ở step3)
# để chạy Nhánh A (centrality nội tại) trong step7.
# Chạy: bash run_branch_b.sh
# ============================================================
set -e

echo ">>> [1/2] Crawl dữ liệu GitHub thật (cần GITHUB_TOKEN trong file .env)..."
echo "    Lưu ý: bước này có thể mất 1-3 giờ tùy rate-limit và MAX_USERS đã cấu hình."
python src/step2_crawl_github_supplement.py

echo ">>> [2/2] RQ3 — Tính centrality + tương quan Spearman với star/fork thật..."
python src/step7_centrality_correlation.py

echo ">>> Hoàn tất Nhánh B. Kết quả nằm trong outputs/"
echo "Báo Leader để chạy step8_visualize_report.py tổng hợp báo cáo cuối."
