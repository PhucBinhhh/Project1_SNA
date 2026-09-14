"""
STEP 8 — Trực quan hóa tổng hợp toàn bộ kết quả 3 RQ, xuất ra outputs/ để đưa vào báo cáo.

Chạy SAU CÙNG, khi đã có đủ output của step4, step5, step6, step7.
"""
import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def plot_rq1_cluster_distribution():
    path = os.path.join(OUT_DIR, "leiden_cluster_distribution.csv")
    if not os.path.exists(path):
        print("[Bỏ qua RQ1 plot] Chưa có leiden_cluster_distribution.csv")
        return
    df = pd.read_csv(path).head(15)  # top 15 cụm lớn nhất

    fig, ax = plt.subplots(figsize=(10, 6))
    df_sorted = df.sort_values("n_nodes")
    ax.barh(df_sorted["cluster"].astype(str), df_sorted["pct_web"], label="% Web dev", color="#4C72B0")
    ax.barh(df_sorted["cluster"].astype(str), df_sorted["pct_ml"], left=df_sorted["pct_web"],
            label="% ML dev", color="#DD8452")
    ax.set_xlabel("% trong cụm")
    ax.set_ylabel("Cluster ID (Leiden)")
    ax.set_title("RQ1: Tỷ lệ Web/ML developer trong từng cụm (top 15 cụm lớn nhất)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rq1_cluster_label_distribution.png"), dpi=150)
    plt.close()
    print("Đã lưu: rq1_cluster_label_distribution.png")


def plot_rq2_comparison():
    path = os.path.join(OUT_DIR, "rq2_comparison_table.csv")
    if not os.path.exists(path):
        print("[Bỏ qua RQ2 plot] Chưa có rq2_comparison_table.csv")
        return
    df = pd.read_csv(path)
    metrics = ["NMI", "ARI", "Purity"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.25
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m], width, label=m)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["method"])
    ax.set_ylabel("Điểm số")
    ax.set_title("RQ2: So sánh Louvain vs Leiden vs GCN")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rq2_method_comparison.png"), dpi=150)
    plt.close()
    print("Đã lưu: rq2_method_comparison.png")


def plot_rq3_correlation():
    path = os.path.join(OUT_DIR, "supplement_users_with_centrality.csv")
    if not os.path.exists(path):
        print("[Bỏ qua RQ3 plot] Chưa có supplement_users_with_centrality.csv (cần chạy Nhánh B ở step7)")
        return
    df = pd.read_csv(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(data=df, x="eigenvector_centrality", y="total_stars_received", ax=axes[0],
                scatter_kws={"alpha": 0.5})
    axes[0].set_title("Eigenvector Centrality vs Total Stars")

    sns.regplot(data=df, x="degree_centrality", y="total_forks_received", ax=axes[1],
                scatter_kws={"alpha": 0.5}, color="orange")
    axes[1].set_title("Degree Centrality vs Total Forks")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rq3_centrality_vs_starfork.png"), dpi=150)
    plt.close()
    print("Đã lưu: rq3_centrality_vs_starfork.png")


def generate_summary_report():
    lines = ["# BÁO CÁO TỔNG HỢP KẾT QUẢ\n"]

    hp_path = os.path.join(OUT_DIR, "rq1_homophily_summary.txt")
    if os.path.exists(hp_path):
        lines.append("## RQ1 — Homophily\n")
        with open(hp_path) as f:
            lines.append("```\n" + f.read() + "```\n")

    rq2_path = os.path.join(OUT_DIR, "rq2_comparison_table.csv")
    if os.path.exists(rq2_path):
        lines.append("## RQ2 — So sánh phương pháp\n")
        lines.append(pd.read_csv(rq2_path).to_markdown(index=False) + "\n")

    rq3_path = os.path.join(OUT_DIR, "rq3_real_star_fork_correlation.csv")
    if os.path.exists(rq3_path):
        lines.append("## RQ3 — Tương quan Centrality vs Star/Fork (dữ liệu thật)\n")
        lines.append(pd.read_csv(rq3_path).to_markdown(index=False) + "\n")

    with open(os.path.join(OUT_DIR, "SUMMARY_REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nĐã tạo báo cáo tổng hợp: {os.path.join(OUT_DIR, 'SUMMARY_REPORT.md')}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_rq1_cluster_distribution()
    plot_rq2_comparison()
    plot_rq3_correlation()
    generate_summary_report()


if __name__ == "__main__":
    main()
