"""
STEP 10 — Vẽ các hình thống kê cho báo cáo (xuất PDF vào report/figures/).

  fig_degree_distribution      : phân bố bậc (thang log-log)            -> mục Dữ liệu
  fig_rq1_permutation          : kiểm định hoán vị của homophily         -> RQ1
  fig_rq2_resolution_sweep     : modularity và ARI theo resolution       -> RQ2
  fig_rq3_degree_vs_stars      : quan hệ giữa bậc và số star             -> RQ3
  fig_rq3_delta_aic            : mức giải thích số star của từng độ đo   -> RQ3
  fig_rq3_topk_overlap         : hiệu quả sàng lọc influencer            -> RQ3

Hình nào thiếu dữ liệu đầu vào thì được bỏ qua kèm thông báo.
Chạy: python src/step10_figures.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # hỗ trợ tiếng Việt
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})
WEB, ML, ACCENT = "#1F4E9C", "#E8730C", "#444444"
ONE_COL, TWO_COL = (3.5, 2.6), (7.0, 2.8)   # kích thước cột đơn / cột đôi của IEEE


def save(fig, name: str) -> None:
    path = C.FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Đã lưu: {path.name}")


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


# ---------- 1. Phân bố bậc ----------
def fig_degree_distribution():
    if not C.EDGE_ARRAY_FILE.exists():
        return print("  [Bỏ qua] chưa có edges.npy — hãy chạy step02")
    edges = np.load(C.EDGE_ARRAY_FILE)
    deg = np.bincount(edges.ravel())
    counts = pd.Series(deg[deg > 0]).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=ONE_COL)
    ax.scatter(counts.index, counts.values, s=6, color=WEB, alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Bậc")
    ax.set_ylabel("Số node")
    ax.axvline(np.median(deg[deg > 0]), color=ACCENT, ls="--", lw=1)
    ax.text(np.median(deg[deg > 0]) * 1.2, counts.max() * 0.4,
            f"trung vị = {int(np.median(deg[deg > 0]))}", fontsize=8, color=ACCENT)
    save(fig, "fig_degree_distribution")


# ---------- 2. Kiểm định hoán vị (RQ1) ----------
def fig_rq1_permutation():
    path = C.RES_DIR / "rq1_permutation_null.npy"
    summary = read_csv(C.RES_DIR / "rq1_summary.csv")
    if not path.exists() or summary is None:
        return print("  [Bỏ qua] chưa có kết quả RQ1 — hãy chạy step03")
    null = np.load(path)
    observed = float(summary.iloc[0, 1])

    fig, ax = plt.subplots(figsize=ONE_COL)
    ax.hist(null, bins=40, color="#BBBBBB", edgecolor="white")
    ax.axvline(observed, color=ML, lw=2)
    ax.annotate(f"quan sát = {observed:.3f}", xy=(observed, ax.get_ylim()[1] * 0.8),
                xytext=(-8, 0), textcoords="offset points", ha="right",
                color=ML, fontsize=8)
    ax.set_xlabel("Edge homophily")
    ax.set_ylabel(f"Số lần trong {len(null)} lần xáo nhãn")
    save(fig, "fig_rq1_permutation")


# ---------- 3. Quét resolution (RQ2) ----------
def fig_rq2_resolution_sweep():
    df = read_csv(C.RES_DIR / "rq2_leiden_resolution_sweep.csv")
    if df is None:
        return print("  [Bỏ qua] chưa có bảng quét resolution — hãy chạy step04")
    df = df.sort_values(df.columns[0])
    gamma, mod, ari, std = df.iloc[:, 0], df["Modularity"], df["ARI"], df.iloc[:, 6]

    fig, ax = plt.subplots(figsize=ONE_COL)
    ax.plot(gamma, mod, "o-", color=WEB, lw=1.5, ms=4, label="Modularity")
    ax.set_xlabel(r"Tham số resolution $\gamma$")
    ax.set_ylabel("Modularity", color=WEB)
    ax.tick_params(axis="y", labelcolor=WEB)
    ax.axvline(gamma[mod.idxmax()], color=WEB, ls=":", lw=1)

    ax2 = ax.twinx()
    ax2.errorbar(gamma, ari, yerr=std, fmt="s-", color=ML, lw=1.5, ms=4,
                 capsize=2, label="ARI")
    ax2.set_ylabel("ARI (so với nhãn web/ML)", color=ML)
    ax2.tick_params(axis="y", labelcolor=ML)
    ax2.axvline(gamma[ari.idxmax()], color=ML, ls=":", lw=1)
    ax2.grid(False)

    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, ["Modularity", "ARI"], fontsize=8, loc="upper center")
    save(fig, "fig_rq2_resolution_sweep")


# ---------- Dữ liệu chung cho RQ3 ----------
def load_rq3():
    stars = read_csv(C.STARS_FILE)
    cent = read_csv(C.RES_DIR / "rq3_centrality.csv")
    if stars is None or cent is None:
        return None
    df = (stars[stars["found"].astype(bool)].drop(columns=["degree"])
          .merge(cent, on="id", how="inner"))
    return df[df["stratum"] < C.STAR_N_STRATA]     # chỉ dùng mẫu chính


# ---------- 4. Bậc và số star (RQ3) ----------
def fig_rq3_degree_vs_stars():
    df = load_rq3()
    if df is None:
        return print("  [Bỏ qua] chưa có dữ liệu star/centrality — hãy chạy step07, step08")
    x, y = df["degree"].to_numpy(float), df["stars"].to_numpy(float)

    fig, ax = plt.subplots(figsize=ONE_COL)
    for label, color, mask in [("Web", WEB, df["ml_target"] == 0),
                               ("ML", ML, df["ml_target"] == 1)]:
        ax.scatter(x[mask] + 0.5, y[mask] + 0.5, s=5, alpha=0.4,
                   color=color, label=label, linewidths=0)

    # đường xu hướng trên thang log
    ok = (x > 0) & (y > 0)
    b, a = np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)
    xs = np.logspace(0, np.log10(x.max()), 50)
    ax.plot(xs, 10 ** (a + b * np.log10(xs)), color=ACCENT, lw=1.5)

    from scipy.stats import spearmanr
    rho = spearmanr(x, y)[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Bậc (+0,5)")
    ax.set_ylabel("Số star (+0,5)")
    ax.set_title(f"Spearman $\\rho$ = {rho:.3f}", fontsize=9)
    ax.legend(fontsize=8, markerscale=2, loc="upper left")
    save(fig, "fig_rq3_degree_vs_stars")


# ---------- 5. ΔAIC của từng độ đo (RQ3) ----------
def fig_rq3_delta_aic():
    df = read_csv(C.RES_DIR / "rq3_negative_binomial.csv")
    if df is None:
        return print("  [Bỏ qua] chưa có kết quả hồi quy — hãy chạy step09")
    df = df[df["Mô hình"].str.startswith("+")].copy()
    df["ten"] = df["Mô hình"].str.lstrip("+ ")
    df["daic"] = pd.to_numeric(df["ΔAIC so với cơ sở"], errors="coerce")
    df = df.dropna(subset=["daic"]).sort_values("daic")
    if df.empty:
        return print("  [Bỏ qua] bảng hồi quy không có ΔAIC hợp lệ")

    fig, ax = plt.subplots(figsize=ONE_COL)
    ax.barh(df["ten"], -df["daic"], color=WEB, height=0.65)
    ax.set_xlabel(r"Mức cải thiện độ phù hợp ($-\Delta$AIC)")
    ax.set_xlim(0, (-df["daic"]).max() * 1.15)
    ax.invert_yaxis()
    for y_pos, v in enumerate(-df["daic"]):
        ax.text(v, y_pos, f" {v:.0f}", va="center", fontsize=7)
    save(fig, "fig_rq3_delta_aic")


# ---------- 6. Hiệu quả sàng lọc influencer (RQ3) ----------
def fig_rq3_topk_overlap():
    df = read_csv(C.RES_DIR / "rq3_influencer_overlap.csv")
    if df is None:
        return print("  [Bỏ qua] chưa có bảng trùng top-K — hãy chạy step09")
    col = df.columns[1]
    baseline = float(df[df["Độ đo"] == "Mức ngẫu nhiên"][col].iloc[0])
    df = df[df["Độ đo"] != "Mức ngẫu nhiên"].copy()
    df[col] = df[col].astype(float)
    df = df.sort_values(col, ascending=False)

    fig, ax = plt.subplots(figsize=ONE_COL)
    ax.bar(range(len(df)), df[col], color=WEB, width=0.65)
    ax.axhline(baseline, color=ML, ls="--", lw=1.2,
               label=f"mức ngẫu nhiên = {baseline:.2f}")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["Độ đo"], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(col)
    ax.legend(fontsize=8)
    save(fig, "fig_rq3_topk_overlap")


def main():
    C.ensure_dirs()
    print(f"Xuất hình vào: {C.FIG_DIR}")
    for func in (fig_degree_distribution, fig_rq1_permutation, fig_rq2_resolution_sweep,
                 fig_rq3_degree_vs_stars, fig_rq3_delta_aic, fig_rq3_topk_overlap):
        try:
            func()
        except Exception as e:
            print(f"  [Lỗi] {func.__name__}: {e.__class__.__name__}: {e}")
    print("Xong.")


if __name__ == "__main__":
    main()