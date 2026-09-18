"""
STEP 09 — RQ3 (phần 2): centrality có tương quan với số star/fork nhận được không?

Dữ liệu: mẫu star/fork từ step08 ghép với centrality từ step07 (theo id).
  - Mẫu chính: 5 nhóm bậc ngẫu nhiên (nhóm 0-4). Mỗi nhóm là 1/5 mạng và được lấy
    400 người, nên đây là mẫu tỷ lệ, đại diện cho toàn mạng (trừ top 200).
  - Nhóm 5: toàn bộ 200 người bậc cao nhất, dùng riêng cho phân tích influencer.

Phân tích:
  1. Tỷ lệ tài khoản còn tồn tại theo từng nhóm
  2. Tương quan Spearman + khoảng tin cậy 95% bootstrap (mẫu chính)
  3. Tương quan riêng phần, kiểm soát nhãn web/ML, số repo, tuổi tài khoản
  4. Hồi quy nhị thức âm: số star theo từng độ đo centrality (so sánh bằng AIC)
  5. Phát hiện influencer: tỷ lệ trùng top-K theo centrality và theo số star

Chạy: python src/step09_rq3_correlation.py
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as t_dist

import config as C
from utils import save_table

N_BOOT = 1000
TOP_K = 50
TOP_STRATUM = C.STAR_N_STRATA

PREDICTORS = [
    ("degree", "Bậc"),
    ("pagerank", "PageRank toàn cục"),
    ("eigenvector", "Eigenvector"),
    ("betweenness", "Betweenness"),
    ("core", "k-core"),
    ("pagerank_in", "PageRank trong cụm"),
    ("share_in", "Tỷ lệ kết nối trong cụm"),
    ("participation", "Hệ số tham gia"),
]
OUTCOMES = [("stars", "Số star"), ("forks", "Số fork"), ("followers", "Số follower (hiện tại)")]
CONTROLS = ["ml_target", "n_own_repos", "account_age_years"]


# ---------- Chuẩn bị dữ liệu ----------
def load_data():
    if not C.STARS_FILE.exists():
        raise RuntimeError(f"Chưa có {C.STARS_FILE}. Hãy chạy step08 trước.")
    cent_file = C.RES_DIR / "rq3_centrality.csv"
    if not cent_file.exists():
        raise RuntimeError("Chưa có rq3_centrality.csv. Hãy chạy step07 trước.")

    stars = pd.read_csv(C.STARS_FILE).drop_duplicates("id", keep="last")
    cent = pd.read_csv(cent_file)

    coverage = (stars.groupby("stratum")
                .agg(so_nguoi=("id", "size"), ty_le_con_ton_tai=("found", "mean"))
                .reset_index())
    coverage["ty_le_con_ton_tai"] = coverage["ty_le_con_ton_tai"].round(3)
    coverage.columns = ["Nhóm bậc", "Số người lấy mẫu", "Tỷ lệ tài khoản còn tồn tại"]

    df = (stars[stars["found"].astype(bool)]
          .drop(columns=["degree"])
          .merge(cent, on="id", how="inner"))
    created = pd.to_datetime(df["created_at"], utc=True)
    fetched = pd.to_datetime(df["fetched_at"], utc=True)
    df["account_age_years"] = (fetched - created).dt.days / 365.25
    for col in ["stars", "forks", "followers", "n_own_repos"]:
        df[col] = df[col].astype(int)
    return df, coverage


# ---------- Tương quan ----------
def spearman_with_ci(x: np.ndarray, y: np.ndarray, rng) -> tuple[float, float, float, float]:
    rho, p = spearmanr(x, y)
    n = len(x)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot[b] = spearmanr(x[idx], y[idx])[0]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(rho), float(lo), float(hi), float(p)


def partial_spearman(df: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    """Spearman riêng phần: xếp hạng mọi biến, loại ảnh hưởng của biến kiểm soát bằng hồi quy
    tuyến tính, rồi tính tương quan giữa hai phần dư."""
    ranks = df[[x, y] + CONTROLS].rank()
    Z = np.column_stack([np.ones(len(ranks)), ranks[CONTROLS].to_numpy()])

    def resid(col):
        v = ranks[col].to_numpy()
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    r, _ = pearsonr(resid(x), resid(y))
    dof = len(df) - 2 - len(CONTROLS)
    t = r * np.sqrt(dof / (1 - r ** 2))
    p = 2 * t_dist.sf(abs(t), dof)
    return float(r), float(p)


def correlation_tables(df: pd.DataFrame):
    rng = np.random.default_rng(0)
    simple, partial = [], []
    for col, name in PREDICTORS:
        row_s, row_p = {"Độ đo": name}, {"Độ đo": name}
        for out, out_name in OUTCOMES:
            rho, lo, hi, p = spearman_with_ci(df[col].to_numpy(), df[out].to_numpy(), rng)
            row_s[out_name] = f"{rho:.3f} [{lo:.3f}, {hi:.3f}]"
            r, pp = partial_spearman(df, col, out)
            row_p[out_name] = f"{r:.3f}" + ("*" if pp < 0.05 else "")
        simple.append(row_s)
        partial.append(row_p)
        print(f"  {name:<26} star: {row_s['Số star']} | riêng phần: {row_p['Số star']}")
    return pd.DataFrame(simple), pd.DataFrame(partial)


# ---------- Hồi quy nhị thức âm ----------
def zscore(v: pd.Series) -> pd.Series:
    return (v - v.mean()) / v.std(ddof=0)


def negative_binomial_models(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        import warnings
        import statsmodels.formula.api as smf
    except ImportError:
        print("  [Bỏ qua] chưa cài statsmodels: pip install statsmodels")
        return None

    data = df.copy()
    data["z_repos"] = zscore(np.log1p(data["n_own_repos"]))
    data["z_age"] = zscore(data["account_age_years"])
    base_formula = "stars ~ ml_target + z_repos + z_age"

    def fit(formula):
        model = smf.negativebinomial(formula, data, loglike_method="nb2")
        start = np.zeros(model.exog.shape[1] + 1)
        start[0] = np.log(data["stars"].mean() + 1)    # hệ số chặn
        start[-1] = 1.0                                # tham số phân tán alpha
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # ẩn cảnh báo trung gian khi tối ưu
            res = model.fit(start_params=start, method="nm", maxiter=10000, disp=0)
            res = model.fit(start_params=np.asarray(res.params), method="bfgs",
                            maxiter=2000, disp=0)
        converged = res.mle_retvals.get("converged", False)
        if not converged or not np.isfinite(res.llf) or not np.all(np.isfinite(res.bse)):
            raise RuntimeError("không hội tụ")
        return res

    try:
        base = fit(base_formula)
    except RuntimeError:
        print("  [Bỏ qua] mô hình cơ sở không hội tụ")
        return None
    print(f"  Mô hình cơ sở: AIC = {base.aic:.1f}, alpha = {base.params['alpha']:.3f}")

    rows = [{"Mô hình": "Chỉ biến kiểm soát", "IRR (1 độ lệch chuẩn)": "",
             "KTC 95%": "", "p-value": "", "AIC": f"{base.aic:.1f}",
             "ΔAIC so với cơ sở": "0.0", "Pseudo R²": f"{base.prsquared:.4f}"}]

    for col, name in PREDICTORS:
        raw = data[col]
        if col in ("share_in", "participation"):
            data["x"] = zscore(raw)
        else:
            eps = raw[raw > 0].min()                   # giá trị dương nhỏ nhất của độ đo
            data["x"] = zscore(np.log(raw + eps))      # log nhất quán cho mọi thang đo
        try:
            m = fit(base_formula + " + x")
        except Exception:
            print(f"  [{name}] không hội tụ, bỏ qua")
            continue
        lo, hi = np.exp(m.conf_int().loc["x"])
        rows.append({
            "Mô hình": f"+ {name}",
            "IRR (1 độ lệch chuẩn)": f"{np.exp(m.params['x']):.3f}",
            "KTC 95%": f"[{lo:.3f}, {hi:.3f}]",
            "p-value": f"{m.pvalues['x']:.2g}",
            "AIC": f"{m.aic:.1f}",
            "ΔAIC so với cơ sở": f"{m.aic - base.aic:.1f}",
            "Pseudo R²": f"{m.prsquared:.4f}",
        })
        print(f"  {name:<26} IRR = {rows[-1]['IRR (1 độ lệch chuẩn)']} {rows[-1]['KTC 95%']} "
              f"| p = {rows[-1]['p-value']} | ΔAIC = {rows[-1]['ΔAIC so với cơ sở']}")
    return pd.DataFrame(rows)


# ---------- Phát hiện influencer ----------
def influencer_overlap(df: pd.DataFrame) -> pd.DataFrame:
    top_stars = set(df.nlargest(TOP_K, "stars")["id"])
    top_followers = set(df.nlargest(TOP_K, "followers")["id"])
    rows = []
    for col, name in PREDICTORS:
        top_c = set(df.nlargest(TOP_K, col)["id"])
        rows.append({
            "Độ đo": name,
            f"Trùng top {TOP_K} star": f"{len(top_c & top_stars) / TOP_K:.2f}",
            f"Trùng top {TOP_K} follower": f"{len(top_c & top_followers) / TOP_K:.2f}",
        })
    rows.append({"Độ đo": "Mức ngẫu nhiên",
                 f"Trùng top {TOP_K} star": f"{TOP_K / len(df):.2f}",
                 f"Trùng top {TOP_K} follower": f"{TOP_K / len(df):.2f}"})
    return pd.DataFrame(rows)


def main():
    C.ensure_dirs()
    df, coverage = load_data()

    print("[1] Tỷ lệ tài khoản còn tồn tại")
    print(coverage.to_string(index=False))
    save_table(coverage, "rq3_coverage",
               caption="Mẫu thu thập star/fork và tỷ lệ tài khoản còn tồn tại",
               label="tab:rq3_coverage")

    main_df = df[df["stratum"] < TOP_STRATUM].reset_index(drop=True)
    top_df = df[df["stratum"] == TOP_STRATUM].reset_index(drop=True)
    print(f"\nMẫu chính: {len(main_df)} người | nhóm top: {len(top_df)} người")
    print(f"Số star (mẫu chính): trung vị {main_df['stars'].median():.0f}, "
          f"trung bình {main_df['stars'].mean():.1f}, lớn nhất {main_df['stars'].max()}")

    print(f"\n[2-3] Tương quan Spearman (mẫu chính, bootstrap {N_BOOT} lần)")
    simple, partial = correlation_tables(main_df)
    save_table(simple, "rq3_spearman",
               caption="Tương quan Spearman giữa centrality và mức độ ảnh hưởng (KTC 95% bootstrap)",
               label="tab:rq3_spearman")
    save_table(partial, "rq3_partial_spearman",
               caption=("Tương quan Spearman riêng phần, kiểm soát nhãn web/ML, số repo và "
                        "tuổi tài khoản (* p < 0,05)"),
               label="tab:rq3_partial")

    print("\n[4] Hồi quy nhị thức âm cho số star (mẫu chính)")
    nb = negative_binomial_models(main_df)
    if nb is not None:
        save_table(nb, "rq3_negative_binomial",
                   caption=("Hồi quy nhị thức âm cho số star; mỗi mô hình thêm một độ đo "
                            "centrality (đã chuẩn hóa) vào các biến kiểm soát"),
                   label="tab:rq3_nb")

    print(f"\n[5] Nhóm top 200 bậc cao nhất: tương quan với số star")
    rows = []
    for col, name in PREDICTORS:
        rho, p = spearmanr(top_df[col], top_df["stars"])
        rows.append({"Độ đo": name, "Spearman với số star": f"{rho:.3f}", "p-value": f"{p:.2g}"})
    within_top = pd.DataFrame(rows)
    print(within_top.to_string(index=False))
    save_table(within_top, "rq3_within_top",
               caption="Tương quan trong nhóm 200 người có bậc cao nhất",
               label="tab:rq3_within_top")

    print(f"\n[6] Phát hiện influencer: trùng top {TOP_K} (toàn bộ mẫu)")
    overlap = influencer_overlap(df)
    print(overlap.to_string(index=False))
    save_table(overlap, "rq3_influencer_overlap",
               caption=f"Tỷ lệ trùng giữa top {TOP_K} theo centrality và theo số star, follower",
               label="tab:rq3_influencer")

    print(f"\nĐã lưu kết quả RQ3 vào {C.RES_DIR} và {C.TABLE_DIR}")


if __name__ == "__main__":
    main()