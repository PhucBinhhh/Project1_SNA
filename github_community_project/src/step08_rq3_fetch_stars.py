"""
STEP 08 — RQ3: lấy số star/fork THẬT cho một mẫu người dùng trong chính mạng MUSAE.

- Dùng cột `name` (tên tài khoản GitHub) trong musae_git_target.csv.
- Lấy mẫu phân tầng theo bậc (5 nhóm x 400 người), cộng thêm TOÀN BỘ 200 người
  bậc cao nhất (nhóm 5) để phân tích influencer.
- Mỗi người gọi 2 loại API:
    /users/{name}         -> followers, following, public_repos, created_at
    /users/{name}/repos   -> cộng dồn star/fork trên repo do họ tạo (bỏ repo fork)
- Lưu dần vào CSV: bị ngắt giữa chừng thì chạy lại sẽ tiếp tục từ chỗ dừng.
- Tự chờ khi hết lượt API (5.000 request/giờ với token).

Chạy thử (không gọi API, chỉ xem mẫu):  python src/step08_rq3_fetch_stars.py --dry-run
Chạy thật:                              python src/step08_rq3_fetch_stars.py

LƯU Ý ĐẠO ĐỨC: file kết quả chỉ lưu `id`, không lưu tên tài khoản.
"""
import argparse
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

import config as C

API = "https://api.github.com"
SAVE_EVERY = 25
COLUMNS = ["id", "stratum", "degree", "found", "followers", "following",
           "public_repos", "created_at", "n_own_repos", "stars", "forks",
           "fetched_at"]


# ---------- Lấy mẫu ----------
def build_sample() -> pd.DataFrame:
    edges = pd.read_csv(C.EDGES_FILE)
    target = pd.read_csv(C.TARGET_FILE)
    if "name" not in target.columns:
        raise RuntimeError(
            "musae_git_target.csv không có cột 'name'. Hãy khôi phục dữ liệu gốc "
            "từ bản sao lưu hoặc tải lại từ SNAP."
        )

    degree = pd.concat([edges["id_1"], edges["id_2"]]).value_counts()
    df = target.set_index("id")
    df["degree"] = degree.reindex(df.index).fillna(0).astype(int)
    df["stratum"] = pd.qcut(df["degree"].rank(method="first"),
                            q=C.STAR_N_STRATA, labels=False)

    # Nhóm đặc biệt: top-N người bậc cao nhất (ứng viên influencer), lấy toàn bộ
    top = df.nlargest(C.STAR_TOP_N, "degree").assign(stratum=C.STAR_N_STRATA)
    rest = df.drop(top.index)

    per_stratum = C.STAR_SAMPLE_SIZE // C.STAR_N_STRATA
    parts = [g.sample(n=per_stratum, random_state=C.STAR_SAMPLE_SEED)
             for _, g in rest.groupby("stratum")] + [top]
    sample = pd.concat(parts).rename_axis("id").reset_index()
    return sample[["id", "name", "stratum", "degree"]]


# ---------- Gọi API ----------
class GitHubClient:
    def __init__(self, token: str):
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def get(self, url: str, params: dict | None = None):
        """Trả về JSON, hoặc None nếu 404. Tự chờ khi hết lượt, thử lại khi lỗi mạng."""
        for attempt in range(5):
            try:
                r = self.s.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                wait = 10 * (attempt + 1)
                print(f"  Lỗi mạng ({e.__class__.__name__}), thử lại sau {wait}s")
                time.sleep(wait)
                continue

            if r.status_code == 404:
                return None
            if r.status_code in (403, 429) and r.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - int(time.time()), 0) + 5
                print(f"  Hết lượt API, chờ {wait // 60} phút {wait % 60} giây...")
                time.sleep(wait)
                continue
            if r.status_code in (403, 429):          # giới hạn phụ (secondary rate limit)
                wait = int(r.headers.get("Retry-After", 60))
                print(f"  Bị giới hạn tạm thời, chờ {wait}s")
                time.sleep(wait)
                continue
            if r.status_code == 401:
                raise RuntimeError("Token không hợp lệ hoặc đã hết hạn (401). Kiểm tra file .env")
            if r.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Thất bại sau 5 lần thử: {url}")

    def user_stats(self, name: str) -> dict:
        user = self.get(f"{API}/users/{name}")
        if user is None:
            return {"found": False}

        stars = forks = n_own = 0
        page = 1
        while True:
            repos = self.get(f"{API}/users/{name}/repos",
                             params={"type": "owner", "per_page": 100, "page": page})
            if not repos:
                break
            for repo in repos:
                if not repo.get("fork"):
                    stars += repo.get("stargazers_count", 0)
                    forks += repo.get("forks_count", 0)
                    n_own += 1
            if len(repos) < 100:
                break
            page += 1

        return {
            "found": True,
            "followers": user.get("followers"),
            "following": user.get("following"),
            "public_repos": user.get("public_repos"),
            "created_at": user.get("created_at"),
            "n_own_repos": n_own,
            "stars": stars,
            "forks": forks,
        }


# ---------- Chương trình chính ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Chỉ tạo và in mẫu, không gọi API")
    args = parser.parse_args()

    C.ensure_dirs()
    sample = build_sample()
    print(f"Mẫu: {len(sample)} người ({C.STAR_N_STRATA} nhóm ngẫu nhiên + nhóm top {C.STAR_TOP_N})")
    print(sample.groupby("stratum")["degree"].agg(["count", "min", "max"]).to_string())

    if args.dry_run:
        return

    load_dotenv(C.ENV_FILE)
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(f"Không tìm thấy GITHUB_TOKEN trong {C.ENV_FILE}")
    client = GitHubClient(token)

    done_ids = set()
    if C.STARS_FILE.exists():
        done_ids = set(pd.read_csv(C.STARS_FILE, usecols=["id"])["id"])
    todo = sample[~sample["id"].isin(done_ids)]
    print(f"Đã có {len(done_ids)}, còn {len(todo)} người cần lấy\n")

    buffer, start = [], time.time()
    for i, row in enumerate(todo.itertuples(index=False), 1):
        stats = client.user_stats(row.name)
        buffer.append({
            "id": row.id, "stratum": row.stratum, "degree": row.degree,
            **stats,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

        if i % SAVE_EVERY == 0 or i == len(todo):
            out = pd.DataFrame(buffer).reindex(columns=COLUMNS)
            out.to_csv(C.STARS_FILE, mode="a", index=False,
                       header=not C.STARS_FILE.exists())
            buffer = []
            rate = i / (time.time() - start)
            eta = (len(todo) - i) / rate / 60 if rate else 0
            print(f"  {i}/{len(todo)} | {rate * 60:.0f} người/phút | còn khoảng {eta:.0f} phút")

    result = pd.read_csv(C.STARS_FILE)
    print(f"\nHOÀN TẤT: {len(result)} người, tìm thấy {result['found'].mean():.1%} tài khoản")
    print(f"Đã lưu: {C.STARS_FILE}")


if __name__ == "__main__":
    main()