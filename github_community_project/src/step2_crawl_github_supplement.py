"""
STEP 2 — Crawl bổ sung dữ liệu GitHub THẬT cho RQ3.

LÝ DO CẦN BƯỚC NÀY: dataset MUSAE đã ẩn danh hóa (chỉ có ID số), không có username
thật để tra star/fork. Script này thu thập một MẪU MỚI, ĐỘC LẬP gồm các user GitHub
thật, có đầy đủ: follower/following (để dựng mini-network) + star/fork trên các
repo họ sở hữu (ground-truth cho RQ3).

Yêu cầu: biến môi trường GITHUB_TOKEN (Personal Access Token, không cần quyền đặc biệt,
chỉ cần "public_repo" read) để tránh rate-limit thấp (60 request/giờ nếu không có token,
5000 request/giờ nếu có token).

Chiến lược lấy mẫu: bắt đầu từ danh sách "seed" user (ví dụ trending developers, hoặc
GitHub Search API tìm user có >N follower để đảm bảo có đủ dữ liệu star/fork), sau đó
mở rộng qua follower/following của họ (kiểu snowball sampling), giới hạn ~500-1000 user
để tránh vượt rate-limit / tốn quá nhiều thời gian.
"""
import os
import time
import json
from collections import deque

from github import Github, RateLimitExceededException
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "supplement")
MAX_USERS = 800          # điều chỉnh tuỳ thời gian/rate-limit bạn có
SEED_QUERY = "followers:>500 repos:>5"   # user hoạt động tích cực, đủ dữ liệu star/fork


def safe_call(fn, *args, **kwargs):
    """Gọi API, tự chờ nếu bị rate-limit."""
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededException:
            print("Rate limit — tạm nghỉ 60s...")
            time.sleep(60)
        except Exception as e:
            print(f"Lỗi bỏ qua: {e}")
            return None


def crawl(max_users: int = MAX_USERS):
    if not TOKEN:
        raise RuntimeError(
            "Chưa có GITHUB_TOKEN. Tạo file .env với dòng: GITHUB_TOKEN=ghp_xxx"
        )
    g = Github(TOKEN, per_page=100)

    os.makedirs(OUT_DIR, exist_ok=True)
    users_data = []
    edges = []  # (source_login, target_login) = source follows target
    visited = set()

    # Bước 1: tìm seed users
    seed_users = safe_call(g.search_users, SEED_QUERY)
    queue = deque()
    for u in seed_users[:50]:
        queue.append(u.login)

    pbar = tqdm(total=max_users, desc="Crawling GitHub users")
    while queue and len(visited) < max_users:
        login = queue.popleft()
        if login in visited:
            continue
        visited.add(login)

        user = safe_call(g.get_user, login)
        if user is None:
            continue

        # Tổng số star nhận được = tổng stargazers_count trên các repo user sở hữu (không fork)
        total_stars, total_forks, n_repos = 0, 0, 0
        repos = safe_call(user.get_repos)
        if repos is not None:
            for repo in repos:
                if repo.fork:
                    continue
                total_stars += repo.stargazers_count
                total_forks += repo.forks_count
                n_repos += 1

        users_data.append({
            "login": login,
            "followers": user.followers,
            "following": user.following,
            "public_repos": user.public_repos,
            "total_stars_received": total_stars,
            "total_forks_received": total_forks,
            "n_original_repos": n_repos,
        })

        # Mở rộng snowball qua follower (giới hạn 20 người/user để tránh nổ quá nhanh)
        followers = safe_call(user.get_followers)
        if followers is not None:
            for i, f in enumerate(followers):
                if i >= 20:
                    break
                edges.append((f.login, login))
                if f.login not in visited:
                    queue.append(f.login)

        pbar.update(1)
        time.sleep(0.2)  # lịch sự với API

    pbar.close()

    with open(os.path.join(OUT_DIR, "supplement_users.json"), "w") as f:
        json.dump(users_data, f, indent=2)
    with open(os.path.join(OUT_DIR, "supplement_edges.json"), "w") as f:
        json.dump(edges, f, indent=2)

    print(f"Đã crawl {len(users_data)} user, {len(edges)} cạnh follow.")
    print(f"Lưu tại: {OUT_DIR}")


if __name__ == "__main__":
    crawl()
