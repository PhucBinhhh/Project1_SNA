"""
STEP 1 — Tải dataset MUSAE GitHub Social Network gốc từ SNAP.

Nguồn: https://snap.stanford.edu/data/github-social.html
File zip chứa: musae_git_edges.csv, musae_git_features.json, musae_git_target.csv
"""
import os
import zipfile
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZIP_URL = "https://snap.stanford.edu/data/git_web_ml.zip"
ZIP_PATH = os.path.join(DATA_DIR, "git_web_ml.zip")


def download_file(url: str, dest: str, chunk_size: int = 8192) -> None:
    print(f"Đang tải {url} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    print(f"Đã lưu: {dest}")


def extract_zip(zip_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Đã giải nén vào: {extract_to}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        download_file(ZIP_URL, ZIP_PATH)
    else:
        print("File zip đã tồn tại, bỏ qua tải lại.")

    extract_zip(ZIP_PATH, DATA_DIR)

    expected_files = [
        "git_web_ml/musae_git_edges.csv",
        "git_web_ml/musae_git_features.json",
        "git_web_ml/musae_git_target.csv",
    ]
    for f in expected_files:
        path = os.path.join(DATA_DIR, f)
        status = "OK" if os.path.exists(path) else "THIẾU"
        print(f"[{status}] {path}")


if __name__ == "__main__":
    main()
