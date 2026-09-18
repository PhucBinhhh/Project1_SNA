"""
STEP 05 — RQ2 (phần 2): các phương pháp dựa trên EMBEDDING, sau đó phân cụm bằng KMeans (k = 2).

  svd       : KMeans trực tiếp trên đặc trưng SVD-128            (chỉ đặc trưng)
  featprop  : lan truyền đặc trưng qua mạng  Â^k X, rồi KMeans   (cấu trúc + đặc trưng, không huấn luyện)
  deepwalk  : random walk + Word2Vec, rồi KMeans                 (chỉ cấu trúc)
  gae       : Graph Autoencoder 2 lớp GCN + tái tạo đặc trưng    (cấu trúc + đặc trưng)

Mỗi phương pháp chạy với các seed trong config.SEEDS.
Embedding của seed đầu tiên được lưu lại để dùng cho phân tích chẩn đoán ở step06.

Chạy tất cả:          python src/step05_rq2_embedding_methods.py
Chạy một phần:        python src/step05_rq2_embedding_methods.py --methods svd featprop
"""
import argparse
import time

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize

import config as C
from utils import load_edges, load_labels, load_svd_features, save_partitions

ALL_METHODS = ["svd", "featprop", "deepwalk", "gae"]


# ---------- Hàm chung ----------
def cluster_embedding(X: np.ndarray, seed: int, scaling: str = "standard") -> np.ndarray:
    Z = StandardScaler().fit_transform(X) if scaling == "standard" else normalize(X)
    return KMeans(n_clusters=C.N_CLUSTERS, n_init=20, random_state=seed).fit_predict(Z)


def report(name: str, parts: np.ndarray, y: np.ndarray) -> None:
    from sklearn.metrics import adjusted_rand_score
    sizes = np.bincount(parts[0])
    aris = [adjusted_rand_score(y, p) for p in parts]
    print(f"  [{name}] kích thước cụm (seed đầu) = {sizes.tolist()} | "
          f"ARI = {np.mean(aris):.4f} ± {np.std(aris):.4f}")
    if sizes.max() / sizes.sum() > 0.95:
        print(f"  [{name}] CẢNH BÁO: một cụm chiếm hơn 95% số node (suy biến)")


def normalized_adjacency(edges: np.ndarray, n: int) -> sparse.csr_matrix:
    """Â = D^-1/2 (A + I) D^-1/2, giống ma trận dùng trong GCN."""
    u, v = edges
    A = sparse.coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    A = (A + A.T + sparse.identity(n)).tocsr()
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.asarray(A.sum(axis=1)).ravel()))
    return (d_inv_sqrt @ A @ d_inv_sqrt).tocsr()


# ---------- 1. KMeans trên đặc trưng ----------
def method_svd(X, edges, n):
    return np.stack([cluster_embedding(X, s) for s in C.SEEDS])


# ---------- 2. Lan truyền đặc trưng ----------
def method_featprop(X, edges, n):
    S = normalized_adjacency(edges, n)
    H = X.astype(np.float64)
    for _ in range(C.FEATURE_PROP_K):
        H = S @ H
    np.save(C.EMB_DIR / "featprop.npy", H.astype(np.float32))
    return np.stack([cluster_embedding(H, s) for s in C.SEEDS])


# ---------- 3. DeepWalk ----------
def random_walks(indptr, indices, n: int, rng) -> np.ndarray:
    """Sinh đồng thời tất cả random walk; mỗi bước chọn đều một láng giềng."""
    starts = np.tile(np.arange(n), C.DW_NUM_WALKS)
    walks = np.empty((len(starts), C.DW_WALK_LENGTH), dtype=np.int64)
    walks[:, 0] = starts
    for t in range(1, C.DW_WALK_LENGTH):
        cur = walks[:, t - 1]
        deg = indptr[cur + 1] - indptr[cur]
        offset = (rng.random(len(cur)) * deg).astype(np.int64)
        walks[:, t] = indices[indptr[cur] + offset]
    return walks[rng.permutation(len(walks))]


class WalkCorpus:
    """Đưa từng walk cho Word2Vec dưới dạng danh sách chuỗi, không giữ hết trong bộ nhớ."""

    def __init__(self, walks: np.ndarray, tokens: np.ndarray):
        self.walks, self.tokens = walks, tokens

    def __iter__(self):
        for row in self.walks:
            yield self.tokens[row].tolist()


def method_deepwalk(X, edges, n):
    from gensim.models import Word2Vec

    S = normalized_adjacency(edges, n)          # chỉ dùng để lấy danh sách láng giềng
    S.setdiag(0)
    S.eliminate_zeros()
    indptr, indices = S.indptr, S.indices
    tokens = np.array([str(i) for i in range(n)], dtype=object)

    parts = []
    for s in C.SEEDS:
        t = time.time()
        rng = np.random.default_rng(s)
        walks = random_walks(indptr, indices, n, rng)
        model = Word2Vec(
            sentences=WalkCorpus(walks, tokens), vector_size=C.DW_DIM,
            window=C.DW_WINDOW, min_count=0, sg=1, negative=5,
            workers=C.DW_WORKERS, epochs=C.DW_EPOCHS, seed=s,
        )
        emb = np.stack([model.wv[str(i)] for i in range(n)])
        if s == C.SEEDS[0]:
            np.save(C.EMB_DIR / "deepwalk_seed0.npy", emb)
        # embedding skip-gram mang thông tin chủ yếu ở hướng -> chuẩn hóa L2
        parts.append(cluster_embedding(emb, s, scaling="l2"))
        print(f"    seed {s}: xong sau {time.time() - t:.0f} giây")
    return np.stack(parts)


# ---------- 4. Graph Autoencoder ----------
def method_gae(X, edges, n):
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GAE, GCNConv

    class Encoder(torch.nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.conv1 = GCNConv(d_in, C.GAE_HIDDEN)
            self.conv2 = GCNConv(C.GAE_HIDDEN, C.GAE_DIM)

        def forward(self, x, edge_index):
            return self.conv2(self.conv1(x, edge_index).relu(), edge_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Huấn luyện trên: {device}")

    # Node i của PyG chính là node có id = i, vì mảng cạnh và đặc trưng đều theo id
    x = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float, device=device)
    ei = torch.tensor(edges, dtype=torch.long)
    edge_index = torch.cat([ei, ei.flip(0)], dim=1).to(device)   # hai chiều

    parts = []
    for s in C.SEEDS:
        t = time.time()
        torch.manual_seed(s)
        model = GAE(Encoder(x.size(1))).to(device)
        feat_decoder = torch.nn.Linear(C.GAE_DIM, x.size(1)).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(feat_decoder.parameters()),
                               lr=C.GAE_LR)

        for epoch in range(1, C.GAE_EPOCHS + 1):
            model.train()
            opt.zero_grad()
            z = model.encode(x, edge_index)
            loss_struct = model.recon_loss(z, edge_index)
            loss_feat = F.mse_loss(feat_decoder(z), x)
            loss = loss_struct + C.GAE_ALPHA * loss_feat
            loss.backward()
            opt.step()
            if epoch == 1 or epoch % 100 == 0:
                print(f"    seed {s} | epoch {epoch:3d} | cấu trúc {loss_struct.item():.4f} "
                      f"| đặc trưng {loss_feat.item():.4f}")

        model.eval()
        with torch.no_grad():
            emb = model.encode(x, edge_index).cpu().numpy()
        if emb.std(axis=0).mean() < 1e-3:
            print("    CẢNH BÁO: embedding gần như không biến thiên (collapse)")
        if s == C.SEEDS[0]:
            np.save(C.EMB_DIR / "gae_seed0.npy", emb)
        parts.append(cluster_embedding(emb, s))
        print(f"    seed {s}: xong sau {time.time() - t:.0f} giây")
    return np.stack(parts)


RUNNERS = {"svd": method_svd, "featprop": method_featprop,
           "deepwalk": method_deepwalk, "gae": method_gae}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    args = parser.parse_args()

    C.ensure_dirs()
    y = load_labels()
    edges = load_edges()
    X = load_svd_features()
    n = len(y)

    for name in args.methods:
        print(f"\n===== {name} =====")
        t = time.time()
        parts = RUNNERS[name](X, edges, n)
        save_partitions(f"{name}_kmeans", parts)
        report(name, parts, y)
        print(f"  Tổng thời gian: {time.time() - t:.0f} giây")

    print(f"\nĐã lưu kết quả vào {C.PART_DIR} và {C.EMB_DIR}")
    print("Chạy tiếp: python src/step06_rq2_evaluate.py")


if __name__ == "__main__":
    main()
