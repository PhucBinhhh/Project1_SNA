"""
STEP 5 — Trả lời một phần RQ2: phân cụm bằng GCN unsupervised.

Dùng Graph Autoencoder (GAE) / Variational GAE (VGAE) của torch_geometric:
  - Encoder: 2 lớp GCNConv, học embedding từ (cấu trúc mạng + feature SVD-128)
  - Decoder: inner-product, tối ưu để tái tạo lại cấu trúc cạnh gốc
  - Sau khi có embedding -> chạy KMeans (k=2, khớp số nhãn web/ML) để ra cụm

So với Louvain/Leiden (chỉ dùng cấu trúc), GCN ở đây dùng CẢ cấu trúc VÀ feature riêng
của từng lập trình viên -> đây chính là phép so sánh cốt lõi của RQ2.
"""
import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, VGAE
from sklearn.cluster import KMeans

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

EMBED_DIM = 32
EPOCHS = 200
LR = 0.01
N_CLUSTERS = 2   # khớp với 2 nhãn web/ML — có thể thử k lớn hơn để phân tích sâu hơn


class VGAEEncoder(torch.nn.Module):
    """Encoder 2 lớp GCN cho VGAE: 1 lớp chung + 2 lớp riêng cho mu và logstd."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


def build_pyg_data():
    with open(os.path.join(PROC_DIR, "graph.gpickle"), "rb") as f:
        G = pickle.load(f)
    features = np.load(os.path.join(PROC_DIR, "features_svd128.npy"))

    data = from_networkx(G)
    data.x = torch.tensor(features[np.array(sorted(G.nodes()))], dtype=torch.float)
    return data, sorted(G.nodes())


def train_vgae(data) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training trên: {device}")

    model = VGAE(VGAEEncoder(data.x.size(1), 64, EMBED_DIM)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
    return z.cpu().numpy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data, node_order = build_pyg_data()
    print(f"PyG Data: {data.num_nodes} node, {data.num_edges} cạnh (có hướng x2), "
          f"feature dim = {data.x.size(1)}")

    embeddings = train_vgae(data)
    np.save(os.path.join(OUT_DIR, "gcn_embeddings.npy"), embeddings)

    print(f"\nChạy KMeans (k={N_CLUSTERS}) trên embedding ...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    result_df = pd.DataFrame({"node": node_order, "cluster": cluster_labels})
    result_df.to_csv(os.path.join(OUT_DIR, "community_gcn.csv"), index=False)

    print(f"Phân bố cụm GCN: {pd.Series(cluster_labels).value_counts().to_dict()}")
    print(f"Đã lưu kết quả vào: {OUT_DIR}")


if __name__ == "__main__":
    main()
