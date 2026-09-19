"""Xuất file .gexf cho Gephi, kèm nhãn, cụm Leiden và các độ đo centrality."""
import networkx as nx
import numpy as np
import pandas as pd

import config as C
from utils import load_graph, load_labels, load_partitions

OUT = C.OUT_DIR / "gephi"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    G = load_graph()
    y = load_labels()
    cent = pd.read_csv(C.RES_DIR / "rq3_centrality.csv").set_index("id")
    leiden = load_partitions("leiden_selected")[0]

    attrs = {
        "nghe": {i: ("ML" if y[i] == 1 else "Web") for i in G.nodes()},
        "cum_leiden": {i: int(leiden[i]) for i in G.nodes()},
        "bac": {i: int(cent.loc[i, "degree"]) for i in G.nodes()},
        "pagerank": {i: float(cent.loc[i, "pagerank"]) for i in G.nodes()},
        "he_so_tham_gia": {i: float(cent.loc[i, "participation"]) for i in G.nodes()},
    }
    for name, values in attrs.items():
        nx.set_node_attributes(G, values, name)

    nx.write_gexf(G, OUT / "github_full.gexf")
    print(f"Đã xuất: {OUT / 'github_full.gexf'}")

    # Đồ thị con: cụm hình sao mà GAE luôn tách ra (dùng cho hình chẩn đoán RQ2)
    gae = load_partitions("gae_kmeans")[0]
    small = np.flatnonzero(gae == np.argmin(np.bincount(gae)))
    hub = int(small[np.argmax([G.degree(i) for i in small])])
    star = G.subgraph(list(small) + list(G.neighbors(hub))[:300])
    nx.write_gexf(star, OUT / "gae_star_cluster.gexf")
    print(f"Đã xuất: {OUT / 'gae_star_cluster.gexf'} ({star.number_of_nodes()} node)")


if __name__ == "__main__":
    main()