import numpy as np
import scipy.sparse as sp
import networkx as nx

from bk_maxflow import graph_cut_from_adj 

def run_maxflow(
    adj: sp.csr_matrix, S_index: int, T_index: int, num_pixels: int, algorithm: str = 'bk'
) -> tuple[float, np.ndarray]:
    """
    Runs s-t mincut on the given adjacency matrix using the specified algorithm.
    """
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()

    if algorithm == 'bk':
        return graph_cut_from_adj(adj, S_index, T_index, num_pixels)

    G = nx.DiGraph()
    n = int(adj.shape[0])

    # Try sparse API with .nonzero(); fallback to dense scan (small graphs only)
    try:
        rows, cols = adj.nonzero()
        for u, v in zip(rows, cols):
            if u == v:
                continue
            c = float(adj[u, v])
            if c > 0:
                G.add_edge(int(u), int(v), capacity=c)
    except Exception:
        M = np.asarray(adj, dtype=float)
        for u in range(n):
            for v in range(n):
                if u != v and M[u, v] > 0:
                    G.add_edge(u, v, capacity=float(M[u, v]))

    if algorithm == 'bk_nx':
        flow_func = nx.algorithms.flow.boykov_kolmogorov
    elif algorithm == 'dinic':
        flow_func = nx.algorithms.flow.dinitz
    elif algorithm == 'edmonds_karp':
        flow_func = nx.algorithms.flow.edmonds_karp
    elif algorithm == 'preflow':
        flow_func = nx.algorithms.flow.preflow_push
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}.")

    # Compute min-cut. NetworkX returns the cut value and the partition
    cut_value, (S_side, T_side) = nx.minimum_cut(
        G, S_index, T_index, capacity="capacity", flow_func=flow_func
    )

    # Labels for pixel nodes only: 0 if reachable from S in residual (SOURCE side), else 1
    S_side = set(S_side)
    labels = np.fromiter((0 if i in S_side else 1 for i in range(num_pixels)),
                         dtype=np.uint8, count=num_pixels)

    return float(cut_value), labels
