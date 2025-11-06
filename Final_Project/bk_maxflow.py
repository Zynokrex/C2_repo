# bk_maxflow.py
# Boykov–Kolmogorov min-cut wrapper for your ImageGraph adjacency.
# Tries PyMaxflow (BK). Falls back to a pure-Python Dinic implementation.

from __future__ import annotations
import numpy as np
import scipy.sparse as sp

# Optional BK backend (highly recommended)
_HAS_PYMAXFLOW = False
try:
    import maxflow  # pip install PyMaxflow
    _HAS_PYMAXFLOW = True
except Exception:
    _HAS_PYMAXFLOW = False


def graph_cut_from_adj(
    adj: sp.csr_matrix, S_index: int, T_index: int, num_pixels: int
) -> tuple[float, np.ndarray]:
    """
    Runs s-t mincut on the given adjacency matrix.

    Args:
        adj: csr_matrix of shape (num_pixels+2, num_pixels+2).
             Should include symmetric n-links, and t-links to S_index and T_index.
        S_index: index of source node in the matrix.
        T_index: index of sink node in the matrix.
        num_pixels: number of pixel-nodes (0..num_pixels-1). Terminals are outside this range.

    Returns:
        flow: max-flow value.
        labels: np.uint8 array of shape (num_pixels,), 0 = SOURCE side, 1 = SINK side.
    """
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()

    if _HAS_PYMAXFLOW:
        return _bk_with_pymaxflow(adj, S_index, T_index, num_pixels)
    else:
        return _dinic_fallback(adj, S_index, T_index, num_pixels)


# ----------------------------
# BK via PyMaxflow (fast path)
# ----------------------------
def _bk_with_pymaxflow(
    adj: sp.csr_matrix, S_index: int, T_index: int, num_pixels: int
) -> tuple[float, np.ndarray]:
    # Collect terminal weights and pairwise edges
    # We expect adj to be (roughly) symmetric for pixel-pixel edges, and
    # to contain capacities for (p <-> S) and (p <-> T) t-links.
    A = adj.tocoo()
    source_w = np.zeros(num_pixels, dtype=np.float64)
    sink_w = np.zeros(num_pixels, dtype=np.float64)

    # To avoid double-adding undirected n-links, accumulate only once per unordered pair.
    # We’ll build a map for (i, j) with i<j.
    from collections import defaultdict
    pair_caps = defaultdict(lambda: [0.0, 0.0])  # (i,j) -> [cap_i_to_j, cap_j_to_i]

    for i, j, c in zip(A.row, A.col, A.data):
        if i == j or c <= 0:
            continue
        i_pix = i < num_pixels
        j_pix = j < num_pixels

        # t-links
        if i_pix and j == S_index:
            source_w[i] += float(c)
            continue
        if i_pix and j == T_index:
            sink_w[i] += float(c)
            continue
        if j_pix and i == S_index:
            source_w[j] += float(c)
            continue
        if j_pix and i == T_index:
            sink_w[j] += float(c)
            continue

        # n-links (pixel-pixel)
        if i_pix and j_pix:
            if i < j:
                pair_caps[(i, j)][0] += float(c)  # i -> j
            else:
                pair_caps[(j, i)][1] += float(c)  # j -> i

    g = maxflow.Graph[float](num_pixels, max(len(pair_caps), 1))
    g.add_nodes(num_pixels)

    # add pairwise capacities
    for (i, j), (c_ij, c_ji) in pair_caps.items():
        if c_ij > 0 or c_ji > 0:
            g.add_edge(i, j, c_ij, c_ji)

    # add terminal weights
    for p in range(num_pixels):
        if source_w[p] != 0 or sink_w[p] != 0:
            g.add_tedge(p, source_w[p], sink_w[p])

    flow = g.maxflow()
    labels = np.fromiter((g.get_segment(p) for p in range(num_pixels)), dtype=np.uint8, count=num_pixels)
    return float(flow), labels


# --------------------------------
# Pure-Python Dinic (fallback path)
# --------------------------------
class _Edge:
    __slots__ = ("to", "rev", "cap")
    def __init__(self, to: int, rev: int, cap: float) -> None:
        self.to = to
        self.rev = rev
        self.cap = cap


class _Dinic:
    def __init__(self, N: int) -> None:
        self.N = N
        self.G = [[] for _ in range(N)]
        self.level = [-1] * N
        self.it = [0] * N

    def add_edge(self, fr: int, to: int, cap: float) -> None:
        fwd = _Edge(to, len(self.G[to]), cap)
        rev = _Edge(fr, len(self.G[fr]), 0.0)
        self.G[fr].append(fwd)
        self.G[to].append(rev)

    def bfs(self, s: int, t: int) -> bool:
        from collections import deque
        self.level = [-1] * self.N
        dq = deque([s])
        self.level[s] = 0
        while dq:
            v = dq.popleft()
            for e in self.G[v]:
                if e.cap > 1e-18 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[v] + 1
                    dq.append(e.to)
        return self.level[t] >= 0

    def dfs(self, v: int, t: int, f: float) -> float:
        if v == t:
            return f
        i = self.it[v]
        while i < len(self.G[v]):
            e = self.G[v][i]
            if e.cap > 1e-18 and self.level[v] < self.level[e.to]:
                d = self.dfs(e.to, t, min(f, e.cap))
                if d > 0:
                    e.cap -= d
                    self.G[e.to][e.rev].cap += d
                    return d
            i += 1
            self.it[v] = i
        return 0.0

    def maxflow(self, s: int, t: int) -> float:
        flow = 0.0
        INF = 1e30
        while self.bfs(s, t):
            self.it = [0] * self.N
            while True:
                f = self.dfs(s, t, INF)
                if f <= 0:
                    break
                flow += f
        return flow

    def mincut_reachable(self, s: int) -> np.ndarray:
        # After maxflow, nodes with level >= 0 are reachable from s in residual graph
        return np.array([lvl >= 0 for lvl in self.level], dtype=bool)


def _dinic_fallback(
    adj: sp.csr_matrix, S_index: int, T_index: int, num_pixels: int
) -> tuple[float, np.ndarray]:
    N = adj.shape[0]
    din = _Dinic(N)

    A = adj.tocoo()
    for i, j, c in zip(A.row, A.col, A.data):
        if i == j or c <= 0:
            continue
        din.add_edge(int(i), int(j), float(c))

    flow = din.maxflow(S_index, T_index)
    reachable = din.mincut_reachable(S_index)

    # labels only for pixel nodes (0..num_pixels-1)
    labels = np.where(reachable[:num_pixels], 0, 1).astype(np.uint8)
    return float(flow), labels
