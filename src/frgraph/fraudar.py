"""FRAUDAR style greedy peeling for dense blocks in a bipartite graph (Hooi et al., KDD 2016).

Rows are accounts (reviewers), columns are targets (businesses). Each edge (i, j) gets weight
1 / log(x + d_j), where d_j is the column degree, so edges to popular targets count less. This is
the column weighting FRAUDAR uses to resist camouflage (fraudsters adding edges to popular honest
targets). The density of a node set S is the total weight of edges inside S divided by |S|.

Greedy peeling removes, one at a time, the node whose remaining edges carry the least weight, and
returns the intermediate set with the highest density. Several blocks are found by deleting the
edges of each block and peeling again.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Block:
    rows: np.ndarray
    cols: np.ndarray
    density: float
    n_edges: int


def column_weights(adj: sp.spmatrix, x: float = 5.0) -> np.ndarray:
    deg = np.asarray((sp.csr_matrix(adj) > 0).sum(axis=0)).ravel()
    return 1.0 / np.log(x + deg)


def weighted(adj: sp.spmatrix, col_w: np.ndarray | None = None) -> sp.csr_matrix:
    a = (sp.csr_matrix(adj) > 0).astype(np.float64)
    if col_w is None:
        return a
    return sp.csr_matrix(a @ sp.diags(col_w))


def greedy_peel(w: sp.spmatrix, row_prior: np.ndarray | None = None) -> Block:
    """Peel a weighted bipartite matrix. Returns the densest intermediate block (rows, cols).

    row_prior is FRAUDAR's optional node suspiciousness term: a non negative score per row added to
    the block's mass when the row is in the block. It must not be derived from labels."""
    w = sp.csr_matrix(w, dtype=np.float64)
    w.eliminate_zeros()
    n_r, n_c = w.shape
    wc = sp.csc_matrix(w)
    # node ids: rows 0..n_r-1, cols n_r..n_r+n_c-1
    edge_mass = np.concatenate([np.asarray(w.sum(axis=1)).ravel(), np.asarray(w.sum(axis=0)).ravel()])
    node_prior = np.zeros(n_r + n_c)
    if row_prior is not None:
        node_prior[:n_r] = row_prior
    prio = edge_mass + node_prior
    alive = np.ones(n_r + n_c, dtype=bool)
    # ignore isolated nodes from the start; they only lower density
    alive[edge_mass == 0] = False
    prio[~alive] = 0.0
    total = float(w.sum()) + float(node_prior[alive].sum())
    size = int(alive.sum())
    if size == 0:
        return Block(np.array([], int), np.array([], int), 0.0, 0)
    best_density = total / size
    best_step = 0
    heap = [(prio[v], v) for v in np.where(alive)[0]]
    heapq.heapify(heap)
    removed = []
    while heap:
        p, v = heapq.heappop(heap)
        if not alive[v] or p != prio[v]:
            continue
        alive[v] = False
        removed.append(v)
        total -= prio[v]
        size -= 1
        if v < n_r:
            lo, hi = w.indptr[v], w.indptr[v + 1]
            nbrs, vals = w.indices[lo:hi] + n_r, w.data[lo:hi]
        else:
            j = v - n_r
            lo, hi = wc.indptr[j], wc.indptr[j + 1]
            nbrs, vals = wc.indices[lo:hi], wc.data[lo:hi]
        for u, val in zip(nbrs, vals, strict=True):
            if alive[u]:
                prio[u] -= val
                heapq.heappush(heap, (prio[u], u))
        if size > 0:
            d = total / size
            if d > best_density + 1e-12:
                best_density, best_step = d, len(removed)
    gone = set(removed[:best_step])
    start = np.where(edge_mass > 0)[0]
    keep = np.array([v for v in start if v not in gone], dtype=int)
    rows = keep[keep < n_r]
    cols = keep[keep >= n_r] - n_r
    n_edges = int((w[rows][:, cols] > 0).sum()) if len(rows) and len(cols) else 0
    return Block(rows, cols, float(best_density), n_edges)


def detect_blocks(
    adj: sp.spmatrix, k: int = 10, x: float = 5.0, min_rows: int = 2, row_prior: np.ndarray | None = None
) -> list[Block]:
    """Find up to k blocks, deleting each block's edges before searching for the next."""
    col_w = column_weights(adj, x)
    w = weighted(adj, col_w).tolil()
    blocks = []
    for _ in range(k):
        b = greedy_peel(w.tocsr(), row_prior)
        if len(b.rows) == 0:
            break
        if len(b.rows) >= min_rows:
            blocks.append(b)
        for r in b.rows:
            row_cols = np.array(w.rows[r], dtype=int)
            hit = np.isin(row_cols, b.cols)
            for c in row_cols[hit]:
                w[r, c] = 0
    return blocks
