"""Label free graph features for each review.

Nothing here reads labels, so the features can be computed once on the whole graph. That is the
same situation as production, where the graph around a new review is known before its label is.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from .graph import Entities, reviewer_business_matrix

REL_SHORT = {"net_rur": "rur", "net_rtr": "rtr", "net_rsr": "rsr"}


def pagerank(adj: sp.spmatrix, damping: float = 0.85, tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
    """PageRank on an undirected (or directed, row = source) graph by power iteration. Dangling mass is spread evenly."""
    a = sp.csr_matrix(adj, dtype=np.float64)
    n = a.shape[0]
    out = np.asarray(a.sum(axis=1)).ravel()
    dangling = out == 0
    inv = np.zeros(n)
    inv[~dangling] = 1.0 / out[~dangling]
    p = sp.diags(inv) @ a  # row stochastic where out > 0
    pt = sp.csr_matrix(p.T)
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new = damping * (pt @ r + r[dangling].sum() / n) + (1 - damping) / n
        if np.abs(new - r).sum() < tol:
            r = new
            break
        r = new
    return r / r.sum()


def neighbour_mean(adj: sp.spmatrix, x: np.ndarray) -> np.ndarray:
    """Mean of neighbours' feature rows. Rows with no neighbours are NaN (the model sees missing, not zero)."""
    a = sp.csr_matrix(adj, dtype=np.float64)
    deg = np.asarray(a.sum(axis=1)).ravel()
    s = a @ x
    out = np.full_like(s, np.nan, dtype=np.float64)
    has = deg > 0
    out[has] = s[has] / deg[has, None]
    return out


def co_review_stats(rb: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Per reviewer: max number of businesses shared with any single other reviewer, and the number of
    other reviewers sharing two or more businesses."""
    b = (sp.csr_matrix(rb) > 0).astype(np.float64)
    c = sp.csr_matrix(b @ b.T)
    c.setdiag(0)
    c.eliminate_zeros()
    n = c.shape[0]
    max_shared = np.zeros(n)
    n_multi = np.zeros(n)
    for i in range(n):
        row = c.data[c.indptr[i] : c.indptr[i + 1]]
        if row.size:
            max_shared[i] = row.max()
            n_multi[i] = (row >= 2).sum()
    return max_shared, n_multi


def spectral_embedding(adj: sp.spmatrix, dim: int = 16, seed: int = 0) -> np.ndarray:
    """Eigenvectors of the normalised adjacency of the largest connected component, dropping the trivial
    top one. Nodes outside that component get NaN. Signs are fixed so the result is deterministic."""
    a = sp.csr_matrix(adj, dtype=np.float64)
    n = a.shape[0]
    _, lab = connected_components(a, directed=False)
    big = lab == np.bincount(lab).argmax()
    idx = np.where(big)[0]
    sub = a[idx][:, idx]
    deg = np.asarray(sub.sum(axis=1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(deg))
    norm = dinv @ sub @ dinv
    k = min(dim + 1, len(idx) - 1)
    v0 = np.random.default_rng(seed).random(len(idx))
    vals, vecs = eigsh(norm, k=k, which="LA", v0=v0)
    order = np.argsort(-vals)
    vecs = vecs[:, order[1:]]
    signs = np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(vecs.shape[1])])
    vecs = vecs * signs
    out = np.full((n, vecs.shape[1]), np.nan)
    out[idx] = vecs * np.sqrt(len(idx))
    return out


def group_size(groups: np.ndarray) -> np.ndarray:
    return np.bincount(groups)[groups].astype(np.float64)


def structural_features(rel: dict[str, sp.csr_matrix], ent: Entities) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for r, short in REL_SHORT.items():
        cols.append(np.log1p(np.asarray(rel[r].sum(axis=1)).ravel()))
        names.append(f"log_degree_{short}")

    rb = reviewer_business_matrix(ent)
    rev_n = group_size(ent.reviewer)
    rev_nb = np.asarray((rb > 0).sum(axis=1)).ravel()[ent.reviewer]
    bus_n = group_size(ent.business)
    bus_nu = np.asarray((rb > 0).sum(axis=0)).ravel()[ent.business]
    star_n = group_size(ent.star_group)
    tw_n = group_size(ent.time_window)
    cols += [np.log1p(rev_n), np.log1p(rev_nb), np.log1p(bus_n), np.log1p(bus_nu), star_n / bus_n, tw_n / bus_n]
    names += [
        "reviewer_n_reviews",
        "reviewer_n_businesses",
        "business_n_reviews",
        "business_n_reviewers",
        "star_group_share_of_business",
        "time_window_share_of_business",
    ]

    max_shared, n_multi = co_review_stats(rb)
    cols += [max_shared[ent.reviewer], np.log1p(n_multi[ent.reviewer])]
    names += ["co_review_max_shared", "co_reviewers_2plus"]

    homo = rel["net_rur"] + rel["net_rtr"] + rel["net_rsr"]
    homo = (homo > 0).astype(np.float64)
    pr = pagerank(homo)
    cols.append(np.log(pr * len(pr)))
    names.append("pagerank_review_graph")

    # PageRank of the reviewer in the reviewer-business bipartite graph.
    n_u, n_b = rb.shape
    bip = sp.bmat([[None, rb], [rb.T, None]]).tocsr()
    pr_b = pagerank(bip)[:n_u]
    cols.append(np.log(pr_b * (n_u + n_b))[ent.reviewer])
    names.append("pagerank_reviewer_bipartite")
    return np.column_stack(cols), names


def neighbour_features(rel: dict[str, sp.csr_matrix], x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for r, short in REL_SHORT.items():
        m = neighbour_mean(rel[r], x)
        cols.append(m)
        cols.append(x - m)
        names += [f"nbr_mean_{short}_f{j}" for j in range(x.shape[1])]
        names += [f"self_minus_nbr_{short}_f{j}" for j in range(x.shape[1])]
    return np.column_stack(cols), names


def spectral_features(rel: dict[str, sp.csr_matrix], dim: int = 16) -> tuple[np.ndarray, list[str]]:
    homo = ((rel["net_rur"] + rel["net_rtr"] + rel["net_rsr"]) > 0).astype(np.float64)
    emb = spectral_embedding(homo, dim=dim)
    return emb, [f"spectral_{j}" for j in range(emb.shape[1])]
