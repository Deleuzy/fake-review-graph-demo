"""Train, validation and test splits that keep whole groups together.

YelpChi labels are almost constant per reviewer (a filtered reviewer usually has all reviews
filtered), and reviews by the same reviewer share reviewer level features and an R-U-R edge.
A random split therefore puts near copies of test reviews in the training set. Grouping by
reviewer (or by business) prevents that.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

FRACTIONS = (0.6, 0.2, 0.2)


def group_split(groups: np.ndarray, seed: int, fractions=FRACTIONS) -> np.ndarray:
    """Assign every review to 0 = train, 1 = validation, 2 = test so that each group lands in exactly one part.

    Groups are shuffled with the seed and filled into parts in order until each part reaches its share of
    reviews, so the review counts match the fractions closely even when group sizes vary.
    """
    groups = np.asarray(groups)
    uniq, inv, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(uniq))
    cum = np.cumsum(sizes[order]) / sizes.sum()
    bounds = np.cumsum(fractions)[:-1]
    part_of_ordered = np.searchsorted(bounds, cum - sizes[order] / sizes.sum() / 2, side="right")
    part_of_group = np.empty(len(uniq), dtype=np.int64)
    part_of_group[order] = part_of_ordered
    return part_of_group[inv]


def random_split(n: int, seed: int, fractions=FRACTIONS) -> np.ndarray:
    """Plain random split per review. Kept only to show how much it overstates results."""
    return group_split(np.arange(n), seed, fractions)


def cross_part_edges(adj: sp.spmatrix, part: np.ndarray, a: int = 0, b: int = 2) -> int:
    """Number of undirected edges linking part a to part b (a leakage check)."""
    coo = sp.triu(sp.csr_matrix(adj), k=1).tocoo()
    pr, pc = part[coo.row], part[coo.col]
    return int(((pr == a) & (pc == b)).sum() + ((pr == b) & (pc == a)).sum())
