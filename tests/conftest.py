"""Small synthetic stand in for YelpChi, built the same way the real relations are defined."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from frgraph.data import ReviewData, clean_adjacency


def clique_adjacency(groups: np.ndarray) -> sp.csr_matrix:
    inc = sp.csr_matrix((np.ones(len(groups)), (np.arange(len(groups)), groups)))
    return clean_adjacency(inc @ inc.T)


def window_adjacency(business: np.ndarray, day: np.ndarray, days: int) -> sp.csr_matrix:
    """Same business and posted less than `days` apart. Like YelpChi's R-T-R this is not a union of cliques."""
    same = business[:, None] == business[None, :]
    close = np.abs(day[:, None] - day[None, :]) < days
    return clean_adjacency(sp.csr_matrix((same & close).astype(float)))


def make_synthetic(seed: int = 0, n_reviewers: int = 60, n_business: int = 6, n_features: int = 32):
    rng = np.random.default_rng(seed)
    reviewer, business, month, star = [], [], [], []
    for u in range(n_reviewers):
        k = 1 if u % 3 == 0 else int(rng.integers(2, 5))
        bs = rng.choice(n_business, size=min(k, n_business), replace=False)
        for b in bs:
            reviewer.append(u)
            business.append(int(b))
            month.append(int(rng.integers(0, 90)))  # day of posting
            star.append(int(rng.integers(1, 6)))
    reviewer, business, month, star = map(np.array, (reviewer, business, month, star))
    n = len(reviewer)
    y = (reviewer % 3 == 0).astype(int)  # one review accounts are "fake", like YelpChi's skew
    x = rng.random((n, n_features))
    x[:, 0] += y
    rel = {
        "net_rur": clique_adjacency(reviewer),
        "net_rtr": window_adjacency(business, month, days=30),
        "net_rsr": clique_adjacency(business * 10 + star),
    }
    truth = {"reviewer": reviewer, "business": business, "day": month, "star": star}
    return ReviewData(X=x, y=y, rel=rel), truth


@pytest.fixture
def synthetic():
    return make_synthetic()
