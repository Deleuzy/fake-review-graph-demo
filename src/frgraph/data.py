"""Load the public YelpChi graph (CARE-GNN release) into plain numpy and scipy objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

RELATIONS = ("net_rur", "net_rtr", "net_rsr")
REL_NAMES = {"net_rur": "R-U-R", "net_rtr": "R-T-R", "net_rsr": "R-S-R"}
DEFAULT_PATH = Path("data/raw/YelpChi.mat")


@dataclass
class ReviewData:
    X: np.ndarray  # (n, 32) handcrafted features
    y: np.ndarray  # (n,) 1 = filtered by Yelp (treated as fake)
    rel: dict[str, sp.csr_matrix]  # symmetric 0/1 adjacency per relation, no self loops

    @property
    def n(self) -> int:
        return len(self.y)


def clean_adjacency(a) -> sp.csr_matrix:
    """Symmetric, binary, no self loops."""
    a = sp.csr_matrix(a, dtype=np.float64)
    a = ((a + a.T) > 0).astype(np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    return sp.csr_matrix(a)


def load_yelpchi(path: str | Path = DEFAULT_PATH) -> ReviewData:
    m = sio.loadmat(str(path))
    x = m["features"]
    x = x.toarray() if sp.issparse(x) else np.asarray(x)
    y = np.asarray(m["label"]).ravel().astype(int)
    rel = {r: clean_adjacency(m[r]) for r in RELATIONS}
    return ReviewData(X=x.astype(np.float64), y=y, rel=rel)
