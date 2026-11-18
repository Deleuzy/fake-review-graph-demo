import itertools

import numpy as np
import scipy.sparse as sp

from frgraph.fraudar import column_weights, detect_blocks, greedy_peel, weighted


def density(w: np.ndarray, rows, cols, prior=None) -> float:
    if len(rows) + len(cols) == 0:
        return 0.0
    mass = w[np.ix_(rows, cols)].sum() if len(rows) and len(cols) else 0.0
    if prior is not None:
        mass += prior[list(rows)].sum()
    return mass / (len(rows) + len(cols))


def planted(seed=0, n_r=200, n_c=100, p=0.01, block=(slice(0, 20), slice(0, 10))):
    rng = np.random.default_rng(seed)
    a = (rng.random((n_r, n_c)) < p).astype(float)
    a[block] = 1.0
    return sp.csr_matrix(a)


def test_finds_planted_block():
    b = greedy_peel(weighted(planted()))
    assert set(range(20)) <= set(b.rows.tolist())
    assert set(range(10)) <= set(b.cols.tolist())
    assert len(b.rows) <= 25 and len(b.cols) <= 12


def test_complete_bipartite_returns_everything():
    b = greedy_peel(sp.csr_matrix(np.ones((4, 3))))
    assert len(b.rows) == 4 and len(b.cols) == 3
    assert np.isclose(b.density, 12 / 7)


def test_reported_density_matches_block():
    w = weighted(planted(seed=3))
    b = greedy_peel(w)
    assert np.isclose(b.density, density(w.toarray(), b.rows, b.cols))


def test_greedy_is_at_least_half_of_optimum_on_small_graphs():
    rng = np.random.default_rng(1)
    for _ in range(15):
        a = (rng.random((5, 4)) < 0.45).astype(float)
        if a.sum() == 0:
            continue
        w = a * rng.uniform(0.2, 1.0, size=4)
        got = greedy_peel(sp.csr_matrix(w)).density
        best = 0.0
        for r in range(6):
            for rows in itertools.combinations(range(5), r):
                for c in range(5):
                    for cols in itertools.combinations(range(4), c):
                        best = max(best, density(w, list(rows), list(cols)))
        assert got >= best / 2 - 1e-9
        assert got <= best + 1e-9


def test_column_weights_down_weight_popular_targets():
    a = sp.csr_matrix(np.array([[1, 1], [1, 0], [1, 0]], dtype=float))
    cw = column_weights(a, x=5)
    assert np.isclose(cw[0], 1 / np.log(8)) and np.isclose(cw[1], 1 / np.log(6))
    assert cw[0] < cw[1]


def test_camouflage_edges_to_popular_column_do_not_hide_block():
    a = planted(seed=2).toarray()
    a[:, 99] = 1.0  # every account, honest or not, reviews one hugely popular business
    blocks = detect_blocks(sp.csr_matrix(a), k=1)
    assert set(range(20)) <= set(blocks[0].rows.tolist())


def test_detect_blocks_finds_two_disjoint_blocks():
    a = planted(seed=4).toarray()
    a[100:115, 50:58] = 1.0
    blocks = detect_blocks(sp.csr_matrix(a), k=2)
    found = [set(b.rows.tolist()) for b in blocks]
    assert any(set(range(20)) <= f for f in found)
    assert any(set(range(100, 115)) <= f for f in found)


def test_row_prior_pulls_suspicious_rows_in():
    a = sp.csr_matrix(np.array([[1, 1], [1, 1], [1, 0]], dtype=float))
    without = greedy_peel(a)
    with_prior = greedy_peel(a, row_prior=np.array([0.0, 0.0, 5.0]))
    assert 2 in with_prior.rows.tolist()
    assert with_prior.density > without.density


def test_empty_graph():
    b = greedy_peel(sp.csr_matrix((3, 3)))
    assert len(b.rows) == 0 and b.density == 0.0
