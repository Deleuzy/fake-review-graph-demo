import networkx as nx
import numpy as np
import scipy.sparse as sp

from frgraph.features import (
    co_review_stats,
    neighbour_features,
    neighbour_mean,
    pagerank,
    spectral_embedding,
    spectral_features,
    structural_features,
)
from frgraph.graph import derive_entities


def test_pagerank_matches_networkx():
    g = nx.karate_club_graph()
    a = nx.to_scipy_sparse_array(g, nodelist=range(g.number_of_nodes()), weight=None)
    ours = pagerank(sp.csr_matrix(a))
    ref = nx.pagerank(g, alpha=0.85, weight=None, tol=1e-12)
    assert np.allclose(ours, [ref[i] for i in range(len(ours))], atol=1e-6)


def test_pagerank_handles_dangling_nodes():
    a = sp.csr_matrix(np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=float))
    r = pagerank(a)
    assert np.isclose(r.sum(), 1.0) and (r > 0).all()


def test_neighbour_mean_and_isolated_rows():
    a = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float))
    a = sp.bmat([[a, None], [None, sp.csr_matrix((1, 1))]]).tocsr()
    x = np.array([[1.0], [2.0], [4.0], [9.0]])
    m = neighbour_mean(a, x)
    assert m[0, 0] == 3.0 and m[1, 0] == 1.0
    assert np.isnan(m[3, 0])


def test_co_review_stats():
    # reviewers 0 and 1 share businesses 0 and 1; reviewer 2 shares only business 1 with them
    rb = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=float))
    max_shared, n_multi = co_review_stats(rb)
    assert list(max_shared) == [2, 2, 1]
    assert list(n_multi) == [1, 1, 0]


def test_spectral_embedding_separates_two_communities_and_skips_small_components():
    blocks = [np.ones((10, 10)), np.ones((10, 10))]
    a = sp.block_diag(blocks).tolil()
    a[0, 10] = a[10, 0] = 1  # one bridge makes a single large component
    a = sp.block_diag([a, np.ones((2, 2))]).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    emb = spectral_embedding(a, dim=2)
    assert emb.shape == (22, 2)
    assert np.isnan(emb[20:]).all()
    first = emb[:20, 0]
    assert np.sign(first[1:10]).tolist().count(np.sign(first[1])) == 9
    assert np.sign(first[1]) != np.sign(first[11])


def test_spectral_embedding_is_deterministic():
    g = nx.karate_club_graph()
    a = sp.csr_matrix(nx.to_scipy_sparse_array(g, weight=None))
    assert np.allclose(spectral_embedding(a, dim=3), spectral_embedding(a, dim=3))


def test_feature_blocks_shapes_and_no_label_use(synthetic):
    data, _ = synthetic
    ent = derive_entities(data.rel)
    s, s_names = structural_features(data.rel, ent)
    nb, nb_names = neighbour_features(data.rel, data.X)
    sp_, sp_names = spectral_features(data.rel, dim=4)
    assert s.shape == (data.n, len(s_names)) and nb.shape == (data.n, len(nb_names)) and sp_.shape == (data.n, 4)
    assert len(nb_names) == 6 * data.X.shape[1]
    # features must not change when labels change
    data.y[:] = 1 - data.y
    s2, _ = structural_features(data.rel, ent)
    assert np.allclose(s, s2, equal_nan=True)
