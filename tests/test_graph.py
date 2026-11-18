import numpy as np
import scipy.io as sio
import scipy.sparse as sp

from frgraph.data import RELATIONS, clean_adjacency, load_yelpchi
from frgraph.graph import (
    build_typed_graph,
    derive_entities,
    is_union_of_cliques,
    load_schema,
    reviewer_business_matrix,
    validate_against_schema,
)


def same_partition(a, b):
    # two labelings describe the same partition if the pair mapping is one to one
    pairs = set(zip(a.tolist(), b.tolist(), strict=True))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def test_derive_entities_recovers_reviewers_and_businesses(synthetic):
    data, truth = synthetic
    ent = derive_entities(data.rel)
    assert same_partition(ent.reviewer, truth["reviewer"])
    assert same_partition(ent.business, truth["business"])
    assert same_partition(ent.star_group, truth["business"] * 10 + truth["star"])
    assert ent.checks["rur_is_union_of_cliques"]
    assert ent.checks["rtr_within_one_business"]


def test_union_of_cliques_detects_a_path():
    path = clean_adjacency(sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])))
    assert not is_union_of_cliques(path)
    tri = clean_adjacency(sp.csr_matrix(np.ones((3, 3))))
    assert is_union_of_cliques(tri)


def test_clean_adjacency_symmetric_binary_no_loops():
    a = clean_adjacency(sp.csr_matrix(np.array([[1, 2, 0], [0, 0, 0], [0, 3, 0]])))
    d = a.toarray()
    assert (d == d.T).all() and set(np.unique(d)) <= {0, 1} and d.trace() == 0


def test_typed_graph_matches_schema(synthetic):
    data, _ = synthetic
    ent = derive_entities(data.rel)
    g = build_typed_graph(ent)
    assert validate_against_schema(g, load_schema()) == []
    n_reviews = sum(1 for _, d in g.nodes(data=True) if d["type"] == "Review")
    assert n_reviews == data.n


def test_schema_validation_catches_wrong_endpoints(synthetic):
    data, _ = synthetic
    g = build_typed_graph(derive_entities(data.rel))
    g.add_edge(("Business", 0), ("Reviewer", 0), type="WROTE")
    g.add_node(("Device", 0), type="Device")
    errors = validate_against_schema(g, load_schema())
    assert any("WROTE" in e for e in errors) and any("Device" in e for e in errors)


def test_schema_states_source_for_every_type():
    s = load_schema()
    for section in ("node_types", "relation_types"):
        for name, spec in s[section].items():
            assert spec["source"] in {"given", "derived"}, name


def test_reviewer_business_matrix_counts(synthetic):
    data, truth = synthetic
    ent = derive_entities(data.rel)
    rb = reviewer_business_matrix(ent)
    assert rb.sum() == data.n


def test_load_yelpchi_from_mat(tmp_path, synthetic):
    data, _ = synthetic
    homo = data.rel["net_rur"] + data.rel["net_rtr"] + data.rel["net_rsr"]
    payload = {r: sp.csc_matrix(data.rel[r]) for r in RELATIONS}
    payload.update(features=sp.csc_matrix(data.X), label=data.y[None, :], homo=sp.csc_matrix(homo))
    sio.savemat(tmp_path / "toy.mat", payload)
    loaded = load_yelpchi(tmp_path / "toy.mat")
    assert loaded.n == data.n
    assert np.allclose(loaded.X, data.X)
    assert (loaded.rel["net_rur"] != data.rel["net_rur"]).nnz == 0
