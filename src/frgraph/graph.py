"""Build the typed review graph (the ontology in schema/review_graph.yaml) from the three given relations.

YelpChi publishes only review nodes and three review to review relations. Reviewer, Business,
StarGroup and TimeWindow nodes are derived here from the structure of those relations. No labels
are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import yaml
from scipy.sparse.csgraph import connected_components

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "review_graph.yaml"


def components(adj: sp.spmatrix) -> np.ndarray:
    """Component id per node (isolated nodes get their own id)."""
    _, lab = connected_components(sp.csr_matrix(adj), directed=False)
    return lab.astype(np.int64)


def is_union_of_cliques(adj: sp.spmatrix) -> bool:
    """True if every connected component is a complete graph (each node links to all others in it)."""
    adj = sp.csr_matrix(adj)
    lab = components(adj)
    sizes = np.bincount(lab)
    deg = np.diff(adj.indptr)
    return bool(np.all(deg == sizes[lab] - 1))


@dataclass
class Entities:
    reviewer: np.ndarray  # reviewer id per review
    business: np.ndarray  # business id per review
    star_group: np.ndarray  # (business, star) group id per review
    time_window: np.ndarray  # burst id per review
    checks: dict[str, bool]

    def counts(self) -> dict[str, int]:
        return {
            "Review": len(self.reviewer),
            "Reviewer": int(self.reviewer.max()) + 1,
            "Business": int(self.business.max()) + 1,
            "BusinessWith2PlusReviews": int((np.bincount(self.business) >= 2).sum()),
            "StarGroup": int(self.star_group.max()) + 1,
            "TimeWindow": int(self.time_window.max()) + 1,
        }


def derive_entities(rel: dict[str, sp.csr_matrix]) -> Entities:
    rur, rtr, rsr = rel["net_rur"], rel["net_rtr"], rel["net_rsr"]
    reviewer = components(rur)
    business = components(rsr + rtr)
    star_group = components(rsr)
    time_window = components(rtr)
    # Checks on how the relations were built that could fail on other data. They are reported, not assumed.
    # (That R-T-R and R-S-R stay inside one business is not checked: it holds by construction of `business`.)
    stars_per_business = np.array([len(np.unique(star_group[business == b])) for b in np.unique(business)])
    checks = {
        "rur_is_union_of_cliques": is_union_of_cliques(rur),
        "rsr_is_union_of_cliques": is_union_of_cliques(rsr),
        "rtr_is_union_of_cliques": is_union_of_cliques(rtr),
        "at_most_5_star_groups_per_business": bool(stars_per_business.max() <= 5),
    }
    return Entities(reviewer, business, star_group, time_window, checks)


def incidence(groups: np.ndarray, n_groups: int | None = None) -> sp.csr_matrix:
    """Review x group 0/1 matrix."""
    n = len(groups)
    k = int(groups.max()) + 1 if n_groups is None else n_groups
    return sp.csr_matrix((np.ones(n), (np.arange(n), groups)), shape=(n, k))


def reviewer_business_matrix(ent: Entities) -> sp.csr_matrix:
    """Reviewer x business count of reviews (the bipartite graph used for ring detection)."""
    n_u = int(ent.reviewer.max()) + 1
    n_b = int(ent.business.max()) + 1
    m = sp.csr_matrix((np.ones(len(ent.reviewer)), (ent.reviewer, ent.business)), shape=(n_u, n_b))
    m.sum_duplicates()
    return m


def load_schema(path: Path = SCHEMA_PATH) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def build_typed_graph(ent: Entities, x: np.ndarray | None = None) -> nx.MultiDiGraph:
    """Materialise the ontology as a typed networkx graph (entity level, review to review links omitted)."""
    g = nx.MultiDiGraph()
    for i in range(len(ent.reviewer)):
        attrs = {"type": "Review"}
        if x is not None:
            attrs["features_32"] = x[i]
        g.add_node(("Review", i), **attrs)
    for kind, arr in (
        ("Reviewer", ent.reviewer),
        ("Business", ent.business),
        ("StarGroup", ent.star_group),
        ("TimeWindow", ent.time_window),
    ):
        for k in np.unique(arr):
            g.add_node((kind, int(k)), type=kind)
    for i in range(len(ent.reviewer)):
        g.add_edge(("Reviewer", int(ent.reviewer[i])), ("Review", i), type="WROTE")
        g.add_edge(("Review", i), ("Business", int(ent.business[i])), type="ABOUT")
        g.add_edge(("Review", i), ("StarGroup", int(ent.star_group[i])), type="IN_STAR_GROUP")
        g.add_edge(("Review", i), ("TimeWindow", int(ent.time_window[i])), type="IN_TIME_WINDOW")
    rb = reviewer_business_matrix(ent)
    co = sp.triu((rb > 0).astype(np.float64) @ (rb > 0).astype(np.float64).T, k=1).tocoo()
    for u, v, w in zip(co.row, co.col, co.data, strict=True):
        g.add_edge(("Reviewer", int(u)), ("Reviewer", int(v)), type="CO_REVIEWED", weight=int(w))
    return g


def validate_against_schema(g: nx.MultiDiGraph, schema: dict) -> list[str]:
    """Return a list of violations: unknown node or edge types, or edges between the wrong node types."""
    errors = []
    node_types = set(schema["node_types"])
    rels = schema["relation_types"]
    for n, d in g.nodes(data=True):
        if d.get("type") not in node_types:
            errors.append(f"node {n} has unknown type {d.get('type')}")
    for u, v, d in g.edges(data=True):
        t = d.get("type")
        if t not in rels:
            errors.append(f"edge {u}->{v} has unknown type {t}")
            continue
        want = rels[t]["between"]
        got = [g.nodes[u]["type"], g.nodes[v]["type"]]
        if got != want:
            errors.append(f"edge {t} {u}->{v} connects {got}, schema says {want}")
    return errors
