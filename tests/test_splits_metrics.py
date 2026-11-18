import numpy as np

from frgraph.graph import derive_entities
from frgraph.metrics import at_threshold, evaluate, threshold_at_fpr
from frgraph.splits import cross_part_edges, group_split, random_split


def test_group_split_keeps_groups_whole():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 300, size=3000)
    part = group_split(groups, seed=1)
    for g in np.unique(groups):
        assert len(np.unique(part[groups == g])) == 1


def test_group_split_fractions_and_determinism():
    groups = np.repeat(np.arange(1000), 3)
    a, b, c = group_split(groups, 7), group_split(groups, 7), group_split(groups, 8)
    assert (a == b).all() and not (a == c).all()
    shares = np.bincount(a) / len(a)
    assert np.allclose(shares, [0.6, 0.2, 0.2], atol=0.01)


def test_reviewer_split_has_no_same_reviewer_leakage_random_split_does(synthetic):
    data, _ = synthetic
    ent = derive_entities(data.rel)
    rur = data.rel["net_rur"]
    assert cross_part_edges(rur, group_split(ent.reviewer, 0)) == 0
    assert sum(cross_part_edges(rur, random_split(data.n, s)) for s in range(5)) > 0


def test_threshold_at_fpr_respects_budget():
    rng = np.random.default_rng(0)
    neg = rng.random(1000)
    for f in (0.01, 0.05, 0.2):
        t = threshold_at_fpr(neg, f)
        assert (neg > t).mean() <= f
        assert (neg > t).mean() >= f - 0.002


def test_threshold_with_ties():
    neg = np.array([0.5] * 10 + [0.1] * 90)
    t = threshold_at_fpr(neg, 0.05)
    assert (neg > t).mean() <= 0.05


def test_at_threshold_counts():
    y = np.array([1, 1, 0, 0])
    m = at_threshold(y, np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert m == {"recall": 0.5, "precision": 0.5, "fpr": 0.5, "flagged": 2}


def test_evaluate_perfect_scores():
    y = np.array([0] * 90 + [1] * 10)
    s = y + np.linspace(0, 0.1, 100)
    m = evaluate(y, s, y, s)
    assert np.isclose(m["pr_auc"], 1.0) and m["recall_at_fpr1pct"] == 1.0
    assert m["fpr_at_fpr1pct"] <= 0.01 and m["fpr_at_fpr5pct"] <= 0.05
