import numpy as np

from frgraph import experiment
from frgraph.graph import derive_entities


def test_build_blocks_do_not_depend_on_labels(synthetic, tmp_path, monkeypatch):
    data, _ = synthetic
    ent = derive_entities(data.rel)
    from frgraph.features import spectral_features

    monkeypatch.setattr(experiment, "spectral_features", lambda rel: spectral_features(rel, dim=4))
    a = experiment.build_blocks(data, ent, cache=tmp_path / "a.npz")
    data.y[:] = np.random.default_rng(0).permutation(data.y)
    b = experiment.build_blocks(data, ent, cache=tmp_path / "b.npz")
    for k in a:
        assert np.allclose(a[k][0], b[k][0], equal_nan=True), k


def test_cache_is_reused_only_with_matching_code_hash(synthetic, tmp_path, monkeypatch):
    data, _ = synthetic
    ent = derive_entities(data.rel)
    from frgraph.features import spectral_features

    monkeypatch.setattr(experiment, "spectral_features", lambda rel: spectral_features(rel, dim=4))
    cache = tmp_path / "c.npz"
    first = experiment.build_blocks(data, ent, cache=cache)
    calls = []
    monkeypatch.setattr(experiment, "structural_features", lambda *a: calls.append(1) or first["structure"])
    experiment.build_blocks(data, ent, cache=cache)
    assert calls == []  # same code: cache used
    monkeypatch.setattr(experiment, "code_hash", lambda: "changed")
    experiment.build_blocks(data, ent, cache=cache)
    assert calls == [1]  # code changed: recomputed


def test_accounts_seen_in_train():
    reviewer = np.array([0, 0, 1, 2, 2])
    part = np.array([0, 2, 2, 1, 0])
    assert experiment.accounts_seen_in_train(part, reviewer).tolist() == [True, True, False, True, True]


def test_rows_from_csv_round_trip(tmp_path):
    import csv

    rows = [
        {"split": "reviewer", "seed": s, "feature_set": "A: x", "n_features": 3} | {m: 0.5 + s for m in experiment.METRICS}
        for s in range(2)
    ]
    path = tmp_path / "runs.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert experiment.rows_from_csv(path) == rows


def test_summary_reports_worst_case_fpr():
    rows = [
        {"split": "business", "seed": s, "feature_set": "A: x", "n_features": 1} | {m: 0.01 * (s + 1) for m in experiment.METRICS}
        for s in range(3)
    ]
    s = experiment.summarise(rows)[0]
    assert np.isclose(s["fpr_at_fpr1pct_max"], 0.03)
    assert "(max 0.030)" in experiment.markdown_table([s], "business")
