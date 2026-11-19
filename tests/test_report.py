import json

import pytest

from frgraph import experiment, report


def fake_results(tmp_path):
    rows = []
    for split in experiment.SPLITS:
        sets = experiment.RANDOM_SPLIT_SETS if split == "random" else tuple(experiment.FEATURE_SETS)
        for seed in range(2):
            for i, fs in enumerate(sets):
                rows.append(
                    {"split": split, "seed": seed, "feature_set": fs, "n_features": 32 + i}
                    | {m: 0.1 * (i + 1) + 0.01 * seed for m in experiment.METRICS}
                )
    summary = experiment.summarise(rows)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "table.md").write_text(experiment.markdown_table(summary, "reviewer"))
    with (tmp_path / "runs.csv").open("w", newline="") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    leak = {
        s: {
            "rur_edges_train_test": 0,
            "test_share": 0.2,
            "test_positive_rate": 0.1,
            "test_reviews_from_accounts_seen_in_train": 0.4,
        }
        for s in ("reviewer", "business", "random")
    }
    gs = {
        "n_reviews": 10,
        "positive_rate": 0.1,
        "entity_counts": {"Reviewer": 5, "Business": 2, "BusinessWith2PlusReviews": 1, "StarGroup": 3, "TimeWindow": 4},
        "edges": {"net_rur": 1, "net_rtr": 2, "net_rsr": 3},
        "reviewers_with_2plus_reviews": 3,
        "reviewer_label_mixed_share_among_2plus": 0.0,
        "single_review_businesses": 1,
        "split_leakage_seed0": leak,
    }
    (tmp_path / "graph_summary.json").write_text(json.dumps(gs))
    ctx = {
        "n_reviewers": 5,
        "fake_rate_one_review_accounts": 0.2,
        "fake_rate_multi_review_accounts": 0.05,
        "share_reviews_from_one_review_accounts": 0.5,
    }
    runs = [
        {
            "graph": "g",
            "n_blocks": 1,
            "n_mostly_fake": 0,
            "base_fake_rate": 0.1,
            "blocks": [{"fake_share": 0.0, "n_reviewers": 2}],
        }
    ]
    (tmp_path / "rings.json").write_text(json.dumps({"context": ctx, "runs": runs}))


def test_summary_mean_and_spread():
    rows = [
        {"split": "reviewer", "seed": s, "feature_set": "A: x", "n_features": 1} | {m: float(s) for m in experiment.METRICS}
        for s in range(3)
    ]
    s = experiment.summarise(rows)[0]
    assert s["pr_auc_mean"] == 1.0 and s["pr_auc_std"] == 1.0 and s["n_seeds"] == 3


def test_render_fills_markers_and_check_detects_drift(tmp_path, monkeypatch):
    fake_results(tmp_path)
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    readme = tmp_path / "README.md"
    names = ("HEADLINE", "RINGS", "DATA", "TABLE", "GAINS")
    readme.write_text("\n".join(f"<!-- BEGIN {n} -->\nold\n<!-- END {n} -->" for n in names))
    monkeypatch.setattr(report, "README", readme)
    with pytest.raises(SystemExit):
        report.main(check=True)
    report.main()
    text = readme.read_text()
    assert "old" not in text and "0.100" in text
    assert "| reviewer | E: A + all graph features | +0.400 | 2 of 2 |" in text
    report.main(check=True)


def test_render_requires_markers(tmp_path, monkeypatch):
    fake_results(tmp_path)
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    with pytest.raises(SystemExit):
        report.render("no markers here")
