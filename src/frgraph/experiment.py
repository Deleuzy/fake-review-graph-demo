"""Measured comparison: gradient boosting on the 32 YelpChi features, with and without graph features.

Every number reported in the README is written by this module to results/.
Usage: python -m frgraph.experiment [--seeds 5]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import DEFAULT_PATH, load_yelpchi
from .features import neighbour_features, spectral_features, structural_features
from .graph import derive_entities
from .metrics import evaluate
from .splits import cross_part_edges, group_split, random_split

RESULTS = Path("results")
CACHE = Path("data/cache/features.npz")

FEATURE_SETS = {
    "A: 32 handcrafted features": ["base"],
    "B: A + structural graph features": ["base", "structure"],
    "C: A + neighbour aggregates": ["base", "neighbour"],
    "D: A + spectral embedding": ["base", "spectral"],
    "E: A + all graph features": ["base", "structure", "neighbour", "spectral"],
    "F: structural graph features only": ["structure"],
}
SPLITS = ("reviewer", "business", "business_new_accounts", "random")
# business_new_accounts is not a separate split: it scores the business split models only on validation and test
# reviews whose account has no review in the training part, so account memory cannot help.
SPLIT_TITLES = {
    "reviewer": "Split by reviewer",
    "business": "Split by business",
    "business_new_accounts": "Split by business, scored only on accounts with no review in training",
    "random": "Random split (leakage check only)",
}
# The random split is only there to show leakage, so it runs for the first and the full model only.
RANDOM_SPLIT_SETS = ("A: 32 handcrafted features", "E: A + all graph features")
METRICS = (
    "recall_at_fpr1pct",
    "precision_at_fpr1pct",
    "fpr_at_fpr1pct",
    "recall_at_fpr5pct",
    "precision_at_fpr5pct",
    "fpr_at_fpr5pct",
    "pr_auc",
    "roc_auc",
)


def code_hash() -> str:
    """Hash of the code that computes features, so a stale cache is never reused."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("data.py", "graph.py", "features.py"):
        h.update((here / name).read_bytes())
    return h.hexdigest()


def build_blocks(data, ent, cache: Path | None = None) -> dict[str, tuple[np.ndarray, list[str]]]:
    cache = CACHE if cache is None else cache
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if "code_hash" in z and str(z["code_hash"]) == code_hash() and len(z["base"]) == data.n:
            return {k: (z[k], list(z[f"{k}_names"])) for k in ("base", "structure", "neighbour", "spectral")}
    blocks = {
        "base": (data.X, [f"f{j}" for j in range(data.X.shape[1])]),
        "structure": structural_features(data.rel, ent),
        "neighbour": neighbour_features(data.rel, data.X),
        "spectral": spectral_features(data.rel),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        code_hash=np.array(code_hash()),
        **{k: v[0] for k, v in blocks.items()},
        **{f"{k}_names": v[1] for k, v in blocks.items()},
    )
    return blocks


def make_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def accounts_seen_in_train(part: np.ndarray, reviewer: np.ndarray) -> np.ndarray:
    """True for each review whose account has at least one review in the training part."""
    seen = np.zeros(int(reviewer.max()) + 1, dtype=bool)
    seen[reviewer[part == 0]] = True
    return seen[reviewer]


def split_for(name: str, seed: int, ent, n: int) -> np.ndarray:
    if name == "reviewer":
        return group_split(ent.reviewer, seed)
    if name == "business":
        return group_split(ent.business, seed)
    return random_split(n, seed)


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    keys = sorted({(r["split"], r["feature_set"]) for r in rows}, key=lambda k: (SPLITS.index(k[0]), k[1]))
    for split, fs in keys:
        sub = [r for r in rows if r["split"] == split and r["feature_set"] == fs]
        s = {"split": split, "feature_set": fs, "n_seeds": len(sub), "n_features": sub[0]["n_features"]}
        for m in METRICS:
            v = np.array([r[m] for r in sub], dtype=float)
            s[f"{m}_mean"] = float(v.mean())
            s[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            s[f"{m}_max"] = float(v.max())
        out.append(s)
    return out


def fmt(s: dict, m: str) -> str:
    return f"{s[m + '_mean']:.3f} ± {s[m + '_std']:.3f}"


def markdown_table(summary: list[dict], split: str) -> str:
    head = (
        "| Features | # | Recall at 1% FPR | Precision at 1% FPR | Test FPR (target 1%) "
        "| Recall at 5% FPR | Precision at 5% FPR | Test FPR (target 5%) | PR-AUC (average precision) |\n"
    )
    head += "|---|---|---|---|---|---|---|---|---|\n"
    lines = [
        f"| {s['feature_set']} | {s['n_features']} | {fmt(s, 'recall_at_fpr1pct')} | {fmt(s, 'precision_at_fpr1pct')} "
        f"| {fmt(s, 'fpr_at_fpr1pct')} (max {s['fpr_at_fpr1pct_max']:.3f}) | {fmt(s, 'recall_at_fpr5pct')} "
        f"| {fmt(s, 'precision_at_fpr5pct')} | {fmt(s, 'fpr_at_fpr5pct')} (max {s['fpr_at_fpr5pct_max']:.3f}) "
        f"| {fmt(s, 'pr_auc')} |"
        for s in summary
        if s["split"] == split
    ]
    return head + "\n".join(lines) + "\n"


def write_summary(rows: list[dict]) -> None:
    summary = summarise(rows)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    present = [split for split in SPLITS if any(s["split"] == split for s in summary)]
    md = [f"### {SPLIT_TITLES[split]}\n\n" + markdown_table(summary, split) for split in present]
    (RESULTS / "table.md").write_text("\n".join(md))


def rows_from_csv(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ("split", "feature_set"):
                r[k] = float(v)
        r["seed"], r["n_features"] = int(r["seed"]), int(r["n_features"])
    return rows


def main(seeds: int = 5, path: Path = DEFAULT_PATH) -> None:
    t0 = time.time()
    data = load_yelpchi(path)
    ent = derive_entities(data.rel)
    blocks = build_blocks(data, ent)
    RESULTS.mkdir(exist_ok=True)

    # Graph summary and a leakage check: how many same reviewer links cross from train to test.
    leak = {}
    for split in ("reviewer", "business", "random"):
        part = split_for(split, 0, ent, data.n)
        leak[split] = {
            "rur_edges_train_test": cross_part_edges(data.rel["net_rur"], part),
            "test_share": float((part == 2).mean()),
            "test_positive_rate": float(data.y[part == 2].mean()),
            "test_reviews_from_accounts_seen_in_train": float(accounts_seen_in_train(part, ent.reviewer)[part == 2].mean()),
        }
    per_account = np.bincount(ent.reviewer)
    mixed = (lambda f: (f > 0) & (f < 1))(np.bincount(ent.reviewer, weights=data.y) / per_account)
    graph_summary = {
        "n_reviews": data.n,
        "positive_rate": float(data.y.mean()),
        "entity_counts": ent.counts(),
        "derivation_checks": ent.checks,
        "edges": {k: int(v.nnz // 2) for k, v in data.rel.items()},
        "reviewers_with_2plus_reviews": int((per_account >= 2).sum()),
        "reviewer_label_mixed_share_among_2plus": float(mixed[per_account >= 2].mean()),
        "single_review_businesses": int((np.bincount(ent.business) == 1).sum()),
        "split_leakage_seed0": leak,
    }
    (RESULTS / "graph_summary.json").write_text(json.dumps(graph_summary, indent=2) + "\n")

    rows = []
    for split in ("reviewer", "business", "random"):
        sets = RANDOM_SPLIT_SETS if split == "random" else tuple(FEATURE_SETS)
        for seed in range(seeds):
            part = split_for(split, seed, ent, data.n)
            tr, va, te = part == 0, part == 1, part == 2
            seen = accounts_seen_in_train(part, ent.reviewer)
            for fs in sets:
                x = np.column_stack([blocks[b][0] for b in FEATURE_SETS[fs]])
                model = make_model(seed).fit(x[tr], data.y[tr])
                s = model.predict_proba(x)[:, 1]
                m = evaluate(data.y[va], s[va], data.y[te], s[te])
                rows.append({"split": split, "seed": seed, "feature_set": fs, "n_features": x.shape[1], **m})
                print(f"{split:8s} seed={seed} {fs:40s} rec@1%={m['recall_at_fpr1pct']:.3f} pr_auc={m['pr_auc']:.3f}", flush=True)
                if split == "business":
                    new = ~seen
                    m2 = evaluate(data.y[va & new], s[va & new], data.y[te & new], s[te & new])
                    rows.append(
                        {"split": "business_new_accounts", "seed": seed, "feature_set": fs, "n_features": x.shape[1], **m2}
                    )

    with (RESULTS / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_summary(rows)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--from-runs", action="store_true", help="rebuild summary.json and table.md from results/runs.csv")
    a = ap.parse_args()
    if a.from_runs:
        write_summary(rows_from_csv(RESULTS / "runs.csv"))
    else:
        main(a.seeds, a.data)
