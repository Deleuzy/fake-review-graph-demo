"""Fill the generated sections of README.md from the files in results/.

Every number in those sections comes from results/*.json or results/*.md, which the experiment and
ring scripts write. `--check` fails if README.md is out of date, so CI catches hand edited numbers.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RESULTS = Path("results")
README = Path("README.md")


def pct(v: float, digits: int = 1) -> str:
    return f"{100 * v:.{digits}f}%"


def find(summary, split, prefix):
    return next(s for s in summary if s["split"] == split and s["feature_set"].startswith(prefix))


def gains_table() -> str:
    """Paired by seed: each set minus set A on the same split and seed."""
    import csv

    with (RESULTS / "runs.csv").open() as f:
        rows = list(csv.DictReader(f))
    by = {(r["split"], r["feature_set"], r["seed"]): r for r in rows}
    out = [
        "| Split | Features | Change in recall at 1% FPR | Seeds better than A | Change in PR-AUC | Seeds better than A |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("reviewer", "business"):
        sets = sorted({r["feature_set"] for r in rows if r["split"] == split and not r["feature_set"].startswith("A:")})
        for fs in sets:
            seeds = sorted({r["seed"] for r in rows if r["split"] == split and r["feature_set"] == fs})
            cells = []
            for m in ("recall_at_fpr1pct", "pr_auc"):
                d = [float(by[(split, fs, s)][m]) - float(by[(split, "A: 32 handcrafted features", s)][m]) for s in seeds]
                cells += [f"{sum(d) / len(d):+.3f}", f"{sum(x > 0 for x in d)} of {len(d)}"]
            out.append(f"| {split} | {fs} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def sections() -> dict[str, str]:
    summary = json.loads((RESULTS / "summary.json").read_text())
    gs = json.loads((RESULTS / "graph_summary.json").read_text())
    ring_file = json.loads((RESULTS / "rings.json").read_text())
    rings, ctx = ring_file["runs"], ring_file["context"]
    table = (RESULTS / "table.md").read_text().strip()

    a = find(summary, "reviewer", "A:")
    e = find(summary, "reviewer", "E:")
    ba = find(summary, "business", "A:")
    be = find(summary, "business", "E:")
    ra = find(summary, "random", "A:")
    re_ = find(summary, "random", "E:")
    best = max((s for s in summary if s["split"] == "reviewer"), key=lambda s: s["recall_at_fpr1pct_mean"])
    ec = gs["entity_counts"]
    leak = gs["split_leakage_seed0"]

    data = (
        f"* {gs['n_reviews']:,} reviews, {pct(gs['positive_rate'])} labelled fake (filtered by Yelp).\n"
        f"* Derived: {ec['Reviewer']:,} reviewers, {ec['Business']} businesses, {ec['StarGroup']} star groups, "
        f"{ec['TimeWindow']:,} time windows.\n"
        f"* Given edges: R-U-R {gs['edges']['net_rur']:,}, R-T-R {gs['edges']['net_rtr']:,}, R-S-R {gs['edges']['net_rsr']:,}.\n"
        f"* Only {pct(gs['reviewer_label_mixed_share'], 2)} of reviewers have a mix of fake and genuine reviews, "
        "so the label is almost a property of the account.\n"
        f"* Reviews from accounts with a single review: {pct(ctx['share_reviews_from_one_review_accounts'])} of all reviews, "
        f"{pct(ctx['fake_rate_one_review_accounts'])} labelled fake, against {pct(ctx['fake_rate_multi_review_accounts'])} "
        "for accounts with more than one review.\n"
        f"* Same reviewer links from train to test (seed 0): random split {leak['random']['rur_edges_train_test']:,}, "
        f"business split {leak['business']['rur_edges_train_test']:,}, "
        f"reviewer split {leak['reviewer']['rur_edges_train_test']:,}."
    )

    headline = (
        f"* Split by reviewer, at 1% false positive rate: recall {a['recall_at_fpr1pct_mean']:.3f} with the 32 features (A) "
        f"and {e['recall_at_fpr1pct_mean']:.3f} with all graph features added (E). "
        f"PR-AUC {a['pr_auc_mean']:.3f} against {e['pr_auc_mean']:.3f}.\n"
        f"* Best row at 1% false positive rate on the reviewer split: {best['feature_set']} "
        f"(recall {best['recall_at_fpr1pct_mean']:.3f} ± {best['recall_at_fpr1pct_std']:.3f}).\n"
        f"* A random split reports PR-AUC {ra['pr_auc_mean']:.3f} (A) and {re_['pr_auc_mean']:.3f} (E), "
        f"against {a['pr_auc_mean']:.3f} and {e['pr_auc_mean']:.3f} on the reviewer split.\n"
        f"* Split by business (unseen businesses), at 1% false positive rate: recall {ba['recall_at_fpr1pct_mean']:.3f} (A) "
        f"and {be['recall_at_fpr1pct_mean']:.3f} (E), and the realised test false positive rate was "
        f"{ba['fpr_at_fpr1pct_mean']:.3f} and {be['fpr_at_fpr1pct_mean']:.3f} against the 0.010 target."
    )

    ring_lines = []
    for r in rings:
        ring_lines.append(
            f"* {r['graph']}: {r['n_mostly_fake']} of {r['n_blocks']} blocks are mostly fake "
            f"(base rate {r['base_fake_rate']:.3f}). Fake share per block, in rank order: "
            + ", ".join(f"{g['fake_share']:.2f}" for g in r["blocks"])
            + "."
        )
    ring_lines.append("\nFull evidence per block is in [results/rings.md](results/rings.md) and results/rings.json.")
    return {"DATA": data, "HEADLINE": headline, "TABLE": table, "GAINS": gains_table(), "RINGS": "\n".join(ring_lines)}


def render(text: str) -> str:
    for name, body in sections().items():
        pat = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(\n<!-- END {name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README is missing the {name} markers")
        text = pat.sub(lambda m, body=body: m.group(1) + body + m.group(2), text)
    return text


def main(check: bool = False) -> None:
    old = README.read_text()
    new = render(old)
    if check:
        if new != old:
            sys.exit("README.md is out of date with results/. Run: python -m frgraph.report")
        print("README matches results/")
        return
    README.write_text(new)
    print("README updated")


if __name__ == "__main__":
    main("--check" in sys.argv)
