"""Ring detection with FRAUDAR style peeling, with evidence per flagged group.

Two pre specified runs, both without a node prior and with k = 10 blocks:
  1. reviewer x business (the classic FRAUDAR setting: accounts x targets)
  2. reviewer x burst, where a burst is a set of reviews of one business with the same star rating
     linked by the same month relation (components of R-T-R AND R-S-R), the review bombing view.
Labels are only used afterwards, to score each block. They never steer the search.
Usage: python -m frgraph.rings
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .data import DEFAULT_PATH, load_yelpchi
from .fraudar import Block, detect_blocks
from .graph import Entities, components, derive_entities

RESULTS = Path("results")
K = 10
MOSTLY_FAKE = 0.5


def burst_ids(rel) -> np.ndarray:
    return components(rel["net_rtr"].multiply(rel["net_rsr"]))


def pair_share(adj: sp.csr_matrix, idx: np.ndarray, same: np.ndarray) -> float:
    """Among pairs of block reviews about the same business, the share linked by adj."""
    if len(idx) < 2:
        return float("nan")
    sub = adj[idx][:, idx]
    linked = sub.nnz / 2
    _, counts = np.unique(same[idx], return_counts=True)
    pairs = float((counts * (counts - 1) / 2).sum())
    return linked / pairs if pairs else float("nan")


def block_evidence(b: Block, target_of_review: np.ndarray, ent: Entities, y: np.ndarray, rel, rank: int) -> dict:
    in_rows = np.isin(ent.reviewer, b.rows)
    idx = np.where(in_rows & np.isin(target_of_review, b.cols))[0]
    n_reviews_of = np.bincount(ent.reviewer)
    biz, biz_counts = np.unique(ent.business[idx], return_counts=True)
    top = np.argsort(-biz_counts)[:5]
    acct_reviews = n_reviews_of[b.rows]
    return {
        "rank": rank,
        "density": round(b.density, 4),
        "n_reviewers": int(len(b.rows)),
        "n_targets": int(len(b.cols)),
        "n_reviews_in_block": int(len(idx)),
        "fake_share": float(y[idx].mean()) if len(idx) else float("nan"),
        "mostly_fake": bool(len(idx) and y[idx].mean() > MOSTLY_FAKE),
        "evidence": {
            "sample_reviewer_ids": [int(r) for r in b.rows[:10]],
            "top_shared_businesses": [{"business_id": int(biz[t]), "reviews_in_block": int(biz_counts[t])} for t in top],
            "median_reviews_per_member_account": float(np.median(acct_reviews)),
            "share_one_review_accounts": float(np.mean(acct_reviews == 1)),
            "same_month_pair_share": pair_share(rel["net_rtr"], idx, ent.business),
            "same_star_pair_share": pair_share(rel["net_rsr"], idx, ent.business),
        },
    }


def run_one(name: str, target: np.ndarray, ent: Entities, y: np.ndarray, rel) -> dict:
    n_u = int(ent.reviewer.max()) + 1
    m = sp.csr_matrix((np.ones(len(target)), (ent.reviewer, target)), shape=(n_u, int(target.max()) + 1))
    blocks = detect_blocks(m, k=K)
    groups = [block_evidence(b, target, ent, y, rel, i + 1) for i, b in enumerate(blocks)]
    return {
        "graph": name,
        "k": K,
        "base_fake_rate": float(y.mean()),
        "n_blocks": len(groups),
        "n_mostly_fake": int(sum(g["mostly_fake"] for g in groups)),
        "blocks": groups,
    }


def markdown(res: list[dict]) -> str:
    out = []
    for r in res:
        out.append(
            f"### {r['graph']}\n\nBase fake rate {r['base_fake_rate']:.3f}. "
            f"Blocks found: {r['n_blocks']}, mostly fake (share above {MOSTLY_FAKE}): {r['n_mostly_fake']}.\n"
        )
        out.append(
            "| Rank | Reviewers | Targets | Reviews | Fake share | Median reviews per account | "
            "One review accounts | Same month pairs | Same star pairs |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|")
        for g in r["blocks"]:
            e = g["evidence"]
            out.append(
                f"| {g['rank']} | {g['n_reviewers']} | {g['n_targets']} | {g['n_reviews_in_block']} | "
                f"{g['fake_share']:.2f} | {e['median_reviews_per_member_account']:.0f} | "
                f"{e['share_one_review_accounts']:.2f} | {e['same_month_pair_share']:.2f} | {e['same_star_pair_share']:.2f} |"
            )
        out.append("")
    return "\n".join(out)


def main(path: Path = DEFAULT_PATH) -> None:
    data = load_yelpchi(path)
    ent = derive_entities(data.rel)
    res = [
        run_one("Reviewer x business", ent.business, ent, data.y, data.rel),
        run_one("Reviewer x burst (business, star, month)", burst_ids(data.rel), ent, data.y, data.rel),
    ]
    n_rev = np.bincount(ent.reviewer)[ent.reviewer]
    context = {
        "n_reviewers": int(ent.reviewer.max()) + 1,
        "fake_rate_one_review_accounts": float(data.y[n_rev == 1].mean()),
        "fake_rate_multi_review_accounts": float(data.y[n_rev > 1].mean()),
        "share_reviews_from_one_review_accounts": float((n_rev == 1).mean()),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "rings.json").write_text(json.dumps({"context": context, "runs": res}, indent=2) + "\n")
    (RESULTS / "rings.md").write_text(markdown(res))
    print(markdown(res))


if __name__ == "__main__":
    main()
