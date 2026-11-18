import numpy as np

from frgraph.graph import derive_entities
from frgraph.rings import burst_ids, markdown, run_one


def test_ring_report_has_evidence(synthetic):
    data, _ = synthetic
    ent = derive_entities(data.rel)
    res = run_one("toy", ent.business, ent, data.y, data.rel)
    assert res["n_blocks"] >= 1
    g = res["blocks"][0]
    assert g["n_reviews_in_block"] > 0 and 0 <= g["fake_share"] <= 1
    ev = g["evidence"]
    assert ev["top_shared_businesses"] and ev["sample_reviewer_ids"]
    assert "| 1 |" in markdown([res])


def test_bursts_are_within_business(synthetic):
    data, truth = synthetic
    b = burst_ids(data.rel)
    for k in np.unique(b):
        assert len(np.unique(truth["business"][b == k])) == 1
