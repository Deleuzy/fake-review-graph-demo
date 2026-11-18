# fake-review-graph-demo

What do graph features add to fake review detection, measured the way a trust team has to measure it: at a fixed, low false positive rate, on splits that do not leak, over several seeds? And does dense subgraph ring detection find fake groups on a public benchmark?

This is a learning demo on the public YelpChi benchmark. It is not Trustpilot data and makes no claim about Trustpilot's systems.

## Short answer

<!-- BEGIN HEADLINE -->
* Split by reviewer, at 1% false positive rate: recall 0.532 with the 32 features (A) and 0.721 with all graph features added (E). PR-AUC 0.823 against 0.909.
* Best row at 1% false positive rate on the reviewer split: E: A + all graph features (recall 0.721 ± 0.019).
* A random split reports PR-AUC 0.831 (A) and 0.911 (E), against 0.823 and 0.909 on the reviewer split.
* Split by business (unseen businesses), at 1% false positive rate: recall 0.371 (A) and 0.450 (E), and the realised test false positive rate was 0.009 and 0.013 against the 0.010 target.
<!-- END HEADLINE -->

<!-- BEGIN RINGS -->
* Reviewer x business: 0 of 10 blocks are mostly fake (base rate 0.145). Fake share per block, in rank order: 0.00, 0.01, 0.01, 0.03, 0.05, 0.00, 0.00, 0.09, 0.11, 0.18.
* Reviewer x burst (business, star, month): 0 of 10 blocks are mostly fake (base rate 0.145). Fake share per block, in rank order: 0.01, 0.00, 0.04, 0.07, 0.00, 0.15, 0.25, 0.28, 0.20, 0.18.

Full evidence per block is in [results/rings.md](results/rings.md) and results/rings.json.
<!-- END RINGS -->

The discussion of these numbers is in [What helped and what did not](#what-helped-and-what-did-not).

## Data

YelpChi as released with CARE-GNN (Dou et al., CIKM 2020), originally collected by Rayana and Akoglu (KDD 2015) from Yelp reviews of hotels and restaurants in Chicago. The label is whether Yelp's own filter hid the review, which is the usual stand in for "fake" in this literature.

`scripts/download_data.py` fetches `YelpChi.zip` from the CARE-GNN GitHub repository and checks its SHA-256. The file is not committed.

**Licence and terms.** The CARE-GNN repository is released under Apache 2.0. The underlying reviews come from Yelp and were collected for research by Rayana and Akoglu; I found no separate licence for the data itself, so treat it as research use only, cite both papers, and do not redistribute it. That is why this repository downloads it instead of including it.

**What the public file contains, and what it does not.** Only review nodes: 32 handcrafted features per review (text, behaviour and rating statistics computed by the dataset authors), the label, and three review to review relations. There is no review text, no timestamp, no reviewer id and no business id. Everything below that needs those is derived from the relations, and says so.

<!-- BEGIN DATA -->
* 45,954 reviews, 14.5% labelled fake (filtered by Yelp).
* Derived: 29,431 reviewers, 209 businesses, 803 star groups, 1,835 time windows.
* Given edges: R-U-R 49,315, R-T-R 573,616, R-S-R 3,402,743.
* Only 0.30% of reviewers have a mix of fake and genuine reviews, so the label is almost a property of the account.
* Reviews from accounts with a single review: 48.1% of all reviews, 24.7% labelled fake, against 5.1% for accounts with more than one review.
* Same reviewer links from train to test (seed 0): random split 11,902, business split 12,053, reviewer split 0.
<!-- END DATA -->

## The review graph as an ontology

The schema is in [schema/review_graph.yaml](schema/review_graph.yaml). Every node and relation type is marked `given` (in the file) or `derived` (computed here, without labels).

| Type | Source | How |
|---|---|---|
| Review | given | the review nodes in the file |
| R-U-R | given | same reviewer |
| R-T-R | given | same business, same month |
| R-S-R | given | same business, same star rating |
| Reviewer, WROTE | derived | connected components of R-U-R. Checked: R-U-R is an exact union of cliques, so each component is one account |
| Business, ABOUT | derived | components of R-S-R plus R-T-R. Checked: both relations stay inside one business |
| StarGroup | derived | components of R-S-R (an exact union of cliques). The star value itself is not in the file |
| TimeWindow | derived | components of R-T-R. R-T-R is not a union of cliques, so a window is a chain of same month links, not a calendar month |
| CO_REVIEWED | derived | two reviewers who reviewed at least one business in common, weighted by how many |

`frgraph.graph.build_typed_graph` materialises this as a typed networkx graph and `validate_against_schema` checks every node and edge type against the YAML. The checks on how the relations were built are written to `results/graph_summary.json` each run.

## Method

**Model.** scikit-learn `HistGradientBoostingClassifier` (300 trees, learning rate 0.05, fixed settings, no tuning on test data).

**Feature sets.**

| Set | Contents |
|---|---|
| A | the 32 handcrafted features |
| B | A plus 13 structural graph features: log degree per relation, reviewer review and business counts, business review and reviewer counts, share of the business's reviews in the same star group and time window, co-review strength (most businesses shared with any one other reviewer, number of reviewers sharing two or more), PageRank of the review in the review graph, PageRank of the reviewer in the reviewer to business graph |
| C | A plus neighbour aggregates: for each relation, the mean of the neighbours' 32 features and the review's difference from that mean (192 columns, missing when a review has no neighbour in that relation) |
| D | A plus a 16 dimensional spectral embedding of the review graph (eigenvectors of the normalised adjacency of its largest component). I used this instead of node2vec to avoid a random walk dependency; it captures the same kind of position in the graph but is not identical |
| E | A plus B, C and D |
| F | the 13 structural graph features alone, no handcrafted features |

All graph features are label free, so they are computed once on the whole graph. That matches production, where the graph around a new review exists before anyone labels it.

**Splits.** 60% train, 20% validation, 20% test, five seeds, with whole groups kept together:

* **by reviewer** (main result): every review of an account is in one part. YelpChi labels are almost constant per account, so this is the split that stops the model from recognising an account it has already seen.
* **by business**: every review of a business is in one part. This is the harder "new business" case. It does not keep accounts together.
* **random**: shown only to measure how much a careless split overstates results.

The file has no timestamps, so a split by time is not possible here.

**Fixed false positive rate.** For each run the threshold is set on the validation set so that at most 1% (or 5%) of genuine validation reviews score above it. That threshold is then applied unchanged to the test set. The table reports recall and precision on the test set, and `results/runs.csv` also has the realised test false positive rate for every run. PR-AUC is reported as a threshold free summary.

## Results

Mean ± standard deviation over five seeds. Written by `python -m frgraph.experiment` to `results/table.md`, `results/summary.json` and `results/runs.csv`.

<!-- BEGIN TABLE -->
### Split by reviewer

| Features | # | Recall at 1% FPR | Precision at 1% FPR | Test FPR (target 1%) | Recall at 5% FPR | Precision at 5% FPR | Test FPR (target 5%) | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| A: 32 handcrafted features | 32 | 0.532 ± 0.024 | 0.901 ± 0.011 | 0.010 ± 0.001 | 0.766 ± 0.020 | 0.715 ± 0.018 | 0.052 ± 0.006 | 0.823 ± 0.012 |
| B: A + structural graph features | 45 | 0.632 ± 0.035 | 0.918 ± 0.007 | 0.010 ± 0.002 | 0.843 ± 0.015 | 0.733 ± 0.010 | 0.053 ± 0.004 | 0.875 ± 0.005 |
| C: A + neighbour aggregates | 224 | 0.710 ± 0.021 | 0.917 ± 0.006 | 0.011 ± 0.001 | 0.871 ± 0.007 | 0.746 ± 0.009 | 0.051 ± 0.003 | 0.904 ± 0.004 |
| D: A + spectral embedding | 48 | 0.618 ± 0.009 | 0.905 ± 0.012 | 0.011 ± 0.002 | 0.823 ± 0.013 | 0.724 ± 0.014 | 0.054 ± 0.003 | 0.866 ± 0.009 |
| E: A + all graph features | 253 | 0.721 ± 0.019 | 0.918 ± 0.010 | 0.011 ± 0.002 | 0.882 ± 0.010 | 0.747 ± 0.008 | 0.051 ± 0.003 | 0.909 ± 0.004 |
| F: structural graph features only | 13 | 0.081 ± 0.007 | 0.560 ± 0.029 | 0.011 ± 0.001 | 0.200 ± 0.020 | 0.402 ± 0.021 | 0.051 ± 0.004 | 0.351 ± 0.014 |

### Split by business

| Features | # | Recall at 1% FPR | Precision at 1% FPR | Test FPR (target 1%) | Recall at 5% FPR | Precision at 5% FPR | Test FPR (target 5%) | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| A: 32 handcrafted features | 32 | 0.371 ± 0.084 | 0.868 ± 0.034 | 0.009 ± 0.003 | 0.651 ± 0.068 | 0.670 ± 0.045 | 0.053 ± 0.011 | 0.723 ± 0.049 |
| B: A + structural graph features | 45 | 0.334 ± 0.093 | 0.839 ± 0.068 | 0.010 ± 0.003 | 0.637 ± 0.068 | 0.695 ± 0.042 | 0.046 ± 0.004 | 0.724 ± 0.067 |
| C: A + neighbour aggregates | 224 | 0.477 ± 0.079 | 0.859 ± 0.090 | 0.013 ± 0.009 | 0.769 ± 0.047 | 0.707 ± 0.076 | 0.054 ± 0.019 | 0.791 ± 0.088 |
| D: A + spectral embedding | 48 | 0.333 ± 0.073 | 0.860 ± 0.059 | 0.010 ± 0.006 | 0.591 ± 0.070 | 0.674 ± 0.047 | 0.048 ± 0.015 | 0.697 ± 0.030 |
| E: A + all graph features | 253 | 0.450 ± 0.092 | 0.854 ± 0.084 | 0.013 ± 0.007 | 0.751 ± 0.066 | 0.707 ± 0.065 | 0.053 ± 0.016 | 0.793 ± 0.068 |
| F: structural graph features only | 13 | 0.032 ± 0.024 | 0.295 ± 0.058 | 0.014 ± 0.011 | 0.118 ± 0.034 | 0.293 ± 0.034 | 0.048 ± 0.017 | 0.263 ± 0.018 |

### Split by random

| Features | # | Recall at 1% FPR | Precision at 1% FPR | Test FPR (target 1%) | Recall at 5% FPR | Precision at 5% FPR | Test FPR (target 5%) | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| A: 32 handcrafted features | 32 | 0.533 ± 0.018 | 0.899 ± 0.010 | 0.010 ± 0.001 | 0.767 ± 0.022 | 0.731 ± 0.011 | 0.049 ± 0.003 | 0.831 ± 0.013 |
| E: A + all graph features | 253 | 0.723 ± 0.009 | 0.924 ± 0.005 | 0.010 ± 0.001 | 0.879 ± 0.007 | 0.755 ± 0.010 | 0.050 ± 0.002 | 0.911 ± 0.002 |
<!-- END TABLE -->

## Ring detection

`frgraph.fraudar` implements FRAUDAR style greedy peeling (Hooi et al., KDD 2016): edges to a target are weighted 1 / log(5 + target degree), so links to popular businesses count less, which is how FRAUDAR resists camouflage. The algorithm repeatedly removes the node with the least remaining edge weight and keeps the densest intermediate set. Further blocks are found by deleting each block's edges and peeling again. The optional node prior from the paper is implemented and tested but not used in the reported runs.

Two runs were fixed before looking at the labels, each asking for 10 blocks:

1. **Reviewer x business**, the classic FRAUDAR setting.
2. **Reviewer x burst**, where a burst is the set of reviews of one business with the same star rating linked by the same month relation. This is the review bombing view.

Labels are used only afterwards, to score each block. Evidence per block (members, shared businesses, how many members are one review accounts, the share of review pairs posted in the same month window and with the same star rating) is in [results/rings.md](results/rings.md).

## What helped and what did not

Change against set A on the same split and seed (paired), from `results/runs.csv`:

<!-- BEGIN GAINS -->
| Split | Features | Change in recall at 1% FPR | Seeds better than A | Change in PR-AUC | Seeds better than A |
|---|---|---|---|---|---|
| reviewer | B: A + structural graph features | +0.100 | 5 of 5 | +0.052 | 5 of 5 |
| reviewer | C: A + neighbour aggregates | +0.178 | 5 of 5 | +0.080 | 5 of 5 |
| reviewer | D: A + spectral embedding | +0.086 | 5 of 5 | +0.043 | 5 of 5 |
| reviewer | E: A + all graph features | +0.189 | 5 of 5 | +0.085 | 5 of 5 |
| reviewer | F: structural graph features only | -0.451 | 0 of 5 | -0.472 | 0 of 5 |
| business | B: A + structural graph features | -0.037 | 1 of 5 | +0.001 | 3 of 5 |
| business | C: A + neighbour aggregates | +0.106 | 5 of 5 | +0.067 | 4 of 5 |
| business | D: A + spectral embedding | -0.038 | 1 of 5 | -0.026 | 1 of 5 |
| business | E: A + all graph features | +0.079 | 5 of 5 | +0.069 | 5 of 5 |
| business | F: structural graph features only | -0.339 | 0 of 5 | -0.461 | 0 of 5 |
<!-- END GAINS -->

* **Neighbour aggregates helped most, and they held up on unseen businesses.** Describing a review by what the reviews around it look like (same account, same business and month, same business and star rating) was the only feature family that improved recall at 1% false positive rate in every seed on both grouped splits. The context of a review carries signal that its own 32 features miss.
* **Structural features and the spectral embedding helped on the reviewer split but not on the business split.** On unseen businesses they were flat or slightly worse in most seeds. A likely reason is that much of what they encode is which business a review belongs to (R-S-R makes up most of the edges), and that does not transfer to a business the model has never seen. I have not tested that explanation.
* **Graph structure alone is weak** (set F). On this data the graph is context for the review features, not a replacement for them.
* **The random split overstated results much less than I expected.** I expected a large gap because labels are nearly constant per account. It did not appear, probably because the 32 features already include account level behaviour, so seeing the account's other reviews in training adds little. Grouped splits are still the right default: the business split is where performance really drops, the spread across seeds is several times larger, and a threshold set on validation data does not hold its false positive rate as tightly on new businesses.
* **Dense block detection did not find fake rings here.** In both runs the densest blocks were mostly genuine prolific reviewers, and no block was mostly fake. YelpChi's filtered accounts are mostly one review accounts, so they do not form dense co-review structure. Two lessons: a density score on its own would flag honest power users first, which is exactly the false positive Trustpilot cannot afford; and a ring detector can only be evaluated against labels that describe collusion, which Yelp's filter labels do not.

## Honest limits

* **This is a benchmark, not Trustpilot data.** Yelp hotels and restaurants in Chicago, collected around 2015. Trustpilot has other signals (invitations, devices, IPs, business replies, verified purchases) that change what works.
* **The labels are Yelp's filter, not ground truth.** A model that agrees with the filter learns the filter, including its mistakes. In this data, accounts with a single review are labelled fake far more often than others (numbers under Data), which shapes everything here.
* **No text and no timestamps** in the public graph version, so no language model features, no split by time, and the time window node is only a chain of same month links.
* **One model family, fixed settings.** No hyperparameter search, so the gaps between rows are between feature sets under one reasonable setting, not between best possible models.
* **No GNN row.** The plan allowed one if time permitted; I left it out rather than add an untuned one.
* **Graph features are computed transductively** on the whole graph, label free. That is realistic for a live system, but it means test reviews contribute to, for example, PageRank.
* **Five seeds** give a spread, not a confidence interval.

## If I were doing this at Trustpilot

1. Start from the current model and how the Content Integrity specialists label cases, and find out which errors cost most: removing a genuine review, or letting a fake one through.
2. Agree the measure first: recall and precision at a fixed false positive rate, on splits by time and by business, with the threshold set on held out data, exactly as above.
3. Build the review graph from the richer signals Trustpilot has (invitation links, devices, IP ranges, timing), keep the ontology explicit so analysts can read it, and add graph features to the existing model one family at a time, keeping only what moves the measure.
4. Use dense block detection as a lead generator for analysts, not as an automatic decision, and show the evidence per group. On this benchmark the densest groups were mostly genuine power reviewers, which is exactly the false positive risk.
5. Track label quality: where the specialists disagree with the automatic filter, those cases are the most valuable training data.
6. Try a GNN only if the features plateau, and test it on disguised fraudsters.

## Run it

```bash
uv sync
uv run python scripts/download_data.py
uv run python -m frgraph.experiment --seeds 5
uv run python -m frgraph.rings
uv run python -m frgraph.report
uv run pytest
```

The experiment took about ten minutes on an M1 laptop with two threads (`OMP_NUM_THREADS=2`). Graph features are cached in `data/cache/`. Tests use a small synthetic graph built the same way as YelpChi's relations, so they run in CI without the dataset. CI also checks that the numbers in this README match the committed results files.

## References

* Dou, Liu, Sun, Deng, Peng, Yu. Enhancing Graph Neural Network based Fraud Detectors against Camouflaged Fraudsters. CIKM 2020. Code and data: https://github.com/YingtongDou/CARE-GNN
* Rayana, Akoglu. Collective Opinion Spam Detection: Bridging Review Networks and Metadata. KDD 2015.
* Hooi, Song, Beutel, Shah, Shin, Faloutsos. FRAUDAR: Bounding Graph Fraud in the Face of Camouflage. KDD 2016.

## Licence

Code: MIT. Data: not included, see above.
