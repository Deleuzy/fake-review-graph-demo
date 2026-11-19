### Split by reviewer

| Features | # | Recall at 1% FPR | Precision at 1% FPR | Test FPR (target 1%) | Recall at 5% FPR | Precision at 5% FPR | Test FPR (target 5%) | PR-AUC (average precision) |
|---|---|---|---|---|---|---|---|---|
| A: 32 handcrafted features | 32 | 0.532 ± 0.024 | 0.901 ± 0.011 | 0.010 ± 0.001 (max 0.012) | 0.766 ± 0.020 | 0.715 ± 0.018 | 0.052 ± 0.006 (max 0.059) | 0.823 ± 0.012 |
| B: A + structural graph features | 45 | 0.632 ± 0.035 | 0.918 ± 0.007 | 0.010 ± 0.002 (max 0.012) | 0.843 ± 0.015 | 0.733 ± 0.010 | 0.053 ± 0.004 (max 0.057) | 0.875 ± 0.005 |
| C: A + neighbour aggregates | 224 | 0.710 ± 0.021 | 0.917 ± 0.006 | 0.011 ± 0.001 (max 0.012) | 0.871 ± 0.007 | 0.746 ± 0.009 | 0.051 ± 0.003 (max 0.054) | 0.904 ± 0.004 |
| D: A + spectral embedding | 48 | 0.618 ± 0.009 | 0.905 ± 0.012 | 0.011 ± 0.002 (max 0.014) | 0.823 ± 0.013 | 0.724 ± 0.014 | 0.054 ± 0.003 (max 0.057) | 0.866 ± 0.009 |
| E: A + all graph features | 253 | 0.721 ± 0.019 | 0.918 ± 0.010 | 0.011 ± 0.002 (max 0.012) | 0.882 ± 0.010 | 0.747 ± 0.008 | 0.051 ± 0.003 (max 0.055) | 0.909 ± 0.004 |
| F: structural graph features only | 13 | 0.081 ± 0.007 | 0.560 ± 0.029 | 0.011 ± 0.001 (max 0.011) | 0.200 ± 0.020 | 0.402 ± 0.021 | 0.051 ± 0.004 (max 0.057) | 0.351 ± 0.014 |

### Split by business

| Features | # | Recall at 1% FPR | Precision at 1% FPR | Test FPR (target 1%) | Recall at 5% FPR | Precision at 5% FPR | Test FPR (target 5%) | PR-AUC (average precision) |
|---|---|---|---|---|---|---|---|---|
| A: 32 handcrafted features | 32 | 0.371 ± 0.084 | 0.868 ± 0.034 | 0.009 ± 0.003 (max 0.013) | 0.651 ± 0.068 | 0.670 ± 0.045 | 0.053 ± 0.011 (max 0.061) | 0.723 ± 0.049 |
| B: A + structural graph features | 45 | 0.334 ± 0.093 | 0.839 ± 0.068 | 0.010 ± 0.003 (max 0.014) | 0.637 ± 0.068 | 0.695 ± 0.042 | 0.046 ± 0.004 (max 0.052) | 0.724 ± 0.067 |
| C: A + neighbour aggregates | 224 | 0.477 ± 0.079 | 0.859 ± 0.090 | 0.013 ± 0.009 (max 0.028) | 0.769 ± 0.047 | 0.707 ± 0.076 | 0.054 ± 0.019 (max 0.078) | 0.791 ± 0.088 |
| D: A + spectral embedding | 48 | 0.333 ± 0.073 | 0.860 ± 0.059 | 0.010 ± 0.006 (max 0.018) | 0.591 ± 0.070 | 0.674 ± 0.047 | 0.048 ± 0.015 (max 0.065) | 0.697 ± 0.030 |
| E: A + all graph features | 253 | 0.450 ± 0.092 | 0.854 ± 0.084 | 0.013 ± 0.007 (max 0.024) | 0.751 ± 0.066 | 0.707 ± 0.065 | 0.053 ± 0.016 (max 0.071) | 0.793 ± 0.068 |
| F: structural graph features only | 13 | 0.032 ± 0.024 | 0.295 ± 0.058 | 0.014 ± 0.011 (max 0.028) | 0.118 ± 0.034 | 0.293 ± 0.034 | 0.048 ± 0.017 (max 0.070) | 0.263 ± 0.018 |

### Split by business, scored only on accounts with no review in training

| Features | # | Recall at 1% FPR | Precision at 1% FPR | Test FPR (target 1%) | Recall at 5% FPR | Precision at 5% FPR | Test FPR (target 5%) | PR-AUC (average precision) |
|---|---|---|---|---|---|---|---|---|
| A: 32 handcrafted features | 32 | 0.325 ± 0.117 | 0.898 ± 0.055 | 0.010 ± 0.006 (max 0.017) | 0.627 ± 0.086 | 0.760 ± 0.072 | 0.056 ± 0.021 (max 0.076) | 0.774 ± 0.055 |
| B: A + structural graph features | 45 | 0.260 ± 0.075 | 0.886 ± 0.072 | 0.009 ± 0.004 (max 0.015) | 0.545 ± 0.102 | 0.760 ± 0.062 | 0.047 ± 0.013 (max 0.067) | 0.742 ± 0.073 |
| C: A + neighbour aggregates | 224 | 0.388 ± 0.093 | 0.870 ± 0.120 | 0.017 ± 0.015 (max 0.043) | 0.667 ± 0.089 | 0.776 ± 0.079 | 0.055 ± 0.026 (max 0.092) | 0.792 ± 0.100 |
| D: A + spectral embedding | 48 | 0.295 ± 0.106 | 0.897 ± 0.056 | 0.011 ± 0.010 (max 0.026) | 0.570 ± 0.094 | 0.768 ± 0.056 | 0.050 ± 0.025 (max 0.086) | 0.758 ± 0.023 |
| E: A + all graph features | 253 | 0.377 ± 0.101 | 0.871 ± 0.113 | 0.015 ± 0.012 (max 0.034) | 0.650 ± 0.094 | 0.762 ± 0.072 | 0.058 ± 0.022 (max 0.086) | 0.792 ± 0.080 |
| F: structural graph features only | 13 | 0.025 ± 0.030 | 0.261 ± 0.062 | 0.019 ± 0.019 (max 0.048) | 0.085 ± 0.033 | 0.293 ± 0.032 | 0.058 ± 0.027 (max 0.090) | 0.270 ± 0.019 |

### Random split (leakage check only)

| Features | # | Recall at 1% FPR | Precision at 1% FPR | Test FPR (target 1%) | Recall at 5% FPR | Precision at 5% FPR | Test FPR (target 5%) | PR-AUC (average precision) |
|---|---|---|---|---|---|---|---|---|
| A: 32 handcrafted features | 32 | 0.533 ± 0.018 | 0.899 ± 0.010 | 0.010 ± 0.001 (max 0.011) | 0.767 ± 0.022 | 0.731 ± 0.011 | 0.049 ± 0.003 (max 0.052) | 0.831 ± 0.013 |
| E: A + all graph features | 253 | 0.723 ± 0.009 | 0.924 ± 0.005 | 0.010 ± 0.001 (max 0.011) | 0.879 ± 0.007 | 0.755 ± 0.010 | 0.050 ± 0.002 (max 0.052) | 0.911 ± 0.002 |
