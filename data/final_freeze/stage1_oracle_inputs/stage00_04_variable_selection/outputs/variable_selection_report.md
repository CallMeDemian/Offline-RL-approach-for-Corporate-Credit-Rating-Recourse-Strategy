# Stage 3 Variable Selection Report

## Selected Variables

| category   | variable_id                        | variable_name                      | source       |    score |   rank |   spearman_rho |   abs_spearman |       iv |   kw_eta2 |   monotonicity | direction       | direction_weak   |   missing_rate_dev |
|:-----------|:-----------------------------------|:-----------------------------------|:-------------|---------:|-------:|---------------:|---------------:|---------:|----------:|---------------:|:----------------|:-----------------|-------------------:|
| 수익성        | R006                               | R006                               | financial    | 0.914105 |      1 |     -0.356114  |      0.356114  | 0.625798 | 0.135959  |       0.777778 | value_up_good   | False            |         0.162474   |
| 안정성        | R064                               | R064                               | financial    | 0.915026 |      1 |     -0.492614  |      0.492614  | 1.36921  | 0.266399  |       0.888889 | value_up_good   | False            |         0.162062   |
| 부채상환능력     | R085                               | R085                               | financial    | 0.921166 |      1 |      0.401221  |      0.42965   | 0.781356 | 0.19132   |       0.888889 | value_down_good | False            |         0.190928   |
| 유동성        | R133                               | R133                               | financial    | 0.569564 |      1 |     -0.237148  |      0.237148  | 0.297492 | 0.0564425 |       0.777778 | value_up_good   | False            |         0.162062   |
| 활동성        | R157                               | R157                               | financial    | 0.96     |      1 |     -0.392371  |      0.392371  | 0.720457 | 0.174376  |       0.777778 | value_up_good   | False            |         0.19134    |
| 성장성        | R182                               | R182                               | financial    | 0.625546 |      1 |     -0.117009  |      0.117009  | 0.190064 | 0.0176567 |       0.666667 | value_up_good   | False            |         0.19134    |
| 산업위험       | industry_avg_rating_lag1_self_excl | industry_avg_rating_lag1_self_excl | nonfinancial | 0.580165 |      1 |      0.236054  |      0.236054  | 0.237642 | 0.0545018 |       0.666667 | value_down_good | False            |         0.00907216 |
| 경영위험       | cap_change_count_3y                | cap_change_count_3y                | nonfinancial | 1        |      1 |      0.247165  |      0.247165  | 0.33249  | 0.0630207 |       0.75     | value_down_good | False            |         0          |
| 영업위험       | log_assets                         | log_assets                         | nonfinancial | 1        |      1 |     -0.471802  |      0.471802  | 0.802728 | 0.242333  |       0.888889 | value_up_good   | False            |         0.162062   |
| 재무위험       | operating_loss_freq_3y             | operating_loss_freq_3y             | nonfinancial | 0.9      |      1 |      0.3822    |      0.3822    | 0.4824   | 0.182814  |       1        | value_down_good | False            |         0.147629   |
| 신뢰도        | ratio_missing_rate                 | ratio_missing_rate                 | nonfinancial | 0.702668 |      1 |      0.0174209 |      0.0174209 | 0.330424 | 0.0444496 |       0.5      | value_down_good | True             |         0          |

## Methodology

- Dev sample: 2002-2019
- OOT sample: 2020-2023
- 4-metric screening: Spearman, IV, KW eta², monotonicity
- Selection score: 0.3·|ρ| + 0.3·IV + 0.2·KW + 0.2·mono
- Collinearity threshold: |corr| > 0.7
- Stage 3 sub-fix v3: R185 and financial_data_completeness prior scores overridden to custom floor; audit log generated.
- Domain filters: liquidity exclusion + Stage 1C v3.2 industry-risk lag1 self-excluded allowlist.
