# Stage 1C v2 vs v3.2 — Diff Report

**Comparison date**: 2026-06-06T17:18:18.725861+09:00

## 1. Row / panel shape

| item | v3.2 |
|---|---:|
| total firm-year | 4924 |
| eligible_for_stage2 | 4924 |
| panel shape | (4924, 52) |

## 2. Variable count

| item | v3.2 |
|---|---:|
| quality report variables | 33 |
| quality_pass variables | 33 |
| selected_eligible variables | 28 |

## 3. Category counts

| 평가항목 | quality_pass | selected_eligible |
|---|---:|---:|
| 산업위험 | 14 | 9 |
| 경영위험 | 6 | 6 |
| 영업위험 | 3 | 3 |
| 재무위험 | 6 | 6 |
| 신뢰도 | 4 | 4 |

## 4. Industry-risk expansion

v2는 `industry_avg_rating` 중심의 산업위험 후보군이었다. v3.2는 아래 9개 lagged/self-excluded 산업위험 proxy를 추가한다.

- `industry_avg_rating_lag1_self_excl`
- `industry_median_rating_lag1_self_excl`
- `industry_bad_grade_share_lag1_self_excl`
- `industry_b_or_lower_share_lag1_self_excl`
- `industry_rating_std_lag1_self_excl`
- `industry_rating_iqr_lag1_self_excl`
- `industry_downgrade_rate_lag1_self_excl`
- `industry_upgrade_rate_lag1_self_excl`
- `industry_net_downgrade_rate_lag1_self_excl`


### Stage 3 selected-eligible industry candidates

- `industry_avg_rating_lag1_self_excl`
- `industry_median_rating_lag1_self_excl`
- `industry_bad_grade_share_lag1_self_excl`
- `industry_b_or_lower_share_lag1_self_excl`
- `industry_rating_std_lag1_self_excl`
- `industry_rating_iqr_lag1_self_excl`
- `industry_downgrade_rate_lag1_self_excl`
- `industry_upgrade_rate_lag1_self_excl`
- `industry_net_downgrade_rate_lag1_self_excl`


## 5. Fallback distribution for level-based industry proxy

- Level 1: 4486
- Level 2: 268
- Level 3: 41
- Level 4: 107
- NaN: 22

## 6. Stage 3 input files

- `nonfinancial_metadata_panel.parquet`
- `nonfinancial_candidate_pool_by_item.csv`
- `nonfinancial_variable_quality_report.csv`

Stage 3 must use `selected_eligible=True`, `diagnostic_only=False`, and `leakage_safe=True` when screening nonfinancial candidates.
