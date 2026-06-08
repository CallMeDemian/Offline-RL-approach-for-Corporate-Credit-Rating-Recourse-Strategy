# Stage 1C v3.2 — Summary

**Stage**: 1C v3.2 — Nonfinancial Metadata Panel (Industry Risk Expanded)
**Verdict**: ✅ **PASS**
**Run date**: 2026-05-01
**Version**: v3.2

## 산출 panel

- shape: (4924, 52)
- firm-year: 4924 (full panel 유지)
- eligible_for_stage2: 4924
- year range: 2002–2024

## 변수 후보 요약

| 평가항목 | quality_pass | selected_eligible |
|---|---:|---:|
| 산업위험 | 14 | 9 |
| 경영위험 | 6 | 6 |
| 영업위험 | 3 | 3 |
| 재무위험 | 6 | 6 |
| 신뢰도 | 4 | 4 |

총 quality_pass: 33/33  
총 selected_eligible: 28/33

## v3.2 산업위험 신규 proxy

- ✅ `industry_avg_rating_lag1_self_excl`
- ✅ `industry_median_rating_lag1_self_excl`
- ✅ `industry_bad_grade_share_lag1_self_excl`
- ✅ `industry_b_or_lower_share_lag1_self_excl`
- ✅ `industry_rating_std_lag1_self_excl`
- ✅ `industry_rating_iqr_lag1_self_excl`
- ✅ `industry_downgrade_rate_lag1_self_excl`
- ✅ `industry_upgrade_rate_lag1_self_excl`
- ✅ `industry_net_downgrade_rate_lag1_self_excl`


## industry proxy fallback level 분포

- Level 1 (sector × t-1 leave-one-out): 4486
- Level 2 (sector × t-3..t-1 rolling): 268
- Level 3 (sector × past all): 41
- Level 4 (all × past all): 107
- NaN: 22

## Acceptance Criteria

| Criterion | Threshold | Actual | Pass |
|---|---|---|---|
| 신규 산업위험 9개 column 존재 | 9 | 9 | ✅ |
| 신규 산업위험 quality_pass ≥ 기준 | 1 | 9 | ✅ |
| 신규 산업위험 selected_eligible ≥ 기준 | 1 | 9 | ✅ |
| 경영위험 후보 ≥ 1 | 1 | 6 | ✅ |
| 영업위험 후보 ≥ 1 | 1 | 3 | ✅ |
| 재무위험 후보 ≥ 1 | 1 | 6 | ✅ |
| 신뢰도 후보 ≥ 1 | 1 | 4 | ✅ |
| firm-year rows retained or expanded after KONEX merge | 4908 | 4924 | ✅ |
| eligible_for_stage2 커버 ≥ configured threshold | 4662 | 4924 | ✅ |
| Task 1-10 필수 output 모두 생성 | 12/12 | 12/12 | ✅ |

## 다음 단계

→ Stage 3 v3.2-NF — Stage 1C v3.2 비재무 후보군을 사용해 4-metric screening + collinearity replacement 수행.
