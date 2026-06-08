# Growth Ratio Policy

_Generated: 2026-06-06 17:17:52 KST_

## Positive-base policy
- base_year_offset: t - 1
- positive_base_only: True
- cap_extreme: |growth| > 100.0 → NaN

## 근거
- 분모 (base_value) ≤ 0 인 경우 성장률 부호 의미가 모호
- 흑자 전환/적자 전환 케이스는 별도 indicator로 처리하는 것이 정확
- 분모가 0인 경우 division-by-zero → NaN
