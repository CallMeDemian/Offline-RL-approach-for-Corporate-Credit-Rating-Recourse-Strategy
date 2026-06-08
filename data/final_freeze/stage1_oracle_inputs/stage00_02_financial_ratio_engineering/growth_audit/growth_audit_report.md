# Stage 2 v3.2 Growth Candidate Audit Report

_Generated: 2026-06-06 17:17:52 KST_

## 1. 작업 요약

| 항목 | 내용 |
|---|---|
| 기존 성장성 후보 | 11개 |
| 기존 main_eligible | 8개 |
| 신규 GX 계산 | 3개 |
| 신규 main_eligible | 0개 |
| 덮어쓴 main 산출물 | 없음 (보호 파일 유지) |

## 2. 기존 성장성 — main_eligible=True

- `R169` : miss_dev=12.8%, ρ=-0.0786
- `R180` : miss_dev=19.1%, ρ=-0.1104
- `R184` : miss_dev=19.4%, ρ=-0.0448
- `R204` : miss_dev=16.2%, ρ=-0.0447
- `R181` : miss_dev=19.1%, ρ=-0.0578
- `R182` : miss_dev=19.1%, ρ=-0.117
- `R188` : miss_dev=19.6%, ρ=-0.0339
- `R170` : miss_dev=21.5%, ρ=-0.0116

## 3. 기존 성장성 — 제외 사유

- `R174` : miss_dev=32.4%>=30%
- `R205` : denom_instability
- `R185` : miss_dev=31.3%>=30%

## 4. 신규 GX 변수 — main_eligible=True

  (없음)

## 5. 핵심 설계 원칙

- **분모 처리**: base_value ≤ 0 → NaN (stage_config growth_ratio.positive_base_only 정책 동일)
- **3-year simple growth**: CAGR 대신 단순 비율 (음수 base 문제 최소화)
- **Extreme cap**: |growth| > 100 → NaN
- **main_eligible 기준**: miss_dev < 0.3, n_dev ≥ 100, unique > 2, inf_rate ≤ 0.01
- **sector-relative**: sector_7은 Stage 1C 산출물이므로 Stage 2 Task 9에서 생성하지 않음

## 6. 산출물 위치

```
STAGE2_OUT/growth_audit/
  growth_audit_existing.csv                       기존 성장성 audit
  growth_audit_new.csv                            신규 GX audit
  engineered_financial_ratios_growth_expanded.parquet
  candidate_ratio_pool_growth_expanded.csv
  growth_audit_report.md
```

## 7. Stage 3 사용 방법

Stage 3 v3.2에서 성장성 후보 pool을 확장하려면:
- `growth_audit/candidate_ratio_pool_growth_expanded.csv`에서 `main_eligible_v3_2=True` 행 사용
- 재무비율 데이터는 `growth_audit/engineered_financial_ratios_growth_expanded.parquet` 사용
- 기존 main output(`candidate_ratio_pool_by_item.csv`)은 Stage 3 v2 호환용으로 유지
