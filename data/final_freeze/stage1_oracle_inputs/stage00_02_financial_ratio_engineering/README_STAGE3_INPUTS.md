# Stage 3 Variable Selection Inputs (Stage 2 outputs)

Stage 3 (Variable Selection)에서 사용할 Stage 2 산출물:

## 핵심 입력
- `engineered_financial_ratios.parquet` — 4,746 firm-year × 164 ratio wide panel
- `financial_ratio_formula_dictionary_final.csv` — 164 ratio formula 정의
- `candidate_ratio_pool_by_item.csv` — 평가항목별 후보 ratio pool
- `ratio_quality_report.csv` — quality_pass=True인 134개 ratio 식별
- `growth_audit/candidate_ratio_pool_growth_expanded.csv` — 성장성 확장 pool (v3.2)

## 보조 (선택)
- `ratio_audit_against_precomputed_panel.csv` — NICE 검증 결과
- `stage2_formula_investigation_targets.csv` — 추가 검토 대상 formula
- `lag_support_coverage_report.csv` — lag 데이터 커버리지

## 사용 예시
```python
import pandas as pd
ratios = pd.read_parquet("engineered_financial_ratios.parquet")
qual = pd.read_csv("ratio_quality_report.csv")
pass_cols = qual[qual["quality_pass"] == True]["ratio_name"].tolist()
X = ratios[["거래소코드", "year"] + pass_cols]

# v3.2 성장성 확장 pool 사용 시
# pool_exp = pd.read_csv("growth_audit/candidate_ratio_pool_growth_expanded.csv")
```
