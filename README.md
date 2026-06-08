# Offline RL 기반 기업 신용등급 개선정책 학습
### Offline RL approach for Corporate Credit Rating Recourse Strategy

| | |
|---|---|
| **학번 / 이름** | `<A67035>` / 조종선 (Demian) |
| **소속** | 서강대학교 AI·SW대학원 — 강화학습의 기초 (GITA401) |
| **GitHub** | https://github.com/CallMeDemian/Offline-RL-approach-for-Corporate-Credit-Rating-Recourse-Strategy |
| **팀 구성** | 1인 (단독) |

> 한국 상장기업(제조업)의 재무 패널에서, **신용등급을 끌어올리는 재무 개선 행동(recourse policy)** 을 오프라인 강화학습(IQL)으로 학습한다. 환경과의 온라인 상호작용 없이, 과거 데이터에서 구성한 전이(transition)만으로 정책을 학습하는 것이 핵심이다.

---

## 1. 문제 정의 및 목표

- **목표**: 기업의 현재 재무 상태가 주어졌을 때, 등급 향상 기대값(Oracle 점수 상승)을 최대화하는 재무 조정 행동을 처방하는 정책을 학습.
- **왜 offline RL인가**: 기업 재무에는 "행동 → 결과"를 실시간으로 시뮬레이션할 수 있는 환경이 없다. 따라서 과거 재무제표에서 (상태, 행동, 보상, 다음상태) 전이를 구성하고, 분포 외 행동을 억제하는 offline RL(IQL)로 학습한다.
- **핵심 발견(요약)**: 과거 데이터의 행동 분포는 보수적 행동(CX1, 낮은 Oracle 가치)에 집중되어 있으나, 학습된 정책(C3)은 데이터에 희소하지만 등급 향상 기여가 큰 행동(DL2/OE2)으로 분포를 이동시킨다

## 2. 파이프라인 개요 (6 stages)

| Stage | 모듈 | 역할 |
|---|---|---|
| **0** | `oracle.stage0.build_stage0_foundation_from_raw` | Raw 재무/등급 데이터 → 정규화된 canonical 패널 |
| **1** | `oracle.stage1.run_stage1_oracle_development` | Oracle 신용모형 3종(alpha=isotonic-binning, beta=ordered logit, gamma=ML) 개발 |
| **2** | `rl.pipelines.final_stage2_*` | Action Projection · counterfactual 전이 생성 · 학습/검증 분할 |
| **3** | `rl.pipelines.final_stage3_acd_ssl` | SSL Transformer Encoder (masked reconstruction, 보상 독립) |
| **4** | `rl.pipelines.final_stage4_candidate_bc` | Class + Family Balanced Behavior Cloning |
| **5** | `rl.pipelines.final_stage5_candidate_iql` | Candidate-IQL (critic→actor distillation) |
| **6** | `eval.final_stage6_*` | 2024년 기업에 대한 multi-Oracle 평가 + 통계 추론 |

State / Action / Reward 및 알고리즘 설계의 상세는 [`보고서 PPT`](#7-보고서)를 참조.

---

## 3. 디렉토리 구조

```
RL_repo/
├── src/credit_recourse/        # 코드 본체 (PYTHONPATH = src)
│   ├── oracle/                 # Stage 0–1: 신용 Oracle 모형 3종
│   │   ├── stage0/             #   Raw → canonical 패널
│   │   ├── stage1/             #   stage00_01~04 (등급결합/비율공학/비재무/변수선택) + backend 개발
│   │   ├── backends/{alpha,beta,gamma}/
│   │   └── verification/       #   stage별 계약(contract) 검증기
│   ├── rl/                     # Stage 2–5: offline RL
│   │   ├── pipelines/final_stage2_*  #   행동투영·counterfactual전이·분할
│   │   ├── pipelines/final_stage3_acd_ssl    #   SSL encoder
│   │   ├── pipelines/final_stage4_candidate_bc  #   BC
│   │   └── pipelines/final_stage5_candidate_iql #   IQL
│   ├── eval/                   # Stage 6: multi-Oracle 평가 + 통계 추론
│   ├── simulator/              # 재무 시뮬레이터 / Oracle evaluator
│   ├── configs/                # action contract, candidate library, seed protocol 등
│   └── utils/, contracts/, common/
├── tools/                      # 실행 러너 (PowerShell)
│   ├── run_oracle_stage0_stage1_RL_repo.ps1     #  Stage 0–1
│   └── run_rl_unified_stage3456.ps1             #  Stage 2–6
├── data/                       # (git 미포함) final_freeze 산출물 + raw 입력 자리
├── results/                    # (git 미포함) 시드별 실험 셀 + 학습된 .pt
├── requirements.txt
└── README.md
```

> `data/`와 `results/`는 용량·데이터 보호 때문에 `.gitignore` 처리되어 있다 (§5, §6 참조).

---

## 4. 환경 설정

- **Python**: 3.10+ 권장 *(코드에 버전 핀은 없음 — pandas 2.x / modern typing 기준 권장값)*
- **OS**: 러너는 Windows PowerShell 기준으로 작성됨. Linux/macOS에서는 §8의 동등 `python -m ...` 커맨드를 직접 사용.
- **GPU**: 선택. `torch.cuda.is_available()`로 자동 감지하여 없으면 CPU 폴백.

```bash
# 가상환경
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

# PYTHONPATH를 src로 지정해야 credit_recourse 패키지가 import됨
# (PowerShell 러너는 내부에서 자동 설정함)
$env:PYTHONPATH = "src"         # Windows PowerShell
# export PYTHONPATH=src         # Linux/macOS
```

---

## 5. 데이터

| 구분 | 포함 여부 | 비고 |
|---|---|---|
| `data/raw/` (원시 상장사 재무·등급) | ❌ 미배포 | 데이터 라이선스/보호. Stage0부터 재현하려면 사용자가 직접 배치 |
| `data/final_freeze/` (Stage0–1 파생 산출물) | ⚠️ git 미포함 | canonical 패널 · Oracle 백엔드 산출물 등. 별도 다운로드(§6) |

- 데이터 전처리(비율 공학, 결측/이상치 필터, 비재무 메타 결합, 변수 선택)는 모두 Stage 0–1 파이프라인에 코드로 포함되어 있다 (`oracle/stage0`, `oracle/stage1/stage00_01~04`).
- **재현 범위**: 배포된 `data/final_freeze` 산출물이 있으면 RL Stage 2–6을 그대로 재현 가능. Stage 0(Raw→canonical)은 Raw 데이터가 있어야 재실행 가능.

---

## 6. 학습된 모델 / 데이터 다운로드

> 과제 요건에 따라 학습된 모델은 리포에 커밋하지 않고 다운로드 링크로 제공

**학습된 모델 (대표 시드 1개 셀 = `m0p05_f0p45_liq0p45_s1 / P50_L1p5_G0p6_T0p85`)**

| 파일 | 크기 | 설명 |
|---|---|---|
| `stage3_encoder_avs256_final_refit_fulltrain.pt` | ~14.5 MB | SSL Transformer Encoder |
| `stage4_bc_final_refit_fulltrain.pt` | ~29 MB | Class+Family Balanced BC |
| `stage5_candidate_iql_final_refit_fulltrain.pt` | ~52.6 MB | **최종 IQL 정책** |

**다운로드**: `<여기에 링크 — 아래 둘 중 하나로 채우세요>`

- **방법 A (권장) GitHub Release**: 파일당 최대 2GB까지 첨부 가능. §9의 절차 참조.
- **방법 B Google Drive 등 외부 스토리지**: 공유 링크를 위 표/이 줄에 기입.

---

## 7. 보고서

프로젝트 주제·설계·구현·실험 결과를 담은 PPT 보고서는 리포 루트에 포함되어 있으며, Cyber Campus에도 별도 제출한다.
→ `Offline RL approach for Corporate Credit Rating Recourse Strategy.pdf` (또는 `.pptx`)

---

## 8. 실행 방법

> 두 러너 모두 멱등(idempotent)하게 설계됨: 산출물이 이미 있으면 SKIP, 다시 만들려면 clean/force 플래그 사용.
> **⚠️ 경로 주의**: `run_rl_unified_stage3456.ps1`의 기본 `-ProjectRoot`는 `thesis_repo`로 되어 있다. 실제 폴더명이 `RL_repo`이면 **반드시 `-ProjectRoot`를 명시**하거나 스크립트 기본값을 수정할 것.

### 8.1 Stage 0–1 — Oracle 신용모형 개발

```powershell
# 전체 (resume-safe). 원시 데이터가 있을 때 Stage0부터.
.\tools\run_oracle_stage0_stage1_RL_repo.ps1 `
  -ProjectRoot "C:\Users\<you>\Desktop\RL_repo" `
  -RawAllDir   "<원시 재무 폴더>" `
  -RawRatingDir "<원시 등급 폴더>" `
  -RunVerifiers

# 부분 재실행 예: 백엔드만 다시
.\tools\run_oracle_stage0_stage1_RL_repo.ps1 -Only backend_alpha -ProjectRoot "...\RL_repo"
```

내부적으로 실행되는 핵심 커맨드 (Linux 등에서 직접 호출 시):

```bash
python -m credit_recourse.utils.materialize_final_freeze_configs --project-root <ROOT> --overwrite
python -m credit_recourse.verification.verify_final_no_regression_contract --project-root <ROOT>
python -m credit_recourse.oracle.stage0.build_stage0_foundation_from_raw --project-root <ROOT> \
    --raw-all-dir <...> --raw-rating-dir <...>
python -m credit_recourse.oracle.stage1.run_stage1_oracle_development --project-root <ROOT> \
    --score-end-year 2023 --dev-start-year 2002 --dev-end-year 2019 --oot-start-year 2020 \
    --start-step stage00_01 --end-step substrate_validation --resume
```

**시간 분할**: dev 2002–2019 · OOT 2020 · score-end 2023 (Stage6 평가는 2024 기업).

### 8.2 Stage 2–6 — Offline RL 학습 및 평가

아래는 배포된 모델 셀과 **동일한 최종 설정**으로 재현하는 커맨드다 (값 출처: 셀 매니페스트 `00_README_RUN_STATUS.md`).

```powershell
.\tools\run_rl_unified_stage3456.ps1 `
  -ProjectRoot "C:\Users\<you>\Desktop\RL_repo" `
  -Stage3Mode skip -Stage3Seed 1 `        # SSL encoder는 보상 독립 → 재사용(skip)
  -MagnitudeQuantile 50 `                  # P50
  -Stage4Epochs 80 -Stage4BatchSize 512 -Stage4Seed 1 -FamilyBalanced `
  -Stage5Epochs 80 -Stage5BatchSize 128 -Stage5Seed 1 `
  -Gamma 0.6 -ExpectileTau 0.85 -Beta 15 `
  -Stage5LearningRate 1e-4 -Stage5WeightDecay 2e-3 `
  -DistillLambda 1.5 -DistillMarginMin 0.01 `
  -Stage5CriticHeadArch cross_attention -CrossAttnBlocks 2 -CrossAttnHeads 4 -CrossAttnDropout 0.25 `
  -CounterfactualTransitions -CounterfactualRewardMode phi_merton_fcff_liquidity
```

> Stage3를 처음부터 학습하려면 `-Stage3Mode train`. 7-시드 실험은 `-Stage5Seed`(및 seed grid)를 바꿔 반복한다.

---

## 9. 실험 결과 (요약)

대표 셀(seed 1) 기준 multi-Oracle 평가 핵심 수치:

| 지표 | 값 |
|---|---|
| C3 − C0 (α) | **+0.628** |
| C2 − C0 (α) | +0.506 |
| C3 − C2 (α) | +0.122 |
| 정책 순위 | **1 / 18** |
| C3 dominant action | DL2 (38.6%) |

**7-시드 신뢰도**: 평균 C3 − C0 α = **+0.63 ± 0.03**, 7개 시드 중 6개가 순위 1/18.

> 위 수치는 배포된 셀 산출물에 대응한다. 셀별 권위 있는 요약은
> `outputs/stage6_candidate_selector_eval/final_policy_summary.csv`를 사용한다
> (셀 루트의 `00_POLICY_SUMMARY_CORE.csv`는 stage6 재실행 시 stale일 수 있음).

---

## 10. 재현성 노트 (읽어두면 좋은 함정들)

- **러너 기본값 ≠ 최종 config**: `run_rl_unified_stage3456.ps1`의 param 기본값(Stage5 epochs 150 / batch 256 / γ 0.8 / τ 0.9 / β 10 / linear head)은 탐색용 기본치이며, **최종 실험 값은 §8.2의 명시 인자**(epochs 80 / batch 128 / γ 0.6 / τ 0.85 / β 15 / cross_attention)다.
- **skip 모드 주의**: `-Stage3Mode skip`은 Stage 1/2 일부를 디스크에서 로드한다. 중간 산출물이 시드 간 달라지면 평가 패널이 어긋날 수 있다.
- **권위 있는 결과 CSV**: stage6를 수동 재실행한 경우, 셀 루트의 `00_POLICY_SUMMARY_CORE.csv`가 아니라 `stage6_candidate_selector_eval/final_policy_summary.csv`를 본다.
- **결과 폴더 네이밍**: 바깥 폴더 `m{merton}_f{fcff}_liq{liquidity}_s{seed}`는 Stage2 counterfactual 보상 가중 프로파일과 시드를, 안쪽 셀 `P{quantile}_L{distill_λ}_G{gamma}_T{tau}`는 Stage4/5 하이퍼파라미터를 인코딩한다.

---