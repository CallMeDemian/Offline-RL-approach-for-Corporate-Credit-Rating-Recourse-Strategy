"""Stage 1 Loop B1 substrate-validation gate.

Tests whether the Oracle score *change* tracks the real 10-grade rating *change*
(RESEARCH_FINAL_METHODOLOGY_AND_DESIGN.md §4.2, RQ0). Simulator-independent: runs
entirely on the Stage 1 backend firm-year outputs.

Preliminary analysis found the contemporaneous relationship null and the
one-year-lead relationship strong, so this verifier tests the LEAD relationship:
the Oracle score change over (t-1 -> t) versus the real rating change over
(t -> t+1). Loops A and B2 (simulator-mediated) are NOT part of this gate; they
run as a Stage 2 extension.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from credit_recourse.rl.common.io import write_json

KEY = "거래소코드"

# Pre-registered thresholds (RESEARCH §4.2; fixed before the final run).
# rating_num_10 is encoded lower-is-better (AAA=1 ... D=10), while R_score_* is
# higher-is-better. Therefore the raw level Spearman should be negative when the
# backend is valid. Verdicts use the oriented value: -raw_spearman.
LEVEL_SPEARMAN_MIN = 0.55
LOOPB1_AGREEMENT_TARGET = 0.65
LOOPB1_SPEARMAN_TARGET = 0.35
RATING_NUM_10_ORIENTATION = "lower_is_better"
SCORE_ORIENTATION = "higher_is_better"

BACKENDS = {
    "alpha": ("alpha/oracle_firm_year_output_alpha.parquet", "R_score_alpha"),
    "beta": ("beta/benchmark_firm_year_output_beta.parquet", "R_score_beta"),
    "gamma": ("gamma/benchmark_firm_year_output_gamma.parquet", "R_score_gamma"),
}


def _auc_upgrade_vs_downgrade(score: np.ndarray, improvement: np.ndarray) -> float | None:
    """Probability a randomly chosen upgrade has a higher score change than a downgrade."""
    up = score[improvement > 0]
    dn = score[improvement < 0]
    if len(up) == 0 or len(dn) == 0:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in up for b in dn)
    return float(wins / (len(up) * len(dn)))


def _lead_pairs(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Build (rating change t->t+1, score change t-1->t) records per firm.

    Requires three consecutive fiscal years (t-1, t, t+1) for the lead score change.
    """
    recs: list[dict[str, Any]] = []
    for _, g in df.sort_values([KEY, "year"]).groupby(KEY):
        yrs = g["year"].to_numpy()
        rat = g["rating_num_10"].to_numpy(dtype=float)
        sc = g[score_col].to_numpy(dtype=float)
        sp = g["split_stage4"].to_numpy() if "split_stage4" in g.columns else np.array([""] * len(g))
        for i in range(1, len(g) - 1):
            if yrs[i + 1] - yrs[i] != 1 or yrs[i] - yrs[i - 1] != 1:
                continue
            if np.isnan(rat[i]) or np.isnan(rat[i + 1]) or np.isnan(sc[i]) or np.isnan(sc[i - 1]):
                continue
            recs.append({
                "d_rating": rat[i + 1] - rat[i],          # >0 = downgrade (higher num = worse)
                "improvement": -(rat[i + 1] - rat[i]),     # >0 = upgrade
                "d_score_lead": sc[i] - sc[i - 1],         # score change t-1 -> t
                "split": sp[i + 1],
            })
    return pd.DataFrame(recs)


def _evaluate_split(pairs: pd.DataFrame, level_df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    movers = pairs[pairs["d_rating"] != 0]
    out: dict[str, Any] = {"n_pairs": int(len(pairs)), "n_movers": int(len(movers))}
    # level validity (premise): score vs rating level
    lv = level_df.dropna(subset=[score_col, "rating_num_10"])
    raw_level_rho = (
        float(stats.spearmanr(lv[score_col], lv["rating_num_10"]).statistic) if len(lv) > 2 else None
    )
    # R_score_* is higher-is-better, but rating_num_10 is lower-is-better.
    # A valid backend should therefore show a negative raw rho. The pre-registered
    # level-validity threshold is applied to the oriented value.
    oriented_level_rho = -raw_level_rho if raw_level_rho is not None else None
    out["level_validity_spearman_raw"] = raw_level_rho
    out["level_validity_spearman_oriented"] = oriented_level_rho
    # Backward-compatible alias, but do not use this field for verdicts.
    out["level_validity_spearman"] = raw_level_rho
    out["rating_num_10_orientation"] = RATING_NUM_10_ORIENTATION
    out["score_orientation"] = SCORE_ORIENTATION
    if len(movers) < 5:
        out["status"] = "insufficient_movers"
        return out
    sign_match = np.sign(movers["d_score_lead"]) == np.sign(movers["improvement"])
    n = int(len(movers))
    k = int(sign_match.sum())
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95)
    out["lead_direction_agreement"] = float(k / n)
    out["lead_direction_agreement_ci95"] = [float(ci.low), float(ci.high)]
    rho = stats.spearmanr(movers["d_score_lead"], movers["improvement"])
    out["lead_spearman"] = float(rho.statistic)
    out["lead_spearman_p"] = float(rho.pvalue)
    out["upgrade_vs_downgrade_auc"] = _auc_upgrade_vs_downgrade(
        movers["d_score_lead"].to_numpy(), movers["improvement"].to_numpy()
    )
    return out


def _verdict(oot: dict[str, Any]) -> str:
    """Three-tier verdict from the out-of-sample (held-out) mover statistics."""
    if oot.get("status") == "insufficient_movers" or "lead_direction_agreement" not in oot:
        return "fail"
    ci_low = oot["lead_direction_agreement_ci95"][0]
    agree = oot["lead_direction_agreement"]
    lv = oot.get("level_validity_spearman_oriented")
    if lv is not None and lv < LEVEL_SPEARMAN_MIN:
        return "fail"               # premise (oriented level validity) not met
    if ci_low <= 0.50:
        return "fail"               # not significantly above chance
    if agree >= LOOPB1_AGREEMENT_TARGET:
        return "strong_pass"
    return "partial_pass"


def verify_backend(name: str, path: Path, score_col: str, rating_map: pd.DataFrame,
                   errors: list[str]) -> dict[str, Any]:
    res: dict[str, Any] = {"backend": name}
    if not path.exists():
        errors.append(f"missing backend output: {path}")
        res["status"] = "missing_output"
        return res
    df = pd.read_parquet(path)
    if score_col not in df.columns:
        # fall back: resolve any R_score_* column
        cand = [c for c in df.columns if str(c).startswith("R_score")]
        if not cand:
            errors.append(f"{name}: no score column found in {path.name}")
            res["status"] = "no_score_column"
            return res
        score_col = cand[0]
    res["score_column"] = score_col
    for c in (KEY, "year"):
        if c not in df.columns:
            errors.append(f"{name}: output missing key column {c}")
            res["status"] = "missing_key_column"
            return res
    # authoritative 10-grade rating: use the file's own column if present, else join from alpha
    if "rating_num_10" not in df.columns:
        df = df.merge(rating_map, on=[KEY, "year"], how="left")
    df = df[[KEY, "year", score_col, "rating_num_10"] + (["split_stage4"] if "split_stage4" in df.columns else [])].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["rating_num_10"] = pd.to_numeric(df["rating_num_10"], errors="coerce")
    pairs = _lead_pairs(df, score_col)
    res["all"] = _evaluate_split(pairs, df, score_col)
    if "split" in pairs.columns and (pairs["split"] == "oot").any():
        res["oot"] = _evaluate_split(pairs[pairs["split"] == "oot"], df, score_col)
        res["dev"] = _evaluate_split(pairs[pairs["split"] == "dev"], df, score_col)
        res["verdict"] = _verdict(res["oot"])
        res["verdict_basis"] = "oot"
    else:
        res["verdict"] = _verdict(res["all"])
        res["verdict_basis"] = "all"
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 1 Loop B1 substrate-validation gate")
    p.add_argument("--project-root", required=True)
    args = p.parse_args(argv)
    root = Path(args.project_root).resolve()
    final = root / "data" / "final_freeze"
    backends_dir = final / "stage1_oracle_backends"
    errors: list[str] = []

    # authoritative 10-grade rating map from the Alpha output
    alpha_path = backends_dir / BACKENDS["alpha"][0]
    rating_map = pd.DataFrame(columns=[KEY, "year", "rating_num_10"])
    if alpha_path.exists():
        a = pd.read_parquet(alpha_path)
        if {KEY, "year", "rating_num_10"}.issubset(a.columns):
            rating_map = a[[KEY, "year", "rating_num_10"]].drop_duplicates()
        else:
            errors.append("alpha output missing one of 거래소코드/year/rating_num_10 for rating map")
    else:
        errors.append(f"missing alpha output for authoritative rating map: {alpha_path}")

    per_backend = {
        name: verify_backend(name, backends_dir / rel, col, rating_map, errors)
        for name, (rel, col) in BACKENDS.items()
    }

    verdicts = {n: r.get("verdict", "fail") for n, r in per_backend.items()}
    # the gate verdict is the Alpha (main backend) verdict; Beta/Gamma reported alongside
    gate_verdict = verdicts.get("alpha", "fail")

    result = {
        "stage": "verify_stage1_substrate_validation",
        "loop": "B1",
        "specification": "lead (Oracle score change t-1->t vs real 10-grade rating change t->t+1)",
        "scale": "rating_num_10",
        "thresholds": {
            "level_validity_spearman_oriented_min": LEVEL_SPEARMAN_MIN,
            "level_validity_orientation_rule": "oriented = -spearman(R_score, rating_num_10), because R_score is higher-is-better and rating_num_10 is lower-is-better",
            "loopB1_lead_direction_agreement_target": LOOPB1_AGREEMENT_TARGET,
            "loopB1_lead_direction_agreement_min_rule": "95pct_CI_lower_bound_gt_0.50",
            "loopB1_lead_spearman_target": LOOPB1_SPEARMAN_TARGET,
        },
        "per_backend": per_backend,
        "backend_verdicts": verdicts,
        "gate_verdict": gate_verdict,
        "gate_verdict_basis": "alpha_main_backend",
        "note": "Loops A and B2 (simulator-mediated) are a Stage 2 extension and are not part of this gate.",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    out = final / "ledgers" / "stage1_substrate_validation_loopB1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # exit non-zero only on infrastructure errors; a 'fail' verdict is a finding, not a crash
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
