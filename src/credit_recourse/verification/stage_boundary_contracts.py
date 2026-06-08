from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from credit_recourse.contracts.stage_paths import CANONICAL_STAGE_DIRS, DEPRECATED_STAGE_DIR_ALIASES, final_root
from credit_recourse.oracle.stage0.rating_contract_repair import validate_stage0_contract
from credit_recourse.utils.io_contract import read_json, write_json, resolve_selected_variables, sha256_file
from credit_recourse.rl.common.actions import load_action_space
from credit_recourse.rl.contracts.avs256_acd_v2 import SCHEMA_VERSION, CONTINUOUS_COLUMNS, ACD_TARGET_COLUMNS, CATEGORICAL_COLUMNS, EXPECTED_BLOCK_COUNTS, DIRECTION_VOCAB
from credit_recourse.rl.final_candidate.encoder import FINAL_ENCODER_D_MODEL, FINAL_ENCODER_N_HEADS, FINAL_ENCODER_N_LAYERS, FINAL_ENCODER_FF_MULTIPLIER

V32_LABELS = [
    "A0_noop",
    "DL1_deleverage_mild",
    "DL2_deleverage_moderate",
    "RF1_short_debt_refinance",
    "CX1_capex_discipline",
    "WC1_working_capital_tightening",
    "WC2_supplier_financing",
    "OE1_cost_efficiency_mild",
    "OE2_cost_efficiency_moderate",
    "MX1_cost_and_deleverage",
    "MX2_liquidity_rescue",
]
ACTION_COLS = [
    "action__ppe_pct", "action__inv_turnover_chg", "action__ar_turnover_chg", "action__ap_turnover_chg",
    "action__short_debt_pct", "action__long_debt_pct", "action__bond_pct", "action__revenue_growth",
    "action__cogs_ratio_chg", "action__sga_ratio_chg",
]

ACTION_MIN_OBSERVED_RATE = {
    "ppe_pct": 0.20,
    "inv_turnover_chg": 0.20,
    "ar_turnover_chg": 0.20,
    "ap_turnover_chg": 0.20,
    "short_debt_pct": 0.10,
    "long_debt_pct": 0.10,
    "bond_pct": 0.05,
    "revenue_growth": 0.20,
    "cogs_ratio_chg": 0.20,
    "sga_ratio_chg": 0.20,
}

def action_cols(root: Path) -> list[str]:
    return list(load_action_space(root).columns)

def train_labels(root: Path) -> list[str]:
    return list(load_action_space(root).train_labels)

REWARD_COLS = [
    "reward_raw_notch", "reward_raw", "phi_t", "phi_tplusH", "delta_phi", "delta_phi_clipped",
    "lambda_phi", "reward_aux_phi", "reward_total_raw", "reward_mean_train", "reward_std_train",
    "reward_train", "reward_original", "reward",
]

MERTON_AUX_REWARD_COLS = [
    "merton_default_point_t", "merton_default_point_tplusH", "merton_badness_t", "merton_badness_tplusH",
    "delta_merton_badness", "delta_merton_badness_scaled", "lambda_merton", "reward_aux_merton",
]
FCFF_AUX_REWARD_COLS = [
    "fcff_capacity_t", "fcff_capacity_tplusH", "delta_fcff_capacity", "delta_fcff_capacity_scaled",
    "lambda_fcff", "reward_aux_fcff",
]
LIQUIDITY_AUX_REWARD_COLS = [
    "liquid_capacity_t", "liquid_capacity_tplusH", "delta_liquid_capacity",
    "delta_liquid_capacity_scaled", "lambda_liquidity", "reward_aux_liquidity",
]
AUX_REWARD_COLS = MERTON_AUX_REWARD_COLS + FCFF_AUX_REWARD_COLS + LIQUIDITY_AUX_REWARD_COLS

STAGE6_REQUIRED_FIRMSTATE_FIELDS = [
    "revenue", "cogs", "sga", "total_assets", "current_assets", "current_liabilities", "cash",
    "inventory", "receivables", "payables", "ppe", "short_term_debt", "long_term_debt", "bonds",
    "total_liabilities", "total_equity",
]

STAGE6_FIRMSTATE_FIELD_ALIASES = {
    "receivables": ["accounts_receivable"],
    "payables": ["accounts_payable"],
    "short_term_debt": ["short_debt"],
    "long_term_debt": ["long_debt"],
    "bonds": ["bond"],
}


# Claim-mode policy for sensitivity-grid verification.
#
# The stage-boundary verifier is used in two different contexts:
#   1. fixed-config/local downstream cells, where sensitivity grids may
#      explicitly register candidate configurations that were intentionally
#      not run; and
#   2. completed full-sweep claims, where every registered row must be run.
#
# Default CLI mode is (1).  Use --strict-full-sweep-claim for (2).
STRICT_FULL_SWEEP_CLAIM = False

REGISTERED_NOT_RUN_SOFT_GATE_CODE = "FIXED_CONFIG_SOFT_GATE"
REGISTERED_NOT_RUN_STRICT_GATE_CODE = "STRICT_FULL_SWEEP_CLAIM"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def nonempty(path: Path, errors: list[str], label: str | None = None) -> bool:
    if not path.exists():
        errors.append(f"missing {label or path}")
        return False
    if path.is_file() and path.stat().st_size <= 0:
        errors.append(f"empty {label or path}")
        return False
    return True



def check_sensitivity_grid(path: Path, errors: list[str], warnings: list[str], label: str, *, require_selected: bool = True) -> dict[str, Any]:
    """Validate a phase sensitivity grid with explicit claim-mode semantics.

    REGISTERED_NOT_RUN is not intrinsically a stage-boundary failure.  It means
    that the grid contains registered configurations that were not executed.
    That is acceptable for a fixed-config/local downstream cell, but it is not
    acceptable when the run is being presented as a completed full sensitivity
    sweep.  The verifier therefore treats it as:

    * warning in the default fixed-config boundary-check mode;
    * error only when --strict-full-sweep-claim is supplied.
    """
    if not nonempty(path, errors, label):
        return {"status": "MISSING"}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"cannot read {label}: {exc!r}")
        return {"status": "FAIL"}
    if "status" not in df.columns:
        errors.append(f"{label} missing status column")
        return {"status": "FAIL", "rows": int(len(df))}

    counts = df["status"].astype(str).value_counts().to_dict()
    selected_n = int(counts.get("SELECTED_RUN", 0))
    registered_not_run_n = int(counts.get("REGISTERED_NOT_RUN", 0))

    if require_selected and selected_n < 1:
        errors.append(f"{label} must contain at least one SELECTED_RUN row")
    elif (not require_selected) and selected_n < 1:
        warnings.append(f"{label} has no SELECTED_RUN row; accepted for fixed-config/local runs only")

    if registered_not_run_n > 0:
        msg = (
            f"{label} contains REGISTERED_NOT_RUN rows; "
            f"rows={registered_not_run_n}; "
            "acceptable for fixed-config/local downstream cells, "
            "not acceptable for a completed full sensitivity sweep claim"
        )
        if STRICT_FULL_SWEEP_CLAIM:
            errors.append(f"[{REGISTERED_NOT_RUN_STRICT_GATE_CODE}] {msg}")
        else:
            warnings.append(f"[{REGISTERED_NOT_RUN_SOFT_GATE_CODE}] {msg}")

    status = "PASS" if (selected_n >= 1 or not require_selected) else "FAIL"
    if registered_not_run_n > 0 and STRICT_FULL_SWEEP_CLAIM:
        status = "FAIL_FULL_SWEEP_CLAIM"

    return {
        "status": status,
        "rows": int(len(df)),
        "status_counts": counts,
        "selected_run_rows": selected_n,
        "registered_not_run_rows": registered_not_run_n,
        "registered_not_run_policy": (
            "ERROR_STRICT_FULL_SWEEP_CLAIM" if STRICT_FULL_SWEEP_CLAIM else "WARNING_FIXED_CONFIG_SOFT_GATE"
        ),
        "strict_full_sweep_claim": bool(STRICT_FULL_SWEEP_CLAIM),
    }

def read_parquet(path: Path, errors: list[str], label: str | None = None) -> pd.DataFrame:
    if not nonempty(path, errors, label):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        errors.append(f"cannot read parquet {label or path}: {exc!r}")
        return pd.DataFrame()


def check_cols(df: pd.DataFrame, cols: list[str], errors: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        errors.append(f"{where} missing columns: {missing}")


def _stage6_field_alias_matches(column: str, field: str) -> bool:
    c = str(column)
    aliases = [field] + list(STAGE6_FIRMSTATE_FIELD_ALIASES.get(field, []))
    for alias in aliases:
        if c == alias or c == f"sim__{alias}":
            return True
        if c == f"raw__{alias}" or c == f"raw__sim__{alias}" or c == f"avs__{alias}" or c == f"raw__avs__{alias}":
            return True
        if c.endswith(f"__{alias}"):
            return True
    return False


def check_stage6_serving_state_columns(df: pd.DataFrame, errors: list[str], where: str, *, min_coverage: float = 0.50) -> dict[str, Any]:
    rows = []
    for field in STAGE6_REQUIRED_FIRMSTATE_FIELDS:
        matches = [c for c in df.columns if _stage6_field_alias_matches(str(c), field)]
        coverage = 0.0
        if matches:
            coverage = max(float(pd.to_numeric(df[c], errors="coerce").notna().mean()) for c in matches)
        rows.append({"field": field, "matched_columns": matches[:20], "coverage": coverage, "status": "PASS" if coverage >= min_coverage else "LOW_COVERAGE"})
    bad = [r for r in rows if r["status"] != "PASS"]
    if bad:
        errors.append(f"{where} missing/low coverage Stage6 FirmState serving columns: {bad[:8]}")
    return {"status": "PASS" if not bad else "FAIL", "min_coverage": min_coverage, "rows": rows}


def check_unique(df: pd.DataFrame, keys: list[str], errors: list[str], where: str) -> int:
    if not all(k in df.columns for k in keys):
        errors.append(f"{where} cannot check duplicates; missing key columns {keys}")
        return -1
    dup = int(df.duplicated(keys).sum())
    if dup:
        errors.append(f"{where} duplicate key rows for {keys}: {dup}")
    return dup


def metadata_status(path: Path, errors: list[str], allow_prefix: tuple[str, ...] = ("PASS",)) -> dict[str, Any]:
    if not nonempty(path, errors):
        return {}
    try:
        meta = read_json(path)
    except Exception as exc:
        errors.append(f"cannot read metadata json {path}: {exc!r}")
        return {}
    status = str(meta.get("status", ""))
    if status and not status.startswith(allow_prefix):
        errors.append(f"metadata status not allowed at {path}: {status}")
    return meta


def verify_stage0(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    stage0 = final_root(root) / CANONICAL_STAGE_DIRS["stage0"]
    ok, errs, meta = validate_stage0_contract(stage0)
    if not ok:
        errors.extend(errs)
    nonempty(stage0 / "canonical_panel" / "stage0_canonical_panel.parquet", errors)
    nonempty(stage0 / "canonical_panel" / "statement_items_panel.parquet", errors)
    metadata_status(stage0 / "stage0_manifest.json", errors)
    return {"stage0_contract": meta}


def verify_stage1_bridge(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    p = final / "stage1_oracle_inputs" / "alpha_vanilla_input_candidate.parquet"
    meta_p = final / "stage1_oracle_inputs" / "alpha_vanilla_input_candidate_metadata.json"
    df = read_parquet(p, errors, "Stage1→Stage2 bridge")
    selected_contract = resolve_selected_variables(final)
    selected = selected_contract.get("selected_variables", [])
    if not df.empty:
        check_cols(df, ["firm_id", "fiscal_year", "거래소코드", "year", "rating_num_10", "rating_num", "split", "selected_variables_all_complete"], errors, "bridge")
        check_unique(df, ["firm_id", "fiscal_year"], errors, "bridge")
        missing = [v for v in selected if v not in df.columns]
        if missing:
            errors.append(f"bridge missing dynamic selected variables: {missing}")
        if "selected_variables_all_complete" in df.columns and int(df["selected_variables_all_complete"].sum()) == 0:
            errors.append("bridge has zero selected_variables_all_complete rows")
        if "rating_num_10" in df.columns and "rating_num" in df.columns:
            neq = int((pd.to_numeric(df["rating_num"], errors="coerce") != pd.to_numeric(df["rating_num_10"], errors="coerce")).fillna(False).sum())
            if neq:
                errors.append(f"bridge rating_num alias differs from rating_num_10 rows={neq}")
    bmeta = metadata_status(meta_p, errors)
    return {"rows": int(len(df)), "selected_variable_contract": selected_contract, "bridge_metadata_status": bmeta.get("status")}


def verify_stage1(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    ledger = final / "ledgers" / "stage1_oracle_backends_full_development.json"
    meta = metadata_status(ledger, errors)
    for p in [
        final / "stage1_oracle_backends" / "alpha" / "oracle_alpha_params.json",
        final / "stage1_oracle_backends" / "alpha" / "oracle_firm_year_output_alpha.parquet",
        final / "stage1_oracle_backends" / "beta" / "benchmark_beta_params.json",
        final / "stage1_oracle_backends" / "beta" / "benchmark_firm_year_output_beta.parquet",
        final / "stage1_oracle_backends" / "gamma" / "benchmark_gamma_params.json",
        final / "stage1_oracle_backends" / "gamma" / "benchmark_firm_year_output_gamma.parquet",
        final / "configs" / "oracle_backend_registry.yaml",
    ]:
        nonempty(p, errors)
    bridge = verify_stage1_bridge(root, errors, warnings)
    substrate_path = final / "ledgers" / "stage1_substrate_validation_loopB1.json"
    substrate = metadata_status(substrate_path, errors, allow_prefix=("PASS",))
    if substrate.get("status") not in {"PASS", "PASS_PARTIAL"}:
        errors.append(f"Stage1 substrate validation Loop B1 gate missing or failed: {substrate_path}")
    return {"ledger_status": meta.get("status"), "bridge": bridge, "substrate_validation_loopB1_status": substrate.get("status")}




def _root_from_final_path(path: Path) -> Path:
    parts = list(path.resolve().parts)
    for i in range(len(parts)-1):
        if parts[i] == "data" and i+1 < len(parts) and parts[i+1] == "final_freeze":
            return Path(*parts[:i])
    return path.resolve().parents[3]

def check_action_source_coverage(path: Path, errors: list[str], warnings: list[str], where: str, *, min_observed_rate: float = 0.05) -> dict[str, Any]:
    if not nonempty(path, errors, f"{where} action_source_coverage.csv"):
        return {"status": "missing"}
    try:
        cov = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        cov = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:
        errors.append(f"cannot read {where} action_source_coverage.csv: {exc!r}")
        return {"status": "unreadable"}
    required = [c.replace("action__", "") for c in action_cols(_root_from_final_path(path))]
    if "action_dim" not in cov.columns:
        errors.append(f"{where} action_source_coverage.csv missing action_dim column")
        return {"status": "invalid"}
    dims = set(cov["action_dim"].astype(str))
    missing_rows = sorted(set(required) - dims)
    if missing_rows:
        errors.append(f"{where} action coverage missing dimensions: {missing_rows}")
    bad = []
    for _, r in cov.iterrows():
        dim = str(r.get("action_dim", ""))
        if dim not in required:
            continue
        src = r.get("source_column", None)
        src_missing = pd.isna(src) or str(src).strip() == ""
        rate = pd.to_numeric(pd.Series([r.get("observed_rate", None)]), errors="coerce").iloc[0]
        dim_threshold = max(float(min_observed_rate), float(ACTION_MIN_OBSERVED_RATE.get(dim, min_observed_rate)))
        if src_missing or pd.isna(rate) or float(rate) < dim_threshold:
            bad.append({
                "action_dim": dim,
                "source_column": None if src_missing else str(src),
                "observed_rate": None if pd.isna(rate) else float(rate),
                "min_required_observed_rate": dim_threshold,
            })
    if bad:
        errors.append(
            f"{where} has insufficient direct raw pseudo-action source coverage; proxy fallback is forbidden. "
            + json.dumps(bad, ensure_ascii=False)
        )
    low_warning = []
    if "observed_rate" in cov.columns:
        for _, r in cov.iterrows():
            dim = str(r.get("action_dim", ""))
            if dim not in required:
                continue
            rate = pd.to_numeric(pd.Series([r.get("observed_rate", None)]), errors="coerce").iloc[0]
            if pd.notna(rate) and float(rate) < 0.50:
                low_warning.append({"action_dim": dim, "observed_rate": float(rate), "warning_reference_rate": 0.50})
    if low_warning:
        warnings.append(f"{where} direct raw source coverage below 50% warning reference for structurally sparse dimensions: " + json.dumps(low_warning, ensure_ascii=False))
    return {
        "status": "PASS" if not bad and not missing_rows else "FAIL",
        "global_min_observed_rate": min_observed_rate,
        "action_min_observed_rate": ACTION_MIN_OBSERVED_RATE,
        "warning_reference_observed_rate": 0.50,
        "rows": int(len(cov)),
        "bad_dimensions": bad,
        "low_warning_dimensions": low_warning,
    }


def check_operating_cf_degeneracy(df: pd.DataFrame, *, require_non_degenerate: bool, errors: list[str], warnings: list[str], where: str) -> dict[str, Any]:
    if "sim__operating_cf" not in df.columns:
        msg = f"{where} missing sim__operating_cf for cash-flow degeneracy check"
        (errors if require_non_degenerate else warnings).append(msg)
        return {"status": "MISSING", "nonzero_rate": 0.0}
    x = pd.to_numeric(df["sim__operating_cf"], errors="coerce").fillna(0.0)
    nonzero_rate = float((x.abs() > 1e-12).mean()) if len(x) else 0.0
    status = "PASS" if nonzero_rate > 0.05 else "DEGENERATE"
    if status != "PASS":
        msg = f"{where} sim__operating_cf degenerate: nonzero_rate={nonzero_rate:.4f}"
        (errors if require_non_degenerate else warnings).append(msg)
    return {"status": status, "nonzero_rate": nonzero_rate, "rows": int(len(df))}


def verify_stage2_input(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    d = final / "stage2_candidate_projection" / "input_splits"
    rows = {}
    for name in ["phase1_pretrain", "phase2_bc", "phase3_iql", "phase_eval"]:
        df = read_parquet(d / f"{name}.parquet", errors, name)
        rows[name] = int(len(df))
        if not df.empty:
            if name == "phase_eval":
                check_cols(df, ["firm_id", "fiscal_year"] + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS, errors, name)
                check_stage6_serving_state_columns(df, errors, name)
                forbidden = [c for c in df.columns if str(c).startswith(("action__", "action_observed__", "next__", "soft_cand_")) or c in REWARD_COLS or c in {"candidate_id", "projection_distance", "done"}]
                if forbidden:
                    errors.append(f"phase_eval must be state-only; forbidden columns present: {forbidden[:20]}")
            else:
                check_cols(df, ["firm_id", "fiscal_year"] + ACTION_COLS + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS, errors, name)
                if name == "phase1_pretrain":
                    check_cols(df, [f"next__{c}" for c in ACD_TARGET_COLUMNS], errors, name)
            check_unique(df, ["firm_id", "fiscal_year"], errors, name)
            clipped = [c for c in df.columns if str(c).startswith("clipped_")]
            if clipped:
                errors.append(f"{name} contains forbidden clipped_* columns: {clipped[:20]}")
    meta = metadata_status(d / "metadata.json", errors)
    cf_join = meta.get("cash_flow_substrate_join", {}) if isinstance(meta, dict) else {}
    require_ocf = bool(cf_join.get("cash_flow_substrate_joined", False))
    ocf_checks = {}
    for _phase in ["phase2_bc", "phase3_iql"]:
        _df = read_parquet(d / f"{_phase}.parquet", errors, f"{_phase} OCF check")
        if not _df.empty:
            ocf_checks[_phase] = check_operating_cf_degeneracy(_df, require_non_degenerate=require_ocf, errors=errors, warnings=warnings, where=f"Stage2 input {_phase}")
    coverage_check = check_action_source_coverage(d / "action_source_coverage.csv", errors, warnings, "Stage2 input splits")
    if nonempty(d / "transition_gap_diagnostics.json", errors):
        try:
            _diag = read_json(d / "transition_gap_diagnostics.json")
            _avs = _diag.get("avs256_enrichment", {}) if isinstance(_diag, dict) else {}
            for _name in ("broad", "rated"):
                _meta = _avs.get(_name, {}) if isinstance(_avs, dict) else {}
                _status = _meta.get("transition_proximity_status")
                if _status != "computed":
                    errors.append(f"Stage2 input transition_proximity_status for {_name} must be computed, got {_status!r}")
        except Exception as exc:
            errors.append(f"cannot inspect transition_gap_diagnostics.json transition proximity status: {exc!r}")
    for _f in ["transition_proximity_metadata.json", "transition_proximity_prototypes.parquet"]:
        nonempty(d / _f, errors, _f)
    if (d / "transition_proximity_metadata.json").exists():
        try:
            _tm = read_json(d / "transition_proximity_metadata.json")
            if _tm.get("status") != "PASS":
                errors.append(f"Stage2 transition_proximity_metadata status must be PASS, got {_tm.get('status')!r}")
        except Exception as exc:
            errors.append(f"cannot inspect transition_proximity_metadata.json: {exc!r}")
    return {"rows": rows, "metadata_status": meta.get("status"), "action_source_coverage": coverage_check, "operating_cf_degeneracy": ocf_checks}



def check_recalibrated_candidate_library_uniqueness(path: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing recalibrated candidate library: {path.name}")
        return {"exists": False}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fixed = data.get("fixed_candidates") or {}
        seen: dict[tuple[float, ...], str] = {}
        duplicates = []
        for cid, vec in fixed.items():
            key = tuple(round(float((vec or {}).get(c, 0.0) or 0.0), 12) for c in ACTION_COLS)
            if key in seen:
                duplicates.append((seen[key], cid))
            else:
                seen[key] = str(cid)
        if duplicates:
            errors.append(f"{path.name}: duplicate fixed candidate action vectors after recalibration: {duplicates}")
        meta = data.get("magnitude_recalibration") or {}
        if meta.get("method") != "tier_preserving_per_dimension_inner_train_abs_action_quantile":
            warnings.append(f"{path.name}: unexpected recalibration method {meta.get('method')!r}")
        return {"exists": True, "fixed_candidate_count": len(fixed), "unique_action_vector_count": len(seen), "duplicate_count": len(duplicates)}
    except Exception as exc:
        errors.append(f"cannot inspect recalibrated candidate library {path.name}: {exc!r}")
        return {"exists": True, "error": repr(exc)}

def check_stage2_aux_reward_contract(d: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    meta_path = d / "metadata.json"
    if not meta_path.exists():
        return {"status": "metadata_missing"}
    meta = read_json(meta_path)
    aux = meta.get("aux_reward_stats", {}) if isinstance(meta, dict) else {}
    m_enabled = bool(aux.get("merton_aux_enabled"))
    f_enabled = bool(aux.get("fcff_aux_enabled"))
    l_enabled = bool(aux.get("liquidity_aux_enabled"))
    enabled = bool(m_enabled or f_enabled or l_enabled)
    out = {
        "enabled": enabled,
        "merton_aux_enabled": m_enabled,
        "fcff_aux_enabled": f_enabled,
        "liquidity_aux_enabled": l_enabled,
        "lambda_merton": aux.get("lambda_merton"),
        "lambda_fcff": aux.get("lambda_fcff"),
        "lambda_liquidity": aux.get("lambda_liquidity"),
    }
    if not enabled:
        return out
    if aux.get("reference_oracle_scores_used") is not False or aux.get("reference_oracle_variables_used") is not False or aux.get("r_code_fallback_allowed") is not False:
        errors.append("Stage2 Merton/FCFF/liquidity aux metadata must state oracle_scores_used=false, oracle_variables_used=false, r_code_fallback_allowed=false")
    df = read_parquet(d / "phase3_iql_candidate.parquet", errors, "phase3_iql_candidate aux reward contract")
    if df.empty:
        return out
    required_by_component: list[tuple[str, list[str]]] = []
    if m_enabled:
        required_by_component.append(("Merton", MERTON_AUX_REWARD_COLS))
    if f_enabled:
        required_by_component.append(("FCFF", FCFF_AUX_REWARD_COLS))
    if l_enabled:
        required_by_component.append(("liquidity", LIQUIDITY_AUX_REWARD_COLS))
    for label, cols in required_by_component:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            errors.append(f"phase3_iql_candidate missing enabled {label} auxiliary reward columns: {missing}")
    for c in [x for _, cols in required_by_component for x in cols if x in df.columns]:
        if str(c).startswith("R") or str(c).startswith("next__R"):
            errors.append(f"Stage2 aux reward column path contains forbidden R-code column label: {c}")
    lambda_checks = [
        ("lambda_merton", "Merton"),
        ("lambda_fcff", "FCFF"),
        ("lambda_liquidity", "liquidity"),
    ]
    for col, label in lambda_checks:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist()
            if len(vals) != 1 or abs(float(vals[0]) - float(aux.get(col, 0.0))) > 1e-9:
                errors.append(f"phase3_iql_candidate {col} does not match metadata for {label}: values={vals}, metadata={aux.get(col)}")
        elif abs(float(aux.get(col, 0.0) or 0.0)) > 1e-12:
            errors.append(f"phase3_iql_candidate missing {col} despite nonzero metadata value {aux.get(col)}")
    return out

def verify_stage2(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    d = final / "stage2_candidate_projection"
    rows = {}
    for name in ["phase1_pretrain_candidate", "phase2_bc_candidate", "phase3_iql_candidate", "phase_eval_candidate"]:
        df = read_parquet(d / f"{name}.parquet", errors, name)
        rows[name] = int(len(df))
        if not df.empty:
            if name == "phase_eval_candidate":
                required = ["firm_id", "fiscal_year"] + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
                check_cols(df, required, errors, name)
                check_stage6_serving_state_columns(df, errors, name)
                forbidden = [c for c in df.columns if str(c).startswith(("action__", "action_observed__", "next__", "soft_cand_")) or c in REWARD_COLS or c in {"candidate_id", "projection_distance", "done"}]
                if forbidden:
                    errors.append(f"phase_eval_candidate must be state-only; forbidden columns present: {forbidden[:20]}")
                continue
            required = ["firm_id", "fiscal_year", "candidate_id"] + ACTION_COLS + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
            if name == "phase3_iql_candidate":
                required += REWARD_COLS
            check_cols(df, required, errors, name)
            if name == "phase2_bc_candidate" and any(c in df.columns for c in REWARD_COLS):
                warnings.append(f"{name}: broad BC phase should not require external-rating reward; reward columns are ignored if present")
            if "candidate_id" in df.columns:
                labels = set(df["candidate_id"].dropna().astype(str).unique())
                if "C2" in labels or any(x.startswith("C2") for x in labels):
                    errors.append(f"{name}: C2 appears as train/projected candidate label")
                unknown = sorted(labels - set(V32_LABELS))
                if unknown:
                    errors.append(f"{name}: labels outside v32 main training labels: {unknown}")
    for f in ["metadata.json", "feature_manifest.json", "candidate_library_metadata.json", "projection_support_by_candidate.csv", "magnitude_recalibrated_libraries_metadata.json"]:
        nonempty(d / f, errors)
    # Stage2 input-split magnitude audit artifacts are produced by the split builder,
    # while candidate projection produces the recalibrated P-library metadata.
    for f in ["candidate_magnitude_audit.csv", "magnitude_calibration_metadata.json"]:
        nonempty(d / "input_splits" / f, errors, f"input_splits/{f}")
    # Guard against reusing old Stage2 runs where 8/10 action dimensions were
    # silently zero-imputed. This verifier gate forces a rebuild from Stage2
    # input splits onward when the action source mapper changes.
    coverage_check = check_action_source_coverage(d / "input_splits" / "action_source_coverage.csv", errors, warnings, "Stage2 projection")
    recalibrated_library_checks = {}
    for q in [50, 65, 75, 85]:
        recalibrated_library_checks[f"P{q}"] = check_recalibrated_candidate_library_uniqueness(d / f"final_candidate_library__P{q}.yaml", errors, warnings)
    aux_reward_contract = check_stage2_aux_reward_contract(d, errors, warnings)
    meta = metadata_status(d / "metadata.json", errors)
    return {"rows": rows, "metadata_status": meta.get("status"), "action_source_coverage": coverage_check, "recalibrated_candidate_libraries": recalibrated_library_checks, "aux_reward_contract": aux_reward_contract}





def check_stage3_encoder_architecture(path: Path, schema: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    expected = {
        "d_model": FINAL_ENCODER_D_MODEL,
        "n_heads": FINAL_ENCODER_N_HEADS,
        "n_layers": FINAL_ENCODER_N_LAYERS,
        "ff_multiplier": FINAL_ENCODER_FF_MULTIPLIER,
    }
    observed: dict[str, Any] = {}
    if path.exists() and path.stat().st_size > 0:
        try:
            payload = torch.load(path, map_location="cpu")
            cfg = payload.get("model_config", {}) if isinstance(payload, dict) else {}
            observed = {
                "d_model": cfg.get("d_model"),
                "n_heads": cfg.get("n_heads"),
                "n_layers": cfg.get("n_layers"),
                "ff_multiplier": cfg.get("ff_multiplier", FINAL_ENCODER_FF_MULTIPLIER),
            }
        except Exception as exc:
            errors.append(f"cannot inspect Stage3 ssl_encoder.pt architecture: {exc!r}")
            return {"status": "FAIL", "expected": expected, "observed": observed}
    schema_arch = schema.get("encoder_architecture") or {}
    for key, val in expected.items():
        actual = observed.get(key, schema_arch.get(key))
        try:
            actual_int = int(actual)
        except Exception:
            actual_int = None
        if actual_int != int(val):
            errors.append(f"Stage3 encoder architecture {key} must be {val}, got {actual}")
    return {"status": "PASS" if all(int(observed.get(k, schema_arch.get(k, -999))) == int(v) for k, v in expected.items() if observed.get(k, schema_arch.get(k)) is not None) else "CHECKED", "expected": expected, "observed": observed, "schema_encoder_architecture": schema_arch}

def verify_stage3_downstream_schema(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    s3_meta = final / "stage3_acd_ssl" / "feature_schema.json"
    if not s3_meta.exists():
        return {"status": "SKIP_NO_STAGE3_SCHEMA"}
    try:
        schema = read_json(s3_meta)
    except Exception as exc:
        errors.append(f"cannot read Stage3 feature schema: {exc!r}")
        return {"status": "FAIL"}
    features = list(schema.get("continuous_columns") or [])
    cats = list(schema.get("categorical_columns") or [])
    checks = {}
    phase_paths = {
        "phase1_pretrain": "stage2_candidate_projection/input_splits/phase1_pretrain.parquet",
        "phase2_bc": "stage2_candidate_projection/input_splits/phase2_bc.parquet",
        "phase3_iql": "stage2_candidate_projection/input_splits/phase3_iql.parquet",
        "phase_eval": "stage2_candidate_projection/input_splits/phase_eval.parquet",
        "phase1_pretrain_candidate": "stage2_candidate_projection/phase1_pretrain_candidate.parquet",
        "phase2_bc_candidate": "stage2_candidate_projection/phase2_bc_candidate.parquet",
        "phase3_iql_candidate": "stage2_candidate_projection/phase3_iql_candidate.parquet",
        "phase_eval_candidate": "stage2_candidate_projection/phase_eval_candidate.parquet",
    }
    for name, rel in phase_paths.items():
        df = read_parquet(final / rel, errors, name)
        if df.empty:
            checks[name] = {"rows": 0, "missing_features": features[:20], "missing_categorical": cats}
            continue
        missing = [c for c in features if c not in df.columns]
        missing_cat = [c for c in cats if c not in df.columns]
        missing_next = []
        if name == "phase1_pretrain":
            missing_next = [f"next__{c}" for c in ACD_TARGET_COLUMNS if f"next__{c}" not in df.columns]
        clipped = [c for c in df.columns if str(c).startswith("clipped_")]
        checks[name] = {
            "rows": int(len(df)),
            "missing_feature_count": len(missing),
            "missing_features_sample": missing[:20],
            "missing_categorical": missing_cat,
            "missing_acd_next_target_count": len(missing_next),
            "missing_acd_next_target_sample": missing_next[:20],
            "clipped_columns_sample": clipped[:20],
        }
        if missing:
            errors.append(f"{name} missing Stage3 encoder feature columns; first missing: {missing[:20]}")
        if missing_cat:
            errors.append(f"{name} missing Stage3 categorical columns: {missing_cat}")
        if missing_next:
            errors.append(f"{name} missing ACD next target columns; first missing: {missing_next[:20]}")
        if clipped:
            errors.append(f"{name} contains forbidden clipped_* columns: {clipped[:20]}")
    fail = any(v.get("missing_feature_count", 0) or v.get("missing_categorical") or v.get("missing_acd_next_target_count", 0) or v.get("clipped_columns_sample") for v in checks.values())
    return {"status": "FAIL" if fail else "PASS", "n_features": len(features), "checks": checks}


def verify_stage3(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    d = final_root(root) / "stage3_acd_ssl"
    for f in ["ssl_encoder.pt", "metadata.json", "feature_schema.json", "preprocess_stats.json", "training_log.csv", "feature_leakage_audit.json"]:
        nonempty(d / f, errors)
    meta = metadata_status(d / "metadata.json", errors, allow_prefix=("PASS",))
    # Final-freeze checkpoint contract: final-refit cells must expose the fulltrain
    # encoder alias, while inner-dev winner is required only for actual inner-dev
    # selection runs.  A fixed-config Phase0/final_refit cell intentionally trains
    # no inner-dev winner; treating that absence as a hard failure stops valid
    # downstream Stage4/5/6 archives even though they consume the fulltrain alias.
    fulltrain_alias = d / "stage3_encoder_avs256_final_refit_fulltrain.pt"
    innerdev_alias = d / "stage3_encoder_avs256_innerdev_winner.pt"
    nonempty(fulltrain_alias, errors)
    train_mode = str(meta.get("train_mode") or "").lower()
    if train_mode == "final_refit":
        if not innerdev_alias.exists():
            warnings.append("Stage3 innerdev winner checkpoint missing; accepted for final_refit/fixed-config run because downstream Stage4/5/6 consume stage3_encoder_avs256_final_refit_fulltrain.pt")
        elif innerdev_alias.stat().st_size <= 0:
            errors.append(f"empty {innerdev_alias}")
    else:
        nonempty(innerdev_alias, errors)
    schema = {}
    if (d / "feature_schema.json").exists():
        try:
            schema = read_json(d / "feature_schema.json")
        except Exception as exc:
            errors.append(f"cannot read Stage3 feature_schema.json: {exc!r}")
    if schema:
        if schema.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"Stage3 schema_version must be {SCHEMA_VERSION}, got {schema.get('schema_version')}")
        if int(schema.get("n_continuous_features", -1)) != 129:
            errors.append(f"Stage3 n_continuous_features must equal 129, got {schema.get('n_continuous_features')}")
        if int(schema.get("n_categorical_fields", -1)) != 2:
            errors.append(f"Stage3 n_categorical_fields must equal 2, got {schema.get('n_categorical_fields')}")
        if int(schema.get("n_acd_targets", -1)) != 118:
            errors.append(f"Stage3 n_acd_targets must equal 118, got {schema.get('n_acd_targets')}")
        if schema.get("acd_head_class") != "ActionConditionalForwardHead":
            errors.append(f"Stage3 acd_head_class must be ActionConditionalForwardHead, got {schema.get('acd_head_class')}")
        if schema.get("acd_uses_interaction") is not True:
            errors.append("Stage3 acd_uses_interaction must be true")
        schema_arch = schema.get("encoder_architecture") or {}
        for _k, _v in {"d_model": FINAL_ENCODER_D_MODEL, "n_heads": FINAL_ENCODER_N_HEADS, "n_layers": FINAL_ENCODER_N_LAYERS, "ff_multiplier": FINAL_ENCODER_FF_MULTIPLIER}.items():
            _actual = schema_arch.get(_k)
            if _actual is not None and int(_actual) != int(_v):
                errors.append(f"Stage3 feature_schema encoder_architecture {_k} must be {_v}, got {_actual}")
        if list(schema.get("categorical_columns") or []) != CATEGORICAL_COLUMNS:
            errors.append(f"Stage3 categorical_columns must be {CATEGORICAL_COLUMNS}, got {schema.get('categorical_columns')}")
        if list(schema.get("continuous_columns") or []) != CONTINUOUS_COLUMNS:
            errors.append("Stage3 continuous_columns do not exactly match AVS256 binding order")
        if list(schema.get("acd_target_columns") or []) != ACD_TARGET_COLUMNS:
            errors.append("Stage3 acd_target_columns do not exactly match AVS256 ACD target order")
        if dict(schema.get("block_realized_counts") or {}) != EXPECTED_BLOCK_COUNTS:
            errors.append(f"Stage3 block counts mismatch: {schema.get('block_realized_counts')}")
        if set((schema.get("direction_vocab") or {}).keys()) != set(DIRECTION_VOCAB.keys()):
            errors.append(f"Stage3 direction_vocab keys mismatch: {schema.get('direction_vocab')}")
        for k, expected in [("mcm_weight", 1.0), ("acd_weight", 0.5), ("contrastive_weight", 0.3)]:
            if float(schema.get(k, -999)) != expected:
                errors.append(f"Stage3 {k} must be {expected}, got {schema.get(k)}")
        if any(str(c).startswith("clipped_") for c in (schema.get("continuous_columns") or [])):
            errors.append("Stage3 continuous_columns contain forbidden clipped_* column")
    for k in ["feature_schema_hash", "candidate_library_hash", "final_action_contract_hash", "optimizer", "learning_rate", "weight_decay", "lr_scheduler", "total_steps_mode", "optimizer_steps", "epochs"]:
        if k not in meta:
            errors.append(f"Stage3 metadata missing {k}")
    if meta.get("optimizer") not in (None, "AdamW"):
        errors.append(f"Stage3 optimizer must be AdamW, got {meta.get('optimizer')}")
    if "learning_rate" in meta and float(meta.get("learning_rate", 0.0)) <= 0.0:
        errors.append(f"Stage3 learning_rate must be positive, got {meta.get('learning_rate')}")
    if "weight_decay" in meta and float(meta.get("weight_decay", -1.0)) < 0.0:
        errors.append(f"Stage3 weight_decay must be non-negative, got {meta.get('weight_decay')}")
    if meta.get("lr_scheduler") not in (None, "none"):
        errors.append(f"Stage3 lr_scheduler must be 'none' until scheduler semantics are implemented, got {meta.get('lr_scheduler')}")
    if meta.get("total_steps_mode") not in (None, "epoch_based"):
        errors.append(f"Stage3 total_steps_mode must be epoch_based, got {meta.get('total_steps_mode')}")
    encoder_architecture = check_stage3_encoder_architecture(d / "ssl_encoder.pt", schema, errors)
    downstream_schema = verify_stage3_downstream_schema(root, errors, warnings)
    sweep_grid = check_sensitivity_grid(d / "stage3_sensitivity_phase_alpha.csv", errors, warnings, "Stage3 phase-alpha sensitivity grid", require_selected=False)
    return {"metadata_status": meta.get("status"), "schema_version": schema.get("schema_version"), "encoder_architecture": encoder_architecture, "downstream_schema_compatibility": downstream_schema, "sweep_grid": sweep_grid}


def _load_recalibrated_library_hash(root: Path, q: int, errors: list[str]) -> str:
    path = final_root(root) / "stage2_candidate_projection" / f"final_candidate_library__P{int(q)}.yaml"
    if not path.exists():
        errors.append(f"missing selected recalibrated candidate library for P{q}: {path}")
        return ""
    return sha256_file(path)


def _check_policy_checkpoint_recalibrated_action_payload(root: Path, ckpt_path: Path, stage: str, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    if not ckpt_path.exists():
        errors.append(f"{stage} missing checkpoint for recalibrated action payload check: {ckpt_path.name}")
        return {"exists": False}
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        errors.append(f"{stage} cannot load checkpoint for recalibrated action payload check {ckpt_path.name}: {exc!r}")
        return {"exists": True, "load_error": repr(exc)}
    q = payload.get("selected_magnitude_quantile") or payload.get("magnitude_quantile")
    if q is None:
        errors.append(f"{stage} checkpoint missing selected_magnitude_quantile/magnitude_quantile: {ckpt_path.name}")
        return {"exists": True, "selected_magnitude_quantile": None}
    expected_hash = _load_recalibrated_library_hash(root, int(q), errors)
    found_hash = payload.get("selected_recalibrated_candidate_library_hash")
    if not found_hash:
        errors.append(f"{stage} checkpoint missing selected_recalibrated_candidate_library_hash: {ckpt_path.name}")
    elif expected_hash and str(found_hash) != str(expected_hash):
        errors.append(f"{stage} selected recalibrated candidate library hash mismatch for P{q}: checkpoint={found_hash} expected={expected_hash}")
    if payload.get("candidate_action_values_source") != "stage2_recalibrated_candidate_library":
        errors.append(f"{stage} candidate_action_values_source must be stage2_recalibrated_candidate_library, got {payload.get('candidate_action_values_source')}")
    rows = payload.get("candidate_action_values")
    labels = train_labels(root)
    cols = action_cols(root)
    if not isinstance(rows, list) or not rows:
        errors.append(f"{stage} checkpoint missing non-empty candidate_action_values: {ckpt_path.name}")
        return {"exists": True, "selected_magnitude_quantile": int(q), "candidate_action_value_count": 0}
    by_id = {str(r.get("candidate_id")): r for r in rows if isinstance(r, dict) and r.get("candidate_id") is not None}
    missing = [x for x in labels if x not in by_id]
    if missing:
        errors.append(f"{stage} candidate_action_values missing train labels: {missing[:8]}")
    duplicate_vectors = []
    seen = {}
    for cid in labels:
        r = by_id.get(cid)
        if not r:
            continue
        vec = tuple(round(float(r.get(c, 0.0) or 0.0), 12) for c in cols)
        if vec in seen:
            duplicate_vectors.append((seen[vec], cid))
        else:
            seen[vec] = cid
    if duplicate_vectors:
        errors.append(f"{stage} duplicate fixed candidate action vectors in checkpoint payload: {duplicate_vectors[:8]}")
    return {
        "exists": True,
        "selected_magnitude_quantile": int(q),
        "expected_selected_recalibrated_candidate_library_hash": expected_hash,
        "checkpoint_selected_recalibrated_candidate_library_hash": found_hash,
        "candidate_action_value_count": len(rows),
        "fixed_candidate_payload_count": len([x for x in labels if x in by_id]),
        "unique_fixed_action_vector_count": len(seen),
        "duplicate_fixed_action_vector_count": len(duplicate_vectors),
    }

def verify_stage4(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    d = final_root(root) / "stage4_candidate_bc"
    for f in ["candidate_bc_policy.pt", "metadata.json", "validation_metrics.json"]:
        nonempty(d / f, errors)
    sweep_grid = check_sensitivity_grid(d / "stage4_bc_sensitivity_grid.csv", errors, warnings, "Stage4 BC sensitivity grid")
    # Require at least one explicit final-refit checkpoint artifact.
    if not any(d.glob("stage4_bc_final_refit__P*__*.pt")):
        errors.append("Stage4 missing stage4_bc_final_refit__P{q}__{mode}.pt checkpoint")
    meta = metadata_status(d / "metadata.json", errors)
    if meta.get("hard_target_fallback_used"):
        errors.append("Stage4 hard target fallback was used; final mode forbids it")
    if meta.get("action_vocabulary") and list(meta.get("action_vocabulary")) != train_labels(root):
        errors.append("Stage4 action vocabulary differs from v32 main labels")
    if bool(meta.get("class_balanced_loss")):
        nonempty(d / str(meta.get("class_balance_audit_file") or "class_balance_audit.csv"), errors, "Stage4 class balance audit")
    if bool(meta.get("family_balanced_loss")):
        fam_file = str(meta.get("family_balance_audit_file") or "family_balance_audit.csv")
        q_file = str(meta.get("stage4_label_quality_audit_file") or "stage4_label_quality_audit.csv")
        nonempty(d / fam_file, errors, "Stage4 family balance audit")
        nonempty(d / q_file, errors, "Stage4 label quality audit")
        if not isinstance(meta.get("action_family_by_candidate"), dict) or not meta.get("action_family_by_candidate"):
            errors.append("Stage4 family-balanced run missing action_family_by_candidate metadata")
        if not isinstance(meta.get("loss_weights_by_candidate"), dict) or not meta.get("loss_weights_by_candidate"):
            errors.append("Stage4 family-balanced run missing loss_weights_by_candidate metadata")
        for k in ["family_balance_power", "family_weight_cap", "combined_weight_cap"]:
            if k not in meta:
                errors.append(f"Stage4 family-balanced run missing {k} metadata")
        labels = train_labels(root)
        fam_keys = set(str(x) for x in (meta.get("action_family_by_candidate") or {}).keys())
        if labels and fam_keys and set(labels) != fam_keys:
            errors.append("Stage4 family-balanced action_family_by_candidate keys differ from train labels")
    q = int(meta.get("magnitude_quantile") or 50)
    mode = "lr_scale_0.1" if bool(meta.get("encoder_finetune")) else "frozen"
    final_ckpt = d / f"stage4_bc_final_refit__P{q}__{mode}.pt"
    action_payload = _check_policy_checkpoint_recalibrated_action_payload(root, final_ckpt, "Stage4", errors, warnings)
    return {"metadata_status": meta.get("status"), "sweep_grid": sweep_grid, "recalibrated_action_payload": action_payload}


def verify_stage5(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    d = final_root(root) / "stage5_candidate_iql"
    for f in ["candidate_iql_policy.pt", "metadata.json", "checkpoint_selection_report.json", "validation_metrics.json"]:
        nonempty(d / f, errors)
    # Final Stage6 consumes the full-train final_refit checkpoint.  The inner-dev
    # winner is optional for fast/local final_refit-only reruns and should not block
    # the final Stage6 path when the final_refit artifact exists.
    if not (d / "stage5_candidate_iql_final_refit_fulltrain.pt").exists():
        nonempty(d / "stage5_candidate_iql_final_refit_fulltrain.pt", errors)
    if not (d / "stage5_candidate_iql_innerdev_winner.pt").exists():
        warnings.append("Stage5 innerdev winner checkpoint missing; accepted for fast/local final_refit-only run because Stage6 consumes stage5_candidate_iql_final_refit_fulltrain.pt")
    meta = metadata_status(d / "metadata.json", errors, allow_prefix=("PASS",))
    if meta.get("action_vocabulary") and list(meta.get("action_vocabulary")) != train_labels(root):
        errors.append("Stage5 action vocabulary differs from v32 main labels")
    sweep_grid = check_sensitivity_grid(d / "stage5_sensitivity_phase_gamma.csv", errors, warnings, "Stage5 phase-gamma sensitivity grid")
    for k in ["candidate_library_hash", "final_action_contract_hash", "stage4_checkpoint_hash"]:
        if k not in meta:
            errors.append(f"Stage5 metadata missing {k}")
    if "stage3_schema_hash" not in meta and "stage3_feature_schema_hash" not in meta:
        errors.append("Stage5 metadata missing stage3_schema_hash/stage3_feature_schema_hash")
    action_payload = _check_policy_checkpoint_recalibrated_action_payload(root, d / "stage5_candidate_iql_final_refit_fulltrain.pt", "Stage5", errors, warnings)
    return {"metadata_status": meta.get("status"), "sweep_grid": sweep_grid, "recalibrated_action_payload": action_payload}


def verify_stage6_actions(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    d = final_root(root) / "stage6_candidate_selector_eval"
    df = read_parquet(d / "policy_actions.parquet", errors, "policy_actions")
    if not df.empty:
        check_cols(df, ["row_id", "policy", "candidate_id"] + action_cols(root), errors, "policy_actions")
        policies = set(df["policy"].astype(str)) if "policy" in df.columns else set()
        if "C_obs" in policies:
            errors.append("Stage6 primary policy_actions must not contain C_obs; C_obs is secondary inner-dev only")
        if "C0_noop" not in policies:
            errors.append("Stage6 policy_actions lacks C0_noop policy rows")
        else:
            all_ids=set(pd.to_numeric(df["row_id"], errors="coerce").dropna().astype(int).tolist())
            noop_ids=set(pd.to_numeric(df.loc[df["policy"].astype(str)=="C0_noop", "row_id"], errors="coerce").dropna().astype(int).tolist())
            missing=sorted(all_ids-noop_ids)
            if missing:
                errors.append(f"Stage6 policy_actions no-op pairing violation; missing row_ids sample={missing[:20]} count={len(missing)}")
    meta = metadata_status(d / "metadata.json", errors, allow_prefix=("PASS",))
    return {"rows": int(len(df)), "metadata_status": meta.get("status")}


def verify_stage6(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    d = final_root(root) / "stage6_candidate_selector_eval"
    for f in ["policy_actions.parquet", "simulated_oracle_input_frame.parquet", "action_effect_audit.parquet", "oracle_scores_alpha.parquet", "oracle_scores_beta.parquet", "oracle_scores_gamma.parquet", "multi_oracle_policy_eval.parquet", "multi_oracle_metadata.json"]:
        nonempty(d / f, errors)
    df = read_parquet(d / "multi_oracle_policy_eval.parquet", errors, "multi_oracle_policy_eval")
    sim_df = read_parquet(d / "simulated_oracle_input_frame.parquet", errors, "simulated_oracle_input_frame")
    if not sim_df.empty:
        try:
            _tc = __import__("credit_recourse.rl.common.temporal", fromlist=["load_temporal_contract"]).load_temporal_contract(root)
            # Stage6 currently simulates the serving eval-base state and stamps
            # predicted_fiscal_year with temporal_split.eval_base_year.  Older
            # verifier code expected a removed rollout_target_year attribute,
            # which made the boundary check fail even when Stage6 metadata and
            # outputs were otherwise PASS.  If a future temporal_split.yaml
            # explicitly contains rollout_target_year, honor it from raw;
            # otherwise use the binding eval_base_year used by Stage6 pipelines.
            _expected_pred_year = int(getattr(_tc, "rollout_target_year", (_tc.raw or {}).get("rollout_target_year", _tc.eval_base_year)))
            if "predicted_fiscal_year" not in sim_df.columns:
                errors.append("simulated_oracle_input_frame missing predicted_fiscal_year")
            else:
                _years = sorted([int(x) for x in pd.to_numeric(sim_df["predicted_fiscal_year"], errors="coerce").dropna().unique().tolist()])
                if _years != [_expected_pred_year]:
                    errors.append(f"simulated_oracle_input_frame predicted_fiscal_year must be {_expected_pred_year}, got {_years}")
        except Exception as exc:
            errors.append(f"cannot verify Stage6 predicted_fiscal_year: {exc!r}")
    if not df.empty:
        delta_cols = [c for c in df.columns if str(c).startswith("delta_R_score_")]
        if not delta_cols:
            errors.append("multi_oracle_policy_eval has no explicit delta_R_score_* columns")
        if "policy" in df.columns:
            policies=set(df["policy"].astype(str))
            if "C0_noop" not in policies:
                errors.append("Stage6 evaluation lacks C0_noop policy rows")
            if "C_obs" in policies:
                errors.append("Stage6 primary evaluation must not contain C_obs; C_obs is secondary inner-dev only")
            if "C0_noop" in policies and "row_id" in df.columns:
                all_ids=set(pd.to_numeric(df["row_id"], errors="coerce").dropna().astype(int).tolist())
                noop_ids=set(pd.to_numeric(df.loc[df["policy"].astype(str)=="C0_noop", "row_id"], errors="coerce").dropna().astype(int).tolist())
                missing=sorted(all_ids-noop_ids)
                if missing:
                    errors.append(f"Stage6 evaluation no-op pairing violation; missing row_ids sample={missing[:20]} count={len(missing)}")
    meta = metadata_status(d / "multi_oracle_metadata.json", errors, allow_prefix=("PASS",))
    for extra in ["final_policy_summary.csv", "firm_state_input_audit.json", "policy_pairing_audit.json", "stage6_policy_actions_inner_dev.parquet"]:
        nonempty(d / extra, errors, extra)
    supply = d / "variable_supply_manifest.json"
    if supply.exists():
        sm = read_json(supply)
        missing = sm.get("missing_required_variables_by_backend") or {}
        bad = {k: v for k, v in missing.items() if v}
        if bad:
            errors.append(f"Stage6 missing required variables by backend: {bad}")
    else:
        warnings.append("Stage6 variable_supply_manifest.json not found")
    return {"rows": int(len(df)), "metadata_status": meta.get("status")}



ALL_STAGE_NAMES = [
    "stage0", "stage1", "stage1_bridge", "stage2_input", "stage2",
    "stage3", "stage4", "stage5", "stage6_actions", "stage6",
]

def verify(root: Path, stage: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if stage == "stage0":
        checks = verify_stage0(root, errors, warnings)
    elif stage == "stage1":
        checks = verify_stage1(root, errors, warnings)
    elif stage == "stage1_bridge":
        checks = verify_stage1_bridge(root, errors, warnings)
    elif stage == "stage2_input":
        checks = verify_stage2_input(root, errors, warnings)
    elif stage == "stage2":
        checks = verify_stage2(root, errors, warnings)
    elif stage == "stage3":
        checks = verify_stage3(root, errors, warnings)
    elif stage == "stage4":
        checks = verify_stage4(root, errors, warnings)
    elif stage == "stage5":
        checks = verify_stage5(root, errors, warnings)
    elif stage == "stage6_actions":
        checks = verify_stage6_actions(root, errors, warnings)
    elif stage == "stage6":
        checks = verify_stage6(root, errors, warnings)
    else:
        raise ValueError(f"Unknown stage verifier: {stage}")
    return {
        "stage_name": f"verify_{stage}",
        "contract_version": "stage_boundary_contract_v1_runner_paths",
        "created_utc": now(),
        "status": "PASS" if not errors else "FAIL",
        "sweep_claim_mode": "strict_full_sweep_claim" if STRICT_FULL_SWEEP_CLAIM else "fixed_config_boundary_check",
        "strict_full_sweep_claim": bool(STRICT_FULL_SWEEP_CLAIM),
        "canonical_stage_dirs": CANONICAL_STAGE_DIRS,
        "deprecated_stage_dir_aliases": DEPRECATED_STAGE_DIR_ALIASES,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--stage", default="all", help="Stage to verify, or 'all' to run the full boundary suite")
    ap.add_argument(
        "--strict-full-sweep-claim",
        action="store_true",
        help=(
            "Promote REGISTERED_NOT_RUN sensitivity-grid rows to errors. "
            "Use only when claiming a completed full sensitivity sweep."
        ),
    )
    args = ap.parse_args(argv)
    global STRICT_FULL_SWEEP_CLAIM
    STRICT_FULL_SWEEP_CLAIM = bool(args.strict_full_sweep_claim)
    root = Path(args.project_root).resolve()
    ledger_dir = final_root(root) / "ledgers"
    if str(args.stage).lower() == "all":
        results = [verify(root, stage) for stage in ALL_STAGE_NAMES]
        result = {
            "stage_name": "verify_all",
            "contract_version": "stage_boundary_contract_v1_runner_paths",
            "created_utc": now(),
            "status": "PASS" if all(r.get("status") == "PASS" for r in results) else "FAIL",
            "sweep_claim_mode": "strict_full_sweep_claim" if STRICT_FULL_SWEEP_CLAIM else "fixed_config_boundary_check",
            "strict_full_sweep_claim": bool(STRICT_FULL_SWEEP_CLAIM),
            "stages": results,
            "errors": {r.get("stage_name", "unknown"): r.get("errors", []) for r in results if r.get("errors")},
            "warnings": {r.get("stage_name", "unknown"): r.get("warnings", []) for r in results if r.get("warnings")},
        }
        out = ledger_dir / "verify_all.json"
    else:
        result = verify(root, args.stage)
        out = ledger_dir / f"verify_{args.stage}.json"
    write_json(out, result)
    # Keep CLI output ASCII-escaped so Windows PowerShell 5.x can pipe/capture
    # verifier JSON without corrupting Korean column names into invalid JSON.
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
