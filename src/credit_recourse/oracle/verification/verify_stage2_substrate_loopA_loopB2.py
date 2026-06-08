from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.rl.common.io import final_root, read_parquet_required, write_json
from credit_recourse.rl.common.actions import load_action_space
from credit_recourse.simulator.financial_simulator import FinancialSimulator
from credit_recourse.simulator.business_plan import calibrate_business_plan, BusinessPlan
from credit_recourse.simulator.action import Action, clip_action
from credit_recourse.simulator.firm_state import FirmState, load_firm_state_from_columns
from credit_recourse.contracts.account_registry import resolved_field_values
from credit_recourse.eval.final_stage6_multi_oracle_eval.pipeline import score_alpha, resolve_backend_artifact


MERTON_FIDELITY_FIELDS = {"total_assets", "short_term_debt", "long_term_debt", "bonds"}
FCFF_FIDELITY_FIELDS = {"operating_cf"}
IDENTITY_FIELDS = {"total_assets", "total_liabilities", "total_equity"}
_STATE_VALUE_FIELDS = [
    f.name
    for f in fields(FirmState)
    if f.name not in {"firm_id", "year", "sector", "rating_num", "rating_grade"}
]


# LoopA fidelity must compare simulator-predicted t+1 states against observed
# raw t+1 values.  Stage2 artifacts use several historical naming conventions;
# keep strict actual-side aliases raw-only so strict fidelity does not compare
# simulator output against simulator output.
LOOPA_PREDICTED_ALIASES: dict[str, tuple[str, ...]] = {
    "total_assets": ("total_assets", "next__sim__total_assets", "sim__total_assets"),
    "short_term_debt": ("short_term_debt", "next__sim__short_term_debt", "sim__short_term_debt"),
    "long_term_debt": ("long_term_debt", "next__sim__long_term_debt", "sim__long_term_debt"),
    "bonds": ("bonds", "bond", "next__sim__bonds", "next__sim__bond", "sim__bonds", "sim__bond"),
    "operating_cf": ("operating_cf", "next__sim__operating_cf", "sim__operating_cf"),
}
LOOPA_ACTUAL_NEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "total_assets": ("next__raw__total_assets", "next__total_assets", "total_assets__next"),
    "short_term_debt": ("next__raw__short_term_debt", "next__raw__short_debt", "next__short_term_debt", "short_term_debt__next"),
    "long_term_debt": ("next__raw__long_term_debt", "next__raw__long_debt", "next__long_term_debt", "long_term_debt__next"),
    "bonds": ("next__raw__bonds", "next__raw__bond", "next__bonds", "bonds__next"),
    "operating_cf": ("next__raw__operating_cf", "next__operating_cf", "operating_cf__next"),
}
LOOPA_REQUIRED_ACTUAL_NEXT_COLUMNS = {
    "total_assets": ("next__raw__total_assets",),
    "short_term_debt": ("next__raw__short_term_debt", "next__raw__short_debt"),
    "long_term_debt": ("next__raw__long_term_debt", "next__raw__long_debt"),
    "bonds": ("next__raw__bonds", "next__raw__bond"),
}


def _finite_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _row_to_state(row: pd.Series):
    d = row.to_dict()
    firm_id = str(row.get("firm_id", row.get("corp_code", "UNKNOWN")))
    year = int(float(row.get("fiscal_year", row.get("year", 0)) or 0))
    sector = str(row.get("sector_7", row.get("industry_class", "Unknown")))
    fs = load_firm_state_from_columns(d, firm_id=firm_id, year=year, sector=sector)
    fs.rating_grade = row.get("rating_grade", None)
    fs.rating_num = row.get("rating_num", row.get("rating_num_10", None))
    return fs


def _action(row, space):
    return clip_action(Action(**{c.replace("action__", ""): float(row.get(c, 0.0) or 0.0) for c in space.columns}))


def _select_business_plan(mode: str, history: list[FirmState], *, rating_grade=None) -> BusinessPlan:
    if mode == "default":
        return BusinessPlan()
    if mode == "calibrated":
        return calibrate_business_plan(history, grade=rating_grade) if history else BusinessPlan()
    raise ValueError(f"Unsupported sim_business_plan_mode={mode}")


def _first_present_value(row_dict: dict[Any, Any], candidates: tuple[str, ...]) -> tuple[Any, str | None]:
    for key in candidates:
        if key not in row_dict:
            continue
        val = row_dict.get(key)
        if val is None:
            continue
        try:
            if pd.isna(val):
                continue
        except Exception:
            pass
        return val, key
    return None, None


def _actual_next_values(row: pd.Series) -> dict[str, Any]:
    """Resolve observed t+1 values from Stage2 handoff columns.

    Strict LoopA fidelity must use observed next raw values as the actual side.
    Do not let next__sim__* satisfy the actual side: that would compare the
    simulator against its own simulated transition and would mask exploitation.
    """
    d = row.to_dict()
    out: dict[str, Any] = {}

    # Explicit raw-next aliases first.  These are the canonical actual side for
    # observed t+1 comparisons in phase3_iql_candidate__P50 artifacts.
    for field in _STATE_VALUE_FIELDS:
        aliases = LOOPA_ACTUAL_NEXT_ALIASES.get(field, (f"next__raw__{field}", f"next__{field}", f"{field}__next"))
        val, _ = _first_present_value(d, aliases)
        if val is not None:
            out[field] = val

    # Registry/U-code fallback can recover legacy raw next U-code columns, but
    # keep the explicit raw aliases above dominant.
    reg_out = dict(resolved_field_values(d, next_state=True))
    for field, val in reg_out.items():
        if field not in out and field in _STATE_VALUE_FIELDS:
            out[field] = val
    return out


def _predicted_next_values(pred_row: dict[str, Any]) -> dict[str, Any]:
    """Resolve simulator-predicted t+1 values to canonical dimensions."""
    out: dict[str, Any] = {}
    for field in _STATE_VALUE_FIELDS:
        aliases = LOOPA_PREDICTED_ALIASES.get(field, (field, f"next__sim__{field}", f"sim__{field}"))
        val, _ = _first_present_value(pred_row, aliases)
        if val is not None:
            out[field] = val
    return out


def _component_group(dimension: str) -> str:
    if dimension in MERTON_FIDELITY_FIELDS:
        return "merton"
    if dimension in FCFF_FIDELITY_FIELDS:
        return "fcff"
    if dimension in IDENTITY_FIELDS:
        return "accounting_identity"
    return "state"


def _numeric_state_dict(fs: FirmState) -> dict[str, float]:
    d = fs.to_dict()
    return {k: _finite_float(v) for k, v in d.items() if k in _STATE_VALUE_FIELDS}


def _compute_error_rows(phase: pd.DataFrame, sim_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for (_, r), pred_row_raw in zip(phase.iterrows(), sim_rows):
        actual_next = _actual_next_values(r)
        pred_row = _predicted_next_values(pred_row_raw)
        current_assets = _finite_float(r.get("sim__total_assets", r.get("raw__total_assets", r.get("total_assets", np.nan))))
        actual_assets = _finite_float(actual_next.get("total_assets", np.nan))
        predicted_assets = _finite_float(pred_row.get("total_assets", np.nan))
        asset_scale = np.nanmax(np.abs([current_assets, actual_assets, predicted_assets]))
        if not np.isfinite(asset_scale) or asset_scale <= 1e-9:
            asset_scale = np.nan
        for field in _STATE_VALUE_FIELDS:
            if field not in pred_row or field not in actual_next:
                continue
            pred = _finite_float(pred_row.get(field))
            actual = _finite_float(actual_next.get(field))
            if not (np.isfinite(pred) and np.isfinite(actual)):
                continue
            signed = pred - actual
            abs_err = abs(signed)
            rows.append(
                {
                    "firm_id": str(r.get("firm_id", r.get("corp_code", "UNKNOWN"))),
                    "fiscal_year": int(float(r.get("fiscal_year", r.get("year", 0)) or 0)),
                    "dimension": field,
                    "component_group": _component_group(field),
                    "predicted": pred,
                    "actual": actual,
                    "signed_error": signed,
                    "abs_error": abs_err,
                    "asset_scale": asset_scale,
                    "abs_err_over_assets": abs_err / asset_scale if np.isfinite(asset_scale) and asset_scale > 0 else np.nan,
                    "signed_err_over_assets": signed / asset_scale if np.isfinite(asset_scale) and asset_scale > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return float("nan")
    if float(x[m].nunique()) < 2 or float(y[m].nunique()) < 2:
        return float("nan")
    return float(x[m].rank().corr(y[m].rank()))


def _summarize_errors(errs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dimension",
        "component_group",
        "count",
        "coverage_share",
        "mean_signed_error",
        "median_signed_error",
        "std_signed_error",
        "mean_abs_error",
        "median_abs_error",
        "p95_abs_error",
        "mean_abs_err_over_assets",
        "median_abs_err_over_assets",
        "p95_abs_err_over_assets",
        "spearman",
        "predicted_nonzero_rate",
        "actual_nonzero_rate",
    ]
    if errs.empty:
        return pd.DataFrame(columns=cols)

    total_rows = max(int(errs[["firm_id", "fiscal_year"]].drop_duplicates().shape[0]), 1)
    records = []
    for dim, g in errs.groupby("dimension", dropna=False):
        pred = pd.to_numeric(g["predicted"], errors="coerce")
        actual = pd.to_numeric(g["actual"], errors="coerce")
        signed = pd.to_numeric(g["signed_error"], errors="coerce")
        abs_err = pd.to_numeric(g["abs_error"], errors="coerce")
        rel = pd.to_numeric(g["abs_err_over_assets"], errors="coerce")
        records.append(
            {
                "dimension": str(dim),
                "component_group": _component_group(str(dim)),
                "count": int(len(g)),
                "coverage_share": float(len(g) / total_rows),
                "mean_signed_error": float(signed.mean()) if signed.notna().any() else np.nan,
                "median_signed_error": float(signed.median()) if signed.notna().any() else np.nan,
                "std_signed_error": float(signed.std()) if signed.notna().sum() > 1 else np.nan,
                "mean_abs_error": float(abs_err.mean()) if abs_err.notna().any() else np.nan,
                "median_abs_error": float(abs_err.median()) if abs_err.notna().any() else np.nan,
                "p95_abs_error": float(abs_err.quantile(0.95)) if abs_err.notna().any() else np.nan,
                "mean_abs_err_over_assets": float(rel.mean()) if rel.notna().any() else np.nan,
                "median_abs_err_over_assets": float(rel.median()) if rel.notna().any() else np.nan,
                "p95_abs_err_over_assets": float(rel.quantile(0.95)) if rel.notna().any() else np.nan,
                "spearman": _spearman(pred, actual),
                "predicted_nonzero_rate": float((pred.fillna(0.0).abs() > 1e-12).mean()),
                "actual_nonzero_rate": float((actual.fillna(0.0).abs() > 1e-12).mean()),
            }
        )
    return pd.DataFrame(records, columns=cols)


def _group_gate(summary: pd.DataFrame, *, max_rel_err_assets: float, min_coverage_share: float) -> dict:
    required = {
        "merton": sorted(MERTON_FIDELITY_FIELDS),
    }
    out = {"max_rel_err_assets": float(max_rel_err_assets), "min_coverage_share": float(min_coverage_share), "groups": {}, "violations": []}
    for group, dims in required.items():
        sub = summary[summary["component_group"].astype(str).eq(group)].copy() if not summary.empty else pd.DataFrame()
        present = sorted(set(sub["dimension"].astype(str))) if not sub.empty else []
        missing = sorted(set(dims) - set(present))
        rel = pd.to_numeric(sub.get("median_abs_err_over_assets", pd.Series(dtype=float)), errors="coerce")
        cov = pd.to_numeric(sub.get("coverage_share", pd.Series(dtype=float)), errors="coerce")
        worst_rel = float(rel.max()) if rel.notna().any() else float("nan")
        min_cov = float(cov.min()) if cov.notna().any() else 0.0
        rec = {"required_dimensions": dims, "present_dimensions": present, "missing_dimensions": missing, "worst_median_abs_err_over_assets": worst_rel, "min_coverage_share": min_cov}
        out["groups"][group] = rec
        if missing:
            out["violations"].append({"component_group": group, "reason": "missing_required_dimensions", "missing": missing})
        if (not np.isfinite(worst_rel)) or worst_rel > float(max_rel_err_assets):
            out["violations"].append({"component_group": group, "reason": "relative_error_threshold", "worst_median_abs_err_over_assets": worst_rel, "threshold": float(max_rel_err_assets)})
        if min_cov < float(min_coverage_share):
            out["violations"].append({"component_group": group, "reason": "coverage_below_threshold", "min_coverage_share": min_cov, "threshold": float(min_coverage_share)})
    out["status"] = "PASS" if not out["violations"] else "FAIL"
    return out


def _write_compatibility_outputs(primary_out: Path, final: Path, files: list[str]) -> list[str]:
    targets = [
        final / "stage2_substrate_validation",
        final / "stage2_candidate_projection" / "verification",
        final / "verification",
    ]
    written = []
    for d in targets:
        if d == primary_out:
            continue
        d.mkdir(parents=True, exist_ok=True)
        for name in files:
            src = primary_out / name
            if src.exists():
                dst = d / name
                shutil.copy2(src, dst)
                written.append(str(dst))
    return written


def _normalise_loopa_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "firm_id" not in out.columns and "corp_code" in out.columns:
        out["firm_id"] = out["corp_code"]
    if "fiscal_year" not in out.columns and "year" in out.columns:
        out["fiscal_year"] = out["year"]
    if "firm_id" not in out.columns or "fiscal_year" not in out.columns:
        raise ValueError("LoopA fidelity requires firm_id and fiscal_year/year columns for observed-next alignment")
    out["firm_id"] = out["firm_id"].astype(str)
    out["fiscal_year"] = pd.to_numeric(out["fiscal_year"], errors="coerce").astype("Int64")
    return out


def _has_merton_actual_next_columns(df: pd.DataFrame) -> bool:
    cols = set(map(str, df.columns))
    return all(any(alias in cols for alias in aliases) for aliases in LOOPA_REQUIRED_ACTUAL_NEXT_COLUMNS.values())


def _load_observed_next_handoff(final: Path) -> tuple[pd.DataFrame, str]:
    """Load a Stage2 handoff that carries observed next raw columns.

    phase_eval_candidate.parquet is the driver for LoopA, but some runs keep
    observed next raw columns only in phase3_iql_candidate__P50.parquet.  Use
    observed artifacts only for the actual side; never use next__sim__ fallback
    as an observed target in strict fidelity.
    """
    candidates = [
        final / "stage2_candidate_projection" / "phase_eval_candidate.parquet",
        final / "stage2_candidate_projection" / "phase3_iql_candidate__P50.parquet",
        final / "stage2_candidate_projection" / "phase3_iql_counterfactual_candidate__P50.parquet",
    ]
    available: list[tuple[Path, pd.DataFrame]] = []
    for path in candidates:
        if not path.exists():
            continue
        df = _normalise_loopa_keys(read_parquet_required(path))
        available.append((path, df))
        if _has_merton_actual_next_columns(df):
            return df, str(path)

    details = {str(path): list(map(str, df.columns[:80])) for path, df in available}
    raise ValueError(
        "No LoopA observed-next handoff with required raw next Merton columns. "
        "Expected aliases include next__raw__total_assets, next__raw__short_term_debt, "
        "next__raw__long_term_debt, next__raw__bonds. Available leading columns: "
        + json.dumps(details, ensure_ascii=False, default=str)[:4000]
    )


def _merge_observed_next_into_phase(phase: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    phase_n = _normalise_loopa_keys(phase)
    obs_n = _normalise_loopa_keys(observed)
    actual_cols: list[str] = []
    for aliases in LOOPA_ACTUAL_NEXT_ALIASES.values():
        for alias in aliases:
            if alias in obs_n.columns and alias not in actual_cols:
                actual_cols.append(alias)
    if not actual_cols:
        raise ValueError("Observed next handoff has no recognized LoopA actual-next columns after alias resolution")
    keep = ["firm_id", "fiscal_year"] + actual_cols
    obs_small = obs_n[keep].drop_duplicates(subset=["firm_id", "fiscal_year"], keep="first")

    # Preserve phase columns; add missing actual-next columns from observed handoff.
    missing_actual = [c for c in actual_cols if c not in phase_n.columns]
    if not missing_actual and _has_merton_actual_next_columns(phase_n):
        phase_n.attrs["loopA_actual_next_source"] = "phase_eval_candidate.parquet"
        return phase_n

    merged = phase_n.merge(obs_small[["firm_id", "fiscal_year"] + missing_actual], on=["firm_id", "fiscal_year"], how="left", validate="many_to_one")
    merged.attrs["loopA_actual_next_source"] = "observed_stage2_handoff_merge"
    matched = int(merged[missing_actual].notna().any(axis=1).sum()) if missing_actual else int(len(merged))
    if matched <= 0:
        raise ValueError("LoopA observed-next merge produced zero rows with actual next values; check firm_id/fiscal_year alignment")
    return merged


def _actual_next_match_count(df: pd.DataFrame) -> int:
    """Count rows that carry at least one required observed raw-next Merton value."""
    if df.empty:
        return 0
    cols = [alias for aliases in LOOPA_REQUIRED_ACTUAL_NEXT_COLUMNS.values() for alias in aliases if alias in df.columns]
    if not cols:
        return 0
    return int(df[cols].notna().any(axis=1).sum())


def _build_loopa_phase(final: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the LoopA driver with observed raw next-state targets.

    `phase_eval_candidate.parquet` is the preferred driver when it can be
    aligned with observed next raw columns.  Some final-freeze runs use OOT
    evaluation rows whose `firm_id + fiscal_year` keys do not overlap the
    observed transition handoff.  In that case, fidelity must fall back to the
    observed Stage2 handoff itself instead of raising before any error can be
    measured.  The fallback still compares simulator predictions against
    `next__raw__*` observed targets; it never uses `next__sim__*` as actuals.
    """
    phase_path = final / "stage2_candidate_projection" / "phase_eval_candidate.parquet"
    phase_raw = read_parquet_required(phase_path)
    if phase_raw.empty:
        raise ValueError("LoopA/B2 verifier received empty phase_eval_candidate.parquet")

    observed_next, observed_next_source = _load_observed_next_handoff(final)
    meta: dict[str, Any] = {
        "loopA_phase_eval_source": str(phase_path),
        "loopA_observed_next_source": observed_next_source,
        "loopA_driver_source": str(phase_path),
        "loopA_driver_fallback_reason": None,
    }

    try:
        phase_candidate = _merge_observed_next_into_phase(phase_raw, observed_next)
        match_rows = _actual_next_match_count(phase_candidate)
        if match_rows > 0:
            meta["loopA_driver_source"] = str(phase_path)
            meta["loopA_observed_next_match_rows"] = int(match_rows)
            meta["loopA_actual_next_source"] = phase_candidate.attrs.get("loopA_actual_next_source", observed_next_source)
            return phase_candidate, meta
        meta["loopA_driver_fallback_reason"] = "phase_eval_merge_produced_zero_actual_next_rows"
    except Exception as exc:
        meta["loopA_driver_fallback_reason"] = f"phase_eval_merge_failed: {type(exc).__name__}: {exc}"

    # Fallback: use the observed Stage2 transition handoff as the LoopA driver.
    # This is the only table guaranteed to carry observed next raw financials.
    fallback = _normalise_loopa_keys(observed_next)
    match_rows = _actual_next_match_count(fallback)
    if match_rows <= 0:
        raise ValueError(
            "LoopA observed handoff fallback has zero rows with actual next raw values; "
            "check phase3_iql_candidate__P50 observed next columns"
        )
    meta["loopA_driver_source"] = observed_next_source
    meta["loopA_actual_next_source"] = observed_next_source
    meta["loopA_observed_next_match_rows"] = int(match_rows)
    meta["loopA_phase_eval_rows"] = int(len(phase_raw))
    meta["loopA_driver_fallback_used"] = True
    return fallback, meta


def run(
    project_root: Path,
    *,
    sim_business_plan_mode: str = "default",
    max_rel_err_assets: float = 0.05,
    min_coverage_share: float = 0.80,
    output_dir: Path | None = None,
) -> dict:
    root = project_root.resolve()
    final = final_root(root)
    out = output_dir if output_dir is not None else final / "stage2_substrate_loopA_loopB2"
    out.mkdir(parents=True, exist_ok=True)

    phase, loopa_driver_meta = _build_loopa_phase(final)

    space = load_action_space(root)
    sim = FinancialSimulator()
    sim_rows = []
    for _, r in phase.iterrows():
        fs = _row_to_state(r)
        act = _action(r, space)
        hist = (
            phase[
                (phase.get("firm_id", "").astype(str) == str(fs.firm_id))
                & (pd.to_numeric(phase.get("fiscal_year", 0), errors="coerce") <= float(fs.year))
            ]
            .sort_values("fiscal_year")
            .tail(3)
            if "firm_id" in phase.columns
            else pd.DataFrame()
        )
        h = []
        for _, hr in hist.iterrows():
            try:
                h.append(_row_to_state(hr))
            except Exception:
                pass
        bp = _select_business_plan(sim_business_plan_mode, h, rating_grade=fs.rating_grade)
        res = sim.simulate(fs, bp, act)
        row = {"firm_id": fs.firm_id, "fiscal_year": fs.year}
        row.update(_numeric_state_dict(res.state_t1))
        row["simulator_sustainability"] = res.sustainability
        row["simulator_plug_used"] = res.plug_used
        row["simulator_plug_amount"] = float(res.plug_amount)
        sim_rows.append(row)

    pred = pd.DataFrame(sim_rows)
    errs = _compute_error_rows(phase, sim_rows)
    summary = _summarize_errors(errs)
    gate = _group_gate(summary, max_rel_err_assets=max_rel_err_assets, min_coverage_share=min_coverage_share)

    if errs.empty:
        gate["status"] = "FAIL"
        gate.setdefault("violations", []).append({"reason": "no_comparable_next_state_values"})
    pred.to_parquet(out / "loopA_predicted_tplus1_states.parquet", index=False)
    errs.to_csv(out / "loopA_dimension_errors.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "loopA_dimension_bias_summary.csv", index=False, encoding="utf-8-sig")

    # Loop B2: Alpha score movement from simulated t+1; real lead rating movement if available.
    registry_path = final / "oracle_backend_registry.json"
    b2 = {"status": "SKIP_NO_REGISTRY", "reason": f"missing {registry_path}"}
    if registry_path.exists() and not pred.empty:
        reg = json.loads(registry_path.read_text(encoding="utf-8"))
        params = resolve_backend_artifact(root, final, reg.get("alpha", {}).get("params", ""))
        if params.exists():
            score = score_alpha(pred, params)
            base_cols = ["firm_id", "fiscal_year"]
            if "rating_num_10" in phase.columns:
                base_cols.append("rating_num_10")
            b2_df = phase[base_cols].copy() if set(base_cols).issubset(phase.columns) else phase[["firm_id", "fiscal_year"]].copy()
            b2_df["pred_alpha_score_tplus1"] = score.to_numpy()
            if "rating_num_10__next" in phase.columns and "rating_num_10" in phase.columns:
                b2_df["real_rating_delta_t_to_tplus1"] = pd.to_numeric(phase["rating_num_10"], errors="coerce") - pd.to_numeric(phase["rating_num_10__next"], errors="coerce")
            b2_df.to_csv(out / "loopB2_alpha_predicted_score_vs_real_rating_change.csv", index=False, encoding="utf-8-sig")
            b2 = {"status": "PASS", "rows": int(len(b2_df))}
        else:
            b2 = {"status": "SKIP_NO_ALPHA_PARAMS", "reason": str(params)}

    files = [
        "loopA_predicted_tplus1_states.parquet",
        "loopA_dimension_errors.csv",
        "loopA_dimension_bias_summary.csv",
        "substrate_loopA_loopB2_report.json",
    ]
    if (out / "loopB2_alpha_predicted_score_vs_real_rating_change.csv").exists():
        files.append("loopB2_alpha_predicted_score_vs_real_rating_change.csv")

    meta = {
        "status": "PASS" if gate["status"] == "PASS" else "FAIL",
        "sim_business_plan_mode": sim_business_plan_mode,
        "loopA_rows": int(len(pred)),
        "loopA_phase_rows": int(len(phase)),
        **loopa_driver_meta,
        "loopA_comparable_error_rows": int(len(errs)),
        "loopA_dimensions": int(summary["dimension"].nunique()) if not summary.empty else 0,
        "loopA_fidelity_gate": gate,
        "loopB2": b2,
        "outputs": files,
        "primary_output_dir": str(out),
        "compatibility_output_dirs": [],
    }
    write_json(out / "substrate_loopA_loopB2_report.json", meta)
    compat = _write_compatibility_outputs(out, final, files)
    meta["compatibility_output_dirs"] = sorted(set(str(Path(x).parent) for x in compat))
    write_json(out / "substrate_loopA_loopB2_report.json", meta)
    _write_compatibility_outputs(out, final, files)

    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--sim-business-plan-mode", choices=["default", "calibrated"], default="default")
    ap.add_argument("--max-rel-err-assets", type=float, default=0.05)
    ap.add_argument("--min-coverage-share", type=float, default=0.80)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)
    meta = run(
        Path(args.project_root).resolve(),
        sim_business_plan_mode=args.sim_business_plan_mode,
        max_rel_err_assets=args.max_rel_err_assets,
        min_coverage_share=args.min_coverage_share,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0 if meta.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
