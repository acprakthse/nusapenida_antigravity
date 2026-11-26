"""
Economic analysis module for the Fully Isolated Nusa Penida Microgrid (20 kV AC-Coupled).
Topology: PLTD (PLN + Leased Diesel) + Solar PV + BESS on isolated 20 kV bus (no external grid).

In this fully isolated system:
  - `grid_price_usd_kwh` represents the Diesel Generation Production Cost (BPP PLTD, USD/kWh).
  - `total_mdu_pancabina_kw` (when present in the input CSV) represents the real measured
    PLTD generation used as the Diesel-Only Baseline.

Computes:
  1. Solar PV Only LCOE (USD/kWh)
  2. Combined Solar PV + BESS LCOE (USD/kWh)
  3. Hybrid Isolated System Blended Cost (Dispatched Diesel * BPP + Annualized PV/BESS)
  4. Measured Diesel-Only Baseline (using `total_mdu_pancabina_kw` * BPP PLTD)
  5. Unit cost spread ($/kWh) and annual cost comparison ($/year).
"""

from __future__ import annotations
from typing import Dict, Any
import os
import numpy as np
import pandas as pd


DEFAULT_ECONOMICS: Dict[str, float] = {
    "pv_capacity_kwp": 4200.0,           # Suana Solar PV DC capacity (kWp)
    "pv_capex_usd_per_kwp": 850.0,       # Solar PV CAPEX (USD/kWp)
    "pv_opex_usd_per_kwp_yr": 15.0,      # Solar PV fixed O&M (USD/kWp-year)
    "bess_power_kw": 2000.0,             # BESS rated power (kW)
    "bess_energy_kwh": 2200.0,           # BESS nominal energy capacity (kWh)
    "bess_capex_usd_per_kwh": 320.0,     # BESS CAPEX (USD/kWh)
    "bess_opex_usd_per_kwh_yr": 8.0,     # BESS fixed O&M (USD/kWh-year)
    "discount_rate": 0.08,               # Discount rate / WACC (fraction, e.g. 0.08 = 8%)
    "project_lifetime_yr": 20.0,         # Project lifetime (years)
    "bess_replacement_yr": 10.0,         # BESS battery replacement year (0 = no replacement)
    "bess_replacement_cost_frac": 0.60,  # Replacement cost as fraction of initial BESS CAPEX
}

ECONOMICS_METADATA = [
    ("pv_capacity_kwp",            "kWp",        "Solar PV installed DC capacity"),
    ("pv_capex_usd_per_kwp",       "USD/kWp",    "Solar PV capital expenditure (CAPEX)"),
    ("pv_opex_usd_per_kwp_yr",     "USD/kWp-yr", "Solar PV annual O&M cost (OPEX)"),
    ("bess_power_kw",              "kW",         "BESS inverter power capacity"),
    ("bess_energy_kwh",            "kWh",        "BESS nominal energy capacity"),
    ("bess_capex_usd_per_kwh",     "USD/kWh",    "BESS capital expenditure (CAPEX)"),
    ("bess_opex_usd_per_kwh_yr",   "USD/kWh-yr", "BESS annual O&M cost (OPEX)"),
    ("discount_rate",              "fraction",   "Discount rate / WACC (e.g. 0.08 = 8%)"),
    ("project_lifetime_yr",        "years",      "Project economic lifetime"),
    ("bess_replacement_yr",        "years",      "BESS replacement year (0 to disable)"),
    ("bess_replacement_cost_frac", "fraction",   "BESS replacement cost fraction of initial CAPEX"),
]


def save_economics_template_csv(
    csv_path: str = "input/input_economics.csv",
    params: Dict[str, float] | None = None,
) -> None:
    """Save a clean, editable PV + BESS CAPEX/OPEX economics template CSV."""
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cfg = dict(DEFAULT_ECONOMICS)
    if params:
        cfg.update(params)
    rows = [
        {
            "parameter": key,
            "value": cfg[key],
            "unit": unit,
            "description": desc,
        }
        for key, unit, desc in ECONOMICS_METADATA
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def load_economics_from_csv(csv_path: str) -> Dict[str, float]:
    """Load PV + BESS CAPEX/OPEX parameters from a key-value CSV file."""
    df = pd.read_csv(csv_path)
    if "parameter" not in df.columns or "value" not in df.columns:
        raise ValueError(
            f"Economics CSV must contain 'parameter' and 'value' columns. "
            f"Found columns: {list(df.columns)}"
        )
    params = dict(DEFAULT_ECONOMICS)
    for _, row in df.iterrows():
        key = str(row["parameter"]).strip()
        if key in params:
            params[key] = float(row["value"])
    return params


def capital_recovery_factor(discount_rate: float, lifetime_yr: float) -> float:
    """Compute the Capital Recovery Factor (CRF) for annualizing CAPEX."""
    if lifetime_yr <= 0:
        raise ValueError("Project lifetime must be positive.")
    if discount_rate <= 0:
        return 1.0 / lifetime_yr
    factor = (1.0 + discount_rate) ** lifetime_yr
    return (discount_rate * factor) / (factor - 1.0)


def compute_lcoe_and_comparison(
    df_dispatch: pd.DataFrame,
    dt_hours: float = 0.5,
    economics: Dict[str, float] | None = None,
    default_diesel_bpp: float = 0.2488,
) -> Dict[str, Any]:
    """
    Compute PV-only LCOE, combined PV+BESS LCOE, hybrid blended system cost,
    and comparison against the real measured Diesel-Only Baseline (`total_mdu_pancabina_kw`).
    """
    cfg = dict(DEFAULT_ECONOMICS)
    if economics:
        cfg.update(economics)

    r = float(cfg["discount_rate"])
    n_yr = float(cfg["project_lifetime_yr"])
    crf = capital_recovery_factor(r, n_yr)

    # 1. PV & BESS CAPEX + OPEX Annualized Costs (USD)
    pv_capex_total = cfg["pv_capacity_kwp"] * cfg["pv_capex_usd_per_kwp"]
    pv_annual_capex = pv_capex_total * crf
    pv_annual_opex = cfg["pv_capacity_kwp"] * cfg["pv_opex_usd_per_kwp_yr"]
    pv_annual_total_cost = pv_annual_capex + pv_annual_opex

    bess_capex_total = cfg["bess_energy_kwh"] * cfg["bess_capex_usd_per_kwh"]
    repl_yr = float(cfg["bess_replacement_yr"])
    bess_repl_pv = 0.0
    if 0 < repl_yr < n_yr:
        bess_repl_pv = (bess_capex_total * cfg["bess_replacement_cost_frac"]) / ((1.0 + r) ** repl_yr)
    bess_annual_capex = (bess_capex_total + bess_repl_pv) * crf
    bess_annual_opex = cfg["bess_energy_kwh"] * cfg["bess_opex_usd_per_kwh_yr"]
    bess_annual_total_cost = bess_annual_capex + bess_annual_opex

    total_capex_usd = pv_capex_total + bess_capex_total
    total_annual_pv_bess_cost_usd = pv_annual_total_cost + bess_annual_total_cost

    # 2. Annualized Energy Flows from Dispatch Horizon
    horizon_hours = len(df_dispatch) * dt_hours
    horizon_days = max(horizon_hours / 24.0, 1e-6)
    annual_factor = 365.0 / horizon_days

    load_arr = df_dispatch["load_kw"].to_numpy(dtype=float)
    pv_arr = df_dispatch["pv_used_kw"].to_numpy(dtype=float)
    diesel_arr = df_dispatch["pltd_total_kw"].to_numpy(dtype=float)

    if "grid_price_usd_kwh" in df_dispatch.columns:
        bpp_arr = df_dispatch["grid_price_usd_kwh"].to_numpy(dtype=float)
    else:
        bpp_arr = np.full(len(df_dispatch), default_diesel_bpp, dtype=float)

    # Use real measured PLTD (`total_mdu_pancabina_kw`) for Diesel-Only baseline if available
    if "total_mdu_pancabina_kw" in df_dispatch.columns:
        baseline_pltd_arr = (
            pd.to_numeric(df_dispatch["total_mdu_pancabina_kw"], errors="coerce")
            .ffill()
            .bfill()
            .fillna(0.0)
            .clip(lower=0.0)
            .to_numpy(dtype=float)
        )
        baseline_source = "Measured PLTD (total_mdu_pancabina_kw)"
    else:
        baseline_pltd_arr = load_arr
        baseline_source = "System Load (load_kw)"

    annual_pv_used_kwh = float(np.sum(pv_arr) * dt_hours) * annual_factor
    annual_load_kwh = float(np.sum(load_arr) * dt_hours) * annual_factor
    annual_diesel_kwh = float(np.sum(diesel_arr) * dt_hours) * annual_factor
    annual_baseline_pltd_kwh = float(np.sum(baseline_pltd_arr) * dt_hours) * annual_factor

    # 3. LCOE (USD/kWh)
    lcoe_pv_only = pv_annual_total_cost / annual_pv_used_kwh if annual_pv_used_kwh > 0 else float("nan")
    lcoe_pv_bess = total_annual_pv_bess_cost_usd / annual_pv_used_kwh if annual_pv_used_kwh > 0 else float("nan")

    # 4. Measured Diesel-Only Baseline vs. Hybrid (PV + BESS + PLTD)
    diesel_bpp_avg = float(np.mean(bpp_arr))
    diesel_only_annual_cost_usd = float(np.sum(baseline_pltd_arr * bpp_arr) * dt_hours) * annual_factor
    diesel_bpp_weighted = (
        diesel_only_annual_cost_usd / annual_baseline_pltd_kwh
        if annual_baseline_pltd_kwh > 0
        else diesel_bpp_avg
    )

    hybrid_diesel_annual_cost_usd = float(np.sum(diesel_arr * bpp_arr) * dt_hours) * annual_factor
    hybrid_total_annual_cost_usd = hybrid_diesel_annual_cost_usd + total_annual_pv_bess_cost_usd
    hybrid_blended_cost_usd_kwh = (
        hybrid_total_annual_cost_usd / annual_load_kwh if annual_load_kwh > 0 else float("nan")
    )

    # 5. Cost Differences & Savings
    # 5a. Unit cost difference (Measured PLTD BPP - PV+BESS LCOE) & PV+BESS displaced savings
    unit_diff_usd_kwh = diesel_bpp_weighted - lcoe_pv_bess
    unit_diff_pct = (
        100.0 * unit_diff_usd_kwh / diesel_bpp_weighted if diesel_bpp_weighted > 0 else 0.0
    )
    displaced_annual_savings_usd = unit_diff_usd_kwh * annual_pv_used_kwh

    # 5b. Blended unit cost difference (Measured PLTD BPP - Hybrid Blended)
    hybrid_savings_usd_kwh = diesel_bpp_weighted - hybrid_blended_cost_usd_kwh
    hybrid_savings_pct = (
        100.0 * hybrid_savings_usd_kwh / diesel_bpp_weighted if diesel_bpp_weighted > 0 else 0.0
    )

    # 5c. Direct annual cost difference against Measured PLTD Baseline (total_mdu_pancabina_kw)
    annual_diff_vs_measured_pltd_usd = diesel_only_annual_cost_usd - hybrid_total_annual_cost_usd

    return {
        "diesel_baseline_source": baseline_source,
        "crf": crf,
        "pv_capex_total_usd": pv_capex_total,
        "bess_capex_total_usd": bess_capex_total,
        "bess_repl_pv_usd": bess_repl_pv,
        "total_capex_usd": total_capex_usd,
        "pv_annual_cost_usd": pv_annual_total_cost,
        "bess_annual_cost_usd": bess_annual_total_cost,
        "total_annual_cost_usd": total_annual_pv_bess_cost_usd,
        "annual_pv_used_mwh": annual_pv_used_kwh / 1000.0,
        "annual_diesel_mwh": annual_diesel_kwh / 1000.0,
        "annual_load_mwh": annual_load_kwh / 1000.0,
        "annual_baseline_pltd_mwh": annual_baseline_pltd_kwh / 1000.0,
        "lcoe_pv_only_usd_kwh": lcoe_pv_only,
        "lcoe_pv_bess_usd_kwh": lcoe_pv_bess,
        "diesel_bpp_avg_usd_kwh": diesel_bpp_avg,
        "diesel_bpp_load_weighted_usd_kwh": diesel_bpp_weighted,
        "diesel_only_annual_cost_usd": diesel_only_annual_cost_usd,
        "hybrid_diesel_annual_cost_usd": hybrid_diesel_annual_cost_usd,
        "hybrid_total_annual_cost_usd": hybrid_total_annual_cost_usd,
        "hybrid_blended_cost_usd_kwh": hybrid_blended_cost_usd_kwh,
        "cost_diff_vs_bpp_usd_kwh": unit_diff_usd_kwh,
        "cost_diff_vs_bpp_pct": unit_diff_pct,
        "displaced_annual_savings_usd": displaced_annual_savings_usd,
        "hybrid_savings_vs_diesel_usd_kwh": hybrid_savings_usd_kwh,
        "hybrid_savings_vs_diesel_pct": hybrid_savings_pct,
        "hybrid_annual_savings_vs_diesel_usd": annual_diff_vs_measured_pltd_usd,
    }


def print_economic_summary(summary: Dict[str, Any]) -> None:
    """Print a clean English summary of LCOE vs. Measured Diesel-Only Baseline (PLTD)."""
    diff_u = summary["cost_diff_vs_bpp_usd_kwh"]
    diff_u_pct = summary["cost_diff_vs_bpp_pct"]
    disp_sav = summary["displaced_annual_savings_usd"]
    hyb_sav_kwh = summary["hybrid_savings_vs_diesel_usd_kwh"]
    hyb_sav_pct = summary["hybrid_savings_vs_diesel_pct"]
    ann_diff = summary["hybrid_annual_savings_vs_diesel_usd"]

    print("\n===== ISOLATED MICROGRID LCOE vs. MEASURED PLTD BASELINE =====")
    print(f"Diesel-Only Baseline Source         : {summary['diesel_baseline_source']}")
    print(f"Total CAPEX (PV + BESS)             : ${summary['total_capex_usd']:,.0f}")
    print(f"  - Solar PV CAPEX                  : ${summary['pv_capex_total_usd']:,.0f}")
    print(f"  - BESS CAPEX                      : ${summary['bess_capex_total_usd']:,.0f} (Repl. PV: ${summary['bess_repl_pv_usd']:,.0f})")
    print(f"Annualized PV + BESS Cost           : ${summary['total_annual_cost_usd']:,.0f} / year")
    print(f"Annual Utilized Solar PV            : {summary['annual_pv_used_mwh']:,.1f} MWh / year")
    print(f"Annual MILP Dispatched PLTD         : {summary['annual_diesel_mwh']:,.1f} MWh / year")
    print(f"Annual Total System Load            : {summary['annual_load_mwh']:,.1f} MWh / year")
    print(f"Annual Measured PLTD Baseline       : {summary['annual_baseline_pltd_mwh']:,.1f} MWh / year")
    print("--------------------------------------------------------------")
    print("UNIT ENERGY COST COMPARISON (USD/kWh):")
    print(f"  1. Solar PV Only LCOE             : ${summary['lcoe_pv_only_usd_kwh']:.4f} / kWh")
    print(f"  2. Solar PV + BESS LCOE           : ${summary['lcoe_pv_bess_usd_kwh']:.4f} / kWh")
    print(f"  3. Hybrid System (Blended)        : ${summary['hybrid_blended_cost_usd_kwh']:.4f} / kWh")
    print(f"  4. Measured PLTD Baseline (BPP)   : ${summary['diesel_bpp_load_weighted_usd_kwh']:.4f} / kWh")
    print("--------------------------------------------------------------")
    print("COST DIFFERENCES & ANNUAL COMPARISON:")
    print(f"  PV+BESS LCOE vs. PLTD BPP (unit)  : ${diff_u:+.4f} / kWh ({diff_u_pct:+.1f}%)")
    print(f"  PV+BESS Displaced Fuel Savings    : ${disp_sav:+,.0f} / year")
    print(f"  Hybrid vs. PLTD BPP (blended)     : ${hyb_sav_kwh:+.4f} / kWh ({hyb_sav_pct:+.1f}%)")
    print("--------------------------------------------------------------")
    print("TOTAL ANNUAL SUPPLY COST:")
    print(f"  - Measured PLTD Baseline          : ${summary['diesel_only_annual_cost_usd']:,.0f} / year ({summary['annual_baseline_pltd_mwh']:,.1f} MWh)")
    print(f"  - Hybrid (MILP PLTD + PV + BESS)  : ${summary['hybrid_total_annual_cost_usd']:,.0f} / year ({summary['annual_load_mwh']:,.1f} MWh)")
    print(f"  - Direct Annual Difference        : ${ann_diff:+,.0f} / year")
    print("==============================================================")
