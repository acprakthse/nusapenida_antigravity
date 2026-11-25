"""
Economic analysis module for the Nusa Penida Microgrid (PV + BESS + Diesel).
Computes Levelized Cost of Energy (LCOE) from simple CAPEX & OPEX inputs
and compares PV+BESS against both:
  1. Grid electricity import price (USD/kWh), and
  2. Diesel-only (PLTD-only) supply baseline (USD/kWh and annual system cost).
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd


DEFAULT_ECONOMICS: Dict[str, float] = {
    "pv_capacity_kwp": 4200.0,            # Suana Solar PV DC capacity (kWp)
    "pv_capex_usd_per_kwp": 850.0,        # Solar PV CAPEX (USD/kWp)
    "pv_opex_usd_per_kwp_yr": 15.0,       # Solar PV fixed O&M (USD/kWp-year)
    "bess_power_kw": 2000.0,              # BESS rated power (kW)
    "bess_energy_kwh": 2200.0,            # BESS nominal energy capacity (kWh)
    "bess_capex_usd_per_kwh": 320.0,      # BESS CAPEX (USD/kWh)
    "bess_opex_usd_per_kwh_yr": 8.0,      # BESS fixed O&M (USD/kWh-year)
    "diesel_pln_gen_cost_usd_kwh": 0.185, # PLN Diesel (MFO) generation cost: fuel + O&M (USD/kWh)
    "diesel_panca_gen_cost_usd_kwh": 0.215, # Leased Diesel (Panca Bina B50) generation cost (USD/kWh)
    "discount_rate": 0.08,                # Discount rate / WACC (fraction, e.g. 0.08 = 8%)
    "project_lifetime_yr": 20.0,          # Project lifetime (years)
    "bess_replacement_yr": 10.0,          # BESS battery replacement year (0 = no replacement)
    "bess_replacement_cost_frac": 0.60,   # Replacement cost as fraction of initial BESS CAPEX
}

ECONOMICS_METADATA = [
    ("pv_capacity_kwp",               "kWp",        "Solar PV installed DC capacity"),
    ("pv_capex_usd_per_kwp",          "USD/kWp",    "Solar PV capital expenditure (CAPEX)"),
    ("pv_opex_usd_per_kwp_yr",        "USD/kWp-yr", "Solar PV annual O&M cost (OPEX)"),
    ("bess_power_kw",                 "kW",         "BESS inverter power capacity"),
    ("bess_energy_kwh",               "kWh",        "BESS nominal energy capacity"),
    ("bess_capex_usd_per_kwh",        "USD/kWh",    "BESS capital expenditure (CAPEX)"),
    ("bess_opex_usd_per_kwh_yr",      "USD/kWh-yr", "BESS annual O&M cost (OPEX)"),
    ("diesel_pln_gen_cost_usd_kwh",   "USD/kWh",    "PLN Diesel (MFO) generation cost (fuel + O&M)"),
    ("diesel_panca_gen_cost_usd_kwh", "USD/kWh",    "Leased Diesel (Panca Bina B50) generation cost (fuel + lease + O&M)"),
    ("discount_rate",                 "fraction",   "Discount rate / WACC (e.g. 0.08 = 8%)"),
    ("project_lifetime_yr",           "years",      "Project economic lifetime"),
    ("bess_replacement_yr",           "years",      "BESS replacement year (0 to disable)"),
    ("bess_replacement_cost_frac",    "fraction",   "BESS replacement cost fraction of initial CAPEX"),
]


def save_economics_template_csv(csv_path: str = "input_economics.csv", params: Dict[str, float] | None = None) -> None:
    """Save a clean, editable CAPEX/OPEX & Diesel cost economics template CSV."""
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
    """Load CAPEX/OPEX and diesel cost parameters from a key-value CSV file."""
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
    default_grid_price: float = 0.125,
) -> Dict[str, Any]:
    """
    Compute PV-only LCOE, combined PV+BESS LCOE, hybrid blended system cost,
    grid import benchmark, and Diesel-only (PLTD-only) baseline comparison.
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

    # 2. Annualized Energy Output from Dispatch Horizon
    horizon_hours = len(df_dispatch) * dt_hours
    horizon_days = max(horizon_hours / 24.0, 1e-6)
    annual_factor = 365.0 / horizon_days

    load_arr = df_dispatch["load_kw"].to_numpy(dtype=float)
    pv_arr = df_dispatch["pv_used_kw"].to_numpy(dtype=float)
    pln_arr = df_dispatch["pltd_pln_kw"].to_numpy(dtype=float)
    panca_arr = df_dispatch["pltd_panca_kw"].to_numpy(dtype=float)

    pv_used_kwh = float(np.sum(pv_arr) * dt_hours)
    load_kwh = float(np.sum(load_arr) * dt_hours)
    pln_kwh = float(np.sum(pln_arr) * dt_hours)
    panca_kwh = float(np.sum(panca_arr) * dt_hours)

    annual_pv_used_kwh = pv_used_kwh * annual_factor
    annual_load_kwh = load_kwh * annual_factor
    annual_pln_kwh = pln_kwh * annual_factor
    annual_panca_kwh = panca_kwh * annual_factor

    # 3. LCOE (USD/kWh)
    lcoe_pv_only = pv_annual_total_cost / annual_pv_used_kwh if annual_pv_used_kwh > 0 else float("nan")
    lcoe_pv_bess = total_annual_pv_bess_cost_usd / annual_pv_used_kwh if annual_pv_used_kwh > 0 else float("nan")

    # 4. Grid Electricity Import Price Benchmark (USD/kWh)
    if "grid_price_usd_kwh" in df_dispatch.columns:
        prices = df_dispatch["grid_price_usd_kwh"].to_numpy(dtype=float)
    else:
        prices = np.full(len(df_dispatch), default_grid_price, dtype=float)

    grid_price_avg = float(np.mean(prices))
    grid_price_load_weighted = (
        float(np.sum(load_arr * prices) / np.sum(load_arr)) if np.sum(load_arr) > 0 else grid_price_avg
    )
    grid_only_annual_cost_usd = grid_price_load_weighted * annual_load_kwh

    # 5. Diesel-Only (PLTD-Only) Baseline vs. Hybrid Microgrid (PV + BESS + PLTD)
    c_pln = float(cfg["diesel_pln_gen_cost_usd_kwh"])
    c_panca = float(cfg["diesel_panca_gen_cost_usd_kwh"])

    # Under Diesel-Only baseline (no PV/BESS), merit order dispatches cheaper PLN Diesel
    # up to its observed peak available output, with Leased Panca Bina covering the remainder.
    pln_max_kw = max(float(np.max(pln_arr)), 5417.75)
    base_pln_arr = np.minimum(load_arr, pln_max_kw)
    base_panca_arr = np.maximum(0.0, load_arr - base_pln_arr)

    annual_base_pln_kwh = float(np.sum(base_pln_arr) * dt_hours) * annual_factor
    annual_base_panca_kwh = float(np.sum(base_panca_arr) * dt_hours) * annual_factor

    diesel_only_annual_cost_usd = annual_base_pln_kwh * c_pln + annual_base_panca_kwh * c_panca
    diesel_only_cost_usd_kwh = (
        diesel_only_annual_cost_usd / annual_load_kwh if annual_load_kwh > 0 else float("nan")
    )

    # Actual Hybrid Microgrid Total Cost (Dispatched PLTD + Annualized PV + BESS)
    hybrid_diesel_annual_cost_usd = annual_pln_kwh * c_pln + annual_panca_kwh * c_panca
    hybrid_total_annual_cost_usd = hybrid_diesel_annual_cost_usd + total_annual_pv_bess_cost_usd
    hybrid_blended_cost_usd_kwh = (
        hybrid_total_annual_cost_usd / annual_load_kwh if annual_load_kwh > 0 else float("nan")
    )

    # 6. Cost Differences (Positive = Savings vs. Benchmark)
    # 6a. PV+BESS LCOE vs. Grid Import Price
    diff_vs_grid_usd_kwh = grid_price_load_weighted - lcoe_pv_bess
    diff_vs_grid_pct = (
        100.0 * diff_vs_grid_usd_kwh / grid_price_load_weighted if grid_price_load_weighted > 0 else 0.0
    )
    annual_savings_vs_grid_usd = diff_vs_grid_usd_kwh * annual_pv_used_kwh

    # 6b. PV+BESS LCOE vs. Diesel-Only Cost (per-kWh displaced)
    diff_vs_diesel_usd_kwh = diesel_only_cost_usd_kwh - lcoe_pv_bess
    diff_vs_diesel_pct = (
        100.0 * diff_vs_diesel_usd_kwh / diesel_only_cost_usd_kwh if diesel_only_cost_usd_kwh > 0 else 0.0
    )

    # 6c. Full-System Hybrid (PV+BESS+PLTD) vs. Diesel-Only (PLTD Only)
    hybrid_savings_vs_diesel_usd_kwh = diesel_only_cost_usd_kwh - hybrid_blended_cost_usd_kwh
    hybrid_savings_vs_diesel_pct = (
        100.0 * hybrid_savings_vs_diesel_usd_kwh / diesel_only_cost_usd_kwh if diesel_only_cost_usd_kwh > 0 else 0.0
    )
    hybrid_annual_savings_vs_diesel_usd = diesel_only_annual_cost_usd - hybrid_total_annual_cost_usd

    return {
        "crf": crf,
        "pv_capex_total_usd": pv_capex_total,
        "bess_capex_total_usd": bess_capex_total,
        "bess_repl_pv_usd": bess_repl_pv,
        "total_capex_usd": total_capex_usd,
        "pv_annual_cost_usd": pv_annual_total_cost,
        "bess_annual_cost_usd": bess_annual_total_cost,
        "total_annual_cost_usd": total_annual_pv_bess_cost_usd,
        "annual_pv_used_mwh": annual_pv_used_kwh / 1000.0,
        "annual_load_mwh": annual_load_kwh / 1000.0,
        "lcoe_pv_only_usd_kwh": lcoe_pv_only,
        "lcoe_pv_bess_usd_kwh": lcoe_pv_bess,
        "grid_price_avg_usd_kwh": grid_price_avg,
        "grid_price_load_weighted_usd_kwh": grid_price_load_weighted,
        "grid_only_annual_cost_usd": grid_only_annual_cost_usd,
        "diesel_only_cost_usd_kwh": diesel_only_cost_usd_kwh,
        "diesel_only_annual_cost_usd": diesel_only_annual_cost_usd,
        "hybrid_blended_cost_usd_kwh": hybrid_blended_cost_usd_kwh,
        "hybrid_total_annual_cost_usd": hybrid_total_annual_cost_usd,
        "cost_diff_usd_kwh": diff_vs_grid_usd_kwh,
        "cost_diff_pct": diff_vs_grid_pct,
        "annual_cost_diff_usd": annual_savings_vs_grid_usd,
        "cost_diff_vs_diesel_usd_kwh": diff_vs_diesel_usd_kwh,
        "cost_diff_vs_diesel_pct": diff_vs_diesel_pct,
        "hybrid_savings_vs_diesel_usd_kwh": hybrid_savings_vs_diesel_usd_kwh,
        "hybrid_savings_vs_diesel_pct": hybrid_savings_vs_diesel_pct,
        "hybrid_annual_savings_vs_diesel_usd": hybrid_annual_savings_vs_diesel_usd,
    }


def print_economic_summary(summary: Dict[str, Any]) -> None:
    """Print a clean English summary table of LCOE, Grid Import, and Diesel-Only comparison."""
    diff_g = summary["cost_diff_usd_kwh"]
    diff_g_pct = summary["cost_diff_pct"]
    ann_g = summary["annual_cost_diff_usd"]
    diff_d = summary["cost_diff_vs_diesel_usd_kwh"]
    diff_d_pct = summary["cost_diff_vs_diesel_pct"]
    hyb_sav_kwh = summary["hybrid_savings_vs_diesel_usd_kwh"]
    hyb_sav_pct = summary["hybrid_savings_vs_diesel_pct"]
    hyb_sav_yr = summary["hybrid_annual_savings_vs_diesel_usd"]

    print("\n========= LCOE, GRID IMPORT & DIESEL-ONLY COMPARISON =========")
    print(f"Total CAPEX (PV + BESS)             : ${summary['total_capex_usd']:,.0f}")
    print(f"  - Solar PV CAPEX                  : ${summary['pv_capex_total_usd']:,.0f}")
    print(f"  - BESS CAPEX                      : ${summary['bess_capex_total_usd']:,.0f} (Repl. PV: ${summary['bess_repl_pv_usd']:,.0f})")
    print(f"Annualized PV + BESS Cost           : ${summary['total_annual_cost_usd']:,.0f} / year")
    print(f"Annual Utilized Solar PV            : {summary['annual_pv_used_mwh']:,.1f} MWh / year")
    print(f"Annual Total System Load            : {summary['annual_load_mwh']:,.1f} MWh / year")
    print("--------------------------------------------------------------")
    print("UNIT ENERGY COST BENCHMARKS (USD/kWh):")
    print(f"  1. Solar PV Only LCOE             : ${summary['lcoe_pv_only_usd_kwh']:.4f} / kWh")
    print(f"  2. Solar PV + BESS LCOE           : ${summary['lcoe_pv_bess_usd_kwh']:.4f} / kWh")
    print(f"  3. Grid Import Price (Load-Wtd)   : ${summary['grid_price_load_weighted_usd_kwh']:.4f} / kWh")
    print(f"  4. Hybrid Microgrid (Blended)     : ${summary['hybrid_blended_cost_usd_kwh']:.4f} / kWh")
    print(f"  5. Diesel-Only Supply (PLTD Only) : ${summary['diesel_only_cost_usd_kwh']:.4f} / kWh")
    print("--------------------------------------------------------------")
    print("COST DIFFERENCES & ANNUAL SAVINGS:")
    print(f"  PV+BESS vs. Grid Import           : ${diff_g:+.4f} / kWh ({diff_g_pct:+.1f}%) | ${ann_g:+,.0f} / yr")
    print(f"  PV+BESS vs. Diesel-Only (unit)    : ${diff_d:+.4f} / kWh ({diff_d_pct:+.1f}%)")
    print(f"  Hybrid vs. Diesel-Only (system)   : ${hyb_sav_kwh:+.4f} / kWh ({hyb_sav_pct:+.1f}%) | ${hyb_sav_yr:+,.0f} / yr")
    print("--------------------------------------------------------------")
    print("TOTAL ANNUAL SYSTEM SUPPLY COST (100% OF LOAD):")
    print(f"  - Diesel-Only (PLTD Only)         : ${summary['diesel_only_annual_cost_usd']:,.0f} / year")
    print(f"  - Hybrid (PV + BESS + PLTD)       : ${summary['hybrid_total_annual_cost_usd']:,.0f} / year")
    print(f"  - Grid Import Only (Reference)    : ${summary['grid_only_annual_cost_usd']:,.0f} / year")
    print("==============================================================")
