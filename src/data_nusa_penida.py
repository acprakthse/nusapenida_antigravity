"""
System parameters for the Nusa Penida Isolated Microgrid.
Source: Nusa Penida Power System Report (PLN IP UBP Bali / PIPS / PT Panca Bina / CSIRO).
All components (Solar PV, BESS, Diesel Generators) are AC-coupled on the 20 kV bus.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math
import os
import pandas as pd


@dataclass(frozen=True)
class DieselUnit:
    unit_id: str
    owner: str          # "PLN_PIPS" or "PANCA_BINA"
    dmp_kw: float       # Net Dependable Capacity / DMP (kW) per Tables 1 & 2
    mode: str           # "LOAD_SHARE", "FIX", or "OFFLINE"
    fuel: str           # "MFO" or "B50"
    p_min_ratio: float  # Minimum loading ratio relative to DMP when ON
    fix_ratio: float    # Load setpoint ratio relative to DMP for FIX-mode units
    var_cost: float     # Relative variable cost weight (merit-order tie-breaker)
    startup_cost: float # Relative start-up cost weight (prevents excessive cycling)
    min_up_hr: float    # Minimum Up Time (hours)
    min_down_hr: float  # Minimum Down Time (hours)
    ramp_kw_per_s: float = 30.0  # Diesel ramp rate ±30 kW/s (Chapter 9)


# ---------------------------------------------------------------------------
# Table 1: PLN-owned Diesel Units (operated by PIPS) - 7 Hyundai 9H21/32 units
# Snapshot data: 30 October 2026 at 06:00 WITA (Total DMP = 6,250 kW)
# Note: excludes auxiliary (PS) load of 242 kW / 400 kW DMP
# ---------------------------------------------------------------------------
PLTD_PLN_UNITS: List[DieselUnit] = [
    DieselUnit("MDU#1", "PLN_PIPS", 1400.0, "FIX",        "MFO", 0.35, 0.70, 1.00, 50.0, 2.0, 1.0),
    DieselUnit("MDU#2", "PLN_PIPS", 1200.0, "FIX",        "MFO", 0.35, 0.70, 1.00, 50.0, 2.0, 1.0),
    DieselUnit("MDU#3", "PLN_PIPS", 1300.0, "FIX",        "MFO", 0.35, 0.70, 1.00, 50.0, 2.0, 1.0),
    DieselUnit("MDU#4", "PLN_PIPS", 1000.0, "LOAD_SHARE", "MFO", 0.35, 0.80, 1.02, 50.0, 2.0, 1.0),
    DieselUnit("MDU#5", "PLN_PIPS",  750.0, "FIX",        "MFO", 0.35, 0.70, 1.00, 50.0, 2.0, 1.0),
    DieselUnit("MDU#6", "PLN_PIPS", 1000.0, "LOAD_SHARE", "MFO", 0.35, 0.80, 1.02, 50.0, 2.0, 1.0),
    DieselUnit("MDU#7", "PLN_PIPS",    0.0, "OFFLINE",    "MFO", 0.35, 0.70, 1.00, 50.0, 2.0, 1.0),
]

# ---------------------------------------------------------------------------
# Table 2: Leased / IPP Diesel Units (PT Panca Bina, PJBTL scheme) - 19 units
# Snapshot data: 30 October 2026 at 06:00 WITA (Total DMP = 10,433 kW)
# Only 6 of 19 units support load sharing (snapshot active LS: UNIT#2, 3, 8)
# ---------------------------------------------------------------------------
PLTD_PANCA_UNITS: List[DieselUnit] = [
    DieselUnit("PB#1",  "PANCA_BINA",   0.0, "OFFLINE",    "B50", 0.40, 0.90, 1.15, 40.0, 2.0, 1.0),
    DieselUnit("PB#2",  "PANCA_BINA", 850.0, "LOAD_SHARE", "B50", 0.40, 0.90, 1.15, 40.0, 2.0, 1.0),
    DieselUnit("PB#3",  "PANCA_BINA", 850.0, "LOAD_SHARE", "B50", 0.40, 0.90, 1.15, 40.0, 2.0, 1.0),
    DieselUnit("PB#4",  "PANCA_BINA", 712.0, "FIX",        "B50", 0.40, 0.90, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#5",  "PANCA_BINA", 592.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#6",  "PANCA_BINA", 285.0, "FIX",        "B50", 0.40, 0.90, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#7",  "PANCA_BINA",   0.0, "OFFLINE",    "B50", 0.40, 0.90, 1.15, 40.0, 2.0, 1.0),
    DieselUnit("PB#8",  "PANCA_BINA", 850.0, "LOAD_SHARE", "B50", 0.40, 0.90, 1.15, 40.0, 2.0, 1.0),
    DieselUnit("PB#9",  "PANCA_BINA", 508.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#10", "PANCA_BINA", 600.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#11", "PANCA_BINA", 622.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#12", "PANCA_BINA", 518.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#13", "PANCA_BINA", 615.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#14", "PANCA_BINA", 708.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#15", "PANCA_BINA", 678.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#16", "PANCA_BINA",   0.0, "OFFLINE",    "B50", 0.40, 0.90, 1.15, 40.0, 2.0, 1.0),
    DieselUnit("PB#17", "PANCA_BINA", 825.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#18", "PANCA_BINA", 304.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
    DieselUnit("PB#19", "PANCA_BINA", 916.0, "FIX",        "B50", 0.40, 0.95, 1.12, 40.0, 2.0, 1.0),
]

ALL_PLTD_UNITS: List[DieselUnit] = PLTD_PLN_UNITS + PLTD_PANCA_UNITS


# ---------------------------------------------------------------------------
# Suana Solar PV Parameters (Chapter 5)
# ---------------------------------------------------------------------------
PLTS_SUANA_PARAMS: Dict[str, float] = {
    "dc_capacity_kwp": 4200.0,   # 4.2 MWp DC (7,778 x 540 Wp JA Solar)
    "ac_capacity_kw":  3500.0,   # 3.5 MWac (18 x Huawei SUN2000-215KTL-H0)
    "inverter_mva":    3.6,      # ~3.6 MVA
    "bus_kv":          20.0,     # AC-coupled 20 kV (Step-Up 2 x 2,100 kVA 20/0.8 kV)
}


# ---------------------------------------------------------------------------
# Suana BESS Parameters (Chapters 6 & 9)
# Available scenarios:
#   - "full_3pcs": 3 active PCS (3 MW / 3 MWh nominal design, 1.84 MWh usable EoL)
#   - "derated_2pcs": existing Nov 2026 condition (1 PCS offline -> 2 MW / 2.2 MWh)
# ---------------------------------------------------------------------------
BESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "full_3pcs": {
        "p_max_kw":          3000.0,  # 3 MW (3 x 1500 kVA ABB PS1000)
        "e_nom_kwh":         3000.0,  # 3 MWh nominal (3.375 MWh DC nameplate)
        "soc_min":           0.15,    # 15% LFP lower protection limit
        "soc_max":           0.95,    # 95% upper limit
        "soc_init":          0.50,    # 50% initial SoC
        "eta_ch":            0.94,    # AC->DC charging efficiency
        "eta_dis":           0.94,    # DC->AC discharging efficiency (RTE ~88.4%)
        "bridge_duration_hr": 10.0 / 60.0,  # 10-min bridging for genset start/prelube
    },
    "full_eol": {
        "p_max_kw":          3000.0,
        "e_nom_kwh":         1840.0,  # 1.84 MWh usable energy (End of Life)
        "soc_min":           0.05,
        "soc_max":           0.95,
        "soc_init":          0.50,
        "eta_ch":            0.94,
        "eta_dis":           0.94,
        "bridge_duration_hr": 10.0 / 60.0,
    },
    "derated_2pcs": {
        "p_max_kw":          2000.0,  # 2 MW (1 PCS offline 12 months as of Nov 2026)
        "e_nom_kwh":         2200.0,  # 2.2 MWh
        "soc_min":           0.15,
        "soc_max":           0.95,
        "soc_init":          0.50,
        "eta_ch":            0.94,
        "eta_dis":           0.94,
        "bridge_duration_hr": 10.0 / 60.0,
    },
}


# ---------------------------------------------------------------------------
# Isolated System Operating Parameters (Chapters 3 & 9)
# ---------------------------------------------------------------------------
SYSTEM_PARAMS: Dict[str, float] = {
    "bus_voltage_kv":       20.0,
    "ps_load_kw":           242.0,  # PLN Diesel auxiliary load (set 0 if already in load profile)
    "min_spin_res_kw":      700.0,  # Operating procedure: minimum 700 kW load-share spinning reserve
    "cloud_ramp_factor":    0.685,  # ±2.4 MW / 3.5 MWac ≈ 68.5% PV output for cloud ramp protection
    "min_diesel_res_kw":    0.0,    # Set > 0 to enforce a pure diesel spinning reserve floor
}


def generate_sample_profiles(
    n_days: int = 1,
    dt_hours: float = 0.5,
    peak_load_kw: float = 15780.0,
    pv_peak_kw: float = 3500.0,
) -> pd.DataFrame:
    """
    Generate synthetic load, Solar PV generation, and grid electricity price profiles
    (30-min resolution) calibrated to Nusa Penida report characteristics (April 2026):
      - Evening peak load ~15.78 MW at 19:30 WITA
      - Early morning valley ~9.2 MW at 03:30–04:30 WITA
      - Daytime load ~11.5–12.8 MW
      - Suana Solar PV generation 06:00–18:00, peaking at 3.5 MWac around 12:00 WITA
      - Time-of-Use (TOU) grid import tariff: $0.11/kWh off-peak, $0.13/kWh daytime,
        $0.17/kWh evening peak (18:00–22:00 WITA).
    """
    steps_per_day = int(round(24.0 / dt_hours))
    rows = []
    for d in range(n_days):
        for s in range(steps_per_day):
            hour_of_day = s * dt_hours
            elapsed_hour = (d * steps_per_day + s) * dt_hours
            # Daily Nusa Penida load pattern (tourism + residential)
            # Base + evening peak (19:30) + midday shoulder (13:00) - night valley (04:00)
            evening_peak = math.exp(-0.5 * ((hour_of_day - 19.5) / 2.4) ** 2)
            day_shoulder = math.exp(-0.5 * ((hour_of_day - 13.0) / 3.2) ** 2)
            night_valley = math.exp(-0.5 * ((hour_of_day - 4.0) / 2.2) ** 2)
            load_norm = 0.68 + 0.32 * evening_peak + 0.11 * day_shoulder - 0.12 * night_valley
            load_kw = round(peak_load_kw * load_norm / 1.00, 1)

            # Suana Solar PV generation pattern (06:00 - 18:00 WITA)
            if 6.0 <= hour_of_day <= 18.0:
                pv_raw = math.sin(math.pi * (hour_of_day - 6.0) / 12.0) ** 1.25
                # Slight midday cloud dip around 13:30
                cloud_dip = 0.85 if abs(hour_of_day - 13.5) < 0.6 else 1.0
                pv_kw = round(min(pv_peak_kw, pv_peak_kw * pv_raw * cloud_dip), 1)
            else:
                pv_kw = 0.0

            # Sample Time-of-Use (TOU) grid electricity import price (USD/kWh)
            if 18.0 <= hour_of_day < 22.0:
                grid_price = 0.170  # Evening peak tariff
            elif 8.0 <= hour_of_day < 18.0:
                grid_price = 0.130  # Daytime shoulder tariff
            else:
                grid_price = 0.110  # Night off-peak tariff

            rows.append({
                "t": d * steps_per_day + s,
                "hour": elapsed_hour,
                "load_kw": load_kw,
                "pv_avail_kw": pv_kw,
                "grid_price_usd_kwh": grid_price,
            })

    return pd.DataFrame(rows)


def export_sample_profiles_csv(
    csv_path: str = "input/input_profiles.csv",
    n_days: int = 1,
    dt_hours: float = 0.5,
) -> pd.DataFrame:
    """Export a clean CSV template for load, PV generation, and electricity price profiles."""
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df = generate_sample_profiles(n_days=n_days, dt_hours=dt_hours)
    df[["hour", "load_kw", "pv_avail_kw", "grid_price_usd_kwh"]].to_csv(csv_path, index=False)
    return df


def load_profiles_from_csv(
    csv_path: str,
    load_col: str = "load_kw",
    pv_col: str = "pv_avail_kw",
    price_col: str = "grid_price_usd_kwh",
    measured_pltd_col: str = "total_mdu_pancabina_kw",
    default_grid_price: float = 0.2488,
    dt_hours: float = 0.5,
) -> pd.DataFrame:
    """
    Load user-provided load, Solar PV generation, electricity price (BPP PLTD),
    and optional measured PLTD baseline (`total_mdu_pancabina_kw`) from a CSV file.
    Required columns: `load_col` (kW) and `pv_col` (kW).
    Optional columns: `hour` (numeric hours or datetime string), `price_col` (USD/kWh),
    and `measured_pltd_col` (kW, real measured PLTD generation).
    Negative PV values (nighttime inverter standby losses) are clipped to 0.0 kW.
    """
    df = pd.read_csv(csv_path)
    if load_col not in df.columns or pv_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{load_col}' and '{pv_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "t": range(len(df)),
        "load_kw": pd.to_numeric(df[load_col], errors="coerce").fillna(0.0).clip(lower=0.0),
        "pv_avail_kw": pd.to_numeric(df[pv_col], errors="coerce").fillna(0.0).clip(lower=0.0),
    })

    if "hour" in df.columns:
        numeric_hr = pd.to_numeric(df["hour"], errors="coerce")
        if numeric_hr.notna().all():
            out["hour"] = numeric_hr.astype(float)
        else:
            # Handle datetime strings such as "7/2/25 0:30"
            out["timestamp"] = df["hour"].astype(str)
            out["hour"] = [i * dt_hours for i in range(len(df))]
    else:
        out["hour"] = [i * dt_hours for i in range(len(df))]

    if price_col in df.columns:
        out["grid_price_usd_kwh"] = (
            pd.to_numeric(df[price_col], errors="coerce").fillna(default_grid_price).astype(float)
        )
    else:
        out["grid_price_usd_kwh"] = float(default_grid_price)

    if measured_pltd_col in df.columns:
        out["total_mdu_pancabina_kw"] = (
            pd.to_numeric(df[measured_pltd_col], errors="coerce")
            .ffill()
            .bfill()
            .fillna(0.0)
            .clip(lower=0.0)
            .astype(float)
        )

    return out
