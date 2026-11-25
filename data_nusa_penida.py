"""
Data parameter sistem kelistrikan isolated microgrid Nusa Penida.
Sumber: Laporan Sistem Kelistrikan Nusa Penida (PLN IP UBP Bali / PIPS / PT Panca Bina / CSIRO).
Semua komponen (PLTS, BESS, PLTD) terhubung AC-coupled pada bus 20 kV.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math
import pandas as pd


@dataclass(frozen=True)
class DieselUnit:
    unit_id: str
    owner: str          # "PLN_PIPS" atau "PANCA_BINA"
    dmp_kw: float       # Daya Mampu Pasok (kW) sesuai Tabel 1 & Tabel 2
    mode: str           # "LOAD_SHARE", "FIX", atau "OFFLINE"
    fuel: str           # "MFO" atau "B50"
    p_min_ratio: float  # Rasio beban minimum terhadap DMP saat ON
    fix_ratio: float    # Setpoint beban untuk unit mode FIX (rasio terhadap DMP)
    var_cost: float     # Bobot biaya variabel relatif (tie-breaker merit order)
    startup_cost: float # Bobot biaya start-up relatif (mencegah cycling berlebih)
    min_up_hr: float    # Minimum Up Time (jam)
    min_down_hr: float  # Minimum Down Time (jam)
    ramp_kw_per_s: float = 30.0  # Ramp rate PLTD ±30 kW/detik (Bab 9)


# ---------------------------------------------------------------------------
# Tabel 1: Unit PLTD milik PLN (dioperasikan PIPS) - 7 unit Hyundai 9H21/32
# Data snapshot 30 Oktober 2026 pukul 06.00 WITA (Total DMP = 6.250 kW)
# Catatan: tidak mencakup PS (pemakaian sendiri) 242 kW beban / 400 kW DMP
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
# Tabel 2: Unit PLTD sewa / IPP (PT Panca Bina, skema PJBTL) - 19 unit
# Data snapshot 30 Oktober 2026 pukul 06.00 WITA (Total DMP = 10.433 kW)
# Hanya 6 dari 19 unit punya kapabilitas load sharing (di snapshot: UNIT#2,3,8 aktif LS)
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
# Parameter PLTS Suana (Bab 5)
# ---------------------------------------------------------------------------
PLTS_SUANA_PARAMS: Dict[str, float] = {
    "dc_capacity_kwp": 4200.0,   # 4,2 MWp DC (7.778 x 540 Wp JA Solar)
    "ac_capacity_kw":  3500.0,   # 3,5 MWac (18 x Huawei SUN2000-215KTL-H0)
    "inverter_mva":    3.6,      # ~3,6 MVA
    "bus_kv":          20.0,     # AC-coupled 20 kV (Step-Up 2 x 2.100 kVA 20/0.8 kV)
}


# ---------------------------------------------------------------------------
# Parameter BESS Suana (Bab 6 & Bab 9)
# Tersedia 2 skenario:
#   - "full_3pcs": 3 PCS aktif (desain 3 MW / 3 MWh nominal, usable EoL 1,84 MWh)
#   - "derated_2pcs": kondisi eksisting Nov 2026 (1 PCS offline -> 2 MW / 2,2 MWh)
# ---------------------------------------------------------------------------
BESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "full_3pcs": {
        "p_max_kw":          3000.0,  # 3 MW (3 x 1500 kVA ABB PS1000)
        "e_nom_kwh":         3000.0,  # 3 MWh nominal (3,375 MWh DC nameplate)
        "soc_min":           0.15,    # 15% batas bawah proteksi LFP
        "soc_max":           0.95,    # 95% batas atas
        "soc_init":          0.50,    # 50% SoC awal
        "eta_ch":            0.94,    # Efisiensi charging AC->DC
        "eta_dis":           0.94,    # Efisiensi discharging DC->AC (RTE ~88,4%)
        "bridge_duration_hr": 10.0 / 60.0,  # 10 menit bridging untuk start/prelube genset
    },
    "full_eol": {
        "p_max_kw":          3000.0,
        "e_nom_kwh":         1840.0,  # 1,84 MWh usable energy (End of Life)
        "soc_min":           0.05,
        "soc_max":           0.95,
        "soc_init":          0.50,
        "eta_ch":            0.94,
        "eta_dis":           0.94,
        "bridge_duration_hr": 10.0 / 60.0,
    },
    "derated_2pcs": {
        "p_max_kw":          2000.0,  # 2 MW (1 PCS offline 12 bulan per Nov 2026)
        "e_nom_kwh":         2200.0,  # 2,2 MWh
        "soc_min":           0.15,
        "soc_max":           0.95,
        "soc_init":          0.50,
        "eta_ch":            0.94,
        "eta_dis":           0.94,
        "bridge_duration_hr": 10.0 / 60.0,
    },
}


# ---------------------------------------------------------------------------
# Parameter Operasi Sistem Isolated (Bab 3 & Bab 9)
# ---------------------------------------------------------------------------
SYSTEM_PARAMS: Dict[str, float] = {
    "bus_voltage_kv":       20.0,
    "ps_load_kw":           242.0,  # Pemakaian sendiri PLTD PLN (set 0 jika sudah masuk load profile)
    "min_spin_res_kw":      700.0,  # Prosedur resmi: minimum 700 kW cadangan putar loadshare
    "cloud_ramp_factor":    0.685,  # ±2,4 MW / 3,5 MWac ≈ 68,5% output PV untuk proteksi cloud ramp
    "min_diesel_res_kw":    0.0,    # Set > 0 jika tetap mensyaratkan lantai cadangan putar murni PLTD
}


def generate_sample_profiles(
    n_days: int = 1,
    dt_hours: float = 0.5,
    peak_load_kw: float = 15780.0,
    pv_peak_kw: float = 3500.0,
) -> pd.DataFrame:
    """
    Membuat profil beban & generasi PLTS sintetis (resolusi 30 menit) yang
    dikalibrasi ke karakteristik laporan Nusa Penida (April 2026):
      - Beban puncak malam ~15,78 MW pukul 19:30 WITA
      - Lembah dini hari ~9,2 MW pukul 03:30-04:30 WITA
      - Beban siang hari ~11,5 - 12,8 MW
      - PLTS Suana berproduksi 06:00 - 18:00, puncak 3,5 MWac sekitar 12:00 WITA.
    """
    steps_per_day = int(round(24.0 / dt_hours))
    rows = []
    for d in range(n_days):
        for s in range(steps_per_day):
            hour = s * dt_hours
            # Pola beban harian Nusa Penida (pariwisata + domestik)
            # Basis + puncak malam (19:30) + bahu siang (13:00) - lembah dini hari (04:00)
            evening_peak = math.exp(-0.5 * ((hour - 19.5) / 2.4) ** 2)
            day_shoulder = math.exp(-0.5 * ((hour - 13.0) / 3.2) ** 2)
            night_valley = math.exp(-0.5 * ((hour - 4.0) / 2.2) ** 2)
            load_norm = 0.68 + 0.32 * evening_peak + 0.11 * day_shoulder - 0.12 * night_valley
            load_kw = round(peak_load_kw * load_norm / 1.00, 1)

            # Pola generasi PLTS Suana (06:00 - 18:00 WITA)
            if 6.0 <= hour <= 18.0:
                pv_raw = math.sin(math.pi * (hour - 6.0) / 12.0) ** 1.25
                # Sedikit variasi awan siang pada pukul 13:30
                cloud_dip = 0.85 if abs(hour - 13.5) < 0.6 else 1.0
                pv_kw = round(min(pv_peak_kw, pv_peak_kw * pv_raw * cloud_dip), 1)
            else:
                pv_kw = 0.0

            rows.append({
                "t": d * steps_per_day + s,
                "hour": hour,
                "load_kw": load_kw,
                "pv_avail_kw": pv_kw,
            })

    return pd.DataFrame(rows)


def load_profiles_from_csv(
    csv_path: str,
    load_col: str = "load_kw",
    pv_col: str = "pv_avail_kw",
) -> pd.DataFrame:
    """
    Memuat data profil beban dan generasi PLTS dari file CSV pengguna.
    Kolom wajib: `load_col` (kW) dan `pv_col` (kW).
    """
    df = pd.read_csv(csv_path)
    if load_col not in df.columns or pv_col not in df.columns:
        raise ValueError(
            f"CSV harus memiliki kolom '{load_col}' dan '{pv_col}'. "
            f"Kolom ditemukan: {list(df.columns)}"
        )
    out = df[[load_col, pv_col]].copy()
    out.columns = ["load_kw", "pv_avail_kw"]
    out["t"] = range(len(out))
    return out
