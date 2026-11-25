"""
MILP Optimization Model (Pyomo) for the Nusa Penida Isolated Microgrid System.
Topology: Suana Solar PV (3.5 MWac) + Suana BESS (3 MW/3 MWh or 2 MW/2.2 MWh)
          + PLN/PIPS Diesel (7 units) + Leased Panca Bina Diesel (19 units),
          all AC-coupled on the 20 kV bus, with no external grid import/export.

Objective Function:
  1. Primary priority: Maximize Solar PV energy utilization (drive curtailment -> 0)
     by leveraging BESS flexibility (energy shifting + bridging/spinning reserve).
  2. Secondary priority (tie-breaker): Minimize diesel operating & start-up costs
     plus a small BESS cycling penalty for realistic dispatch without unnecessary cycling.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import math
import pandas as pd
import pyomo.environ as pyo

from src.data_nusa_penida import (
    ALL_PLTD_UNITS,
    BESS_SCENARIOS,
    PLTS_SUANA_PARAMS,
    SYSTEM_PARAMS,
    DieselUnit,
)


def build_nusa_penida_milp(
    profiles: pd.DataFrame,
    dt_hours: float = 0.5,
    bess_scenario: str = "derated_2pcs",
    pltd_units: Optional[List[DieselUnit]] = None,
    include_ps_load: bool = False,
    allow_bess_reserve: bool = True,
    enforce_zero_curtailment: bool = False,
    fix_mode_tolerance: float = 0.05,
    enforce_mut_mdt: bool = True,
    cyclic_soc: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> pyo.ConcreteModel:
    """
    Build a Pyomo ConcreteModel (MILP) for the Nusa Penida isolated microgrid.

    Parameters
    ----------
    profiles : DataFrame
        Must contain 'load_kw' and 'pv_avail_kw' columns.
    dt_hours : float
        Time step resolution in hours (default 0.5 h = 30 minutes).
    bess_scenario : str
        BESS scenario selection: 'derated_2pcs' (2 MW / 2.2 MWh),
        'full_3pcs' (3 MW / 3 MWh), or 'full_eol' (3 MW / 1.84 MWh).
    pltd_units : List[DieselUnit], optional
        List of diesel generator units; defaults to all units from Tables 1 & 2.
    include_ps_load : bool
        If True, adds 242 kW auxiliary (PS) load to the load profile.
    allow_bess_reserve : bool
        If True, BESS can contribute spinning/bridging reserve to meet the
        700 kW minimum spinning reserve and PV cloud ramp requirements.
    enforce_zero_curtailment : bool
        If True, adds a hard constraint P_curt[t] == 0 (caution: may become
        infeasible if load is too low and BESS is full). If False (recommended),
        curtailment is heavily penalized in the objective function.
    fix_mode_tolerance : float
        Operating band tolerance (±%) for FIX-mode diesel units when ON (default 5%).
        Set to 0.0 for exact constant output P_fix * u[g,t].
    enforce_mut_mdt : bool
        Enable diesel Minimum Up Time (MUT) and Minimum Down Time (MDT) constraints.
    cyclic_soc : bool
        If True, end-of-horizon BESS energy >= initial BESS energy.
    weights : dict, optional
        Objective function weight overrides.
    """
    if bess_scenario not in BESS_SCENARIOS:
        raise ValueError(f"Unknown BESS scenario '{bess_scenario}'. Choose from: {list(BESS_SCENARIOS)}")

    bess = BESS_SCENARIOS[bess_scenario]
    units_raw = pltd_units if pltd_units is not None else ALL_PLTD_UNITS
    # Only available units (DMP > 0 and mode != OFFLINE)
    active_units = [u for u in units_raw if u.dmp_kw > 0 and u.mode != "OFFLINE"]
    unit_map: Dict[str, DieselUnit] = {u.unit_id: u for u in active_units}

    w = {
        "pv_reward":    1000.0,   # High reward per kWh of utilized Solar PV energy
        "curt_penalty": 10000.0,  # Very high penalty per kWh of Solar PV curtailment
        "diesel_scale": 1.0,      # Diesel variable & startup cost scale (tie-breaker)
        "bess_cycle":   0.02,     # Small penalty per kWh of BESS throughput
    }
    if weights:
        w.update(weights)

    n_steps = len(profiles)
    load_series = profiles["load_kw"].astype(float).tolist()
    pv_series = profiles["pv_avail_kw"].astype(float).tolist()
    ps_kw = SYSTEM_PARAMS["ps_load_kw"] if include_ps_load else 0.0

    m = pyo.ConcreteModel(name="Nusa_Penida_Isolated_Microgrid_MILP")

    # =========================================================================
    # 1. SETS
    # =========================================================================
    m.T = pyo.RangeSet(0, n_steps - 1)           # Time interval index
    m.T_soc = pyo.RangeSet(0, n_steps)           # BESS energy state index (0..N)
    m.G = pyo.Set(initialize=[u.unit_id for u in active_units])
    m.G_LS = pyo.Set(within=m.G, initialize=[u.unit_id for u in active_units if u.mode == "LOAD_SHARE"])
    m.G_FIX = pyo.Set(within=m.G, initialize=[u.unit_id for u in active_units if u.mode == "FIX"])

    # =========================================================================
    # 2. PARAMETERS
    # =========================================================================
    m.dt = pyo.Param(initialize=dt_hours)
    m.P_load = pyo.Param(m.T, initialize={t: load_series[t] + ps_kw for t in range(n_steps)})
    m.P_pv_avail = pyo.Param(
        m.T,
        initialize={t: min(pv_series[t], PLTS_SUANA_PARAMS["ac_capacity_kw"]) for t in range(n_steps)},
    )
    m.P_pv_max_ac = pyo.Param(initialize=PLTS_SUANA_PARAMS["ac_capacity_kw"])

    # Diesel generator parameters
    m.DMP = pyo.Param(m.G, initialize={u.unit_id: u.dmp_kw for u in active_units})
    m.P_min = pyo.Param(m.G, initialize={u.unit_id: u.dmp_kw * u.p_min_ratio for u in active_units})
    m.P_fix = pyo.Param(m.G, initialize={u.unit_id: u.dmp_kw * u.fix_ratio for u in active_units})
    m.c_var = pyo.Param(m.G, initialize={u.unit_id: u.var_cost for u in active_units})
    m.c_start = pyo.Param(m.G, initialize={u.unit_id: u.startup_cost for u in active_units})
    m.ramp_step = pyo.Param(
        m.G,
        initialize={u.unit_id: u.ramp_kw_per_s * 3600.0 * dt_hours for u in active_units},
    )

    # BESS parameters
    m.P_bess_max = pyo.Param(initialize=bess["p_max_kw"])
    m.E_bess_nom = pyo.Param(initialize=bess["e_nom_kwh"])
    m.E_bess_min = pyo.Param(initialize=bess["e_nom_kwh"] * bess["soc_min"])
    m.E_bess_max = pyo.Param(initialize=bess["e_nom_kwh"] * bess["soc_max"])
    m.E_bess_init = pyo.Param(initialize=bess["e_nom_kwh"] * bess["soc_init"])
    m.eta_ch = pyo.Param(initialize=bess["eta_ch"])
    m.eta_dis = pyo.Param(initialize=bess["eta_dis"])
    m.tau_bridge = pyo.Param(initialize=bess["bridge_duration_hr"])

    # Isolated system parameters
    m.R_base = pyo.Param(initialize=SYSTEM_PARAMS["min_spin_res_kw"])
    m.alpha_cloud = pyo.Param(initialize=SYSTEM_PARAMS["cloud_ramp_factor"])
    m.R_diesel_min = pyo.Param(initialize=SYSTEM_PARAMS["min_diesel_res_kw"])

    # =========================================================================
    # 3. DECISION VARIABLES
    # =========================================================================
    # Solar PV (AC-coupled 20 kV)
    m.P_pv = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.P_curt = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # BESS (AC-coupled 20 kV)
    m.P_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.P_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.u_ch = pyo.Var(m.T, domain=pyo.Binary)  # 1 = charging, 0 = discharging/idle
    m.E_bess = pyo.Var(m.T_soc, bounds=(pyo.value(m.E_bess_min), pyo.value(m.E_bess_max)))
    m.R_bess = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Diesel Unit Commitment & Dispatch
    m.u_gen = pyo.Var(m.G, m.T, domain=pyo.Binary)      # ON/OFF status
    m.y_start = pyo.Var(m.G, m.T, domain=pyo.Binary)    # Start-up indicator
    m.z_stop = pyo.Var(m.G, m.T, domain=pyo.Binary)     # Shut-down indicator
    m.P_gen = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    m.R_gen = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)

    # =========================================================================
    # 4. CONSTRAINTS
    # =========================================================================

    # 4.1 20 kV AC-Coupled Bus Power Balance (Isolated System: No Export/Import)
    def ac_bus_balance_rule(mod, t):
        return (
            sum(mod.P_gen[g, t] for g in mod.G) + mod.P_pv[t] + mod.P_dis[t]
            == mod.P_load[t] + mod.P_ch[t]
        )
    m.c_ac_balance = pyo.Constraint(m.T, rule=ac_bus_balance_rule)

    # 4.2 Suana Solar PV Generation & Curtailment
    def pv_balance_rule(mod, t):
        return mod.P_pv[t] + mod.P_curt[t] == mod.P_pv_avail[t]
    m.c_pv_balance = pyo.Constraint(m.T, rule=pv_balance_rule)

    if enforce_zero_curtailment:
        m.c_zero_curt = pyo.Constraint(m.T, rule=lambda mod, t: mod.P_curt[t] == 0.0)

    # 4.3 Diesel Operating Limits (LOAD_SHARE vs. FIX)
    def pltd_min_rule(mod, g, t):
        if g in mod.G_LS:
            return mod.P_gen[g, t] >= mod.P_min[g] * mod.u_gen[g, t]
        # FIX mode: operates within tolerance band around fixed setpoint when ON
        p_low = max(pyo.value(mod.P_min[g]), pyo.value(mod.P_fix[g]) * (1.0 - fix_mode_tolerance))
        return mod.P_gen[g, t] >= p_low * mod.u_gen[g, t]
    m.c_pltd_min = pyo.Constraint(m.G, m.T, rule=pltd_min_rule)

    def pltd_max_rule(mod, g, t):
        if g in mod.G_LS:
            return mod.P_gen[g, t] + mod.R_gen[g, t] <= mod.DMP[g] * mod.u_gen[g, t]
        # FIX mode: does not provide spinning reserve (R_gen = 0)
        p_high = min(pyo.value(mod.DMP[g]), pyo.value(mod.P_fix[g]) * (1.0 + fix_mode_tolerance))
        return mod.P_gen[g, t] <= p_high * mod.u_gen[g, t]
    m.c_pltd_max = pyo.Constraint(m.G, m.T, rule=pltd_max_rule)

    def pltd_fix_no_reserve_rule(mod, g, t):
        return mod.R_gen[g, t] == 0.0
    m.c_pltd_fix_no_res = pyo.Constraint(m.G_FIX, m.T, rule=pltd_fix_no_reserve_rule)

    # 4.4 Diesel Start-Up / Shut-Down Logic
    def uc_logic_rule(mod, g, t):
        if t == 0:
            # Free initial condition (no startup penalty for units already ON at t=0)
            return pyo.Constraint.Skip
        return mod.u_gen[g, t] - mod.u_gen[g, t - 1] == mod.y_start[g, t] - mod.z_stop[g, t]
    m.c_uc_logic = pyo.Constraint(m.G, m.T, rule=uc_logic_rule)

    def uc_excl_rule(mod, g, t):
        return mod.y_start[g, t] + mod.z_stop[g, t] <= 1
    m.c_uc_excl = pyo.Constraint(m.G, m.T, rule=uc_excl_rule)

    # 4.5 Diesel Minimum Up Time (MUT) & Minimum Down Time (MDT)
    if enforce_mut_mdt:
        def mut_rule(mod, g, t):
            mut_steps = max(1, int(math.ceil(unit_map[g].min_up_hr / dt_hours)))
            if t == 0:
                return pyo.Constraint.Skip
            end_t = min(n_steps - 1, t + mut_steps - 1)
            return sum(mod.u_gen[g, k] for k in range(t, end_t + 1)) >= (end_t - t + 1) * mod.y_start[g, t]
        m.c_mut = pyo.Constraint(m.G, m.T, rule=mut_rule)

        def mdt_rule(mod, g, t):
            mdt_steps = max(1, int(math.ceil(unit_map[g].min_down_hr / dt_hours)))
            if t == 0:
                return pyo.Constraint.Skip
            end_t = min(n_steps - 1, t + mdt_steps - 1)
            return sum(1 - mod.u_gen[g, k] for k in range(t, end_t + 1)) >= (end_t - t + 1) * mod.z_stop[g, t]
        m.c_mdt = pyo.Constraint(m.G, m.T, rule=mdt_rule)

    # 4.6 Diesel Ramp Rate (±30 kW/s)
    def ramp_up_rule(mod, g, t):
        if t == 0:
            return pyo.Constraint.Skip
        return mod.P_gen[g, t] - mod.P_gen[g, t - 1] <= mod.ramp_step[g] * mod.u_gen[g, t - 1] + mod.DMP[g] * mod.y_start[g, t]
    m.c_ramp_up = pyo.Constraint(m.G, m.T, rule=ramp_up_rule)

    def ramp_down_rule(mod, g, t):
        if t == 0:
            return pyo.Constraint.Skip
        return mod.P_gen[g, t - 1] - mod.P_gen[g, t] <= mod.ramp_step[g] * mod.u_gen[g, t] + mod.DMP[g] * mod.z_stop[g, t]
    m.c_ramp_down = pyo.Constraint(m.G, m.T, rule=ramp_down_rule)

    # 4.7 BESS Operation & Energy Dynamics (AC-Coupled 20 kV)
    m.c_bess_ch_lim = pyo.Constraint(m.T, rule=lambda mod, t: mod.P_ch[t] <= mod.P_bess_max * mod.u_ch[t])
    m.c_bess_dis_lim = pyo.Constraint(m.T, rule=lambda mod, t: mod.P_dis[t] <= mod.P_bess_max * (1 - mod.u_ch[t]))

    m.c_bess_init = pyo.Constraint(expr=m.E_bess[0] == m.E_bess_init)

    def bess_soc_dyn_rule(mod, t):
        return (
            mod.E_bess[t + 1]
            == mod.E_bess[t] + (mod.eta_ch * mod.P_ch[t] - (1.0 / mod.eta_dis) * mod.P_dis[t]) * mod.dt
        )
    m.c_bess_dyn = pyo.Constraint(m.T, rule=bess_soc_dyn_rule)

    if cyclic_soc:
        m.c_bess_cyclic = pyo.Constraint(expr=m.E_bess[n_steps] >= m.E_bess_init)

    # 4.8 BESS Spinning Reserve / Bridging Power Contribution
    if allow_bess_reserve:
        # Power headroom: stopping charge + switching to discharge provides R_bess
        m.c_bess_res_power = pyo.Constraint(
            m.T,
            rule=lambda mod, t: mod.P_dis[t] - mod.P_ch[t] + mod.R_bess[t] <= mod.P_bess_max,
        )
        # Energy headroom: sufficient to sustain bridging for tau_bridge (10 minutes)
        m.c_bess_res_energy = pyo.Constraint(
            m.T,
            rule=lambda mod, t: (mod.R_bess[t] * mod.tau_bridge) / mod.eta_dis <= mod.E_bess[t] - mod.E_bess_min,
        )
    else:
        m.c_bess_res_zero = pyo.Constraint(m.T, rule=lambda mod, t: mod.R_bess[t] == 0.0)

    # 4.9 Isolated System Spinning Reserve Requirement (700 kW Procedure + PV Cloud Ramp)
    def spin_res_base_rule(mod, t):
        return sum(mod.R_gen[g, t] for g in mod.G_LS) + mod.R_bess[t] >= mod.R_base
    m.c_spin_res_base = pyo.Constraint(m.T, rule=spin_res_base_rule)

    def spin_res_cloud_rule(mod, t):
        return sum(mod.R_gen[g, t] for g in mod.G_LS) + mod.R_bess[t] >= mod.alpha_cloud * mod.P_pv[t]
    m.c_spin_res_cloud = pyo.Constraint(m.T, rule=spin_res_cloud_rule)

    if pyo.value(m.R_diesel_min) > 0:
        m.c_spin_res_diesel_floor = pyo.Constraint(
            m.T,
            rule=lambda mod, t: sum(mod.R_gen[g, t] for g in mod.G_LS) >= mod.R_diesel_min,
        )

    # =========================================================================
    # 5. OBJECTIVE FUNCTION
    #    Maximize Solar PV energy utilization & drive curtailment to 0
    # =========================================================================
    def objective_rule(mod):
        pv_energy_term = sum(w["pv_reward"] * mod.P_pv[t] * mod.dt for t in mod.T)
        curt_penalty_term = sum(w["curt_penalty"] * mod.P_curt[t] * mod.dt for t in mod.T)
        diesel_cost_term = w["diesel_scale"] * sum(
            mod.c_var[g] * mod.P_gen[g, t] * mod.dt + mod.c_start[g] * mod.y_start[g, t]
            for g in mod.G
            for t in mod.T
        )
        bess_cycle_term = sum(w["bess_cycle"] * (mod.P_ch[t] + mod.P_dis[t]) * mod.dt for t in mod.T)
        return pv_energy_term - curt_penalty_term - diesel_cost_term - bess_cycle_term

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    return m


def extract_results(model: pyo.ConcreteModel, profiles: pd.DataFrame) -> pd.DataFrame:
    """Extract optimal dispatch results from the Pyomo model into a DataFrame."""
    dt = pyo.value(model.dt)
    pln_ids = [g for g in model.G if g.startswith("MDU#")]
    panca_ids = [g for g in model.G if g.startswith("PB#")]
    e_nom = pyo.value(model.E_bess_nom)

    rows = []
    for t in model.T:
        p_pln = sum(pyo.value(model.P_gen[g, t]) for g in pln_ids)
        p_panca = sum(pyo.value(model.P_gen[g, t]) for g in panca_ids)
        r_diesel = sum(pyo.value(model.R_gen[g, t]) for g in model.G_LS)
        r_bess = pyo.value(model.R_bess[t])
        n_pln_on = sum(int(round(pyo.value(model.u_gen[g, t]))) for g in pln_ids)
        n_panca_on = sum(int(round(pyo.value(model.u_gen[g, t]))) for g in panca_ids)

        rows.append({
            "t": t,
            "hour": profiles["hour"].iloc[t] if "hour" in profiles.columns else t * dt,
            "load_kw": pyo.value(model.P_load[t]),
            "pv_avail_kw": pyo.value(model.P_pv_avail[t]),
            "pv_used_kw": pyo.value(model.P_pv[t]),
            "pv_curt_kw": pyo.value(model.P_curt[t]),
            "bess_ch_kw": pyo.value(model.P_ch[t]),
            "bess_dis_kw": pyo.value(model.P_dis[t]),
            "bess_soc_pct": 100.0 * pyo.value(model.E_bess[t + 1]) / e_nom,
            "pltd_pln_kw": p_pln,
            "pltd_panca_kw": p_panca,
            "pltd_total_kw": p_pln + p_panca,
            "units_pln_on": n_pln_on,
            "units_panca_on": n_panca_on,
            "res_diesel_kw": r_diesel,
            "res_bess_kw": r_bess,
            "res_total_kw": r_diesel + r_bess,
            "grid_price_usd_kwh": (
                float(profiles["grid_price_usd_kwh"].iloc[t])
                if "grid_price_usd_kwh" in profiles.columns
                else 0.125
            ),
        })

    return pd.DataFrame(rows)
