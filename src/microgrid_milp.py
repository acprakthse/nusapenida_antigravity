"""
MILP Optimization Model (Pyomo) for the Nusa Penida Isolated Microgrid System.
Supports both single-shot MILP and Rolling Horizon MILP (default 48-hour lookahead).

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
from typing import Any, Dict, List, Optional
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
    initial_state: Optional[Dict[str, Any]] = None,
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
        If True, adds a hard constraint P_curt[t] == 0.
    fix_mode_tolerance : float
        Operating band tolerance (±%) for FIX-mode diesel units when ON (default 5%).
    enforce_mut_mdt : bool
        Enable diesel Minimum Up Time (MUT) and Minimum Down Time (MDT) constraints.
    cyclic_soc : bool
        If True, end-of-horizon BESS energy >= initial BESS energy.
    weights : dict, optional
        Objective function weight overrides.
    initial_state : dict, optional
        Carry-over state from a previous rolling-horizon window containing:
          - 'e_bess_kwh': float
          - 'u_init': Dict[str, int]
          - 'p_init': Dict[str, float]
          - 'up_run_init': Dict[str, int]
          - 'down_run_init': Dict[str, int]
    """
    if bess_scenario not in BESS_SCENARIOS:
        raise ValueError(f"Unknown BESS scenario '{bess_scenario}'. Choose from: {list(BESS_SCENARIOS)}")

    bess = BESS_SCENARIOS[bess_scenario]
    units_raw = pltd_units if pltd_units is not None else ALL_PLTD_UNITS
    active_units = [u for u in units_raw if u.dmp_kw > 0 and u.mode != "OFFLINE"]
    unit_map: Dict[str, DieselUnit] = {u.unit_id: u for u in active_units}

    w = {
        "pv_reward":        10.0,    # High reward (USD/kWh) to prioritize Solar PV utilization
        "curt_penalty":     100.0,   # High penalty (USD/kWh) to drive PV curtailment -> 0
        "diesel_scale":     1.0,     # Scales real Diesel BPP (USD/kWh from grid_price_usd_kwh)
        "startup_usd":      1.0,     # Scales diesel start-up cost (USD/start)
        "bess_cycle":       0.005,   # Small cycling cost (USD/kWh throughput) to avoid idle cycling
        "unserved_penalty": 1000.0,  # Very high penalty (USD/kWh) for unserved load safety slack
    }
    if weights:
        w.update(weights)

    n_steps = len(profiles)
    load_series = profiles["load_kw"].astype(float).tolist()
    pv_series = profiles["pv_avail_kw"].astype(float).tolist()
    bpp_series = (
        profiles["grid_price_usd_kwh"].astype(float).tolist()
        if "grid_price_usd_kwh" in profiles.columns
        else [0.2488] * n_steps
    )
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
        initialize={t: max(0.0, min(pv_series[t], PLTS_SUANA_PARAMS["ac_capacity_kw"])) for t in range(n_steps)},
    )
    m.P_pv_max_ac = pyo.Param(initialize=PLTS_SUANA_PARAMS["ac_capacity_kw"])
    # Isolated 20 kV grid diesel production cost / BPP PLTD (USD/kWh)
    m.bpp_pltd = pyo.Param(m.T, initialize={t: max(0.0, bpp_series[t]) for t in range(n_steps)})

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
    e_min_kwh = bess["e_nom_kwh"] * bess["soc_min"]
    e_max_kwh = bess["e_nom_kwh"] * bess["soc_max"]
    e_init_kwh = (
        float(initial_state["e_bess_kwh"])
        if initial_state and "e_bess_kwh" in initial_state
        else bess["e_nom_kwh"] * bess["soc_init"]
    )
    # Clamp slightly for numerical safety across rolling windows
    e_init_kwh = min(max(e_init_kwh, e_min_kwh), e_max_kwh)

    m.P_bess_max = pyo.Param(initialize=bess["p_max_kw"])
    m.E_bess_nom = pyo.Param(initialize=bess["e_nom_kwh"])
    m.E_bess_min = pyo.Param(initialize=e_min_kwh)
    m.E_bess_max = pyo.Param(initialize=e_max_kwh)
    m.E_bess_init = pyo.Param(initialize=e_init_kwh)
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
    m.P_pv = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.P_curt = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    m.P_ch = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.P_dis = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.u_ch = pyo.Var(m.T, domain=pyo.Binary)
    m.E_bess = pyo.Var(m.T_soc, bounds=(e_min_kwh, e_max_kwh))
    m.R_bess = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    m.u_gen = pyo.Var(m.G, m.T, domain=pyo.Binary)
    m.y_start = pyo.Var(m.G, m.T, domain=pyo.Binary)
    m.z_stop = pyo.Var(m.G, m.T, domain=pyo.Binary)
    m.P_gen = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    m.R_gen = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    # Safety slack for unserved load during extreme peaks / SCADA anomalies (heavily penalized)
    m.P_unserved = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # =========================================================================
    # 4. CONSTRAINTS
    # =========================================================================

    # 4.1 20 kV AC-Coupled Bus Power Balance (Fully Isolated System: No External Grid Import/Export)
    def ac_bus_balance_rule(mod, t):
        return (
            sum(mod.P_gen[g, t] for g in mod.G) + mod.P_pv[t] + mod.P_dis[t] + mod.P_unserved[t]
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
        p_low = max(pyo.value(mod.P_min[g]), pyo.value(mod.P_fix[g]) * (1.0 - fix_mode_tolerance))
        return mod.P_gen[g, t] >= p_low * mod.u_gen[g, t]
    m.c_pltd_min = pyo.Constraint(m.G, m.T, rule=pltd_min_rule)

    def pltd_max_rule(mod, g, t):
        if g in mod.G_LS:
            return mod.P_gen[g, t] + mod.R_gen[g, t] <= mod.DMP[g] * mod.u_gen[g, t]
        p_high = min(pyo.value(mod.DMP[g]), pyo.value(mod.P_fix[g]) * (1.0 + fix_mode_tolerance))
        return mod.P_gen[g, t] <= p_high * mod.u_gen[g, t]
    m.c_pltd_max = pyo.Constraint(m.G, m.T, rule=pltd_max_rule)

    def pltd_fix_no_reserve_rule(mod, g, t):
        return mod.R_gen[g, t] == 0.0
    m.c_pltd_fix_no_res = pyo.Constraint(m.G_FIX, m.T, rule=pltd_fix_no_reserve_rule)

    # 4.4 Diesel Start-Up / Shut-Down Logic (with Rolling Horizon Initial State Continuity)
    u_init_map = initial_state.get("u_init") if initial_state else None
    p_init_map = initial_state.get("p_init") if initial_state else None
    up_run_map = initial_state.get("up_run_init") if initial_state else None
    down_run_map = initial_state.get("down_run_init") if initial_state else None

    def uc_logic_rule(mod, g, t):
        if t == 0:
            if u_init_map is None or g not in u_init_map:
                return pyo.Constraint.Skip
            return mod.u_gen[g, 0] - int(u_init_map[g]) == mod.y_start[g, 0] - mod.z_stop[g, 0]
        return mod.u_gen[g, t] - mod.u_gen[g, t - 1] == mod.y_start[g, t] - mod.z_stop[g, t]
    m.c_uc_logic = pyo.Constraint(m.G, m.T, rule=uc_logic_rule)

    def uc_excl_rule(mod, g, t):
        return mod.y_start[g, t] + mod.z_stop[g, t] <= 1
    m.c_uc_excl = pyo.Constraint(m.G, m.T, rule=uc_excl_rule)

    # 4.5 Diesel Minimum Up Time (MUT) & Minimum Down Time (MDT)
    if enforce_mut_mdt:
        def mut_rule(mod, g, t):
            mut_steps = max(1, int(math.ceil(unit_map[g].min_up_hr / dt_hours)))
            if t == 0 and u_init_map is None:
                return pyo.Constraint.Skip
            end_t = min(n_steps - 1, t + mut_steps - 1)
            return sum(mod.u_gen[g, k] for k in range(t, end_t + 1)) >= (end_t - t + 1) * mod.y_start[g, t]
        m.c_mut = pyo.Constraint(m.G, m.T, rule=mut_rule)

        def mdt_rule(mod, g, t):
            mdt_steps = max(1, int(math.ceil(unit_map[g].min_down_hr / dt_hours)))
            if t == 0 and u_init_map is None:
                return pyo.Constraint.Skip
            end_t = min(n_steps - 1, t + mdt_steps - 1)
            return sum(1 - mod.u_gen[g, k] for k in range(t, end_t + 1)) >= (end_t - t + 1) * mod.z_stop[g, t]
        m.c_mdt = pyo.Constraint(m.G, m.T, rule=mdt_rule)

        # Carry-over MUT / MDT obligations from the previous rolling-horizon window
        if u_init_map is not None and up_run_map is not None and down_run_map is not None:
            m.c_init_mut_mdt = pyo.ConstraintList()
            for g in active_units:
                gid = g.unit_id
                if gid not in u_init_map:
                    continue
                mut_steps = max(1, int(math.ceil(g.min_up_hr / dt_hours)))
                mdt_steps = max(1, int(math.ceil(g.min_down_hr / dt_hours)))
                if int(u_init_map[gid]) == 1:
                    rem_up = max(0, mut_steps - int(up_run_map.get(gid, mut_steps)))
                    for k in range(min(rem_up, n_steps)):
                        m.c_init_mut_mdt.add(m.u_gen[gid, k] == 1)
                else:
                    rem_down = max(0, mdt_steps - int(down_run_map.get(gid, mdt_steps)))
                    for k in range(min(rem_down, n_steps)):
                        m.c_init_mut_mdt.add(m.u_gen[gid, k] == 0)

    # 4.6 Diesel Ramp Rate (±30 kW/s) with Initial State Continuity
    def ramp_up_rule(mod, g, t):
        if t == 0:
            if u_init_map is None or p_init_map is None or g not in p_init_map:
                return pyo.Constraint.Skip
            return (
                mod.P_gen[g, 0] - float(p_init_map[g])
                <= mod.ramp_step[g] * int(u_init_map[g]) + mod.DMP[g] * mod.y_start[g, 0]
            )
        return mod.P_gen[g, t] - mod.P_gen[g, t - 1] <= mod.ramp_step[g] * mod.u_gen[g, t - 1] + mod.DMP[g] * mod.y_start[g, t]
    m.c_ramp_up = pyo.Constraint(m.G, m.T, rule=ramp_up_rule)

    def ramp_down_rule(mod, g, t):
        if t == 0:
            if p_init_map is None or g not in p_init_map:
                return pyo.Constraint.Skip
            return (
                float(p_init_map[g]) - mod.P_gen[g, 0]
                <= mod.ramp_step[g] * mod.u_gen[g, 0] + mod.DMP[g] * mod.z_stop[g, 0]
            )
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
        m.c_bess_res_power = pyo.Constraint(
            m.T,
            rule=lambda mod, t: mod.P_dis[t] - mod.P_ch[t] + mod.R_bess[t] <= mod.P_bess_max,
        )
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
    #    Maximize Solar PV utilization & minimize real Diesel BPP production cost
    # =========================================================================
    def objective_rule(mod):
        pv_energy_term = sum(w["pv_reward"] * mod.P_pv[t] * mod.dt for t in mod.T)
        curt_penalty_term = sum(w["curt_penalty"] * mod.P_curt[t] * mod.dt for t in mod.T)
        diesel_cost_term = w["diesel_scale"] * sum(
            mod.bpp_pltd[t] * mod.c_var[g] * mod.P_gen[g, t] * mod.dt
            + w["startup_usd"] * mod.c_start[g] * mod.y_start[g, t]
            for g in mod.G
            for t in mod.T
        )
        bess_cycle_term = sum(w["bess_cycle"] * (mod.P_ch[t] + mod.P_dis[t]) * mod.dt for t in mod.T)
        unserved_penalty_term = sum(w["unserved_penalty"] * mod.P_unserved[t] * mod.dt for t in mod.T)
        return (
            pv_energy_term
            - curt_penalty_term
            - diesel_cost_term
            - bess_cycle_term
            - unserved_penalty_term
        )

    m.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    return m


def extract_results(
    model: pyo.ConcreteModel,
    profiles: pd.DataFrame,
    n_commit: Optional[int] = None,
    t_offset: int = 0,
    hour_offset: Optional[float] = None,
) -> pd.DataFrame:
    """Extract optimal dispatch results (first `n_commit` steps) from a solved Pyomo model."""
    dt = pyo.value(model.dt)
    pln_ids = [g for g in model.G if g.startswith("MDU#")]
    panca_ids = [g for g in model.G if g.startswith("PB#")]
    e_nom = pyo.value(model.E_bess_nom)
    n_total = len(model.T)
    limit = n_total if n_commit is None else min(n_commit, n_total)

    rows = []
    for t in range(limit):
        p_pln = sum(pyo.value(model.P_gen[g, t]) for g in pln_ids)
        p_panca = sum(pyo.value(model.P_gen[g, t]) for g in panca_ids)
        r_diesel = sum(pyo.value(model.R_gen[g, t]) for g in model.G_LS)
        r_bess = pyo.value(model.R_bess[t])
        n_pln_on = sum(int(round(pyo.value(model.u_gen[g, t]))) for g in pln_ids)
        n_panca_on = sum(int(round(pyo.value(model.u_gen[g, t]))) for g in panca_ids)

        if hour_offset is not None:
            hr_val = hour_offset + t * dt
        elif "hour" in profiles.columns:
            hr_val = float(profiles["hour"].iloc[t])
        else:
            hr_val = (t_offset + t) * dt

        row = {
            "t": t_offset + t,
            "hour": hr_val,
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
            "unserved_kw": pyo.value(model.P_unserved[t]),
            "grid_price_usd_kwh": float(pyo.value(model.bpp_pltd[t])),
        }
        if "timestamp" in profiles.columns:
            row["timestamp"] = str(profiles["timestamp"].iloc[t])
        rows.append(row)

    return pd.DataFrame(rows)


def extract_terminal_state(
    model: pyo.ConcreteModel,
    commit_steps: int,
    prev_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract exact carry-over state at the end of `commit_steps` to initialize
    the next rolling-horizon optimization window (BESS energy, unit status,
    generator dispatch, and consecutive up/down step counts for MUT/MDT).
    """
    last_t = commit_steps - 1
    e_bess_end = float(pyo.value(model.E_bess[commit_steps]))

    u_init: Dict[str, int] = {}
    p_init: Dict[str, float] = {}
    up_run_init: Dict[str, int] = {}
    down_run_init: Dict[str, int] = {}

    prev_up = prev_state.get("up_run_init", {}) if prev_state else {}
    prev_down = prev_state.get("down_run_init", {}) if prev_state else {}

    for g in model.G:
        seq = [int(round(pyo.value(model.u_gen[g, k]))) for k in range(commit_steps)]
        u_last = seq[-1]
        u_init[g] = u_last
        p_init[g] = float(pyo.value(model.P_gen[g, last_t]))

        # Count trailing consecutive steps with the same ON/OFF state
        run = 0
        for val in reversed(seq):
            if val == u_last:
                run += 1
            else:
                break

        # If the entire committed block stayed in the same state, add prior run length
        if run == commit_steps:
            if u_last == 1:
                run += int(prev_up.get(g, 0))
            else:
                run += int(prev_down.get(g, 0))

        up_run_init[g] = run if u_last == 1 else 0
        down_run_init[g] = run if u_last == 0 else 0

    return {
        "e_bess_kwh": e_bess_end,
        "u_init": u_init,
        "p_init": p_init,
        "up_run_init": up_run_init,
        "down_run_init": down_run_init,
    }


def _build_window_profiles(
    profiles: pd.DataFrame,
    start_idx: int,
    horizon_steps: int,
    dt_hours: float,
    wrap_tail: bool = True,
) -> pd.DataFrame:
    """Slice a `horizon_steps` lookahead window starting at `start_idx`, padding tail if needed."""
    total_steps = len(profiles)
    end_idx = start_idx + horizon_steps
    if end_idx <= total_steps:
        win = profiles.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    elif wrap_tail and total_steps > 0:
        # Pad the lookahead buffer beyond the end of the dataset cyclically to avoid end-of-horizon BESS myopia
        indices = [(start_idx + k) % total_steps for k in range(horizon_steps)]
        win = profiles.iloc[indices].copy().reset_index(drop=True)
    else:
        win = profiles.iloc[start_idx:total_steps].copy().reset_index(drop=True)

    start_hr = (
        float(profiles["hour"].iloc[start_idx])
        if "hour" in profiles.columns
        else start_idx * dt_hours
    )
    win["hour"] = [start_hr + k * dt_hours for k in range(len(win))]
    return win


def solve_rolling_horizon_milp(
    profiles: pd.DataFrame,
    opt: Any,
    solver_name: str,
    dt_hours: float = 0.5,
    horizon_hours: float = 48.0,
    step_hours: float = 24.0,
    bess_scenario: str = "derated_2pcs",
    include_ps_load: bool = False,
    allow_bess_reserve: bool = True,
    enforce_mut_mdt: bool = True,
    wrap_tail: bool = True,
    mip_gap: float = 0.005,
    time_limit: float = 60.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Solve the Nusa Penida MILP using a Rolling Horizon strategy (default: 48-hour
    lookahead window with 24-hour commit step and exact BESS/Diesel state carry-over).
    """
    total_steps = len(profiles)
    horizon_steps = max(1, int(round(horizon_hours / dt_hours)))
    step_steps = max(1, min(int(round(step_hours / dt_hours)), horizon_steps))
    n_windows = int(math.ceil(total_steps / step_steps))

    if verbose:
        print(
            f"Starting Rolling Horizon MILP: {horizon_hours:.1f}h lookahead ({horizon_steps} steps) | "
            f"{step_hours:.1f}h commit step ({step_steps} steps) | {n_windows} window(s) total"
        )

    committed_frames: List[pd.DataFrame] = []
    state: Optional[Dict[str, Any]] = None

    for w_idx in range(n_windows):
        start_idx = w_idx * step_steps
        commit_steps = min(step_steps, total_steps - start_idx)
        win_profiles = _build_window_profiles(
            profiles=profiles,
            start_idx=start_idx,
            horizon_steps=horizon_steps,
            dt_hours=dt_hours,
            wrap_tail=wrap_tail,
        )

        # In rolling horizon, the 48h lookahead naturally preserves BESS value across days,
        # so cyclic_soc is disabled per window to allow smooth multi-day SoC trajectories.
        model = build_nusa_penida_milp(
            profiles=win_profiles,
            dt_hours=dt_hours,
            bess_scenario=bess_scenario,
            include_ps_load=include_ps_load,
            allow_bess_reserve=allow_bess_reserve,
            enforce_mut_mdt=enforce_mut_mdt,
            cyclic_soc=False,
            initial_state=state,
        )

        if solver_name == "appsi_highs":
            opt.config.mip_gap = mip_gap
            opt.config.time_limit = time_limit
            res = opt.solve(model)
        else:
            res = opt.solve(model, tee=False)

        start_hr = (
            float(profiles["hour"].iloc[start_idx])
            if "hour" in profiles.columns
            else start_idx * dt_hours
        )
        df_commit = extract_results(
            model,
            win_profiles,
            n_commit=commit_steps,
            t_offset=start_idx,
            hour_offset=start_hr,
        )
        committed_frames.append(df_commit)
        state = extract_terminal_state(model, commit_steps=commit_steps, prev_state=state)

        if verbose:
            end_hr = start_hr + commit_steps * dt_hours
            end_soc = df_commit["bess_soc_pct"].iloc[-1]
            print(
                f"  Window {w_idx + 1:2d}/{n_windows} | Commit: {start_hr:5.1f}h–{end_hr:5.1f}h "
                f"| Lookahead: {len(win_profiles) * dt_hours:.1f}h "
                f"| Status: {res.solver.termination_condition} | End SoC: {end_soc:.1f}%"
            )

    return pd.concat(committed_frames, ignore_index=True)
