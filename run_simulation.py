"""
Main script to run Rolling Horizon MILP simulation (default 48-hour lookahead),
optimization, and LCOE economic comparison for the Nusa Penida Isolated Microgrid.
Supports CSV inputs from `input/` and saves all generated results to `output/`.
"""

from __future__ import annotations
import argparse
import os
import sys
import pandas as pd
import pyomo.environ as pyo

from src.data_nusa_penida import (
    export_sample_profiles_csv,
    generate_sample_profiles,
    load_profiles_from_csv,
)
from src.economics import (
    DEFAULT_ECONOMICS,
    compute_lcoe_and_comparison,
    load_economics_from_csv,
    print_economic_summary,
    save_economics_template_csv,
)
from src.microgrid_milp import (
    build_nusa_penida_milp,
    extract_results,
    solve_rolling_horizon_milp,
)


def get_available_solver(preferred: str = "appsi_highs"):
    """Find an available MILP solver in the Pyomo environment."""
    candidates = [preferred, "appsi_highs", "highs", "cbc", "glpk", "gurobi", "cplex", "scip"]
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            opt = pyo.SolverFactory(name)
            if opt is not None and opt.available(exception_flag=False):
                return name, opt
        except Exception:
            continue
    return None, None


def _ensure_parent_dir(file_path: str) -> None:
    """Create parent directory if it does not already exist."""
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Nusa Penida Isolated Microgrid Rolling Horizon MILP (48h) & LCOE Analysis"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV containing load_kw, pv_avail_kw, and optional grid_price_usd_kwh (default: input/NP_Genprofile_PV1_PV2.csv)",
    )
    parser.add_argument(
        "--econ-csv",
        type=str,
        default="input/input_economics.csv",
        help="Path to PV + BESS + Diesel CAPEX/OPEX CSV (default: input/input_economics.csv if present)",
    )
    parser.add_argument("--dt", type=float, default=0.5, help="Time resolution in hours (default: 0.5 = 30 min)")
    parser.add_argument("--days", type=int, default=3, help="Number of days for synthetic profiles / template export (default: 3)")
    parser.add_argument(
        "--max-days",
        type=float,
        default=None,
        help="Optional limit on number of days to simulate from the loaded CSV (default: full CSV)",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=48.0,
        help="Rolling horizon lookahead window in hours (default: 48.0)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=24.0,
        help="Rolling horizon commit step in hours (default: 24.0)",
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="Disable rolling horizon and solve the entire profile in a single MILP",
    )
    parser.add_argument(
        "--bess",
        type=str,
        default="derated_2pcs",
        choices=["derated_2pcs", "full_3pcs", "full_eol"],
        help="Suana BESS scenario (default: derated_2pcs = 2 MW / 2.2 MWh)",
    )
    parser.add_argument("--solver", type=str, default="appsi_highs", help="Pyomo MILP solver name")
    parser.add_argument("--out", type=str, default="output/dispatch_results.csv", help="Output CSV path for dispatch results")
    parser.add_argument("--econ-out", type=str, default="output/economic_summary.csv", help="Output CSV path for LCOE summary")
    parser.add_argument("--include-ps", action="store_true", help="Add 242 kW auxiliary (PS) load to load profile")
    parser.add_argument("--no-bess-reserve", action="store_true", help="Disable BESS spinning reserve contribution")
    parser.add_argument("--plot", action="store_true", help="Generate PNG dashboard chart after optimization")
    parser.add_argument("--plot-out", type=str, default="output/nusa_penida_dispatch.png", help="Output PNG path when --plot is enabled")
    parser.add_argument(
        "--export-templates",
        action="store_true",
        help="Export ready-to-edit CSV templates into input/ and exit",
    )
    args = parser.parse_args()

    if args.export_templates:
        export_sample_profiles_csv("input/input_profiles.csv", n_days=args.days, dt_hours=args.dt)
        save_economics_template_csv("input/input_economics.csv")
        print(f"Exported CSV input templates ({args.days} day(s), dt={args.dt} h):")
        print("  - input/input_profiles.csv  (columns: hour, load_kw, pv_avail_kw, grid_price_usd_kwh)")
        print("  - input/input_economics.csv (PV, BESS & Diesel CAPEX/OPEX and financial parameters)")
        return

    default_profile_path = "input/NP_Genprofile_PV1_PV2.csv"
    fallback_profile_path = "input/input_profiles.csv"
    if args.csv:
        profiles = load_profiles_from_csv(args.csv, dt_hours=args.dt)
        print(f"Loaded profile data from: {args.csv} ({len(profiles)} timesteps)")
    elif os.path.exists(default_profile_path):
        profiles = load_profiles_from_csv(default_profile_path, dt_hours=args.dt)
        print(f"Loaded profile data from: {default_profile_path} ({len(profiles)} timesteps)")
    elif os.path.exists(fallback_profile_path):
        profiles = load_profiles_from_csv(fallback_profile_path, dt_hours=args.dt)
        print(f"Loaded profile data from: {fallback_profile_path} ({len(profiles)} timesteps)")
    else:
        profiles = generate_sample_profiles(n_days=args.days, dt_hours=args.dt)
        print(f"Using calibrated Nusa Penida synthetic profiles ({len(profiles)} timesteps, {args.days} days, dt={args.dt} h)")

    if args.max_days is not None and args.max_days > 0:
        max_steps = int(round(args.max_days * 24.0 / args.dt))
        profiles = profiles.iloc[:max_steps].copy().reset_index(drop=True)
        print(f"Sliced profile to first {args.max_days:g} day(s) ({len(profiles)} timesteps)")

    if args.econ_csv and os.path.exists(args.econ_csv):
        econ_params = load_economics_from_csv(args.econ_csv)
        print(f"Loaded economic parameters from: {args.econ_csv}")
    else:
        econ_params = dict(DEFAULT_ECONOMICS)
        print("Using built-in default PV + BESS + Diesel economic parameters")

    solver_name, opt = get_available_solver(args.solver)
    if opt is None:
        print(
            "\n[ERROR] No MILP solver installed (appsi_highs / cbc / glpk / gurobi).\n"
            "Please install the lightweight HiGHS solver with:\n"
            "    pip install highspy\n"
        )
        sys.exit(1)

    if args.single_shot:
        print(f"Running Single-Shot MILP with solver '{solver_name}' (BESS scenario: {args.bess})...")
        model = build_nusa_penida_milp(
            profiles=profiles,
            dt_hours=args.dt,
            bess_scenario=args.bess,
            include_ps_load=args.include_ps,
            allow_bess_reserve=not args.no_bess_reserve,
        )
        if solver_name == "appsi_highs":
            opt.config.mip_gap = 0.005
            opt.config.time_limit = 60.0
            res = opt.solve(model)
        else:
            res = opt.solve(model, tee=False)
        print(f"Termination status: {res.solver.termination_condition}")
        df_res = extract_results(model, profiles)
        mode_label = "Single-Shot MILP"
    else:
        print(f"Running Rolling Horizon MILP with solver '{solver_name}' (BESS scenario: {args.bess})...")
        df_res = solve_rolling_horizon_milp(
            profiles=profiles,
            opt=opt,
            solver_name=solver_name,
            dt_hours=args.dt,
            horizon_hours=args.horizon,
            step_hours=args.step,
            bess_scenario=args.bess,
            include_ps_load=args.include_ps,
            allow_bess_reserve=not args.no_bess_reserve,
        )
        mode_label = f"Rolling Horizon {args.horizon:.0f}h / Step {args.step:.0f}h"

    _ensure_parent_dir(args.out)
    df_res.to_csv(args.out, index=False)

    dt = args.dt
    e_load = df_res["load_kw"].sum() * dt / 1000.0
    e_pv_avail = df_res["pv_avail_kw"].sum() * dt / 1000.0
    e_pv_used = df_res["pv_used_kw"].sum() * dt / 1000.0
    e_pv_curt = df_res["pv_curt_kw"].sum() * dt / 1000.0
    e_bess_ch = df_res["bess_ch_kw"].sum() * dt / 1000.0
    e_bess_dis = df_res["bess_dis_kw"].sum() * dt / 1000.0
    e_pln = df_res["pltd_pln_kw"].sum() * dt / 1000.0
    e_panca = df_res["pltd_panca_kw"].sum() * dt / 1000.0

    pv_util_pct = (100.0 * e_pv_used / e_pv_avail) if e_pv_avail > 0 else 0.0
    pv_share_pct = (100.0 * e_pv_used / e_load) if e_load > 0 else 0.0
    sim_hours = len(df_res) * dt

    print("\n================ OPTIMIZATION SUMMARY ================")
    print(f"Optimization Mode          : {mode_label}")
    print(f"Simulated Horizon          : {sim_hours:.1f} hours ({sim_hours / 24.0:.2f} days)")
    print(f"Total System Load          : {e_load:8.2f} MWh")
    print(f"Available Solar PV         : {e_pv_avail:8.2f} MWh")
    print(f"Utilized Solar PV          : {e_pv_used:8.2f} MWh ({pv_util_pct:.2f}% of available, {pv_share_pct:.2f}% of load)")
    print(f"Solar PV Curtailment       : {e_pv_curt:8.4f} MWh")
    print(f"BESS Charging              : {e_bess_ch:8.2f} MWh")
    print(f"BESS Discharging           : {e_bess_dis:8.2f} MWh")
    print(f"BESS SoC Min / Max         : {df_res['bess_soc_pct'].min():.1f}% / {df_res['bess_soc_pct'].max():.1f}%")
    print(f"PLN Diesel (PIPS)          : {e_pln:8.2f} MWh")
    print(f"Leased Diesel (Panca Bina) : {e_panca:8.2f} MWh")
    print(f"Min Total Spinning Reserve : {df_res['res_total_kw'].min():.1f} kW")
    print(f"Dispatch results saved to  : {args.out}")
    print("======================================================")

    econ_summary = compute_lcoe_and_comparison(df_res, dt_hours=dt, economics=econ_params)
    print_economic_summary(econ_summary)

    _ensure_parent_dir(args.econ_out)
    pd.DataFrame([econ_summary]).T.reset_index().rename(
        columns={"index": "metric", 0: "value"}
    ).to_csv(args.econ_out, index=False)
    print(f"Economic summary saved to  : {args.econ_out}")

    if args.plot:
        from src.visualize_results import plot_dispatch_dashboard
        plot_dispatch_dashboard(
            df_res,
            dt_hours=args.dt,
            title=f"Nusa Penida Microgrid MILP ({mode_label} | BESS: {args.bess})",
            save_path=args.plot_out,
            show=False,
            econ_summary=econ_summary,
        )


if __name__ == "__main__":
    main()
