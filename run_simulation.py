"""
Main script to run MILP simulation & optimization for the Nusa Penida Isolated Microgrid.
Supports CSV input for load & Solar PV profiles or report-calibrated synthetic profiles.
"""

from __future__ import annotations
import argparse
import sys
import pyomo.environ as pyo

from data_nusa_penida import generate_sample_profiles, load_profiles_from_csv
from microgrid_milp import build_nusa_penida_milp, extract_results


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


def main():
    parser = argparse.ArgumentParser(description="Nusa Penida Isolated Microgrid MILP Optimization")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV containing load_kw and pv_avail_kw columns")
    parser.add_argument("--dt", type=float, default=0.5, help="Time resolution in hours (default: 0.5 = 30 min)")
    parser.add_argument(
        "--bess",
        type=str,
        default="derated_2pcs",
        choices=["derated_2pcs", "full_3pcs", "full_eol"],
        help="Suana BESS scenario (default: derated_2pcs = 2 MW / 2.2 MWh)",
    )
    parser.add_argument("--solver", type=str, default="appsi_highs", help="Pyomo MILP solver name")
    parser.add_argument("--out", type=str, default="dispatch_results.csv", help="Output CSV path for dispatch results")
    parser.add_argument("--include-ps", action="store_true", help="Add 242 kW auxiliary (PS) load to load profile")
    parser.add_argument("--no-bess-reserve", action="store_true", help="Disable BESS spinning reserve contribution")
    parser.add_argument("--plot", action="store_true", help="Generate PNG dashboard chart after optimization")
    parser.add_argument("--plot-out", type=str, default="nusa_penida_dispatch.png", help="Output PNG path when --plot is enabled")
    args = parser.parse_args()

    if args.csv:
        profiles = load_profiles_from_csv(args.csv)
        print(f"Loaded profile data from: {args.csv} ({len(profiles)} timesteps)")
    else:
        profiles = generate_sample_profiles(n_days=1, dt_hours=args.dt)
        print(f"Using calibrated Nusa Penida synthetic profiles ({len(profiles)} timesteps, dt={args.dt} h)")

    model = build_nusa_penida_milp(
        profiles=profiles,
        dt_hours=args.dt,
        bess_scenario=args.bess,
        include_ps_load=args.include_ps,
        allow_bess_reserve=not args.no_bess_reserve,
    )

    solver_name, opt = get_available_solver(args.solver)
    if opt is None:
        print(
            "\n[ERROR] No MILP solver installed (appsi_highs / cbc / glpk / gurobi).\n"
            "Please install the lightweight HiGHS solver with:\n"
            "    pip install highspy\n"
        )
        sys.exit(1)

    print(f"Running MILP optimization with solver '{solver_name}' (BESS scenario: {args.bess})...")
    if solver_name == "appsi_highs":
        opt.config.mip_gap = 0.005
        opt.config.time_limit = 60.0
        res = opt.solve(model)
    else:
        res = opt.solve(model, tee=False)

    print(f"Termination status: {res.solver.termination_condition}")

    df_res = extract_results(model, profiles)
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

    print("\n================ OPTIMIZATION SUMMARY ================")
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

    if args.plot:
        from visualize_results import plot_dispatch_dashboard
        plot_dispatch_dashboard(
            df_res,
            dt_hours=args.dt,
            title=f"Nusa Penida Isolated Microgrid MILP Optimization (BESS: {args.bess})",
            save_path=args.plot_out,
            show=False,
        )


if __name__ == "__main__":
    main()
