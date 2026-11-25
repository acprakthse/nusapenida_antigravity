"""
Visualization of MILP optimization & LCOE economic results for the
Nusa Penida Isolated Microgrid (20 kV AC-Coupled).
Generates a 5-panel dashboard:
  1. 20 kV Bus Power Dispatch (PLN Diesel, Panca Bina Diesel, Solar PV, BESS vs. Load)
  2. BESS Operation (Charge/Discharge Power & State of Charge %)
  3. Spinning Reserve (Diesel Spinning Reserve + BESS Bridging vs. System Requirement)
  4. Diesel Unit Commitment (Online Units: PLN/PIPS vs. Panca Bina)
  5. Economics: Grid Electricity Import Price Profile vs. PV+BESS LCOE & Cost Difference
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from data_nusa_penida import SYSTEM_PARAMS, load_profiles_from_csv
from economics import (
    DEFAULT_ECONOMICS,
    compute_lcoe_and_comparison,
    load_economics_from_csv,
    print_economic_summary,
)


# Modern, consistent color palette across all panels
COLORS = {
    "panca":      "#E07A2B",  # Warm amber (Leased Diesel - Panca Bina)
    "pln":        "#2B6CB0",  # Steel blue (PLN Diesel - PIPS)
    "pv":         "#F4B942",  # Sunlit gold (Suana Solar PV)
    "pv_curt":    "#B91C1C",  # Dark red (PV Curtailment)
    "bess_dis":   "#14B8A6",  # Crisp teal (BESS Discharge)
    "bess_ch":    "#7C3AED",  # Clean violet (BESS Charge)
    "load":       "#0F172A",  # Deep slate (System Load)
    "load_ch":    "#6D28D9",  # Violet dashed (Load + BESS Charge)
    "soc":        "#DC2626",  # Crimson (SoC %)
    "res_diesel": "#60A5FA",  # Sky blue (Diesel Spinning Reserve)
    "res_bess":   "#2DD4BF",  # Mint teal (BESS Bridging Reserve)
    "res_req":    "#DC2626",  # Crimson dashed (Reserve Requirement)
    "grid_price": "#1D4ED8",  # Royal blue (Grid Import Price)
    "lcoe_pv":    "#D97706",  # Amber (PV Only LCOE)
    "lcoe_comb":  "#0F766E",  # Deep teal (PV + BESS LCOE)
    "hybrid":     "#4F46E5",  # Indigo (Hybrid Blended Cost)
    "diesel_only": "#B45309", # Burnt orange/brown (Diesel-Only PLTD Cost)
    "savings":    "#15803D",  # Green (Positive savings)
    "premium":    "#B91C1C",  # Red (Cost premium)
    "text":       "#1E293B",  # Dark slate text
    "muted":      "#475569",  # Muted slate text
    "spine":      "#94A3B8",  # Soft grey spine
    "grid":       "#CBD5E1",  # Soft gridline
    "bg_ax":      "#FFFFFF",  # Clean white axes background
    "bg_kpi":     "#F1F5F9",  # Soft KPI pill background
}


def _style_axis(ax: plt.Axes, hide_x_labels: bool = True) -> None:
    """Apply clean and consistent visual styling to each subplot."""
    ax.set_facecolor(COLORS["bg_ax"])
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=COLORS["grid"], linestyle="--", linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["spine"])
    ax.spines["bottom"].set_color(COLORS["spine"])
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", colors=COLORS["muted"], labelsize=9, length=3.5, width=0.8)
    if hide_x_labels:
        ax.tick_params(axis="x", labelbottom=False, length=0)


def _place_panel_header(
    ax: plt.Axes,
    title: str,
    handles=None,
    labels=None,
    ncol: int = 4,
    title_pad: float = 24.0,
    show_legend: bool = True,
) -> None:
    """Place panel title and legend cleanly above the plot area with zero data overlap."""
    ax.set_title(
        title,
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["text"],
        pad=title_pad,
    )
    if not show_legend:
        return
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(-0.005, 1.01),
        ncol=ncol,
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.3,
        borderaxespad=0.0,
        labelcolor=COLORS["muted"],
    )


def plot_dispatch_dashboard(
    df: pd.DataFrame,
    dt_hours: float = 0.5,
    title: str = "Nusa Penida Isolated Microgrid MILP Optimization (AC-Coupled 20 kV)",
    save_path: str | None = "nusa_penida_dispatch.png",
    show: bool = True,
    econ_summary: Dict[str, Any] | None = None,
) -> plt.Figure:
    """Create a clean 5-panel dashboard (dispatch + LCOE vs. grid import comparison)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.edgecolor": COLORS["spine"],
    })

    if econ_summary is None:
        econ_summary = compute_lcoe_and_comparison(df, dt_hours=dt_hours)

    t_hr = df["hour"].to_numpy(dtype=float)
    load_mw = df["load_kw"].to_numpy(dtype=float) / 1000.0
    pv_avail_mw = df["pv_avail_kw"].to_numpy(dtype=float) / 1000.0
    pv_used_mw = df["pv_used_kw"].to_numpy(dtype=float) / 1000.0
    pv_curt_mw = df["pv_curt_kw"].to_numpy(dtype=float) / 1000.0
    bess_ch_mw = np.clip(df["bess_ch_kw"].to_numpy(dtype=float), 0.0, None) / 1000.0
    bess_dis_mw = np.clip(df["bess_dis_kw"].to_numpy(dtype=float), 0.0, None) / 1000.0
    bess_soc = df["bess_soc_pct"].to_numpy(dtype=float)
    pln_mw = df["pltd_pln_kw"].to_numpy(dtype=float) / 1000.0
    panca_mw = df["pltd_panca_kw"].to_numpy(dtype=float) / 1000.0
    res_diesel_mw = np.clip(df["res_diesel_kw"].to_numpy(dtype=float), 0.0, None) / 1000.0
    res_bess_mw = np.clip(df["res_bess_kw"].to_numpy(dtype=float), 0.0, None) / 1000.0
    units_panca = df["units_panca_on"].to_numpy(dtype=float)
    units_pln = df["units_pln_on"].to_numpy(dtype=float)

    if "grid_price_usd_kwh" in df.columns:
        grid_price = df["grid_price_usd_kwh"].to_numpy(dtype=float)
    else:
        grid_price = np.full_like(t_hr, econ_summary["grid_price_avg_usd_kwh"])

    res_req_mw = np.maximum(
        SYSTEM_PARAMS["min_spin_res_kw"] / 1000.0,
        SYSTEM_PARAMS["cloud_ramp_factor"] * pv_used_mw,
    )

    # Energy summary & KPIs
    e_load = load_mw.sum() * dt_hours
    e_pv_avail = pv_avail_mw.sum() * dt_hours
    e_pv_used = pv_used_mw.sum() * dt_hours
    e_pv_curt = pv_curt_mw.sum() * dt_hours
    pv_util = 100.0 * e_pv_used / e_pv_avail if e_pv_avail > 0 else 0.0
    pv_share = 100.0 * e_pv_used / e_load if e_load > 0 else 0.0
    peak_idx = int(np.argmax(load_mw))
    peak_mw = load_mw[peak_idx]
    peak_hr = t_hr[peak_idx]
    soc_min, soc_max = bess_soc.min(), bess_soc.max()

    lcoe_pv = econ_summary["lcoe_pv_only_usd_kwh"]
    lcoe_comb = econ_summary["lcoe_pv_bess_usd_kwh"]
    grid_wtd = econ_summary["grid_price_load_weighted_usd_kwh"]
    diesel_only = econ_summary["diesel_only_cost_usd_kwh"]
    hybrid_blended = econ_summary["hybrid_blended_cost_usd_kwh"]
    cost_diff = econ_summary["cost_diff_usd_kwh"]
    cost_diff_pct = econ_summary["cost_diff_pct"]
    annual_diff = econ_summary["annual_cost_diff_usd"]
    hyb_sav_yr = econ_summary["hybrid_annual_savings_vs_diesel_usd"]
    hyb_sav_pct = econ_summary["hybrid_savings_vs_diesel_pct"]

    fig = plt.figure(figsize=(14.8, 15.0), dpi=140, facecolor="#FFFFFF")
    gs = gridspec.GridSpec(
        5, 1,
        height_ratios=[2.25, 1.30, 1.25, 1.05, 1.50],
        hspace=0.48,
    )

    # Main header & KPI summary bar
    fig.text(
        0.065, 0.978,
        title,
        fontsize=14,
        fontweight="bold",
        color=COLORS["text"],
        ha="left",
        va="top",
    )
    kpi_text = (
        f"Load: {e_load:.1f} MWh (Peak {peak_mw:.2f} MW)   •   "
        f"PV: {e_pv_used:.2f}/{e_pv_avail:.2f} MWh ({pv_util:.0f}% util.)   •   "
        f"LCOE PV+BESS: ${lcoe_comb:.4f}/kWh vs. Grid: ${grid_wtd:.4f}/kWh vs. Diesel-Only: ${diesel_only:.4f}/kWh   •   "
        f"Hybrid Saves ${hyb_sav_yr/1e6:.2f}M/yr vs. Diesel"
    )
    fig.text(
        0.065, 0.952,
        kpi_text,
        fontsize=9.0,
        color=COLORS["muted"],
        ha="left",
        va="top",
        parse_math=False,
        bbox=dict(
            boxstyle="round,pad=0.42,rounding_size=0.35",
            facecolor=COLORS["bg_kpi"],
            edgecolor="#E2E8F0",
            linewidth=0.8,
        ),
    )

    x_pad = dt_hours * 0.5
    x_min = t_hr[0] - x_pad
    x_max = t_hr[-1] + x_pad
    x_ticks = np.arange(0, np.floor(t_hr[-1]) + 0.1, 2.0)
    x_tick_labels = [f"{int(h % 24):02d}:00" for h in x_ticks]

    # -------------------------------------------------------------------------
    # Panel 1: 20 kV AC Bus Power Balance (Stacked Area)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    _style_axis(ax1, hide_x_labels=True)

    ax1.stackplot(
        t_hr,
        panca_mw,
        pln_mw,
        pv_used_mw,
        bess_dis_mw,
        labels=[
            "Leased Diesel (Panca Bina)",
            "PLN Diesel (PIPS)",
            "Suana Solar PV",
            "BESS Discharging",
        ],
        colors=[COLORS["panca"], COLORS["pln"], COLORS["pv"], COLORS["bess_dis"]],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )

    if bess_ch_mw.max() > 1e-4:
        ax1.fill_between(
            t_hr,
            0.0,
            -bess_ch_mw,
            color=COLORS["bess_ch"],
            alpha=0.45,
            hatch="///",
            edgecolor=COLORS["bess_ch"],
            linewidth=0.5,
            label="BESS Charging",
            zorder=2,
        )

    ax1.plot(
        t_hr,
        load_mw,
        color=COLORS["load"],
        linewidth=2.1,
        label="Nusa Penida Load",
        zorder=5,
    )
    if bess_ch_mw.max() > 1e-4:
        ax1.plot(
            t_hr,
            load_mw + bess_ch_mw,
            color=COLORS["load_ch"],
            linewidth=1.3,
            linestyle="--",
            label="Load + BESS Charge",
            zorder=4,
        )
    if pv_curt_mw.max() > 1e-4:
        ax1.plot(
            t_hr,
            panca_mw + pln_mw + pv_used_mw + bess_dis_mw + pv_curt_mw,
            color=COLORS["pv_curt"],
            linewidth=1.6,
            linestyle=":",
            label="PV Curtailment",
            zorder=4,
        )

    peak_hh = int(peak_hr % 24)
    peak_mm = int(round((peak_hr % 1) * 60))
    ax1.scatter([peak_hr], [peak_mw], color=COLORS["load"], s=28, zorder=6)
    ax1.annotate(
        f"Peak {peak_mw:.2f} MW ({peak_hh:02d}:{peak_mm:02d})",
        xy=(peak_hr, peak_mw),
        xytext=(peak_hr - 3.6 if peak_hr > 18 else peak_hr + 1.0, peak_mw + 1.15),
        fontsize=8.3,
        fontweight="semibold",
        color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CBD5E1", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=COLORS["muted"], lw=0.9, connectionstyle="arc3,rad=0.12"),
        zorder=7,
    )

    ax1.axhline(0, color=COLORS["text"], linewidth=0.9, zorder=3)
    y1_max = max((load_mw + bess_ch_mw).max(), (panca_mw + pln_mw + pv_used_mw + bess_dis_mw).max()) * 1.16
    y1_min = -max(bess_ch_mw.max() * 1.45, 0.5) if bess_ch_mw.max() > 1e-4 else 0.0
    ax1.set_ylim(y1_min, y1_max)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylabel("Power (MW)", fontsize=9.5, fontweight="bold", color=COLORS["text"])
    _place_panel_header(
        ax1,
        "1. 20 kV AC-Coupled Bus Power Dispatch (Isolated System)",
        ncol=4,
        title_pad=36.0,
    )

    # -------------------------------------------------------------------------
    # Panel 2: BESS Operation (Charge/Discharge Power & SoC %)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _style_axis(ax2, hide_x_labels=True)

    bar_w = dt_hours * 0.72
    ax2.bar(
        t_hr, bess_dis_mw, width=bar_w,
        color=COLORS["bess_dis"], alpha=0.88, edgecolor="white", linewidth=0.4,
        label="BESS Discharge (+MW)", zorder=3,
    )
    ax2.bar(
        t_hr, -bess_ch_mw, width=bar_w,
        color=COLORS["bess_ch"], alpha=0.82, edgecolor="white", linewidth=0.4,
        label="BESS Charge (-MW)", zorder=3,
    )
    ax2.axhline(0, color=COLORS["text"], linewidth=0.9, zorder=4)

    p_bess_lim = max(bess_dis_mw.max(), bess_ch_mw.max(), 0.25) * 1.25
    ax2.set_ylim(-p_bess_lim, p_bess_lim)
    ax2.set_ylabel("BESS Power (MW)", fontsize=9.5, fontweight="bold", color=COLORS["text"])

    ax2_soc = ax2.twinx()
    ax2_soc.grid(False)
    ax2_soc.spines["top"].set_visible(False)
    ax2_soc.spines["left"].set_visible(False)
    ax2_soc.spines["bottom"].set_visible(False)
    ax2_soc.spines["right"].set_color(COLORS["soc"])
    ax2_soc.spines["right"].set_linewidth(0.9)
    ax2_soc.plot(
        t_hr, bess_soc,
        color=COLORS["soc"], linewidth=2.1, marker="o", markersize=3.0,
        markerfacecolor="white", markeredgewidth=1.3,
        label="BESS SoC (%)", zorder=5,
    )
    ax2_soc.set_ylabel("SoC (%)", fontsize=9.5, fontweight="bold", color=COLORS["soc"])
    ax2_soc.set_ylim(0, 105)
    ax2_soc.set_yticks([0, 25, 50, 75, 100])
    ax2_soc.tick_params(axis="y", labelcolor=COLORS["soc"], colors=COLORS["soc"], labelsize=9, length=3.5)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_soc.get_legend_handles_labels()
    _place_panel_header(
        ax2,
        "2. Suana BESS Utilization (Charge/Discharge Power & State of Charge)",
        handles=lines_1 + lines_2,
        labels=labels_1 + labels_2,
        ncol=3,
        title_pad=22.0,
    )

    # -------------------------------------------------------------------------
    # Panel 3: Spinning Reserve & PV Cloud Ramp Protection
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    _style_axis(ax3, hide_x_labels=True)

    ax3.stackplot(
        t_hr,
        res_diesel_mw,
        res_bess_mw,
        labels=["Diesel Spinning Reserve (Load Share)", "BESS Bridging Reserve"],
        colors=[COLORS["res_diesel"], COLORS["res_bess"]],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    ax3.plot(
        t_hr,
        res_req_mw,
        color=COLORS["res_req"],
        linewidth=2.0,
        linestyle="--",
        label="Minimum Requirement (max 0.7 MW / PV Cloud Ramp)",
        zorder=5,
    )
    y3_max = max((res_diesel_mw + res_bess_mw).max(), res_req_mw.max(), 0.8) * 1.22
    ax3.set_ylim(0.0, y3_max)
    ax3.set_ylabel("Reserve (MW)", fontsize=9.5, fontweight="bold", color=COLORS["text"])
    _place_panel_header(
        ax3,
        "3. Isolated System Spinning Reserve (700 kW Procedure + PV Cloud Ramp)",
        ncol=3,
        title_pad=22.0,
    )

    # -------------------------------------------------------------------------
    # Panel 4: Diesel Unit Commitment (Online Units)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    _style_axis(ax4, hide_x_labels=False)

    ax4.step(
        t_hr, units_panca, where="mid",
        color=COLORS["panca"], linewidth=2.1,
        label="Panca Bina Diesel Units ON", zorder=4,
    )
    ax4.fill_between(
        t_hr, 0, units_panca, step="mid",
        color=COLORS["panca"], alpha=0.10, zorder=2,
    )
    ax4.step(
        t_hr, units_pln, where="mid",
        color=COLORS["pln"], linewidth=2.1,
        label="PLN (PIPS) Diesel Units ON", zorder=4,
    )
    ax4.fill_between(
        t_hr, 0, units_pln, step="mid",
        color=COLORS["pln"], alpha=0.10, zorder=2,
    )

    u_max = max(units_panca.max(), units_pln.max(), 1.0)
    ax4.set_ylim(0, u_max + 2.5)
    ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    ax4.set_ylabel("Units Online", fontsize=9.5, fontweight="bold", color=COLORS["text"])
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels(x_tick_labels)
    _place_panel_header(
        ax4,
        "4. Diesel Generator Unit Commitment",
        ncol=2,
        title_pad=22.0,
    )

    # -------------------------------------------------------------------------
    # Panel 5: Economic Analysis (5a: Price & Cost Benchmarks | 5b: Cost Spread)
    # -------------------------------------------------------------------------
    gs_econ = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[4], width_ratios=[1.35, 1.15], wspace=0.20)

    # Panel 5a: Grid Import Price Profile vs. Diesel-Only & PV+BESS LCOE
    ax5a = fig.add_subplot(gs_econ[0])
    _style_axis(ax5a, hide_x_labels=False)

    ax5a.axhline(
        diesel_only,
        color=COLORS["diesel_only"], linewidth=2.0, linestyle="-.",
        label=f"Diesel-Only (${diesel_only:.4f}/kWh)", zorder=4,
    )
    ax5a.step(
        t_hr, grid_price, where="mid",
        color=COLORS["grid_price"], linewidth=2.1,
        label="Grid Import Price ($/kWh)", zorder=4,
    )
    ax5a.axhline(
        lcoe_comb,
        color=COLORS["lcoe_comb"], linewidth=2.0, linestyle="--",
        label=f"PV+BESS LCOE (${lcoe_comb:.4f}/kWh)", zorder=5,
    )
    ax5a.axhline(
        lcoe_pv,
        color=COLORS["lcoe_pv"], linewidth=1.6, linestyle=":",
        label=f"PV Only LCOE (${lcoe_pv:.4f}/kWh)", zorder=4,
    )
    ax5a.fill_between(
        t_hr, lcoe_comb, grid_price,
        where=(grid_price >= lcoe_comb),
        step="mid", color=COLORS["savings"], alpha=0.13, zorder=2,
    )
    ax5a.fill_between(
        t_hr, lcoe_comb, grid_price,
        where=(grid_price < lcoe_comb),
        step="mid", color=COLORS["premium"], alpha=0.12, zorder=2,
    )

    p5_max = max(grid_price.max(), diesel_only, lcoe_comb, lcoe_pv) * 1.22
    ax5a.set_ylim(0.0, p5_max)
    ax5a.set_xlim(x_min, x_max)
    ax5a.set_xticks(x_ticks)
    ax5a.set_xticklabels(x_tick_labels)
    ax5a.set_ylabel("Cost / Price ($/kWh)", fontsize=9.5, fontweight="bold", color=COLORS["text"])
    ax5a.set_xlabel("Time (WITA / UTC+8)", fontsize=9.5, fontweight="bold", color=COLORS["text"], labelpad=6)
    _place_panel_header(
        ax5a,
        "5a. Grid Import Price vs. Diesel-Only & PV+BESS LCOE",
        ncol=2,
        title_pad=34.0,
    )

    # Panel 5b: LCOE vs. Grid Import vs. Hybrid vs. Diesel-Only Bar Chart
    ax5b = fig.add_subplot(gs_econ[1])
    _style_axis(ax5b, hide_x_labels=False)
    ax5b.grid(axis="x", visible=False)

    categories = [
        "PV Only\nLCOE",
        "PV+BESS\nLCOE",
        "Grid\nImport",
        "Hybrid\nBlended",
        "Diesel\nOnly",
    ]
    values = [lcoe_pv, lcoe_comb, grid_wtd, hybrid_blended, diesel_only]
    bar_colors = [
        COLORS["lcoe_pv"],
        COLORS["lcoe_comb"],
        COLORS["grid_price"],
        COLORS["hybrid"],
        COLORS["diesel_only"],
    ]

    bars = ax5b.bar(categories, values, width=0.56, color=bar_colors, alpha=0.90, edgecolor="white", linewidth=0.6, zorder=3)
    b_max = max(values) * 1.62
    ax5b.set_ylim(0.0, b_max)
    ax5b.set_ylabel("Cost ($/kWh)", fontsize=9.5, fontweight="bold", color=COLORS["text"])
    ax5b.set_xlabel("Supply Option / Benchmark", fontsize=9.5, fontweight="bold", color=COLORS["text"], labelpad=6)
    ax5b.tick_params(axis="x", labelsize=8.3)

    for bar, val in zip(bars, values):
        ax5b.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + b_max * 0.022,
            f"${val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=COLORS["text"],
        )

    # Dual cost-spread callout badge (vs. Grid & vs. Diesel-Only)
    badge_color = COLORS["savings"] if hyb_sav_yr >= 0 else COLORS["premium"]
    badge_bg = "#DCFCE7" if hyb_sav_yr >= 0 else "#FEE2E2"
    diff_text = (
        f"PV+BESS vs. Grid Import: ${cost_diff:+.4f}/kWh ({cost_diff_pct:+.1f}%)  |  ${annual_diff:+,.0f}/yr\n"
        f"Hybrid vs. Diesel-Only:  ${(diesel_only - hybrid_blended):+.4f}/kWh ({hyb_sav_pct:+.1f}%)  |  ${hyb_sav_yr:+,.0f}/yr"
    )
    ax5b.text(
        0.50, 0.97,
        diff_text,
        transform=ax5b.transAxes,
        ha="center",
        va="top",
        fontsize=7.9,
        fontweight="semibold",
        color=badge_color,
        parse_math=False,
        bbox=dict(boxstyle="round,pad=0.34", facecolor=badge_bg, edgecolor=badge_color, linewidth=0.9, alpha=0.94),
        zorder=6,
    )
    _place_panel_header(
        ax5b,
        "5b. Unit Cost Comparison (LCOE vs. Grid vs. Diesel-Only)",
        title_pad=34.0,
        show_legend=False,
    )

    fig.subplots_adjust(top=0.882, bottom=0.045, left=0.065, right=0.935)

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Chart saved to: {save_path}")
    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize Nusa Penida Microgrid MILP dispatch & LCOE results")
    parser.add_argument("--csv", type=str, default="dispatch_results.csv", help="CSV output file from run_simulation.py")
    parser.add_argument("--profiles-csv", type=str, default="input_profiles.csv", help="Optional profiles CSV if grid_price_usd_kwh is missing from dispatch CSV")
    parser.add_argument("--econ-csv", type=str, default="input_economics.csv", help="Optional PV + BESS CAPEX/OPEX CSV")
    parser.add_argument("--dt", type=float, default=0.5, help="Time resolution in hours (default: 0.5)")
    parser.add_argument("--out", type=str, default="nusa_penida_dispatch.png", help="Output PNG image path")
    parser.add_argument("--no-show", action="store_true", help="Save image only without opening plot window")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "grid_price_usd_kwh" not in df.columns and args.profiles_csv and os.path.exists(args.profiles_csv):
        prof = load_profiles_from_csv(args.profiles_csv)
        if len(prof) == len(df):
            df["grid_price_usd_kwh"] = prof["grid_price_usd_kwh"].to_numpy()

    if args.econ_csv and os.path.exists(args.econ_csv):
        econ_params = load_economics_from_csv(args.econ_csv)
    else:
        econ_params = dict(DEFAULT_ECONOMICS)

    econ_summary = compute_lcoe_and_comparison(df, dt_hours=args.dt, economics=econ_params)
    print_economic_summary(econ_summary)

    plot_dispatch_dashboard(
        df,
        dt_hours=args.dt,
        save_path=args.out,
        show=not args.no_show,
        econ_summary=econ_summary,
    )


if __name__ == "__main__":
    main()
