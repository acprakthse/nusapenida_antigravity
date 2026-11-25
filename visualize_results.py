"""
Visualisasi hasil optimasi MILP Microgrid Isolated Nusa Penida (20 kV AC-Coupled).
Menghasilkan dashboard 4 panel:
  1. Dispatch Daya Bus 20 kV (PLTD PLN, PLTD Panca Bina, PLTS, BESS vs Beban)
  2. Operasi BESS (Daya Charge/Discharge & State of Charge %)
  3. Cadangan Putar (Spinning Reserve PLTD + Bridging BESS vs Kebutuhan Sistem)
  4. Jumlah Unit PLTD Beroperasi (PLN/PIPS vs Panca Bina) & Curtailment PLTS
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_nusa_penida import SYSTEM_PARAMS


def plot_dispatch_dashboard(
    df: pd.DataFrame,
    dt_hours: float = 0.5,
    title: str = "Optimasi MILP Isolated Microgrid Nusa Penida (AC-Coupled 20 kV)",
    save_path: str | None = "nusa_penida_dispatch.png",
    show: bool = True,
) -> plt.Figure:
    """Membuat visualisasi 4-panel dari DataFrame hasil dispatch MILP."""
    plt.style.use("seaborn-v0_8-whitegrid")

    t_hr = df["hour"].to_numpy()
    load_mw = df["load_kw"].to_numpy() / 1000.0
    pv_avail_mw = df["pv_avail_kw"].to_numpy() / 1000.0
    pv_used_mw = df["pv_used_kw"].to_numpy() / 1000.0
    pv_curt_mw = df["pv_curt_kw"].to_numpy() / 1000.0
    bess_ch_mw = df["bess_ch_kw"].to_numpy() / 1000.0
    bess_dis_mw = df["bess_dis_kw"].to_numpy() / 1000.0
    bess_soc = df["bess_soc_pct"].to_numpy()
    pln_mw = df["pltd_pln_kw"].to_numpy() / 1000.0
    panca_mw = df["pltd_panca_kw"].to_numpy() / 1000.0
    res_diesel_mw = df["res_diesel_kw"].to_numpy() / 1000.0
    res_bess_mw = df["res_bess_kw"].to_numpy() / 1000.0

    res_req_mw = np.maximum(
        SYSTEM_PARAMS["min_spin_res_kw"] / 1000.0,
        SYSTEM_PARAMS["cloud_ramp_factor"] * pv_used_mw,
    )

    # Ringkasan energi (MWh)
    e_load = load_mw.sum() * dt_hours
    e_pv_avail = pv_avail_mw.sum() * dt_hours
    e_pv_used = pv_used_mw.sum() * dt_hours
    e_pv_curt = pv_curt_mw.sum() * dt_hours
    pv_util = 100.0 * e_pv_used / e_pv_avail if e_pv_avail > 0 else 0.0

    fig = plt.figure(figsize=(14, 11), dpi=130)
    gs = gridspec.GridSpec(4, 1, height_ratios=[2.2, 1.3, 1.3, 1.1], hspace=0.32)

    fig.suptitle(
        f"{title}\n"
        f"Total Beban: {e_load:.1f} MWh | PLTS Terkonsumsi: {e_pv_used:.2f}/{e_pv_avail:.2f} MWh "
        f"({pv_util:.1f}%) | Curtailment: {e_pv_curt:.3f} MWh",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    # -------------------------------------------------------------------------
    # Panel 1: Keseimbangan Daya Bus 20 kV (Stacked Area)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    ax1.stackplot(
        t_hr,
        panca_mw,
        pln_mw,
        pv_used_mw,
        bess_dis_mw,
        labels=[
            "PLTD Sewa (Panca Bina)",
            "PLTD PLN (PIPS)",
            "PLTS Suana (Terkonsumsi)",
            "BESS Discharging",
        ],
        colors=["#E67E22", "#2E86C1", "#F1C40F", "#1ABC9C"],
        alpha=0.88,
    )
    # BESS charging digambarkan sebagai tambahan beban di atas kurva beban (atau area negatif)
    ax1.fill_between(
        t_hr,
        0,
        -bess_ch_mw,
        color="#16A085",
        alpha=0.55,
        hatch="//",
        label="BESS Charging (dari Bus 20 kV)",
    )
    ax1.plot(t_hr, load_mw, color="#1B2631", linewidth=2.2, label="Beban Nusa Penida")
    ax1.plot(
        t_hr,
        load_mw + bess_ch_mw,
        color="#117864",
        linewidth=1.3,
        linestyle="--",
        label="Beban + BESS Charge",
    )
    if pv_curt_mw.max() > 1e-4:
        ax1.plot(
            t_hr,
            panca_mw + pln_mw + pv_used_mw + bess_dis_mw + pv_curt_mw,
            color="#C0392B",
            linewidth=1.5,
            linestyle=":",
            label="PLTS Curtailment",
        )

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Daya (MW)", fontweight="bold")
    ax1.set_title("1. Dispatch Daya Bus 20 kV AC-Coupled (Sistem Isolated)", loc="left", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper left", ncol=3, fontsize=8.5, framealpha=0.9)
    ax1.set_xlim(t_hr[0], t_hr[-1])

    # -------------------------------------------------------------------------
    # Panel 2: Operasi BESS (Daya Net & SoC %)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    bar_w = dt_hours * 0.78
    ax2.bar(t_hr, bess_dis_mw, width=bar_w, color="#1ABC9C", alpha=0.85, label="BESS Discharge (+MW)")
    ax2.bar(t_hr, -bess_ch_mw, width=bar_w, color="#8E44AD", alpha=0.80, label="BESS Charge (-MW)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Daya BESS (MW)", fontweight="bold")
    ax2.set_title("2. Utilisasi BESS Suana (Daya Charge/Discharge & State of Charge)", loc="left", fontsize=10.5, fontweight="bold")

    ax2_soc = ax2.twinx()
    ax2_soc.plot(t_hr, bess_soc, color="#C0392B", linewidth=2.2, marker="o", markersize=2.8, label="BESS SoC (%)")
    ax2_soc.set_ylabel("SoC (%)", fontweight="bold", color="#C0392B")
    ax2_soc.set_ylim(0, 105)
    ax2_soc.tick_params(axis="y", labelcolor="#C0392B")

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_soc.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", ncol=3, fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 3: Cadangan Putar (Spinning Reserve) & Proteksi Cloud Ramp PV
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.stackplot(
        t_hr,
        res_diesel_mw,
        res_bess_mw,
        labels=["Cadangan Putar PLTD (Load Share)", "Bridging Reserve BESS"],
        colors=["#5DADE2", "#48C9B0"],
        alpha=0.85,
    )
    ax3.plot(
        t_hr,
        res_req_mw,
        color="#C0392B",
        linewidth=2.0,
        linestyle="--",
        label="Kebutuhan Minimum (max 0,7 MW / Cloud Ramp PV)",
    )
    ax3.set_ylabel("Cadangan (MW)", fontweight="bold")
    ax3.set_title("3. Cadangan Putar Sistem Isolated (Prosedur 700 kW + Cloud Ramp PV)", loc="left", fontsize=10.5, fontweight="bold")
    ax3.legend(loc="upper left", ncol=3, fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 4: Unit Commitment PLTD (Jumlah Unit ON)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.step(t_hr, df["units_panca_on"], where="mid", color="#E67E22", linewidth=2.0, label="Unit PLTD Panca Bina ON")
    ax4.step(t_hr, df["units_pln_on"], where="mid", color="#2E86C1", linewidth=2.0, label="Unit PLTD PLN (PIPS) ON")
    ax4.set_ylabel("Jumlah Unit ON", fontweight="bold")
    ax4.set_xlabel("Waktu (Jam WITA)", fontweight="bold")
    ax4.set_title("4. Komitmen Unit PLTD (Unit Commitment)", loc="left", fontsize=10.5, fontweight="bold")
    ax4.legend(loc="upper left", ncol=2, fontsize=8.5, framealpha=0.9)

    # Format sumbu X setiap 2 jam
    x_ticks = np.arange(0, t_hr[-1] + 0.1, 2.0)
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels([f"{int(h % 24):02d}:00" for h in x_ticks])

    fig.subplots_adjust(top=0.91, bottom=0.06, left=0.07, right=0.93)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grafik disimpan ke: {save_path}")
    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualisasi hasil dispatch MILP Microgrid Nusa Penida")
    parser.add_argument("--csv", type=str, default="dispatch_results.csv", help="File CSV hasil run_simulation.py")
    parser.add_argument("--dt", type=float, default=0.5, help="Resolusi waktu dalam jam (default: 0.5)")
    parser.add_argument("--out", type=str, default="nusa_penida_dispatch.png", help="Path file gambar output")
    parser.add_argument("--no-show", action="store_true", help="Simpan gambar saja tanpa membuka jendela plot")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    plot_dispatch_dashboard(df, dt_hours=args.dt, save_path=args.out, show=not args.no_show)


if __name__ == "__main__":
    main()
