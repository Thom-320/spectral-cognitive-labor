#!/usr/bin/env python3
"""
07_entropy_analysis.py
======================
Reporte informacional y figura resumen para el analisis primario.

Lee spectral_comparison_results.csv para garantizar que MI/JSD
usen exactamente el mismo set primario de diadas que el analisis
espectral principal. Ademas genera una figura-resumen pensada
para compartir con Andrade.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import numpy as np


def build_entropy_figure(results_df, out_path):
    """Genera figura de MI/JSD usando el set primario."""
    grouped = results_df.copy()
    grouped["Group"] = np.where(grouped["Type"] == "MIXED", "Mixed", "Axial")
    axial = grouped[grouped["Group"] == "Axial"]
    mixed = grouped[grouped["Group"] == "Mixed"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    bp = ax.boxplot(
        [axial["MI"].values, mixed["MI"].values],
        tick_labels=["Axial", "Mixed"],
        patch_artist=True,
        widths=0.55,
    )
    bp["boxes"][0].set_facecolor("#4c78a8")
    bp["boxes"][1].set_facecolor("#e45756")
    ax.set_ylabel("I(Player; Cell) [bits]")
    ax.set_title("(a) Informacion mutua")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    bp = ax.boxplot(
        [axial["JSD"].values, mixed["JSD"].values],
        tick_labels=["Axial", "Mixed"],
        patch_artist=True,
        widths=0.55,
    )
    bp["boxes"][0].set_facecolor("#4c78a8")
    bp["boxes"][1].set_facecolor("#e45756")
    ax.set_ylabel("JSD(p1 || p2)")
    ax.set_title("(b) Jensen-Shannon divergence")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.scatter(axial["MI"], axial["eta"], s=75, color="#4c78a8", edgecolors="black", label="Axial")
    ax.scatter(mixed["MI"], mixed["eta"], s=75, color="#e45756", edgecolors="black", label="Mixed")
    rho, p_value = scipy_stats.spearmanr(grouped["MI"], grouped["eta"])
    ax.text(
        0.03,
        0.97,
        f"rho = {rho:.3f}\np = {p_value:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9},
    )
    ax.axhline(y=1.0, color="blue", linestyle="--", linewidth=1.5)
    ax.set_xlabel("MI")
    ax.set_ylabel("eta")
    ax.set_title("(c) Eficiencia espectral vs MI")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.scatter(axial["JSD"], axial["eta"], s=75, color="#4c78a8", edgecolors="black", label="Axial")
    ax.scatter(mixed["JSD"], mixed["eta"], s=75, color="#e45756", edgecolors="black", label="Mixed")
    rho, p_value = scipy_stats.spearmanr(grouped["JSD"], grouped["eta"])
    ax.text(
        0.03,
        0.97,
        f"rho = {rho:.3f}\np = {p_value:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9},
    )
    ax.axhline(y=1.0, color="blue", linestyle="--", linewidth=1.5)
    ax.set_xlabel("JSD")
    ax.set_ylabel("eta")
    ax.set_title("(d) Eficiencia espectral vs JSD")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_andrade_summary(results_df, out_path):
    """Genera la figura resumen de cuatro paneles para compartir."""
    grouped = results_df.copy()
    grouped["Group"] = np.where(grouped["Type"] == "MIXED", "Mixed", "Axial")
    axial = grouped[grouped["Group"] == "Axial"]
    mixed = grouped[grouped["Group"] == "Mixed"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))

    ax = axes[0, 0]
    x = np.linspace(-1.0, 1.0, 200)
    ax.plot(x, x, color="#808080", linewidth=2, label="Base diagonal")
    ax.plot(x, -x, color="#b0b0b0", linewidth=2, label="Base diagonal rotada")
    ax.axhline(0.0, color="#2ca02c", linestyle="--", linewidth=2, label="Direccion LR")
    ax.axvline(0.0, color="#1f77b4", linestyle="--", linewidth=2, label="Direccion TB")
    circle = plt.Circle((0, 0), 0.98, fill=False, color="#444444", alpha=0.4)
    ax.add_patch(circle)
    ax.set_title("(a) Eigenspace degenerado de Fiedler")
    ax.set_xlabel("Componente en v2")
    ax.set_ylabel("Componente en v3")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    bp = ax.boxplot(
        [axial["h_obs"].values, mixed["h_obs"].values],
        tick_labels=["Axial", "Mixed"],
        patch_artist=True,
        widths=0.55,
    )
    bp["boxes"][0].set_facecolor("#4c78a8")
    bp["boxes"][1].set_facecolor("#e45756")
    ax.axhline(grouped["h_fiedler_naive"].iloc[0], color="blue", linestyle="--", linewidth=1.8, label="Fiedler ingenuo")
    ax.axhline(grouped["h_best_axis_baseline"].iloc[0], color="green", linestyle=":", linewidth=1.8, label="Best axis")
    ax.set_ylabel("Conductancia h(S_obs)")
    ax.set_title("(b) Conductancia axial vs mixed")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.scatter(axial["DLIndex_mean"], axial["eta"], s=75, color="#4c78a8", edgecolors="black", label="Axial")
    ax.scatter(mixed["DLIndex_mean"], mixed["eta"], s=75, color="#e45756", edgecolors="black", label="Mixed")
    rho, p_value = scipy_stats.spearmanr(grouped["DLIndex_mean"], grouped["eta"])
    ax.text(
        0.03,
        0.97,
        f"rho = {rho:.3f}\np = {p_value:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9},
    )
    ax.axhline(1.0, color="blue", linestyle="--", linewidth=1.5)
    ax.set_xlabel("DLIndex medio")
    ax.set_ylabel("eta")
    ax.set_title("(c) Lente espectral vs DLIndex")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.scatter(axial["MI"], axial["JSD"], s=75, color="#4c78a8", edgecolors="black", label="Axial")
    ax.scatter(mixed["MI"], mixed["JSD"], s=75, color="#e45756", edgecolors="black", label="Mixed")
    ax.set_xlabel("MI")
    ax.set_ylabel("JSD")
    ax.set_title("(d) Apoyo informacional")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("REPORTE INFORMACIONAL DEL SET PRIMARIO")
    print("=" * 70)

    csv_path = ROOT / "data" / "results" / "spectral_comparison_results.csv"
    if not csv_path.exists():
        print("ERROR: Falta spectral_comparison_results.csv")
        print("Ejecuta antes: python src/02_full_comparison.py")
        return

    results_df = pd.read_csv(csv_path)
    grouped = results_df.copy()
    grouped["Group"] = np.where(grouped["Type"] == "MIXED", "Mixed", "Axial")
    axial = grouped[grouped["Group"] == "Axial"]
    mixed = grouped[grouped["Group"] == "Mixed"]

    print(f"  Diadas en set primario: {len(grouped)}")
    print(f"  Axiales: {len(axial)} | Mixed: {len(mixed)}")

    print("\nResumen MI/JSD por grupo:")
    summary = grouped.groupby("Group")[["MI", "JSD", "eta"]].agg(["mean", "std", "count"]).round(3)
    print(summary.to_string())

    print("\nComparacion axial vs mixed (Mann-Whitney):")
    for column in ["MI", "JSD"]:
        stat, p_value = scipy_stats.mannwhitneyu(axial[column], mixed[column], alternative="two-sided")
        print(f"  {column}: U = {stat:.3f}, p = {p_value:.6f}")

    for column in ["MI", "JSD"]:
        rho, p_value = scipy_stats.spearmanr(grouped["eta"], grouped[column])
        print(f"  Spearman eta vs {column}: rho = {rho:.3f}, p = {p_value:.3e}")

    out_data = ROOT / "data" / "results"
    out_fig = ROOT / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    entropy_csv = out_data / "entropy_analysis_results.csv"
    grouped.to_csv(entropy_csv, index=False)

    entropy_fig = out_fig / "entropy_analysis.png"
    andrade_fig = out_fig / "andrade_summary.png"
    build_entropy_figure(grouped, entropy_fig)
    build_andrade_summary(grouped, andrade_fig)

    print(f"\nGuardado: {entropy_csv}")
    print(f"Guardado: {entropy_fig}")
    print(f"Guardado: {andrade_fig}")


if __name__ == '__main__':
    main()
