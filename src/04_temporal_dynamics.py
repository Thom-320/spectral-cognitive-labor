#!/usr/bin/env python3
"""
04_temporal_dynamics.py
=======================
Curva de aprendizaje espectral: h(S_obs(t)) para t = 1..60.
Muestra cuándo la cognición humana cruza el umbral de Fiedler.

Correcciones aplicadas:
  - Grafo 8×8 8-conectividad: 210 aristas (no 224).
  - Columnas 'a' filtradas por Player (PL1 vs PL2), no a/b.
  - Itera solo sobre rondas existentes en los datos.
  - Todas las díadas, agrupadas LR/TB vs Mixed con bandas de std.

Autor: Thomas Chísica
Fecha: Febrero 2026
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. GRAFO: 8×8 King Grid (8-conectividad), 64 nodos, 210 aristas
# ============================================================

def build_adjacency(n: int = 8) -> np.ndarray:
    """Matriz de adyacencia del king grid n×n."""
    N = n * n
    A = np.zeros((N, N), dtype=np.float64)
    for r in range(n):
        for c in range(n):
            node = r * n + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        A[node, nr * n + nc] = 1.0
    return A


def fiedler_conductance(A: np.ndarray, degrees: np.ndarray) -> float:
    """Conductancia de la bisección de Fiedler (mediana del 2do eigenvector)."""
    L = np.diag(degrees) - A
    evals, evecs = np.linalg.eigh(L)
    idx = np.argsort(evals)
    fv = evecs[:, idx[1]]
    S = set(np.where(fv >= np.median(fv))[0])
    return _conductance(S, A, degrees)


def _conductance(S: set, A: np.ndarray, degrees: np.ndarray) -> float:
    """h(S) = cut(S, S̄) / min(vol(S), vol(S̄))."""
    if not S or len(S) == A.shape[0]:
        return np.nan
    S_arr = np.array(list(S))
    Sbar_arr = np.array(list(set(range(A.shape[0])) - S))
    cut = A[np.ix_(S_arr, Sbar_arr)].sum()
    vol_S = degrees[S_arr].sum()
    vol_Sbar = degrees[Sbar_arr].sum()
    denom = min(vol_S, vol_Sbar)
    return cut / denom if denom > 0 else np.nan


# ============================================================
# 2. CLASIFICACIÓN DE DÍADAS
# ============================================================

def classify_dyads(df: pd.DataFrame, min_round: int = 40) -> dict:
    """Clasifica cada díada como 'LR', 'TB', o 'MIXED'."""
    late = df.loc[df["Round"] >= min_round]
    result = {}
    for dyad, grp in late.groupby("Dyad"):
        cats = set(grp["Category"].unique())
        if {"LEFT", "RIGHT"} <= cats:
            result[dyad] = "LR"
        elif {"TOP", "BOTTOM"} <= cats:
            result[dyad] = "TB"
        else:
            result[dyad] = "MIXED"
    return result


# ============================================================
# 3. DINÁMICA TEMPORAL — VECTORIZADA
# ============================================================

def temporal_conductance_all(
    df: pd.DataFrame, A: np.ndarray, degrees: np.ndarray
) -> pd.DataFrame:
    """
    Para cada díada y cada ronda existente t, calcula h(S_obs(t))
    donde S_obs(t) se define por frecuencias acumuladas hasta t.

    Retorna DataFrame con columnas:
        Dyad, Round, h_obs, partition_size, cut_edges
    """
    a_cols = [f"a{i}{j}" for i in range(1, 9) for j in range(1, 9)]
    records = []

    for dyad, dyad_df in df.groupby("Dyad"):
        players = sorted(dyad_df["Player"].unique())
        if len(players) != 2:
            continue
        p1, p2 = players

        # Separar datos por jugador, ordenar por ronda
        p1_df = dyad_df.loc[dyad_df["Player"] == p1, ["Round"] + a_cols].sort_values("Round")
        p2_df = dyad_df.loc[dyad_df["Player"] == p2, ["Round"] + a_cols].sort_values("Round")

        # Rondas donde AMBOS jugadores tienen datos
        common_rounds = sorted(set(p1_df["Round"]) & set(p2_df["Round"]))
        if not common_rounds:
            continue

        # Matrices de visitas: filas = rondas, columnas = 64 celdas
        p1_mat = p1_df.set_index("Round").loc[common_rounds, a_cols].values.astype(np.float64)
        p2_mat = p2_df.set_index("Round").loc[common_rounds, a_cols].values.astype(np.float64)

        # Acumulado temporal (cumsum por filas)
        p1_cumul = np.cumsum(p1_mat, axis=0)
        p2_cumul = np.cumsum(p2_mat, axis=0)

        for t_idx, rnd in enumerate(common_rounds):
            # S_obs(t) = {v : freq1_cumul(v) > freq2_cumul(v)}
            dominance = p1_cumul[t_idx] - p2_cumul[t_idx]
            S = set(np.where(dominance > 0)[0])

            sz = len(S)
            if sz < 5 or sz > 59:
                records.append((dyad, rnd, np.nan, sz, np.nan))
                continue

            h = _conductance(S, A, degrees)
            # Aristas de corte
            S_arr = np.array(list(S))
            Sbar_arr = np.array(list(set(range(64)) - S))
            cut = int(A[np.ix_(S_arr, Sbar_arr)].sum())

            records.append((dyad, rnd, h, sz, cut))

    return pd.DataFrame(
        records, columns=["Dyad", "Round", "h_obs", "partition_size", "cut_edges"]
    )


# ============================================================
# 4. FIGURAS
# ============================================================

def plot_dynamics(
    temporal_df: pd.DataFrame,
    dyad_types: dict,
    h_fiedler: float,
    h_optimal: float = 22 / 210,
    outdir: Path = Path("."),
):
    """Genera las gráficas de dinámica temporal."""

    temporal_df = temporal_df.copy()
    temporal_df["Type"] = temporal_df["Dyad"].map(dyad_types)
    temporal_df["eta"] = h_fiedler / temporal_df["h_obs"]

    clear = temporal_df[temporal_df["Type"].isin(["LR", "TB"])]
    mixed = temporal_df[temporal_df["Type"] == "MIXED"]

    # --- FIGURA 1: Subplots LR/TB vs Mixed con trayectorias + media ± std ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, subset, title, color in [
        (ax1, clear, "Clear Splits (LR / TB)", "#2ecc71"),
        (ax2, mixed, "Mixed Splits", "#e74c3c"),
    ]:
        # Trayectorias individuales
        for dyad, grp in subset.groupby("Dyad"):
            valid = grp.dropna(subset=["h_obs"])
            ax.plot(valid["Round"], valid["h_obs"], "-", alpha=0.2, color=color, linewidth=1)

        # Media ± std por ronda
        stats = subset.groupby("Round")["h_obs"].agg(["mean", "std"]).dropna()
        if len(stats) > 0:
            ax.plot(stats.index, stats["mean"], "o-", color=color,
                    linewidth=2.5, markersize=4, label="Mean h(S$_{obs}$)", zorder=5)
            ax.fill_between(
                stats.index,
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                alpha=0.15, color=color,
            )

        ax.axhline(y=h_fiedler, color="blue", linestyle="--", linewidth=2,
                    label=f"h(Fiedler) = {h_fiedler:.4f}")
        ax.axhline(y=h_optimal, color="green", linestyle=":", linewidth=2,
                    label=f"h(LR optimal) = {h_optimal:.4f}")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylim(0, 0.85)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)

    ax1.set_ylabel("Conductance h(S$_{obs}$(t))", fontsize=12)

    plt.tight_layout()
    fig.savefig(outdir / "temporal_dynamics.png", dpi=300, bbox_inches="tight")
    print(f"  Guardado: {outdir / 'temporal_dynamics.png'}")

    # --- FIGURA 2: Eta temporal ---
    fig2, ax = plt.subplots(figsize=(10, 6))

    for subset, label, color, marker in [
        (clear, "LR/TB", "#2ecc71", "o"),
        (mixed, "Mixed", "#e74c3c", "^"),
    ]:
        stats = subset.groupby("Round")["eta"].agg(["mean", "std"]).dropna()
        if len(stats) > 0:
            ax.plot(stats.index, stats["mean"], f"{marker}-", color=color,
                    linewidth=2, markersize=5, label=f"{label} mean η")
            ax.fill_between(stats.index, stats["mean"] - stats["std"],
                            stats["mean"] + stats["std"], alpha=0.12, color=color)

    ax.axhline(y=1.0, color="blue", linestyle="--", linewidth=2, label="η = 1 (Fiedler)")
    ax.axhspan(1.0, ax.get_ylim()[1] or 3, alpha=0.06, color="green")
    ax.axhspan(0, 1.0, alpha=0.03, color="red")
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("η = h(Fiedler) / h(S$_{obs}$)", fontsize=13)
    ax.set_title("Spectral Efficiency Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))

    plt.tight_layout()
    fig2.savefig(outdir / "temporal_eta_evolution.png", dpi=300, bbox_inches="tight")
    print(f"  Guardado: {outdir / 'temporal_eta_evolution.png'}")

    # --- FIGURA 3: Barras por fase ---
    fig3, ax = plt.subplots(figsize=(8, 5))
    phases = [(1, 20, "Early\n(R1-20)"), (21, 40, "Mid\n(R21-40)"), (41, 60, "Late\n(R41-60)")]
    x = np.arange(len(phases))
    w = 0.35

    for offset, subset, label, color in [
        (-w / 2, clear, "LR/TB", "#2ecc71"),
        (w / 2, mixed, "Mixed", "#e74c3c"),
    ]:
        means, stds = [], []
        for lo, hi, _ in phases:
            h = subset.loc[(subset["Round"] >= lo) & (subset["Round"] <= hi), "h_obs"].dropna()
            means.append(h.mean())
            stds.append(h.std())
        ax.bar(x + offset, means, w, yerr=stds, label=label, color=color,
               edgecolor="black", capsize=5)

    ax.axhline(y=h_fiedler, color="blue", linestyle="--", linewidth=2, label="h(Fiedler)")
    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in phases])
    ax.set_ylabel("Mean h(S$_{obs}$)", fontsize=12)
    ax.set_title("Conductance by Experimental Phase", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig3.savefig(outdir / "temporal_phases.png", dpi=300, bbox_inches="tight")
    print(f"  Guardado: {outdir / 'temporal_phases.png'}")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 70)
    print("DINÁMICA TEMPORAL DE EFICIENCIA ESPECTRAL")
    print("  Grafo: King Grid 8×8 | 64 nodos | 210 aristas")
    print("=" * 70)

    # --- Grafo ---
    A = build_adjacency(8)
    degrees = A.sum(axis=1)
    h_fiedler = fiedler_conductance(A, degrees)
    h_optimal = 22 / 210  # LR/TB corte perfecto

    print(f"\n  |E| = {int(A.sum() / 2)}")
    print(f"  h(Fiedler)  = {h_fiedler:.6f}  ({28}/{210})")
    print(f"  h(LR optim) = {h_optimal:.6f}  ({22}/{210})")

    # --- Datos ---
    csv_path = ROOT / "data" / "raw" / "humans_only_absent.csv"
    df = pd.read_csv(csv_path)
    print(f"  Díadas: {df['Dyad'].nunique()} | Filas: {len(df)}")

    # --- Clasificación ---
    dyad_types = classify_dyads(df)
    n_clear = sum(1 for v in dyad_types.values() if v in ("LR", "TB"))
    n_mixed = sum(1 for v in dyad_types.values() if v == "MIXED")
    print(f"  LR/TB: {n_clear} | Mixed: {n_mixed}")

    # --- Temporal ---
    print("\n  Calculando h(S_obs(t)) por ronda...")
    temporal_df = temporal_conductance_all(df, A, degrees)
    temporal_df["Type"] = temporal_df["Dyad"].map(dyad_types)

    outdir_data = ROOT / "data" / "results"
    outdir_data.mkdir(parents=True, exist_ok=True)
    temporal_df.to_csv(outdir_data / "temporal_conductance_results.csv", index=False)
    print(f"  Guardado: temporal_conductance_results.csv ({len(temporal_df)} filas)")

    # --- Estadísticas por fase ---
    print("\n" + "=" * 70)
    print("RESULTADOS POR FASE")
    print("=" * 70)

    for label, types in [("LR/TB", ["LR", "TB"]), ("MIXED", ["MIXED"])]:
        sub = temporal_df[temporal_df["Type"].isin(types)]
        for lo, hi, phase in [(1, 20, "Early"), (21, 40, "Mid"), (41, 60, "Late")]:
            h = sub.loc[(sub["Round"] >= lo) & (sub["Round"] <= hi), "h_obs"].dropna()
            print(f"  {label:6s} {phase:5s} (R{lo:02d}-{hi:02d}): "
                  f"h = {h.mean():.4f} ± {h.std():.4f}  (n={len(h)})")
        print()

    # --- Cruce del umbral de Fiedler ---
    print(f"--- Cruce del umbral h(Fiedler) = {h_fiedler:.4f} ---")
    clear_dyads = [d for d, t in dyad_types.items() if t in ("LR", "TB")]
    for dyad in sorted(clear_dyads):
        traj = temporal_df.loc[temporal_df["Dyad"] == dyad].dropna(subset=["h_obs"])
        crossed = traj.loc[traj["h_obs"] <= h_fiedler]
        if len(crossed) > 0:
            first = crossed.iloc[0]
            print(f"  {dyad} ({dyad_types[dyad]}): ronda {int(first['Round']):2d}  "
                  f"(h = {first['h_obs']:.4f})")
        else:
            print(f"  {dyad} ({dyad_types[dyad]}): NO cruza")

    # --- Figuras ---
    outdir_fig = ROOT / "figures"
    outdir_fig.mkdir(parents=True, exist_ok=True)
    print("\n  Generando figuras...")
    plot_dynamics(temporal_df, dyad_types, h_fiedler, h_optimal, outdir_fig)

    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
