#!/usr/bin/env python3
"""
06_partition_robustness.py
==========================
Analisis de robustez para la particion observada S_obs.

Objetivos:
1. Comparar ventanas temporales 30-60, 40-60 y 50-60.
2. Tratar empates de forma explicita.
3. Medir especializacion suave con el margen m(v) = f1(v) - f2(v).
4. Derivar orientacion geometrica desde el campo de dominancia, sin usar Category.
5. Cuantificar estabilidad con indice de Jaccard entre ventanas.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import deque


WINDOWS = [(30, 60), (40, 60), (50, 60)]
MIN_PARTITION_SIZE = 10
MAX_PARTITION_SIZE = 54
LOW_MARGIN_TAU = 1
ORIENT_MIN_SCORE = 0.25
ORIENT_MIN_RATIO = 1.15


def build_grid_graph(n=8):
    """Construye la grilla king n x n."""
    num_nodes = n * n
    A = np.zeros((num_nodes, num_nodes))

    for i in range(n):
        for j in range(n):
            node = i * n + j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        neighbor = ni * n + nj
                        A[node, neighbor] = 1

    return A


def compute_laplacian(A):
    """Calcula L = D - A."""
    degrees = np.sum(A, axis=1)
    return np.diag(degrees) - A, degrees


def compute_normalized_laplacian(A):
    """Calcula L_norm = D^(-1/2) L D^(-1/2)."""
    L, degrees = compute_laplacian(A)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    return D_inv_sqrt @ L @ D_inv_sqrt


def conductance(S, A, degrees):
    """h(S) = cut(S, S_bar) / min(vol(S), vol(S_bar))."""
    S = set(S)
    S_bar = set(range(len(degrees))) - S

    if len(S) == 0 or len(S_bar) == 0:
        return np.nan

    cut = sum(A[u, v] for u in S for v in S_bar)
    vol_S = sum(degrees[list(S)])
    vol_S_bar = sum(degrees[list(S_bar)])

    return cut / min(vol_S, vol_S_bar)


def is_connected(S, A):
    """Verifica conectividad del subgrafo inducido por S."""
    if len(S) == 0:
        return False

    S = list(S)
    visited = {S[0]}
    queue = deque([S[0]])

    while queue:
        node = queue.popleft()
        for neighbor in S:
            if neighbor not in visited and A[node, neighbor] > 0:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(S)


def jaccard_index(S_a, S_b):
    """Indice de Jaccard entre dos particiones duras."""
    union = len(S_a | S_b)
    if union == 0:
        return np.nan
    return len(S_a & S_b) / union


def external_category_label(df, dyad, min_round=40):
    """Etiqueta externa solo para comparacion diagnostica."""
    late = df[(df["Dyad"] == dyad) & (df["Round"] >= min_round)]
    cats = set(late["Category"].unique())

    if {"LEFT", "RIGHT"} <= cats:
        return "LR"
    if {"TOP", "BOTTOM"} <= cats:
        return "TB"
    return "MIXED"


def build_orientation_templates():
    """Construye plantillas LR y TB sobre la grilla 8x8."""
    x_coords = np.tile(np.arange(8), 8).reshape(8, 8)
    y_coords = np.repeat(np.arange(8), 8).reshape(8, 8)

    lr_template = x_coords - x_coords.mean()
    tb_template = y_coords.mean() - y_coords

    lr_template = lr_template / np.linalg.norm(lr_template)
    tb_template = tb_template / np.linalg.norm(tb_template)

    return lr_template, tb_template


def orientation_from_margin(margin_grid, lr_template, tb_template):
    """Clasifica orientacion desde el campo firmado de dominancia."""
    margin_norm = np.linalg.norm(margin_grid)
    if margin_norm == 0:
        return {
            "orientation": "MIXED",
            "corr_lr": 0.0,
            "corr_tb": 0.0,
            "dominant_score": 0.0,
            "score_ratio": 0.0,
        }

    normalized_margin = margin_grid / margin_norm
    corr_lr = float(np.sum(normalized_margin * lr_template))
    corr_tb = float(np.sum(normalized_margin * tb_template))

    abs_lr = abs(corr_lr)
    abs_tb = abs(corr_tb)
    dominant_score = max(abs_lr, abs_tb)
    secondary_score = min(abs_lr, abs_tb)
    score_ratio = dominant_score / (secondary_score + 1e-12)

    if dominant_score < ORIENT_MIN_SCORE or score_ratio < ORIENT_MIN_RATIO:
        orientation = "MIXED"
    else:
        orientation = "LR" if abs_lr >= abs_tb else "TB"

    return {
        "orientation": orientation,
        "corr_lr": corr_lr,
        "corr_tb": corr_tb,
        "dominant_score": dominant_score,
        "score_ratio": score_ratio,
    }


def hard_partition_from_margin(margin_flat):
    """Construye la particion dura a partir del signo del margen."""
    return set(np.where(margin_flat > 0)[0])


def soft_specialization(f1_flat, f2_flat):
    """Mide especializacion suave usando la masa del margen."""
    total = np.sum(f1_flat + f2_flat)
    if total == 0:
        return np.nan
    return float(np.sum(np.abs(f1_flat - f2_flat)) / total)


def extract_window_counts(dyad_df, player_ids, a_cols, start_round, end_round):
    """Acumula visitas por jugador en una ventana temporal."""
    counts = []

    for player_id in player_ids:
        player_df = dyad_df[
            (dyad_df["Player"] == player_id)
            & (dyad_df["Round"] >= start_round)
            & (dyad_df["Round"] <= end_round)
        ]
        counts.append(player_df[a_cols].to_numpy(dtype=float).sum(axis=0))

    return counts[0], counts[1]


def compute_window_metrics(
    dyad,
    dyad_df,
    player_ids,
    start_round,
    end_round,
    A,
    degrees,
    lr_template,
    tb_template,
):
    """Calcula metricas de una ventana para una diada."""
    a_cols = [f"a{i}{j}" for i in range(1, 9) for j in range(1, 9)]
    f1_flat, f2_flat = extract_window_counts(
        dyad_df, player_ids, a_cols, start_round, end_round
    )

    margin_flat = f1_flat - f2_flat
    margin_grid = margin_flat.reshape(8, 8)
    S_obs = hard_partition_from_margin(margin_flat)
    S_bar = set(range(64)) - S_obs

    partition_size = len(S_obs)
    tie_count = int(np.sum(margin_flat == 0))
    low_margin_count = int(np.sum(np.abs(margin_flat) <= LOW_MARGIN_TAU))
    valid_partition = MIN_PARTITION_SIZE <= partition_size <= MAX_PARTITION_SIZE

    orientation_info = orientation_from_margin(margin_grid, lr_template, tb_template)
    h_obs = conductance(S_obs, A, degrees) if valid_partition else np.nan

    connected = False
    if valid_partition:
        connected = is_connected(S_obs, A) and is_connected(S_bar, A)

    return {
        "Dyad": dyad,
        "window_start": start_round,
        "window_end": end_round,
        "partition_size": partition_size,
        "tie_count": tie_count,
        "low_margin_count": low_margin_count,
        "soft_specialization": soft_specialization(f1_flat, f2_flat),
        "h_obs": h_obs,
        "connected": connected,
        "valid_partition": valid_partition,
        "orientation": orientation_info["orientation"],
        "corr_lr": orientation_info["corr_lr"],
        "corr_tb": orientation_info["corr_tb"],
        "dominant_score": orientation_info["dominant_score"],
        "score_ratio": orientation_info["score_ratio"],
        "S_obs": S_obs,
    }


def stable_orientation(window_orientations):
    """Resume la orientacion geometrica a traves de ventanas."""
    labels = [label for label in window_orientations if label in {"LR", "TB", "MIXED"}]
    if not labels:
        return "INVALID"

    counts = pd.Series(labels).value_counts()
    if len(counts) == 1:
        return counts.index[0]

    top_label = counts.index[0]
    top_count = counts.iloc[0]

    if top_label in {"LR", "TB"} and top_count >= 2:
        return top_label

    if top_label == "MIXED":
        return "MIXED"

    return "UNSTABLE"


def summarize_stability(window_metrics, external_label):
    """Construye resumen por diada a traves de ventanas."""
    valid_metrics = window_metrics[window_metrics["valid_partition"]].copy()

    if len(valid_metrics) == 0:
        return {
            "Dyad": window_metrics["Dyad"].iloc[0],
            "orientation_30_60": window_metrics.loc[window_metrics["window_start"] == 30, "orientation"].iloc[0],
            "orientation_40_60": window_metrics.loc[window_metrics["window_start"] == 40, "orientation"].iloc[0],
            "orientation_50_60": window_metrics.loc[window_metrics["window_start"] == 50, "orientation"].iloc[0],
            "stable_orientation": "INVALID",
            "external_label_40_60": external_label,
            "agrees_with_external": False,
            "mean_h_obs": np.nan,
            "h_obs_40_60": np.nan,
            "mean_soft_specialization": window_metrics["soft_specialization"].mean(),
            "soft_specialization_40_60": window_metrics.loc[window_metrics["window_start"] == 40, "soft_specialization"].iloc[0],
            "mean_tie_count": window_metrics["tie_count"].mean(),
            "tie_count_40_60": window_metrics.loc[window_metrics["window_start"] == 40, "tie_count"].iloc[0],
            "valid_windows": 0,
            "mean_jaccard": np.nan,
            "min_jaccard": np.nan,
            "jaccard_30-60_vs_40-60": np.nan,
            "jaccard_40-60_vs_50-60": np.nan,
            "jaccard_30-60_vs_50-60": np.nan,
        }

    by_window = {
        f"{int(row['window_start'])}-{int(row['window_end'])}": row
        for _, row in valid_metrics.iterrows()
    }

    windows_present = sorted(by_window.keys())
    partitions = {name: by_window[name]["S_obs"] for name in windows_present}

    jaccards = {}
    for a, b in [("30-60", "40-60"), ("40-60", "50-60"), ("30-60", "50-60")]:
        if a in partitions and b in partitions:
            jaccards[f"jaccard_{a}_vs_{b}"] = jaccard_index(partitions[a], partitions[b])
        else:
            jaccards[f"jaccard_{a}_vs_{b}"] = np.nan

    orientation_labels = [by_window[name]["orientation"] for name in windows_present]
    stable_label = stable_orientation(orientation_labels)

    finite_jaccards = [value for value in jaccards.values() if np.isfinite(value)]
    mean_jaccard = float(np.mean(finite_jaccards)) if finite_jaccards else np.nan
    min_jaccard = float(np.min(finite_jaccards)) if finite_jaccards else np.nan

    summary = {
        "Dyad": window_metrics["Dyad"].iloc[0],
        "orientation_30_60": window_metrics.loc[window_metrics["window_start"] == 30, "orientation"].iloc[0],
        "orientation_40_60": window_metrics.loc[window_metrics["window_start"] == 40, "orientation"].iloc[0],
        "orientation_50_60": window_metrics.loc[window_metrics["window_start"] == 50, "orientation"].iloc[0],
        "stable_orientation": stable_label,
        "external_label_40_60": external_label,
        "agrees_with_external": stable_label == external_label if stable_label in {"LR", "TB", "MIXED"} else False,
        "mean_h_obs": valid_metrics["h_obs"].mean(),
        "h_obs_40_60": window_metrics.loc[window_metrics["window_start"] == 40, "h_obs"].iloc[0],
        "mean_soft_specialization": window_metrics["soft_specialization"].mean(),
        "soft_specialization_40_60": window_metrics.loc[window_metrics["window_start"] == 40, "soft_specialization"].iloc[0],
        "mean_tie_count": window_metrics["tie_count"].mean(),
        "tie_count_40_60": window_metrics.loc[window_metrics["window_start"] == 40, "tie_count"].iloc[0],
        "valid_windows": int(valid_metrics["valid_partition"].sum()),
        "mean_jaccard": mean_jaccard,
        "min_jaccard": min_jaccard,
    }

    summary.update(jaccards)
    return summary


def mann_whitney_report(summary_df, left_mask, right_mask, column):
    """Ejecuta Mann-Whitney entre dos grupos cuando es posible."""
    left = summary_df.loc[left_mask, column].dropna()
    right = summary_df.loc[right_mask, column].dropna()

    if len(left) < 2 or len(right) < 2:
        return np.nan, np.nan

    stat, p_value = stats.mannwhitneyu(left, right, alternative="two-sided")
    return stat, p_value


def plot_summary(summary_df, window_df, out_path):
    """Genera figura resumen del analisis de robustez."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    color_map = {
        "LR": "#1f77b4",
        "TB": "#2ca02c",
        "MIXED": "#d62728",
        "UNSTABLE": "#ff7f0e",
        "INVALID": "#7f7f7f",
    }

    base_40 = window_df[window_df["window_start"] == 40].copy()
    stable_lookup = summary_df.set_index("Dyad")["stable_orientation"].to_dict()
    base_40["stable_orientation"] = base_40["Dyad"].map(stable_lookup)

    ax1 = axes[0]
    for label, subset in base_40.groupby("stable_orientation"):
        ax1.scatter(
            subset["corr_lr"].abs(),
            subset["corr_tb"].abs(),
            s=60,
            alpha=0.8,
            color=color_map.get(label, "#7f7f7f"),
            label=label,
        )
    ax1.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax1.set_xlabel("|corr(LR template)|")
    ax1.set_ylabel("|corr(TB template)|")
    ax1.set_title("Orientacion geometrica derivada del margen")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    ax2 = axes[1]
    axial = summary_df[summary_df["stable_orientation"].isin(["LR", "TB"])]["mean_jaccard"].dropna()
    mixed = summary_df[summary_df["stable_orientation"] == "MIXED"]["mean_jaccard"].dropna()
    data = [axial.values, mixed.values]
    labels = ["Axial", "Mixed"]
    box = ax2.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(box["boxes"], ["#4c78a8", "#e45756"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax2.set_ylabel("Mean Jaccard across windows")
    ax2.set_title("Estabilidad temporal de la particion")
    ax2.grid(axis="y", alpha=0.25)

    ax3 = axes[2]
    for label, subset in summary_df.groupby("stable_orientation"):
        ax3.scatter(
            subset["soft_specialization_40_60"],
            subset["h_obs_40_60"],
            s=60,
            alpha=0.8,
            color=color_map.get(label, "#7f7f7f"),
            label=label,
        )
    ax3.set_xlabel("Soft specialization (40-60)")
    ax3.set_ylabel("Conductance h(S_obs)")
    ax3.set_title("Especializacion suave vs conductancia")
    ax3.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("ROBUSTEZ DE S_obs Y ORIENTACION GEOMETRICA")
    print("=" * 70)

    csv_path = ROOT / "data" / "raw" / "humans_only_absent.csv"
    df = pd.read_csv(csv_path)
    print(f"  Filas: {len(df)} | Diadas: {df['Dyad'].nunique()}")

    A = build_grid_graph(8)
    _, degrees = compute_laplacian(A)
    lr_template, tb_template = build_orientation_templates()

    window_rows = []
    summary_rows = []

    for dyad, dyad_df in df.groupby("Dyad"):
        player_ids = sorted(dyad_df["Player"].unique())
        if len(player_ids) != 2:
            continue

        local_rows = []
        for start_round, end_round in WINDOWS:
            row = compute_window_metrics(
                dyad,
                dyad_df,
                player_ids,
                start_round,
                end_round,
                A,
                degrees,
                lr_template,
                tb_template,
            )
            local_rows.append(row)

            clean_row = row.copy()
            clean_row.pop("S_obs")
            window_rows.append(clean_row)

        local_df = pd.DataFrame(local_rows)
        summary_rows.append(
            summarize_stability(
                local_df,
                external_category_label(df, dyad, min_round=40),
            )
        )

    window_df = pd.DataFrame(window_rows)
    summary_df = pd.DataFrame(summary_rows)

    out_data = ROOT / "data" / "results"
    out_fig = ROOT / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    window_df.to_csv(out_data / "partition_window_metrics.csv", index=False)
    summary_df.to_csv(out_data / "partition_stability_summary.csv", index=False)

    fig_path = out_fig / "partition_robustness_summary.png"
    plot_summary(summary_df, window_df, fig_path)

    print("\nResumen por orientacion estable:")
    valid_summary = summary_df[summary_df["stable_orientation"] != "INVALID"].copy()
    grouped = (
        valid_summary.groupby("stable_orientation")[
            ["h_obs_40_60", "soft_specialization_40_60", "mean_jaccard", "tie_count_40_60"]
        ]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print(grouped.to_string())

    axial_mask = summary_df["stable_orientation"].isin(["LR", "TB"])
    mixed_mask = summary_df["stable_orientation"] == "MIXED"

    print("\nComparacion axial vs mixed (Mann-Whitney):")
    for column in ["h_obs_40_60", "soft_specialization_40_60", "mean_jaccard"]:
        stat, p_value = mann_whitney_report(summary_df, axial_mask, mixed_mask, column)
        print(f"  {column}: U = {stat:.3f}, p = {p_value:.6f}")

    agreement = valid_summary["agrees_with_external"].mean()
    print(f"\nAcuerdo con etiqueta externa (diagnostico, 40-60): {agreement:.3f}")

    unstable = valid_summary.sort_values("mean_jaccard").head(10)
    print("\nDiadas menos estables entre ventanas:")
    print(
        unstable[
            [
                "Dyad",
                "stable_orientation",
                "h_obs_40_60",
                "soft_specialization_40_60",
                "mean_jaccard",
                "tie_count_40_60",
            ]
        ].to_string(index=False)
    )

    print(f"\nGuardado: {out_data / 'partition_window_metrics.csv'}")
    print(f"Guardado: {out_data / 'partition_stability_summary.csv'}")
    print(f"Guardado: {fig_path}")


if __name__ == "__main__":
    main()
