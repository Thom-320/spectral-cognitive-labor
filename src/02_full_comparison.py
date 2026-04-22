#!/usr/bin/env python3
"""
02_full_comparison.py
=====================
Analisis primario de las diadas SODCL sobre humans_only_absent.csv.

Objetivos:
1. Construir un set primario de diadas con particion dura valida.
2. Auditar explicitamente las 45 diadas y documentar exclusiones.
3. Comparar h(S_obs) contra Fiedler ingenuo y baselines axis-aligned.
4. Integrar metricas originales del paper (DLIndex) con la lente espectral.
5. Guardar un CSV principal listo para manuscrito y figuras.
"""

from pathlib import Path
from collections import deque

ROOT = Path(__file__).parent.parent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


MIN_ROUND = 40
MIN_PARTITION_SIZE = 10
MAX_PARTITION_SIZE = 54
ORIENT_MIN_SCORE = 0.25
ORIENT_MIN_RATIO = 1.15
AXIS_TOL = 1e-9


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
    """Calcula L = D - A y el vector de grados."""
    degrees = np.sum(A, axis=1)
    return np.diag(degrees) - A, degrees


def conductance(S, A, degrees):
    """h(S) = cut(S, S_bar) / min(vol(S), vol(S_bar))."""
    S = set(S)
    S_bar = set(range(len(degrees))) - S

    if len(S) == 0 or len(S_bar) == 0:
        return np.nan

    cut = sum(A[u, v] for u in S for v in S_bar)
    vol_S = sum(degrees[list(S)])
    vol_S_bar = sum(degrees[list(S_bar)])
    denom = min(vol_S, vol_S_bar)

    if denom == 0:
        return np.nan

    return float(cut / denom)


def is_connected(S, A):
    """Verifica conectividad del subgrafo inducido por S."""
    if len(S) == 0:
        return False

    nodes = list(S)
    visited = {nodes[0]}
    queue = deque([nodes[0]])

    while queue:
        node = queue.popleft()
        for neighbor in nodes:
            if neighbor not in visited and A[node, neighbor] > 0:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(nodes)


def axis_partition(kind, n=8):
    """Construye una particion axial fija."""
    if kind == "LR":
        return {i * n + j for i in range(n) for j in range(n // 2)}
    if kind == "TB":
        return {i * n + j for i in range(n // 2) for j in range(n)}
    raise ValueError(f"Particion axial desconocida: {kind}")


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
            "Type": "MIXED",
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
        "Type": orientation,
        "corr_lr": corr_lr,
        "corr_tb": corr_tb,
        "dominant_score": dominant_score,
        "score_ratio": score_ratio,
    }


def mutual_information(freq1, freq2):
    """Informacion mutua I(Player; Cell) en bits."""
    total = freq1.sum() + freq2.sum()
    if total == 0:
        return 0.0

    p_joint = np.zeros((2, 64))
    p_joint[0] = freq1 / total
    p_joint[1] = freq2 / total

    p_x = np.array([freq1.sum() / total, freq2.sum() / total])
    p_c = (freq1 + freq2) / total

    mi = 0.0
    for x in range(2):
        for c in range(64):
            if p_joint[x, c] > 0 and p_x[x] > 0 and p_c[c] > 0:
                mi += p_joint[x, c] * np.log2(p_joint[x, c] / (p_x[x] * p_c[c]))
    return float(mi)


def jensen_shannon_divergence(freq1, freq2):
    """Divergencia de Jensen-Shannon entre distribuciones de visita."""
    sum1 = freq1.sum()
    sum2 = freq2.sum()
    if sum1 == 0 or sum2 == 0:
        return 0.0

    p1 = freq1 / sum1
    p2 = freq2 / sum2
    mean_dist = 0.5 * (p1 + p2)

    def kl_divergence(p, q):
        kl_value = 0.0
        for i in range(len(p)):
            if p[i] > 0 and q[i] > 0:
                kl_value += p[i] * np.log2(p[i] / q[i])
        return kl_value

    return float(
        0.5 * kl_divergence(p1, mean_dist) + 0.5 * kl_divergence(p2, mean_dist)
    )


def summarize_behavioral_columns(late_df):
    """Resume columnas conductuales del dataset oficial."""
    round_level = late_df.groupby("Round").agg(
        DLIndex=("DLIndex", "first"),
        Joint=("Joint", "first"),
    )

    categories = sorted(set(late_df["Category"].dropna()))
    category_mode = late_df["Category"].mode()

    return {
        "late_rounds": int(late_df["Round"].nunique()),
        "late_rows": int(len(late_df)),
        "DLIndex_mean": float(round_level["DLIndex"].mean()),
        "Joint_mean": float(round_level["Joint"].mean()),
        "Similarity_mean": float(late_df["Similarity"].mean()),
        "Consistency_mean": float(late_df["Consistency"].mean()),
        "Size_visited_mean": float(late_df["Size_visited"].mean()),
        "late_category_mode": category_mode.iloc[0] if len(category_mode) > 0 else "NA",
        "late_categories": ",".join(categories) if categories else "NA",
    }


def summarize_dyad(
    dyad,
    dyad_df,
    A,
    degrees,
    h_fiedler,
    h_lr,
    h_tb,
    h_best_axis,
    lr_template,
    tb_template,
):
    """Construye una fila de auditoria y, si aplica, una fila del analisis primario."""
    late_df = dyad_df[dyad_df["Round"] >= MIN_ROUND].copy()
    audit_row = {"Dyad": dyad}

    if len(late_df) == 0:
        audit_row.update(
            {
                "included_primary": False,
                "exclusion_reason": "sin_rondas_absent_ge_40",
            }
        )
        return audit_row, None

    behavior = summarize_behavioral_columns(late_df)
    audit_row.update(behavior)

    player_ids = sorted(late_df["Player"].unique())
    audit_row["player_count"] = int(len(player_ids))
    if len(player_ids) != 2:
        audit_row.update(
            {
                "included_primary": False,
                "exclusion_reason": "conteo_de_jugadores_invalido",
            }
        )
        return audit_row, None

    a_cols = [f"a{i}{j}" for i in range(1, 9) for j in range(1, 9)]
    freq1 = late_df[late_df["Player"] == player_ids[0]][a_cols].to_numpy(dtype=float).sum(axis=0)
    freq2 = late_df[late_df["Player"] == player_ids[1]][a_cols].to_numpy(dtype=float).sum(axis=0)

    margin_flat = freq1 - freq2
    ties = int(np.sum(margin_flat == 0))
    S_obs = set(np.where(margin_flat > 0)[0])
    partition_size = len(S_obs)

    audit_row["partition_size"] = partition_size
    audit_row["tie_count"] = ties

    if partition_size < MIN_PARTITION_SIZE:
        audit_row.update(
            {
                "included_primary": False,
                "exclusion_reason": "particion_muy_pequena",
            }
        )
        return audit_row, None

    if partition_size > MAX_PARTITION_SIZE:
        audit_row.update(
            {
                "included_primary": False,
                "exclusion_reason": "particion_muy_grande",
            }
        )
        return audit_row, None

    h_obs = conductance(S_obs, A, degrees)
    if not np.isfinite(h_obs) or h_obs == 0:
        audit_row.update(
            {
                "included_primary": False,
                "exclusion_reason": "conductancia_invalida",
            }
        )
        return audit_row, None

    S_bar = set(range(64)) - S_obs
    margin_grid = margin_flat.reshape(8, 8)
    orientation = orientation_from_margin(margin_grid, lr_template, tb_template)
    mi = mutual_information(freq1, freq2)
    jsd = jensen_shannon_divergence(freq1, freq2)

    conn_s = is_connected(S_obs, A)
    conn_sbar = is_connected(S_bar, A)
    eta = h_fiedler / h_obs
    delta_vs_fiedler = h_obs - h_fiedler
    delta_vs_best_axis = h_obs - h_best_axis

    if abs(delta_vs_best_axis) <= AXIS_TOL:
        baseline_status = "alcanza_axis"
    elif h_obs < h_fiedler:
        baseline_status = "mejora_fiedler_sin_alcanzar_axis"
    else:
        baseline_status = "no_mejora_fiedler"

    primary_row = {
        "Dyad": dyad,
        "Type": orientation["Type"],
        "S_obs_size": partition_size,
        "tie_count": ties,
        "late_rounds": behavior["late_rounds"],
        "late_rows": behavior["late_rows"],
        "h_obs": h_obs,
        "h_fiedler_naive": h_fiedler,
        "h_lr_baseline": h_lr,
        "h_tb_baseline": h_tb,
        "h_best_axis_baseline": h_best_axis,
        "eta": eta,
        "delta_vs_fiedler_naive": delta_vs_fiedler,
        "delta_vs_best_axis": delta_vs_best_axis,
        "baseline_status": baseline_status,
        "connected_S": conn_s,
        "connected_Sbar": conn_sbar,
        "connected_both": conn_s and conn_sbar,
        "corr_lr": orientation["corr_lr"],
        "corr_tb": orientation["corr_tb"],
        "dominant_score": orientation["dominant_score"],
        "score_ratio": orientation["score_ratio"],
        "MI": mi,
        "JSD": jsd,
        "DLIndex_mean": behavior["DLIndex_mean"],
        "Joint_mean": behavior["Joint_mean"],
        "Similarity_mean": behavior["Similarity_mean"],
        "Consistency_mean": behavior["Consistency_mean"],
        "Size_visited_mean": behavior["Size_visited_mean"],
        "late_category_mode": behavior["late_category_mode"],
        "late_categories": behavior["late_categories"],
    }

    audit_row.update(
        {
            "included_primary": True,
            "exclusion_reason": "incluida",
            "Type": orientation["Type"],
            "h_obs": h_obs,
            "eta": eta,
            "baseline_status": baseline_status,
            "DLIndex_mean": behavior["DLIndex_mean"],
        }
    )

    return audit_row, primary_row


def mann_whitney_report(df, left_mask, right_mask, column):
    """Ejecuta Mann-Whitney entre dos grupos cuando es posible."""
    left = df.loc[left_mask, column].dropna()
    right = df.loc[right_mask, column].dropna()

    if len(left) < 2 or len(right) < 2:
        return np.nan, np.nan

    statistic, p_value = stats.mannwhitneyu(left, right, alternative="two-sided")
    return statistic, p_value


def build_summary_figure(results_df, h_fiedler, h_best_axis, out_path):
    """Genera la figura resumen del analisis primario."""
    grouped = results_df.copy()
    grouped["Group"] = np.where(grouped["Type"] == "MIXED", "Mixed", "Axial")

    axial = grouped[grouped["Group"] == "Axial"]
    mixed = grouped[grouped["Group"] == "Mixed"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1 = axes[0]
    data = [axial["h_obs"].values, mixed["h_obs"].values]
    box = ax1.boxplot(data, tick_labels=["Axial", "Mixed"], patch_artist=True, widths=0.6)
    for patch, color in zip(box["boxes"], ["#4c78a8", "#e45756"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax1.axhline(y=h_fiedler, color="blue", linestyle="--", linewidth=2, label="Fiedler ingenuo")
    ax1.axhline(y=h_best_axis, color="green", linestyle=":", linewidth=2, label="Best axis-aligned")
    ax1.set_ylabel("Conductancia h(S_obs)")
    ax1.set_title("Conductancia humana vs baselines")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = axes[1]
    if len(axial) > 0:
        ax2.scatter(
            axial["DLIndex_mean"],
            axial["eta"],
            s=70,
            color="#4c78a8",
            edgecolors="black",
            label="Axial",
        )
    if len(mixed) > 0:
        ax2.scatter(
            mixed["DLIndex_mean"],
            mixed["eta"],
            s=70,
            color="#e45756",
            edgecolors="black",
            label="Mixed",
        )
    rho, p_value = stats.spearmanr(grouped["DLIndex_mean"], grouped["eta"])
    ax2.text(
        0.03,
        0.97,
        f"Spearman rho = {rho:.3f}\np = {p_value:.2e}",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9},
    )
    ax2.axhline(y=1.0, color="blue", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("DLIndex medio")
    ax2.set_ylabel("eta = h(Fiedler ingenuo) / h(S_obs)")
    ax2.set_title("Lente espectral vs medida original")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("ANALISIS PRIMARIO DE DIADAS SODCL")
    print("=" * 70)

    out_data = ROOT / "data" / "results"
    out_fig = ROOT / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    csv_path = ROOT / "data" / "raw" / "humans_only_absent.csv"
    if not csv_path.exists():
        print("ERROR: No se encuentra humans_only_absent.csv")
        return

    df = pd.read_csv(csv_path)
    print(f"  Filas: {len(df)} | Diadas: {df['Dyad'].nunique()}")

    A = build_grid_graph(8)
    L, degrees = compute_laplacian(A)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    fiedler_vector = eigenvectors[:, idx[1]]
    S_fiedler = set(np.where(fiedler_vector >= np.median(fiedler_vector))[0])

    h_fiedler = conductance(S_fiedler, A, degrees)
    h_lr = conductance(axis_partition("LR"), A, degrees)
    h_tb = conductance(axis_partition("TB"), A, degrees)
    h_best_axis = min(h_lr, h_tb)
    lr_template, tb_template = build_orientation_templates()

    print(f"  h(Fiedler ingenuo) = {h_fiedler:.6f}")
    print(f"  h(LR)              = {h_lr:.6f}")
    print(f"  h(TB)              = {h_tb:.6f}")

    np.savez(
        out_data / "spectral_results.npz",
        lambda2=eigenvalues[1],
        h_fiedler=h_fiedler,
        h_lr=h_lr,
        h_tb=h_tb,
        h_best_axis=h_best_axis,
        S_fiedler=np.array(sorted(S_fiedler)),
        fiedler_vector=fiedler_vector,
        eigenvalues=eigenvalues,
    )

    audit_rows = []
    primary_rows = []

    for dyad, dyad_df in df.groupby("Dyad"):
        audit_row, primary_row = summarize_dyad(
            dyad,
            dyad_df,
            A,
            degrees,
            h_fiedler,
            h_lr,
            h_tb,
            h_best_axis,
            lr_template,
            tb_template,
        )
        audit_rows.append(audit_row)
        if primary_row is not None:
            primary_rows.append(primary_row)

    audit_df = pd.DataFrame(audit_rows).sort_values("Dyad").reset_index(drop=True)
    results_df = pd.DataFrame(primary_rows).sort_values(["Type", "h_obs", "Dyad"]).reset_index(drop=True)

    audit_path = out_data / "spectral_analysis_audit.csv"
    results_path = out_data / "spectral_comparison_results.csv"
    audit_df.to_csv(audit_path, index=False)
    results_df.to_csv(results_path, index=False)

    figure_path = out_fig / "spectral_comparison_summary.png"
    build_summary_figure(results_df, h_fiedler, h_best_axis, figure_path)

    print("\n" + "=" * 70)
    print("AUDITORIA DE MUESTRA")
    print("=" * 70)
    status_counts = audit_df["exclusion_reason"].value_counts()
    print(status_counts.to_string())

    grouped = results_df.copy()
    grouped["Group"] = np.where(grouped["Type"] == "MIXED", "Mixed", "Axial")
    print("\nTipo geometrico en set primario:")
    print(results_df["Type"].value_counts().to_string())

    summary = (
        grouped.groupby("Group")[
            ["h_obs", "eta", "DLIndex_mean", "Joint_mean", "Similarity_mean", "Size_visited_mean", "MI", "JSD"]
        ]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print("\nResumen por grupo:")
    print(summary.to_string())

    axial_mask = grouped["Group"] == "Axial"
    mixed_mask = grouped["Group"] == "Mixed"
    print("\nComparacion axial vs mixed (Mann-Whitney):")
    for column in ["h_obs", "DLIndex_mean", "MI", "JSD"]:
        statistic, p_value = mann_whitney_report(grouped, axial_mask, mixed_mask, column)
        print(f"  {column}: U = {statistic:.3f}, p = {p_value:.6f}")

    rho_eta_dl, p_eta_dl = stats.spearmanr(grouped["eta"], grouped["DLIndex_mean"])
    rho_h_dl, p_h_dl = stats.spearmanr(grouped["h_obs"], grouped["DLIndex_mean"])

    print("\nNovelty gate:")
    print(f"  Spearman eta vs DLIndex: rho = {rho_eta_dl:.3f}, p = {p_eta_dl:.3e}")
    print(f"  Spearman h_obs vs DLIndex: rho = {rho_h_dl:.3f}, p = {p_h_dl:.3e}")
    print("  Nota: LR/TB y best axis-aligned coinciden numericamente en P8 x P8.")

    print("\nEstado frente a baselines:")
    print(results_df["baseline_status"].value_counts().to_string())

    print(f"\nGuardado: {audit_path}")
    print(f"Guardado: {results_path}")
    print(f"Guardado: {figure_path}")


if __name__ == "__main__":
    main()
