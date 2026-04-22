#!/usr/bin/env python3
"""
08_early_prediction.py
======================
Prediccion temprana de especializacion axial en SODCL.

Objetivos:
1. Construir una senal geometrica temprana con las primeras 5 rondas absent comunes.
2. Compararla contra metricas tempranas ya presentes en el dataset.
3. Medir transferencia hacia desempeno posterior en performances.csv.
4. Reportar un diagnostico de valor incremental para h_obs tardia.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EARLY_COMMON_ROUNDS = 5
PRIMARY_MIN_ROUND = 40


def build_grid_graph(n=8):
    """Construye la grilla king n x n."""
    num_nodes = n * n
    adjacency = np.zeros((num_nodes, num_nodes))

    for i in range(n):
        for j in range(n):
            node = i * n + j
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        adjacency[node, ni * n + nj] = 1

    return adjacency


def conductance(partition, adjacency, degrees):
    """Calcula h(S) para una particion dura."""
    partition = set(partition)
    complement = set(range(len(degrees))) - partition

    if len(partition) == 0 or len(complement) == 0:
        return np.nan

    cut_size = sum(adjacency[u, v] for u in partition for v in complement)
    vol_partition = sum(degrees[list(partition)])
    vol_complement = sum(degrees[list(complement)])
    denom = min(vol_partition, vol_complement)

    if denom == 0:
        return np.nan

    return float(cut_size / denom)


def build_orientation_templates():
    """Construye plantillas LR y TB sobre la grilla 8x8."""
    x_coords = np.tile(np.arange(8), 8).reshape(8, 8)
    y_coords = np.repeat(np.arange(8), 8).reshape(8, 8)

    lr_template = x_coords - x_coords.mean()
    tb_template = y_coords.mean() - y_coords

    lr_template = lr_template / np.linalg.norm(lr_template)
    tb_template = tb_template / np.linalg.norm(tb_template)

    return lr_template, tb_template


def early_rounds_for_dyad(dyad_df):
    """Toma las primeras rondas absent comunes a ambos jugadores."""
    common_rounds = sorted(
        round_id
        for round_id, round_df in dyad_df.groupby("Round")
        if round_df["Player"].nunique() == 2
    )
    return common_rounds[:EARLY_COMMON_ROUNDS]


def early_feature_row(dyad, dyad_df, lr_template, tb_template, adjacency, degrees):
    """Extrae una fila de features tempranos para una diada."""
    player_ids = sorted(dyad_df["Player"].unique())
    early_rounds = early_rounds_for_dyad(dyad_df)
    early_df = dyad_df[dyad_df["Round"].isin(early_rounds)].copy()
    a_cols = [f"a{i}{j}" for i in range(1, 9) for j in range(1, 9)]

    if len(player_ids) != 2 or len(early_rounds) == 0:
        return {
            "Dyad": dyad,
            "early_round_count": int(len(early_rounds)),
            "early_rounds": ",".join(str(r) for r in early_rounds) if early_rounds else "NA",
            "dominant_score": np.nan,
            "axis_gap": np.nan,
            "abs_corr_lr": np.nan,
            "abs_corr_tb": np.nan,
            "score_ratio": np.nan,
            "early_h_obs": np.nan,
            "early_partition_size": np.nan,
            "early_tie_count": np.nan,
            "DLIndex_early": np.nan,
            "Similarity_early": np.nan,
            "Consistency_early": np.nan,
            "Joint_early": np.nan,
            "Size_visited_early": np.nan,
        }

    freq1 = (
        early_df[early_df["Player"] == player_ids[0]][a_cols]
        .to_numpy(dtype=float)
        .sum(axis=0)
    )
    freq2 = (
        early_df[early_df["Player"] == player_ids[1]][a_cols]
        .to_numpy(dtype=float)
        .sum(axis=0)
    )

    margin_flat = freq1 - freq2
    margin_grid = margin_flat.reshape(8, 8)
    norm_margin = np.linalg.norm(margin_grid)

    if norm_margin == 0:
        corr_lr = 0.0
        corr_tb = 0.0
        dominant_score = 0.0
        score_ratio = 0.0
    else:
        normalized_margin = margin_grid / norm_margin
        corr_lr = float(np.sum(normalized_margin * lr_template))
        corr_tb = float(np.sum(normalized_margin * tb_template))
        abs_lr = abs(corr_lr)
        abs_tb = abs(corr_tb)
        dominant_score = max(abs_lr, abs_tb)
        score_ratio = dominant_score / (min(abs_lr, abs_tb) + 1e-12)

    early_partition = set(np.where(margin_flat > 0)[0])
    round_level = early_df.groupby("Round").agg(
        DLIndex=("DLIndex", "first"),
        Joint=("Joint", "first"),
    )

    return {
        "Dyad": dyad,
        "early_round_count": int(len(early_rounds)),
        "early_rounds": ",".join(str(r) for r in early_rounds),
        "dominant_score": dominant_score,
        "axis_gap": abs(abs(corr_lr) - abs(corr_tb)),
        "abs_corr_lr": abs(corr_lr),
        "abs_corr_tb": abs(corr_tb),
        "score_ratio": score_ratio,
        "early_h_obs": conductance(early_partition, adjacency, degrees),
        "early_partition_size": int(len(early_partition)),
        "early_tie_count": int(np.sum(margin_flat == 0)),
        "DLIndex_early": float(round_level["DLIndex"].mean()),
        "Similarity_early": float(early_df["Similarity"].mean()),
        "Consistency_early": float(early_df["Consistency"].mean()),
        "Joint_early": float(round_level["Joint"].mean()),
        "Size_visited_early": float(early_df["Size_visited"].mean()),
    }


def prediction_summary_rows(feature_df, split_name, valid_mask):
    """Evalua AUC in-sample y LOOCV para varios modelos."""
    rows = []
    subset = feature_df.loc[valid_mask].copy()
    y = subset["axial_target"].to_numpy(dtype=int)

    model_features = {
        "geometry_dominant": ["dominant_score"],
        "official_trio": ["DLIndex_early", "Similarity_early", "Consistency_early"],
        "geometry_plus_official": [
            "dominant_score",
            "DLIndex_early",
            "Similarity_early",
            "Consistency_early",
        ],
    }

    for model_name, feature_cols in model_features.items():
        X = subset[feature_cols].to_numpy(dtype=float)
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(max_iter=1000, solver="liblinear")),
            ]
        )
        pipeline.fit(X, y)
        in_sample = float(roc_auc_score(y, pipeline.predict_proba(X)[:, 1]))

        loo = LeaveOneOut()
        probs = []
        for train_idx, test_idx in loo.split(X):
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("logit", LogisticRegression(max_iter=1000, solver="liblinear")),
                ]
            )
            pipeline.fit(X[train_idx], y[train_idx])
            probs.append(float(pipeline.predict_proba(X[test_idx])[:, 1][0]))

        loocv_auc = float(roc_auc_score(y, probs))
        rows.append(
            {
                "split": split_name,
                "model": model_name,
                "feature_set": ",".join(feature_cols),
                "n": int(len(subset)),
                "positives": int(y.sum()),
                "negatives": int(len(y) - y.sum()),
                "auc_in_sample": in_sample,
                "auc_loocv": loocv_auc,
            }
        )

    return rows


def performance_transfer_summary(performance_df, stability_df):
    """Resume desempeno en performances.csv segun orientacion estable."""
    valid = stability_df[stability_df["stable_orientation"].isin(["LR", "TB", "MIXED"])].copy()
    valid["Group"] = valid["stable_orientation"].replace({"LR": "AXIAL", "TB": "AXIAL"})

    merged = performance_df.merge(valid[["Dyad", "Group"]], on="Dyad", how="inner").copy()
    correct = (
        ((merged["Answer"] == "Present") & (merged["Is_there"] == "Unicorn_Present"))
        | ((merged["Answer"] == "Absent") & (merged["Is_there"] == "Unicorn_Absent"))
    ).astype(int)
    merged = merged.assign(correct=correct)

    round_level = merged.groupby(
        ["Dyad", "Round", "Is_there", "Group"], as_index=False
    ).agg(
        Score=("Score", "first"),
        Joint=("Joint", "first"),
        accuracy=("correct", "mean"),
    )

    summary = round_level.groupby(["Is_there", "Group"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        score_mean=("Score", "mean"),
        joint_mean=("Joint", "mean"),
        n_rounds=("Round", "size"),
    )

    dyad_summary = round_level.groupby(["Dyad", "Is_there"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        score_mean=("Score", "mean"),
        joint_mean=("Joint", "mean"),
    )

    return summary, dyad_summary


def incremental_value_summary(spectral_df, dyad_performance_df):
    """Diagnostica si h_obs agrega valor sobre metricas oficiales tardias."""
    present_df = dyad_performance_df[dyad_performance_df["Is_there"] == "Unicorn_Present"].copy()
    merged = spectral_df.merge(present_df, on="Dyad", how="inner")

    models = {
        "official_only": ["DLIndex_mean", "Similarity_mean", "Consistency_mean"],
        "official_plus_h": ["DLIndex_mean", "Similarity_mean", "Consistency_mean", "h_obs"],
        "official_plus_eta": ["DLIndex_mean", "Similarity_mean", "Consistency_mean", "eta"],
    }
    targets = {
        "score_present": "score_mean",
        "acc_present": "accuracy_mean",
        "joint_present": "joint_mean",
    }

    rows = []
    for target_name, target_col in targets.items():
        y = merged[target_col].to_numpy(dtype=float)
        for model_name, feature_cols in models.items():
            X = merged[feature_cols].to_numpy(dtype=float)
            loo = LeaveOneOut()
            predictions = []

            for train_idx, test_idx in loo.split(X):
                pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("linear", LinearRegression()),
                    ]
                )
                pipeline.fit(X[train_idx], y[train_idx])
                predictions.append(float(pipeline.predict(X[test_idx])[0]))

            predictions = np.array(predictions)
            rows.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "feature_set": ",".join(feature_cols),
                    "n": int(len(merged)),
                    "loocv_r2": float(r2_score(y, predictions)),
                    "loocv_mae": float(np.mean(np.abs(y - predictions))),
                }
            )

    return pd.DataFrame(rows)


def build_summary_figure(feature_df, prediction_df, transfer_df, output_path):
    """Construye una figura resumen para la tesis de prediccion temprana."""
    display_df = feature_df.copy()
    display_df["Target"] = display_df["axial_target"].map({1: "Axial estable", 0: "No axial"})

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    groups = ["No axial", "Axial estable"]
    values = [
        display_df.loc[display_df["Target"] == group, "dominant_score"].to_numpy()
        for group in groups
    ]
    box = ax.boxplot(values, tick_labels=groups, patch_artist=True)
    colors = ["#d9d9d9", "#4c78a8"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
    for idx, group in enumerate(groups, start=1):
        y_vals = display_df.loc[display_df["Target"] == group, "dominant_score"].to_numpy()
        x_vals = np.full_like(y_vals, idx, dtype=float) + np.linspace(-0.08, 0.08, len(y_vals))
        ax.scatter(x_vals, y_vals, s=25, alpha=0.75, color="#222222")
    ax.set_title("Senal geometrica temprana")
    ax.set_ylabel("dominant_score")
    ax.grid(alpha=0.2, axis="y")

    ax = axes[0, 1]
    loocv_df = prediction_df[prediction_df["split"] == "all_dyads"].copy()
    labels = {
        "geometry_dominant": "Geometria",
        "official_trio": "Metricas oficiales",
        "geometry_plus_official": "Combinado",
    }
    x_positions = np.arange(len(loocv_df))
    ax.bar(
        x_positions,
        loocv_df["auc_loocv"],
        color=["#4c78a8", "#f58518", "#54a24b"],
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([labels[name] for name in loocv_df["model"]], rotation=0)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC LOOCV")
    ax.set_title("Prediccion de orientacion axial estable")
    for idx, value in enumerate(loocv_df["auc_loocv"]):
        ax.text(idx, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1, 0]
    acc_df = transfer_df.copy()
    acc_pivot = acc_df.pivot(index="Is_there", columns="Group", values="accuracy_mean")
    acc_pivot = acc_pivot.reindex(["Unicorn_Absent", "Unicorn_Present"])
    x_positions = np.arange(len(acc_pivot.index))
    width = 0.35
    ax.bar(x_positions - width / 2, acc_pivot["AXIAL"], width=width, label="Axial", color="#4c78a8")
    ax.bar(x_positions + width / 2, acc_pivot["MIXED"], width=width, label="Mixed", color="#e45756")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Absent", "Present"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy media")
    ax.set_title("Transferencia a desempeno")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1, 1]
    score_pivot = transfer_df.pivot(index="Is_there", columns="Group", values="score_mean")
    score_pivot = score_pivot.reindex(["Unicorn_Absent", "Unicorn_Present"])
    ax.bar(x_positions - width / 2, score_pivot["AXIAL"], width=width, label="Axial", color="#4c78a8")
    ax.bar(x_positions + width / 2, score_pivot["MIXED"], width=width, label="Mixed", color="#e45756")
    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Absent", "Present"])
    ax.set_ylabel("Score medio")
    ax.set_title("Ventaja conductual asociada")
    ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    """Ejecuta el analisis completo."""
    absent_path = ROOT / "data/raw/humans_only_absent.csv"
    performance_path = ROOT / "data/raw/performances.csv"
    stability_path = ROOT / "data/results/partition_stability_summary.csv"
    spectral_path = ROOT / "data/results/spectral_comparison_results.csv"

    absent_df = pd.read_csv(absent_path)
    performance_df = pd.read_csv(performance_path)
    stability_df = pd.read_csv(stability_path)
    spectral_df = pd.read_csv(spectral_path)

    adjacency = build_grid_graph(8)
    degrees = np.sum(adjacency, axis=1)
    lr_template, tb_template = build_orientation_templates()

    feature_rows = []
    for dyad, dyad_df in absent_df.groupby("Dyad"):
        feature_rows.append(
            early_feature_row(
                dyad,
                dyad_df,
                lr_template,
                tb_template,
                adjacency,
                degrees,
            )
        )

    feature_df = pd.DataFrame(feature_rows)
    stability_labels = stability_df[["Dyad", "stable_orientation"]].copy()
    feature_df = feature_df.merge(stability_labels, on="Dyad", how="left")
    feature_df["axial_target"] = feature_df["stable_orientation"].isin(["LR", "TB"]).astype(int)
    feature_df["stable_label_set"] = np.where(
        feature_df["stable_orientation"].isin(["LR", "TB", "MIXED"]),
        "valid",
        "invalid",
    )

    prediction_rows = []
    prediction_rows.extend(
        prediction_summary_rows(
            feature_df,
            split_name="all_dyads",
            valid_mask=feature_df["stable_orientation"].notna(),
        )
    )
    prediction_rows.extend(
        prediction_summary_rows(
            feature_df,
            split_name="valid_only",
            valid_mask=feature_df["stable_orientation"].isin(["LR", "TB", "MIXED"]),
        )
    )
    prediction_df = pd.DataFrame(prediction_rows)

    transfer_df, dyad_transfer_df = performance_transfer_summary(performance_df, stability_df)
    increment_df = incremental_value_summary(spectral_df, dyad_transfer_df)

    feature_output = ROOT / "data/results/early_prediction_features.csv"
    prediction_output = ROOT / "data/results/early_prediction_summary.csv"
    transfer_output = ROOT / "data/results/performance_transfer_summary.csv"
    increment_output = ROOT / "data/results/present_performance_increment.csv"
    figure_output = ROOT / "figures/early_prediction_summary.png"

    feature_df.to_csv(feature_output, index=False)
    prediction_df.to_csv(prediction_output, index=False)
    transfer_df.to_csv(transfer_output, index=False)
    increment_df.to_csv(increment_output, index=False)
    build_summary_figure(feature_df, prediction_df, transfer_df, figure_output)

    print("=" * 70)
    print("PREDICCION TEMPRANA DE ESPECIALIZACION AXIAL")
    print("=" * 70)
    print(f"  Diadas auditadas: {feature_df['Dyad'].nunique()}")
    print(
        f"  Target axial estable: {int(feature_df['axial_target'].sum())} / {len(feature_df)}"
    )
    print()
    print("Resumen AUC:")
    print(
        prediction_df[
            ["split", "model", "auc_in_sample", "auc_loocv", "n", "positives", "negatives"]
        ].round(3).to_string(index=False)
    )
    print()
    print("Transferencia a desempeno:")
    print(transfer_df.round(3).to_string(index=False))
    print()
    print("Valor incremental de h_obs tardia:")
    print(increment_df.round(3).to_string(index=False))
    print()
    print(f"Guardado: {feature_output}")
    print(f"Guardado: {prediction_output}")
    print(f"Guardado: {transfer_output}")
    print(f"Guardado: {increment_output}")
    print(f"Guardado: {figure_output}")


if __name__ == "__main__":
    main()
