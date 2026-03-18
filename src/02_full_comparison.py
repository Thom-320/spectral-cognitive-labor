#!/usr/bin/env python3
"""
Análisis espectral completo de todas las díadas SODCL.
Compara particiones humanas con bisección de Fiedler.

Autor: Thomas Chísica
Fecha: Febrero 2026
"""

from pathlib import Path

# Directorio raíz del proyecto (un nivel arriba de src/)
ROOT = Path(__file__).parent.parent

import numpy as np
import pandas as pd
from scipy import sparse, stats
from collections import deque
import matplotlib.pyplot as plt
import os

# ============================================================
# FUNCIONES BASE
# ============================================================

def build_grid_graph(n=8):
    """Construye grafo de grilla 8x8 con 8-conectividad (incluye diagonales)"""
    num_nodes = n * n
    edges = []
    for i in range(n):
        for j in range(n):
            node = i * n + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        neighbor = ni * n + nj
                        edges.append((node, neighbor))
    
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        A[i, j] = 1
    return A


def compute_laplacian(A):
    """L = D - A (Laplaciano combinatorial)"""
    D = np.diag(np.sum(A, axis=1))
    return D - A


def spectral_analysis(L):
    """Calcula eigenvalores y eigenvectores del Laplaciano"""
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def fiedler_bisection(fiedler_vector, threshold='median'):
    """Bisección usando el vector de Fiedler"""
    if threshold == 'median':
        thresh = np.median(fiedler_vector)
    else:
        thresh = 0
    S = set(np.where(fiedler_vector >= thresh)[0])
    return S


def conductance(S, A, degrees):
    """h(S) = cut(S, S̄) / min(vol(S), vol(S̄))"""
    S_set = set(S)
    S_bar = set(range(len(A))) - S_set
    
    if len(S_set) == 0 or len(S_bar) == 0:
        return float('inf')
    
    cut = 0
    for i in S_set:
        for j in S_bar:
            cut += A[i, j]
    
    vol_S = sum(degrees[i] for i in S_set)
    vol_S_bar = sum(degrees[i] for i in S_bar)
    
    if min(vol_S, vol_S_bar) == 0:
        return float('inf')
    
    return cut / min(vol_S, vol_S_bar)


def is_connected(S, A):
    """Verifica si el subgrafo inducido por S es conexo (BFS)"""
    if len(S) == 0:
        return True
    S_list = list(S)
    n = len(A)
    
    visited = set()
    queue = deque([S_list[0]])
    visited.add(S_list[0])
    
    while queue:
        node = queue.popleft()
        for neighbor in range(n):
            if A[node, neighbor] > 0 and neighbor in S and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(S)


def get_observed_partition(dyad_name, df_all, min_round=40):
    """
    Calcula S_obs basándose en frecuencias de visita.
    S_obs = {celdas donde P1 visitó más que P2}
    """
    dyad_data = df_all[(df_all['Dyad'] == dyad_name) & (df_all['Round'] >= min_round)]
    
    if len(dyad_data) == 0:
        return None, None, None
    
    players = dyad_data['Player'].unique()
    if len(players) != 2:
        return None, None, None
    
    p1, p2 = sorted(players)
    
    freq1 = np.zeros(64)
    freq2 = np.zeros(64)
    
    a_cols = [f'a{i}{j}' for i in range(1,9) for j in range(1,9)]
    
    p1_data = dyad_data[dyad_data['Player'] == p1]
    p2_data = dyad_data[dyad_data['Player'] == p2]
    
    for _, row in p1_data.iterrows():
        for idx, col in enumerate(a_cols):
            if col in row and row[col] == 1:
                freq1[idx] += 1
    
    for _, row in p2_data.iterrows():
        for idx, col in enumerate(a_cols):
            if col in row and row[col] == 1:
                freq2[idx] += 1
    
    S_obs = set()
    for v in range(64):
        if freq1[v] > freq2[v]:
            S_obs.add(v)
    
    return S_obs, freq1, freq2


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*70)
    print("ANÁLISIS ESPECTRAL DE DIVISIÓN DEL TRABAJO COGNITIVO")
    print("="*70)
    
    # --- 1. Análisis del grafo base ---
    print("\n[1] Construyendo grafo 8x8 con 8-conectividad...")
    A = build_grid_graph(8)
    degrees = np.sum(A, axis=1)
    L = compute_laplacian(A)
    eigenvalues, eigenvectors = spectral_analysis(L)
    
    lambda2 = eigenvalues[1]
    fiedler_vector = eigenvectors[:, 1]
    S_fiedler = fiedler_bisection(fiedler_vector)
    h_fiedler = conductance(S_fiedler, A, degrees)
    
    print(f"    λ₂ = {lambda2:.4f}")
    print(f"    h(S_Fiedler) = {h_fiedler:.4f}")
    print(f"    |S_Fiedler| = {len(S_fiedler)}")
    
    # Guardar resultados espectrales
    np.savez(ROOT / 'data' / 'results' / 'spectral_results.npz',
             lambda2=lambda2,
             h_fiedler=h_fiedler,
             S_fiedler=np.array(list(S_fiedler)),
             fiedler_vector=fiedler_vector,
             eigenvalues=eigenvalues)
    print("    Guardado: spectral_results.npz")
    
    # --- 2. Cargar datos humanos ---
    print("\n[2] Cargando datos humanos...")
    if not (ROOT / 'data' / 'raw' / 'humans_only_absent.csv').exists():
        print("    ERROR: No se encuentra humans_only_absent.csv")
        print("    Descárgalo de: https://github.com/EAndrade-Lotero/SODCL/blob/master/Data/humans_only_absent.csv")
        return
    
    df = pd.read_csv(ROOT / 'data' / 'raw' / 'humans_only_absent.csv')
    print(f"    Filas: {len(df)}, Díadas: {df['Dyad'].nunique()}")
    
    # --- 3. Identificar díadas por tipo ---
    print("\n[3] Clasificando díadas...")
    late = df[df['Round'] >= 40]
    
    lr_dyads = []
    tb_dyads = []
    mixed_dyads = []
    
    for dyad in df['Dyad'].unique():
        dyad_late = late[late['Dyad'] == dyad]
        cats = dyad_late['Category'].unique()
        
        if 'LEFT' in cats and 'RIGHT' in cats:
            lr_dyads.append(dyad)
        elif 'TOP' in cats and 'BOTTOM' in cats:
            tb_dyads.append(dyad)
        else:
            mixed_dyads.append(dyad)
    
    print(f"    Left-Right: {len(lr_dyads)}")
    print(f"    Top-Bottom: {len(tb_dyads)}")
    print(f"    Mixed: {len(mixed_dyads)}")
    
    # --- 4. Analizar todas las díadas ---
    print("\n[4] Analizando díadas con splits claros...")
    
    results = []
    clear_dyads = lr_dyads + tb_dyads
    
    for dyad in clear_dyads:
        split_type = 'LR' if dyad in lr_dyads else 'TB'
        S_obs, f1, f2 = get_observed_partition(dyad, df)
        
        if S_obs is None or len(S_obs) == 0 or len(S_obs) == 64:
            continue
        
        h_obs = conductance(S_obs, A, degrees)
        S_bar = set(range(64)) - S_obs
        conn = is_connected(S_obs, A) and is_connected(S_bar, A)
        eta = h_fiedler / h_obs if h_obs > 0 else float('inf')
        
        results.append({
            'Dyad': dyad,
            'Type': split_type,
            'S_obs_size': len(S_obs),
            'h_obs': h_obs,
            'h_fiedler': h_fiedler,
            'eta': eta,
            'connected': conn
        })
    
    print("\n[5] Analizando díadas mixed...")
    for dyad in mixed_dyads:
        S_obs, f1, f2 = get_observed_partition(dyad, df)
        
        if S_obs is None or len(S_obs) < 10 or len(S_obs) > 54:
            continue
        
        h_obs = conductance(S_obs, A, degrees)
        if h_obs == float('inf'):
            continue
        
        eta = h_fiedler / h_obs if h_obs > 0 else 0
        
        results.append({
            'Dyad': dyad,
            'Type': 'MIXED',
            'S_obs_size': len(S_obs),
            'h_obs': h_obs,
            'h_fiedler': h_fiedler,
            'eta': eta,
            'connected': True
        })
    
    # --- 5. Guardar resultados ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(ROOT / 'data' / 'results' / 'spectral_comparison_results.csv', index=False)
    print(f"\n    Guardado: spectral_comparison_results.csv ({len(results_df)} díadas)")
    
    # --- 6. Estadísticas ---
    print("\n" + "="*70)
    print("RESULTADOS")
    print("="*70)
    
    clear = results_df[results_df['Type'].isin(['LR', 'TB'])]
    mixed = results_df[results_df['Type'] == 'MIXED']
    
    print(f"\n{'Métrica':<30} {'Clear (LR/TB)':>15} {'Mixed':>15}")
    print("-"*60)
    print(f"{'N díadas':<30} {len(clear):>15} {len(mixed):>15}")
    print(f"{'h(S_obs) promedio':<30} {clear['h_obs'].mean():>15.4f} {mixed['h_obs'].mean():>15.4f}")
    print(f"{'h(S_obs) std':<30} {clear['h_obs'].std():>15.4f} {mixed['h_obs'].std():>15.4f}")
    print(f"{'η promedio':<30} {clear['eta'].mean():>15.2f} {mixed['eta'].mean():>15.2f}")
    print(f"{'η > 1 (mejor que Fiedler)':<30} {(clear['eta'] > 1).sum():>15} {(mixed['eta'] > 1).sum():>15}")
    
    # Test estadístico
    if len(clear) > 0 and len(mixed) > 0:
        t_stat, p_val = stats.ttest_ind(clear['h_obs'], mixed['h_obs'])
        print(f"\n--- Test t (h_obs clear vs mixed) ---")
        print(f"t = {t_stat:.3f}, p = {p_val:.6f}")
    
    # --- 7. Figuras ---
    print("\n[6] Generando figuras...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    ax1 = axes[0]
    clear_h = clear['h_obs'].values
    mixed_h = mixed['h_obs'].values
    bp = ax1.boxplot([clear_h, mixed_h], tick_labels=['Clear Splits\n(LR/TB)', 'Mixed'],
                      patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax1.axhline(y=h_fiedler, color='blue', linestyle='--', linewidth=2, 
                label=f'h(Fiedler) = {h_fiedler:.4f}')
    ax1.set_ylabel('Conductance h(S)', fontsize=12)
    ax1.set_title('Human Partitions vs Fiedler Bisection', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Scatter plot
    ax2 = axes[1]
    clear_eta = clear['eta'].values
    mixed_eta = mixed['eta'].values
    ax2.scatter(range(1, len(clear_eta)+1), clear_eta, s=100, c='#2ecc71', 
                label='Clear Splits', edgecolors='black', zorder=3)
    ax2.scatter(range(len(clear_eta)+1, len(clear_eta)+len(mixed_eta)+1), mixed_eta, 
                s=100, c='#e74c3c', label='Mixed', edgecolors='black', zorder=3)
    ax2.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='η = 1')
    ax2.set_xlabel('Dyad Index', fontsize=12)
    ax2.set_ylabel('η = h(Fiedler) / h(S_obs)', fontsize=12)
    ax2.set_title('Spectral Efficiency Ratio', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.5)
    ax2.grid(alpha=0.3)
    ax2.axhspan(1.0, 1.5, alpha=0.1, color='green')
    ax2.axhspan(0, 1.0, alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.savefig(ROOT / 'figures' / 'spectral_comparison_summary.png', dpi=150, bbox_inches='tight')
    print("    Guardado: spectral_comparison_summary.png")
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == '__main__':
    main()
