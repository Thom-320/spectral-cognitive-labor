"""
01_single_dyad_analysis.py
==========================
Análisis de UNA díada para demostrar el pipeline completo.

Díada seleccionada: 435-261 (divergencia LR del 93%)
"""

from pathlib import Path

# Directorio raíz del proyecto (un nivel arriba de src/)
ROOT = Path(__file__).parent.parent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# Cargar funciones del análisis espectral
# (En un notebook real, importaríamos del módulo)

def build_grid_graph(n=8):
    """Construye grafo 8x8 con 8-conectividad."""
    num_nodes = n * n
    A = np.zeros((num_nodes, num_nodes))
    
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for i in range(n):
        for j in range(n):
            node = i * n + j
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    neighbor = ni * n + nj
                    A[node, neighbor] = 1
    
    return A

def conductance(S, A, degrees):
    """Calcula conductancia h(S)."""
    S = set(S)
    n = len(degrees)
    S_bar = set(range(n)) - S
    
    cut = sum(A[u, v] for u in S for v in S_bar)
    vol_S = sum(degrees[u] for u in S)
    vol_S_bar = sum(degrees[v] for v in S_bar)
    
    min_vol = min(vol_S, vol_S_bar)
    if min_vol == 0:
        return float('inf')
    
    return cut / min_vol

def is_connected(S, A):
    """Verifica conectividad con BFS."""
    if len(S) == 0:
        return True
    
    S = list(S)
    visited = set([S[0]])
    queue = deque([S[0]])
    
    while queue:
        node = queue.popleft()
        for neighbor in S:
            if neighbor not in visited and A[node, neighbor] > 0:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(S)

# =============================================================================
# ANÁLISIS DE DÍADA 435-261
# =============================================================================

def main():
    print("=" * 70)
    print("ANÁLISIS DE DÍADA 435-261 (LEFT-RIGHT SPLIT)")
    print("=" * 70)
    
    # 1. Cargar datos
    print("\n[1] Cargando dataset SODCL...")
    df = pd.read_csv(ROOT / 'data' / 'raw' / 'humans_only_absent.csv')
    
    # Filtrar díada 435-261
    dyad_data = df[df['Dyad'] == '435-261']
    print(f"    Filas para díada 435-261: {len(dyad_data)}")
    
    # 2. Separar jugadores
    p1_data = dyad_data[dyad_data['Player'].str.contains('PL1')]
    p2_data = dyad_data[dyad_data['Player'].str.contains('PL2')]
    print(f"    Rondas P1: {len(p1_data)}, Rondas P2: {len(p2_data)}")
    
    # 3. Construir matrices de frecuencia (rondas 40-60 para convergencia)
    print("\n[2] Calculando frecuencias de visita (rondas 40-60)...")
    
    # Columnas de visitas: a11 a a88
    visit_cols = [f'a{i}{j}' for i in range(1, 9) for j in range(1, 9)]
    
    # Filtrar rondas tardías
    late_rounds = range(40, 61)
    p1_late = p1_data[p1_data['Round'].isin(late_rounds)]
    p2_late = p2_data[p2_data['Round'].isin(late_rounds)]
    
    # Sumar visitas por celda
    f1 = p1_late[visit_cols].sum().values.reshape(8, 8)
    f2 = p2_late[visit_cols].sum().values.reshape(8, 8)
    
    print(f"    Rondas analizadas: {len(p1_late)}")
    print(f"    Total visitas P1: {f1.sum()}")
    print(f"    Total visitas P2: {f2.sum()}")
    
    # 4. Definir partición observada
    print("\n[3] Definiendo partición observada S_obs...")
    
    # S_obs = {v : f1(v) > f2(v)}
    # Usamos > en vez de >= para evitar empates
    tau = 0  # umbral mínimo
    diff = f1 - f2
    
    S_obs = set()
    for i in range(8):
        for j in range(8):
            node = i * 8 + j
            if diff[i, j] > tau:
                S_obs.add(node)
    
    print(f"    |S_obs| = {len(S_obs)} (celdas dominadas por P1)")
    print(f"    |S̄_obs| = {64 - len(S_obs)} (celdas dominadas por P2)")
    
    # 5. Verificar conectividad con BFS
    print("\n[4] Verificando conectividad (BFS)...")
    A = build_grid_graph(8)
    degrees = np.sum(A, axis=1)
    
    S_obs_connected = is_connected(S_obs, A)
    S_bar_connected = is_connected(set(range(64)) - S_obs, A)
    
    print(f"    S_obs es conexo: {S_obs_connected}")
    print(f"    S̄_obs es conexo: {S_bar_connected}")
    
    # 6. Calcular conductancia
    print("\n[5] Calculando conductancia h(S_obs)...")
    h_obs = conductance(S_obs, A, degrees)
    print(f"    h(S_obs) = {h_obs:.6f}")
    
    # 7. Cargar resultados espectrales
    print("\n[6] Comparando con bisección de Fiedler...")
    spectral = np.load(ROOT / 'data' / 'results' / 'spectral_results.npz')
    lambda2 = spectral['lambda2']
    h_fiedler = spectral['h_fiedler']
    S_fiedler = set(spectral['S_fiedler'])
    
    print(f"    λ₂ = {lambda2:.6f}")
    print(f"    h(S_Fiedler) = {h_fiedler:.6f}")
    
    # 8. Calcular métricas
    print("\n[7] Métricas de eficiencia espectral:")
    eta = h_fiedler / h_obs if h_obs > 0 else float('inf')
    delta = h_obs - lambda2 / 2
    
    print(f"    η = h(S_Fiedler) / h(S_obs) = {eta:.4f}")
    print(f"    Δ = h(S_obs) - λ₂/2 = {delta:.6f}")
    
    if eta > 0.8:
        print("    → Interpretación: Partición CERCA del óptimo espectral")
    elif eta > 0.5:
        print("    → Interpretación: Partición MODERADAMENTE eficiente")
    else:
        print("    → Interpretación: Partición LEJOS del óptimo espectral")
    
    # 9. Visualización
    print("\n[8] Generando visualizaciones...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Frecuencia P1
    ax1 = axes[0, 0]
    im1 = ax1.imshow(f1, cmap='Blues', aspect='equal')
    ax1.set_title('Frecuencia Visitas P1 (rondas 40-60)', fontsize=12)
    plt.colorbar(im1, ax=ax1)
    for i in range(8):
        for j in range(8):
            ax1.text(j, i, int(f1[i, j]), ha='center', va='center', fontsize=8)
    
    # Panel 2: Frecuencia P2
    ax2 = axes[0, 1]
    im2 = ax2.imshow(f2, cmap='Oranges', aspect='equal')
    ax2.set_title('Frecuencia Visitas P2 (rondas 40-60)', fontsize=12)
    plt.colorbar(im2, ax=ax2)
    for i in range(8):
        for j in range(8):
            ax2.text(j, i, int(f2[i, j]), ha='center', va='center', fontsize=8)
    
    # Panel 3: Diferencia (P1 - P2)
    ax3 = axes[1, 0]
    im3 = ax3.imshow(diff, cmap='RdBu', aspect='equal', 
                     vmin=-diff.max(), vmax=diff.max())
    ax3.set_title('Diferencia: P1 - P2', fontsize=12)
    plt.colorbar(im3, ax=ax3)
    
    # Panel 4: Partición observada vs Fiedler
    ax4 = axes[1, 1]
    partition_grid = np.zeros((8, 8))
    fiedler_grid = np.zeros((8, 8))
    
    for node in S_obs:
        i, j = node // 8, node % 8
        partition_grid[i, j] = 1
    
    for node in S_fiedler:
        i, j = node // 8, node % 8
        fiedler_grid[i, j] = 1
    
    # Crear visualización combinada
    combined = np.zeros((8, 8, 3))
    for i in range(8):
        for j in range(8):
            node = i * 8 + j
            if node in S_obs and node in S_fiedler:
                combined[i, j] = [0.5, 0, 0.5]  # Púrpura: coinciden
            elif node in S_obs:
                combined[i, j] = [0, 0, 1]  # Azul: solo S_obs
            elif node in S_fiedler:
                combined[i, j] = [1, 0, 0]  # Rojo: solo S_Fiedler
            else:
                combined[i, j] = [0.9, 0.9, 0.9]  # Gris: ninguno
    
    ax4.imshow(combined, aspect='equal')
    ax4.set_title(f'S_obs (azul) vs S_Fiedler (rojo)\nη = {eta:.3f}', fontsize=12)
    
    # Marcar celdas
    for i in range(8):
        for j in range(8):
            node = i * 8 + j
            if node in S_obs and node in S_fiedler:
                marker = '●'
            elif node in S_obs:
                marker = 'H'  # Human
            elif node in S_fiedler:
                marker = 'F'  # Fiedler
            else:
                marker = '○'
            ax4.text(j, i, marker, ha='center', va='center', fontsize=10,
                    color='white' if node in S_obs or node in S_fiedler else 'gray')
    
    plt.tight_layout()
    plt.savefig(ROOT / 'figures' / 'dyad_435_261_analysis.png', 
                dpi=150, bbox_inches='tight')
    print("    Figura guardada: dyad_435_261_analysis.png")
    plt.show()
    
    # 10. Resumen
    print("\n" + "=" * 70)
    print("RESUMEN: DÍADA 435-261")
    print("=" * 70)
    print(f"  Tipo de división: Left-Right (93% divergencia)")
    print(f"  |S_obs| = {len(S_obs)}, |S̄_obs| = {64 - len(S_obs)}")
    print(f"  Ambas particiones conexas: {S_obs_connected and S_bar_connected}")
    print(f"  h(S_obs) = {h_obs:.6f}")
    print(f"  h(S_Fiedler) = {h_fiedler:.6f}")
    print(f"  η (ratio eficiencia) = {eta:.4f}")
    print(f"  Δ (gap Cheeger) = {delta:.6f}")
    print("=" * 70)
    
    return {
        'dyad': '435-261',
        'S_obs': S_obs,
        'h_obs': h_obs,
        'h_fiedler': h_fiedler,
        'eta': eta,
        'delta': delta,
        'connected': S_obs_connected and S_bar_connected
    }

if __name__ == "__main__":
    results = main()
