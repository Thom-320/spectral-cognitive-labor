"""
00_spectral_grid.py
===================
Análisis Espectral del Grafo del Tablero 8×8
Proyecto: Eficiencia Espectral de la División del Trabajo Humano

Este script calcula las propiedades espectrales del grafo base SIN datos humanos:
- Matriz de adyacencia A (8-conectividad)
- Matriz Laplaciana L = D - A
- Eigenvalores y eigenvectores de L
- Valor de Fiedler λ₂ y vector de Fiedler v₂
- Bisección de Fiedler S_Fiedler
- Conductancia h(S_Fiedler)
- Cota de Cheeger λ₂/2

Autor: [Tu nombre]
Fecha: Febrero 2026
"""

from pathlib import Path

# Directorio raíz del proyecto (un nivel arriba de src/)
ROOT = Path(__file__).parent.parent

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from collections import deque

# =============================================================================
# 1. CONSTRUCCIÓN DEL GRAFO 8×8 CON 8-CONECTIVIDAD
# =============================================================================

def build_grid_graph(n=8):
    """
    Construye el grafo de una grilla n×n con 8-conectividad.
    
    Cada celda (i,j) se conecta con sus vecinos:
    - 4-conectividad: arriba, abajo, izquierda, derecha
    - 8-conectividad: además, las 4 diagonales
    
    Returns:
        A: Matriz de adyacencia (n²×n²)
        pos: Diccionario {nodo: (x, y)} para visualización
    """
    num_nodes = n * n
    A = np.zeros((num_nodes, num_nodes))
    pos = {}
    
    def node_id(i, j):
        """Convierte coordenadas (i,j) a índice de nodo."""
        return i * n + j
    
    # Direcciones de 8-conectividad: (di, dj)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),  # arriba-izq, arriba, arriba-der
        (0, -1),           (0, 1),   # izquierda, derecha
        (1, -1),  (1, 0),  (1, 1)    # abajo-izq, abajo, abajo-der
    ]
    
    for i in range(n):
        for j in range(n):
            node = node_id(i, j)
            pos[node] = (j, n - 1 - i)  # x=columna, y=fila invertida para visualización
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    neighbor = node_id(ni, nj)
                    A[node, neighbor] = 1
    
    return A, pos

# =============================================================================
# 2. CÁLCULO DE MATRICES LAPLACIANAS
# =============================================================================

def compute_laplacian(A):
    """
    Calcula la matriz Laplaciana L = D - A.
    
    Args:
        A: Matriz de adyacencia
    
    Returns:
        L: Matriz Laplaciana
        D: Matriz de grados (diagonal)
        degrees: Vector de grados
    """
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    L = D - A
    return L, D, degrees

def compute_normalized_laplacian(A):
    """
    Calcula la matriz Laplaciana normalizada L_norm = D^{-1/2} L D^{-1/2}.
    
    Útil para comparar con literatura de spectral clustering.
    """
    degrees = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L = np.diag(degrees) - A
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    return L_norm

# =============================================================================
# 3. ANÁLISIS ESPECTRAL
# =============================================================================

def spectral_analysis(L):
    """
    Realiza análisis espectral completo de la matriz Laplaciana.
    
    Returns:
        eigenvalues: Eigenvalores ordenados de menor a mayor
        eigenvectors: Eigenvectores correspondientes (columnas)
        lambda2: Segundo eigenvalor más pequeño (conectividad algebraica)
        fiedler_vector: Eigenvector asociado a λ₂
    """
    eigenvalues, eigenvectors = linalg.eigh(L)
    
    # Ordenar por eigenvalor (ya deberían estar ordenados, pero por seguridad)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # λ₁ ≈ 0 (grafo conexo), λ₂ = conectividad algebraica
    lambda2 = eigenvalues[1]
    fiedler_vector = eigenvectors[:, 1]
    
    return eigenvalues, eigenvectors, lambda2, fiedler_vector

# =============================================================================
# 4. BISECCIÓN DE FIEDLER
# =============================================================================

def fiedler_bisection(fiedler_vector, method='median'):
    """
    Calcula la bisección espectral usando el vector de Fiedler.
    
    Args:
        fiedler_vector: El segundo eigenvector de L
        method: 'median' (corte en la mediana) o 'zero' (corte en cero)
    
    Returns:
        S: Conjunto de nodos en la primera partición
        S_bar: Conjunto de nodos en la segunda partición
    """
    if method == 'median':
        threshold = np.median(fiedler_vector)
    else:
        threshold = 0.0
    
    S = set(np.where(fiedler_vector >= threshold)[0])
    S_bar = set(np.where(fiedler_vector < threshold)[0])
    
    return S, S_bar

def optimal_fiedler_cut(fiedler_vector, A, degrees):
    """
    Encuentra el corte óptimo explorando umbrales sobre el vector de Fiedler.
    
    Este es el algoritmo estándar de particionamiento espectral:
    1. Ordenar nodos por su coordenada en el Fiedler vector
    2. Probar cada posible corte
    3. Quedarse con el de mínima conductancia
    
    Returns:
        best_S: Partición óptima
        best_conductance: Conductancia del corte óptimo
    """
    n = len(fiedler_vector)
    sorted_indices = np.argsort(fiedler_vector)
    
    best_conductance = float('inf')
    best_S = None
    
    # Probar cada posible punto de corte
    for k in range(1, n):
        S = set(sorted_indices[:k])
        S_bar = set(sorted_indices[k:])
        
        h = conductance(S, A, degrees)
        
        if h < best_conductance:
            best_conductance = h
            best_S = S.copy()
    
    return best_S, best_conductance

# =============================================================================
# 5. CÁLCULO DE CONDUCTANCIA
# =============================================================================

def conductance(S, A, degrees):
    """
    Calcula la conductancia de una partición (S, V\S).
    
    h(S) = cut(S, S̄) / min(vol(S), vol(S̄))
    
    donde:
    - cut(S, S̄) = número de aristas entre S y su complemento
    - vol(S) = suma de grados de nodos en S
    
    Args:
        S: Conjunto de nodos en la partición
        A: Matriz de adyacencia
        degrees: Vector de grados
    
    Returns:
        h: Conductancia de la partición
    """
    S = set(S)
    n = len(degrees)
    S_bar = set(range(n)) - S
    
    # Calcular cut(S, S̄)
    cut = 0
    for u in S:
        for v in S_bar:
            cut += A[u, v]
    
    # Calcular volúmenes
    vol_S = sum(degrees[u] for u in S)
    vol_S_bar = sum(degrees[v] for v in S_bar)
    
    # Conductancia
    min_vol = min(vol_S, vol_S_bar)
    if min_vol == 0:
        return float('inf')
    
    h = cut / min_vol
    return h

# =============================================================================
# 6. VERIFICACIÓN DE CONECTIVIDAD (BFS)
# =============================================================================

def is_connected(S, A):
    """
    Verifica si el subgrafo inducido por S es conexo usando BFS.
    
    Args:
        S: Conjunto de nodos
        A: Matriz de adyacencia del grafo completo
    
    Returns:
        True si S induce un subgrafo conexo, False otherwise
    """
    if len(S) == 0:
        return True
    
    S = list(S)
    visited = set()
    queue = deque([S[0]])
    visited.add(S[0])
    
    while queue:
        node = queue.popleft()
        for neighbor in S:
            if neighbor not in visited and A[node, neighbor] > 0:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == len(S)

# =============================================================================
# 7. VISUALIZACIÓN
# =============================================================================

def plot_fiedler_grid(fiedler_vector, S_fiedler, n=8, save_path=None):
    """
    Visualiza la grilla coloreada por el vector de Fiedler y marca la bisección.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Panel 1: Fiedler Vector Heatmap ---
    ax1 = axes[0]
    fiedler_grid = fiedler_vector.reshape(n, n)
    im1 = ax1.imshow(fiedler_grid, cmap='RdBu', aspect='equal')
    ax1.set_title('Vector de Fiedler $\\mathbf{v}_2$', fontsize=14)
    ax1.set_xlabel('Columna')
    ax1.set_ylabel('Fila')
    plt.colorbar(im1, ax=ax1, label='$v_2(i,j)$')
    
    # Añadir valores en cada celda
    for i in range(n):
        for j in range(n):
            val = fiedler_grid[i, j]
            color = 'white' if abs(val) > 0.1 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    fontsize=7, color=color)
    
    # --- Panel 2: Bisección de Fiedler ---
    ax2 = axes[1]
    partition_grid = np.zeros((n, n))
    for node in S_fiedler:
        i, j = node // n, node % n
        partition_grid[i, j] = 1
    
    im2 = ax2.imshow(partition_grid, cmap='coolwarm', aspect='equal', vmin=0, vmax=1)
    ax2.set_title('Bisección de Fiedler $S_{Fiedler}$', fontsize=14)
    ax2.set_xlabel('Columna')
    ax2.set_ylabel('Fila')
    
    # Marcar frontera
    for i in range(n):
        for j in range(n):
            node = i * n + j
            in_S = node in S_fiedler
            ax2.text(j, i, 'S' if in_S else '$\\bar{S}$', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if in_S else 'darkblue')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()
    return fig

def plot_spectrum(eigenvalues, lambda2, save_path=None):
    """
    Visualiza el espectro del Laplaciano.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.bar(range(len(eigenvalues)), eigenvalues, color='steelblue', alpha=0.7)
    ax.axhline(y=lambda2, color='red', linestyle='--', 
               label=f'$\\lambda_2$ = {lambda2:.4f}')
    ax.axhline(y=lambda2/2, color='orange', linestyle=':', 
               label=f'$\\lambda_2/2$ = {lambda2/2:.4f} (Cheeger bound)')
    
    ax.set_xlabel('Índice del eigenvalor', fontsize=12)
    ax.set_ylabel('$\\lambda_i$', fontsize=12)
    ax.set_title('Espectro del Laplaciano del Grafo 8×8', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()
    return fig

# =============================================================================
# 8. EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    print("=" * 70)
    print("ANÁLISIS ESPECTRAL DEL GRAFO 8×8 CON 8-CONECTIVIDAD")
    print("=" * 70)
    
    # 1. Construir grafo
    print("\n[1] Construyendo grafo del tablero...")
    A, pos = build_grid_graph(n=8)
    print(f"    - Nodos: {A.shape[0]}")
    print(f"    - Aristas: {int(np.sum(A)/2)}")
    
    # 2. Calcular Laplaciano
    print("\n[2] Calculando matriz Laplaciana L = D - A...")
    L, D, degrees = compute_laplacian(A)
    print(f"    - Grado mínimo: {int(min(degrees))}")
    print(f"    - Grado máximo: {int(max(degrees))}")
    print(f"    - Grado promedio: {np.mean(degrees):.2f}")
    
    # 3. Análisis espectral
    print("\n[3] Realizando análisis espectral...")
    eigenvalues, eigenvectors, lambda2, fiedler_vector = spectral_analysis(L)
    print(f"    - λ₁ = {eigenvalues[0]:.6f} (≈ 0, grafo conexo)")
    print(f"    - λ₂ = {lambda2:.6f} (conectividad algebraica)")
    print(f"    - λ₃ = {eigenvalues[2]:.6f}")
    print(f"    - λ_max = {eigenvalues[-1]:.6f}")
    
    # 4. Bisección de Fiedler
    print("\n[4] Calculando bisección de Fiedler...")
    S_median, S_bar_median = fiedler_bisection(fiedler_vector, method='median')
    print(f"    - Bisección por mediana: |S| = {len(S_median)}, |S̄| = {len(S_bar_median)}")
    
    # 5. Buscar corte óptimo
    print("\n[5] Buscando corte óptimo sobre Fiedler vector...")
    S_optimal, h_optimal = optimal_fiedler_cut(fiedler_vector, A, degrees)
    print(f"    - Corte óptimo: |S| = {len(S_optimal)}, |S̄| = {64 - len(S_optimal)}")
    print(f"    - h(S_Fiedler) = {h_optimal:.6f}")
    
    # 6. Verificar conectividad
    print("\n[6] Verificando conectividad de particiones...")
    S_connected = is_connected(S_optimal, A)
    S_bar_connected = is_connected(set(range(64)) - S_optimal, A)
    print(f"    - S es conexo: {S_connected}")
    print(f"    - S̄ es conexo: {S_bar_connected}")
    
    # 7. Cota de Cheeger
    print("\n[7] Desigualdad de Cheeger:")
    cheeger_lower = lambda2 / 2
    cheeger_upper = np.sqrt(2 * lambda2)
    print(f"    λ₂/2 ≤ h(G) ≤ √(2λ₂)")
    print(f"    {cheeger_lower:.6f} ≤ h(G) ≤ {cheeger_upper:.6f}")
    print(f"    h(S_Fiedler) = {h_optimal:.6f}")
    print(f"    Gap sobre cota inferior: Δ = h - λ₂/2 = {h_optimal - cheeger_lower:.6f}")
    
    # 8. Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS ESPECTRALES (GRAFO BASE)")
    print("=" * 70)
    print(f"  Grafo: 8×8 grilla con 8-conectividad")
    print(f"  |V| = 64, |E| = {int(np.sum(A)/2)}")
    print(f"  λ₂ (conectividad algebraica) = {lambda2:.6f}")
    print(f"  h(S_Fiedler) = {h_optimal:.6f}")
    print(f"  λ₂/2 (Cheeger lower bound) = {cheeger_lower:.6f}")
    print(f"  √(2λ₂) (Cheeger upper bound) = {cheeger_upper:.6f}")
    print("=" * 70)
    
    # 9. Guardar resultados
    results = {
        'lambda2': lambda2,
        'h_fiedler': h_optimal,
        'cheeger_lower': cheeger_lower,
        'cheeger_upper': cheeger_upper,
        'S_fiedler': S_optimal,
        'fiedler_vector': fiedler_vector,
        'eigenvalues': eigenvalues,
        'A': A,
        'degrees': degrees
    }
    
    np.savez(ROOT / 'data' / 'results' / 'spectral_results.npz',
             lambda2=lambda2,
             h_fiedler=h_optimal,
             cheeger_lower=cheeger_lower,
             cheeger_upper=cheeger_upper,
             S_fiedler=np.array(list(S_optimal)),
             fiedler_vector=fiedler_vector,
             eigenvalues=eigenvalues)
    print("\nResultados guardados en: spectral_results.npz")
    
    # 10. Visualización
    print("\n[8] Generando visualizaciones...")
    plot_fiedler_grid(fiedler_vector, S_optimal, n=8, 
                      save_path=ROOT / 'figures' / 'fiedler_grid.png')
    plot_spectrum(eigenvalues, lambda2,
                  save_path=ROOT / 'figures' / 'spectrum.png')
    
    return results

if __name__ == "__main__":
    results = main()
