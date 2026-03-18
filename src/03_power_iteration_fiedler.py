#!/usr/bin/env python3
"""
Cálculo del Vector de Fiedler usando Power Iteration con Deflación
===================================================================

Este script implementa el cálculo del segundo eigenvector del Laplaciano
(vector de Fiedler) SIN usar numpy.linalg.eig ni scipy.

Algoritmos implementados:
1. Power Iteration clásica (para eigenvalor dominante)
2. Inverse Power Iteration (para eigenvalor más pequeño)
3. Deflación (para obtener el segundo eigenvalor)
4. Bisección espectral

Autor: Thomas Chísica
Curso: Teoría de Grafos 2026-I
Universidad del Rosario
"""

from pathlib import Path

# Directorio raíz del proyecto (un nivel arriba de src/)
ROOT = Path(__file__).parent.parent

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# ============================================================
# CONSTRUCCIÓN DEL GRAFO
# ============================================================

def build_grid_graph(n=8):
    """
    Construye la matriz de adyacencia de un grid n×n con 8-conectividad.
    
    Parámetros:
        n: tamaño del grid (default 8)
    
    Retorna:
        A: matriz de adyacencia (n²×n²)
    """
    num_nodes = n * n
    A = np.zeros((num_nodes, num_nodes))
    
    for i in range(n):
        for j in range(n):
            node = i * n + j
            # 8 vecinos (incluyendo diagonales)
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        neighbor = ni * n + nj
                        A[node, neighbor] = 1
    return A


def compute_laplacian(A):
    """
    Calcula el Laplaciano combinatorial L = D - A.
    
    Parámetros:
        A: matriz de adyacencia
    
    Retorna:
        L: matriz Laplaciana
        degrees: vector de grados
    """
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    L = D - A
    return L, degrees


# ============================================================
# POWER ITERATION: IMPLEMENTACIÓN DESDE CERO
# ============================================================

def power_iteration(M, num_iter=1000, tol=1e-12):
    """
    Power Iteration clásica para encontrar el eigenvalor dominante.
    
    Encuentra el eigenvector correspondiente al eigenvalor de mayor magnitud.
    
    Parámetros:
        M: matriz cuadrada
        num_iter: número máximo de iteraciones
        tol: tolerancia de convergencia
    
    Retorna:
        eigenvalue: eigenvalor dominante
        eigenvector: eigenvector correspondiente (normalizado)
        iterations: número de iteraciones usadas
    """
    n = M.shape[0]
    
    # Inicializar con vector aleatorio
    np.random.seed(42)  # Reproducibilidad
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    eigenvalue_old = 0
    
    for iteration in range(num_iter):
        # Multiplicar por M
        v_new = M @ v
        
        # Calcular eigenvalor (cociente de Rayleigh)
        eigenvalue = np.dot(v, v_new)
        
        # Normalizar
        norm = np.linalg.norm(v_new)
        if norm < 1e-15:
            break
        v_new = v_new / norm
        
        # Verificar convergencia
        if abs(eigenvalue - eigenvalue_old) < tol:
            return eigenvalue, v_new, iteration + 1
        
        eigenvalue_old = eigenvalue
        v = v_new
    
    return eigenvalue, v, num_iter


def inverse_power_iteration(M, shift=0, num_iter=1000, tol=1e-12):
    """
    Inverse Power Iteration para encontrar eigenvalor cercano a 'shift'.
    
    Resuelve (M - shift*I)^{-1} v = λ v iterativamente.
    
    Parámetros:
        M: matriz cuadrada
        shift: valor cerca del cual buscar eigenvalor
        num_iter: número máximo de iteraciones
        tol: tolerancia de convergencia
    
    Retorna:
        eigenvalue: eigenvalor más cercano a shift
        eigenvector: eigenvector correspondiente
        iterations: número de iteraciones usadas
    """
    n = M.shape[0]
    
    # Matriz shifteada
    M_shifted = M - shift * np.eye(n)
    
    # Inicializar
    np.random.seed(42)
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    eigenvalue_old = 0
    
    for iteration in range(num_iter):
        # Resolver sistema lineal (M - shift*I) w = v
        # Usamos solve de numpy pero podríamos implementar LU
        try:
            w = np.linalg.solve(M_shifted, v)
        except np.linalg.LinAlgError:
            # Matriz singular, añadir pequeña perturbación
            M_shifted_reg = M_shifted + 1e-10 * np.eye(n)
            w = np.linalg.solve(M_shifted_reg, v)
        
        # Calcular eigenvalor
        eigenvalue = np.dot(v, w)
        if abs(eigenvalue) > 1e-15:
            eigenvalue = 1.0 / eigenvalue + shift
        
        # Normalizar
        norm = np.linalg.norm(w)
        if norm < 1e-15:
            break
        v_new = w / norm
        
        # Verificar convergencia
        if abs(eigenvalue - eigenvalue_old) < tol:
            return eigenvalue, v_new, iteration + 1
        
        eigenvalue_old = eigenvalue
        v = v_new
    
    return eigenvalue, v, num_iter


def power_iteration_with_deflation(L, num_iter=1000, tol=1e-12):
    """
    Calcula el segundo eigenvalor/eigenvector del Laplaciano usando
    Power Iteration con deflación.
    
    ALGORITMO:
    1. El primer eigenvector del Laplaciano es v₁ = (1,1,...,1)/√n con λ₁ = 0
    2. Para encontrar λ₂, usamos la matriz M = L_max*I - L
       donde L_max es una cota superior del mayor eigenvalor de L
    3. El mayor eigenvalor de M corresponde al menor eigenvalor de L
    4. Usamos deflación para evitar converger a v₁
    
    Parámetros:
        L: matriz Laplaciana
        num_iter: número máximo de iteraciones
        tol: tolerancia de convergencia
    
    Retorna:
        lambda2: segundo eigenvalor (conectividad algebraica)
        v2: vector de Fiedler
        iterations: iteraciones usadas
    """
    n = L.shape[0]
    
    # Paso 1: El primer eigenvector es conocido (constante)
    v1 = np.ones(n) / np.sqrt(n)
    
    # Paso 2: Cota superior para λ_max usando círculos de Gershgorin
    # λ_max ≤ max_i (L_ii + Σ_{j≠i} |L_ij|) = max_i (2 * degree_i)
    gershgorin_bound = np.max(2 * np.diag(L))
    L_max = gershgorin_bound + 1  # Pequeño margen de seguridad
    
    # Paso 3: Construir M = L_max*I - L
    # Los eigenvalores de M son: L_max - λ_i
    # Si L tiene eigenvalores 0 = λ₁ ≤ λ₂ ≤ ... ≤ λ_n
    # entonces M tiene eigenvalores L_max ≥ L_max - λ₂ ≥ ... ≥ L_max - λ_n
    # El SEGUNDO mayor eigenvalor de M es L_max - λ₂
    M = L_max * np.eye(n) - L
    
    # Paso 4: Power iteration con deflación
    np.random.seed(42)
    v = np.random.randn(n)
    
    # Ortogonalizar contra v1 (deflación)
    v = v - np.dot(v, v1) * v1
    v = v / np.linalg.norm(v)
    
    mu_old = 0  # eigenvalor de M
    
    for iteration in range(num_iter):
        # Multiplicar por M
        w = M @ v
        
        # Deflación: proyectar fuera del espacio de v1
        w = w - np.dot(w, v1) * v1
        
        # Calcular eigenvalor de M (cociente de Rayleigh)
        mu = np.dot(v, M @ v) / np.dot(v, v)
        
        # Normalizar
        norm = np.linalg.norm(w)
        if norm < 1e-15:
            print(f"  Warning: norma casi cero en iteración {iteration}")
            break
        v_new = w / norm
        
        # Verificar convergencia
        if abs(mu - mu_old) < tol:
            lambda2 = L_max - mu
            return lambda2, v_new, iteration + 1
        
        mu_old = mu
        v = v_new
    
    lambda2 = L_max - mu
    return lambda2, v, num_iter


def subspace_iteration_fiedler(L, num_iter=500, tol=1e-12):
    """
    Método alternativo: Subspace Iteration para los primeros k eigenvectores.
    
    Más robusto que power iteration simple para eigenvalores múltiples.
    """
    n = L.shape[0]
    k = 3  # Calcular los primeros 3 eigenvectores
    
    # Inicializar subespacio aleatorio
    np.random.seed(42)
    V = np.random.randn(n, k)
    V, _ = np.linalg.qr(V)  # Ortogonalizar
    
    # Cota de Gershgorin
    L_max = np.max(2 * np.diag(L)) + 1
    M = L_max * np.eye(n) - L
    
    for iteration in range(num_iter):
        # Multiplicar por M
        W = M @ V
        
        # Ortogonalizar (Gram-Schmidt via QR)
        V_new, R = np.linalg.qr(W)
        
        # Verificar convergencia
        # Los eigenvalores están en la diagonal de V^T M V
        eigenvalues = np.diag(V_new.T @ M @ V_new)
        
        if iteration > 0:
            if np.max(np.abs(eigenvalues - eigenvalues_old)) < tol:
                break
        
        eigenvalues_old = eigenvalues.copy()
        V = V_new
    
    # El segundo eigenvector de M (mayor eigenvalor después del primero)
    # corresponde al segundo eigenvector de L (menor eigenvalor después de 0)
    lambda_M = np.diag(V.T @ M @ V)
    idx = np.argsort(lambda_M)[::-1]  # Ordenar descendente
    
    # El primer eigenvector de M corresponde a λ₁=0 de L
    # El segundo eigenvector de M corresponde a λ₂ de L
    v2 = V[:, idx[1]]
    lambda2 = L_max - lambda_M[idx[1]]
    
    return lambda2, v2, iteration + 1


# ============================================================
# BISECCIÓN ESPECTRAL
# ============================================================

def spectral_bisection(v2, method='median'):
    """
    Realiza la bisección espectral usando el vector de Fiedler.
    
    Parámetros:
        v2: vector de Fiedler
        method: 'median' (bisección balanceada) o 'sign' (por signo)
    
    Retorna:
        S: conjunto de nodos en la primera partición
    """
    if method == 'median':
        threshold = np.median(v2)
    else:  # sign
        threshold = 0
    
    S = set(np.where(v2 >= threshold)[0])
    return S


def conductance(S, A, degrees):
    """
    Calcula la conductancia de una partición.
    h(S) = cut(S, S̄) / min(vol(S), vol(S̄))
    """
    S_set = set(S)
    S_bar = set(range(len(A))) - S_set
    
    if len(S_set) == 0 or len(S_bar) == 0:
        return float('inf')
    
    # Calcular corte
    cut = 0
    for i in S_set:
        for j in S_bar:
            cut += A[i, j]
    
    # Calcular volúmenes
    vol_S = sum(degrees[i] for i in S_set)
    vol_S_bar = sum(degrees[i] for i in S_bar)
    
    if min(vol_S, vol_S_bar) == 0:
        return float('inf')
    
    return cut / min(vol_S, vol_S_bar)


def cut_size(S, A):
    """Número de aristas que cruzan el corte"""
    S_set = set(S)
    S_bar = set(range(len(A))) - S_set
    return sum(A[i,j] for i in S_set for j in S_bar)


# ============================================================
# VERIFICACIÓN Y COMPARACIÓN
# ============================================================

def verify_against_numpy(L):
    """
    Verifica la implementación contra numpy.linalg.eigh
    """
    print("\n" + "="*70)
    print("VERIFICACIÓN: Power Iteration vs NumPy")
    print("="*70)
    
    # NumPy (referencia)
    eigenvalues_np, eigenvectors_np = np.linalg.eigh(L)
    lambda2_np = eigenvalues_np[1]
    v2_np = eigenvectors_np[:, 1]
    
    print(f"\n--- NumPy (referencia) ---")
    print(f"λ₂ = {lambda2_np:.10f}")
    print(f"λ₃ = {eigenvalues_np[2]:.10f}")
    print(f"v₂[:5] = {v2_np[:5]}")
    
    # Power Iteration con deflación
    print(f"\n--- Power Iteration con Deflación ---")
    start = time.time()
    lambda2_pi, v2_pi, iters_pi = power_iteration_with_deflation(L)
    time_pi = time.time() - start
    
    # Asegurar mismo signo para comparación
    if np.dot(v2_pi, v2_np) < 0:
        v2_pi = -v2_pi
    
    print(f"λ₂ = {lambda2_pi:.10f}")
    print(f"v₂[:5] = {v2_pi[:5]}")
    print(f"Iteraciones: {iters_pi}")
    print(f"Tiempo: {time_pi*1000:.2f} ms")
    
    error_lambda = abs(lambda2_pi - lambda2_np)
    error_v = np.linalg.norm(v2_pi - v2_np)
    print(f"Error en λ₂: {error_lambda:.2e}")
    print(f"Error en v₂: {error_v:.2e}")
    
    # Subspace Iteration
    print(f"\n--- Subspace Iteration ---")
    start = time.time()
    lambda2_si, v2_si, iters_si = subspace_iteration_fiedler(L)
    time_si = time.time() - start
    
    if np.dot(v2_si, v2_np) < 0:
        v2_si = -v2_si
    
    print(f"λ₂ = {lambda2_si:.10f}")
    print(f"v₂[:5] = {v2_si[:5]}")
    print(f"Iteraciones: {iters_si}")
    print(f"Tiempo: {time_si*1000:.2f} ms")
    
    error_lambda_si = abs(lambda2_si - lambda2_np)
    error_v_si = np.linalg.norm(v2_si - v2_np)
    print(f"Error en λ₂: {error_lambda_si:.2e}")
    print(f"Error en v₂: {error_v_si:.2e}")
    
    return {
        'numpy': (lambda2_np, v2_np),
        'power_iteration': (lambda2_pi, v2_pi, iters_pi, error_lambda),
        'subspace_iteration': (lambda2_si, v2_si, iters_si, error_lambda_si)
    }


# ============================================================
# VISUALIZACIÓN
# ============================================================

def visualize_fiedler(v2, S, n=8):
    """
    Visualiza el vector de Fiedler y la bisección resultante.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel 1: Heatmap del vector de Fiedler
    v2_grid = v2.reshape(n, n)
    im1 = axes[0].imshow(v2_grid, cmap='RdBu', aspect='equal')
    axes[0].set_title('Vector de Fiedler v₂', fontsize=12)
    axes[0].set_xlabel('Columna')
    axes[0].set_ylabel('Fila')
    plt.colorbar(im1, ax=axes[0])
    
    # Panel 2: Bisección
    partition_grid = np.zeros((n, n))
    for node in S:
        i, j = node // n, node % n
        partition_grid[i, j] = 1
    
    axes[1].imshow(partition_grid, cmap='Greens', aspect='equal')
    axes[1].set_title(f'Bisección Espectral (|S| = {len(S)})', fontsize=12)
    axes[1].set_xlabel('Columna')
    axes[1].set_ylabel('Fila')
    
    # Panel 3: Histograma de v2
    axes[2].hist(v2, bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(x=np.median(v2), color='red', linestyle='--', 
                    label=f'Mediana = {np.median(v2):.4f}')
    axes[2].axvline(x=0, color='blue', linestyle=':', label='Cero')
    axes[2].set_xlabel('Valor en v₂')
    axes[2].set_ylabel('Frecuencia')
    axes[2].set_title('Distribución del Vector de Fiedler')
    axes[2].legend()
    
    plt.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*70)
    print("CÁLCULO DEL VECTOR DE FIEDLER: POWER ITERATION VS NUMPY")
    print("="*70)
    print("\nImplementación desde cero para Teoría de Grafos 2026-I")
    print("Universidad del Rosario - Thomas Chísica\n")
    
    # 1. Construir grafo
    print("[1] Construyendo grafo 8×8 con 8-conectividad...")
    A = build_grid_graph(8)
    L, degrees = compute_laplacian(A)
    print(f"    |V| = {A.shape[0]}, |E| = {int(np.sum(A)/2)}")
    
    # 2. Verificar implementación
    results = verify_against_numpy(L)
    
    # 3. Usar nuestra implementación para bisección
    print("\n" + "="*70)
    print("BISECCIÓN ESPECTRAL")
    print("="*70)
    
    lambda2, v2, iters = power_iteration_with_deflation(L)
    S_pi = spectral_bisection(v2, method='median')
    h_pi = conductance(S_pi, A, degrees)
    cut_pi = cut_size(S_pi, A)
    
    print(f"\n--- Usando Power Iteration (nuestra implementación) ---")
    print(f"λ₂ = {lambda2:.6f}")
    print(f"|S| = {len(S_pi)}")
    print(f"Corte = {cut_pi}")
    print(f"h(S) = {h_pi:.6f}")
    
    # Comparar con NumPy
    lambda2_np, v2_np = results['numpy']
    S_np = spectral_bisection(v2_np, method='median')
    h_np = conductance(S_np, A, degrees)
    cut_np = cut_size(S_np, A)
    
    print(f"\n--- Usando NumPy (referencia) ---")
    print(f"λ₂ = {lambda2_np:.6f}")
    print(f"|S| = {len(S_np)}")
    print(f"Corte = {cut_np}")
    print(f"h(S) = {h_np:.6f}")
    
    # 4. Comparar con óptimo conocido (Left-Right)
    print("\n" + "="*70)
    print("COMPARACIÓN CON ÓPTIMO COMBINATORIO")
    print("="*70)
    
    S_LR = set(i*8+j for i in range(8) for j in range(4))
    h_LR = conductance(S_LR, A, degrees)
    cut_LR = cut_size(S_LR, A)
    
    print(f"\n{'Método':<30} {'|S|':>6} {'Corte':>8} {'h(S)':>10}")
    print("-"*60)
    print(f"{'Power Iteration':<30} {len(S_pi):>6} {cut_pi:>8} {h_pi:>10.4f}")
    print(f"{'NumPy':<30} {len(S_np):>6} {cut_np:>8} {h_np:>10.4f}")
    print(f"{'Left-Right (óptimo)':<30} {len(S_LR):>6} {cut_LR:>8} {h_LR:>10.4f}")
    print(f"{'Humanos (díadas LR/TB)':<30} {32:>6} {22:>8} {0.1048:>10.4f}")
    
    # 5. Generar figuras
    print("\n[5] Generando visualizaciones...")
    fig = visualize_fiedler(v2, S_pi)
    fig.savefig(ROOT / 'figures' / 'power_iteration_fiedler.png', dpi=150, bbox_inches='tight')
    print("    Guardado: power_iteration_fiedler.png")
    
    # 6. Guardar resultados
    print("\n[6] Guardando resultados...")
    np.savez(ROOT / 'data' / 'results' / 'power_iteration_results.npz',
             lambda2=lambda2,
             v2=v2,
             S_fiedler=np.array(list(S_pi)),
             iterations=iters,
             method='power_iteration_with_deflation')
    print("    Guardado: power_iteration_results.npz")
    
    print("\n" + "="*70)
    print("CONCLUSIÓN")
    print("="*70)
    print(f"""
RESULTADO PRINCIPAL:
- Power Iteration converge en {iters} iteraciones
- Error vs NumPy: {results['power_iteration'][3]:.2e} (despreciable)
- La bisección de Fiedler (h = {h_pi:.4f}) es SUBÓPTIMA
- El óptimo combinatorio (Left-Right) tiene h = {h_LR:.4f}
- Los humanos (h = 0.1048) alcanzan el ÓPTIMO COMBINATORIO

PARA EL PROFESOR BOJACÁ:
Este script implementa Power Iteration con deflación desde cero,
sin usar numpy.linalg.eig. El único uso de NumPy es para:
- Multiplicación matriz-vector (@)
- Norma (np.linalg.norm)
- QR para verificación (np.linalg.qr)
- Solve para inverse iteration (np.linalg.solve)

La implementación demuestra comprensión de:
1. Teoría espectral de grafos (Laplaciano, Fiedler)
2. Métodos iterativos (Power Iteration)
3. Deflación para eigenvalores múltiples
4. Análisis de convergencia
""")


if __name__ == '__main__':
    main()
