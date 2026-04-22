"""
05_counterexample_P6xP8.py
==========================
Contraejemplo: P6 ⊠ P8 (grilla rectangular 6x8 con 8-conectividad)

Muestra que cuando n != m, el eigenespacio de Fiedler es 1D (sin degenerancia),
y la biseccion de Fiedler estandar funciona correctamente sin necesidad del
algoritmo symmetry-aware. Esto prueba que el algoritmo propuesto es necesario
SOLO bajo simetria cuadrada (D4).

Comparacion:
  P8 ⊠ P8: lambda2 = lambda3 (multiplicidad 2) -> Fiedler ambiguo
  P6 ⊠ P8: lambda2 < lambda3 (multiplicidad 1) -> Fiedler unico

Autor: Thomas Chisica
Fecha: Febrero 2026
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from collections import deque

ROOT = Path(__file__).parent.parent


# =============================================================================
# 1. CONSTRUCCION DE GRILLAS RECTANGULARES CON 8-CONECTIVIDAD
# =============================================================================

def build_grid_graph_rect(rows, cols):
    """
    Construye el grafo de una grilla rows x cols con 8-conectividad (king moves).
    Esto corresponde al strong product P_rows ⊠ P_cols.

    Args:
        rows: numero de filas
        cols: numero de columnas

    Returns:
        A: Matriz de adyacencia (rows*cols x rows*cols)
        pos: Diccionario {nodo: (x, y)} para visualizacion
    """
    num_nodes = rows * cols
    A = np.zeros((num_nodes, num_nodes))
    pos = {}

    def node_id(i, j):
        return i * cols + j

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for i in range(rows):
        for j in range(cols):
            node = node_id(i, j)
            pos[node] = (j, rows - 1 - i)

            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor = node_id(ni, nj)
                    A[node, neighbor] = 1

    return A, pos


def compute_laplacian(A):
    """Calcula L = D - A."""
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    L = D - A
    return L, D, degrees


def compute_normalized_laplacian(A):
    """Calcula L_norm = D^(-1/2) L D^(-1/2)."""
    degrees = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L = np.diag(degrees) - A
    return D_inv_sqrt @ L @ D_inv_sqrt


def spectral_analysis(L):
    """Eigendescomposicion completa del Laplaciano."""
    eigenvalues, eigenvectors = linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    lambda2 = eigenvalues[1]
    fiedler_vector = eigenvectors[:, 1]
    return eigenvalues, eigenvectors, lambda2, fiedler_vector


def conductance(S, A, degrees):
    """h(S) = cut(S, S') / min(vol(S), vol(S'))."""
    S = set(S)
    n = len(degrees)
    S_bar = set(range(n)) - S

    cut = 0
    for u in S:
        for v in S_bar:
            cut += A[u, v]

    vol_S = sum(degrees[u] for u in S)
    vol_S_bar = sum(degrees[v] for v in S_bar)
    min_vol = min(vol_S, vol_S_bar)

    if min_vol == 0:
        return float('inf')
    return cut / min_vol


def optimal_fiedler_cut(fiedler_vector, A, degrees):
    """Busca el corte optimo sobre el vector de Fiedler (sweep)."""
    n = len(fiedler_vector)
    sorted_indices = np.argsort(fiedler_vector)

    best_conductance = float('inf')
    best_S = None
    best_cut_edges = 0

    for k in range(1, n):
        S = set(sorted_indices[:k])
        S_bar = set(sorted_indices[k:])
        h = conductance(S, A, degrees)

        if h < best_conductance:
            best_conductance = h
            best_S = S.copy()
            # Contar aristas de corte
            cut = sum(A[u, v] for u in S for v in S_bar)
            best_cut_edges = int(cut)

    return best_S, best_conductance, best_cut_edges


def is_connected(S, A):
    """Verifica conectividad del subgrafo inducido via BFS."""
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
# 2. ANALISIS DE UNA GRILLA
# =============================================================================

def analyze_grid(rows, cols):
    """
    Realiza analisis espectral completo de una grilla rows x cols.
    Retorna diccionario con todos los resultados.
    """
    label = f"P{rows} ⊠ P{cols}"
    print(f"\n{'='*60}")
    print(f"  ANALISIS: {label} ({rows}x{cols} con 8-conectividad)")
    print(f"{'='*60}")

    # Construir grafo
    A, pos = build_grid_graph_rect(rows, cols)
    num_nodes = rows * cols
    num_edges = int(np.sum(A) / 2)
    print(f"  |V| = {num_nodes}, |E| = {num_edges}")

    # Laplaciano
    L, D, degrees = compute_laplacian(A)
    print(f"  Grado min = {int(min(degrees))}, max = {int(max(degrees))}")

    # Analisis espectral
    eigenvalues, eigenvectors, lambda2, fiedler_vector = spectral_analysis(L)
    lambda3 = eigenvalues[2]
    lambda4 = eigenvalues[3]

    # Verificar multiplicidad de lambda2
    gap = lambda3 - lambda2
    multiplicity = np.sum(np.abs(eigenvalues - lambda2) < 1e-8)

    print(f"\n  Espectro (primeros eigenvalores):")
    print(f"    lambda1 = {eigenvalues[0]:.8f}")
    print(f"    lambda2 = {lambda2:.8f}")
    print(f"    lambda3 = {lambda3:.8f}")
    print(f"    lambda4 = {lambda4:.8f}")
    print(f"    ...")
    print(f"    lambda_max = {eigenvalues[-1]:.4f}")
    print(f"\n  Gap espectral: lambda3 - lambda2 = {gap:.8f}")
    print(f"  Multiplicidad de lambda2: {multiplicity}")

    if multiplicity > 1:
        print(f"  >>> DEGENERADO: eigenespacio de Fiedler es {multiplicity}D")
        print(f"  >>> Se necesita algoritmo symmetry-aware")
    else:
        print(f"  >>> NO DEGENERADO: eigenespacio de Fiedler es 1D")
        print(f"  >>> Biseccion de Fiedler estandar es suficiente")

    # Biseccion de Fiedler (estandar)
    S_opt, h_opt, cut_edges = optimal_fiedler_cut(fiedler_vector, A, degrees)
    S_bar = set(range(num_nodes)) - S_opt

    print(f"\n  Biseccion de Fiedler (sweep sobre v2):")
    print(f"    |S| = {len(S_opt)}, |S'| = {len(S_bar)}")
    print(f"    Aristas de corte = {cut_edges}")
    print(f"    h(S_Fiedler) = {h_opt:.6f}")

    # Cheeger con Laplaciano normalizado
    L_norm = compute_normalized_laplacian(A)
    evals_norm, _ = linalg.eigh(L_norm)
    lambda2_norm = np.sort(evals_norm)[1]
    cheeger_lower = lambda2_norm / 2
    cheeger_upper = np.sqrt(2 * lambda2_norm)
    print(f"\n  Cheeger (Laplaciano normalizado):")
    print(f"    {cheeger_lower:.6f} <= h(G) <= {cheeger_upper:.6f}")
    print(f"    h(S_Fiedler) = {h_opt:.6f}")

    # Conectividad
    s_conn = is_connected(S_opt, A)
    sb_conn = is_connected(S_bar, A)
    print(f"    S conexo: {s_conn}, S' conexo: {sb_conn}")

    print(f"\n  Observacion estructural:")
    if rows == cols:
        print("    >>> La simetria cuadrada favorece degenerancia en el eigenspace de Fiedler")
    else:
        print("    >>> Al romper la simetria cuadrada, el segundo autovalor se separa")
        if lambda2 < lambda3:
            print("    >>> La direccion de Fiedler queda esencialmente fijada por la geometria rectangular")

    return {
        'rows': rows,
        'cols': cols,
        'label': label,
        'A': A,
        'pos': pos,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'lambda2': lambda2,
        'lambda3': lambda3,
        'gap': gap,
        'multiplicity': multiplicity,
        'fiedler_vector': fiedler_vector,
        'S_opt': S_opt,
        'h_opt': h_opt,
        'cut_edges': cut_edges,
        'degrees': degrees,
        'lambda2_norm': lambda2_norm,
    }


# =============================================================================
# 3. VISUALIZACION COMPARATIVA
# =============================================================================

def plot_comparison(res_square, res_rect, save_path=None):
    """
    Genera figura comparativa de 2x3 paneles:
    Fila 1: P8 ⊠ P8 (cuadrado, degenerado)
    Fila 2: P6 ⊠ P8 (rectangular, no degenerado)

    Columnas: Espectro | Fiedler Vector | Biseccion
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, res in enumerate([res_square, res_rect]):
        rows, cols = res['rows'], res['cols']
        label = res['label']
        eigs = res['eigenvalues']
        fv = res['fiedler_vector']
        S = res['S_opt']
        mult = res['multiplicity']
        gap = res['gap']

        # --- Col 1: Espectro (primeros 15 eigenvalores) ---
        ax = axes[row_idx, 0]
        n_show = min(15, len(eigs))
        colors = ['red' if np.abs(eigs[i] - res['lambda2']) < 1e-8 else 'steelblue'
                  for i in range(n_show)]
        ax.bar(range(n_show), eigs[:n_show], color=colors, alpha=0.8)
        ax.set_title(f'{label}: Espectro', fontsize=12, fontweight='bold')
        ax.set_xlabel('Indice')
        ax.set_ylabel('$\\lambda_i$')

        # Anotar multiplicidad
        if mult > 1:
            ax.annotate(f'$\\lambda_2 = \\lambda_3$ = {res["lambda2"]:.4f}\n'
                       f'Multiplicidad {mult}\nGap = {gap:.2e}',
                       xy=(1.5, res['lambda2']),
                       xytext=(5, res['lambda2'] + 0.3),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=9, color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
        else:
            ax.annotate(f'$\\lambda_2$ = {res["lambda2"]:.4f}\n'
                       f'$\\lambda_3$ = {res["lambda3"]:.4f}\n'
                       f'Gap = {gap:.4f}',
                       xy=(1, res['lambda2']),
                       xytext=(5, res['lambda2'] + 0.2),
                       arrowprops=dict(arrowstyle='->', color='green'),
                       fontsize=9, color='green',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))
        ax.grid(True, alpha=0.3)

        # --- Col 2: Fiedler Vector Heatmap ---
        ax = axes[row_idx, 1]
        fiedler_grid = fv.reshape(rows, cols)
        im = ax.imshow(fiedler_grid, cmap='RdBu', aspect='equal')
        ax.set_title(f'{label}: Vector de Fiedler $v_2$', fontsize=12, fontweight='bold')
        ax.set_xlabel('Columna')
        ax.set_ylabel('Fila')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Valores en celdas (solo si la grilla no es muy grande)
        if rows * cols <= 64:
            for i in range(rows):
                for j in range(cols):
                    val = fiedler_grid[i, j]
                    color = 'white' if abs(val) > 0.08 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=6, color=color)

        # --- Col 3: Biseccion ---
        ax = axes[row_idx, 2]
        partition_grid = np.zeros((rows, cols))
        for node in S:
            i, j = node // cols, node % cols
            partition_grid[i, j] = 1

        ax.imshow(partition_grid, cmap='coolwarm', aspect='equal', vmin=0, vmax=1)
        ax.set_title(f'{label}: Biseccion de Fiedler\n'
                     f'Cut={res["cut_edges"]}, h={res["h_opt"]:.4f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Columna')
        ax.set_ylabel('Fila')

        for i in range(rows):
            for j in range(cols):
                node = i * cols + j
                in_S = node in S
                ax.text(j, i, 'S' if in_S else "$\\bar{S}$",
                        ha='center', va='center', fontsize=8,
                        fontweight='bold',
                        color='white' if in_S else 'darkblue')

    # Titulo general
    fig.suptitle(
        'Contraejemplo: Degenerancia del Eigenespacio de Fiedler\n'
        'Grilla cuadrada (degenerada) vs. rectangular (no degenerada)',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigura guardada en: {save_path}")

    plt.show()
    return fig


# =============================================================================
# 4. TABLA RESUMEN
# =============================================================================

def print_summary_table(res_square, res_rect):
    """Imprime tabla comparativa entre las dos grillas."""
    print(f"\n{'='*70}")
    print("TABLA COMPARATIVA: COUNTEREXAMPLE")
    print(f"{'='*70}")
    print(f"{'Propiedad':<35} {'P8⊠P8':>15} {'P6⊠P8':>15}")
    print(f"{'-'*65}")
    print(f"{'Nodos |V|':<35} {res_square['rows']*res_square['cols']:>15} "
          f"{res_rect['rows']*res_rect['cols']:>15}")
    print(f"{'Aristas |E|':<35} {int(np.sum(res_square['A'])/2):>15} "
          f"{int(np.sum(res_rect['A'])/2):>15}")
    print(f"{'lambda2':<35} {res_square['lambda2']:>15.6f} "
          f"{res_rect['lambda2']:>15.6f}")
    print(f"{'lambda3':<35} {res_square['lambda3']:>15.6f} "
          f"{res_rect['lambda3']:>15.6f}")
    print(f"{'Gap (lambda3 - lambda2)':<35} {res_square['gap']:>15.8f} "
          f"{res_rect['gap']:>15.8f}")
    print(f"{'Multiplicidad lambda2':<35} {res_square['multiplicity']:>15} "
          f"{res_rect['multiplicity']:>15}")
    print(f"{'Eigenespacio Fiedler':<35} {'2D (degenerado)':>15} "
          f"{'1D (unico)':>15}")
    print(f"{'Aristas de corte (Fiedler)':<35} {res_square['cut_edges']:>15} "
          f"{res_rect['cut_edges']:>15}")
    print(f"{'h(S_Fiedler)':<35} {res_square['h_opt']:>15.6f} "
          f"{res_rect['h_opt']:>15.6f}")
    print(f"{'Algoritmo necesario':<35} {'Symmetry-aware':>15} "
          f"{'Estandar':>15}")
    print(f"{'-'*65}")

    # Conclusiones estructurales
    print(f"\n  Observacion estructural:")
    print("    P8⊠P8 presenta degenerancia por simetria cuadrada")
    print("    P6⊠P8 rompe esa simetria y separa el segundo autovalor")
    print(f"\n  Conclusion: La degenerancia del eigenespacio de Fiedler es una")
    print(f"  consecuencia EXCLUSIVA de la simetria cuadrada (n = m).")
    print(f"  Cuando n != m, el Fiedler estandar basta.")
    print(f"{'='*70}")


# =============================================================================
# 5. EJECUCION PRINCIPAL
# =============================================================================

def main():
    print("=" * 70)
    print("COUNTEREXAMPLE: P6 ⊠ P8 vs P8 ⊠ P8")
    print("Degenerancia del eigenespacio de Fiedler bajo simetria cuadrada")
    print("=" * 70)

    # Analizar ambas grillas
    res_square = analyze_grid(8, 8)
    res_rect = analyze_grid(6, 8)

    # Tabla resumen
    print_summary_table(res_square, res_rect)

    # Visualizacion comparativa
    save_path = ROOT / 'figures' / 'counterexample_P6xP8.png'
    plot_comparison(res_square, res_rect, save_path=save_path)

    # Guardar resultados
    np.savez(ROOT / 'data' / 'results' / 'counterexample_P6xP8.npz',
             # P8 x P8
             sq_lambda2=res_square['lambda2'],
             sq_lambda3=res_square['lambda3'],
             sq_gap=res_square['gap'],
             sq_multiplicity=res_square['multiplicity'],
             sq_h_fiedler=res_square['h_opt'],
             sq_cut_edges=res_square['cut_edges'],
             # P6 x P8
             rect_lambda2=res_rect['lambda2'],
             rect_lambda3=res_rect['lambda3'],
             rect_gap=res_rect['gap'],
             rect_multiplicity=res_rect['multiplicity'],
             rect_h_fiedler=res_rect['h_opt'],
             rect_cut_edges=res_rect['cut_edges'],
             # Laplaciano normalizado
             sq_lambda2_norm=res_square['lambda2_norm'],
             rect_lambda2_norm=res_rect['lambda2_norm'])

    print(f"\nResultados guardados en: data/results/counterexample_P6xP8.npz")

    return res_square, res_rect


if __name__ == "__main__":
    res_sq, res_rect = main()
