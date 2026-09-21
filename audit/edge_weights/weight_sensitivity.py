"""Sensibilidad del optimo de conductancia al peso de las aristas del tablero.

Tres familias de ponderacion, todas geometricas: no usan datos de participantes
y no ajustan modelos.

  A. Descuento uniforme de la diagonal:  w_ortogonal = 1, w_diagonal = w.
  B. Anisotropia:  w_horizontal = 1, w_vertical = b, w_diagonal = 1.
  C. Nucleo gaussiano sobre todas las parejas de casillas, w_ij = exp(-d2/(2 s^2)),
     truncado por debajo de una tolerancia. Añade aristas mas alla de los ocho vecinos.

Para cada ponderacion se resuelve el minimo global de conductancia con el metodo
de Dinkelbach sobre un MILP entero, no solo se comparan particiones candidatas.

Uso: python audit/edge_weights/weight_sensitivity.py
"""
import itertools
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

n = 8
N = n * n
yy, xx = np.indices((n, n))
X = xx.ravel()
Y = yy.ravel()

# ---------------------------------------------------------------- particiones
def _staircase():
    """Frontera en dos escalones: filas 0-3 cortan en x<3, filas 4-7 en x<5."""
    return np.where(Y < 4, X < 3, X < 5)

CANDIDATES = {
    "LR (mitades)": X < 4,
    "TB (mitades)": Y < 4,
    "Diagonal": (X + Y) < 7,
    "IN 6x6 / OUT": (X > 0) & (X < 7) & (Y > 0) & (Y < 7),
    "IN 4x4 / OUT": (X > 1) & (X < 6) & (Y > 1) & (Y < 6),
    "Escalera (dos escalones)": _staircase(),
    "Cuadrantes opuestos": ((X < 4) & (Y < 4)) | ((X >= 4) & (Y >= 4)),
    "Cuadrante 4x4": (X < 4) & (Y < 4),
}

# -------------------------------------------------------------------- grafos
def eight_neighbour_edges():
    edges = []
    for r in range(n):
        for c in range(n):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    kind = "diag" if (dr and dc) else ("horiz" if dr == 0 else "vert")
                    edges.append((r * n + c, rr * n + cc, kind))
    return edges

EIGHT = eight_neighbour_edges()


def graph_uniform_diagonal(w):
    u = np.array([e[0] for e in EIGHT])
    v = np.array([e[1] for e in EIGHT])
    wt = np.array([w if e[2] == "diag" else 1.0 for e in EIGHT])
    return u, v, wt


def graph_anisotropic(b):
    u = np.array([e[0] for e in EIGHT])
    v = np.array([e[1] for e in EIGHT])
    wt = np.array([{"horiz": 1.0, "vert": b, "diag": 1.0}[e[2]] for e in EIGHT])
    return u, v, wt


def graph_gaussian(sigma, tol=1e-3):
    pos = np.stack([X, Y], axis=1).astype(float)
    u, v, wt = [], [], []
    for i, j in itertools.combinations(range(N), 2):
        d2 = float(((pos[i] - pos[j]) ** 2).sum())
        w = np.exp(-d2 / (2.0 * sigma * sigma))
        if w >= tol:
            u.append(i)
            v.append(j)
            wt.append(w)
    return np.array(u), np.array(v), np.array(wt)


def degrees(u, v, wt):
    d = np.zeros(N)
    np.add.at(d, u, wt)
    np.add.at(d, v, wt)
    return d


def conductance(S, u, v, wt, d=None):
    d = degrees(u, v, wt) if d is None else d
    cut = wt[S[u] != S[v]].sum()
    vol = d[S].sum()
    other = d.sum() - vol
    lo = min(vol, other)
    return cut / lo if lo > 0 else np.nan


# ------------------------------------------------------- optimo global (MILP)
def global_optimum(u, v, wt, time_limit=300, max_iter=15):
    """Dinkelbach sobre un MILP entero. Devuelve (h*, S*, iteraciones, estado).

    En cada iteracion se minimiza  sum_e w_e y_e - lam * sum_i d_i x_i  con
    y_e >= |x_u - x_v|. Un valor objetivo no negativo certifica que ningun
    conjunto mejora lam; en ese caso el optimizador devuelto es el conjunto
    vacio y hay que conservar el incumbente, no reportarlo.
    """
    d = degrees(u, v, wt)
    M = len(wt)
    best_S = None
    best = np.inf
    for S in CANDIDATES.values():
        h = conductance(S, u, v, wt, d)
        if h < best:
            best, best_S = h, S.copy()
    lam = best
    rows = []
    lb = []
    ub = []
    for k in range(M):
        for a, b in ((u[k], v[k]), (v[k], u[k])):
            row = np.zeros(N + M)
            row[a] = 1.0
            row[b] = -1.0
            row[N + k] = -1.0
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)
    row = np.zeros(N + M)
    row[:N] = d
    rows.append(row)
    lb.append(1e-6)
    ub.append(d.sum() / 2.0)
    A = np.array(rows)
    cons = LinearConstraint(A, np.array(lb), np.array(ub))
    status = None
    for it in range(1, max_iter + 1):
        cost = np.concatenate([-lam * d, wt])
        res = milp(
            cost,
            constraints=cons,
            integrality=np.ones(N + M),
            bounds=Bounds(0, 1),
            options={"time_limit": time_limit, "mip_rel_gap": 0},
        )
        status = res.status
        if res.x is None:
            return lam, best_S, it, status
        S = res.x[:N] > 0.5
        vol = d[S].sum()
        other = d.sum() - vol
        if min(vol, other) <= 0:
            # conjunto vacio o total: certificado de que lam no se puede mejorar
            return lam, best_S, it, status
        new_lam = wt[S[u] != S[v]].sum() / min(vol, other)
        if new_lam < lam - 1e-12:
            lam, best_S = new_lam, S.copy()
        else:
            return lam, best_S, it, status
    return lam, best_S, max_iter, status


def label(S):
    hits = [k for k, c in CANDIDATES.items() if np.array_equal(c, S) or np.array_equal(~c, S)]
    return ", ".join(hits) if hits else "otra"


def report(title, note, graphs):
    print(f"\n=== {title} ===")
    print(note)
    keys = list(graphs)
    header = "".join(f"{k:>11}" for k in keys)
    print(f"\n{'particion':30s}{header}")
    cached = {k: graphs[k]() for k in keys}
    for name, S in CANDIDATES.items():
        cells = "".join(f"{conductance(S, *cached[k]):11.4f}" for k in keys)
        print(f"{name:30s}{cells}")
    print("\noptimo global (MILP entero, Dinkelbach, gap 0):")
    for k in keys:
        u, v, wt = cached[k]
        t0 = time.time()
        h, S, it, st = global_optimum(u, v, wt)
        print(
            f"  {k:>11}  h*={h:.6f}  |S|={int(S.sum()):2d}  iter={it}  status={st}  "
            f"coincide con: {label(S)}  [{time.time() - t0:.1f}s]"
        )


if __name__ == "__main__":
    report(
        "Familia A: descuento uniforme de la diagonal",
        "w = peso de la arista diagonal; las ortogonales valen 1.",
        {f"w={w:g}": (lambda w=w: graph_uniform_diagonal(w)) for w in (1.0, 0.9, 0.8, 0.7071, 0.6, 0.5, 0.3)},
    )
    report(
        "Familia B: anisotropia horizontal / vertical",
        "b = peso de la arista vertical; horizontal y diagonal valen 1.",
        {f"b={b:g}": (lambda b=b: graph_anisotropic(b)) for b in (1.0, 0.8, 0.6, 0.4, 1.5)},
    )
    report(
        "Familia C: nucleo gaussiano sobre todas las parejas",
        "s = escala del nucleo exp(-d^2/(2 s^2)); se conservan pesos >= 1e-3.",
        {f"s={s:g}": (lambda s=s: graph_gaussian(s)) for s in (0.7, 1.0, 1.5, 2.5)},
    )
