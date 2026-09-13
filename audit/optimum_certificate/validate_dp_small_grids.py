#!/usr/bin/env python3
"""Valida la programación dinámica por columnas contra fuerza bruta en grillas pequeñas.

Para grillas r x c de ocho y de cuatro vecinos con r*c <= 20, enumera todos los
subconjuntos y compara con la DP el corte mínimo por volumen exacto y el mínimo
global de conductancia h(S) = cut / min(vol(S), vol(S^c)).
La DP usa la misma recursión que optimum_dp_exhaustive.py, parametrizada en r y c.
Uso: python validate_dp_small_grids.py   (requiere NumPy)
"""
from fractions import Fraction
import numpy as np


def edges(r, c, king):
    out = []
    for i in range(r):
        for j in range(c):
            for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if not king and di and dj:
                    continue
                ii, jj = i + di, j + dj
                if 0 <= ii < r and 0 <= jj < c:
                    out.append((i * c + j, ii * c + jj))
    return out


def degrees(r, c, king):
    d = np.zeros(r * c, dtype=np.int64)
    for u, v in edges(r, c, king):
        d[u] += 1
        d[v] += 1
    return d


def brute_force(r, c, king):
    n = r * c
    E = edges(r, c, king)
    deg = degrees(r, c, king)
    masks = np.arange(1 << n, dtype=np.int64)
    bits = ((masks[:, None] >> np.arange(n)) & 1).astype(np.int8)
    cut = np.zeros(1 << n, dtype=np.int64)
    for u, v in E:
        cut += bits[:, u] ^ bits[:, v]
    vol = bits.astype(np.int64) @ deg
    V = int(deg.sum())
    cstar = np.full(V + 1, 10 ** 9, dtype=np.int64)
    np.minimum.at(cstar, vol, cut)
    return cstar


def dp(r, c, king):
    """Column-by-column DP: state = column pattern (2^r) x accumulated volume."""
    deg = degrees(r, c, king).reshape(r, c)
    pats = np.arange(1 << r)
    b = ((pats[:, None] >> np.arange(r)) & 1).astype(np.int64)
    within = (b[:, :-1] != b[:, 1:]).sum(1)
    cross = (b[:, None, :] != b[None, :, :]).sum(2)
    if king:
        cross = cross + (b[:, None, :-1] != b[None, :, 1:]).sum(2) + (b[:, None, 1:] != b[None, :, :-1]).sum(2)
    V = int(deg.sum())
    INF = 10 ** 9
    volcol = [b @ deg[:, j] for j in range(c)]
    table = np.full((1 << r, V + 1), INF, dtype=np.int64)
    for s in pats:
        table[s, volcol[0][s]] = within[s]
    for j in range(1, c):
        new = np.full_like(table, INF)
        for t in pats:
            best = (table + cross[:, t][:, None]).min(0)
            vt = volcol[j][t]
            new[t, vt:] = np.minimum(new[t, vt:], best[: V + 1 - vt] + within[t])
        table = new
    return table.min(0)


def hstar(cstar):
    V = len(cstar) - 1
    vals = [Fraction(int(cstar[v]), min(v, V - v)) for v in range(1, V) if cstar[v] < 10 ** 9]
    return min(vals)


def main():
    ok = True
    for r, c in [(3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (4, 5), (5, 4)]:
        for king in (True, False):
            bf, d = brute_force(r, c, king), dp(r, c, king)
            same = bool(np.array_equal(bf, d))
            ok &= same
            print(f"{r}x{c} {'ocho' if king else 'cuatro'} vecinos: corte mínimo por volumen idéntico={same}; "
                  f"h* fuerza bruta={hstar(bf)} DP={hstar(d)}")
    # Consistency with the 8x8 script: the parametric DP must reproduce dp_results.npz.
    try:
        saved = np.load("dp_results.npz")
        for king, key in ((True, "cstar_king"), (False, "cstar_grid4")):
            d8 = dp(8, 8, king)
            # Unreachable volumes are marked 10**6 in the 8x8 script and 10**9 here.
            reach_saved, reach_here = saved[key] < 10 ** 6, d8 < 10 ** 9
            same = bool(np.array_equal(reach_saved, reach_here) and np.array_equal(d8[reach_here], saved[key][reach_saved]))
            ok &= same
            print(f"8x8 {'ocho' if king else 'cuatro'} vecinos: igual a dp_results.npz={same}; h*={hstar(d8)}")
    except FileNotFoundError:
        print("dp_results.npz no encontrado: ejecute antes optimum_dp_exhaustive.py")
        ok = False
    print("TODAS COINCIDEN" if ok else "HAY DIFERENCIAS")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
