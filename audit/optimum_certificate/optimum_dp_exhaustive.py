"""Optimo global EXACTO de h(S)=cut/min(vol(S),vol(Sb)) por programacion dinamica columna a columna.
Estado: patron de la columna (2^8=256) x volumen acumulado (0..420). Exhaustivo sobre los 2^64 subconjuntos.
Tambien: min cut con |S|=k para todo k (bisection width en k=32) y expansion cut/min(|S|,|Sb|)."""
import numpy as np, itertools, time

n = 8
pats = np.arange(256)
bits = ((pats[:, None] >> np.arange(n)[None, :]) & 1).astype(int)  # [256, 8] fila i = bit i


def run(king=True, tag=""):
    t0 = time.time()
    # grados por celda (i,j)
    def deg(i, j):
        c = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0: continue
                if not king and di != 0 and dj != 0: continue
                if 0 <= i + di < n and 0 <= j + dj < n: c += 1
        return c
    D = np.array([[deg(i, j) for j in range(n)] for i in range(n)])
    volV = D.sum(); E = volV // 2
    # within-column cut (vertical edges) para patron s
    within = (bits[:, :-1] != bits[:, 1:]).sum(1)  # [256]
    # cross-column cut entre s (col j) y t (col j+1)
    horiz = (bits[:, None, :] != bits[None, :, :]).sum(2)  # [256,256]
    cross = horiz.copy()
    if king:
        cross = cross + (bits[:, None, :-1] != bits[None, :, 1:]).sum(2) + (bits[:, None, 1:] != bits[None, :, :-1]).sum(2)
    INF = 10 ** 6
    # --- DP sobre volumen ---
    V = volV + 1
    volcol = [bits @ D[:, j] for j in range(n)]  # [256] volumen de patron en columna j
    dp = np.full((256, V), INF, dtype=np.int64)
    for s in range(256):
        dp[s, volcol[0][s]] = within[s]
    for j in range(1, n):
        new = np.full((256, V), INF, dtype=np.int64)
        for t in range(256):
            best = (dp + cross[:, t][:, None]).min(0)  # [V] min sobre s
            vt = volcol[j][t]
            new[t, vt:] = np.minimum(new[t, vt:], best[: V - vt] + within[t])
        dp = new
    cstar = dp.min(0)  # min cut para cada volumen exacto
    hs = np.full(V, np.inf)
    for v in range(1, volV):
        if cstar[v] < INF:
            hs[v] = cstar[v] / min(v, volV - v)
    vbest = int(np.argmin(hs))
    print(f"[{tag}] vol(V)={volV} |E|={E}  tiempo={time.time()-t0:.1f}s")
    print(f"  OPTIMO GLOBAL exacto: h* = {hs[vbest]:.6f} = {int(cstar[vbest])}/{min(vbest, volV-vbest)}  (vol(S)={vbest})")
    ties = [(v, int(cstar[v])) for v in range(1, volV) if abs(hs[v] - hs[vbest]) < 1e-12]
    print(f"  volumenes que alcanzan h*: {ties}")
    order = np.argsort(hs)[:10]
    print("  10 mejores (vol, cut, h):", [(int(v), int(cstar[v]), round(float(hs[v]), 5)) for v in order])
    print(f"  h(LR)= {22 if king else 8}/{E} = {(22 if king else 8)/E:.6f}")
    # --- DP sobre numero de nodos (bisection width / expansion) ---
    K = 65
    cnt = bits.sum(1)
    dpk = np.full((256, K), INF, dtype=np.int64)
    for s in range(256):
        dpk[s, cnt[s]] = within[s]
    for j in range(1, n):
        new = np.full((256, K), INF, dtype=np.int64)
        for t in range(256):
            best = (dpk + cross[:, t][:, None]).min(0)
            kt = cnt[t]
            new[t, kt:] = np.minimum(new[t, kt:], best[: K - kt] + within[t])
        dpk = new
    ck = dpk.min(0)
    print(f"  bisection width (min cut, |S|=32): {int(ck[32])}")
    exp_ = [ck[k] / min(k, 64 - k) for k in range(1, 64)]
    kb = int(np.argmin(exp_)) + 1
    print(f"  min expansion cut/min(|S|,|Sb|) = {exp_[kb-1]:.5f} en |S|={kb} (cut={int(ck[kb])})")
    print("  min cut por |S| (k: cut) para k=8..32:", {k: int(ck[k]) for k in range(8, 33, 2)})
    return cstar, ck


cs_k, ck_k = run(True, "king P8xP8")
cs_4, ck_4 = run(False, "grid4 P8[]P8")
np.savez("dp_results.npz", cstar_king=cs_k, ck_king=ck_k, cstar_grid4=cs_4, ck_grid4=ck_4)
