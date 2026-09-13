#!/usr/bin/env python3
"""Certificate: global minimum conductance of the unweighted 8x8 king graph (P8 strong P8).

Definition: h(S) = cut(S, S^c) / min(vol(S), vol(S^c)), vol = sum of degrees, vol(V) = 420.
Every nontrivial partition has a side with 1 <= vol <= 210, so
    min_S h(S) >= 22/210   <=>   min over {S : 1 <= vol(S) <= 210} of 210*cut(S) - 22*vol(S) >= 0.
All coefficients are integers, so a proven optimum of 0 (MIP gap 0) is exact.
Then enumerate every S with objective <= 0 via no-good cuts, to list all minimizers.
No participant data. Requires numpy and scipy (HiGHS).
"""
import json, sys
import numpy as np
import scipy
from scipy.optimize import milp, LinearConstraint, Bounds

n = 8
N = n * n
E = []
for r in range(n):
    for c in range(n):
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                E.append((r * n + c, rr * n + cc))
M = len(E)
deg = np.zeros(N, dtype=int)
for u, v in E:
    deg[u] += 1
    deg[v] += 1
assert M == 210 and deg.sum() == 420

# Variables: x_i in {0,1} (i in S), y_e in {0,1} with y_e >= |x_u - x_v|.
rows, lb, ub = [], [], []
for k, (u, v) in enumerate(E):
    for a, b in ((u, v), (v, u)):
        row = np.zeros(N + M); row[a] = 1; row[b] = -1; row[N + k] = -1
        rows.append(row); lb.append(-np.inf); ub.append(0)
row = np.zeros(N + M); row[:N] = deg
rows.append(row); lb.append(1); ub.append(210)
cost = np.concatenate([-22 * deg, 210 * np.ones(M)]).astype(float)


def solve(extra_rows, extra_lb, extra_ub):
    A = np.array(rows + extra_rows)
    res = milp(cost, constraints=LinearConstraint(A, lb + extra_lb, ub + extra_ub),
               integrality=np.ones(N + M), bounds=Bounds(0, 1),
               options={"time_limit": 600, "mip_rel_gap": 0})
    return res


def describe(x):
    S = np.where(x > 0.5)[0]
    cut = sum(1 for u, v in E if (x[u] > 0.5) != (x[v] > 0.5))
    vol = int(deg[S].sum())
    grid = x[:N].reshape(n, n).round().astype(int)
    return S, cut, vol, grid


report = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
          "edges": M, "vol_V": int(deg.sum()), "minimizers": []}
extra_rows, extra_lb, extra_ub = [], [], []
for it in range(10):
    res = solve(extra_rows, extra_lb, extra_ub)
    obj = round(res.fun) if res.status == 0 else None
    entry = {"iteration": it, "status": int(res.status), "message": res.message,
             "objective_210cut_minus_22vol": obj,
             "mip_gap": float(getattr(res, "mip_gap", float("nan"))),
             "dual_bound": float(getattr(res, "mip_dual_bound", float("nan")))}
    if res.status != 0 or obj > 0:
        entry["note"] = "no further set with objective <= 0" if res.status == 0 else "solver did not prove optimality"
        report.setdefault("stops", []).append(entry)
        break
    x = res.x[:N]
    S, cut, vol, grid = describe(res.x[:N])
    entry.update({"size": int(len(S)), "cut": int(cut), "vol": vol, "h": cut / vol,
                  "grid_rows": ["".join(map(str, r)) for r in grid]})
    report["minimizers"].append(entry)
    # no-good cut excluding exactly this S: sum_{i in S}(1-x_i) + sum_{i not in S} x_i >= 1
    row = np.zeros(N + M)
    inS = np.zeros(N, bool); inS[S] = True
    row[:N] = np.where(inS, -1.0, 1.0)
    extra_rows.append(row); extra_lb.append(1 - inS.sum()); extra_ub.append(np.inf)
print(json.dumps(report, indent=2))
