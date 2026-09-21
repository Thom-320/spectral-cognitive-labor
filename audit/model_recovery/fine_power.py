#!/usr/bin/env python3
"""Rejilla fina de potencia en el tramo donde la curva principal salta.

parametric_power.py evalua delta en multiplos 0, 0.5, 1, 1.5, 2.5 y 4 del valor
ajustado, y entre el primero y el segundo la potencia pasa de 0,02 a 1,00. Este
script rellena ese tramo para localizar el efecto minimo detectable.

Uso: python audit/model_recovery/fine_power.py [--reps N]
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parametric_models as pm  # noqa: E402
import parametric_power as pp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=60)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=4242)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pre, meta = pm.load_transitions('original')
    fits = pm.fit_ladder(pre, seed=1, polish=True)
    base = fits['FRA'][0].copy()
    d_fit = float(np.exp(base[6]))
    print(f"delta ajustado sobre humanos: {d_fit:.4f}; {args.reps} replicas por punto\n")
    print(f"{'delta':>8}{'efecto oraculo':>16}{'P(FRA elegido)':>16}{'P(IC>0)':>10}"
          f"{'delta medio':>13}")
    rows = []
    for mult in [0.05, 0.10, 0.15, 0.20, 0.30]:
        th = base.copy()
        th[6] = np.log(d_fit * mult)
        od = pp.oracle_delta(th, rng)
        r = pp.run(th, 'FRA', args.reps, rng, folds=args.folds)
        p = r['selected']['FRA'] / args.reps
        print(f'{d_fit * mult:8.4f}{od:16.4f}{p:16.2f}{r["power"]:10.2f}'
              f'{r["delta_mean"]:13.4f}', flush=True)
        rows.append(dict(delta=d_fit * mult, oracle=od, p_select_FRA=p,
                         power=r['power'], delta_mean=r['delta_mean']))
    Path(__file__).with_name('fine_power.json').write_text(
        json.dumps(dict(reps=args.reps, delta_fit=d_fit, meta=meta, rows=rows),
                   indent=2, default=float))
    print('\nEscrito: audit/model_recovery/fine_power.json')


if __name__ == '__main__':
    main()
