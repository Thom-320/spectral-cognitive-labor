#!/usr/bin/env python3
"""¿Cuanto optimismo introduce elegir kappa fuera de los pliegues?

En power_recovery.py la constante de suavizado se elige una vez por replica con
validacion cruzada interna sobre toda la muestra simulada, incluida la diada que
despues se deja fuera. Es una fuga pequena, un escalar, pero es una fuga. Aqui se
compara esa version con la estricta, que vuelve a elegir kappa dentro de cada
pliegue de entrenamiento, con menos replicas porque cuesta unas seis veces mas.

Uso: python audit/model_recovery/nested_kappa_check.py [--reps N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_recovery as pr  # noqa: E402


def strict_loo(S, rng):
    """LOO con kappa reelegido dentro de cada conjunto de entrenamiento."""
    per = pr._dyad_counts(S)
    nd = len(per)
    z, s, o, y, dy = S['z'], S['s'], S['o'], S['y'], S['dyad']
    full = sum(per)
    out = np.zeros((len(pr.MODELS), len(z)))
    for d in range(nd):
        sel = dy == d
        if not sel.any():
            continue
        keep = np.flatnonzero(dy != d)
        train = {k: v[keep] for k, v in S.items()}
        train['dyad'] = np.unique(train['dyad'], return_inverse=True)[1]
        kappas = pr.choose_kappa(train, pr._dyad_counts(train), rng)
        C = full - per[d]
        fits = {k: pr.fit(C, k) for k in set(kappas.values())}
        for m, name in enumerate(pr.MODELS):
            p = pr.predict(fits[kappas[name]], name, z[sel], s[sel], o[sel])
            out[m, sel] = -np.log(p[np.arange(int(sel.sum())), y[sel]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    T, meta = pr.load('absent')
    P = pr.fit(pr.count(T['z'], T['s'], T['o'], T['y']))
    tables = {m: pr.as_table(P, m) for m in pr.MODELS}
    gen = pr.Generator(T, rng)

    print(f"replicas: {args.reps}; diadas: {meta['n_dyads']}; "
          f"transiciones: {meta['n_transitions']}\n")
    print(f"{'generador':12s}{'recupera (rapido)':>20s}{'recupera (estricto)':>22s}"
          f"{'delta rapido':>14s}{'delta estricto':>16s}")
    rows = []
    for g in ['M1', 'M2', 'M3']:
        fast = strict = 0
        df, ds = [], []
        for _ in range(args.reps):
            S = gen.draw(tables[g], rng=rng)
            Lf, _ = pr.loo_losses(S, kappa='cv', rng=rng)
            Ls = strict_loo(S, rng)
            fast += int(pr.MODELS[int(np.argmin(Lf.mean(1)))] == g)
            strict += int(pr.MODELS[int(np.argmin(Ls.mean(1)))] == g)
            df.append(float((Lf[2] - Lf[3]).mean()))
            ds.append(float((Ls[2] - Ls[3]).mean()))
        print(f'{g:12s}{fast / args.reps:20.2f}{strict / args.reps:22.2f}'
              f'{np.mean(df):14.4f}{np.mean(ds):16.4f}')
        rows.append(dict(generator=g, fast=fast / args.reps, strict=strict / args.reps,
                         delta_fast=float(np.mean(df)), delta_strict=float(np.mean(ds))))
    Path(__file__).with_name('nested_kappa_check.json').write_text(
        json.dumps(dict(reps=args.reps, meta=meta, rows=rows), indent=2))
    print('\nEscrito: audit/model_recovery/nested_kappa_check.json')


if __name__ == '__main__':
    main()
