#!/usr/bin/env python3
"""Potencia y recuperacion de MBIASES, WSLS y FRA, los modelos del estudio original.

Continua la [simulacion con tablas](README.md), que mostro que la version no
parametrica del contraste no tiene potencia. Aqui se repite todo con los modelos
parametricos del articulo, reimplementados en `parametric_models.py`, que tienen
4, 6 y 8 parametros libres en lugar de 1.080.

La pregunta es la del proyecto: la ventaja publicada de FRA sobre WSLS se midio
con AIC dentro de muestra sobre 1.244 transiciones agregadas. ¿Sobrevive cuando el
modelo tiene que generalizar a una diada que no vio? Y antes de eso: ¿puede este
diseno distinguir los tres modelos?

No se reporta ninguna comparacion de modelos sobre las personas mas alla de la
reconciliacion con las cifras ya publicadas por los autores. El contraste fuera de
diada sobre humanos sigue sin correrse.

Uso: python audit/model_recovery/parametric_power.py [--reps N] [--folds K]
"""
import os

# Cada replica se reparte entre procesos, asi que las bibliotecas de algebra no
# deben abrir hilos propios: sin esto los trabajadores se estorban y la corrida
# tarda un orden de magnitud mas. Tiene que ir antes de importar NumPy.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parametric_models as pm  # noqa: E402

multiprocessing_context = None

OUT = Path(__file__).resolve().parent
PUBLISHED_DEV = {'MBIASES': 1612.0, 'WSLS': 654.0, 'FRA': 557.0}
PUBLISHED_AIC = {'MBIASES': 1620.0, 'WSLS': 668.0, 'FRA': 577.0}


# ------------------------------------------------------------------ evaluacion

def grouped_folds(dyad, k, rng):
    nd = int(dyad.max()) + 1
    lab = np.empty(nd, dtype=int)
    lab[rng.permutation(nd)] = np.arange(nd) % k
    return lab[dyad]


def cv_losses(S, folds, rng, seed=0):
    """Perdida logaritmica por observacion, dejando fuera grupos de diadas.

    Los tres modelos se ajustan con `fit_ladder`, que impone el anidamiento. Sin
    eso, una ventaja aparente del modelo mayor puede ser solo un fallo del
    optimizador del menor, que es lo que ocurria al usar arranques sueltos.
    """
    g = grouped_folds(S['dyad'], folds, rng)
    out = np.zeros((len(pm.MODELS), len(S['y'])))
    for f in np.unique(g):
        tr, te = g != f, g == f
        if not te.any() or not tr.any():
            continue
        lad = pm.fit_ladder(S, sel=tr, seed=seed)
        sub = pm.subset(S, te)
        y = sub['y']
        for m, name in enumerate(pm.MODELS):
            P = pm.probabilities(lad[name][0], name, sub)
            out[m, te] = -np.log(np.maximum(P[np.arange(len(y)), y], 1e-300))
    return out


def insample(S, seed=0):
    lad = pm.fit_ladder(S, seed=seed)
    return {name: dict(theta=lad[name][0], nll=lad[name][1],
                       aic=2 * lad[name][1] + 2 * pm.N_PUBLISHED[name])
            for name in pm.MODELS}


def _one_rep(job):
    """Una replica completa: simular, validar por diada y ajustar dentro de muestra."""
    theta, model, n_dyads, folds, seed = job
    rng = np.random.default_rng(seed)
    S = pm.simulate(theta, model, n_dyads=n_dyads, rng=rng)
    L = cv_losses(S, folds, rng, seed=seed)
    ins = insample(S, seed=seed)
    diff = L[1] - L[2]
    lo, _ = clustered_ci(diff, S['dyad'], rng)
    return dict(cv_pick=pm.MODELS[int(np.argmin(L.mean(1)))],
                aic_pick=min(ins, key=lambda k: ins[k]['aic']),
                delta=float(diff.mean()), excludes_zero=bool(lo > 0),
                theta=np.asarray(ins[model]['theta']),
                n=len(S['y']))


def clustered_ci(diff, dyad, rng, B=400):
    nd = int(dyad.max()) + 1
    per = np.bincount(dyad, weights=diff, minlength=nd)
    cnt = np.bincount(dyad, minlength=nd)
    idx = rng.integers(0, nd, size=(B, nd))
    boot = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def run(theta, model, reps, rng, folds=5, n_dyads=45, workers=None):
    seeds = rng.integers(0, 2**31 - 1, size=reps)
    jobs = [(theta, model, n_dyads, folds, int(s)) for s in seeds]
    w = workers or max(1, min(os.cpu_count() - 1, 5))
    with ProcessPoolExecutor(max_workers=w) as ex:
        res = list(ex.map(_one_rep, jobs, chunksize=1))
    sel = {m: sum(r['cv_pick'] == m for r in res) for m in pm.MODELS}
    aic_sel = {m: sum(r['aic_pick'] == m for r in res) for m in pm.MODELS}
    deltas = np.array([r['delta'] for r in res])
    return dict(selected=sel, aic_selected=aic_sel,
                power=float(np.mean([r['excludes_zero'] for r in res])),
                delta_mean=float(deltas.mean()), delta_sd=float(deltas.std()),
                n_mean=float(np.mean([r['n'] for r in res])),
                thetas=np.array([r['theta'] for r in res]))


def oracle_delta(theta_fra, rng, n_dyads=600):
    """Ganancia esperada de log-loss de FRA sobre su mejor aproximacion WSLS.

    Se estima ajustando WSLS a una muestra grande generada por FRA, lo que da el
    limite al que tiende el contraste con datos infinitos. Es una propiedad de la
    verdad sintetica, no de los datos humanos.
    """
    big = pm.simulate(theta_fra, 'FRA', n_dyads=n_dyads, rng=rng)
    th_w = pm.fit_ladder(big, seed=0)['WSLS'][0]
    Pw = pm.probabilities(th_w, 'WSLS', big)
    Pf = pm.probabilities(theta_fra, 'FRA', big)
    y = big['y']
    a = np.arange(len(y))
    return float((-np.log(np.maximum(Pw[a, y], 1e-300))
                  + np.log(np.maximum(Pf[a, y], 1e-300))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=60)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=20260921)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    res = {'reps': args.reps, 'folds': args.folds, 'seed': args.seed}

    pre, meta = pm.load_transitions('original')
    print(json.dumps(meta, indent=2))

    print('\n=== 0. Reconciliacion con el articulo ===')
    print('   Conjunto: las 1.244 transiciones ausente-ausente de humans_only_absent.csv,')
    print('   que es el que ajusta fitModel.R. Su "Dev." es -logL, no -2logL, y va')
    print('   con el coeficiente multinomial de la agregacion por situaciones (119,6).')
    print(f"\n{'modelo':9s}{'-logL aqui':>12s}{'menos 119,6':>13s}{'Dev. publicada':>16s}"
          f"{'libres':>8s}{'contados':>10s}")
    fits, recon = {}, {}
    ladder = pm.fit_ladder(pre, seed=1, polish=True)
    for m in pm.MODELS:
        th, nll = ladder[m]
        fits[m] = th
        recon[m] = dict(nll=nll, adjusted=nll - 119.6, published=PUBLISHED_DEV[m],
                        theta=th.tolist())
        print(f'{m:9s}{nll:12.1f}{nll - 119.6:13.1f}{PUBLISHED_DEV[m]:16.1f}'
              f'{pm.N_FREE[m]:8d}{pm.N_PUBLISHED[m]:10d}')
    d_here = (recon['MBIASES']['nll'] - recon['WSLS']['nll'],
              recon['WSLS']['nll'] - recon['FRA']['nll'])
    print(f"\n   diferencias de log-verosimilitud aqui: {d_here[0]:.1f} y {d_here[1]:.1f}")
    print('   publicadas:                            958,0 y 97,0')
    print('   El orden y el orden de magnitud se reproducen; el resto se explica por')
    print('   los parametros redondeados de la Tabla 3 y por las cotas del optimizador.')
    res['reconciliation'] = recon

    print('\n=== 1. ¿Reproduce el modelo ajustado la conducta al simularlo? ===')
    obs = np.bincount(pre['y'], minlength=pm.NS) / len(pre['y'])
    print(f"   P(RS) observada en las transiciones: {obs[pm.RS]:.3f}")
    fwd = {}
    for m in pm.MODELS:
        P = pm.probabilities(fits[m], m, pre)
        sims = [pm.simulate(fits[m], m, rng=rng) for _ in range(20)]
        rs = float(np.mean([(s['y'] == pm.RS).mean() for s in sims]))
        fwd[m] = dict(conditional=float(P[:, pm.RS].mean()), forward=rs)
        print(f'   {m:9s} P(RS) predicha a un paso = {P[:, pm.RS].mean():.3f}   '
              f'al simular la cadena = {rs:.3f}')
    print('   Los tres estan calibrados a un paso y ninguno reproduce la frecuencia')
    print('   marginal al correr hacia adelante. Es una limitacion de la simulacion')
    print('   como retrato de las personas, y a la vez un desajuste de los modelos.')
    res['forward_check'] = fwd

    print('\n=== 2. Recuperacion de modelo ===')
    print(f"   {args.reps} replicas, 45 diadas, validacion cruzada por diada en "
          f"{args.folds} pliegues.")
    hdr = ''.join(f'{m:>10}' for m in pm.MODELS)
    print(f"\n{'generador':10s}{'fuera de diada':>30s}{'':4s}{'AIC dentro de muestra':>30s}")
    print(f"{'':10s}{hdr}{'':4s}{hdr}")
    rec = {}
    for g in pm.MODELS:
        t0 = time.time()
        r = run(fits[g], g, args.reps, rng, folds=args.folds)
        row = ''.join(f"{r['selected'][m] / args.reps:10.2f}" for m in pm.MODELS)
        arow = ''.join(f"{r['aic_selected'][m] / args.reps:10.2f}" for m in pm.MODELS)
        print(f'{g:10s}{row}{"":4s}{arow}   [{time.time() - t0:.0f}s]')
        rec[g] = {k: v for k, v in r.items() if k != 'thetas'}
        if g == 'FRA':
            res['fra_param_recovery'] = summarize_recovery(r['thetas'], fits['FRA'])
    res['recovery'] = rec

    print('\n=== 3. Potencia frente a la fuerza del mecanismo propio de FRA ===')
    print('   delta es el parametro que separa FRA de WSLS; delta = 0 es WSLS.')
    print(f"\n{'delta':>8}{'efecto oraculo':>16}{'P(FRA elegido)':>16}{'P(IC>0)':>10}"
          f"{'delta medio':>13}")
    base = fits['FRA'].copy()
    d_fit = float(np.exp(base[6]))
    curve = []
    for mult in [0.0, 0.5, 1.0, 1.5, 2.5, 4.0]:
        th = base.copy()
        if mult == 0.0:
            th[6] = np.log(1e-9)
        else:
            th[6] = np.log(d_fit * mult)
        od = oracle_delta(th, rng)
        r = run(th, 'FRA', args.reps, rng, folds=args.folds)
        p3 = r['selected']['FRA'] / args.reps
        print(f'{d_fit * mult:8.3f}{od:16.4f}{p3:16.2f}{r["power"]:10.2f}'
              f'{r["delta_mean"]:13.4f}')
        curve.append(dict(delta=d_fit * mult, oracle=od, p_select_FRA=p3,
                          power=r['power'], delta_mean=r['delta_mean']))
    res['power_curve'] = curve

    print('\n=== 4. Cuantas diadas harian falta ===')
    print(f"\n{'diadas':>8}{'transiciones':>14}{'P(FRA elegido)':>16}{'P(IC>0)':>10}{'delta medio':>13}")
    scale = []
    for nd in [45, 90, 180]:
        r = run(fits['FRA'], 'FRA', max(args.reps // 2, 20), rng, folds=args.folds, n_dyads=nd)
        n = max(args.reps // 2, 20)
        ntr = int(r['n_mean'])
        print(f"{nd:8d}{ntr:14d}{r['selected']['FRA'] / n:16.2f}{r['power']:10.2f}"
              f"{r['delta_mean']:13.4f}")
        scale.append(dict(n_dyads=nd, n_transitions=ntr,
                          p_select_FRA=r['selected']['FRA'] / n, power=r['power'],
                          delta_mean=r['delta_mean']))
    res['scale'] = scale

    OUT.joinpath('parametric_power.json').write_text(
        json.dumps(res, indent=2, default=float))
    print('\nEscrito: audit/model_recovery/parametric_power.json')


def summarize_recovery(thetas, truth):
    """Recuperacion de parametros, en el espiritu de la Fig 8 del articulo."""
    names = ['bias ALL', 'bias NOTHING', 'bias LR', 'bias IN/OUT', 'log alpha',
             'gamma', 'log delta', 'z']
    out = {}
    print('\n   Recuperacion de parametros de FRA (verdad frente a estimado):')
    for i, nm in enumerate(names):
        est = thetas[:, i]
        out[nm] = dict(truth=float(truth[i]), mean=float(est.mean()),
                       sd=float(est.std()),
                       bias=float(est.mean() - truth[i]))
        print(f'     {nm:14s} verdad {truth[i]:8.3f}   media {est.mean():8.3f}'
              f'   de {est.std():7.3f}')
    return out


if __name__ == '__main__':
    main()
