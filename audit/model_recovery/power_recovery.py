#!/usr/bin/env python3
"""Potencia y recuperacion del contraste M0/M1/M2/M3 antes de correrlo sobre humanos.

Pregunta: con 45 diadas y el diseno del experimento original, un contraste de
log-loss dejando una diada fuera, ¿puede distinguir los modelos rivales de la
propuesta mecanistica? Y si los distingue, ¿a partir de que tamano de efecto?

Lo que este script SI hace:
  - estima parametros generadores plausibles a partir de humans_full.csv;
  - simula datos sinteticos de esos generadores, con y sin heterogeneidad entre diadas;
  - reajusta los cuatro modelos y mide con que frecuencia se recupera el generador;
  - traza la potencia frente al tamano de efecto y frente al numero de diadas.

Lo que este script NO hace, a proposito: no reporta la comparacion entre modelos
sobre los datos humanos. Ese es el contraste que el equipo todavia no ha acordado
y no debe conocerse antes de fijar el protocolo. Los parametros humanos se usan
solo como anclaje del generador sintetico, y por eso el extremo theta=1 de la
curva es una cota optimista: hereda el sobreajuste del ajuste completo.

Uso: python audit/model_recovery/power_recovery.py [--reps N] [--mode absent|all]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STATES = ['LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'IN', 'OUT', 'ALL', 'NOTHING', 'RS']
NS = len(STATES)
NSC = 3                       # bins de score
NOV = 5                       # direccion del solapamiento: vacio y cuatro cuadrantes
SOURCE_SHA256 = '9452e653a5ffd08f8148f15269d73b754f782ed7f48bc1c3292dc78b42105c54'
CELLS = [f'a{i}{j}' for i in range(1, 9) for j in range(1, 9)]
MODELS = ['M0', 'M1', 'M2', 'M3']
KAPPA = 5.0                   # regla de suavizado jerarquico fijada de antemano
DEFAULT_SEED = 20260921


# --------------------------------------------------------------- 1. los datos

def load(mode):
    """Transiciones observadas y covariables discretizadas.

    mode='absent': transiciones entre rondas ausentes consecutivas del mismo
    jugador, donde la categoria no esta truncada por el hallazgo del objetivo.
    mode='all': transiciones entre rondas de calendario consecutivas.
    """
    path = ROOT / 'data/raw/humans_full.csv'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise AssertionError('humans_full.csv no coincide con el hash documentado')
    d = pd.read_csv(path)
    cat = d.Category.replace({'UP': 'TOP', 'DOWN': 'BOTTOM'}).to_numpy()

    # Solapamiento espacial: interseccion de las casillas destapadas por los dos
    # jugadores de la diada en la misma ronda. Su tamano es la columna Joint, y
    # como Score = (32 o -64) - Joint, condicionar en el score ya fija el tamano.
    # Por eso el predictor de M3 es la direccion, no el tamano.
    vis = d[CELLS].to_numpy().astype(bool).reshape(len(d), 8, 8)
    direction = np.zeros(len(d), dtype=int)
    jsize = np.zeros(len(d), dtype=int)
    order = np.lexsort((d.Player.to_numpy(), d.Round.to_numpy(), d.Dyad.to_numpy()))
    for a, b in zip(order[::2], order[1::2]):
        joint = vis[a] & vis[b]
        tot = int(joint.sum())
        jsize[[a, b]] = tot
        if tot:
            left = joint[:, :4].sum() >= joint[:, 4:].sum()
            top = joint[:4, :].sum() >= joint[4:, :].sum()
            direction[[a, b]] = 1 + 2 * int(left) + int(top)
    if not bool((jsize == d.Joint.to_numpy()).all()):
        raise AssertionError('Joint no coincide con el tamano de la interseccion')
    correct = (d.Is_there == 'Unicorn_Present').to_numpy() == (d.Answer == 'Present').to_numpy()
    if not bool((d.Score.to_numpy() == np.where(correct, 32, -64) - jsize).all()):
        raise AssertionError('la identidad del score no se cumple')

    code = {s: i for i, s in enumerate(STATES)}
    tab = pd.DataFrame(dict(dyad=d.Dyad.to_numpy(), player=d.Player.to_numpy(),
                            rnd=d.Round.to_numpy(), z=[code[c] for c in cat],
                            score=d.Score.to_numpy(), ov=direction,
                            absent=(d.Is_there == 'Unicorn_Absent').to_numpy()))
    if mode == 'absent':
        tab = tab[tab.absent]
    tab = tab.sort_values(['dyad', 'player', 'rnd'])
    cuts = np.quantile(tab.score, [1 / 3, 2 / 3])
    tab['s'] = np.searchsorted(cuts, tab.score.to_numpy(), side='right')
    tab['y'] = tab.groupby('player', sort=False)['z'].shift(-1)
    tr = tab[tab.y.notna()]
    dyads = sorted(tr.dyad.unique())
    didx = {v: i for i, v in enumerate(dyads)}
    T = dict(dyad=tr.dyad.map(didx).to_numpy(), z=tr.z.to_numpy().astype(int),
             s=tr.s.to_numpy().astype(int), o=tr.ov.to_numpy().astype(int),
             y=tr.y.to_numpy().astype(int))
    meta = dict(mode=mode, sha256=digest, n_transitions=int(len(tr)), n_dyads=len(dyads),
                score_cuts=[float(c) for c in cuts], joint_equals_intersection=True,
                score_identity_holds=True)
    return T, meta


# ------------------------------------------------- 2. conteos, ajuste, perdida

def count(z, s, o, y):
    C3 = np.zeros((NS, NSC, NOV, NS))
    np.add.at(C3, (z, s, o, y), 1.0)
    return C3


def fit(C3, kappa=KAPPA):
    """Suavizado jerarquico: cada modelo retrocede a su padre donde falta soporte."""
    C2 = C3.sum(2)
    C1 = C2.sum(1)
    C0 = C1.sum(0)
    p0 = (C0 + 0.5) / (C0.sum() + 0.5 * NS)
    p1 = (C1 + kappa * p0) / (C1.sum(-1, keepdims=True) + kappa)
    p2 = (C2 + kappa * p1[:, None, :]) / (C2.sum(-1, keepdims=True) + kappa)
    p3 = (C3 + kappa * p2[:, :, None, :]) / (C3.sum(-1, keepdims=True) + kappa)
    return {'M0': p0, 'M1': p1, 'M2': p2, 'M3': p3}


def predict(P, name, z, s, o):
    if name == 'M0':
        return np.broadcast_to(P['M0'], (len(z), NS))
    if name == 'M1':
        return P['M1'][z]
    if name == 'M2':
        return P['M2'][z, s]
    return P['M3'][z, s, o]


KAPPA_GRID = (0.5, 2.0, 5.0, 20.0, 80.0)


def _dyad_counts(S):
    z, s, o, y, dy = S['z'], S['s'], S['o'], S['y'], S['dyad']
    nd = int(dy.max()) + 1
    return [count(z[dy == d], s[dy == d], o[dy == d], y[dy == d]) for d in range(nd)]


def _eval_folds(S, per, groups, kappas):
    """Perdida por transicion al dejar fuera cada grupo de diadas. (4, n)."""
    z, s, o, y, dy = S['z'], S['s'], S['o'], S['y'], S['dyad']
    full = sum(per)
    out = np.zeros((len(MODELS), len(z)))
    for g in np.unique(groups):
        heldout = np.flatnonzero(groups == g)
        sel = np.isin(dy, heldout)
        if not sel.any():
            continue
        C = full - sum(per[d] for d in heldout)
        fits = {k: fit(C, k) for k in set(kappas.values())}
        for m, name in enumerate(MODELS):
            p = predict(fits[kappas[name]], name, z[sel], s[sel], o[sel])
            out[m, sel] = -np.log(p[np.arange(int(sel.sum())), y[sel]])
    return out


def choose_kappa(S, per, rng, folds=5, grid=KAPPA_GRID):
    """Elige kappa por modelo con validacion cruzada interna agrupada por diada.

    Sin esto, la eleccion entre modelos depende de una constante fijada a mano:
    con kappa alto M3 se confunde con M2 y con kappa bajo nunca gana.
    """
    nd = len(per)
    groups = np.empty(nd, dtype=int)
    groups[rng.permutation(nd)] = np.arange(nd) % folds
    scores = {}
    for k in grid:
        L = _eval_folds(S, per, groups, {m: k for m in MODELS})
        for m, name in enumerate(MODELS):
            scores.setdefault(name, {})[k] = float(L[m].mean())
    return {name: min(scores[name], key=scores[name].get) for name in MODELS}


def loo_losses(S, kappa=KAPPA, rng=None):
    """Perdida logaritmica por transicion dejando una diada fuera. (4, n).

    kappa='cv' elige la constante de suavizado por modelo dentro del entrenamiento.
    """
    per = _dyad_counts(S)
    nd = len(per)
    if kappa == 'cv':
        kappas = choose_kappa(S, per, rng)
    else:
        kappas = {m: kappa for m in MODELS}
    L = _eval_folds(S, per, np.arange(nd), kappas)
    return L, kappas


def insample(S, kappa=KAPPA):
    """Log-loss dentro de muestra y AIC con parametros efectivos (celdas con soporte)."""
    z, s, o, y = S['z'], S['s'], S['o'], S['y']
    C3 = count(z, s, o, y)
    P = fit(C3, kappa)
    C2, C1 = C3.sum(2), C3.sum(2).sum(1)
    k = {'M0': NS - 1,
         'M1': int((C1.sum(-1) > 0).sum()) * (NS - 1),
         'M2': int((C2.sum(-1) > 0).sum()) * (NS - 1),
         'M3': int((C3.sum(-1) > 0).sum()) * (NS - 1)}
    ll, aic = {}, {}
    for name in MODELS:
        p = predict(P, name, z, s, o)[np.arange(len(y)), y]
        ll[name] = float(-np.log(p).mean())
        aic[name] = float(2 * (-np.log(p)).sum() + 2 * k[name])
    return ll, aic, k


# ---------------------------------------------------- 3. el generador sintetico

def _sample(cum, u):
    """Muestreo categorico vectorizado a partir de acumuladas por fila."""
    return (u[:, None] > cum).sum(1)


class Generator:
    """Covariables desde p(s,o | z), estado desde P(y | z,s,o). Chain sequencial."""

    def __init__(self, T, rng):
        self.rng = rng
        C = np.zeros((NS, NSC * NOV))
        np.add.at(C, (T['z'], T['s'] * NOV + T['o']), 1.0)
        self.cov = (C + 0.5) / (C.sum(-1, keepdims=True) + 0.5 * NSC * NOV)
        self.cov_cum = np.cumsum(self.cov, 1)[:, :-1]
        init = np.zeros(NS)
        np.add.at(init, T['z'], 1.0)
        self.init = (init + 0.5) / (init.sum() + 0.5 * NS)
        self.lengths = np.bincount(T['dyad'])
        self.n_dyads = len(self.lengths)

    def marginal_chain(self, P3):
        """Q[z,y] = sum_{s,o} p(s,o|z) P3[z,s,o,y], y su distribucion estacionaria."""
        Q = np.einsum('zc,zcy->zy', self.cov, P3.reshape(NS, NSC * NOV, NS))
        w, v = np.linalg.eig(Q.T)
        p = np.abs(np.real(v[:, int(np.argmin(np.abs(w - 1)))]))
        return Q, p / p.sum()

    def best_m2(self, P3):
        """Mejor aproximacion M2 del generador: marginaliza la direccion del solape."""
        pc = self.cov.reshape(NS, NSC, NOV)
        w = pc / np.maximum(pc.sum(-1, keepdims=True), 1e-300)
        ref2 = np.einsum('zso,zsoy->zsy', w, P3)
        return np.broadcast_to(ref2[:, :, None, :], P3.shape).copy()

    def oracle_delta(self, P3):
        """Ganancia esperada de log-loss del generador sobre su mejor M2, en nats.

        Analitico: esperanza de la divergencia KL(P3 || refM2) bajo la
        distribucion estacionaria del estado y el modelo de covariables.
        """
        _, pi = self.marginal_chain(P3)
        ref = self.best_m2(P3)
        pc = self.cov.reshape(NS, NSC, NOV)
        kl = (P3 * (np.log(np.maximum(P3, 1e-300)) - np.log(np.maximum(ref, 1e-300)))).sum(-1)
        return float((pi[:, None, None] * pc * kl).sum())

    def draw(self, P3, tau=None, n_dyads=None, rng=None):
        rng = rng or self.rng
        nd0 = self.n_dyads
        nd = nd0 if n_dyads is None else n_dyads
        lengths = self.lengths[np.arange(nd) % nd0]
        if tau is None:
            Pd = np.broadcast_to(P3, (nd,) + P3.shape)
        else:
            g = rng.gamma(np.maximum(tau * P3.reshape(-1, NS), 1e-6), size=(nd, NS * NSC * NOV, NS))
            Pd = (g / g.sum(-1, keepdims=True)).reshape((nd,) + P3.shape)
        Pd_cum = np.cumsum(Pd.reshape(nd, -1, NS), -1)[..., :-1]
        z = _sample(np.cumsum(self.init)[None, :-1].repeat(nd, 0), rng.random(nd))
        L = int(lengths.max())
        rows = []
        for t in range(L):
            act = np.flatnonzero(lengths > t)
            if act.size == 0:
                break
            za = z[act]
            c = _sample(self.cov_cum[za], rng.random(act.size))
            sa, oa = c // NOV, c % NOV
            flat = (za * NSC + sa) * NOV + oa
            ya = _sample(Pd_cum[act, flat], rng.random(act.size))
            rows.append(np.stack([act, za, sa, oa, ya], 1))
            z = z.copy()
            z[act] = ya
        R = np.concatenate(rows, 0)
        return dict(dyad=R[:, 0], z=R[:, 1], s=R[:, 2], o=R[:, 3], y=R[:, 4])


def as_table(P, name):
    if name == 'M0':
        return np.broadcast_to(P['M0'], (NS, NSC, NOV, NS)).copy()
    if name == 'M1':
        return np.broadcast_to(P['M1'][:, None, None, :], (NS, NSC, NOV, NS)).copy()
    if name == 'M2':
        return np.broadcast_to(P['M2'][:, :, None, :], (NS, NSC, NOV, NS)).copy()
    return P['M3'].copy()


# ---------------------------------------------------------------- 4. analisis

def clustered_ci(diff, dyad, rng, B=400):
    nd = int(dyad.max()) + 1
    per = np.bincount(dyad, weights=diff, minlength=nd)
    cnt = np.bincount(dyad, minlength=nd)
    idx = rng.integers(0, nd, size=(B, nd))
    boot = per[idx].sum(1) / np.maximum(cnt[idx].sum(1), 1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def run(gen, P3, reps, rng, tau=None, n_dyads=None, kappa='cv'):
    sel = {m: 0 for m in MODELS}
    aic_sel = {m: 0 for m in MODELS}
    picked = {m: [] for m in MODELS}
    excl, deltas, ins = 0, [], []
    for _ in range(reps):
        S = gen.draw(P3, tau, n_dyads, rng)
        L, chosen = loo_losses(S, kappa, rng)
        for m in MODELS:
            picked[m].append(chosen[m])
        sel[MODELS[int(np.argmin(L.mean(1)))]] += 1
        ll, aic, _ = insample(S, KAPPA if kappa == 'cv' else kappa)
        aic_sel[min(aic, key=aic.get)] += 1
        ins.append(ll['M2'] - ll['M3'])
        diff = L[2] - L[3]                       # M2 menos M3: positivo favorece a M3
        deltas.append(float(diff.mean()))
        lo, _hi = clustered_ci(diff, S['dyad'], rng)
        excl += int(lo > 0)
    return dict(selected=sel, aic_selected=aic_sel, power=excl / reps,
                delta_mean=float(np.mean(deltas)), delta_sd=float(np.std(deltas)),
                insample_delta_mean=float(np.mean(ins)),
                kappa_median={m: float(np.median(picked[m])) for m in MODELS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--mode', default='absent', choices=['absent', 'all'])
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    T, meta = load(args.mode)
    P = fit(count(T['z'], T['s'], T['o'], T['y']))
    tables = {m: as_table(P, m) for m in MODELS}
    gen = Generator(T, rng)
    res = {'meta': meta, 'kappa': KAPPA, 'reps': args.reps, 'seed': args.seed}

    print(json.dumps(meta, indent=2))
    print(f"\nTransiciones: {meta['n_transitions']} en {meta['n_dyads']} diadas "
          f"({meta['n_transitions'] / meta['n_dyads']:.1f} por diada)")
    _, _, k = insample(T)
    print(f"Celdas de condicionamiento con soporte: M1 {k['M1'] // (NS - 1)}/{NS}, "
          f"M2 {k['M2'] // (NS - 1)}/{NS * NSC}, M3 {k['M3'] // (NS - 1)}/{NS * NSC * NOV}")
    print(f"Efecto oraculo de M3 sobre su mejor M2, anclado en humanos: "
          f"{gen.oracle_delta(tables['M3']):.4f} nats")
    res['support_cells'] = {m: k[m] // (NS - 1) for m in MODELS}
    res['anchor_oracle_delta'] = gen.oracle_delta(tables['M3'])

    print('\n=== 1. Recuperacion: quien gana cuando sabemos quien genero ===')
    hdr = ''.join(f'{m:>8}' for m in MODELS)
    print(f"\n{'generador':12s}{'fuera de diada':>32s}{'':6s}{'AIC dentro de muestra':>32s}"
          f"{'':6s}{'kappa mediano elegido':>32s}")
    print(f"{'':12s}{hdr}{'':6s}{hdr}{'':6s}{hdr}")
    rec = {}
    for g in MODELS:
        r = run(gen, tables[g], args.reps, rng)
        rec[g] = r
        row = ''.join(f"{r['selected'][m] / args.reps:8.2f}" for m in MODELS)
        arow = ''.join(f"{r['aic_selected'][m] / args.reps:8.2f}" for m in MODELS)
        krow = ''.join(f"{r['kappa_median'][m]:8.1f}" for m in MODELS)
        print(f'{g:12s}{row}{"":6s}{arow}{"":6s}{krow}')
    res['recovery'] = rec

    print('\n=== 2. Potencia frente al tamano de efecto ===')
    print('   P_theta proporcional a refM2 * (M3/refM2)^theta: theta=1 es el anclaje\n   humano y theta>1 amplifica la dependencia del patron de solapamiento.')
    print(f"\n{'theta':>7}{'efecto oraculo':>16}{'P(M3 elegido)':>15}{'P(IC>0)':>10}"
          f"{'delta medio':>13}{'de delta':>10}")
    curve = []
    ref3 = gen.best_m2(tables['M3'])
    for theta in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        Pt = ref3 * np.power(np.maximum(tables['M3'], 1e-12) / np.maximum(ref3, 1e-12), theta)
        Pt /= Pt.sum(-1, keepdims=True)
        od = gen.oracle_delta(Pt)
        r = run(gen, Pt, args.reps, rng)
        s3 = r['selected']['M3'] / args.reps
        print(f'{theta:7.2f}{od:16.4f}{s3:15.2f}{r["power"]:10.2f}'
              f'{r["delta_mean"]:13.4f}{r["delta_sd"]:10.4f}')
        curve.append(dict(theta=theta, oracle_delta=od, p_select_M3=s3, power=r['power'],
                          delta_mean=r['delta_mean'], delta_sd=r['delta_sd']))
    res['power_curve'] = curve

    print('\n=== 3. ¿Cuantas diadas harian falta? (generador M3) ===')
    print(f"\n{'diadas':>8}{'transiciones':>14}{'P(M3 elegido)':>15}{'P(IC>0)':>10}{'delta medio':>13}")
    scale = []
    for nd in [45, 90, 180, 360]:
        r = run(gen, tables['M3'], max(args.reps // 2, 25), rng, n_dyads=nd)
        n_tr = int(gen.lengths[np.arange(nd) % gen.n_dyads].sum())
        print(f"{nd:8d}{n_tr:14d}{r['selected']['M3'] / max(args.reps // 2, 25):15.2f}"
              f"{r['power']:10.2f}{r['delta_mean']:13.4f}")
        scale.append(dict(n_dyads=nd, n_transitions=n_tr,
                          p_select_M3=r['selected']['M3'] / max(args.reps // 2, 25),
                          power=r['power'], delta_mean=r['delta_mean']))
    res['scale'] = scale

    print('\n=== 4. Heterogeneidad entre diadas, generador M2 ===')
    print('   Cada diada recibe su propia tabla Dirichlet(tau * M2), asi que la')
    print('   dependencia aparente del solapamiento es idiosincratica y no transfiere.')
    print('   Es la confusion que un revisor va a plantear: dentro de muestra el')
    print('   modelo rico gana, fuera de diada pierde, y eso no refuta el mecanismo.')
    print(f"\n{'tau':>10}{'delta dentro':>14}{'delta fuera':>13}{'M3 fuera':>10}"
          f"{'M2 fuera':>10}{'M1 fuera':>10}")
    het = []
    n = max(args.reps // 2, 25)
    for tau in [None, 200.0, 50.0, 10.0]:
        r = run(gen, tables['M2'], n, rng, tau=tau)
        lab = 'homogeneo' if tau is None else f'{tau:g}'
        print(f"{lab:>10}{r['insample_delta_mean']:14.4f}{r['delta_mean']:13.4f}"
              f"{r['selected']['M3'] / n:10.2f}{r['selected']['M2'] / n:10.2f}"
              f"{r['selected']['M1'] / n:10.2f}")
        het.append(dict(tau=tau, selected={k2: v / n for k2, v in r['selected'].items()},
                        insample_delta_mean=r['insample_delta_mean'],
                        delta_mean=r['delta_mean']))
    res['heterogeneity'] = het

    print('\n=== 5. El suavizado no es un detalle ===')
    print('   Con kappa fijo a mano la recuperacion cambia de signo; por eso el')
    print('   protocolo debe elegirlo dentro del entrenamiento.')
    print(f"\n{'kappa':>10}{'recupera M1':>14}{'recupera M2':>14}{'recupera M3':>14}")
    sens = []
    n = max(args.reps // 4, 25)
    for kappa in [0.5, 2.0, 5.0, 20.0, 80.0, 'cv']:
        row = {g: run(gen, tables[g], n, rng, kappa=kappa)['selected'][g] / n
               for g in ['M1', 'M2', 'M3']}
        lab = kappa if isinstance(kappa, str) else f'{kappa:g}'
        print(f"{lab:>10}{row['M1']:14.2f}{row['M2']:14.2f}{row['M3']:14.2f}")
        sens.append(dict(kappa=kappa, **row))
    res['smoothing_sensitivity'] = sens

    # Una corrida con otra semilla no debe pisar el resultado registrado.
    tag = '' if args.seed == DEFAULT_SEED else f'_seed{args.seed}'
    name = f'power_recovery_{args.mode}{tag}.json'
    (OUT / name).write_text(json.dumps(res, indent=2))
    print(f'\nEscrito: audit/model_recovery/{name}')


if __name__ == '__main__':
    main()
