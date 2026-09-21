#!/usr/bin/env python3
"""MBIASES, WSLS y FRA como los define el estudio original, para poder simularlos.

Fuente de las ecuaciones: Andrade-Lotero y Goldstone (2021), *PLOS ONE* 16(7):
e0254532, ecuaciones 1 y 3 a 6, y las Tablas 2 y 3. Reimplementacion de Thomas
Chisica; el modelo y el experimento son de los autores originales.

Un modelo asigna una probabilidad a cada una de las nueve categorias de region
(las ocho focales mas RS) que el jugador explorara en la ronda siguiente, dado el
estado (i, s, j): la region que acaba de destapar, el score obtenido y las casillas
que solapo con el companero.

    attract(k, i, j, s) = bias_k
                        + alpha * thresh(s, beta, gamma) * I(k, i)
                        + delta * thresh(FRAsim(i, j, k), epsilon, z)

    P(k) = attract(k) / sum_r attract(r)

    FRAsim(i, j, k) = sim(i, k) * Focal(k) + sim(j, ~k) * Focal'(k)

con sim(a, b) = |a & b| / |a | b| (uno si ambas son vacias), Focal(k) = 0 solo para
RS y Focal'(k) = 0 para RS y para ALL. MBIASES fija alpha = delta = 0; WSLS fija
delta = 0. beta y epsilon valen 30 y no son parametros libres.

Sobre esa capa va la mano temblorosa: P_ns(k) = NON_SHAKY * P(k) para k focal y
P_ns(RS) = 1 - NON_SHAKY * (1 - P(RS)), con NON_SHAKY = 0,88.

Conteo de parametros. Los valores publicados 4, 7 y 10 se recuperan de la Tabla 3
con AIC = devianza + 2k sobre los pares 1612/1620, 654/668 y 557/577, e **incluyen**
beta y epsilon, que el propio texto declara fijos. Los parametros realmente libres
son 4, 6 y 8. Aqui se ajustan 4, 6 y 8 y se informan ambos conteos.
"""
import numpy as np

STATES = ['LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'IN', 'OUT', 'ALL', 'NOTHING', 'RS']
NS = len(STATES)
IDX = {s: i for i, s in enumerate(STATES)}
RS = IDX['RS']
BETA = 30.0                   # fijo en el articulo
EPSILON = 30.0                # fijo en el articulo
NON_SHAKY = 0.88
LOWER_EPS = 1e-4              # recorte del codigo original, sin renormalizar
HIGH_EPS = 0.9999
MODELS = ['MBIASES', 'WSLS', 'FRA']
N_FREE = {'MBIASES': 4, 'WSLS': 6, 'FRA': 8}
N_PUBLISHED = {'MBIASES': 4, 'WSLS': 7, 'FRA': 10}

# Parametros ajustados de la Tabla 3 del articulo.
PUBLISHED = {
    'MBIASES': dict(bias=dict(ALL=0.13, NOTHING=0.076, LR=0.058, IO=0.005),
                    alpha=0.0, gamma=0.0, delta=0.0, z=0.0),
    'WSLS': dict(bias=dict(ALL=0.1, NOTHING=0.05, LR=0.018, IO=0.002),
                 alpha=38.0, gamma=4.6, delta=0.0, z=0.0),
    'FRA': dict(bias=dict(ALL=0.06, NOTHING=0.05, LR=0.003, IO=0.0001),
                alpha=40.0, gamma=15.0, delta=0.5, z=0.954),
}


def focal_masks():
    """Las ocho regiones focales como vectores de 64 casillas, en el orden de STATES."""
    y, x = np.indices((8, 8))
    inside = (x > 0) & (x < 7) & (y > 0) & (y < 7)
    m = {'LEFT': x < 4, 'RIGHT': x >= 4, 'TOP': y < 4, 'BOTTOM': y >= 4,
         'IN': inside, 'OUT': ~inside, 'ALL': np.ones((8, 8), bool),
         'NOTHING': np.zeros((8, 8), bool)}
    F = np.zeros((NS, 64), dtype=bool)
    for k, v in m.items():
        F[IDX[k]] = v.ravel()
    return F


FOCAL = focal_masks()
NOT_FOCAL = ~FOCAL
FOCAL_SIZE = FOCAL.sum(1).astype(float)
NOT_FOCAL_SIZE = NOT_FOCAL.sum(1).astype(float)
IS_FOCAL = np.array([s != 'RS' for s in STATES], dtype=float)          # Focal(k)
IS_FOCAL_PRIME = np.array([s not in ('RS', 'ALL') for s in STATES], dtype=float)


def sim_to(masks, targets, target_sizes):
    """sim(a, b) entre cada fila de `masks` (n, 64) y cada fila de `targets` (m, 64)."""
    a = masks.astype(np.float64)
    inter = a @ targets.T.astype(np.float64)
    union = a.sum(1, keepdims=True) + target_sizes[None, :] - inter
    out = np.divide(inter, union, out=np.ones_like(inter), where=union > 0)
    return out


def frasim(i_tiles, j_tiles):
    """FRAsim(i, j, k) para cada observacion y cada una de las nueve categorias."""
    return (sim_to(i_tiles, FOCAL, FOCAL_SIZE) * IS_FOCAL
            + sim_to(j_tiles, NOT_FOCAL, NOT_FOCAL_SIZE) * IS_FOCAL_PRIME)


def thresh(v, steep, cut):
    """Sigmoide estable 1 / (1 + exp(-steep * (v - cut)))."""
    zz = steep * (v - cut)
    out = np.empty_like(zz, dtype=np.float64)
    pos = zz >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-zz[pos]))
    e = np.exp(zz[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# ------------------------------------------------------------ parametrizacion

def bias_vector(b_all, b_nothing, b_lr, b_io, b_rs=1.0):
    """Nueve sesgos con la simetria del articulo, normalizados a suma uno."""
    v = np.empty(NS)
    v[[IDX['LEFT'], IDX['RIGHT'], IDX['TOP'], IDX['BOTTOM']]] = b_lr
    v[[IDX['IN'], IDX['OUT']]] = b_io
    v[IDX['ALL']] = b_all
    v[IDX['NOTHING']] = b_nothing
    v[RS] = b_rs
    return v / v.sum()


def unpack(theta, model):
    """De los reales sin restringir a (bias, alpha, gamma, delta, z).

    Los sesgos van en escala logaritmica con el de RS fijo en uno, lo que deja
    exactamente cuatro grados de libertad; alpha y delta tambien son positivos.
    """
    b = np.exp(np.clip(theta[:4], -50.0, 20.0))
    bias = bias_vector(b[0], b[1], b[2], b[3])
    alpha = gamma = delta = z = 0.0
    if model in ('WSLS', 'FRA'):
        alpha, gamma = float(np.exp(np.clip(theta[4], -50.0, 20.0))), theta[5]
    if model == 'FRA':
        delta, z = float(np.exp(np.clip(theta[6], -50.0, 20.0))), theta[7]
    return bias, alpha, gamma, delta, z


def pack(p, model):
    """De un diccionario como los de PUBLISHED al vector sin restringir."""
    b = p['bias']
    rest = 1.0 - (b['ALL'] + b['NOTHING'] + 4 * b['LR'] + 2 * b['IO'])
    rest = max(rest, 1e-6)
    theta = [np.log(max(b[k], 1e-12) / rest) for k in ('ALL', 'NOTHING', 'LR', 'IO')]
    if model in ('WSLS', 'FRA'):
        theta += [np.log(max(p['alpha'], 1e-12)), p['gamma']]
    if model == 'FRA':
        theta += [np.log(max(p['delta'], 1e-12)), p['z']]
    return np.array(theta, dtype=float)


# ------------------------------------------------------- probabilidades y ajuste

def probabilities(theta, model, pre, shaky=True, clip=True):
    """Matriz (n, 9) con la probabilidad de cada categoria en la ronda siguiente.

    `clip` reproduce el recorte a [1e-4, 0.9999] que hace el codigo original
    (MODELpred.R, FRApred1) despues de la mano temblorosa y sin renormalizar.
    """
    bias, alpha, gamma, delta, z = unpack(theta, model)
    attract = np.broadcast_to(bias, (len(pre['score']), NS)).copy()
    if alpha:
        attract += alpha * thresh(pre['score'], BETA, gamma)[:, None] * pre['I']
    if delta:
        attract += delta * thresh(pre['frasim'], EPSILON, z)
    P = attract / attract.sum(1, keepdims=True)
    if not shaky:
        return P
    out = NON_SHAKY * P
    out[:, RS] = 1.0 - NON_SHAKY * (1.0 - P[:, RS])
    if clip:
        out = np.clip(out, LOWER_EPS, HIGH_EPS)
    return out


def neg_loglik(theta, model, pre, sel=None):
    sub = pre if sel is None else subset(pre, sel)
    P = probabilities(theta, model, sub)
    y = sub['y']
    return float(-np.log(np.maximum(P[np.arange(len(y)), y], 1e-300)).sum())


def subset(pre, sel):
    return {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(pre['y']) else v)
            for k, v in pre.items()}


def fit(model, pre, sel=None, starts=None, seed=0, n_starts=3, polish=False):
    """Ajusta por maxima verosimilitud.

    El articulo uso el metodo simplex (NMKB en R). Aqui se usa L-BFGS-B con
    varios arranques por coste, y `polish=True` anade una pasada de Nelder-Mead,
    que es lo que se hace en los ajustes sobre datos reales.
    """
    from scipy.optimize import minimize
    pre = pre if sel is None else subset(pre, sel)
    sel = None
    rng = np.random.default_rng(seed)
    if starts is None:
        base = pack(PUBLISHED[model], model)
        starts = [base] + [base + rng.normal(0, 0.5, size=base.shape)
                           for _ in range(max(n_starts - 1, 0))]
    best, best_v = None, np.inf
    for s0 in starts:
        r = minimize(neg_loglik, s0, args=(model, pre, sel), method='L-BFGS-B')
        if r.fun < best_v:
            best, best_v = r.x, float(r.fun)
    if polish:
        r = minimize(neg_loglik, best, args=(model, pre, sel), method='Nelder-Mead',
                     options=dict(maxiter=8000, maxfev=8000, xatol=1e-7, fatol=1e-7))
        if r.fun < best_v:
            best, best_v = r.x, float(r.fun)
    return best, best_v


def deviance_and_aic(model, pre, sel=None, theta=None, seed=0, published_k=True):
    if theta is None:
        theta, nll = fit(model, pre, sel, seed=seed)
    else:
        nll = neg_loglik(theta, model, pre, sel)
    k = N_PUBLISHED[model] if published_k else N_FREE[model]
    return dict(theta=theta, deviance=2 * nll, aic=2 * nll + 2 * k, k=k)


# ------------------------------------------------------------------- los datos

import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA256 = '9452e653a5ffd08f8148f15269d73b754f782ed7f48bc1c3292dc78b42105c54'
CELLS = [f'a{i}{j}' for i in range(1, 9) for j in range(1, 9)]


def load_transitions(mode='original'):
    """Estado (i, s, j) en la ronda n y categoria observada en la ronda n+1.

    mode='original' reproduce el conjunto del estudio: las rondas ausentes cuya
    ronda siguiente tambien es ausente, 1.244 transiciones, que es exactamente el
    contenido de humans_only_absent.csv. Ni el predictor ni el objetivo estan
    truncados por el hallazgo del objetivo.
    mode='absent' usa rondas ausentes consecutivas saltando las presentes.
    mode='all' usa rondas de calendario consecutivas.
    """
    path = ROOT / 'data/raw/humans_full.csv'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise AssertionError('humans_full.csv no coincide con el hash documentado')
    d = pd.read_csv(path)
    cat = d.Category.replace({'UP': 'TOP', 'DOWN': 'BOTTOM'}).to_numpy()
    tiles = d[CELLS].to_numpy().astype(bool)

    order = np.lexsort((d.Player.to_numpy(), d.Round.to_numpy(), d.Dyad.to_numpy()))
    joint = np.zeros_like(tiles)
    for a, b in zip(order[::2], order[1::2]):
        both = tiles[a] & tiles[b]
        joint[a] = both
        joint[b] = both
    if not bool((joint.sum(1) == d.Joint.to_numpy()).all()):
        raise AssertionError('Joint no coincide con el tamano de la interseccion')
    # Comprobacion ejecutable de la geometria: la categoria del conjunto tiene que
    # ser exactamente la coincidencia con una region focal calculada aqui. Si el
    # orden de las casillas o alguna mascara estuvieran mal, esto falla.
    if not bool((observed_category(tiles) == np.array([IDX[c] for c in cat])).all()):
        raise AssertionError('las mascaras focales no reproducen la columna Category')

    tab = pd.DataFrame(dict(dyad=d.Dyad.to_numpy(), player=d.Player.to_numpy(),
                            rnd=d.Round.to_numpy(), z=[IDX[c] for c in cat],
                            score=d.Score.to_numpy().astype(float),
                            absent=(d.Is_there == 'Unicorn_Absent').to_numpy(),
                            row=np.arange(len(d))))
    tab['lead_absent'] = (d.Is_there_LEAD == 'Unicorn_Absent').to_numpy()
    if mode == 'absent':
        tab = tab[tab.absent]
    tab = tab.sort_values(['dyad', 'player', 'rnd'])
    tab['y'] = tab.groupby('player', sort=False)['z'].shift(-1)
    tr = tab[tab.y.notna()]
    if mode == 'original':
        tr = tr[tr.absent & tr.lead_absent]
    rows = tr.row.to_numpy()
    dyads = sorted(tr.dyad.unique())
    didx = {v: i for i, v in enumerate(dyads)}

    I = np.zeros((len(tr), NS))
    zi = tr.z.to_numpy().astype(int)
    focal_prev = zi != RS
    I[np.arange(len(tr))[focal_prev], zi[focal_prev]] = 1.0
    pre = dict(dyad=tr.dyad.map(didx).to_numpy(), z=zi,
               score=tr.score.to_numpy(), y=tr.y.to_numpy().astype(int),
               I=I, frasim=frasim(tiles[rows], joint[rows]))
    meta = dict(mode=mode, sha256=digest, n=int(len(tr)), n_dyads=len(dyads))
    return pre, meta


# ------------------------------------------------------- simulacion generativa

def empirical_game_constants():
    """Constantes del juego que la simulacion necesita y que no son del modelo.

    Son tres: con que frecuencia el objetivo esta ausente, que forma tiene una
    region RS, y con que probabilidad el jugador acierta segun cuantas casillas
    destapo. Se estiman de los datos y son iguales para los tres modelos, asi que
    no son lo que los distingue.

    La forma de RS importa y no se puede inventar. El articulo describe RS como una
    region al azar con todas las casillas equiprobables, pero la categoria RS de los
    datos recoge todo lo que no es coincidencia exacta con una focal, y esas
    regiones se parecen mucho mas a las focales de lo que se pareceria una region
    aleatoria: la similitud maxima con alguna focal tiene percentil 90 de 0,94 en
    los datos frente a 0,54 si se sortean casillas independientes. Como FRA se
    activa por un umbral sobre esa similitud, sortear casillas independientes
    apagaria el mecanismo por construccion. Por eso las regiones RS se remuestrean
    de las observadas en rondas ausentes.
    """
    d = pd.read_csv(ROOT / 'data/raw/humans_full.csv')
    absent = (d.Is_there == 'Unicorn_Absent').to_numpy()
    correct = ((d.Is_there == 'Unicorn_Present').to_numpy()
               == (d.Answer == 'Present').to_numpy())
    cat = d.Category.replace({'UP': 'TOP', 'DOWN': 'BOTTOM'}).to_numpy()
    size = d.Size_visited.to_numpy()
    edges = np.array([0, 8, 16, 32, 48, 65])
    b = np.clip(np.searchsorted(edges, size, side='right') - 1, 0, len(edges) - 2)
    acc = np.array([correct[absent & (b == i)].mean() if (absent & (b == i)).any() else 0.9
                    for i in range(len(edges) - 1)])
    tiles = d[CELLS].to_numpy().astype(bool)
    return dict(p_absent=float(absent.mean()),
                p_rs_tile=float(size[absent & (cat == 'RS')].mean() / 64.0),
                rs_pool=tiles[absent & (cat == 'RS')],
                acc_edges=edges, acc=acc)


GAME = None


def game_constants():
    global GAME
    if GAME is None:
        GAME = empirical_game_constants()
    return GAME


def _draw(cum, u):
    return (u[:, None] > cum).sum(1)


def simulate(theta, model, n_dyads=45, n_rounds=60, rng=None, game=None,
             shaky=True, mode='original'):
    """Simula diadas jugando el juego bajo el modelo dado y devuelve transiciones.

    Cada ronda los dos jugadores eligen una region con P(k) del modelo; la region
    se realiza en casillas, con mano temblorosa; de ahi salen el solapamiento y el
    score; y el estado resultante alimenta la ronda siguiente. Despues se conserva
    el mismo subconjunto de transiciones que uso el estudio original.

    Limitacion declarada: en las rondas con objetivo presente la partida termina al
    encontrarlo y la region observada queda truncada. Aqui no se simula ese
    truncamiento, porque el conjunto del estudio solo usa pares ausente-ausente.
    """
    rng = rng or np.random.default_rng(0)
    g = game or game_constants()
    n = n_dyads * 2
    dyad_of = np.repeat(np.arange(n_dyads), 2)

    bias, alpha, gamma, delta, z = unpack(theta, model)
    bias_cum = np.cumsum(bias)[:-1]

    k = _draw(np.broadcast_to(bias_cum, (n, NS - 1)), rng.random(n))
    absent = rng.random((n_rounds, n_dyads)) < g['p_absent']

    recs = []
    for t in range(n_rounds):
        tiles = FOCAL[k].copy()
        is_rs = k == RS
        if is_rs.any():
            pool = g['rs_pool']
            tiles[is_rs] = pool[rng.integers(0, len(pool), int(is_rs.sum()))]
        if shaky:
            shake = (~is_rs) & (rng.random(n) >= NON_SHAKY)
            for r in np.flatnonzero(shake):
                flip = rng.choice(64, size=int(rng.integers(1, 4)), replace=False)
                tiles[r, flip] = ~tiles[r, flip]
        obs = observed_category(tiles)

        a, b = tiles[0::2], tiles[1::2]
        joint_pair = a & b
        joint = np.repeat(joint_pair, 2, axis=0)
        jsize = joint.sum(1)
        size = tiles.sum(1)
        bi = np.clip(np.searchsorted(g['acc_edges'], size, side='right') - 1,
                     0, len(g['acc_edges']) - 2)
        correct = rng.random(n) < g['acc'][bi]
        score = np.where(correct, 32.0, -64.0) - jsize

        I = np.zeros((n, NS))
        foc = obs != RS
        I[np.flatnonzero(foc), obs[foc]] = 1.0
        fs = frasim(tiles, joint)

        attract = np.broadcast_to(bias, (n, NS)).copy()
        if alpha:
            attract += alpha * thresh(score, BETA, gamma)[:, None] * I
        if delta:
            attract += delta * thresh(fs, EPSILON, z)
        P = attract / attract.sum(1, keepdims=True)
        if shaky:
            Q = NON_SHAKY * P
            Q[:, RS] = 1.0 - NON_SHAKY * (1.0 - P[:, RS])
            P = Q
        k_next = _draw(np.cumsum(P, 1)[:, :-1], rng.random(n))

        recs.append(dict(t=t, z=obs, score=score, I=I, frasim=fs,
                         absent=np.repeat(absent[t], 2), k_next=k_next))
        k = k_next

    return _assemble(recs, dyad_of, n_rounds, mode)


def observed_category(tiles):
    """Categoria observada: coincidencia exacta con una region focal, si no RS."""
    out = np.full(len(tiles), RS)
    for idx in range(NS):
        if STATES[idx] == 'RS':
            continue
        hit = (tiles == FOCAL[idx]).all(1)
        out[hit] = idx
    return out


def _assemble(recs, dyad_of, n_rounds, mode):
    keep_t, keep_r = [], []
    for t in range(n_rounds - 1):
        if mode == 'original':
            ok = recs[t]['absent'] & recs[t + 1]['absent']
        elif mode == 'absent':
            ok = recs[t]['absent'] & recs[t + 1]['absent']
        else:
            ok = np.ones(len(dyad_of), dtype=bool)
        keep_t.append(np.full(int(ok.sum()), t))
        keep_r.append(np.flatnonzero(ok))
    z, score, I, fs, y, dy = [], [], [], [], [], []
    for t, rows in zip(keep_t, keep_r):
        if len(rows) == 0:
            continue
        tt = t[0]
        z.append(recs[tt]['z'][rows])
        score.append(recs[tt]['score'][rows])
        I.append(recs[tt]['I'][rows])
        fs.append(recs[tt]['frasim'][rows])
        y.append(recs[tt + 1]['z'][rows])
        dy.append(dyad_of[rows])
    return dict(dyad=np.concatenate(dy), z=np.concatenate(z),
                score=np.concatenate(score), I=np.concatenate(I),
                frasim=np.concatenate(fs), y=np.concatenate(y))


# --------------------------------------------- ajuste respetando el anidamiento

def embed(theta, src, dst):
    """Mete la solucion de un modelo en el espacio del siguiente, desactivando
    el mecanismo nuevo. Sirve como punto de arranque garantizado."""
    if src == 'MBIASES' and dst == 'WSLS':
        return np.concatenate([theta, [np.log(1e-8), 0.0]])
    if src == 'WSLS' and dst == 'FRA':
        return np.concatenate([theta, [np.log(1e-8), 0.5]])
    raise ValueError(f'{src} no esta anidado directamente en {dst}')


def fit_ladder(pre, sel=None, seed=0, n_grid=9, polish=False):
    """Ajusta MBIASES, WSLS y FRA imponiendo que la escalera sea monotona.

    Con beta = epsilon = 30 las dos funciones umbral son practicamente escalones,
    asi que la verosimilitud es casi plana en gamma y en z y un optimizador local
    depende mucho del arranque: sin este cuidado WSLS puede quedar decenas de nats
    por debajo de FRA, lo que es imposible porque FRA lo contiene. Aqui cada modelo
    arranca tambien desde la solucion del modelo anterior con el mecanismo nuevo
    apagado, y desde una rejilla sobre el umbral correspondiente. Si aun asi el
    modelo mayor ajustara peor, se le impone la solucion del menor.
    """
    sub = pre if sel is None else subset(pre, sel)
    score, fs = sub['score'], sub['frasim']
    qs = np.linspace(0.05, 0.95, n_grid)
    gamma_grid = np.quantile(score, qs)
    z_grid = np.quantile(fs[fs > 0], qs) if (fs > 0).any() else np.linspace(0, 2, n_grid)

    out = {}
    th, v = fit('MBIASES', pre, sel, seed=seed, n_starts=3, polish=polish)
    out['MBIASES'] = (th, v)

    starts = [pack(PUBLISHED['WSLS'], 'WSLS')]
    base = embed(out['MBIASES'][0], 'MBIASES', 'WSLS')
    for g in gamma_grid:
        s = base.copy()
        s[4], s[5] = np.log(10.0), g
        starts.append(s)
    starts.append(base)
    th, v = fit('WSLS', pre, sel, starts=starts, polish=polish)
    if v > out['MBIASES'][1]:
        th, v = base, out['MBIASES'][1]
    out['WSLS'] = (th, v)

    starts = [pack(PUBLISHED['FRA'], 'FRA')]
    base = embed(out['WSLS'][0], 'WSLS', 'FRA')
    for zz in z_grid:
        s = base.copy()
        s[6], s[7] = np.log(0.5), zz
        starts.append(s)
    starts.append(base)
    th, v = fit('FRA', pre, sel, starts=starts, polish=polish)
    if v > out['WSLS'][1]:
        th, v = base, out['WSLS'][1]
    out['FRA'] = (th, v)

    # Pasada descendente: si el modelo mayor encontro una region mejor del espacio
    # comun, el menor tiene que poder alcanzarla. Sin esto, una ventaja aparente
    # del modelo mayor puede ser solo un fallo del optimizador del menor.
    th_w, v_w = fit('WSLS', pre, sel, starts=[out['FRA'][0][:6]], polish=polish)
    if v_w < out['WSLS'][1]:
        out['WSLS'] = (th_w, v_w)
        if out['FRA'][1] > v_w:
            out['FRA'] = (embed(th_w, 'WSLS', 'FRA'), v_w)
    th_m, v_m = fit('MBIASES', pre, sel, starts=[out['WSLS'][0][:4]], polish=polish)
    if v_m < out['MBIASES'][1]:
        out['MBIASES'] = (th_m, v_m)
    return out
