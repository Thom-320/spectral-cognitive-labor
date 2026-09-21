#!/usr/bin/env python3
"""Cadena de Markov empirica sobre las estrategias del estudio original, con las 60 rondas.

Fuente: data/raw/humans_full.csv del repositorio SODCL
(https://raw.githubusercontent.com/EAndrade-Lotero/SODCL/master/Data/humans_full.csv,
sha256 9452e653a5ffd08f8148f15269d73b754f782ed7f48bc1c3292dc78b42105c54).
Las categorias verticales se llaman UP/DOWN alli; se renombran a TOP/BOTTOM.

Descriptivo. No construye ningun outcome ni ajusta modelos predictivos sobre
especializacion. La comparacion de ordenes de la seccion 5 ajusta tablas de
frecuencias por conteo y se evalua dejando una diada fuera; es un diagnostico
del proceso, no una evaluacion predictiva del proyecto.

Uso: python audit/strategy_chain/build_strategy_chain.py
"""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
S = ['LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'IN', 'OUT', 'ALL', 'NOTHING', 'RS']
COMP_UNORDERED = {('LEFT', 'RIGHT'), ('BOTTOM', 'TOP'), ('IN', 'OUT'), ('ALL', 'NOTHING')}
SOURCE_SHA256 = '9452e653a5ffd08f8148f15269d73b754f782ed7f48bc1c3292dc78b42105c54'
CELLS = [f'a{i}{j}' for i in range(1, 9) for j in range(1, 9)]


# --- 1. carga y procedencia ejecutable ---------------------------------------

def focal_regions():
    """Las ocho regiones focales del estudio original, en el orden de S sin RS."""
    y, x = np.indices((8, 8))
    inside = (x > 0) & (x < 7) & (y > 0) & (y < 7)
    return {'ALL': np.ones((8, 8), bool), 'NOTHING': np.zeros((8, 8), bool),
            'BOTTOM': y >= 4, 'TOP': y < 4, 'LEFT': x < 4, 'RIGHT': x >= 4,
            'IN': inside, 'OUT': ~inside}


def load_and_check():
    """Carga humans_full.csv y comprueba hash, regla de Category y coherencia."""
    path = ROOT / 'data/raw/humans_full.csv'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    d = pd.read_csv(path)
    d['Category'] = d.Category.replace({'UP': 'TOP', 'DOWN': 'BOTTOM'})

    regions = focal_regions()
    visits = d[CELLS].to_numpy().astype(bool)
    rebuilt = np.array([next((k for k, m in regions.items() if (m.ravel() == v).all()), 'RS')
                        for v in visits])
    category_rule = bool((rebuilt == d.Category.to_numpy()).all())

    hoa = pd.read_csv(ROOT / 'data/raw/humans_only_absent.csv')
    shared = d.merge(hoa[['Dyad', 'Round', 'Player', 'Category', 'DLIndex']],
                     on=['Dyad', 'Round', 'Player'], suffixes=('_full', '_hoa'))
    checks = {
        'sha256': digest,
        'sha256_matches_documented': digest == SOURCE_SHA256,
        'rows': int(len(d)),
        'dyads': int(d.Dyad.nunique()),
        'category_equals_exact_focal_match': category_rule,
        'non_rs_rows': int((d.Category != 'RS').sum()),
        'shared_rows_with_filtered_file': int(len(shared)),
        'shared_category_matches': bool((shared.Category_full == shared.Category_hoa).all()),
        'shared_dlindex_max_abs_diff': float((shared.DLIndex_full - shared.DLIndex_hoa).abs().max()),
        'note_consistency': ('Consistency no se compara: el archivo filtrado la define contra la '
                             'ronda ausente anterior del mismo jugador'),
    }
    for key in ['sha256_matches_documented', 'category_equals_exact_focal_match', 'shared_category_matches']:
        if not checks[key]:
            raise AssertionError(f'Comprobacion de procedencia fallida: {key}')
    if checks['shared_dlindex_max_abs_diff'] > 1e-12:
        raise AssertionError('DLIndex no coincide en las filas compartidas')
    return d.sort_values(['Dyad', 'Player', 'Round']), checks


# --- 2. utilidades de cadena --------------------------------------------------

def counts(pairs, states):
    idx = {s: i for i, s in enumerate(states)}
    C = np.zeros((len(states), len(states)))
    for a, b in pairs:
        if a in idx and b in idx:
            C[idx[a], idx[b]] += 1
    return C


def normalize(C):
    row = C.sum(1, keepdims=True)
    return np.divide(C, row, out=np.full_like(C, np.nan), where=row > 0)


def irreducible(C):
    """True si todos los estados con datos se alcanzan entre si."""
    keep = C.sum(1) > 0
    A = (C[np.ix_(keep, keep)] > 0).astype(int)
    R = A.copy()
    for _ in range(len(A)):
        R = ((R + R @ A) > 0).astype(int)
    return bool(R.all())


def stationary(P, C):
    keep = C.sum(1) > 0
    Q = np.nan_to_num(P[np.ix_(keep, keep)])
    Q = Q / Q.sum(1, keepdims=True)
    w, v = np.linalg.eig(Q.T)
    p = np.abs(np.real(v[:, np.argmin(np.abs(w - 1))]))
    full = np.zeros(len(P))
    full[keep] = p / p.sum()
    return full


def hitting(P, states, target):
    non = [i for i, s in enumerate(states) if s not in target]
    A = np.eye(len(non)) - np.nan_to_num(P[np.ix_(non, non)])
    try:
        h = np.linalg.solve(A, np.ones(len(non)))
    except np.linalg.LinAlgError:
        h = np.full(len(non), np.nan)
    return {states[i]: float(v) for i, v in zip(non, h)}


def empirical_runs(d):
    """Duracion observada de cada racha, marcando las cortadas por el final de la partida."""
    runs = defaultdict(list)
    censored = defaultdict(list)
    for _, g in d.groupby(['Dyad', 'Player']):
        c = g.Category.tolist()
        i = 0
        while i < len(c):
            j = i
            while j + 1 < len(c) and c[j + 1] == c[i]:
                j += 1
            runs[c[i]].append(j - i + 1)
            censored[c[i]].append(j == len(c) - 1)
            i = j + 1
    out = {}
    for s in S:
        if not runs[s]:
            continue
        arr = np.array(runs[s], float)
        complete = arr[~np.array(censored[s])]
        out[s] = {'n_runs': int(len(arr)), 'mean_all_runs': round(float(arr.mean()), 2),
                  'mean_complete_runs_only': (None if not len(complete) else round(float(complete.mean()), 2)),
                  'right_censored_share': round(float(np.mean(censored[s])), 3)}
    return out


def player_pairs(d, rounds=None, condition=None, consecutive_only=True):
    out = []
    for _, g in d.groupby(['Dyad', 'Player']):
        if condition:
            g = g[g.Is_there == condition]
        if rounds:
            g = g[(g.Round >= rounds[0]) & (g.Round <= rounds[1])]
        r, c = g.Round.tolist(), g.Category.tolist()
        for i in range(len(c) - 1):
            if consecutive_only and r[i + 1] != r[i] + 1:
                continue
            out.append((c[i], c[i + 1]))
    return out


# --- 3. estados de diada, con y sin identidad ---------------------------------

def dyad_states(d, rounds=None, ordered=False):
    """Estado por ronda. ordered=True conserva quien es quien, con los jugadores ordenados por id."""
    rows = []
    for (dy, rd), g in d.groupby(['Dyad', 'Round']):
        if rounds and not (rounds[0] <= rd <= rounds[1]):
            continue
        if g.Player.nunique() != 2:
            continue
        by_player = g.set_index('Player').Category
        players = sorted(by_player.index)
        state = (by_player[players[0]], by_player[players[1]])
        rows.append((dy, rd, state if ordered else tuple(sorted(state))))
    return pd.DataFrame(rows, columns=['Dyad', 'Round', 'state']).sort_values(['Dyad', 'Round'])


def role_transitions(d):
    """Desde un par focal complementario: permanece, intercambia roles o sale."""
    comp_ordered = {(a, b) for pair in COMP_UNORDERED for a, b in (pair, pair[::-1])}
    ds = dyad_states(d, ordered=True)
    tally = {'stay': 0, 'direct_swap': 0, 'exit': 0}
    by_family = defaultdict(lambda: {'stay': 0, 'direct_swap': 0, 'exit': 0})
    for _, g in ds.groupby('Dyad'):
        seq = dict(zip(g.Round, g.state))
        for rd, state in seq.items():
            nxt = seq.get(rd + 1)
            if nxt is None or state not in comp_ordered:
                continue
            key = '/'.join(sorted(state))
            kind = 'stay' if nxt == state else ('direct_swap' if nxt == state[::-1] else 'exit')
            tally[kind] += 1
            by_family[key][kind] += 1
    return tally, dict(by_family)


# --- 4. orden del proceso -----------------------------------------------------

def sequences(d, condition=None, min_len=3):
    """Tramos de rondas consecutivas por jugador, opcionalmente solo de una condicion."""
    out = []
    for _, g in d.groupby(['Dyad', 'Player']):
        gg = g if condition is None else g[g.Is_there == condition]
        r, c = gg.Round.tolist(), gg.Category.tolist()
        current = c[:1]
        for i in range(1, len(c)):
            if r[i] == r[i - 1] + 1:
                current.append(c[i])
            else:
                if len(current) >= min_len:
                    out.append((g.Dyad.iloc[0], current))
                current = [c[i]]
        if len(current) >= min_len:
            out.append((g.Dyad.iloc[0], current))
    return out


def order_nll(train, evaluate_on, order, start=2, alpha=0.5):
    """Log-loss por observacion, dejando una diada fuera.

    Se entrena con `train` y se evalua siempre en las mismas posiciones de
    `evaluate_on`, desde `start`, para que los ordenes sean comparables.
    """
    dyads = sorted({dy for dy, _ in evaluate_on})
    rec = []
    for held in dyads:
        table = defaultdict(Counter)
        for dy, s in train:
            if dy == held:
                continue
            for i in range(order, len(s)):
                table[tuple(s[i - order:i])][s[i]] += 1
        for dy, s in evaluate_on:
            if dy != held:
                continue
            for i in range(start, len(s)):
                c = table[tuple(s[i - order:i])]
                total = sum(c.values()) + alpha * len(S)
                probs = {st: (c[st] + alpha) / total for st in S}
                rec.append((dy, -np.log(probs[s[i]]), max(probs, key=probs.get) == s[i]))
    return pd.DataFrame(rec, columns=['Dyad', 'nll', 'hit'])


def order_comparison(d, seed=0, reps=2000):
    """Ordenes 0, 1 y 2 sobre el mismo soporte, con una rejilla de sensibilidad.

    Varia el suavizado Dirichlet y si los ordenes bajos se entrenan tambien con
    los tramos de longitud dos, que el orden 2 no puede usar.
    """
    rng = np.random.default_rng(seed)
    result = {}
    for label, condition in [('ausentes consecutivas', 'Unicorn_Absent'), ('todas las rondas', None)]:
        evaluate_on = sequences(d, condition, min_len=3)
        variants = {}
        for train_min_len in (2, 3):
            train = sequences(d, condition, min_len=train_min_len)
            for alpha in (0.1, 0.5, 1.0):
                per_order = {o: order_nll(train, evaluate_on, o, alpha=alpha) for o in (0, 1, 2)}
                diff = per_order[1].nll.to_numpy() - per_order[2].nll.to_numpy()
                groups = {k: diff[v] for k, v in per_order[1].groupby('Dyad').indices.items()}
                keys = list(groups)
                boot = [float(np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))]).mean())
                        for _ in range(reps)]
                variants[f'tramos>={train_min_len}, dirichlet={alpha}'] = {
                    'log_loss': {o: round(float(t.nll.mean()), 4) for o, t in per_order.items()},
                    'accuracy': {o: round(float(t.hit.mean()), 3) for o, t in per_order.items()},
                    'order2_minus_order1_gain': round(float(diff.mean()), 4),
                    'ci95_clustered_by_dyad': [round(float(np.percentile(boot, 2.5)), 4),
                                               round(float(np.percentile(boot, 97.5)), 4)]}
        gains = [v['order2_minus_order1_gain'] for v in variants.values()]
        result[label] = {
            'evaluation_positions': int(sum(max(0, len(s) - 2) for _, s in evaluate_on)),
            'note': 'mismo conjunto de evaluacion para los tres ordenes en cada variante',
            'variants': variants,
            'gain_range_across_variants': [round(min(gains), 4), round(max(gains), 4)],
            'sign_stable_across_variants': bool(min(gains) > 0 or max(gains) < 0),
            'every_ci_excludes_zero': all(v['ci95_clustered_by_dyad'][0] > 0 or v['ci95_clustered_by_dyad'][1] < 0
                                          for v in variants.values())}
    return result


# --- 5. informe ---------------------------------------------------------------

def main():
    d, provenance = load_and_check()
    report = {'provenance_checks': provenance,
              'rounds': [int(d.Round.min()), int(d.Round.max())]}

    chains = {
        'todas las rondas 1-60': player_pairs(d),
        'rondas tempranas 1-20': player_pairs(d, rounds=(1, 20)),
        'rondas tardias 41-60': player_pairs(d, rounds=(41, 60)),
        'solo ausentes consecutivas': player_pairs(d, condition='Unicorn_Absent'),
    }
    hoa = pd.read_csv(ROOT / 'data/raw/humans_only_absent.csv').sort_values(['Dyad', 'Player', 'Round'])
    chains['archivo filtrado (comparacion)'] = [
        (a, b) for _, g in hoa.groupby(['Dyad', 'Player'])
        for a, b in zip(g.Category.tolist()[:-1], g.Category.tolist()[1:])]

    report['player_chains'] = {}
    for name, pairs in chains.items():
        C = counts(pairs, S)
        P = normalize(C)
        irr = irreducible(C)
        pi = stationary(P, C) if irr else None
        report['player_chains'][name] = {
            'transitions': int(C.sum()),
            'n_by_state': {s: int(C[i].sum()) for i, s in enumerate(S)},
            'stay_probability': {s: (None if C[i].sum() == 0 else round(float(P[i, i]), 3)) for i, s in enumerate(S)},
            'markov_implied_mean_dwell': {
                s: (None if C[i].sum() == 0 or P[i, i] >= 1 else round(1 / (1 - float(P[i, i])), 2))
                for i, s in enumerate(S)},
            'irreducible': irr,
            'stationary': ({s: round(float(pi[i]), 3) for i, s in enumerate(S)} if irr else
                           'no se reporta: la cadena estimada es reducible'),
        }
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        pd.DataFrame(C, index=S, columns=S).to_csv(OUT / f'counts_player_{slug}.csv')
        if name == 'todas las rondas 1-60':
            pd.DataFrame(P, index=S, columns=S).round(4).to_csv(OUT / 'transition_player_all_rounds.csv')
            ev = np.sort(np.abs(np.linalg.eigvals(np.nan_to_num(P))))[::-1]
            report['player_chains'][name]['eigenvalue_moduli'] = [round(float(v), 4) for v in ev[:5]]

    report['empirical_dwell_calendar_rounds'] = empirical_runs(d)
    report['dwell_note'] = ('La duracion observada esta censurada por la derecha en las rachas que llegan '
                            'a la ronda 60, asi que subestima la duracion real. La diferencia con el valor '
                            'implicado por la cadena no mide por si sola el ajuste del supuesto de Markov.')

    by_cond = d.groupby('Is_there').Category.value_counts(normalize=True).unstack().round(3)
    report['share_by_condition'] = by_cond.fillna(0).to_dict('index')
    report['mean_tiles_uncovered_by_condition'] = {
        k: float(v) for k, v in d.groupby('Is_there').Size_visited.mean().round(2).to_dict().items()}

    ds = dyad_states(d)
    obs = list(pd.unique(ds.state))
    lab = {t: '/'.join(t) for t in obs}
    pairs = [(a, b) for _, g in ds.groupby('Dyad')
             for (r1, a), (r2, b) in zip(zip(g.Round, g.state), list(zip(g.Round, g.state))[1:]) if r2 == r1 + 1]
    C = counts(pairs, obs)
    P = normalize(C)
    freq = ds.state.value_counts()
    comp_states = [t for t in obs if t in COMP_UNORDERED]
    report['dyad_chain'] = {
        'dyad_rounds': int(len(ds)),
        'distinct_states': len(obs),
        'transitions': int(C.sum()),
        'identity_note': 'estados sin identidad de jugador; ver role_transitions para la version ordenada',
        'top_states': {lab[t]: int(freq[t]) for t in freq.index[:12]},
        'share_complementary_focal': round(float(sum(freq.get(t, 0) for t in comp_states) / len(ds)), 4),
        'stay_probability': {lab[t]: round(float(P[i, i]), 3) for i, t in enumerate(obs) if freq[t] >= 5},
        'markov_implied_mean_dwell': {lab[t]: (None if P[i, i] >= 1 else round(1 / (1 - float(P[i, i])), 2))
                                      for i, t in enumerate(obs) if freq[t] >= 5},
    }
    hit = hitting(P, obs, set(comp_states))
    report['dyad_chain']['expected_rounds_to_reach_a_complementary_focal_pair'] = {
        lab[s]: (None if not np.isfinite(h) else round(float(h), 1))
        for s, h in hit.items() if freq.get(s, 0) >= 5}
    report['dyad_chain']['hitting_note'] = ('Valor implicado por una cadena homogenea estimada mezclando las '
                                            '45 diadas y las 60 rondas. El proceso no es homogeneo, asi que no '
                                            'es una estimacion de cuanto tardan las personas.')
    pd.DataFrame(C, index=[lab[t] for t in obs], columns=[lab[t] for t in obs]).to_csv(OUT / 'counts_dyad_pairs.csv')
    pd.DataFrame(P, index=[lab[t] for t in obs], columns=[lab[t] for t in obs]).round(4).to_csv(OUT / 'transition_dyad_pairs.csv')

    for name, window in [('temprana 1-20', (1, 20)), ('tardia 41-60', (41, 60))]:
        sub = dyad_states(d, rounds=window)
        pr = [(a, b) for _, g in sub.groupby('Dyad')
              for (r1, a), (r2, b) in zip(zip(g.Round, g.state), list(zip(g.Round, g.state))[1:]) if r2 == r1 + 1]
        C2 = counts(pr, obs)
        inside = [i for i, t in enumerate(obs) if t in COMP_UNORDERED]
        outside = [i for i in range(len(obs)) if i not in inside]
        stay = C2[inside][:, inside].trace() / max(C2[inside].sum(), 1)
        enter = C2[np.ix_(outside, inside)].sum() / max(C2[outside].sum(), 1)
        report['dyad_chain'][f'ventana_{name}'] = {
            'dyad_rounds': int(len(sub)),
            'share_complementary_focal': round(float(sub.state.isin(comp_states).mean()), 4),
            'stay_in_complementary_focal': round(float(stay), 3),
            'enter_complementary_focal_from_outside': round(float(enter), 3)}

    tally, by_family = role_transitions(d)
    report['role_transitions_identity_preserved'] = {
        'from_a_complementary_focal_pair': tally,
        'by_family': by_family,
        'note': ('Un intercambio directo es que los dos jugadores cambien de lado entre rondas consecutivas. '
                 'Su ausencia no excluye intercambios que pasen por otro estado intermedio.')}

    report['process_order'] = order_comparison(d)
    report['process_order_note'] = ('Tablas de frecuencias por conteo, evaluadas dejando una diada fuera sobre '
                                    'el mismo conjunto de posiciones para los tres ordenes. Se reporta una rejilla '
                                    'de suavizado y de soporte de entrenamiento porque el resultado en rondas '
                                    'ausentes depende de esas decisiones. Diagnostico del proceso, no una '
                                    'evaluacion de especializacion.')

    (OUT / 'strategy_chain_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: report[k] for k in ['provenance_checks', 'role_transitions_identity_preserved', 'process_order']},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
