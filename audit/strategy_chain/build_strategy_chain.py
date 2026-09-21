#!/usr/bin/env python3
"""Cadena de Markov empirica sobre las estrategias del estudio original, con las 60 rondas.

Fuente: data/raw/humans_full.csv del repositorio SODCL
(https://raw.githubusercontent.com/EAndrade-Lotero/SODCL/master/Data/humans_full.csv,
sha256 9452e653a5ffd08f8148f15269d73b754f782ed7f48bc1c3292dc78b42105c54).
Las categorias verticales se llaman UP/DOWN alli; se renombran a TOP/BOTTOM.
Descriptivo: no ajusta modelos predictivos ni construye ningun outcome.
Uso: python audit/strategy_chain/build_strategy_chain.py
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
S = ['LEFT', 'RIGHT', 'TOP', 'BOTTOM', 'IN', 'OUT', 'ALL', 'NOTHING', 'RS']
COMP = {('LEFT', 'RIGHT'), ('BOTTOM', 'TOP'), ('IN', 'OUT'), ('ALL', 'NOTHING')}


def load():
    d = pd.read_csv(ROOT / 'data/raw/humans_full.csv')
    d['Category'] = d.Category.replace({'UP': 'TOP', 'DOWN': 'BOTTOM'})
    return d.sort_values(['Dyad', 'Player', 'Round'])


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
    """True if every state with data can reach every other (single communicating class)."""
    keep = C.sum(1) > 0
    A = (C[np.ix_(keep, keep)] > 0).astype(int)
    R = A.copy()
    for _ in range(len(A)):
        R = ((R + R @ A) > 0).astype(int)
    return bool(R.all())


def stationary(P, C):
    """Left eigenvector of the estimated chain, restricted to states with data."""
    keep = C.sum(1) > 0
    Q = np.nan_to_num(P[np.ix_(keep, keep)])
    Q = Q / Q.sum(1, keepdims=True)
    w, v = np.linalg.eig(Q.T)
    p = np.real(v[:, np.argmin(np.abs(w - 1))])
    p = np.abs(p) / np.abs(p).sum()
    full = np.zeros(len(P))
    full[keep] = p
    return full


def hitting(P, states, target):
    """Expected steps to first enter `target`, from each non-target state."""
    non = [i for i, s in enumerate(states) if s not in target]
    A = np.eye(len(non)) - np.nan_to_num(P[np.ix_(non, non)])
    try:
        h = np.linalg.solve(A, np.ones(len(non)))
    except np.linalg.LinAlgError:
        h = np.full(len(non), np.nan)
    return {states[i]: float(v) for i, v in zip(non, h)}


def player_pairs(d, rounds=None, consecutive_only=True, condition=None):
    out = []
    for _, g in d.groupby(['Dyad', 'Player']):
        if condition:
            g = g[g.Is_there == condition]
        if rounds:
            g = g[(g.Round >= rounds[0]) & (g.Round <= rounds[1])]
        r = g.Round.tolist()
        c = g.Category.tolist()
        for i in range(len(c) - 1):
            if consecutive_only and r[i + 1] != r[i] + 1:
                continue
            out.append((c[i], c[i + 1]))
    return out


def dyad_states(d, rounds=None):
    rows = []
    for (dy, rd), g in d.groupby(['Dyad', 'Round']):
        if rounds and not (rounds[0] <= rd <= rounds[1]):
            continue
        if g.Player.nunique() == 2:
            rows.append((dy, rd, tuple(sorted(g.Category.tolist()))))
    return pd.DataFrame(rows, columns=['Dyad', 'Round', 'state']).sort_values(['Dyad', 'Round'])


def main():
    d = load()
    report = {'source_rows': int(len(d)), 'dyads': int(d.Dyad.nunique()), 'rounds': [int(d.Round.min()), int(d.Round.max())]}

    # --- player-level chains -------------------------------------------------
    chains = {
        'todas las rondas 1-60': player_pairs(d),
        'rondas tempranas 1-20': player_pairs(d, rounds=(1, 20)),
        'rondas tardias 41-60': player_pairs(d, rounds=(41, 60)),
        'solo ausentes consecutivas': player_pairs(d, condition='Unicorn_Absent'),
    }
    hoa = pd.read_csv(ROOT / 'data/raw/humans_only_absent.csv').sort_values(['Dyad', 'Player', 'Round'])
    chains['archivo filtrado (comparacion)'] = [
        (a, b) for _, g in hoa.groupby(['Dyad', 'Player'])
        for a, b in zip(g.Category.tolist()[:-1], g.Category.tolist()[1:])
    ]
    report['player_chains'] = {}
    for name, pr in chains.items():
        C = counts(pr, S)
        P = normalize(C)
        irr = irreducible(C)
        pi = stationary(P, C) if irr else None
        diag = {s: (None if C[i].sum() == 0 else round(float(P[i, i]), 3)) for i, s in enumerate(S)}
        dwell = {s: (None if C[i].sum() == 0 or P[i, i] >= 1 else round(1 / (1 - float(P[i, i])), 2)) for i, s in enumerate(S)}
        report['player_chains'][name] = {
            'transitions': int(C.sum()),
            'n_by_state': {s: int(C[i].sum()) for i, s in enumerate(S)},
            'stay_probability': diag,
            'mean_dwell_rounds': dwell,
            'irreducible': irr,
            'stationary': ({s: round(float(pi[i]), 3) for i, s in enumerate(S)} if irr else
                           'no se reporta: la cadena estimada es reducible (hay estados absorbentes con pocos datos)'),
        }
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        pd.DataFrame(C, index=S, columns=S).to_csv(OUT / f'counts_player_{slug}.csv')
        if name == 'todas las rondas 1-60':
            pd.DataFrame(P, index=S, columns=S).round(4).to_csv(OUT / 'transition_player_all_rounds.csv')
            ev = np.sort(np.abs(np.linalg.eigvals(np.nan_to_num(P))))[::-1]
            report['player_chains'][name]['eigenvalue_moduli'] = [round(float(v), 4) for v in ev[:5]]
            report['player_chains'][name]['implied_timescales_rounds'] = [
                (None if v >= 1 or v <= 0 else round(float(-1 / np.log(v)), 2)) for v in ev[1:5]]

    # --- how the round condition changes the state ---------------------------
    by_cond = d.groupby('Is_there').Category.value_counts(normalize=True).unstack().round(3)
    report['share_by_condition'] = by_cond.fillna(0).to_dict('index')
    tiles = d.groupby('Is_there').Size_visited.mean().round(2).to_dict()
    report['mean_tiles_uncovered_by_condition'] = {k: float(v) for k, v in tiles.items()}

    # --- dyad-level chain ----------------------------------------------------
    ds = dyad_states(d)
    obs = list(pd.unique(ds.state))
    lab = {t: '/'.join(t) for t in obs}
    pr = [(a, b) for _, g in ds.groupby('Dyad')
          for (r1, a), (r2, b) in zip(zip(g.Round, g.state), list(zip(g.Round, g.state))[1:]) if r2 == r1 + 1]
    C = counts(pr, obs)
    P = normalize(C)
    freq = ds.state.value_counts()
    comp_states = [t for t in obs if t in COMP]
    report['dyad_chain'] = {
        'dyad_rounds': int(len(ds)),
        'distinct_states': len(obs),
        'transitions': int(C.sum()),
        'top_states': {lab[t]: int(freq[t]) for t in freq.index[:12]},
        'share_complementary_focal': round(float(sum(freq.get(t, 0) for t in comp_states) / len(ds)), 4),
        'stay_probability': {lab[t]: (None if C[i].sum() < 5 else round(float(P[i, i]), 3))
                             for i, t in enumerate(obs) if freq[t] >= 5},
        'mean_dwell_rounds': {lab[t]: (None if C[i].sum() < 5 or P[i, i] >= 1 else round(1 / (1 - float(P[i, i])), 2))
                              for i, t in enumerate(obs) if freq[t] >= 5},
    }
    hit = hitting(P, obs, set(comp_states))
    report['dyad_chain']['expected_rounds_to_reach_a_complementary_focal_pair'] = {
        lab[s]: (None if not np.isfinite(h) else round(float(h), 1)) for s, h in hit.items() if freq.get(s, 0) >= 5}
    pd.DataFrame(C, index=[lab[t] for t in obs], columns=[lab[t] for t in obs]).to_csv(OUT / 'counts_dyad_pairs.csv')
    pd.DataFrame(P, index=[lab[t] for t in obs], columns=[lab[t] for t in obs]).round(4).to_csv(OUT / 'transition_dyad_pairs.csv')

    # early vs late at dyad level
    for name, win in [('temprana 1-20', (1, 20)), ('tardia 41-60', (41, 60))]:
        sub = dyad_states(d, rounds=win)
        pr2 = [(a, b) for _, g in sub.groupby('Dyad')
               for (r1, a), (r2, b) in zip(zip(g.Round, g.state), list(zip(g.Round, g.state))[1:]) if r2 == r1 + 1]
        C2 = counts(pr2, obs)
        P2 = normalize(C2)
        idx = [i for i, t in enumerate(obs) if t in COMP]
        stay = np.nansum([C2[i, i] for i in idx]) / max(np.nansum([C2[i].sum() for i in idx]), 1)
        enter = np.nansum([C2[np.ix_([j for j in range(len(obs)) if j not in idx], idx)]])
        leave_pool = np.nansum([C2[j].sum() for j in range(len(obs)) if j not in idx])
        report['dyad_chain'][f'ventana_{name}'] = {
            'dyad_rounds': int(len(sub)),
            'share_complementary_focal': round(float(sub.state.isin(comp_states).mean()), 4),
            'stay_in_complementary_focal': round(float(stay), 3),
            'enter_complementary_focal_from_outside': round(float(enter / leave_pool), 3) if leave_pool else None,
        }

    (OUT / 'strategy_chain_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
