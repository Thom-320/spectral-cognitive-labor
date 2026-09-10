#!/usr/bin/env python3
"""Offline, model-free reconstruction. Writes only to a NEW output directory.

Original experiment/data: Andrade-Lotero and Goldstone (2021).
Metric definitions: EAndrade-Lotero/SODCL at UPSTREAM below, Measures.py/FRA.py.
No outcome is constructed and no historical script or result is overwritten.
Requires Python >=3.10 and NumPy; no sklearn, network, or training.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'b5cf4d4d5334f3b7d048d7c7a1ba9d37722dc89c'
ATOL, RTOL = 1e-10, 1e-8
CELLS = [f'a{i}{j}' for i in range(1, 9) for j in range(1, 9)]


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def focal_regions():
    """Original eight fixed regions, row-major. IN is the central 6x6."""
    y, x = np.indices((8, 8))
    inside = (x > 0) & (x < 7) & (y > 0) & (y < 7)
    return np.array([np.ones((8, 8), bool), np.zeros((8, 8), bool),
                     y >= 4, y < 4, x < 4, x >= 4, inside, ~inside]).reshape(8, 64)


def jaccard(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 1.0


def reconstruct(rows):
    """Past/current-only metrics. First absent consistency is unavailable.

    Consistency follows the original implementation: previous ABSENT round,
    regardless of intervening present rounds. No lead-dependent row filter.
    """
    groups = {}
    keys = set()
    for row in rows:
        key = (row['Dyad'], int(row['Round']), row['Player'])
        if key in keys:
            raise ValueError(f'Duplicate observation: {key}')
        keys.add(key)
        groups.setdefault(key[:2], []).append(row)
    previous = {}
    result = []
    templates = focal_regions()
    for (dyad, rnd), pair in sorted(groups.items()):
        if len(pair) != 2 or len({r['Player'] for r in pair}) != 2:
            raise ValueError(f'Not exactly two players: {dyad}, {rnd}')
        if len({r['Is_there'] for r in pair}) != 1:
            raise ValueError('Inconsistent presence flags')
        if pair[0]['Is_there'] != 'Unicorn_Absent':
            continue
        pair = sorted(pair, key=lambda r: r['Player'])
        raw = np.array([[float(r[c]) for c in CELLS] for r in pair])
        if not np.isin(raw, [0, 1]).all():
            raise ValueError('Nonbinary visits')
        visits = raw.astype(bool)
        joint = int(np.count_nonzero(visits[0] & visits[1]))
        dl = float(np.count_nonzero(visits[0] ^ visits[1]) / 64)
        for row, vector in zip(pair, visits):
            if float(row['Joint']) != joint:
                raise ValueError('Raw Joint disagrees with visits')
            key = (dyad, row['Player'])
            prev = previous.get(key)
            result.append(dict(Dyad=dyad, Round=rnd, Player=row['Player'],
                               previous_absent_round=prev[0] if prev else None,
                               DLIndex=dl, Similarity=max(jaccard(vector, t) for t in templates),
                               Consistency=jaccard(vector, prev[1]) if prev else None,
                               Joint=joint, Size_visited=int(vector.sum()), visits=vector))
            previous[key] = (rnd, vector)
    return result


def representations():
    """Euclidean R^64; unweighted eight-neighbor combinatorial Laplacian."""
    y, x = np.indices((8, 8))
    coords = np.column_stack((x.ravel(), y.ravel()))
    distance = np.abs(coords[:, None, :] - coords[None, :, :]).max(axis=2)
    adjacency = (distance == 1).astype(float)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    values, vectors = np.linalg.eigh(laplacian)
    repeated = np.isclose(values, values[1], atol=ATOL, rtol=RTOL)
    if repeated.sum() != 2 or repeated[0]:
        raise ValueError('Expected an isolated two-dimensional Fiedler space')
    qf, _ = np.linalg.qr(vectors[:, repeated])
    centered = coords - coords.mean(axis=0)
    qxy, _ = np.linalg.qr(centered.astype(float))
    return laplacian, values, qf, qxy


def geometry(margin, qf, qxy):
    norm2 = float(margin @ margin)
    if norm2 == 0:
        return dict(margin_zero=True, E_F=0.0, E_xy=0.0, dominant_score=0.0)
    return dict(margin_zero=False, E_F=float(np.linalg.norm(qf.T @ margin)**2 / norm2),
                E_xy=float(np.linalg.norm(qxy.T @ margin)**2 / norm2),
                dominant_score=float(np.max(np.abs(qxy.T @ margin)) / np.sqrt(norm2)))


def cohort(metrics, qf, qxy):
    result = []
    for dyad in sorted({r['Dyad'] for r in metrics}):
        all_rows = [r for r in metrics if r['Dyad'] == dyad]
        rounds = sorted({r['Round'] for r in all_rows})[:5]
        if len(rounds) != 5:
            raise ValueError(f'Fewer than five absent rounds: {dyad}')
        selected = [r for r in all_rows if r['Round'] in rounds]
        players = sorted({r['Player'] for r in selected})
        if len(players) != 2 or len(selected) != 10:
            raise ValueError('Incomplete early dyad')
        counts = [np.sum([r['visits'].astype(float) for r in selected if r['Player'] == p], axis=0)
                  for p in players]
        valid_consistency = [r['Consistency'] for r in selected if r['Consistency'] is not None]
        if any(r['previous_absent_round'] is not None and r['previous_absent_round'] >= r['Round']
               for r in selected):
            raise ValueError('Future consistency input')
        result.append(dict(Dyad=dyad, early_rounds=','.join(map(str, rounds)),
                           cutoff_round=rounds[-1], elapsed_present_rounds=rounds[-1]-5,
                           early_round_count=5, player_rows=10,
                           consistency_observations=len(valid_consistency),
                           DLIndex_early=float(np.mean([r['DLIndex'] for r in selected])),
                           Similarity_early=float(np.mean([r['Similarity'] for r in selected])),
                           Consistency_early=float(np.mean(valid_consistency)),
                           **geometry(counts[0]-counts[1], qf, qxy)))
    return result


def verify_history(metrics, historical):
    lookup = {(r['Dyad'], r['Round'], r['Player']): r for r in metrics}
    checks = []
    for column in ['DLIndex', 'Similarity', 'Consistency', 'Joint', 'Size_visited']:
        errors, missing = [], 0
        for old in historical:
            new = lookup[(old['Dyad'], int(old['Round']), old['Player'])][column]
            old_value = float(old[column]) if old[column] else None
            if new is None or old_value is None:
                missing += 1
                if new != old_value:
                    raise AssertionError(f'Missingness mismatch: {column}')
            else:
                errors.append(abs(new-old_value))
        maximum = max(errors, default=0)
        if maximum > 1e-12:
            raise AssertionError(f'Historical metric mismatch: {column}, {maximum}')
        checks.append(dict(metric=column, matched=len(errors), matching_missing=missing,
                           max_abs_error=maximum))
    return checks


def projector_checks(laplacian, values, qf, qxy):
    pf, pxy = qf @ qf.T, qxy @ qxy.T
    singular = np.linalg.svd(qf.T @ qxy, compute_uv=False)
    checks = dict(fiedler_eigenvalues=values[1:3].tolist(), next_eigenvalue=float(values[3]),
                  eigenspace_atol=ATOL, eigenspace_rtol=RTOL,
                  principal_angles_degrees=np.degrees(np.arccos(np.clip(singular, 0, 1))).tolist(),
                  projector_frobenius_distance=float(np.linalg.norm(pf-pxy)),
                  projector_operator_distance=float(np.linalg.norm(pf-pxy, 2)))
    theta = 0.731
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    permutations = [np.rot90(np.arange(64).reshape(8, 8), k) for k in range(4)]
    permutations += [np.fliplr(p) for p in permutations]
    for name, q, p in [('F', qf, pf), ('xy', qxy, pxy)]:
        residuals = [np.linalg.norm(q.T @ q-np.eye(2)), np.linalg.norm(p-p.T),
                     np.linalg.norm(p @ p-p), np.linalg.norm((q @ rotation) @ (q @ rotation).T-p),
                     np.linalg.norm(p @ np.ones(64))]
        residuals += [np.linalg.norm(p[np.ix_(perm.ravel(), perm.ravel())]-p) for perm in permutations]
        checks[f'{name}_max_invariance_residual'] = float(max(residuals))
        assert max(residuals) < 1e-10
    checks['eigen_residual'] = float(np.linalg.norm(laplacian @ qf-qf*values[1]))
    assert checks['eigen_residual'] < 1e-10
    zero = geometry(np.zeros(64), qf, qxy)
    assert zero['margin_zero'] and zero['E_F'] == zero['E_xy'] == 0
    single = geometry(qxy[:, 0], qf, qxy)
    mixed = geometry((qxy[:, 0]+qxy[:, 1])/np.sqrt(2), qf, qxy)
    assert np.isclose(single['E_xy'], mixed['E_xy'])
    assert single['dominant_score'] > mixed['dominant_score']
    checks['energy_not_axial_orientation_example'] = dict(single=single, mixed=mixed)
    return checks


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Must not exist')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new directory to preserve prior runs')
    raw_path = ROOT/'data/raw/performances.csv'
    old_path = ROOT/'data/raw/humans_only_absent.csv'
    raw = read_csv(raw_path)
    historical = read_csv(old_path)
    metrics = reconstruct(raw)
    checks = verify_history(metrics, historical)
    laplacian, values, qf, qxy = representations()
    math_checks = projector_checks(laplacian, values, qf, qxy)
    early = cohort(metrics, qf, qxy)
    # Recompute from each actually available prefix; future observations must be unnecessary.
    for row in early:
        prefix = [r for r in raw if r['Dyad'] == row['Dyad'] and int(r['Round']) <= row['cutoff_round']]
        assert cohort(reconstruct(prefix), qf, qxy)[0] == row
    lookup = {(r['Dyad'], r['Player'], int(r['Round'])): r for r in raw}
    expected = {k for k, r in lookup.items() if r['Is_there'] == 'Unicorn_Absent'
                and lookup.get((k[0], k[1], k[2]+1), {}).get('Is_there') == 'Unicorn_Absent'}
    assert expected == {(r['Dyad'], r['Player'], int(r['Round'])) for r in historical}
    old_features = {r['Dyad']: r for r in read_csv(ROOT/'data/results/early_prediction_features.csv')}
    comparison = [dict(Dyad=r['Dyad'], historical_rounds=old_features[r['Dyad']]['early_rounds'],
                       historical_cutoff=max(map(int, old_features[r['Dyad']]['early_rounds'].split(','))),
                       reconstructed_rounds=r['early_rounds'], reconstructed_cutoff=r['cutoff_round']) for r in early]
    ef, exy = np.array([[r['E_F'], r['E_xy']] for r in early]).T
    summary = dict(raw_rows=len(raw), absent_player_rows=len(metrics), historical_player_rows=len(historical),
                   dyads=len(early), cutoff_min=min(r['cutoff_round'] for r in early),
                   cutoff_max=max(r['cutoff_round'] for r in early),
                   historical_cutoffs_ge_30=sum(r['historical_cutoff'] >= 30 for r in comparison),
                   zero_margins=sum(r['margin_zero'] for r in early),
                   historical_metrics=checks, prefix_invariance_dyads=len(early),
                   energy_pearson=float(np.corrcoef(ef, exy)[0, 1]),
                   energy_mean_abs_difference=float(np.mean(np.abs(ef-exy))),
                   energy_max_abs_difference=float(np.max(np.abs(ef-exy))),
                   projectors=math_checks, models_fitted=0, outcome='pending team decision')
    # Snapshot current historical paths, not new audit outputs; verify hashes after writing.
    preserved = sorted([*ROOT.glob('data/raw/*'), *ROOT.glob('data/results/*'), *ROOT.glob('figures/*'),
                        *ROOT.glob('src/*.py'), *ROOT.glob('docs/*.md'), ROOT/'README.md', ROOT/'STATUS.md'])
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in preserved if p.is_file()}
    payloads = {
        'early_cohort.csv': csv_bytes(early),
        'historical_vs_reconstructed_rounds.csv': csv_bytes(comparison),
        'absent_round_metrics.csv': csv_bytes([{k: v for k, v in r.items() if k != 'visits'} for r in metrics]),
        'summary.json': (json.dumps(summary, indent=2, allow_nan=False)+'\n').encode(),
    }
    args.output.mkdir(parents=True)
    for name, content in payloads.items():
        (args.output/name).write_bytes(content)
    with tarfile.open(args.output/'historical_evaluation.tar.gz', 'w:gz') as archive:
        for name in before:
            archive.add(ROOT/name, arcname=name)
    np.savez(args.output/'projectors.npz', Q_F=qf, Q_xy=qxy, P_F=qf@qf.T, P_xy=qxy@qxy.T,
             eigenvalues=values)
    provenance = dict(historical_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      upstream_commit=UPSTREAM, python=platform.python_version(), numpy=np.__version__,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      preserved_sha256=before,
                      output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()})
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    assert before == {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in before}
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
