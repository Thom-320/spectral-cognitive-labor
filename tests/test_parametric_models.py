"""Pruebas de la reimplementacion de MBIASES, WSLS y FRA.

Comprueban las propiedades que el articulo y el codigo original establecen:
anidamiento de los tres modelos, forma de la similitud, capa de mano temblorosa
y coherencia del simulador. No dependen de scipy salvo donde se indica.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'audit' / 'model_recovery'))
import parametric_models as pm  # noqa: E402


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    tiles = rng.random((n, 64)) < 0.5
    joint = tiles & (rng.random((n, 64)) < 0.5)
    z = rng.integers(0, pm.NS, n)
    I = np.zeros((n, pm.NS))
    foc = z != pm.RS
    I[np.flatnonzero(foc), z[foc]] = 1.0
    return dict(dyad=rng.integers(0, 10, n), z=z,
                score=rng.uniform(-128, 32, n), I=I,
                frasim=pm.frasim(tiles, joint), y=rng.integers(0, pm.NS, n))


class TestGeometry(unittest.TestCase):
    def test_focal_sizes_match_the_paper(self):
        got = {s: int(pm.FOCAL[pm.IDX[s]].sum()) for s in pm.STATES}
        self.assertEqual(got['ALL'], 64)
        self.assertEqual(got['NOTHING'], 0)
        for s in ('LEFT', 'RIGHT', 'TOP', 'BOTTOM'):
            self.assertEqual(got[s], 32, s)
        self.assertEqual(got['IN'], 36)     # el 6x6 interior
        self.assertEqual(got['OUT'], 28)
        self.assertTrue((pm.FOCAL[pm.IDX['IN']] | pm.FOCAL[pm.IDX['OUT']]).all())

    def test_sim_is_jaccard_with_the_empty_convention(self):
        # El orden de las casillas es por filas, asi que las 32 primeras son TOP.
        a = np.zeros((3, 64), bool)
        a[0, :32] = True
        a[1, :16] = True
        s = pm.sim_to(a, pm.FOCAL, pm.FOCAL_SIZE)
        self.assertAlmostEqual(s[0, pm.IDX['TOP']], 1.0)       # identica
        self.assertAlmostEqual(s[0, pm.IDX['ALL']], 0.5)       # 32 de 64
        self.assertAlmostEqual(s[0, pm.IDX['LEFT']], 16 / 48)  # cruza a la mitad
        self.assertAlmostEqual(s[1, pm.IDX['TOP']], 0.5)       # 16 de 32
        self.assertAlmostEqual(s[2, pm.IDX['NOTHING']], 1.0)   # vacia con vacia

    def test_frasim_is_zero_for_rs_and_single_term_for_all(self):
        rng = np.random.default_rng(1)
        i = rng.random((5, 64)) < 0.5
        j = rng.random((5, 64)) < 0.3
        f = pm.frasim(i, j)
        np.testing.assert_allclose(f[:, pm.RS], 0.0, atol=1e-12)
        only_first = pm.sim_to(i, pm.FOCAL, pm.FOCAL_SIZE)[:, pm.IDX['ALL']]
        np.testing.assert_allclose(f[:, pm.IDX['ALL']], only_first, atol=1e-12)
        self.assertTrue((f >= 0).all() and (f <= 2).all())


class TestThresh(unittest.TestCase):
    def test_matches_the_sigmoid_and_is_stable(self):
        x = np.array([-1e4, -50.0, 0.0, 3.0, 1e4])
        got = pm.thresh(x, 30.0, 3.0)
        self.assertTrue(np.isfinite(got).all())
        self.assertAlmostEqual(got[2 + 1], 0.5)
        self.assertAlmostEqual(got[0], 0.0)
        self.assertAlmostEqual(got[-1], 1.0)
        mid = pm.thresh(np.array([3.1]), 30.0, 3.0)[0]
        self.assertAlmostEqual(mid, 1 / (1 + np.exp(-30 * 0.1)))

    def test_is_monotone(self):
        x = np.linspace(-40, 40, 200)
        self.assertTrue((np.diff(pm.thresh(x, 30.0, 0.0)) >= 0).all())


class TestParametrisation(unittest.TestCase):
    def test_bias_sums_to_one_and_respects_the_symmetry(self):
        b = pm.bias_vector(0.13, 0.076, 0.058, 0.005,
                           1 - (0.13 + 0.076 + 4 * 0.058 + 2 * 0.005))
        self.assertAlmostEqual(b.sum(), 1.0)
        for s in ('RIGHT', 'TOP', 'BOTTOM'):
            self.assertAlmostEqual(b[pm.IDX[s]], b[pm.IDX['LEFT']])
        self.assertAlmostEqual(b[pm.IDX['OUT']], b[pm.IDX['IN']])
        self.assertAlmostEqual(b[pm.IDX['ALL']], 0.13)

    def test_pack_unpack_reproduces_the_published_values(self):
        for m in pm.MODELS:
            p = pm.PUBLISHED[m]
            bias, alpha, gamma, delta, z = pm.unpack(pm.pack(p, m), m)
            self.assertAlmostEqual(bias[pm.IDX['ALL']], p['bias']['ALL'], places=6)
            self.assertAlmostEqual(bias[pm.IDX['LEFT']], p['bias']['LR'], places=6)
            self.assertAlmostEqual(alpha, p['alpha'], places=4)
            self.assertAlmostEqual(delta, p['delta'], places=4)

    def test_free_parameter_counts(self):
        for m in pm.MODELS:
            self.assertEqual(len(pm.pack(pm.PUBLISHED[m], m)), pm.N_FREE[m], m)
        # Los conteos publicados anaden beta y epsilon, que el codigo original
        # optimiza dentro de [29, 30].
        self.assertEqual([pm.N_PUBLISHED[m] for m in pm.MODELS], [4, 7, 10])


class TestProbabilities(unittest.TestCase):
    def setUp(self):
        self.pre = toy()

    def test_rows_are_distributions_without_clipping(self):
        for m in pm.MODELS:
            P = pm.probabilities(pm.pack(pm.PUBLISHED[m], m), m, self.pre, clip=False)
            np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-10, err_msg=m)
            self.assertTrue((P > 0).all(), m)

    def test_shaky_hand_follows_equation_6(self):
        th = pm.pack(pm.PUBLISHED['FRA'], 'FRA')
        raw = pm.probabilities(th, 'FRA', self.pre, shaky=False, clip=False)
        ns = pm.probabilities(th, 'FRA', self.pre, shaky=True, clip=False)
        focal = [i for i in range(pm.NS) if i != pm.RS]
        np.testing.assert_allclose(ns[:, focal], pm.NON_SHAKY * raw[:, focal], atol=1e-12)
        np.testing.assert_allclose(ns[:, pm.RS],
                                   1 - pm.NON_SHAKY * (1 - raw[:, pm.RS]), atol=1e-12)
        self.assertTrue((ns[:, pm.RS] >= 1 - pm.NON_SHAKY - 1e-12).all())

    def test_models_are_nested(self):
        """FRA con delta = 0 es WSLS, y WSLS con alpha = 0 es MBIASES."""
        base = pm.PUBLISHED['FRA']
        th_fra = pm.pack(dict(bias=base['bias'], alpha=base['alpha'],
                              gamma=base['gamma'], delta=1e-12, z=base['z']), 'FRA')
        th_wsls = pm.pack(dict(bias=base['bias'], alpha=base['alpha'],
                               gamma=base['gamma']), 'WSLS')
        np.testing.assert_allclose(pm.probabilities(th_fra, 'FRA', self.pre),
                                   pm.probabilities(th_wsls, 'WSLS', self.pre), atol=1e-9)
        th_w0 = pm.pack(dict(bias=base['bias'], alpha=1e-12, gamma=base['gamma']), 'WSLS')
        th_mb = pm.pack(dict(bias=base['bias']), 'MBIASES')
        np.testing.assert_allclose(pm.probabilities(th_w0, 'WSLS', self.pre),
                                   pm.probabilities(th_mb, 'MBIASES', self.pre), atol=1e-9)

    def test_mbiases_ignores_the_state(self):
        P = pm.probabilities(pm.pack(pm.PUBLISHED['MBIASES'], 'MBIASES'),
                             'MBIASES', self.pre)
        np.testing.assert_allclose(P, np.broadcast_to(P[0], P.shape), atol=1e-12)

    def test_win_stay_only_raises_the_repeated_focal_region(self):
        pre = toy(50, seed=3)
        th = pm.pack(pm.PUBLISHED['WSLS'], 'WSLS')
        P = pm.probabilities(th, 'WSLS', pre, clip=False)
        th0 = pm.pack(dict(bias=pm.PUBLISHED['WSLS']['bias'], alpha=1e-12,
                           gamma=pm.PUBLISHED['WSLS']['gamma']), 'WSLS')
        P0 = pm.probabilities(th0, 'WSLS', pre, clip=False)
        rep = pre['I'].argmax(1)
        has = pre['I'].sum(1) > 0
        good = has & (pre['score'] > pm.PUBLISHED['WSLS']['gamma'] + 1)
        rows = np.flatnonzero(good)
        self.assertGreater(len(rows), 0)
        self.assertTrue((P[rows, rep[rows]] > P0[rows, rep[rows]]).all())
        idle = np.flatnonzero(~has)
        if len(idle):
            np.testing.assert_allclose(P[idle], P0[idle], atol=1e-12)

    def test_clipping_matches_the_original_bounds(self):
        pre = toy(20, seed=4)
        th = pm.pack(dict(bias=dict(ALL=1e-9, NOTHING=1e-9, LR=1e-9, IO=1e-9),
                          alpha=0.0, gamma=0.0, delta=0.0, z=0.0), 'MBIASES')
        P = pm.probabilities(th, 'MBIASES', pre, clip=True)
        self.assertTrue((P >= pm.LOWER_EPS - 1e-15).all())
        self.assertTrue((P <= pm.HIGH_EPS + 1e-15).all())


class TestObservedCategory(unittest.TestCase):
    def test_exact_match_only(self):
        t = np.stack([pm.FOCAL[pm.IDX['LEFT']], pm.FOCAL[pm.IDX['ALL']],
                      pm.FOCAL[pm.IDX['NOTHING']], pm.FOCAL[pm.IDX['LEFT']].copy()])
        t[3, 0] = ~t[3, 0]
        got = pm.observed_category(t)
        self.assertEqual(list(got[:3]), [pm.IDX['LEFT'], pm.IDX['ALL'], pm.IDX['NOTHING']])
        self.assertEqual(got[3], pm.RS)


if __name__ == '__main__':
    unittest.main()
