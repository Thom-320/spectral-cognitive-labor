"""Pruebas del simulador de potencia y recuperacion.

No tocan humans_full.csv: comprueban las propiedades matematicas del ajuste,
del generador y del efecto oraculo sobre entradas sinteticas.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'audit' / 'model_recovery'))
import power_recovery as pr  # noqa: E402


def toy(n=4000, seed=0):
    """Transiciones sinteticas con dependencia real de z, s y o."""
    rng = np.random.default_rng(seed)
    z = rng.integers(0, pr.NS, n)
    s = rng.integers(0, pr.NSC, n)
    o = rng.integers(0, pr.NOV, n)
    y = (z + s + o) % pr.NS
    dyad = rng.integers(0, 10, n)
    return dict(dyad=dyad, z=z, s=s, o=o, y=y)


class TestFit(unittest.TestCase):
    def test_rows_are_distributions(self):
        P = pr.fit(pr.count(**{k: v for k, v in toy().items() if k != 'dyad'}))
        for name, shape in [('M0', ()), ('M1', (pr.NS,)), ('M2', (pr.NS, pr.NSC)),
                            ('M3', (pr.NS, pr.NSC, pr.NOV))]:
            arr = P[name]
            self.assertEqual(arr.shape, shape + (pr.NS,), name)
            np.testing.assert_allclose(arr.sum(-1), np.ones(shape), atol=1e-12,
                                       err_msg=name)
            self.assertTrue((arr > 0).all(), name)

    def test_empty_cell_backs_off_to_parent(self):
        """Sin datos en (z,s,o), M3 debe ser exactamente M2; asi el contraste es justo."""
        C3 = np.zeros((pr.NS, pr.NSC, pr.NOV, pr.NS))
        C3[0, 0, 0] = [5, 3, 0, 0, 0, 0, 0, 0, 2]
        P = pr.fit(C3)
        np.testing.assert_allclose(P['M3'][0, 0, 1], P['M2'][0, 0], atol=1e-12)
        self.assertGreater(np.abs(P['M3'][0, 0, 0] - P['M2'][0, 0]).max(), 1e-6)

    def test_more_smoothing_pulls_child_towards_parent(self):
        C3 = pr.count(**{k: v for k, v in toy().items() if k != 'dyad'})
        far = np.abs(pr.fit(C3, 0.5)['M3'] - pr.fit(C3, 0.5)['M2'][:, :, None, :]).mean()
        near = np.abs(pr.fit(C3, 200.0)['M3'] - pr.fit(C3, 200.0)['M2'][:, :, None, :]).mean()
        self.assertLess(near, far)


class TestLoo(unittest.TestCase):
    def test_shape_and_finiteness(self):
        S = toy()
        L, kappas = pr.loo_losses(S, kappa=5.0)
        self.assertEqual(L.shape, (4, len(S['z'])))
        self.assertTrue(np.isfinite(L).all())
        self.assertEqual(set(kappas), set(pr.MODELS))

    def test_true_structure_beats_marginal(self):
        """Con dependencia fuerte y muchos datos, M3 debe batir a M0."""
        L, _ = pr.loo_losses(toy(n=9000), kappa=2.0)
        self.assertLess(L[3].mean(), L[0].mean())

    def test_cv_chooses_from_the_grid(self):
        _, kappas = pr.loo_losses(toy(n=1500), kappa='cv', rng=np.random.default_rng(1))
        for m, k in kappas.items():
            self.assertIn(k, pr.KAPPA_GRID, m)


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = pr.Generator(toy(), np.random.default_rng(2))

    def test_draw_matches_the_real_shape(self):
        P3 = np.full((pr.NS, pr.NSC, pr.NOV, pr.NS), 1 / pr.NS)
        S = self.gen.draw(P3, rng=np.random.default_rng(3))
        self.assertEqual(len(S['z']), int(self.gen.lengths.sum()))
        self.assertEqual(int(S['dyad'].max()) + 1, self.gen.n_dyads)
        for k in ('z', 's', 'o', 'y'):
            self.assertTrue((S[k] >= 0).all())
        self.assertLess(S['y'].max(), pr.NS)
        self.assertLess(S['o'].max(), pr.NOV)

    def test_scaling_multiplies_the_sample(self):
        P3 = np.full((pr.NS, pr.NSC, pr.NOV, pr.NS), 1 / pr.NS)
        big = self.gen.draw(P3, n_dyads=2 * self.gen.n_dyads, rng=np.random.default_rng(4))
        self.assertEqual(len(big['z']), 2 * int(self.gen.lengths.sum()))

    def test_chain_is_sequential(self):
        """El estado de una transicion es el resultado de la anterior."""
        P3 = np.zeros((pr.NS, pr.NSC, pr.NOV, pr.NS))
        P3[..., :] = np.eye(pr.NS)[:, None, None, :]   # y = z siempre
        S = self.gen.draw(P3, rng=np.random.default_rng(5))
        np.testing.assert_array_equal(S['z'], S['y'])

    def test_oracle_delta_is_zero_without_overlap_dependence(self):
        rng = np.random.default_rng(6)
        base = rng.dirichlet(np.ones(pr.NS), size=(pr.NS, pr.NSC))
        flat = np.broadcast_to(base[:, :, None, :], (pr.NS, pr.NSC, pr.NOV, pr.NS)).copy()
        self.assertAlmostEqual(self.gen.oracle_delta(flat), 0.0, places=10)

    def test_oracle_delta_is_positive_with_overlap_dependence(self):
        rng = np.random.default_rng(7)
        P3 = rng.dirichlet(np.ones(pr.NS), size=(pr.NS, pr.NSC, pr.NOV))
        self.assertGreater(self.gen.oracle_delta(P3), 0.0)

    def test_best_m2_removes_the_overlap_axis(self):
        rng = np.random.default_rng(8)
        P3 = rng.dirichlet(np.ones(pr.NS), size=(pr.NS, pr.NSC, pr.NOV))
        ref = self.gen.best_m2(P3)
        for o in range(1, pr.NOV):
            np.testing.assert_allclose(ref[:, :, o], ref[:, :, 0], atol=1e-12)
        np.testing.assert_allclose(ref.sum(-1), 1.0, atol=1e-12)


class TestCI(unittest.TestCase):
    def test_interval_brackets_a_clear_shift(self):
        rng = np.random.default_rng(9)
        dyad = np.repeat(np.arange(30), 40)
        diff = rng.normal(0.5, 0.1, size=dyad.size)
        lo, hi = pr.clustered_ci(diff, dyad, rng)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreater(lo, 0.0)

    def test_coverage_is_close_to_nominal_under_the_null(self):
        """Un intervalo unico puede excluir cero por azar; lo que debe cumplirse
        es la cobertura nominal sobre replicas."""
        rng = np.random.default_rng(10)
        dyad = np.repeat(np.arange(30), 40)
        hits = 0
        reps = 600
        for _ in range(reps):
            diff = rng.normal(0.0, 0.5, size=dyad.size)
            lo, hi = pr.clustered_ci(diff, dyad, rng, B=300)
            hits += int(lo <= 0.0 <= hi)
        # El bootstrap percentil agrupado cubre algo por debajo del 95 % nominal
        # con pocas diadas; medido en 0,92-0,94 con 30 y con 45.
        self.assertGreater(hits / reps, 0.90)
        self.assertLess(hits / reps, 0.99)

    def test_clustering_widens_the_interval_when_dyads_differ(self):
        """Con efecto por diada, ignorar el agrupamiento subestimaria la incertidumbre."""
        rng = np.random.default_rng(11)
        dyad = np.repeat(np.arange(30), 40)
        shift = rng.normal(0.0, 0.4, size=30)[dyad]
        diff = shift + rng.normal(0.0, 0.05, size=dyad.size)
        lo, hi = pr.clustered_ci(diff, dyad, rng)
        naive = 1.96 * diff.std() / np.sqrt(diff.size)
        self.assertGreater((hi - lo) / 2, 2 * naive)


if __name__ == '__main__':
    unittest.main()
