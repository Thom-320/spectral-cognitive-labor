"""Scientific integrity checks; no predictive fitting or output generation."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_temporal_repair as audit


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lap, cls.values, cls.qf, cls.qxy = audit.representations()

    def test_original_focal_region_encoding(self):
        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789;:'
        codes = [alphabet, '', 'GHIJKLMNOPQRSTUVWXYZ0123456789;:',
                 'abcdefghijklmnopqrstuvwxyzABCDEF', 'abcdijklqrstyzABGHIJOPQRWXYZ4567',
                 'efghmnopuvwxCDEFKLMNSTUV012389;:', 'jklmnorstuvwzABCDEHIJKLMPQRSTUXYZ012',
                 'abcdefghipqxyFGNOVW3456789;:']
        original = np.array([[letter in code for letter in alphabet] for code in codes])
        np.testing.assert_array_equal(audit.focal_regions(), original)

    def test_graph_and_projector_properties(self):
        audit.projector_checks(self.lap, self.values, self.qf, self.qxy)
        self.assertEqual(np.trace(self.lap)/2, 210)
        self.assertEqual(np.count_nonzero(np.isclose(self.values, self.values[1],
                                                   atol=audit.ATOL, rtol=audit.RTOL)), 2)

    def test_change_basis_player_swap_and_scale(self):
        margin = np.arange(64, dtype=float)-19
        expected = audit.geometry(margin, self.qf, self.qxy)
        for changed in [-margin, 3*margin]:
            actual = audit.geometry(changed, self.qf, self.qxy)
            for key in ['E_F', 'E_xy', 'dominant_score']:
                self.assertAlmostEqual(actual[key], expected[key])
        rotated = audit.geometry(margin, self.qf[:, ::-1]*[-1, 1], self.qxy)
        self.assertAlmostEqual(rotated['E_F'], expected['E_F'])

    def test_empty_sets_and_zero_margin(self):
        zero = np.zeros(64, bool)
        self.assertEqual(audit.jaccard(zero, zero), 1)
        self.assertEqual(audit.jaccard(zero, ~zero), 0)
        result = audit.geometry(zero.astype(float), self.qf, self.qxy)
        self.assertTrue(result['margin_zero'])
        self.assertEqual(result['E_F'], 0)
        self.assertEqual(result['E_xy'], 0)

    def test_future_rows_cannot_change_features(self):
        # Five absent opportunities separated by present rounds; sixth is future.
        rows = []
        for rnd in range(1, 12):
            for player in ['A', 'B']:
                vector = [int((i < 32) == (player == 'A')) for i in range(64)]
                rows.append(dict(Dyad='synthetic', Round=str(rnd), Player=player,
                                 Is_there='Unicorn_Absent' if rnd % 2 else 'Unicorn_Present',
                                 Joint='0', **dict(zip(audit.CELLS, map(str, vector)))))
        expected = audit.cohort(audit.reconstruct(rows), self.qf, self.qxy)
        self.assertEqual(expected[0]['cutoff_round'], 9)
        self.assertEqual(expected[0]['consistency_observations'], 8)
        self.assertEqual(expected[0]['DLIndex_early'], 1)
        self.assertEqual(expected[0]['Similarity_early'], 1)
        self.assertEqual(expected[0]['Consistency_early'], 1)
        for row in rows:
            if int(row['Round']) > 9:
                for cell in audit.CELLS:
                    row[cell] = '0'
        self.assertEqual(audit.cohort(audit.reconstruct(rows), self.qf, self.qxy), expected)
        self.assertEqual(audit.cohort(audit.reconstruct(rows[:18]), self.qf, self.qxy), expected)

    def test_duplicate_and_incomplete_pairs_rejected(self):
        row = dict(Dyad='x', Round='1', Player='A', Is_there='Unicorn_Absent')
        with self.assertRaises(ValueError):
            audit.reconstruct([row, row])
        with self.assertRaises(ValueError):
            audit.reconstruct([row])


if __name__ == '__main__':
    unittest.main()
