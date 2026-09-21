"""Pruebas del artefacto de la cadena de estrategias. Usan datos sinteticos, no el CSV completo."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'audit' / 'strategy_chain'))
import build_strategy_chain as chain


def frame(rows):
    """rows: (Dyad, Round, Player, Category, Is_there)."""
    return pd.DataFrame(rows, columns=['Dyad', 'Round', 'Player', 'Category', 'Is_there'])


class StrategyChainTests(unittest.TestCase):
    def test_focal_regions_match_the_original_definition(self):
        regions = chain.focal_regions()
        self.assertEqual(regions['LEFT'].sum(), 32)
        self.assertEqual(regions['RIGHT'].sum(), 32)
        self.assertEqual(regions['TOP'].sum(), 32)
        self.assertEqual(regions['IN'].sum(), 36)
        self.assertEqual(regions['OUT'].sum(), 28)
        np.testing.assert_array_equal(regions['IN'], ~regions['OUT'])
        np.testing.assert_array_equal(regions['LEFT'], ~regions['RIGHT'])

    def test_counts_and_normalisation(self):
        C = chain.counts([('LEFT', 'LEFT'), ('LEFT', 'RS'), ('RS', 'RS')], chain.S)
        i, j = chain.S.index('LEFT'), chain.S.index('RS')
        self.assertEqual(C[i, i], 1)
        self.assertEqual(C[i, j], 1)
        P = chain.normalize(C)
        self.assertAlmostEqual(P[i, i], 0.5)
        self.assertTrue(np.isnan(P[chain.S.index('IN')]).all())

    def test_irreducibility(self):
        self.assertTrue(chain.irreducible(chain.counts([('LEFT', 'RS'), ('RS', 'LEFT')], chain.S)))
        self.assertFalse(chain.irreducible(chain.counts([('LEFT', 'RS'), ('RS', 'RS')], chain.S)))

    def test_empirical_runs_flag_right_censoring(self):
        d = frame([('d', r, 'p', c, 'Unicorn_Absent') for r, c in
                   enumerate(['LEFT', 'LEFT', 'RS', 'TOP', 'TOP', 'TOP'], start=1)])
        runs = chain.empirical_runs(d)
        self.assertEqual(runs['LEFT']['n_runs'], 1)
        self.assertEqual(runs['LEFT']['mean_all_runs'], 2.0)
        self.assertEqual(runs['LEFT']['right_censored_share'], 0.0)
        self.assertEqual(runs['TOP']['mean_all_runs'], 3.0)
        self.assertEqual(runs['TOP']['right_censored_share'], 1.0)
        self.assertIsNone(runs['TOP']['mean_complete_runs_only'])

    def test_dyad_states_keep_or_drop_identity(self):
        d = frame([('d', 1, 'a', 'LEFT', 'Unicorn_Absent'), ('d', 1, 'b', 'RIGHT', 'Unicorn_Absent')])
        self.assertEqual(chain.dyad_states(d, ordered=True).state.iloc[0], ('LEFT', 'RIGHT'))
        d2 = frame([('d', 1, 'a', 'RIGHT', 'Unicorn_Absent'), ('d', 1, 'b', 'LEFT', 'Unicorn_Absent')])
        self.assertEqual(chain.dyad_states(d2, ordered=True).state.iloc[0], ('RIGHT', 'LEFT'))
        self.assertEqual(chain.dyad_states(d2).state.iloc[0], ('LEFT', 'RIGHT'))

    def test_role_transitions_separate_stay_swap_and_exit(self):
        rows = []
        for rnd, (ca, cb) in enumerate([('LEFT', 'RIGHT'), ('LEFT', 'RIGHT'),
                                        ('RIGHT', 'LEFT'), ('RS', 'RS')], start=1):
            rows += [('d', rnd, 'a', ca, 'Unicorn_Absent'), ('d', rnd, 'b', cb, 'Unicorn_Absent')]
        tally, by_family = chain.role_transitions(frame(rows))
        self.assertEqual(tally, {'stay': 1, 'direct_swap': 1, 'exit': 1})
        self.assertEqual(by_family['LEFT/RIGHT']['direct_swap'], 1)

    def test_sequences_split_on_gaps_and_condition(self):
        d = frame([('d', 1, 'p', 'LEFT', 'Unicorn_Absent'), ('d', 2, 'p', 'LEFT', 'Unicorn_Absent'),
                   ('d', 3, 'p', 'RS', 'Unicorn_Present'), ('d', 4, 'p', 'TOP', 'Unicorn_Absent'),
                   ('d', 5, 'p', 'TOP', 'Unicorn_Absent'), ('d', 6, 'p', 'TOP', 'Unicorn_Absent')])
        self.assertEqual([s for _, s in chain.sequences(d, 'Unicorn_Absent')], [['TOP', 'TOP', 'TOP']])
        self.assertEqual(len(chain.sequences(d, None)[0][1]), 6)

    def test_order_one_beats_order_zero_on_a_persistent_sequence(self):
        data = [(f'd{k}', ['LEFT'] * 6 + ['TOP'] * 6) for k in range(4)]
        zero = chain.order_nll(data, data, 0).nll.mean()
        one = chain.order_nll(data, data, 1).nll.mean()
        self.assertLess(one, zero)


if __name__ == '__main__':
    unittest.main()
