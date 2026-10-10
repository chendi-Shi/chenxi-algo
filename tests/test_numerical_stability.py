"""Financial-unit changes must not amplify machine roundoff into rank changes."""
import copy
import unittest

import engine
from validation.enterprise_checks import AS_OF, MONEY, universe


class RankingPrecisionTests(unittest.TestCase):
    def test_machine_roundoff_remains_a_tie(self):
        self.assertEqual(engine.percentile(1.2, [1.2, 1.2 + 2e-16, 1.2 - 2e-16]), 50)
        self.assertEqual(engine.percentile(1.2 + 1e-6, [1.2, 1.2 + 1e-6, 1.2 - 1e-6]), 100)

    def test_yuan_and_ten_thousand_yuan_keep_scores_and_statuses(self):
        rows, values = universe(200, 1701)
        before = engine.screen(rows, values, AS_OF)['companies']
        converted, converted_values = copy.deepcopy(rows), copy.deepcopy(values)
        for row in converted:
            row['unit_scale'] = 10000
            for field in MONEY:
                row[field] /= 10000
        for value in converted_values:
            value['unit_scale'] = 10000
            value['market_cap'] /= 10000
        after = engine.screen(converted, converted_values, AS_OF)['companies']
        self.assertEqual([(r['ticker'], r['score'], r['status']) for r in before],
                         [(r['ticker'], r['score'], r['status']) for r in after])


if __name__ == '__main__':
    unittest.main()
