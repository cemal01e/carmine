import unittest
import pandas as pd
import numpy as np

from .context import carmine
from .context import X
from .context import y

from carmine.prime import PrimeMBA


class Test_PrimeMBA(unittest.TestCase):

    def test_primes_and_unique_list(self):
        m = PrimeMBA(X, y)
        prime_list, F_unique = m._primes_and_unique_list()
        self.assertIsNotNone(prime_list)
        self.assertIsNotNone(prime_list)
        self.assertGreater(len(prime_list), 0)
        self.assertGreater(len(F_unique), 0)
        self.assertIsInstance(prime_list, list)
        self.assertIsInstance(F_unique, np.ndarray)

    def test_calc_prod(self):
        m = PrimeMBA(X, y)
        prime_list, F_unique = m._primes_and_unique_list()
        prod = m._calc_prod(prime_list, F_unique)
        self.assertIsInstance(prod, pd.Series)
        self.assertEqual(prod.shape[0], X.shape[0])

    def test_MBA_calc(self):
        m = PrimeMBA(X, y)
        prime_list, F_unique = m._primes_and_unique_list()
        prod = m._calc_prod(prime_list, F_unique)
        r1 = pd.DataFrame(data=prime_list, columns=["id"])
        r1["rule"] = r1["id"].replace(to_replace=prime_list,
                                      value=F_unique)
        r1["depth"] = 1
        id_event = np.array(prime_list)[F_unique=="y=True"][0]
        r1 = m._MBA_calc(r1, prod, id_event)

        self.assertIsNotNone(r1)
        self.assertIsInstance(r1, pd.DataFrame)
        expected_cols = ["support", "matches",
                         "support (X, Event)",
                         "confidence (X -> Event)"]
        for x in expected_cols:
            self.assertIn(x, r1.columns.values.tolist())

        self.assertFalse(r1.empty) # df is not empty


    def test_ids_for_r2(self):
        m = PrimeMBA(X, y)
        prime_list, F_unique = m._primes_and_unique_list()
        prod = m._calc_prod(prime_list, F_unique)
        r1 = pd.DataFrame(data=prime_list, columns=["id"])
        r1["rule"] = r1["id"].replace(to_replace=prime_list,
                                      value=F_unique)
        r1["depth"] = 1
        id_event = np.array(prime_list)[F_unique=="y=True"][0]
        r1 = m._MBA_calc(r1, prod, id_event)

        new_ids = m._ids_for_r2(r1, True)

        self.assertIsNotNone(new_ids)
        self.assertGreater(len(new_ids), 0)

    def test_train(self):
        m = PrimeMBA(X, y)
        m.train(depth=1)
        df = m.rule_df
        expected_cols = ["support", "matches",
                         "support (X, Event)",
                         "confidence (X -> Event)",
                         "rule", "id",
                        ]
        for x in expected_cols:
            self.assertIn(x, df.columns.values.tolist())

        self.assertFalse(df.empty)

    def test_train_depth_2(self):
        m = PrimeMBA(X, y)
        m.train(depth=2)
        df = m.rule_df
        expected_cols = ["support", "matches",
                         "support (X, Event)",
                         "confidence (X -> Event)",
                         "rule", "id",
                        ]
        for x in expected_cols:
            self.assertIn(x, df.columns.values.tolist())

        self.assertFalse(df.empty)



if __name__ == "__main__":
    unittest.main()
