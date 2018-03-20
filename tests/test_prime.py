import unittest
import pandas as pd
import numpy as np

from .context import carmine
from .context import X
from .context import y

from carmine.prime import PrimeMBA


class Test_PrimeMBA(unittest.TestCase):

    def test_replace_with_prime(self):
        m = PrimeMBA(X, y)
        df_primed = m._replace_with_prime()
        self.assertIsNotNone(df_primed)
        self.assertEqual(df_primed.shape[0], X.shape[0])
        self.assertEqual(df_primed.shape[1], X.shape[1] + 1)
        self.assertEqual(df_primed.shape, m.data.shape)
        self.assertIsInstance(df_primed, pd.DataFrame)

    def test_MBA_calc(self):
        m = PrimeMBA(X, y)
        df_primed = m._replace_with_prime()
        prod = df_primed.prod(axis=1)
        m.primed_data = df_primed
        prime_list = np.array(m.prime_dict.values())
        value_list = np.array(m.prime_dict.keys())

        r1 = pd.DataFrame(data=prime_list, columns=["id"])

        r1["rule"] = r1["id"].replace(to_replace=prime_list, value=value_list)
        r1["depth"] = 1

        id_event = prime_list[value_list=="y=True"][0]

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
        df_primed = m._replace_with_prime()
        prod = df_primed.prod(axis=1)
        m.primed_data = df_primed
        prime_list = np.array(m.prime_dict.values())
        value_list = np.array(m.prime_dict.keys())

        r1 = pd.DataFrame(data=prime_list, columns=["id"])
        r1["rule"] = r1["id"].replace(to_replace=prime_list, value=value_list)
        r1["depth"] = 1

        id_event = prime_list[value_list=="y=True"][0]
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
