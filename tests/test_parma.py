import unittest
import pandas as pd
import numpy as np

from .context import carmine
from .context import X
from .context import y, Y

from carmine.parma import Parma


class Test_Parma(unittest.TestCase):

    def test_train(self):
        m = Parma(X, y)
        m.train()
        df = m.rule_df
        expected_cols = ["support(x)",
                         "matches(x)",
                         "support(x,y)",
                         "confidence(x,y)",
                         "itemset",
                         "lift(x,y)",
                        ]
        for x in expected_cols:
            self.assertIn(x, df.columns.values.tolist())

        self.assertFalse(df.empty)

    def test_clean(self):
        m = Parma(X, y)
        m.train()
        m.clean()
        df = m.rule_df
        expected_cols = ["support(x)",
                         "matches(x)",
                         "support(x,y)",
                         "confidence(x,y)",
                         "itemset",
                         "lift(x,y)",
                         "equivalent_sets",
                        ]

        for x in expected_cols:
            self.assertIn(x, df.columns.values.tolist())

        self.assertFalse(df.empty)

    def test_clean_with_mins(self):
        m = Parma(X, y)
        m.train()
        m.clean(min_support=0.1)
        self.assertFalse(m.rule_df.empty)
        m.clean(min_confidence=0.1)
        self.assertFalse(m.rule_df.empty)
        m.clean(min_matches=2)
        self.assertFalse(m.rule_df.empty)
        self.assertGreater(m.rule_df_uncleaned.size, m.rule_df.size)

#TODO: There must be a smarter way of doing this rather than just a copy paste!!
class Test_Parma_multiple(unittest.TestCase):

    def test_train(self):
        m = Parma(X, Y)
        m.train()
        df = m.rule_df
        expected_cols = ["support(x)",
                         "matches(x)",
                         "support(x,y)",
                         "confidence(x,y)",
                         "itemset",
                         "lift(x,y)",
                        ]
        for x in expected_cols:
            self.assertIn(x, df.columns.values.tolist())

        self.assertFalse(df.empty)

    def test_clean(self):
        m = Parma(X, Y)
        m.train()
        m.clean()
        df = m.rule_df
        expected_cols = ["support(x)",
                         "matches(x)",
                         "support(x,y)",
                         "confidence(x,y)",
                         "itemset",
                         "lift(x,y)",
                         "equivalent_sets",
                        ]

        for x in expected_cols:
            self.assertIn(x, df.columns.values.tolist())

        self.assertFalse(df.empty)

    def test_clean_with_mins(self):
        m = Parma(X, Y)
        m.train()
        m.clean(min_support=0.1)
        self.assertFalse(m.rule_df.empty)
        m.clean(min_confidence=0.1)
        self.assertFalse(m.rule_df.empty)
        m.clean(min_matches=2)
        self.assertFalse(m.rule_df.empty)
        self.assertGreater(m.rule_df_uncleaned.size, m.rule_df.size)




if __name__ == "__main__":
    unittest.main()
