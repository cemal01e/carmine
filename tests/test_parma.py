import unittest
import pandas as pd
import numpy as np

from .context import carmine
from .context import X
from .context import y

from carmine.parma import Parma


class Test_Parma(unittest.TestCase):

    def test_train(self):
        m =Parma(X, y)
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




if __name__ == "__main__":
    unittest.main()
