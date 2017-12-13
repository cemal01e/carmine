import unittest

from .context import carmine
from .context import X
from .context import y


class TestTreeExtraction(unittest.TestCase):
    def setUp(self):
        self.cls = carmine.DecisionTreeRuleExtractor(X, y)
        self.cls.train()

    def test_extract_rules(self):
        # extract rules
        self.assertGreaterEqual(len(self.cls.rules), 0)

    def test_that_extracted_rules_make_sense(self):
        # extract rules
        rules = self.cls.rules
        for rule in rules.to_list():
            self.assertIn(rule["class"], [0, 1])
            self.assertGreaterEqual(rule["score"], 0)
            self.assertIn(" is ", rule["conditions"])
