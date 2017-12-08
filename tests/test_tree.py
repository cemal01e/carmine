import unittest

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

from .context import carmine
from .context import X
from .context import y


class TestTreeExtraction(unittest.TestCase):
    def setUp(self):
        # transform features
        self.dictionaries = [{
                "feat 0": str(row[0]),
                "feat 1": str(row[1]),
                "feat 2": str(row[2])
            } for row in [X[i, :] for i in range(0, X.shape[0])]]
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(self.dictionaries)
        self.vec = self.dv.transform(self.dictionaries)

        # create classifier
        self.cls = DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            class_weight="balanced",
            max_depth=3,
            max_leaf_nodes=10
        )

        self.features_values = [x.split("=") for x in self.dv.feature_names_]

    def test_extract_rules(self):
        # extract rules
        self.cls.fit(X=self.vec, y=y)
        dtre = carmine.tree.DecisionTreeRuleExtractor(
            self.cls.tree_,
            features_values=self.features_values
        )
        dtre.extract(include_negations=True)

    def test_that_extracted_rules_make_sense(self):
        # extract rules
        self.cls.fit(X=self.vec, y=y)
        dtre = carmine.tree.DecisionTreeRuleExtractor(
            self.cls.tree_,
            features_values=self.features_values
        )

        rules = dtre.extract(include_negations=True).to_list()

        for rule in rules:
            self.assertIn(rule["class"], [0, 1])
            self.assertGreaterEqual(rule["score"], 0)
            self.assertIn(" is ", rule["conditions"])
