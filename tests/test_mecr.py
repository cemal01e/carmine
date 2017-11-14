import unittest
import numpy as np
from .context import carmine


# verification dataset from paper below
# DOI: http://dx.doi.org/10.1016/j.eswa.2012.10.035
dataset = np.array([
    [1, 1, 1, 1],
    [1, 2, 1, 0],
    [2, 2, 1, 0],
    [3, 3, 1, 1],
    [3, 1, 2, 0],
    [3, 3, 1, 1],
    [1, 3, 2, 1],
    [2, 2, 2, 0]
])

X = dataset[:, :-1]  # attributes / features
y = dataset[:, -1]  # class labels


class TestNode(unittest.TestCase):
    def test_create_match_all_node(self):
        n = carmine.mecr.Node(X, y, matches=None)

    def test_create_partial_match_node(self):
        n = carmine.mecr.Node(X, y, matches=[1])

    def test_create_child(self):
        i = carmine.mecr.Node(X, y, matches=[3, 4, 5])
        j = carmine.mecr.Node(X, y, matches=[1, 2, 3])
        child = i.create_child(j)
        self.assertIsNotNone(child)
        self.assertGreater(len(child.matches), 0)
        self.assertEqual(sorted(child.matches)[0], 3)


class TestMECRTree(unittest.TestCase):
    def test_construct_root_node(self):
        m = carmine.MECRTree(X, y)
        n = m._construct_root_node(X, y, min_support=0.1)
        self.assertIsNotNone(n)

    def test_match_all_objects_with_zero_support(self):
        m = carmine.MECRTree(X, y)
        n = m._construct_root_node(X, y, min_support=0.0)
        self.assertEqual(len(n.matches), y.size)

    def test_train_rules(self):
        feature_names = ["feature 1", "feature 2", "feature 3"]
        m = carmine.MECRTree(X, y, feature_names=feature_names)
        m.train(0.25, 0.6)
        self.assertIsNotNone(m.rules)
        self.assertGreater(len(m.rules), 0)

    def test_feature_names(self):
        feature_names = ["feature 1", "feature 2", "feature 3"]
        m = carmine.MECRTree(X, y, feature_names=feature_names)
        m.train(0.1, 0.3)
        self.assertIsNotNone(m.rules)
        self.assertGreater(len(m.rules), 0)
        for rule in m.rules:
            features = list(rule["values"].keys())
            for feature in features:
                self.assertIn("feature ", feature)

    def test_multiple_conditions_in_rule(self):
        m = carmine.MECRTree(X, y)
        m.train(0.25, 0.6)
        max_conditions = max([len(rule["values"]) for rule in m.rules])
        self.assertGreater(max_conditions, 1)

    def test_min_support(self):
        min_support = 0.25
        m = carmine.MECRTree(X, y)
        m.train(min_support, 0.6)
        self.assertIsNotNone(m.rules)
        self.assertGreater(len(m.rules), 0)
        for rule in m.rules:
            self.assertGreaterEqual(rule["support"], min_support)

    def test_min_confidence(self):
        min_confidence = 0.6
        m = carmine.MECRTree(X, y)
        m.train(0.25, min_confidence)
        self.assertIsNotNone(m.rules)
        self.assertGreater(len(m.rules), 0)
        for rule in m.rules:
            self.assertGreaterEqual(rule["confidence"], min_confidence)


if __name__ == "__main__":
    unittest.main()
