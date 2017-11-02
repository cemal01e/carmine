import unittest
import numpy as np
from .context import carmine


# verification dataset from paper below
# DOI: http://dx.doi.org/10.1016/j.eswa.2012.10.035
dataset = np.array([
    [ 1, 1, 1, 1 ],
    [ 1, 2, 1, 0 ],
    [ 2, 2, 1, 0 ],
    [ 3, 3, 1, 1 ],
    [ 3, 1, 2, 0 ],
    [ 3, 3, 1, 1 ],
    [ 1, 3, 2, 1 ],
    [ 2, 2, 2, 0 ]
])

X = dataset[:,:-1]  # attributes / features
y = dataset[:,-1]  # class labels


class TestNode(unittest.TestCase):
    def test_create_match_all_node(self):
        n = carmine.Node(X, y, matches=None)

    def test_create_partial_match_node(self):
        n = carmine.Node(X, y, matches=[1])

    def test_create_child(self):
        pass


class TestMECRTree(unittest.TestCase):
    def test_construct_root_node(self):
        m = carmine.MECRTree(X, y)
        self.assertIsNotNone(m.root)

    def test_match_all_objects(self):
        m = carmine.MECRTree(X, y)
        self.assertEqual(m.root.matches.size, y.size)

    def test_feature_names(self):
        feature_names = ["feature 1", "feature 2", "feature 3"]
        m = carmine.MECRTree(X, y, feature_names=feature_names)
        m.train(1, 0.3)
        self.assertIsNotNone(m.rules)
        for rule in m.rules:
            features = list(rule["values"].keys())
            for feature in features:
                self.assertIn("feature ", feature)

    def test_train_rules(self):
        m = carmine.MECRTree(X, y)
        m.train(2, 0.3)  # TODO: replace this with supp/conf in paper
        self.assertIsNotNone(m.rules)


if __name__ == "__main__":
    unittest.main()
