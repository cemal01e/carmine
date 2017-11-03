# -*- coding: utf-8 -*-
"""
Implementation of the "Modified Equivalence Class Rule" algorithm for
mining class association rules. This is a technique for discovering
classification rules for categorical data derived from the simple counting
approaches in Market Basket Analysis.

Recommended reading:
    1.) "A Novel Classification Algorithm Based on Association Rule Mining"
        Vo, Le. (Pacific Rim Knowledge Acquisition Workshop, 2008).
        DOI: http://dx.doi.org/10.1007/978-3-642-01715-5_6

    2.) "An Efficient Algorithm for Mining Class-Association Rules"
        Nguyen, Vo, Hong, Thanh. (Expert Systems with Applications, 2013).
        DOI: http://dx.doi.org/10.1016/j.eswa.2012.10.035

Author:
    Charles Newey <charlie.newey@flightdataservices.com>, 2017
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder


class CategoricalDataTransformer(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.n_objs = X.shape[0]
        self.n_features = X.shape[1]
        self.encoders = [LabelEncoder() for i in range(0, self.n_features)]

    def encode(self):
        X = self.X
        for i in range(0, self.n_features):
            X[:, i] = self.encoders[i].fit_transform(self.X[:, i])
        return X

    def decode(self, feature_index, feature_values):
        return self.encoders[feature_index].inverse_transform(feature_values)


class Node(object):
    """
    Tree node class. Implements confidence, support, actual occurrence, and
    classification metrics (used for constructing ECR/MECR trees).

    Args:
        X (:obj:`numpy.array`): An array containing a categorical dataset.
        y (:obj:`numpy.array`): An array containing class labels.
        matches (:obj:`numpy.array`): A set of matching objects (default: None)

    Attributes:
        n_objs (int): The number of objects in the dataset.
        n_feats (int): The number of features in the dataset.
        matches (:obj:`numpy.array`): A set of matching objects ("obidset").
        values (:obj:`numpy.array`):
        children (:obj:`list`): A list of this node's children (n+1-itemsets).
    """
    def __init__(self, X, y, matches=None):
        # store references to dataset
        self.X = X
        self.y = y

        # compute number of objects and features in dataset
        self.n_objs = X.shape[0]
        self.n_feats = X.shape[1]

        # compute matching object indices and values
        if matches is None:
            self.matches = np.arange(0, self.n_objs)
        else:
            self.matches = matches

        # create masked array to hold class values
        self.values = np.ma.zeros(shape=self.n_feats, dtype="int")
        self.values.mask = True

        # compute classes and counts
        _, counts = np.unique(y[self.matches], return_counts=True)
        self.n_classes = np.unique(y).size
        self.counts = counts

        # children for tree node
        self.children = []

    def create_child(self, other):
        self_nn = ~self.values.mask
        other_nn = ~other.values.mask
        shared = (self_nn & other_nn)

        # if parents don't share attributes, combine rules
        no_shared_attrs = ~np.any(shared)
        # if parents share attributes but have the same values, combine
        attrs_same_vals = np.any(self.values[shared] == other.values[shared])
        if no_shared_attrs or attrs_same_vals:
            matches = np.intersect1d(self.matches, other.matches)
            # make sure child doesn't just match the same objects as parent
            matches_parents = (len(matches) == len(self.matches) or
                len(matches) == len(other.matches))
            if matches.size > 0 and not matches_parents:
                child = Node(self.X, self.y, matches=matches)
                child.values.data[self_nn] = self.values.data[self_nn]
                child.values.data[other_nn] = other.values.data[other_nn]
                child.values.mask[(self_nn | other_nn)] = False
                return child
        return None

    @property
    def classification(self):
        return np.argmax(self.counts)

    @property
    def confidence(self):
        return np.max(self.counts) / self.actual_occurrence

    @property
    def actual_occurrence(self):
        return self.matches.size

    @property
    def support(self):
        return np.max(self.counts) / self.n_objs


class MECRTree(object):
    """
    Implementation of the MECR tree class association rule mining algorithm
    described in [this paper][1] (also mentioned in the module docstring).

    [1]: http://dx.doi.org/10.1016/j.eswa.2012.10.035
    """
    def __init__(self, X, y, feature_names=None):
        self.transformer = CategoricalDataTransformer(X, y)
        self.X = self.transformer.encode()
        self.y = y
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            n_features = self.X.shape[1]
            self.feature_names = np.arange(0, n_features)

        self.rules = None

    def _construct_root_node(self, X, y, min_support):
        """
        Generate a root node with all 1-itemsets, extracted from data.
        """
        n = Node(X, y)
        n_feats = X.shape[1]
        for feat in np.arange(0, n_feats):
            values = np.unique(X[:, feat])
            for value in values:
                matches = np.nonzero(X[:, feat] == value)[0]
                c = Node(X, y, matches=matches)
                c.values[feat] = value
                if c.support >= min_support:
                    n.children.append(c)
        return n

    def _mine(self, root, min_support, min_confidence):
        rules = []

        queue = [root]
        while len(queue) > 0:
            node = queue.pop()
            for i, l_i in enumerate(node.children):
                # enumerate rules
                if (len(l_i.children) == 0 and
                        (l_i.confidence >= min_confidence)):
                    notnan = ~l_i.values.mask
                    rule = {
                        "values": {
                            str(self.feature_names[i]):
                            self.transformer.decode(i, l_i.values[i])
                            for i in range(0, len(l_i.values)) if notnan[i]
                        },
                        "class": l_i.classification,
                        "confidence": l_i.confidence,
                        "support": l_i.support
                    }
                    rules.append(rule)

                for l_j in node.children[i+1:]:
                    child = l_i.create_child(l_j)
                    if child is not None and child.support >= min_support:
                        l_i.children.append(child)
                queue.append(l_i)
        return rules

    def train(self, min_support, min_confidence):
        """
        Train the MECR tree by mining and filtering rules according to minimum
        support and confidence criteria.

        For a gentle introduction to market basket analysis concepts (including
        mathematical definitions of support and confidence, etc), refer to
        [this post on Medium][1].

        [1]: https://goo.gl/n3VzB7

        Arguments:
            min_support (float): Minimum support for rules.
            min_confidence (float): Minimum confidence for rules.
        """
        self.root = self._construct_root_node(self.X, self.y, min_support)
        self.rules = self._mine(self.root, min_support, min_confidence)

    def get_classification_rules(self, target_class):
        """
        Return classification rules for a particular target class.

        Arguments:
            target_class: The target class to extract rules for.
        Returns:
            list: A list of rules with class labels matching target_class.
        """
        return [rule for rule in self.rules if rule["class"] == target_class]
