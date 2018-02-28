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
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from carmine.rule import Rule, RuleList


class CategoricalDataTransformer(object):
    def __init__(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.n_objs = self.X.shape[0]
        self.n_features = self.X.shape[1]
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
        self.n_objs = float(X.shape[0])
        self.n_feats = X.shape[1]

        # compute matching object indices and values
        if matches is None:
            self.matches = set(np.arange(0, self.n_objs, dtype="int32"))
        else:
            self.matches = set(matches)

        # create masked array to hold class values
        self.values = np.ma.zeros(self.n_feats, dtype="int32")
        self.values.mask = True

        # compute classes and counts
        s = pd.value_counts(
            y[list(self.matches)],
            sort=False
        ).sort_index()
        self.n_classes, self.counts = s.index.size, s.values

        # children for tree node
        self.children = []

    def create_child(self, other):
        def _make_child(n, n_nn, o, o_nn):
            # make sure child doesn't just match the same objects as parent
            matches = n.matches & o.matches
            matches_parents = (len(matches) == len(self.matches) or
                               len(matches) == len(other.matches))
            if len(matches) > 0 and not matches_parents:
                c = Node(n.X, n.y, matches=matches)
                c.values.data[n_nn] = np.compress(n_nn, n.values.data)
                c.values.data[o_nn] = np.compress(o_nn, o.values.data)
                c.values.mask[(n_nn | o_nn)] = False
                return c

        self_nn = ~self.values.mask
        other_nn = ~other.values.mask
        shared = (self_nn & other_nn)

        # if parents don't share attributes, combine rules
        no_shared_attrs = ~np.any(shared)
        if no_shared_attrs:
            return _make_child(self, self_nn, other, other_nn)

        # if parents share attributes but have the same values, combine
        attrs_same_vals = np.any(np.compress(shared, self.values) ==
                                 np.compress(shared, other.values))
        if attrs_same_vals:
            return _make_child(self, self_nn, other, other_nn)

        return None

    @property
    def classification(self):
        return np.argmax(self.counts)

    @property
    def confidence(self):
        return np.max(self.counts) / self.actual_occurrence

    @property
    def actual_occurrence(self):
        return float(len(self.matches))

    @property
    def support(self):
        print(self.counts)
        print(self.n_objs)
        return np.max(self.counts) / self.n_objs


class MECRTree(object):
    """
    Implementation of the MECR tree class association rule mining algorithm
    described in [this paper][1] (also mentioned in the module docstring).

    [1]: http://dx.doi.org/10.1016/j.eswa.2012.10.035
    """
    def __init__(self, X, y, feature_names=None, class_names=None):
        self.transformer = CategoricalDataTransformer(X, y)
        self.X = self.transformer.encode()
        self.y = y

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            n_features = self.X.shape[1]
            self.feature_names = np.arange(0, n_features)

        if class_names is not None:
            self.class_names = class_names
        else:
            n_classes = np.unique(y).size
            self.class_names = np.arange(0, n_classes)

        self.rules = None

    def _construct_root_node(self, X, y, min_support):
        """
        Generate a root node with all 1-itemsets, extracted from data.
        """
        n = Node(X, y)
        n_feats = X.shape[1]
        for feat in np.arange(0, n_feats):
            values = pd.unique(X[:, feat])
            for value in values:
                matches = np.nonzero(X[:, feat] == value)[0]
                c = Node(X, y, matches=matches)
                c.values[feat] = value
                if c.support >= min_support:
                    n.children.append(c)

        print(n.children)
        return n

    def _create_rule(self, values, classification, confidence, support):
        rule = Rule()
        notnan = ~values.mask
        for i in range(0, len(values)):
            if notnan[i]:
                clause = (
                    str(self.feature_names[i]),
                    Rule.EQ,
                    self.transformer.decode(i, values[i])
                )
                rule.add(clause)
        rule.classification = self.class_names[classification]
        rule.purity = confidence
        rule.proportion = support
        rule.matches = confidence * support * self.X.shape[0]
        rule.score = (rule.purity, rule.proportion)

        return rule

    def _mine(self, root, min_support, min_confidence):
        rules = RuleList()
        queue = [root]
        while len(queue) > 0:
            node = queue.pop()
            for i, l_i in enumerate(node.children):
                # enumerate rules
                if (len(l_i.children) == 0 and
                        (l_i.confidence >= min_confidence)):

                    rules.add(
                        self._create_rule(
                            l_i.values,
                            l_i.classification,
                            l_i.confidence,
                            l_i.support
                        )
                    )

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
