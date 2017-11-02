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
import functools
import numpy as np


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
        child = None
        self_notnull = ~self.values.mask
        other_notnull = ~other.values.mask
        notnull = (self_notnull & other_notnull)
        if ~np.any(notnull):
            # if all attributes are different, then combine rules
            matches = np.intersect1d(self.values, other.values)
            child = Node(self.X, self.y, matches=matches)
            child.values.data[self_notnull] = self.values.data[self_notnull]
            child.values.data[other_notnull] = other.values.data[other_notnull]
            child.values.mask = (self_notnull | other_notnull)
        return child

    @property
    @functools.lru_cache(maxsize=1)
    def actual_occurrence(self):
        return (self.matches.size / self.n_objs)

    @property
    @functools.lru_cache(maxsize=1)
    def classification(self):
        return np.argmax(self.counts)

    @property
    @functools.lru_cache(maxsize=1)
    def confidence(self):
        return (self.support / self.n_objs)

    @property
    @functools.lru_cache(maxsize=1)
    def support(self):
        return np.max(self.counts)


class MECRTree(object):
    """
    Implementation of the MECR tree class association rule mining algorithm
    described in [this paper][1] (also mentioned in the module docstring).

    [1]: http://dx.doi.org/10.1016/j.eswa.2012.10.035
    """
    def __init__(self, X, y, feature_names=None):
        self.root = self._construct_root_node(X, y)

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = np.arange(0, self.root.n_feats)

        self.rules = None

    def _construct_root_node(self, X, y):
        """
        Generate a root node with all 1-itemsets, extracted from data.
        """
        n = Node(X, y)
        n_feats = X.shape[1]
        for feat in np.arange(0, n_feats):
            values = np.unique(X[:, feat])
            for value in values:
                matches = np.where(X[:, feat] == value)
                c = Node(X, y, matches=matches)
                c.values[feat] = value
                n.children.append(c)
        return n

    def _mine(self, node, min_support, min_confidence):
        rules = []
        for i, l_i in enumerate(node.children):
            # enumerate rules
            if l_i.confidence >= min_confidence:
                notnan = ~l_i.values.mask
                values = { self.feature_names[i]: l_i.values[i]
                            for i in range(0, len(l_i.values))
                            if notnan[i] }
                rule = {
                    "values": values,
                    "class": l_i.classification,
                    "confidence": l_i.confidence,
                    "support": l_i.support
                }
                rules.append(rule)

            for l_j in node.children[i+1:]:
                child = l_i.create_child(l_j)
                if child is not None and (child.support >= min_support):
                    l_i.children.append(child)

            self._mine(l_i, min_support, min_confidence)
        return rules

    def train(self, min_support, min_confidence):
        self.rules = self._mine(self.root, min_support, min_confidence)
