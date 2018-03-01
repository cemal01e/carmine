# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

from carmine.rule import Rule, RuleList


class DecisionTreeRuleExtractor(object):
    def __init__(self, X, y, feature_names=None,
                 include_negations=True, class_names=None):
        # prepare dataset
        if not feature_names:
            feature_names = [str(i) for i in range(X.shape[1])]
        self.X, self.y, fv = self._preprocess_dataset(X, y, feature_names)
        self.features_values = fv
        self.class_names = class_names
        self.include_negations = include_negations

    def __matrix_to_dict(self, X, feature_names):
        assert len(feature_names) == X.shape[1]
        for i in range(X.shape[0]):
            row = X[i, :]
            d = {feature_names[j]: str(row[j])
                 for j in range(X.shape[1])}
            yield d

    def _preprocess_dataset(self, X, y, feature_names):
        dictionaries = list(self.__matrix_to_dict(X, feature_names))
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dictionaries)

        # transform data
        y = y.ravel()  # ensure data is 1-dimensional
        features_values = [v.split("=") for v in dv.feature_names_]

        return (X, y, features_values)

    def train(self, **kwargs):
        # construct decision tree
        tree = DecisionTreeClassifier(**kwargs)
        tree.fit(self.X, self.y)

        # extract rules
        self.tree = tree.tree_
        self.total_samples = self.tree.value.max(axis=2).reshape(-1)[0]
        self.rules = self.extract(self.include_negations)

    def extract(self, include_negations=True):
        """
        Recursively extract classification rules from a decision tree.
        """
        rules = RuleList()
        classes = self.tree.value.argmax(axis=2).reshape(-1)
        samples = self.tree.value

        def __recurse(tree, node, rule=Rule()):
            # get left and right child nodes
            left = tree.children_left[node]
            right = tree.children_right[node]

            # fetch impurity, classification, and feature name
            impurity = float(tree.impurity[node])

            f_index = tree.feature[node]
            if self.features_values:
                feature = self.features_values[f_index]
            else:
                feature = (f_index, f_index)

            class_ = classes[node]
            if self.class_names:
                class_ = self.class_names[class_]

            # if child nodes exist, recursively extract information from them
            if include_negations and left >= 0:
                rule_left = rule.copy()
                rule_left.add((feature[0], Rule.NEQ, feature[1]))
                __recurse(tree, left, rule=rule_left)

            # ignore negation of leaf condition if option is set
            if right >= 0:
                rule_right = rule.copy()
                rule_right.add((feature[0], Rule.EQ, feature[1]))
                __recurse(tree, right, rule=rule_right)

            # if the current rule state has one or more conditions, add it
            if len(rule) > 0:
                rule.classification = class_
                rule.purity = (1 - impurity)
                rule.proportion = (samples[node].sum() / self.total_samples)
                rule.matches = samples[node].sum()
                rule.score = (rule.purity, rule.proportion)
                rules.add(rule)

        # start off recursion on the root node
        __recurse(self.tree, 0)

        return rules
