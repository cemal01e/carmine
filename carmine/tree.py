# -*- coding: utf-8 -*-

from carmine.rule import Rule
from carmine.rule import RuleList


class DecisionTreeRuleExtractor(object):
    def __init__(self, tree, features_values=None, class_names=None):
        self.total_samples = tree.value.max(axis=2).reshape(-1)[0]
        self.features_values = features_values
        self.class_names = class_names
        self.tree = tree
        self.rules = self.extract()

    def _score_rule(self, impurity, num_samples):
        """
        Returns:
            :int: (node purity) * (fraction of dataset covered)
        """
        return (1 - impurity) * (num_samples / self.total_samples)

    def extract(self):
        """
        Recursively extract classification rules from a decision tree.
        """
        rules = RuleList()
        classes = self.tree.value.argmax(axis=2).reshape(-1)
        samples = self.tree.value.max(axis=2).reshape(-1)

        def __recurse(tree, node, rule_state=Rule()):
            # get left and right child nodes
            left = tree.children_left[node]
            right = tree.children_right[node]

            # fetch impurity and feature name
            impurity = float(tree.impurity[node])
            feature = tree.feature[node]
            if self.features_values:
                feature = self.features_values[feature]
            class_ = classes[node]
            if self.class_names:
                class_ = self.class_names[class_]

            # if child nodes exist, recursively extract information from them
            if left >= 0:
                rule_left = rule_state.copy()
                rule_left.add((feature[0], Rule.NEQ, feature[1]))
                __recurse(tree, left, rule_state=rule_left)

            if right >= 0:
                rule_right = rule_state.copy()
                rule_right.add((feature[0], Rule.EQ, feature[1]))
                __recurse(tree, right, rule_state=rule_right)

            # if the current rule state has one or more conditions, add it
            if len(rule_state) > 0:
                rule = {
                    "class": class_,
                    "conditions": rule_state,
                    "score": self._score_rule(impurity, samples[node])
                }
                rules.add(rule)

        # start off recursion on the root node
        __recurse(self.tree, 0)

        return rules
