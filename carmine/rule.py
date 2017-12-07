# -*- coding: utf-8 -*-

from collections import defaultdict


class Rule(object):
    EQ = "is"
    NEQ = "is not"

    def __init__(self, conditions={}):
        self.conditions = defaultdict(set)
        for key, value in conditions.items():
            self.conditions[key] = set(value)

    def __len__(self):
        return len(self.conditions)

    def add(self, condition):
        """
        Add a condition to the ruleset, checking and removing redundant
        information and logical inconsistencies.
        """
        feature, relation, value = condition

        # if the relationship of this new condition is an equality
        # and the feature is already used for this rule, then the
        # new rule must (by definition) have precedence
        if relation == Rule.EQ:
            self.conditions[feature].clear()
        self.conditions[feature].add((relation, value))

    def copy(self):
        return Rule(conditions=self.conditions)


class RuleList(object):
    def __init__(self):
        self.rules = []

    def add(self, rule):
        self.rules.append(rule)

    def to_list(self, filter_func=None):
        # sort current rule state
        rules = sorted(self.rules, key=lambda x: x["score"], reverse=True)

        # filter rules according to function
        if filter_func is not None:
            rules = [r for r in rules if filter_func(r)]

        pretty_rules = []
        for rule in rules:
            # if rule doesn't match the desired criteria, skip it
            if filter_func is not None and not filter_func(rule):
                continue

            # express score as proportion of maximum
            pretty = {}
            pretty["class"] = rule["class"]
            pretty["score"] = rule["score"]

            # express rules in string representation
            conditions = []
            for key, value in rule["conditions"].conditions.items():
                for clause in value:
                    conditions.append("{k} {c} {v}".format(
                            k=key, c=clause[0], v=clause[1]))
            pretty["conditions"] = " and ".join(conditions)

            # add rule to list
            pretty_rules.append(pretty)
        return pretty_rules

    def to_html(self, filter_func=None):
        """
        Return a nice HTML representation of all rules in the rule set.

        Returns:
            html: An HTML table containing all rules.
        """
        import pandas as pd

        pretty_rules = self.to_list(filter_func=filter_func)
        df = pd.DataFrame(pretty_rules)

        return df.to_html(index=None, justify="inherit")
