# -*- coding: utf-8 -*-

import math
from collections import defaultdict


class Rule(object):
    EQ = "is"
    NEQ = "is not"

    def __init__(self, conditions={}, ignore_full_negations=False):
        self.conditions = defaultdict(set)
        self.ignore_full_negations = ignore_full_negations
        for feat, rule_part in conditions.items():
            for rel, val in rule_part:
                self.add((feat, rel, val))

    def __len__(self):
        return len(self.conditions)

    def add(self, condition):
        """
        Add a condition to the ruleset, checking and removing redundant
        information and logical inconsistencies.
        """
        feature, relation, value = condition

        # if the relationship of this new condition is an equality and the
        # feature is already used for inequalities, then the equality
        # supercedes previous conditions, making them irrelevant
        if relation == Rule.EQ:
            self.conditions[feature].clear()
            self.conditions[feature].add((relation, value))

        if relation == Rule.NEQ and not self.ignore_full_negations:
            self.conditions[feature].add((relation, value))

    def copy(self, ignore_full_negations=False):
        r = Rule(conditions=self.conditions,
                 ignore_full_negations=ignore_full_negations)
        # copy missing properties over
        for prop in dir(self):
            try:
                r.__getattribute__(prop)
            except AttributeError:
                r.__setattr__(prop, self.__getattribute__(prop))
        return r

    def __hash__(self):
        conds = []
        for feat, rule_part in sorted(self.conditions.items()):
            for rel, val in sorted(rule_part):
                conds.append((feat, rel, val))
        return hash(tuple(conds))

    def __eq__(self, other):
        return hash(self) == hash(other)


class RuleList(object):
    def __init__(self, ignore_full_negations=False):
        self.ignore_full_negations = ignore_full_negations
        self.rules = set()

    def __iter__(self):
        for rule in self.rules:
            yield rule

    def __len__(self):
        return len(self.rules)

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(msg="{item} has no attribute \"{key}\"".format(
                    item=type(self),
                    key=key
                ))

    def add(self, rule):
        rule_without_negations = rule.copy(ignore_full_negations=True)
        if (not self.ignore_full_negations or
                (self.ignore_full_negations and
                    len(rule_without_negations) > 0)):
            self.rules.add(rule)

    def merge(self, rule_list):
        for rule in rule_list.rules:
            if rule not in self.rules:
                self.add(rule)

    def to_list(self, filter_func=None):
        # sort current rule state
        rules = sorted(self.rules, key=lambda rule: rule.score, reverse=True)

        # filter rules according to function
        if filter_func is not None:
            rules = [r for r in rules if filter_func(r)]

        pretty_rules = []
        for rule in rules:
            # if rule doesn't match the desired criteria, skip it
            if filter_func is not None and not filter_func(rule):
                continue

            # express some human-friendly quality metrics
            pretty = {}
            pretty["class"] = rule.classification
            pretty["purity"] = rule.purity
            pretty["proportion"] = rule.proportion
            pretty["matches"] = int(math.ceil(rule.matches))

            # express rules in string representation
            conditions = []
            for feat, value in rule.conditions.items():
                for rel, val in value:
                    conditions.append((feat, rel, val))

            pretty["conditions"] = " and ".join([
                "{} {} {}".format(*c)
                for c in sorted(conditions, key=lambda x: x[1])
            ])

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

        human_readable_rules = []
        for r in self.to_list(filter_func=filter_func):
            invalid = int(math.floor(r["purity"] * r["matches"]))
            hr_rule = {
                "conditions": r["conditions"],
                "total": r["matches"],
                "invalid": invalid,
                "valid": r["matches"] - invalid,
                "invalid %": int(round(r["purity"] * 100.0))
            }
            human_readable_rules.append(hr_rule)

        df = pd.DataFrame(human_readable_rules)

        if len(df) > 0:
            df = df[["conditions", "invalid %", "invalid", "valid", "total"]]

            # disable string truncation because pandas
            with pd.option_context("display.max_colwidth", -1):
                return df.to_html(
                    index=None,
                    float_format=lambda f: "{:.3f}".format(f),
                    classes=["tbl", "display"]
                )
        else:
            return None
