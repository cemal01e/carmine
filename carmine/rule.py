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


class RuleScorer(object):
    # TODO: implement metrics
    def __init__(self, metric="entropy"):
        self.metric = metric

    def score(self, args):
        return getattr(self, self.metric)(*args)

    def entropy(self):
        return -1.0

    def confidence_support(self):
        return -1.0


def get_rule_table(rules, filter_func=None):
    """
    Return a nice HTML representation of all mined association rules.

    Returns:
        html: An HTML table containing all rules.
    """
    import prettytable

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
        pretty["confidence"] = rule["confidence"]
        pretty["score"] = rule["score"]
        pretty["support"] = rule["support"]

        # express rules in string representation
        conditions = []
        for k, v in rule["conditions"].items():
            conditions.append("{k} is {v}".format(k=k, v=v))
        pretty["conditions"] = " and ".join(conditions)

        # add rule to list
        pretty_rules.append(pretty)

    html = None
    if len(pretty_rules) > 0:
        field_names = pretty_rules[0].keys()
        tbl = prettytable.PrettyTable(field_names=field_names)
        for pretty in pretty_rules:
            tbl.add_row(pretty.values())
        html = tbl.get_html_string()

    return html
