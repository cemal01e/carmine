# -*- coding: utf-8 -*-
"""
PARMA

Panda Association Rule Mining Algorithm

An implementation of association rule mining
relying on pandas group by functions

Author:
    Eron Cemal <eron.cemal@flightdataservices.com>, 2018
"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
import itertools

import numpy as np
import pandas as pd


class Parma(object):
    """
    This is a novel implementation of MBA using prime numbers
    """
    def __init__(self, X, y, feature_names=None):

        Z = np.hstack((X, y[:, None]))

        if feature_names is None:
            n_features = X.shape[1]
            feature_names = np.arange(0, n_features)\
                    .astype(str).tolist()
        feature_names.append("y")


        data = pd.DataFrame(data=Z, columns=feature_names)
        # TODO: Do we want to force it that y is bool? Assert it maybe?
        data["y"] = data["y"].astype(bool)
        self.data = data

        self.rule_df = None

    def train(self):
        """
        Calculate the matches(x), support(x), support(x,y), confidence(x,y), and lift(x,y)

        :returns: returns a dataframe with association rules and their relevant metrics
        :rtype: pd.DataFrame
        """
        df = self.data.copy()
        result = pd.DataFrame(columns=["count", "mean"])
        # calculate column combinations
        column_combinations = []
        for i in range(1, len(df.columns[:-1])):
            for j in itertools.combinations(df.columns[:-1], i):
                column_combinations.append(j)

        # caclute #matches and confidence for each column combination
        for column_combination in column_combinations:
            aggregated_df = df.groupby(by=column_combination)["y"]\
                    .agg(["count", "mean"]).reset_index()
            association_rules_df = aggregated_df.iloc[:, :-2]
            association_rules_df = association_rules_df.astype(str)
            for col in association_rules_df:
                association_rules_df[col] = "(" + col + " = " + association_rules_df[col] + ")"
            subresult = pd.concat([association_rules_df.sum(axis=1),
                                   aggregated_df.iloc[:, -2:]], axis=1)
            result = pd.concat([result, subresult], axis=0, ignore_index=True)


        #tidy result + add support and lift
        result = result.rename(columns={0:"itemset",
                                        "count":"matches(x)",
                                        "mean":"confidence(x,y)"})
        result["support(x)"] = result["matches(x)"] / df.shape[0]
        result["support(x,y)"] = result["confidence(x,y)"] * result["support(x)"]
        result["lift(x,y)"] = result["confidence(x,y)"]/df["y"].mean()
        #change order of columns
        result = result.loc[:, ["itemset", "matches(x)", "support(x)",
                                "support(x,y)", "confidence(x,y)", "lift(x,y)"]]
        self.rule_df = result

