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
        self.rule_df_uncleaned = None
        self.data_cleaned = False

    def train(self):
        """
        Calculate the matches(x), support(x), support(x,y), confidence(x,y), and lift(x,y).
        Calculated rules will be stored in rule_df.
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
        self.rule_df_uncleaned = result.copy()

    def clean(self, min_matches=None, min_support=None, min_confidence=None):
        """
        Clean rules_df. Remove equivalent rules, and create an equivalent rules column
        where we will store the equivalent rules. If cleaning has been done, the data_cleaned
        property will state True.
        """

        if self.rule_df_uncleaned is None or self.rule_df_uncleaned.empty:
            print('There are no rules. Have you trained before you clean the rules?')
            self.data_cleaned = True

        df = self.rule_df_uncleaned.copy()

        # filter based on support and confidence
        if min_support:
            df = df.loc[df['support(x)'] >= min_support, :].reset_index(drop=True)
        if min_confidence:
            df = df.loc[df['confidence(x,y)'] >= min_confidence, :].reset_index(drop=True)
        if min_matches:
            df = df.loc[df['matches(x)'] >= min_matches, :].reset_index(drop=True)

        df['sets']=df['itemset'].apply(lambda x: set(x.replace(')(','),(').split(',')))
        df['sets'].iloc[0].issubset(df['sets'].iloc[2])
        df = df.reset_index()

        df['hash'] = df['support(x)'].astype(str) + ',' + df['confidence(x,y)'].astype(str)
        df['equivalent_index'] = None
        df['equivalent_sets'] = None


        for i in df['hash'].unique():
            b = df.loc[df['hash']==i,['index','sets']]
            b.loc[:,'equivalent_index'] = b.loc[:,'sets'].apply(lambda x: b.loc[b.loc[:,'sets']>x,'index'].values.tolist())
            b.loc[:,'equivalent_sets'] = b.loc[:,'sets'].apply(lambda x: b.loc[b.loc[:,'sets']>x,'sets'].values.tolist() )
            df.loc[df['hash']==i,['equivalent_index', 'equivalent_sets']] = b

        index_to_delete = np.unique(np.hstack(df['equivalent_index'].values))
        before_length = df.shape[0]
        df = df.drop(index_to_delete)
        df = df.drop(columns=['index', 'equivalent_index','sets','hash'])
        print('Rules have been reduced from {} rows to {} rows'.format(before_length, df.shape[0]))
        df = df.reset_index(drop=True)
        self.rule_df = df.copy()
        self.data_cleaned = True

