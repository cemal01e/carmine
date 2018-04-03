# -*- coding: utf-8 -*-
"""
PARMA

Panda Association Rule Mining Algorithm

An implementation of association rule mining
relying on pandas group by functions.

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
    This is a novel implementation of MBA using pandas groupby
    """
    def __init__(self, X, Y, feature_names=None, y_col_names=None):

        # TODO: Instead of giving X and Y, it should be possible to just
        # supply data and the column names. Currently, I usualy start from
        # data, but break it down to X and Y, then rejoin it in __init__

        # if pd.series or np.array, convert to list
        if feature_names is not None and not isinstance(feature_names, list):
            feature_names = feature_names.tolist()
        if y_col_names is not None and not isinstance(y_col_names, list):
            y_col_names = y_col_names.tolist()

        #convert Y to matrix if it was only an array
        if len(Y.shape) == 1:
            Y = Y[:, None]

        Z = np.hstack((X, Y.astype(float)))


        if feature_names is None:
            feature_names = np.arange(0, X.shape[1]).astype(str).tolist()
        if y_col_names is None:
            y_col_names = np.arange(0, Y.shape[1]).astype(str).tolist()
            y_col_names = ['y='+ x for x in y_col_names]


        data = pd.DataFrame(data=Z, columns=feature_names + y_col_names)

        # TODO: Do we want to force it that y is bool? Assert it maybe?
        data.loc[:,y_col_names]=(data.loc[:, y_col_names]>0)
        data[y_col_names]= data[y_col_names].apply(pd.to_numeric)

        self.data = data
        self.x_cols = feature_names
        self.y_cols = y_col_names

        self.rule_df = None
        self.rule_df_uncleaned = None
        self.data_cleaned = False

    def train(self, min_matches=None, min_support=None, min_confidence=None):
        """
        Calculate the matches(x), support(x), support(x,y), confidence(x,y), and lift(x,y).
        Calculated rules will be stored in rule_df.
        """
        df = self.data.copy()
        # calculate column combinations
        column_combinations = []
        for i in range(1, len(self.x_cols)):
            for j in itertools.combinations(self.x_cols, i):
                column_combinations.append(j)


        # caclute #matches and confidence for each column combination
        result = pd.DataFrame()

        for column_combination in column_combinations:
            aggregated_df = df.groupby(by=column_combination)[self.y_cols]\
                    .agg(["count", "mean"]).reset_index()
            association_rules_df = aggregated_df.iloc[:, :len(column_combination)]
            association_rules_df = association_rules_df.astype(str)
            for col in association_rules_df:
                _col = col[0] if isinstance(col, tuple) else col
                #TODO: change to association_rules_df.apply(lambda x:'(%s = %s)' % (_col, x[col]),axis=1)
                association_rules_df[col] = "(" + _col + " = " + association_rules_df[col] + ")"
            subresult = pd.concat([association_rules_df.sum(axis=1),
                                   aggregated_df.iloc[:, len(column_combination):]], axis=1)
            result = pd.concat([result, subresult], axis=0, ignore_index=True)

        #tidy result + add support and lift
        result = result.rename(columns={0:("itemset", '')})

        #TODO: apply min confidence limits at this point as at the moment, melt and 
        # rejoin takes a while
        #TODO: optimise melt and rejoin!
        melt = result.melt(id_vars='itemset')
        rc = melt.loc[melt['variable_1']=='count',['itemset','variable_0','value']]
        rc = rc.rename(columns={'value':'counts', 'variable_0':'y'})
        rm = melt.loc[melt['variable_1']=='mean',['itemset','variable_0','value']]
        rm = rm.rename(columns={'value':'mean', 'variable_0':'y'})
        rr = pd.merge(rc,rm,on=['itemset','y'])

        rr = rr.rename(columns={'counts':'matches(x)', 'mean':"confidence(x,y)",})

        #filter before we do any transformations
        if min_confidence:
            rr = rr.loc[rr['confidence(x,y)'] >= min_confidence, :].reset_index(drop=True)
        if min_matches:
            rr = rr.loc[rr['matches(x)'] >= min_matches, :].reset_index(drop=True)

        rr['support(x)'] = rr['matches(x)']/df.shape[0]

        #filter again
        if min_support:
            rr = rr.loc[rr['support(x)'] >= min_support, :].reset_index(drop=True)

        rr["support(x,y)"] = rr["confidence(x,y)"] * rr["support(x)"]
        rr["lift(x,y)"] = rr["confidence(x,y)"]

        #TODO: this bit takes the longest!!!!
        for y_val in rr['y'].unique():
            rr.loc[rr['y'] == y_val, "lift(x,y)"] /= df[y_val].mean()

        #change order of columns
        result = result.loc[:, ["itemset", 'y', "matches(x)", "support(x)",
                                "support(x,y)", "confidence(x,y)", "lift(x,y)"]]
        self.rule_df = rr
        self.rule_df_uncleaned = rr.copy()

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

        df['sets'] = df['itemset'].apply(lambda x: set(x.replace(')(','),(').split(',')))
        df['sets'].iloc[0].issubset(df['sets'].iloc[2])
        df = df.reset_index()
        df['hash'] = df.apply(lambda x:'%s, %s, %s' % (x['y'], x['support(x)'], x['confidence(x,y)']), axis=1)
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
