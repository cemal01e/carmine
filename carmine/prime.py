# -*- coding: utf-8 -*-
"""
An implementation of a novel Market Basket Analysis (MBA) algorithm
which uses prime numbers.

Author:
    Eron Cemal <eron.cemal@flightdataservices.com>, 2018
"""
import numpy as np
import pandas as pd
import sympy  #TODO: can get rid of this dependency by locally storing list of primes


class PrimeMBA(object):
    """
    This is a novel implementation of MBA using prime numbers
    """
    def __init__(self, X, y, feature_names=None):

        Z = np.hstack((X, y[:, None]))

        if feature_names is None:
            n_features = X.shape[1]
            feature_names = np.arange(0, n_features)
        feature_names.append("y")


        data = pd.DataFrame(data=Z, columns=feature_names)
        data["y"] = data["y"].astype(bool)
        self.data = data
        self.primed_data = data
        self.rule_df = None

    def _primes_and_unique_list(self):
        df = self.data.copy()
        df = df.astype(str)
        for column in df:
            df[column]= column + '=' + df[column]

        F = df.as_matrix()
        F_unique = np.unique(F)


        n = F_unique.size
        nth = sympy.prime(n)

        prime_list = [x for x in sympy.primerange(0,nth+1)]

        return prime_list, F_unique

    def _calc_prod(self, prime_list, F_unique):
        df = self.data.copy()
        df = df.astype(str)
        for column in df:
            df[column]= column + '=' + df[column]

        # TODO: Why is this the replacing so slow!!!
        df = df.replace(to_replace=F_unique,value=prime_list)
        #TODO: is it ok to define a self. something here?
        self.primed_data = df

        return df.prod(axis=1)

    def _MBA_calc(self, dataframe, prod, id_event):
        df = dataframe.copy()
        df["support"] = df["id"].apply(lambda x: np.mean(np.mod(prod, x) == 0))
        df["matches"] = (df["support"]*len(prod)).astype(int)
        df["support (X, Event)"] = df["id"].apply(lambda x: np.mean(np.mod(prod, x*id_event) == 0))
        df["confidence (X -> Event)"] = df["support (X, Event)"] / df["support"]
        return df

    def _ids_for_r2(self, r1, filter_ids=True):

        # For speed, you might want to consider only cases where an event has occured for depth 2
        ids_for_depth2 = r1[r1["confidence (X -> Event)"] > 0]["id"]

        uniqe_column_list_of_lists = []
        for col in self.primed_data.iloc[:,:-1]:  # we dont want column "y" to be added into this
            uniqe_column_list_of_lists.append(self.primed_data[col].unique())

        new_ids = []

        # for uniqe_column_list in uniqe_column_list_of_lists:
        while uniqe_column_list_of_lists:
            current_column_list = uniqe_column_list_of_lists.pop()

            for uniqe_column_list in uniqe_column_list_of_lists:
                for i in current_column_list:
                    for j in uniqe_column_list:
                        if filter_ids == True:
                            if i in ids_for_depth2 and j in ids_for_depth2:
                                new_ids.append([i,j])
                        else:
                            new_ids.append([i,j])
        return new_ids

    def train(self, depth=1, optimise_y_true=True):
        """
        Calculate the support and confidence using the novel prime number
        MBA method.

        :param depth: the maximum size of the set for which support will be calculated
        :type depth: int
        :param optimise_y_true: If true, will calcualte support for depth 2 only\
                if confidence(x)>0
        :type optimise_y_true: bool

        """
        prime_list, F_unique = self._primes_and_unique_list()

        prod = self._calc_prod(prime_list, F_unique)
        print(prod)

        r1 = pd.DataFrame(data=prime_list, columns=["id"])

        r1["rule"] = r1["id"].replace(to_replace=prime_list, value=F_unique)
        r1["depth"] = 1

        id_event = np.array(prime_list)[F_unique=="y=True"][0]

        r1 = self._MBA_calc(r1, prod, id_event)
        if depth ==1:
            self.rule_df = r1
        elif depth == 2:
            new_ids = self._ids_for_r2(r1, optimise_y_true)
            r2 = pd.DataFrame(data=new_ids)
            r2["id"] = r2.prod(axis=1)
            r2=r2.replace(to_replace=prime_list, value=F_unique)
            r2["rule"] = r2[0] + " and " +  r2[1]
            r2 = r2.drop(columns=[0, 1])
            r2["depth"] = 2
            r2 = self._MBA_calc(r2, prod, id_event)
            self.rule_df = pd.concat([r1, r2], ignore_index=True)
        else:
            # TODO: not supported!
            pass
