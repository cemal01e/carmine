import functools
import numpy as np

## start test dataset

dataset = np.array([
    [ 1, 1, 1, 1 ],
    [ 1, 2, 1, 0 ],
    [ 2, 2, 1, 0 ],
    [ 3, 3, 1, 1 ],
    [ 3, 1, 2, 0 ],
    [ 3, 3, 1, 1 ],
    [ 1, 3, 2, 1 ],
    [ 2, 2, 2, 0 ]
])

X = dataset[:,:-1]
y = dataset[:,-1]

### end test dataset


class Node(object):
    def __init__(self, X, y, matches=None):
        # compute number of objects and features in dataset
        self.n_objs = X.shape[0]
        self.n_feats = X.shape[1]

        # compute matching object indices and values
        if matches is None:
            self.matches = np.arange(0, self.n_objs)
        else:
            self.matches = matches

        self.values = np.full(
            shape=self.n_feats,
            fill_value=np.nan,
            dtype="int"
        )

        # compute classes and counts
        _, counts = np.unique(y[self.matches], return_counts=True)
        self.n_classes = np.unique(y).size
        self.counts = counts

        # children for tree node
        self.children = []

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


def carmine(node, min_support, min_confidence):
    rules = []
    for i, l_i in enumerate(node.children):
        # enumerate rules
        if l_i.confidence >= min_confidence:
            rule = {
                "values": l_i.values,
                "class": l_i.classification,
                "confidence": l_i.confidence,
                "support": l_i.support
            }
            rules.append(rule)

        for l_j in node.children[i+1:]:
            l_i_nn = ~np.isnan(l_i.values)
            l_j_nn = ~np.isnan(l_j.values)
            nn = (l_i_nn & l_j_nn)
            if ~np.any(nn):
                # if different attributes, combine rules
                matches = np.intersect1d(l_i.values, l_j.values)
                o = Node(X, y, matches=matches)
                o.values[l_i_nn] = l_i[l_i_nn]
                o.values[l_j_nn] = l_j[l_j_nn]

                if o.support >= min_support:
                    l_i.children.append(o)

        carmine(l_i, min_support, min_confidence)
    return rules


def construct_root_node(X, y):
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

n = construct_root_node(X, y)
rules = carmine(n, 1, 0.3)
