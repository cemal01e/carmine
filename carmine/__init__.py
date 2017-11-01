import numpy as np


class Node(object):
    """
    Implementation for the "Equivalence Class Rule" algorithm for mining class
    association rules. This is a technique for discovering classification rules
    for categorical data, derived from the simple counting approaches in Market
    Basket Analysis.

    Recommended reading:
        1.) "A Novel Classification Algorithm Based on Association Rule Mining"
            Vo, Le. (Pacific Rim Knowledge Acquisition Workshop, 2008).
            DOI: http://dx.doi.org/10.1007/978-3-642-01715-5_6
        2.) "An efficient algorithm for mining class-association rules"
            Nguyen, Vo, Hong, Thanh. (Expert Systems with Applications, 2013).
            DOI: http://dx.doi.org/10.1016/j.eswa.2012.10.035

    Args:
        n_classes (int): The number of classes to distinguish between.

    Attributes:
        attrs (:obj:`numpy.array`): Array showing which features are used.
        values (:obj:`numpy.array`): ???
        counts (:obj:`numpy.array`): A count of objects which match each class.
        matches (:obj:`numpy.array`): A set of object IDs ("obidset")
        children (:obj:`list`): A list of this node's children in the tree.
    """
    def __init__(self, n_classes=2):
        self.attrs = np.zeros(shape=n_classes, dtype="bool")
        self.values = np.zeros(shape=n_classes, dtype="int")
        self.counts = np.zeros(shape=n_classes, dtype="int")
        self.matches = np.zeros(shape=n_classes, dtype="int")
        self.children = []

    @property
    def classification(self):  # "pos" in paper
        return np.argmax(self.counts)

    @property
    def support(self):
        return np.max(self.counts)


def carmine(root, min_support, min_confidence):
    rules = set()
    for i, l_i in enumerate(root.children):
        p_i = set()
        for j, l_j in enumerate(root.children[i + 1:]):
            if not np.array_equal(l_i.attrs, l_j.attrs):
                o = Node(n_classes=root.n_classes)
                o.values = np.union1d(l_i.values, l_j.values)
                o.matches = np.intersect1d(l_i.matches, l_j.matches)

                if o.matches.size == l_i.matches.size:
                    o.counts = l_i.counts
                elif o.matches.size == l_j.matches.size:
                    o.counts = l_j.counts
                else:
                    # TODO: count implementation
                    o.counts = None

                if o.classification >= min_support:
                    p_i.add(o)

    return rules
