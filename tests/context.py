import numpy as np
import os
import sys

up_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, up_path)

import carmine  # NOQA


# verification dataset from paper below
# DOI: http://dx.doi.org/10.1016/j.eswa.2012.10.035
dataset = np.array([
    [1, 1, 1, 1, 0],
    [1, 2, 1, 0, 0],
    [2, 2, 1, 0, 1],
    [3, 3, 1, 1, 1],
    [3, 1, 2, 0, 1],
    [3, 3, 1, 1, 0],
    [1, 3, 2, 1, 1],
    [2, 2, 2, 0, 0]
])

X = dataset[:, :-1]  # attributes / features
y = dataset[:, -2]  # class labels
Y = dataset[:, -2:]  #  second class labels
