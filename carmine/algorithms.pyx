# -*- coding: utf-8 -*-
# cython: language_level=3

cimport cython
cimport numpy as np
import numpy as np


@cython.boundscheck(False)
@cython.infer_types(False)
@cython.initializedcheck(False)
def unique1d(np.ndarray[np.int32_t, ndim=1] arr):
    """
    Extremely naive, high-performance unique value counter.
    """
    cdef np.ndarray[np.int32_t, ndim=1] ht = \
            np.zeros((arr.shape[0]), dtype=np.int32)

    cdef np.int32_t maxidx = 0
    for i in range(arr.shape[0]):
        ht[arr[i]] += 1
        if ht[arr[i]] == 1:
            maxidx += 1

    cdef np.ndarray[np.int32_t, ndim=1] counts = \
        np.zeros((maxidx), dtype=np.int32)

    for i in range(maxidx):
        counts[i] = ht[i]

    return (maxidx, counts)
