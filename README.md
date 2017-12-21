# Carmine

[![Build Status](https://travis-ci.org/FlightDataServices/carmine.svg?branch=master)][1]

Carmine is a Class Association Rule discovery framework, aiming to implement
various fast mining algorithms in Python.

[1]: https://travis-ci.org/FlightDataServices/carmine


## Background

Class association rule (CAR) mining is a branch of data mining that is derived
from some of the techniques used in Market Basket Analysis and the development
of recommendation engines. These techniques are designed to work on categorical
data (i.e. with items in a shopping basket), and they view the problem of CAR
mining as essentially a task in combinatorial optimisation --- effectively
enumerating all possible combinations of each category within each feature and
thresholding them to ensure that they meet certain quality criteria (namely,
"support", "confidence", "lift", and others --- see [this Wikipedia
article][2]).

[2]: https://en.wikipedia.org/wiki/Association_rule_learning#Useful_Concepts


## Algorithms

Currently there's only one algorithm implemented - the MECR tree.


### MECR Tree

This is based on two papers (and these are recommended reading for
understanding these algorithms). Essentially this algorithm builds a tree of
categorical conditions by joining leaf nodes in a tree structure. It's not the
quickest algorithm and definitely needs optimisation, but seems to create
high-quality decision rules.

1.) "A Novel Classification Algorithm Based on Association Rule Mining"
    Vo, Le. (Pacific Rim Knowledge Acquisition Workshop, 2008).
    DOI: http://dx.doi.org/10.1007/978-3-642-01715-5_6

2.) "An Efficient Algorithm for Mining Class-Association Rules"
    Nguyen, Vo, Hong, Thanh. (Expert Systems with Applications, 2013).
    DOI: http://dx.doi.org/10.1016/j.eswa.2012.10.035
