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

Currently there's three different algorithms implemented:
- Parma (recommended)
- MECR Tree
- PrimeMBA


### Parma (recommended):

Parma stands for Pandas Association Rule Mining Algorithm. It relies on Pandas'
groupby() functions mean and count aggregations. These metrics are then converted
into the familiar Market Basket Analysis(MBA) terms such as support, confidence,
and lift. Out of all the other algorithms in this package, we have found this to
be the most efficient. However, it is unclear if with really large data sizes
this could change.


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

### PrimeMBA

PrimeMBA converts maps each tpye of item in the dataset to a prime number.
It then takes a product of all itemsets. If we want to know if item $$x_{i}$$
exists in the dataset, we simply divide this product by the unique identifier of that item
$$p_i{}$$. This algorthim only calculates association rules up to length two example: 
(bread, milk)-> (eggs)
