# -*- coding: utf-8 -*-
"""
Carmine is a Class Association Rule discovery framework, aiming to implement
various fast mining algorithms in Python.
"""

import os
from codecs import open
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as reqs:
    install_requires = [l.rstrip() for l in reqs.readlines()]


setup(
    name='carmine',
    version='0.1.1',
    description='Association rule mining algorithms for Numpy and Pandas',
    long_description=long_description,
    url='https://github.com/pypa/sampleproject',
    author='Charles Newey',
    author_email='charlie.newey@flightdataservices.com',
    license='BSD 3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='machine learning association rule mining market basket analysis',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=install_requires,
)
