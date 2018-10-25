niftidataset
=======================

[![Build Status](https://travis-ci.org/jcreinhold/niftidataset.svg?branch=master)](https://travis-ci.org/jcreinhold/niftidataset)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/niftidataset/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/niftidataset?branch=master)
[![Documentation Status](https://readthedocs.org/projects/niftidataset/badge/?version=latest)](http://niftidataset.readthedocs.io/en/latest/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

This package simply provides appropriate `dataset` and `transforms` classes for NIfTI images 
for use with PyTorch or PyTorch wrappers

** Note that this is an **alpha** release. If you have feedback or problems, please submit an issue (it is very appreciated) **

Requirements
------------

- nibabel
- numpy
- torch
- torchvision

Installation
------------

    pip install git+git://github.com/jcreinhold/niftidataset.git

Tutorial
--------

[5 minute Overview](https://github.com/jcreinhold/niftidataset/blob/master/tutorials/5min-tutorial.ipynb)

In addition to the above small tutorial, there is consolidated documentation [here](https://niftidataset.readthedocs.io/en/latest/).

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v tests
