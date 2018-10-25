niftidataset
=======================

[![Build Status](https://travis-ci.org/jcreinhold/niftidataset.svg?branch=master)](https://travis-ci.org/jcreinhold/niftidataset)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/niftidataset/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/niftidataset?branch=master)
[![Documentation Status](https://readthedocs.org/projects/niftidataset/badge/?version=latest)](http://niftidataset.readthedocs.io/en/latest/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

This package simply provides appropriate `dataset` and `transforms` for NIfTI images 
for use with PyTorch or PyTorch wrappers

** Note that while this release was carefully inspected, there may be bugs. Please submit an issue if you encounter a problem. **

This package was developed by [Jacob Reinhold](https://jcreinhold.github.io) and the other students and researchers of the 
[Image Analysis and Communication Lab (IACL)](http://iacl.ece.jhu.edu/index.php/Main_Page).

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

[5 minute Overview](https://github.com/jcreinhold/intensity-normalization/blob/master/tutorials/5min_tutorial.md)

In addition to the above small tutorial, there is consolidated documentation [here](https://intensity-normalization.readthedocs.io/en/latest/).

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v tests
