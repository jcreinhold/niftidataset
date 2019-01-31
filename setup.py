#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Module installs the niftidataset package
Can be run via command: python setup.py install (or develop)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 24, 2018
"""

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='niftidataset',
    version='0.1.4',
    description="dataset and transforms classes for NIfTI data in pytorch",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://github.com/jcreinhold/niftidataset',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'tutorials')),
    keywords="nifti dataset",
)

setup(install_requires=['nibabel',
                        'numpy',
                        'pillow',
                        'torch',
                        'torchvision'], **args)
