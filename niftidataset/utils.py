#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
niftidataset.utils

assortment of input/output utilities for the project

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 24, 2018
"""

__all__ = ['split_filename',
           'glob_nii',
           'glob_tiff']

from typing import List, Tuple

from glob import glob
import os


def split_filename(filepath: str) -> Tuple[str, str, str]:
    """ split a filepath into the directory, base, and extension """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def glob_nii(path: str) -> List[str]:
    """ grab all nifti files in a directory and sort them for consistency """
    fns = sorted(glob(os.path.join(path, '*.nii*')))
    return fns


def glob_tiff(path: str) -> List[str]:
    """ grab all .tif or .tiff files in a directory and sort them for consistency """
    fns = sorted(glob(os.path.join(path, '*.tif*')))
    return fns
