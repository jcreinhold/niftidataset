#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
niftidataset.dataset

the actual dataset classes of niftidataset

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 24, 2018
"""

__all__ = ['NiftiDataset']

from typing import Callable, List, Optional

import nibabel as nib
import numpy as np
from torch.utils.data.dataset import Dataset

from .utils import glob_nii


class NiftiDataset(Dataset):
    """
    create a dataset class in PyTorch for reading NIfTI files

    Args:
        source_dir (str): path to source images
        target_dir (str): path to target images
        transform (Callable): transform to apply to both source and target images
        preload (bool): load all data when initializing the dataset
    """

    def __init__(self, source_dir:str, target_dir:str, transform:Optional[Callable]=None, preload:bool=False):
        self.source_dir, self.target_dir = source_dir, target_dir
        self.source_fns, self.target_fns = glob_nii(source_dir), glob_nii(target_dir)
        self.transform = transform
        self.preload = preload
        if len(self.source_fns) != len(self.target_fns) or len(self.source_fns) == 0:
            raise ValueError(f'Number of source and target images must be equal and non-zero')
        if preload:
            self.imgs = [(nib.load(s).get_data(), nib.load(t).get_data())
                         for s, t in zip(self.source_fns, self.target_fns)]

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, idx:int):
        if not self.preload:
            src_fn, tgt_fn = self.source_fns[idx], self.target_fns[idx]
            sample = (nib.load(src_fn).get_data(), nib.load(tgt_fn).get_data())
        else:
            sample = self.imgs[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class MultimodalNiftiDataset(Dataset):
    """
    create a dataset class in PyTorch for reading N types of NIfTI files to M types of output NIfTI files

    ** note that all images must have the same dimensions! **

    Args:
        source_dirs (List[str]): paths to source images
        target_dirs (List[str]): paths to target images
        transform (Callable): transform to apply to both source and target images
    """

    def __init__(self, source_dirs:str, target_dirs: str, transform: Optional[Callable]=None):
        self.source_dirs, self.target_dirs = source_dirs, target_dirs
        self.source_fns, self.target_fns = [glob_nii(sd) for sd in source_dirs], [glob_nii(td) for td in target_dirs]
        self.transform = transform
        if any([len(self.source_fns[0]) != len(sfn) for sfn in self.source_fns]) or \
           any([len(self.target_fns[0]) != len(tfn) for tfn in self.target_fns]) or \
           len(self.source_fns[0]) != len(self.target_fns[0]) or \
           len(self.source_fns[0]) == 0:
            raise ValueError(f'Number of source and target images must be equal and non-zero')

    def __len__(self):
        return len(self.source_fns[0])

    def __getitem__(self, idx:int):
        src_fns, tgt_fns = [sfns[idx] for sfns in self.source_fns], [tfns[idx] for tfns in self.target_fns]
        sample = (np.stack([nib.load(s).get_data() for s in src_fns]),
                  np.stack([nib.load(t).get_data() for t in tgt_fns]))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
