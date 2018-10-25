#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
niftidataset.dataset

the actual dataset classes of niftidataset

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 24, 2018
"""

__all__ = ['NiftiDataset']

from typing import Optional, Callable

import nibabel as nib
from torch.utils.data.dataset import Dataset

from .utils import glob_nii


class NiftiDataset(Dataset):
    """
    create a dataset class in PyTorch for reading NIfTI files

    Args:
        source_dir (str): path to source images
        target_dir (str): path to target images
        transform (Callable): transform to apply to both source and target images
    """

    def __init__(self, source_dir: str, target_dir: str, transform: Optional[Callable]=None):
        self.source_dir, self.target_dir = source_dir, target_dir
        self.source_fns, self.target_fns = glob_nii(source_dir), glob_nii(target_dir)
        self.transform = transform
        if len(self.source_fns) != len(self.target_fns) or len(self.source_fns) == 0:
            raise ValueError(f'Number of source and target images must be equal and non-zero')

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, idx: int):
        src_fn, tgt_fn = self.source_fns[idx], self.target_fns[idx]
        sample = (nib.load(src_fn).get_data(), nib.load(tgt_fn).get_data())
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
