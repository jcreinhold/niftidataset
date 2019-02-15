#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
niftidataset.dataset

the actual dataset classes of niftidataset

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 24, 2018
"""

__all__ = ['NiftiDataset',
           'MultimodalNiftiDataset',
           'MultimodalImageDataset']

from typing import Callable, List, Optional

import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import glob_imgs


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
        self.source_fns, self.target_fns = glob_imgs(source_dir), glob_imgs(target_dir)
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


class MultimodalDataset(Dataset):
    """ base class for Multimodal*Dataset """
    
    def __init__(self, source_dirs:List[str], target_dirs:List[str], transform:Optional[Callable]=None):
        self.source_dirs, self.target_dirs = source_dirs, target_dirs
        self.source_fns, self.target_fns = [self.glob_imgs(sd) for sd in source_dirs], [self.glob_imgs(td) for td in target_dirs]
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
        sample = (self.stack([self.get_data(s) for s in src_fns]),
                  self.stack([self.get_data(t) for t in tgt_fns]))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def glob_imgs(self, path): raise NotImplementedError
    
    def get_data(self, fn): raise NotImplementedError
    
    def stack(self, imgs): raise NotImplementedError
    

class MultimodalNiftiDataset(MultimodalDataset):
    """
    create a dataset class in PyTorch for reading N types of NIfTI files to M types of output NIfTI files

    ** note that all images must have the same dimensions! **

    Args:
        source_dirs (List[str]): paths to source images
        target_dirs (List[str]): paths to target images
        transform (Callable): transform to apply to both source and target images
    """
    
    def glob_imgs(self, path): return glob_imgs(path, ext='*.nii*')
    
    def get_data(self, fn): return nib.load(fn).get_data()

    def stack(self, imgs): return np.stack(imgs)


class MultimodalImageDataset(MultimodalDataset):
    """
    create a dataset class in PyTorch for reading N types of (no channel) image files to M types of output image files.
    can use whatever PIL can open.

    ** note that all images must have the same dimensions! **

    There is no implementation of ImageDataset because it is sufficient to use normal pytorch image
    dataloaders for that use case

    Args:
        source_dirs (List[str]): paths to source images
        target_dirs (List[str]): paths to target images
        transform (Callable): transform to apply to both source and target images
        ext (str): extension of desired images with * to allow glob to pick up all images in directory
            e.g., `*.tif*` to pick up all TIFF images with ext `.tif` or `.tiff` (may pick up more so be careful)
    """

    def __init__(self, source_dirs:List[str], target_dirs:List[str], transform:Optional[Callable]=None, ext:str='*.tif*'):
        self.ext = ext
        super().__init__(source_dirs, target_dirs, transform)

    def glob_imgs(self, path): return glob_imgs(path, ext=self.ext)
    
    def get_data(self, fn): return np.asarray(Image.open(fn),dtype=np.float32)

    def stack(self, imgs): return np.stack(imgs)
