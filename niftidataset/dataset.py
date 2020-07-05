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
           'MultimodalNifti2p5DDataset',
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
        source_fns (List[str]): list of paths to source images
        target_fns (List[str]): list of paths to target images
        transform (Callable): transform to apply to both source and target images
        preload (bool): load all data when initializing the dataset
    """

    def __init__(self, source_fns:str, target_fns:str, transform:Optional[Callable]=None, preload:bool=False):
        self.source_fns, self.target_fns = source_fns, target_fns
        self.transform = transform
        self.preload = preload
        if len(self.source_fns) != len(self.target_fns) or len(self.source_fns) == 0:
            raise ValueError(f'Number of source and target images must be equal and non-zero')
        if preload:
            self.imgs = [(nib.load(s).get_data(), nib.load(t).get_data())
                         for s, t in zip(self.source_fns, self.target_fns)]

    @classmethod
    def setup_from_dir(cls, source_dir:str, target_dir:str, transform:Optional[Callable]=None, preload:bool=False):
        source_fns, target_fns = glob_imgs(source_dir), glob_imgs(target_dir)
        return cls(source_fns, target_fns, transform, preload)

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, idx:int):
        if not self.preload:
            src_fn, tgt_fn = self.source_fns[idx], self.target_fns[idx]
            sample = (nib.load(src_fn).get_fdata(dtype=np.float32), nib.load(tgt_fn).get_fdata(dtype=np.float32))
        else:
            sample = self.imgs[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class MultimodalDataset(Dataset):
    """ base class for Multimodal*Dataset """
    
    def __init__(self, source_fns:List[List[str]], target_fns:List[List[str]],
                 transform:Optional[Callable]=None,
                 segmentation:bool=False, preload:bool=False, **kwargs):
        self.source_fns, self.target_fns = source_fns, target_fns
        self.transform = transform
        self.segmentation = segmentation
        self.preload = preload
        if any([len(self.source_fns[0]) != len(sfn) for sfn in self.source_fns]) or \
           any([len(self.target_fns[0]) != len(tfn) for tfn in self.target_fns]) or \
           len(self.source_fns[0]) != len(self.target_fns[0]) or \
           len(self.source_fns[0]) == 0:
            raise ValueError(f'Number of source and target images must be equal and non-zero')
        if preload:
            self.imgs = []
            for idx in range(len(self.source_fns[0])):
                src_fns, tgt_fns = [sfns[idx] for sfns in self.source_fns], [tfns[idx] for tfns in self.target_fns]
                self.imgs.append((self.stack([self.get_data(s) for s in src_fns]),
                                  self.stack([self.get_data(t) for t in tgt_fns])))

    @classmethod
    def setup_from_dir(cls, source_dirs:List[str], target_dirs:List[str],
                       transform:Optional[Callable]=None, segmentation:bool=False,
                       preload:bool=False, ext:str='*.nii*', **kwargs):
        source_fns = [glob_imgs(sd, ext) for sd in source_dirs]
        target_fns = [glob_imgs(td, ext) for td in target_dirs]
        return cls(source_fns, target_fns, transform, segmentation, preload, **kwargs)

    def __len__(self):
        return len(self.source_fns[0])
    
    def __getitem__(self, idx:int):
        if not self.preload:
            src_fns, tgt_fns = [sfns[idx] for sfns in self.source_fns], [tfns[idx] for tfns in self.target_fns]
            sample = (self.stack([self.get_data(s) for s in src_fns]),
                      self.stack([self.get_data(t) for t in tgt_fns]))
        else:
            sample = self.imgs[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.segmentation:
            sample = (sample[0], sample[1].squeeze().long())  # for segmentation, loss expects no channel dim
        return sample

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
    
    def get_data(self, fn): return nib.load(fn).get_fdata(dtype=np.float32)

    def stack(self, imgs): return np.stack(imgs)


class MultimodalNifti2p5DDataset(MultimodalNiftiDataset):
    """
    create a dataset class in PyTorch for reading N types of NIfTI files to M types of output NIfTI files
    2.5D dataset, so return images stacked in the channel dimension for processing with a 2D CNN

    ** note that all images must have the same dimensions! **

    Args:
        source_dirs (List[str]): paths to source images
        target_dirs (List[str]): paths to target images
        transform (Callable): transform to apply to both source and target images
    """
    def __init__(self, source_dirs:List[str], target_dirs:List[str], transform:Optional[Callable]=None,
                 segmentation:bool=False, preload:bool=False, axis:int=0):
        self.axis = axis
        super().__init__(source_dirs, target_dirs, transform, segmentation, preload)

    def stack(self, imgs): return np.swapaxes(np.concatenate(imgs, axis=self.axis), 0, self.axis)


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
        color (bool): images are color, ie, 3 channels
    """

    def __init__(self, source_dirs:List[str], target_dirs:List[str], transform:Optional[Callable]=None,
                 segmentation:bool=False, color:bool=False, preload:bool=False):
        self.color = color
        super().__init__(source_dirs, target_dirs, transform, segmentation, preload)

    @classmethod
    def setup_from_dir(cls, source_dirs:List[str], target_dirs:List[str],
                       transform:Optional[Callable]=None,
                       segmentation:bool=False,
                       color:bool=False, preload:bool=False,
                       ext:str='*.tif*', **kwargs):
        source_fns = [glob_imgs(sd, ext) for sd in source_dirs]
        target_fns = [glob_imgs(td, ext) for td in target_dirs]
        return cls(source_fns, target_fns, transform, segmentation, color, preload)

    def get_data(self, fn):
        data = np.asarray(Image.open(fn), dtype=np.float32)
        if self.color: data = data.transpose((2,0,1))
        return data

    def stack(self, imgs):
        data = np.stack(imgs)
        if self.color: data = data.squeeze()
        return data


def get_train_and_validation_from_one_directory(source_dir: str, target_dir: str, valid_pct: float = 0.2,
                                                dataset_class: Dataset = NiftiDataset,
                                                transform: Optional[Callable] = None, preload: bool = False):
    """
    :param source_dir: path to source images
    :param target_dir: path to target images
    :param valid_pct: percent of validation set from data
    :param dataset_class: class of Dataset wanted to be returned
    :param transform: transform to apply to both source and target images
    :param preload: load all data when initializing the dataset
    :return: tuple of (train_dataset, validation_dataset)
    """
    if not (0 < valid_pct < 1):
        raise ValueError(f'valid_pct must be between 0 and 1')
    source_fns, target_fns = glob_imgs(source_dir), glob_imgs(target_dir)
    rand_idx = np.random.permutation(list(range(len(self.source_fns))))
    cut = int(valid_pct * len(self.source_fns))
    return (dataset_class(source_fns=[source_fns[i] for i in rand_idx[cut:]],
                          target_fns=[target_fns[i] for i in rand_idx[cut:]],
                          transform=transform, preload=preload),
            dataset_class(source_fns=[source_fns[i] for i in rand_idx[:cut]],
                          target_fns=[target_fns[i] for i in rand_idx[:cut]],
                          transform=transform, preload=preload))

