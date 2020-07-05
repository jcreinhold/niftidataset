#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
niftidataset.dataset

the actual dataset classes of niftidataset

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 24, 2018
"""

__all__ = ['NiftiTrainValidation',
           'NiftiDataset',
           'MultimodalNiftiDataset',
           'MultimodalNifti2p5DDataset',
           'MultimodalImageDataset']

from typing import Callable, List, Optional

import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import glob_imgs


class NiftiTrainValidation(object):
    def __init__(self, source_dir: str, target_dir: str, valid_pct=0.2, transform: Optional[Callable] = None,
                 preload: bool = False):
        self.source_dir, self.target_dir = source_dir, target_dir
        self.source_fns, self.target_fns = glob_imgs(source_dir), glob_imgs(target_dir)
        rand_idx = np.random.permutation(list(range(len(self.source_fns))))
        cut = int(valid_pct * len(self.source_fns))
        self.validation = NiftiLightDataset(source_fns=[self.source_fns[i] for i in rand_idx[:cut]],
                                            target_fns=[self.target_fns[i] for i in rand_idx[:cut]],
                                            transform=transform, preload=preload)
        self.train = NiftiLightDataset(source_fns=[self.source_fns[i] for i in rand_idx[cut:]],
                                       target_fns=[self.target_fns[i] for i in rand_idx[cut:]],
                                       transform=transform, preload=preload)

    def get_train(self):
        return self.train

    def get_validation(self):
        return self.validation


class NiftiLightDataset(Dataset):
    def __init__(self, source_fns, target_fns, transform: Optional[Callable] = None, preload: bool = False):
        self.source_fns, self.target_fns = source_fns, target_fns
        self.transform = transform
        self.preload = preload
        if len(self.source_fns) != len(self.target_fns) or len(self.source_fns) == 0:
            raise ValueError(f'Number of source and target images must be equal and non-zero')
        if preload:
            self.imgs = [(nib.load(s).get_data(), nib.load(t).get_data())
                         for s, t in zip(self.source_fns, self.target_fns)]

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, idx: int):
        if not self.preload:
            src_fn, tgt_fn = self.source_fns[idx], self.target_fns[idx]
            sample = (nib.load(src_fn).get_fdata(dtype=np.float32), nib.load(tgt_fn).get_fdata(dtype=np.float32))
        else:
            sample = self.imgs[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class NiftiDataset(Dataset):
    """
    create a dataset class in PyTorch for reading NIfTI files

    Args:
        source_dir (str): path to source images
        target_dir (str): path to target images
        transform (Callable): transform to apply to both source and target images
        preload (bool): load all data when initializing the dataset
    """

    def __init__(self, source_dir: str, target_dir: str, transform: Optional[Callable] = None, preload: bool = False):
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

    def __getitem__(self, idx: int):
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

    def __init__(self, source_dirs: List[str], target_dirs: List[str], transform: Optional[Callable] = None,
                 segmentation: bool = False, preload: bool = False):
        self.source_dirs, self.target_dirs = source_dirs, target_dirs
        self.source_fns, self.target_fns = [self.glob_imgs(sd) for sd in source_dirs], [self.glob_imgs(td) for td in
                                                                                        target_dirs]
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

    def __len__(self):
        return len(self.source_fns[0])

    def __getitem__(self, idx: int):
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

    def glob_imgs(self, path):
        raise NotImplementedError

    def get_data(self, fn):
        raise NotImplementedError

    def stack(self, imgs):
        raise NotImplementedError


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

    def __init__(self, source_dirs: List[str], target_dirs: List[str], transform: Optional[Callable] = None,
                 segmentation: bool = False, preload: bool = False, axis: int = 0):
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
        ext (str): extension of desired images with * to allow glob to pick up all images in directory
            e.g., `*.tif*` to pick up all TIFF images with ext `.tif` or `.tiff` (may pick up more so be careful)
        color (bool): images are color, ie, 3 channels
    """

    def __init__(self, source_dirs: List[str], target_dirs: List[str], transform: Optional[Callable] = None,
                 segmentation: bool = False,
                 ext: str = '*.tif*', color: bool = False, preload: bool = False):
        self.ext = ext
        self.color = color
        super().__init__(source_dirs, target_dirs, transform, segmentation, preload)

    def glob_imgs(self, path):
        return glob_imgs(path, ext=self.ext)

    def get_data(self, fn):
        data = np.asarray(Image.open(fn), dtype=np.float32)
        if self.color: data = data.transpose((2, 0, 1))
        return data

    def stack(self, imgs):
        data = np.stack(imgs)
        if self.color: data = data.squeeze()
        return data
