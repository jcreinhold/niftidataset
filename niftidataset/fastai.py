#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
niftidataset.fastai

functions to support fastai datasets

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Nov 15, 2018
"""

__all__ = ['open_nii',
           'get_slice',
           'get_patch3d',
           'add_channel',
           'NIfTIItemList',
           'niidatabunch',
           'open_tiff',
           'TIFFImageList',
           'tiffdatabunch',
           'TIFFTupleList']

from functools import singledispatch
import logging
import math
from pathlib import PosixPath
from typing import Callable, List, Optional, Tuple, Union

import fastai.vision as faiv
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
import torch

from .utils import glob_nii, glob_tiff
from .errors import NiftiDatasetError

logger = logging.getLogger(__name__)


def open_nii(fn:str) -> faiv.Image:
    """ Return fastai `Image` object created from NIfTI image in file `fn`."""
    x = nib.load(str(fn)).get_data()
    return faiv.Image(torch.Tensor(x))


@faiv.TfmPixel
@singledispatch
def get_slice(x, pct:faiv.uniform=0.5, axis:int=0) -> torch.Tensor:
    """" Get a random slice of `x` based on axis """
    s = int(x.size(axis) * pct)
    return x[np.newaxis,s,:,:].contiguous() if axis == 0 else \
           x[np.newaxis,:,s,:].contiguous() if axis == 1 else \
           x[np.newaxis,:,:,s].contiguous()


@faiv.TfmPixel
@singledispatch
def get_patch3d(x, ps:int=64, h_pct:faiv.uniform=0.5, w_pct:faiv.uniform=0.5, d_pct:faiv.uniform=0.5) -> torch.Tensor:
    """" Get a random 3d patch of `x` of size ps^3 """
    h, w, d = x.shape
    max_idxs = (h - ps // 2, w - ps // 2, d - ps // 2)
    min_idxs = (ps // 2, ps // 2, ps // 2)
    s_idxs = (int(h * h_pct), int(w * w_pct), int(d * d_pct))
    i, j, k = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
               for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
    o = 0 if ps % 2 == 0 else 1
    return x[np.newaxis, i-ps//2:i+ps//2+o, j-ps//2:j+ps//2+o, k-ps//2:k+ps//2+o].contiguous()


@faiv.TfmPixel
@singledispatch
def add_channel(x) -> torch.Tensor:
    """" add channel to img (used when extracting whole image) """
    return x[np.newaxis, ...].contiguous()


class NIfTIItemList(faiv.ImageItemList):
    """ custom item list for nifti files """
    def open(self, fn:faiv.PathOrStr)->faiv.Image: return open_nii(fn)


def niidatabunch(src_dir:str, tgt_dir:str, split:float=0.2, tfms:Optional[List[Callable]]=None,
                 val_tfms:Optional[List[Callable]]=None, path:str='.', bs:int=32, device:Union[str,torch.device]="cpu",
                 n_jobs=faiv.defaults.cpus, val_src_dir:Optional[str]=None, val_tgt_dir:Optional[str]=None,
                 b_per_epoch:int=1) -> faiv.ImageDataBunch:
    """ create a NIfTI databunch from two directories, returns an image to image databunch """
    idb = __databunch(src_dir, tgt_dir, split, tfms, val_tfms, path, bs, device, n_jobs, val_src_dir, val_tgt_dir,
                      b_per_epoch, type='nii')
    return idb


def __databunch(src_dir:str, tgt_dir:str, split:float=0.2, tfms:Optional[List[Callable]]=None,
                val_tfms:Optional[List[Callable]]=None, path:str='.', bs:int=32, device:Union[str,torch.device]="cpu",
                n_jobs=faiv.defaults.cpus, val_src_dir:Optional[str]=None, val_tgt_dir:Optional[str]=None,
                b_per_epoch:int=1, type:str='nii') -> faiv.ImageDataBunch:
    """ create an X databunch from two directories, returns an image to image databunch """
    if type == 'nii':
        itemlist = NIfTIItemList
    elif type == 'tif':
        itemlist = TIFFImageList
    else:
        raise NiftiDatasetError('type needs to be either `nii` or `tif`')
    use_val_dir = isinstance(val_src_dir, str) and isinstance(val_tgt_dir, str)
    src_fns, tgt_fns = __get_fns(src_dir, tgt_dir, bs, b_per_epoch, use_val_dir, type)
    src = itemlist(src_fns)
    tgt = itemlist(tgt_fns)
    if use_val_dir:
        train_src, train_tgt = src, tgt
        val_src_fns, val_tgt_fns = __get_fns(val_src_dir, val_tgt_dir, bs, 1, use_val_dir, type)
        valid_src = itemlist(val_src_fns)
        valid_tgt = itemlist(val_tgt_fns)
    else:
        val_idxs = np.random.choice(len(src_fns), int(split * len(src_fns)))
        src = src.split_by_idx(val_idxs)
        tgt = tgt.split_by_idx(val_idxs)
        train_src, train_tgt = src.train, tgt.train
        valid_src, valid_tgt = src.valid, tgt.valid
    train_ll = faiv.LabelList(train_src, train_tgt, tfms, tfm_y=True)
    train_ll.transform(tfms, tfm_y=True)
    val_tfms = val_tfms or tfms
    val_ll = faiv.LabelList(valid_src, valid_tgt, val_tfms, tfm_y=True)
    val_ll.transform(val_tfms, tfm_y=True)
    ll = faiv.LabelLists(path, train_ll, val_ll)
    idb = faiv.ImageDataBunch.create_from_ll(ll, bs=bs, device=device, num_workers=n_jobs)
    return idb


def __get_fns(src_dir:str, tgt_dir:str, bs:int, b_per_epoch:int, use_val_dir:bool, type:str):
    m = 1 if use_val_dir else 2
    src_fns = glob_nii(src_dir) if type == 'nii' else glob_tiff(src_dir)
    tgt_fns = glob_nii(tgt_dir) if type == 'nii' else glob_tiff(tgt_dir)
    if len(src_fns) != len(tgt_fns) or len(src_fns) == 0:
        raise ValueError(f'Number of source and target images must be equal and non-zero')
    if len(src_fns) < bs:
        src_fns = src_fns * math.ceil(bs / len(src_fns)) * m
        tgt_fns = tgt_fns * math.ceil(bs / len(tgt_fns)) * m
    if len(src_fns) // bs < b_per_epoch:
        src_fns = src_fns * b_per_epoch
        tgt_fns = tgt_fns * b_per_epoch
    logger.debug(f'Number of batches per epoch: {len(src_fns) // bs}')
    return src_fns, tgt_fns


############## TIFF dataset classes and helper functions ##############

def open_tiff(fn:faiv.PathOrStr)->faiv.Image:
    """ open a 1 channel tif image and transform it into a fastai image """
    return faiv.Image(torch.Tensor(np.asarray(Image.open(fn),dtype=np.float32)[None,...]))


class TIFFImageList(faiv.ImageItemList):
    """ custom item list for TIFF files """
    def open(self, fn:faiv.PathOrStr)->faiv.Image: return open_tiff(fn)


def tiffdatabunch(src_dir:str, tgt_dir:str, split:float=0.2, tfms:Optional[List[Callable]]=None,
                  val_tfms:Optional[List[Callable]]=None, path:str='.', bs:int=32, device:Union[str,torch.device]="cpu",
                  n_jobs=faiv.defaults.cpus, val_src_dir:Optional[str]=None, val_tgt_dir:Optional[str]=None,
                  b_per_epoch:int=1) -> faiv.ImageDataBunch:
    """ create a NIfTI databunch from two directories, returns an image to image databunch """
    idb = __databunch(src_dir, tgt_dir, split, tfms, val_tfms, path, bs, device, n_jobs, val_src_dir, val_tgt_dir,
                      b_per_epoch, type='tif')
    return idb


class ImageTuple(faiv.ItemBase):
    def __init__(self, img1, img2):
        self.img1,self.img2 = img1,img2
        self.obj,self.data = (img1,img2),[img1.data,img2.data]

    def apply_tfms(self, tfms, **kwargs):
        self.img1 = self.img1.apply_tfms(tfms, **kwargs)
        self.img2 = self.img2.apply_tfms(tfms, **kwargs)
        return self

    def to_one(self):
        return faiv.Image(torch.cat(self.data, 2))

    def show_xys(self, xs, ys, figsize: Tuple[int, int] = (9, 10), **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows, rows, figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].show(ax=ax, y=ys[i], **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize: Tuple[int, int] = None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`. 
        `kwargs` are passed to the show method."""
        figsize = faiv.ifnone(figsize, (6, 3 * len(xs)))
        fig, axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            x.show(ax=axs[i, 0], y=y, **kwargs)
            x.show(ax=axs[i, 1], y=z, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__} - im1:{tuple(self.img1.shape)}, im2:{tuple(self.img2.shape)}'

class TargetTupleList(faiv.ItemList):
    def reconstruct(self, t:torch.Tensor):
        if len(t.size()) == 0: return t
        return ImageTuple(faiv.Image(t[0]),faiv.Image(t[1]))


class TIFFTupleList(TIFFImageList):
    _label_cls = TargetTupleList
    def __init__(self, items, itemsB=None, **kwargs):
        self.itemsB = itemsB
        super().__init__(items, **kwargs)

    def new(self, items, **kwargs):
        return super().new(items, itemsB=self.itemsB, **kwargs)

    def get(self, i):
        img1 = super().get(i)
        fn = self.itemsB[i]
        return ImageTuple(img1, open_tiff(fn))

    def reconstruct(self, t:torch.Tensor):
        return ImageTuple(faiv.Image(t[0]),faiv.Image(t[1]))

    @classmethod
    def from_folders(cls, path, folderA, folderB, **kwargs):
        path = PosixPath(path)
        itemsB = TIFFImageList.from_folder(path/folderB).items
        res = super().from_folder(path/folderA, itemsB=itemsB, **kwargs)
        res.path = path
        return res

    def show_xys(self, xs, ys, figsize:Tuple[int,int]=(12,6), **kwargs):
        "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].to_one().show(ax=ax, **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
        """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
        `kwargs` are passed to the show method."""
        figsize = faiv.ifnone(figsize, (12,3*len(xs)))
        fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
        fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
        for i,(x,z) in enumerate(zip(xs,zs)):
            x.to_one().show(ax=axs[i,0], **kwargs)
            z.to_one().show(ax=axs[i,1], **kwargs)
