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
           'NiftiImageList',
           'NiftiNiftiList',
           'open_tiff',
           'TiffImageList',
           'TiffTiffList',
           'TIFFTupleList']

from functools import singledispatch
import logging
import math
from pathlib import PosixPath
from typing import Tuple

import fastai.vision as faiv
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
import torch

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


class NiftiImageList(faiv.ImageItemList):
    """ custom item list for nifti files """
    def open(self, fn:faiv.PathOrStr)->faiv.Image: return open_nii(fn)


class NiftiNiftiList(NiftiImageList):
    """ item list suitable for synthesis tasks """
    _label_cls = NiftiImageList


############## TIFF dataset classes and helper functions ##############

def open_tiff(fn:faiv.PathOrStr)->faiv.Image:
    """ open a 1 channel tif image and transform it into a fastai image """
    return faiv.Image(torch.Tensor(np.asarray(Image.open(fn),dtype=np.float32)[None,...]))


class TiffImageList(faiv.ImageItemList):
    """ custom item list for TIFF files """
    def open(self, fn:faiv.PathOrStr)->faiv.Image: return open_tiff(fn)

class TiffTiffList(TiffImageList):
    _label_cls = TiffImageList

####### The below are for prototypes and should probably be avoided #########

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


class TIFFTupleList(TiffImageList):
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
        itemsB = TiffImageList.from_folder(path/folderB).items
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
