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
           'niidatabunch']

from functools import singledispatch
import logging
import math
from typing import Callable, List, Optional, Union

import fastai as fai
import fastai.vision as faiv
import nibabel as nib
import numpy as np
import torch

from .utils import glob_nii

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


def niidatabunch(src_dir:str, tgt_dir:str, split:float=0.2, tfms:Optional[List[Callable]]=None,
                 val_tfms:Optional[List[Callable]]=None, path:str='.', bs:int=32, device:Union[str,torch.device]="cpu",
                 n_jobs=fai.defaults.cpus, val_src_dir:Optional[str]=None, val_tgt_dir:Optional[str]=None,
                 b_per_epoch:int=1) -> faiv.ImageDataBunch:
    """ create a NIfTI databunch from two directories """
    use_val_dir = isinstance(val_src_dir, str) and isinstance(val_tgt_dir, str)
    src_fns, tgt_fns = __get_fns(src_dir, tgt_dir, bs, b_per_epoch, use_val_dir)
    src = fai.ItemList(src_fns, create_func=open_nii)
    tgt = fai.ItemList(tgt_fns, create_func=open_nii)
    if use_val_dir:
        train_src, train_tgt = src, tgt
        val_src_fns, val_tgt_fns = __get_fns(val_src_dir, val_tgt_dir, bs, 1, use_val_dir)
        valid_src = fai.ItemList(val_src_fns, create_func=open_nii)
        valid_tgt = fai.ItemList(val_tgt_fns, create_func=open_nii)
    else:
        val_idxs = np.random.choice(len(src_fns), int(split * len(src_fns)))
        src = src.split_by_idx(val_idxs)
        tgt = tgt.split_by_idx(val_idxs)
        train_src, train_tgt = src.train, tgt.train
        valid_src, valid_tgt = src.valid, tgt.valid
    train_ll = fai.LabelList(train_src, train_tgt, tfms, tfm_y=True)
    val_tfms = val_tfms or tfms
    val_ll = fai.LabelList(valid_src, valid_tgt, val_tfms, tfm_y=True)
    ll = fai.LabelLists(path, train_ll, val_ll)
    idb = faiv.ImageDataBunch.create_from_ll(ll, bs=bs, device=device, num_workers=n_jobs)
    return idb


def __get_fns(src_dir:str, tgt_dir:str, bs:int, b_per_epoch:int, use_val_dir:bool):
    m = 1 if use_val_dir else 2
    src_fns = glob_nii(src_dir)
    tgt_fns = glob_nii(tgt_dir)
    if len(src_fns) != len(tgt_fns) or len(src_fns) == 0:
        raise ValueError(f'Number of source and target images must be equal and non-zero')
    if len(src_fns) < bs:
        src_fns = src_fns * math.ceil(bs / len(src_fns)) * m
        tgt_fns = tgt_fns * math.ceil(bs / len(tgt_fns)) * m
    if len(src_fns) // bs < b_per_epoch:
        src_fns = src_fns * b_per_epoch
        tgt_fns = tgt_fns * b_per_epoch
    return src_fns, tgt_fns
