#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
niftidataset.transforms

transformations to apply to images in dataset

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 24, 2018
"""

__all__ = ['RandomCrop2D',
           'RandomCrop3D',
           'ToTensor']

from typing import Union, Optional, Tuple

import numpy as np
import torch


class CropBase:
    """ base class for crop transform """

    def __init__(self, out_dim: int, output_size: Union[tuple, int]):
        """ provide the common functionality for RandomCrop2D and RandomCrop3D """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size,)
            for _ in range(out_dim - 1):
                self.output_size += (output_size,)
        else:
            assert len(output_size) == out_dim
            self.output_size = output_size
        self.out_dim = out_dim

    def get_sample_idxs(self, img: np.ndarray, mask: Optional[np.ndarray] = None):
        """ get the set of indices from which to sample (foreground) """
        mask = np.where(img > img.mean())  # returns a tuple of length 3
        c = np.random.randint(0, len(mask[0]))  # choose the set of idxs to use
        hh, ww, dd = [m[c] for m in mask]  # pull out the chosen idxs
        return hh, ww, dd


class RandomCrop2D(CropBase):
    """
    Randomly crop a 2d slice/patch from a 3d image

    Args:
        output_size (tuple or int): Desired output size.
            If int, cube crop is made.
        axis (int or None): along which axis should the patch/slice be extracted
            provide None for random axis
    """

    def __init__(self, output_size: Union[tuple, int], axis: Union[int, None] = 0):
        if axis is not None:
            assert axis <= 2
        super().__init__(2, output_size)
        self.axis = axis

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]):
        axis = self.axis or np.random.randint(0, 3)
        src, tgt = sample
        assert src.shape == tgt.shape
        h, w, d = src.shape
        new_h, new_w = self.output_size
        max_idxs = (np.inf, w - new_h, d - new_w) if axis == 0 else \
            (h - new_h, np.inf, d - new_w) if axis == 1 else \
                (h - new_h, w - new_w, np.inf)
        hh, ww, dd = [min(max_i, i) for max_i, i in zip(max_idxs, super().get_sample_idxs(src))]
        s = src[hh, ww: ww + new_h, dd: dd + new_w] if axis == 0 else \
            src[hh: hh + new_h, ww, dd: dd + new_w] if axis == 1 else \
                src[hh: hh + new_h, ww: ww + new_w, dd]
        t = tgt[hh, ww: ww + new_h, dd: dd + new_w] if axis == 0 else \
            tgt[hh: hh + new_h, ww, dd: dd + new_w] if axis == 1 else \
                tgt[hh: hh + new_h, ww: ww + new_w, dd]
        return s, t


class RandomCrop3D(CropBase):
    """
    Randomly crop a 3d patch from a (pair of) 3d image

    Args:
        output_size (tuple or int): Desired output size.
            If int, cube crop is made.
    """

    def __init__(self, output_size: Union[tuple, int]):
        super().__init__(3, output_size)

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]):
        src, tgt = sample
        assert src.shape == tgt.shape
        h, w, d = src.shape
        new_h, new_w, new_d = self.output_size
        max_idxs = (h - new_h, w - new_w, d - new_d)
        hh, ww, dd = [min(max_i, i) for max_i, i in zip(max_idxs, super().get_sample_idxs(src))]
        s = src[hh: hh + new_h, ww: ww + new_w, dd: dd + new_d]
        t = tgt[hh: hh + new_h, ww: ww + new_w, dd: dd + new_d]
        return s, t


class ToTensor:
    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]):
        src, tgt = sample
        assert src.shape == tgt.shape
        return (torch.from_numpy(src), torch.from_numpy(tgt))
