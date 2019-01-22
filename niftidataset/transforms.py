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
           'RandomSlice',
           'ToTensor',
           'ToFastaiImage',
           'AddChannel',
           'Normalize']

from typing import Optional, Tuple, Union

import numpy as np
import torch


class CropBase:
    """ base class for crop transform """

    def __init__(self, out_dim:int, output_size:Union[tuple,int]):
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

    def _get_sample_idxs(self, img:np.ndarray, mask:Optional[np.ndarray]=None) -> Tuple[int,int,int]:
        """ get the set of indices from which to sample (foreground) """
        mask = np.where(img > img.mean())  # returns a tuple of length 3
        c = np.random.randint(0, len(mask[0]))  # choose the set of idxs to use
        h, w, d = [m[c] for m in mask]  # pull out the chosen idxs
        return h, w, d


class RandomCrop2D(CropBase):
    """
    Randomly crop a 2d slice/patch from a 3d image

    Args:
        output_size (tuple or int): Desired output size.
            If int, cube crop is made.
        axis (int or None): along which axis should the patch/slice be extracted
            provide None for random axis
        include_neighbors (bool): extract 3 neighboring slices instead of just 1
    """

    def __init__(self, output_size:Union[tuple, int], axis:Union[int, None]=0,
                 include_neighbors: bool= False) -> None:
        if axis is not None:
            assert axis <= 2
        super().__init__(2, output_size)
        self.axis = axis
        self.include_neighbors = include_neighbors

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        axis = self.axis if self.axis is not None else np.random.randint(0, 3)
        src, tgt = sample
        assert src.shape == tgt.shape and src.ndim == 3
        h, w, d = src.shape
        new_h, new_w = self.output_size
        max_idxs = (np.inf, w - new_h//2, d - new_w//2) if axis == 0 else \
                   (h - new_h//2, np.inf, d - new_w//2) if axis == 1 else \
                   (h - new_h//2, w - new_w//2, np.inf)
        min_idxs = (-np.inf, new_h//2, new_w//2) if axis == 0 else \
                   (new_h//2, -np.inf, new_w//2) if axis == 1 else \
                   (new_h//2, new_w//2, -np.inf)
        s_idxs = super()._get_sample_idxs(src)
        idxs = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        s = self.__get_slice(src, idxs, axis)
        t = self.__get_slice(tgt, idxs, axis)
        return s, t

    def __get_slice(self, img:np.ndarray, idxs:Tuple[int,int,int], axis:int) -> np.ndarray:
        h, w = self.output_size
        n = 1 if self.include_neighbors else 0
        oh = 0 if h % 2 == 0 else 1
        ow = 0 if w % 2 == 0 else 1
        i, j, k = idxs
        s = img[i-n:i+1+n, j-h//2:j+h//2+oh, k-w//2:k+w//2+ow] if axis == 0 else \
            img[i-h//2:i+h//2+oh, j-n:j+1+n, k-w//2:k+w//2+ow] if axis == 1 else \
            img[i-h//2:i+h//2+oh, j-w//2:j+w//2+ow, k-n:k+1+n]
        if self.include_neighbors:
            s = np.transpose(s, (0,1,2)) if axis == 0 else \
                np.transpose(s, (1,0,2)) if axis == 1 else \
                np.transpose(s, (2,0,1))
        else:
            s = np.squeeze(s)[np.newaxis, ...]  # add empty channel
        return s


class RandomCrop3D(CropBase):
    """
    Randomly crop a 3d patch from a (pair of) 3d image

    Args:
        output_size (tuple or int): Desired output size.
            If int, cube crop is made.
    """

    def __init__(self, output_size:Union[tuple,int]):
        super().__init__(3, output_size)

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        src, tgt = sample
        assert src.shape == tgt.shape
        h, w, d = src.shape
        hh, ww, dd = self.output_size
        max_idxs = (h-hh//2, w-ww//2, d-dd//2)
        min_idxs = (hh//2, ww//2, dd//2)
        s_idxs = super()._get_sample_idxs(src)
        i, j, k = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                   for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        oh = 0 if hh % 2 == 0 else 1
        ow = 0 if ww % 2 == 0 else 1
        od = 0 if dd % 2 == 0 else 1
        s = src[np.newaxis, i-hh//2:i+hh//2+oh, j-ww//2:j+ww//2+ow, k-dd//2:k+dd//2+od]
        t = tgt[np.newaxis, i-hh//2:i+hh//2+oh, j-ww//2:j+ww//2+ow, k-dd//2:k+dd//2+od]
        return s, t


class RandomSlice:
    """
    take a random 2d slice from an image given a sample axis

    Args:
        axis (int): axis on which to take a slice
        div (float): divide the mean by this value in the calculation of mask
            the higher this value, the more background will be "valid"
    """

    def __init__(self, axis:int=0, div:float=2):
        self.axis = axis
        self.div = div

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        src, tgt = sample
        idx = np.random.choice(self._valid_idxs(src)[self.axis])
        s = self._get_slice(src, idx)
        t = self._get_slice(tgt, idx)
        return s, t

    def _get_slice(self, img:np.ndarray, idx:int):
        s = img[idx,:,:] if self.axis == 0 else \
            img[:,idx,:] if self.axis == 1 else \
            img[:,:,idx]
        return s

    def _valid_idxs(self, img:np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """ get the set of indices from which to sample (foreground) """
        mask = np.where(img > img.mean() / self.div)  # returns a tuple of length 3
        h, w, d = [np.arange(np.min(m), np.max(m)+1) for m in mask]  # pull out the valid idx ranges
        return h, w, d


class ToTensor:
    """ Convert ndarrays in sample to Tensors """

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[torch.Tensor,torch.Tensor]:
        src, tgt = sample
        assert src.shape == tgt.shape
        return (torch.from_numpy(src), torch.from_numpy(tgt))


class ToFastaiImage:
    """ convert a 2D image to fastai.Image class """

    def __init__(self):
        from fastai.vision import Image
        self.Image = Image

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        x, y = sample
        return self.Image(x), self.Image(y)


class AddChannel:
    """ Add empty first dimension to sample """

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor]:
        src, tgt = sample
        assert src.shape == tgt.shape
        return (src.unsqueeze(0), tgt.unsqueeze(0))


class Normalize:
    """ put data in range of 0 to 1 """

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        x, y = sample
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        return x, y
