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
           'RandomCrop',
           'RandomSlice',
           'ToTensor',
           'ToFastaiImage',
           'ToPILImage',
           'AddChannel',
           'Normalize',
           'Digitize',
           'RandomAffine',
           'RandomBlock',
           'RandomFlip',
           'RandomGamma',
           'RandomNoise',
           'get_transforms']

import random
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision as tv
import torchvision.transforms.functional as TF

PILImage = type(Image)


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

    def _get_sample_idxs(self, img:np.ndarray) -> Tuple[int,int,int]:
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
                 include_neighbors:bool=False) -> None:
        if axis is not None:
            assert 0 <= axis <= 2
        super().__init__(2, output_size)
        self.axis = axis
        self.include_neighbors = include_neighbors

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        axis = self.axis if self.axis is not None else np.random.randint(0, 3)
        src, tgt = sample
        *cs, h, w, d = src.shape
        *ct, _, _, _ = src.shape
        new_h, new_w = self.output_size
        max_idxs = (np.inf, w - new_h//2, d - new_w//2) if axis == 0 else \
                   (h - new_h//2, np.inf, d - new_w//2) if axis == 1 else \
                   (h - new_h//2, w - new_w//2, np.inf)
        min_idxs = (-np.inf, new_h//2, new_w//2) if axis == 0 else \
                   (new_h//2, -np.inf, new_w//2) if axis == 1 else \
                   (new_h//2, new_w//2, -np.inf)
        s = src[0] if len(cs) > 0 else src  # use the first image to determine sampling if multimodal
        s_idxs = super()._get_sample_idxs(s)
        idxs = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        s = self._get_slice(src, idxs, axis).squeeze()
        t = self._get_slice(tgt, idxs, axis).squeeze()
        if len(cs) == 0 or s.ndim == 2: s = s[np.newaxis,...]  # add channel axis if empty
        if len(ct) == 0 or t.ndim == 2: t = t[np.newaxis,...]
        return s, t

    def _get_slice(self, img:np.ndarray, idxs:Tuple[int,int,int], axis:int) -> np.ndarray:
        h, w = self.output_size
        n = 1 if self.include_neighbors else 0
        oh = 0 if h % 2 == 0 else 1
        ow = 0 if w % 2 == 0 else 1
        i, j, k = idxs
        s = img[..., i-n:i+1+n, j-h//2:j+h//2+oh, k-w//2:k+w//2+ow] if axis == 0 else \
            img[..., i-h//2:i+h//2+oh, j-n:j+1+n, k-w//2:k+w//2+ow] if axis == 1 else \
            img[..., i-h//2:i+h//2+oh, j-w//2:j+w//2+ow, k-n:k+1+n]
        if self.include_neighbors:
            s = np.transpose(s, (0,1,2)) if axis == 0 else \
                np.transpose(s, (1,0,2)) if axis == 1 else \
                np.transpose(s, (2,0,1))
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
        *cs, h, w, d = src.shape
        *ct, _, _, _ = tgt.shape
        hh, ww, dd = self.output_size
        max_idxs = (h-hh//2, w-ww//2, d-dd//2)
        min_idxs = (hh//2, ww//2, dd//2)
        s = src[0] if len(cs) > 0 else src  # use the first image to determine sampling if multimodal
        s_idxs = super()._get_sample_idxs(s)
        i, j, k = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                   for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        oh = 0 if hh % 2 == 0 else 1
        ow = 0 if ww % 2 == 0 else 1
        od = 0 if dd % 2 == 0 else 1
        s = src[..., i-hh//2:i+hh//2+oh, j-ww//2:j+ww//2+ow, k-dd//2:k+dd//2+od]
        t = tgt[..., i-hh//2:i+hh//2+oh, j-ww//2:j+ww//2+ow, k-dd//2:k+dd//2+od]
        if len(cs) == 0: s = s[np.newaxis,...]  # add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis,...]
        return s, t


class RandomCrop:
    """
    Randomly crop a 2d patch from a 2d image

    Args:
        output_size (tuple or int): Desired output size.
            If int, square crop is made.
    """

    def __init__(self, output_size:Union[tuple,int]):
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        src, tgt = sample
        *cs, h, w = src.shape
        *ct, _, _ = tgt.shape
        hh, ww = self.output_size
        max_idxs = (h-hh//2, w-ww//2)
        min_idxs = (hh//2, ww//2)
        s = src[0] if len(cs) > 0 else src  # use the first image to determine sampling if multimodal
        mask = np.where(s > s.mean())  # returns a tuple of length 3
        c = np.random.randint(0, len(mask[0]))  # choose the set of idxs to use
        s_idxs = [m[c] for m in mask]  # pull out the chosen idxs
        i, j = [i if min_i <= i <= max_i else max_i if i > max_i else min_i
                for max_i, min_i, i in zip(max_idxs, min_idxs, s_idxs)]
        oh = 0 if hh % 2 == 0 else 1
        ow = 0 if ww % 2 == 0 else 1
        s = src[..., i-hh//2:i+hh//2+oh, j-ww//2:j+ww//2+ow]
        t = tgt[..., i-hh//2:i+hh//2+oh, j-ww//2:j+ww//2+ow]
        if len(cs) == 0: s = s[np.newaxis,...]  # add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis,...]
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
        assert 0 <= axis <= 2
        self.axis = axis
        self.div = div

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        src, tgt = sample
        *cs, _, _, _ = src.shape
        *ct, _, _, _ = tgt.shape
        s = src[0] if len(cs) > 0 else src  # use the first image to determine sampling if multimodal
        idx = np.random.choice(self._valid_idxs(s)[self.axis])
        s = self._get_slice(src, idx)
        t = self._get_slice(tgt, idx)
        if len(cs) == 0: s = s[np.newaxis,...]  # add channel axis if empty
        if len(ct) == 0: t = t[np.newaxis,...]
        return s, t

    def _get_slice(self, img:np.ndarray, idx:int):
        s = img[...,idx,:,:] if self.axis == 0 else \
            img[...,:,idx,:] if self.axis == 1 else \
            img[...,:,:,idx]
        return s

    def _valid_idxs(self, img:np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """ get the set of indices from which to sample (foreground) """
        mask = np.where(img > img.mean() / self.div)  # returns a tuple of length 3
        h, w, d = [np.arange(np.min(m), np.max(m)+1) for m in mask]  # pull out the valid idx ranges
        return h, w, d


class ToTensor:
    """ Convert images in sample to Tensors """
    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[torch.Tensor,torch.Tensor]:
        src, tgt = sample
        if isinstance(src, np.ndarray) and isinstance(tgt, np.ndarray):
            return torch.from_numpy(src), torch.from_numpy(tgt)
        # handle PIL images
        src, tgt = np.asarray(src)[None,...], np.asarray(tgt)[None,...]
        return torch.from_numpy(src), torch.from_numpy(tgt)


class ToFastaiImage:
    """ convert a 2D image to fastai.Image class """
    def __init__(self):
        from fastai.vision import Image
        self.Image = Image

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        x, y = sample
        return self.Image(x), self.Image(y)


class ToPILImage:
    """ convert 2D image to PIL image """
    def __init__(self, mode:str='F'):
        self.toPIL = tv.transforms.ToPILImage(mode)

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        src, tgt = sample
        src, tgt = np.transpose(src,(1,2,0)), np.transpose(tgt,(1,2,0))
        return self.toPIL(src), self.toPIL(tgt)


class RandomAffine(tv.transforms.RandomAffine):
    """ apply random affine transformations to a sample of images """
    def __init__(self, p:float, degrees:float, translate:Optional[float]=None, scale:Optional[float]=None,
                 resample:int=Image.BILINEAR):
        self.p = p
        self.degrees, self.translate, self.scale = (-degrees,degrees), (translate,translate), (1-scale,1+scale)
        self.resample = resample

    def __call__(self, sample:Tuple[PILImage, PILImage]):
        src, tgt = sample
        ret = self.get_params(self.degrees, self.translate, self.scale, None, src.size)
        if self.degrees[1] > 0 and random.random() < self.p:
            src = TF.affine(src, *ret, resample=self.resample, fillcolor=0)
            tgt = TF.affine(tgt, *ret, resample=self.resample, fillcolor=0)
        return src, tgt


class RandomFlip:
    def __init__(self, p:float, vflip:bool=False, hflip:bool=False):
        self.p = p
        self.vflip, self.hflip = vflip, hflip

    def __call__(self, sample:Tuple[PILImage,PILImage]):
        src, tgt = sample
        if self.vflip and random.random() < self.p:
            src, tgt = TF.vflip(src), TF.vflip(tgt)
        if self.hflip and random.random() < self.p:
            src, tgt = TF.hflip(src), TF.hflip(tgt)
        return src, tgt


class RandomGamma:
    """ apply random gamma transformations to a sample of images """
    def __init__(self, p, tfm_y=False, gamma:float=0., gain:float=0.):
        self.p, self.tfm_y = p, tfm_y
        self.gamma, self.gain = (max(1-gamma,0),1+gamma), (max(1-gain,0),1+gain)

    @staticmethod
    def _make_pos(x): return x.min(), x - x.min()

    def _gamma(self, x, gain, gamma):
        is_pos = torch.all(x >= 0)
        if not is_pos: m, x = self._make_pos(x)
        x = gain * x ** gamma
        if not is_pos: x = x + m
        return x

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        src, tgt = sample
        if random.random() < self.p:
            gamma = random.uniform(self.gamma[0], self.gamma[1])
            gain = random.uniform(self.gain[0], self.gain[1])
            src = self._gamma(src, gain, gamma)
            if self.tfm_y: tgt = self._gamma(tgt, gain, gamma)
        return src, tgt
       

class RandomNoise:
    """ add random gaussian noise to a sample of images """
    def __init__(self, p, tfm_x=True, tfm_y=False, std:float=0):
        self.p, self.tfm_x, self.tfm_y, self.std = p, tfm_x, tfm_y, std

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        src, tgt = sample
        if self.std > 0 and random.random() < self.p:
            if self.tfm_x: src = src + torch.randn_like(src).mul(self.std)
            if self.tfm_y: tgt = tgt + torch.randn_like(tgt).mul(self.std)
        return src, tgt


class RandomBlock:
    """ add random blocks of random intensity to a sample of images """
    def __init__(self, p, sz_range, int_range=None, tfm_x=True, tfm_y=False):
        self.p, self.sz, self.int, self.tfm_x, self.tfm_y = p, sz_range, int_range, tfm_x, tfm_y

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        src, tgt = sample
        _, hmax, wmax = src.shape
        mask = np.where(src >= src.mean())
        c = np.random.randint(0, len(mask[1]))  # choose the set of idxs to use
        h, w = [m[c] for m in mask[1:]]  # pull out the chosen idxs (2D)
        s = random.randrange(*self.sz)
        if h+s >= hmax: s -= hmax - h+s
        if w+s >= wmax: s -= wmax - w+s
        if h-s < 0: s -= h-s
        if w-s < 0: s -= w-s
        o = 0 if s % 2 == 0 else 1
        int_range = self.int if self.int is not None else (int(src.min()), int(src.max())+1)
        if random.random() < self.p:
            if self.tfm_x: src[:,h-s//2:h+s//2+o,w-s//2:w+s//2+o] = random.randrange(*int_range)
            if self.tfm_y: tgt[:,h-s//2:h+s//2+o,w-s//2:w+s//2+o] = random.randrange(*int_range)
        return src, tgt
    
 
class AddChannel:
    """ Add empty first dimension to sample """
    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor]:
        src, tgt = sample
        return (src.unsqueeze(0), tgt.unsqueeze(0))


class Normalize:
    """ put data in range of 0 to 1 """
    def __init__(self, scale:float=1):
        self.scale = scale

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        x, y = sample
        x = self.scale * ((x - x.min()) / (x.max() - x.min()))
        y = self.scale * ((y - y.min()) / (y.max() - y.min()))
        return x, y


class Digitize:
    """ digitize a sample of images """
    def __init__(self, tfm_x=False, tfm_y=True, int_range=(1,100), step=1):
        self.tfm_x, self.tfm_y, self.range, self.step = tfm_x, tfm_y, int_range, step

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        src, tgt = sample
        if self.tfm_x: src = np.digitize(src, np.arange(self.range[0], self.range[1], self.step))
        if self.tfm_y: tgt = np.digitize(tgt, np.arange(self.range[0], self.range[1], self.step))
        return src, tgt
    

def get_transforms(p:Union[list,float], tfm_x:bool=True, tfm_y:bool=False, degrees:Optional[float]=0,
                   translate:Optional[float]=None, scale:Optional[float]=None, vflip:bool=False,
                   hflip:bool=False, gamma:Optional[float]=None, gain:float=1, std:float=0,
                   block:Optional[Tuple[int,int]]=None, norm:float=0):
    """ get many desired transforms in a way s.t. can apply to nifti/tiffdatasets """
    if isinstance(p, float): p = [p] * 5
    tfms = []
    if norm > 0:
        tfms.append(Normalize(norm))
    if degrees > 0 or translate is not None or scale is not None:
        tfms.append(ToPILImage())
        tfms.append(RandomAffine(p[0], degrees, translate, scale))
    if vflip or hflip:
        tfms.append(RandomFlip(p[1], vflip, hflip))
    tfms.append(ToTensor())
    if gamma is not None or gain is not None:
        tfms.append(RandomGamma(p[2], tfm_y, gamma, gain))
    if block is not None:
        tfms.append(RandomBlock(p[3], block, tfm_x=tfm_x, tfm_y=tfm_y))
    if std > 0:
        tfms.append(RandomNoise(p[4], tfm_x, tfm_y, std))
    return tfms
