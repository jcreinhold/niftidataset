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
           'FixIntensityRange',
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


class BaseTransform:
    def __repr__(self): return f'{self.__class__.__name__}'


class CropBase(BaseTransform):
    """ base class for crop transform """

    def __init__(self, out_dim:int, output_size:Union[tuple,int,list], threshold:Optional[float]=None,
                 pct:Tuple[float,float]=(0.,1.), axis=0):
        """ provide the common functionality for RandomCrop2D and RandomCrop3D """
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size,)
            for _ in range(out_dim - 1):
                self.output_size += (output_size,)
        else:
            assert len(output_size) == out_dim
            self.output_size = output_size
        self.out_dim = out_dim
        self.thresh = threshold
        self.pct = pct
        self.axis = axis

    def _get_sample_idxs(self, img: np.ndarray) -> Tuple[int, int, int]:
        """ get the set of indices from which to sample (foreground) """
        mask = np.where(img >= (img.mean() if self.thresh is None else self.thresh))  # returns a tuple of length 3
        c = np.random.randint(0, len(mask[0]))  # choose the set of idxs to use
        h, w, d = [m[c] for m in mask]  # pull out the chosen idxs
        return h, w, d

    def _offset_by_pct(self, h, w, d):
        s = (h, w, d)
        hml = wml = dml = 0
        hmh = wmh = dmh = 0
        i0, i1 = int(s[self.axis] * self.pct[0]), int(s[self.axis] * (1. - self.pct[1]))
        if self.axis == 0:
            hml += i0;
            hmh += i1
        elif self.axis == 1:
            wml += i0;
            wmh += i1
        else:
            dml += i0;
            dmh += i1
        return (hml, wml, dml), (hmh, wmh, dmh)

    def __repr__(self):
        s = '{name}(output_size={output_size}, threshold={thresh})'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)


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

    def __init__(self, output_size:Union[int,tuple,list], axis:Optional[int]=0,
                 include_neighbors:bool=False, threshold:Optional[float]=None) -> None:
        if axis is not None:
            assert 0 <= axis <= 2
        super().__init__(2, output_size, threshold)
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

    def __init__(self, output_size:Union[tuple,int,list], threshold:Optional[float]=None,
                pct:Tuple[float,float]=(0.,1.), axis=0):
        super().__init__(3, output_size, threshold, pct, axis)

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        src, tgt = sample
        *cs, h, w, d = src.shape
        *ct, _, _, _ = tgt.shape
        hh, ww, dd = self.output_size
        (hml, wml, dml), (hmh, wmh, dmh) = self._offset_by_pct(h,w,d)
        max_idxs = (h-hmh-hh//2, w-wmh-ww//2, d-dmh-dd//2)
        min_idxs = (hml+hh//2, wml+ww//2, dml+dd//2)
        s = src[0] if len(cs) > 0 else src  # use the first image to determine sampling if multimodal
        s_idxs = self._get_sample_idxs(s)
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

    def __init__(self, output_size:Union[tuple,int], threshold:Optional[float]=None):
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size
        self.thresh = threshold

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[np.ndarray,np.ndarray]:
        src, tgt = sample
        *cs, h, w = src.shape
        *ct, _, _ = tgt.shape
        hh, ww = self.output_size
        max_idxs = (h-hh//2, w-ww//2)
        min_idxs = (hh//2, ww//2)
        s = src[0] if len(cs) > 0 else src  # use the first image to determine sampling if multimodal
        mask = np.where(s >= (s.mean() if self.thresh is None else self.thresh))
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

    def __repr__(self):
        s = '{name}(output_size={output_size}, threshold={thresh})'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)


class RandomSlice(BaseTransform):
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


class ToTensor(BaseTransform):
    """ Convert images in sample to Tensors """
    def __init__(self, color=False):
        self.color = color

    def __call__(self, sample:Tuple[np.ndarray,np.ndarray]) -> Tuple[torch.Tensor,torch.Tensor]:
        src, tgt = sample
        if isinstance(src, np.ndarray) and isinstance(tgt, np.ndarray):
            return torch.from_numpy(src), torch.from_numpy(tgt)
        if isinstance(src, list): src = np.stack(src)
        if isinstance(tgt, list): src = np.stack(tgt)
        # handle PIL images
        src, tgt = np.asarray(src), np.asarray(tgt)
        if src.ndim == 3 and self.color: src = src.transpose((2,0,1)).astype(np.float32)
        if tgt.ndim == 3 and self.color: tgt = tgt.transpose((2,0,1)).astype(np.float32)
        if src.ndim == 2: src = src[None,...] # add channel dimension
        if tgt.ndim == 2: tgt = tgt[None,...]
        return torch.from_numpy(src), torch.from_numpy(tgt)


class ToFastaiImage(BaseTransform):
    """ convert a 2D image to fastai.Image class """
    def __init__(self):
        from fastai.vision import Image
        self.Image = Image

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        x, y = sample
        return self.Image(x), self.Image(y)


class ToPILImage(BaseTransform):
    """ convert 2D image to PIL image """
    def __init__(self, color=False):
        self.color = color

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        src, tgt = sample
        src, tgt = np.squeeze(src), np.squeeze(tgt)
        if src.ndim == 3 and self.color:
            src = Image.fromarray(src.transpose((1,2,0)).astype(np.uint8))
        elif src.ndim == 2:
            src = Image.fromarray(src)
        else:
            src = [Image.fromarray(s) for s in src]
        if tgt.ndim == 3 and self.color:
            tgt = Image.fromarray(tgt.transpose((1,2,0)).astype(np.uint8))
        elif tgt.ndim == 2:
            tgt = Image.fromarray(tgt)
        else:
            tgt = [Image.fromarray(t) for t in tgt]
        return src, tgt


class RandomAffine(tv.transforms.RandomAffine):
    """ apply random affine transformations to a sample of images """
    def __init__(self, p:float, degrees:float, translate:float=0, scale:float=0, resample:int=Image.BILINEAR):
        self.p = p
        self.degrees, self.translate, self.scale = (-degrees,degrees), (translate,translate), (1-scale,1+scale)
        self.shear, self.fillcolor = None, 0
        self.resample = resample

    def affine(self, x, params):
        return TF.affine(x, *params, resample=self.resample, fillcolor=0)

    def __call__(self, sample:Tuple[PILImage, PILImage]):
        src, tgt = sample
        ret = self.get_params(self.degrees, self.translate, self.scale, None, tgt.size)
        if self.degrees[1] > 0 and random.random() < self.p:
            if not isinstance(src, list):
                src = self.affine(src, ret)
            else:
                src = [self.affine(s, ret) for s in src]
            if not isinstance(tgt, list):
                tgt = self.affine(tgt, ret)
            else:
                tgt = [self.affine(t, ret) for t in tgt]
        return src, tgt


class RandomFlip:
    def __init__(self, p:float, vflip:bool=False, hflip:bool=False):
        self.p = p
        self.vflip, self.hflip = vflip, hflip

    def __call__(self, sample:Tuple[PILImage,PILImage]):
        src, tgt = sample
        if self.vflip and random.random() < self.p:
            if not isinstance(src, list):
                src = TF.vflip(src)
            else:
                src = [TF.vflip(s) for s in src]
            if not isinstance(tgt, list):
                tgt = TF.vflip(tgt)
            else:
                tgt = [TF.vflip(t) for t in tgt]
        if self.hflip and random.random() < self.p:
            if not isinstance(src, list):
                src = TF.hflip(src)
            else:
                src = [TF.hflip(s) for s in src]
            if not isinstance(tgt, list):
                tgt = TF.hflip(tgt)
            else:
                tgt = [TF.hflip(t) for t in tgt]
        return src, tgt

    def __repr__(self):
        s = '{name}(p={p}, vflip={vflip}, hflip={hflip})'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)


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

    def __repr__(self):
        s = '{name}(p={p}, tfm_y={tfm_y}, gamma={gamma}, gain={gain})'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)


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

    def __repr__(self):
        s = '{name}(p={p}, tfm_x={tfm_x}, tfm_y={tfm_y}, std={std})'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)


class RandomBlock:
    """ add random blocks of random intensity to a sample of images """
    def __init__(self, p, sz_range, thresh=None, int_range=None, tfm_x=True, tfm_y=False, is_3d=False):
        self.p, self.int, self.tfm_x, self.tfm_y = p, int_range, tfm_x, tfm_y
        self.sz = sz_range if all([isinstance(szr, (tuple,list)) for szr in sz_range]) else \
                  (sz_range, sz_range, sz_range) if is_3d else (sz_range, sz_range)
        self.thresh = thresh
        self.is_3d = is_3d

    def block2d(self, src, tgt):
        _, hmax, wmax = src.shape
        mask = np.where(src >= (src.mean() if self.thresh is None else self.thresh))
        c = np.random.randint(0, len(mask[1]))  # choose the set of idxs to use
        h, w = [m[c] for m in mask[1:]]  # pull out the chosen idxs (2D)
        sh, sw = random.randrange(*self.sz[0]), random.randrange(*self.sz[1])
        oh = 0 if sh % 2 == 0 else 1
        ow = 0 if sw % 2 == 0 else 1
        if h+(sh//2)+oh >= hmax: h = hmax - (sh//2) - oh
        if w+(sw//2)+ow >= wmax: w = wmax - (sw//2) - ow
        if h-(sh//2) < 0: h = sh//2
        if w-(sw//2) < 0: w = sw//2
        int_range = self.int if self.int is not None else (src.min(), src.max()+1)
        if random.random() < self.p:
            if self.tfm_x: src[:,h-sh//2:h+sh//2+oh,w-sw//2:w+sw//2+ow] = np.random.uniform(*int_range)
            if self.tfm_y: tgt[:,h-sh//2:h+sh//2+oh,w-sw//2:w+sw//2+ow] = np.random.uniform(*int_range)
        return src, tgt

    def block3d(self, src, tgt):
        _, hmax, wmax, dmax = src.shape
        mask = np.where(src >= (src.mean() if self.thresh is None else self.thresh))
        c = np.random.randint(0, len(mask[1]))  # choose the set of idxs to use
        h, w, d = [m[c] for m in mask[1:]]  # pull out the chosen idxs (2D)
        sh, sw, sd = random.randrange(*self.sz[0]), random.randrange(*self.sz[1]), random.randrange(*self.sz[2])
        oh = 0 if sh % 2 == 0 else 1
        ow = 0 if sw % 2 == 0 else 1
        od = 0 if sd % 2 == 0 else 1
        if h+(sh//2)+oh >= hmax: h = hmax - (sh//2) - oh
        if w+(sw//2)+ow >= wmax: w = wmax - (sw//2) - ow
        if d+(sd//2)+od >= dmax: d = dmax - (sd//2) - od
        if h-(sh//2) < 0: h = sh//2
        if w-(sw//2) < 0: w = sw//2
        if d-(sd//2) < 0: d = sd//2
        int_range = self.int if self.int is not None else (src.min(), src.max()+1)
        if isinstance(src, torch.Tensor): src, tgt = src.clone(), tgt.clone()
        if random.random() < self.p:
            if self.tfm_x: src[:,h-sh//2:h+sh//2+oh,w-sw//2:w+sw//2+ow,d-sd//2:d+sd//2+od] = np.random.uniform(*int_range)
            if self.tfm_y: tgt[:,h-sh//2:h+sh//2+oh,w-sw//2:w+sw//2+ow,d-sd//2:d+sd//2+od] = np.random.uniform(*int_range)
        return src, tgt

    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        src, tgt = sample
        src, tgt = self.block2d(src, tgt) if not self.is_3d else self.block3d(src, tgt)
        return src, tgt

    def __repr__(self):
        s = '{name}(p={p}, sz={sz}, int_range={int}, thresh={thresh}, tfm_x={tfm_x}, tfm_y={tfm_y}, is_3d={is_3d})'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)

 
class AddChannel:
    """ Add empty first dimension to sample """
    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor]:
        src, tgt = sample
        return (src.unsqueeze(0), tgt.unsqueeze(0))


class FixIntensityRange:
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


def normalize3d(tensor, mean, std, inplace=False):
    """
    normalize a 3d tensor

    Args:
        tensor (Tensor): Tensor image of size (C, H, W, D) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not inplace:
        tensor = tensor.clone()

    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return tensor


class Normalize:
    """ implement a normalize function for input two images """
    def __init__(self, mean, std, tfm_x=True, tfm_y=False, is_3d=False):
        self.mean = mean
        self.std = std
        self.tfm_x = tfm_x
        self.tfm_y = tfm_y
        self.is_3d = is_3d
        
    def __call__(self, sample:Tuple[torch.Tensor,torch.Tensor]):
        src, tgt = sample
        norm = normalize3d if self.is_3d else tv.transforms.functional.normalize
        if self.tfm_x: src = norm(src, self.mean, self.std)
        if self.tfm_y: tgt = norm(tgt, self.mean, self.std)
        return src, tgt

    def __repr__(self):
        s = '{name}(mean={mean}, std={std}, tfm_x={tfm_x}, tfm_y={tfm_y}, is_3d={is_3d})'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)

    
def get_transforms(p:Union[list,float], tfm_x:bool=True, tfm_y:bool=False, degrees:float=0,
                   translate:float=None, scale:float=None, vflip:bool=False, hflip:bool=False,
                   gamma:float=0, gain:float=0, noise_pwr:float=0, block:Optional[Tuple[int,int]]=None,
                   thresh:Optional[float]=None, is_3d:bool=False,
                   mean:Optional[Tuple[float]]=None, std:Optional[Tuple[float]]=None,
                   color:bool=False):
    """ get many desired transforms in a way s.t. can apply to nifti/tiffdatasets """
    if isinstance(p, float): p = [p] * 5
    tfms = []
    do_affine = p[0] > 0 and (degrees > 0 or translate > 0 or scale > 0)
    do_flip = p[1] > 0 and (vflip or hflip)
    if do_affine or do_flip:
        tfms.append(ToPILImage(color=color))
    if do_affine:
        tfms.append(RandomAffine(p[0], degrees, translate, scale))
    if do_flip:
        tfms.append(RandomFlip(p[1], vflip, hflip))
    tfms.append(ToTensor(color))
    if p[2] > 0 and (gamma is not None or gain is not None):
        tfms.append(RandomGamma(p[2], tfm_y, gamma, gain))
    if p[3] > 0 and (block is not None):
        tfms.append(RandomBlock(p[3], block, thresh=thresh, tfm_x=tfm_x, tfm_y=tfm_y, is_3d=is_3d))
    if p[4] > 0 and (noise_pwr > 0):
        tfms.append(RandomNoise(p[4], tfm_x, tfm_y, noise_pwr))
    if mean is not None and std is not None:
        tfms.append(Normalize(mean, std, tfm_x, tfm_y, is_3d))
    return tfms
