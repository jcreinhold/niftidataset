#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_utilities

test the functions located in utilities submodule for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: 01, 2018
"""

from functools import partial
import os
from pathlib import PosixPath
import shutil
import tempfile
import unittest

import torchvision.transforms as torch_tfms

from niftidataset import *

try:
    import fastai
except ImportError:
    fastai = None


class TestUtilities(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.nii_dir = os.path.join(wd, 'test_data', 'nii')
        self.tif_dir = os.path.join(wd, 'test_data', 'tif')
        self.out_dir = tempfile.mkdtemp()
        self.train_dir = os.path.join(self.out_dir, 'train')
        os.mkdir(self.train_dir)
        os.mkdir(os.path.join(self.train_dir, '1'))
        os.mkdir(os.path.join(self.train_dir, '2'))
        nii = glob_nii(self.nii_dir)[0]
        tif = os.path.join(self.tif_dir, 'test.tif')
        path, base, ext = split_filename(nii)
        for i in range(4):
            shutil.copy(nii, os.path.join(self.train_dir, base + str(i) + ext))
            shutil.copy(tif, os.path.join(self.train_dir, '1', base + str(i) + '.tif'))
            shutil.copy(tif, os.path.join(self.train_dir, '2', base + str(i) + '.tif'))

    def test_niftidataset_2d(self):
        composed = torch_tfms.Compose([RandomCrop2D(10, 0),
                                       ToTensor(),
                                       Normalize()])
        myds = NiftiDataset(self.train_dir, self.train_dir, composed)
        self.assertEqual(myds[0][0].shape, (1,10,10))

    def test_niftidataset_2d_slice(self):
        composed = torch_tfms.Compose([RandomSlice(0),
                                       ToTensor(),
                                       AddChannel()])
        myds = NiftiDataset(self.train_dir, self.train_dir, composed)
        self.assertEqual(myds[0][0].shape, (1,64,64))

    def test_niftidataset_3d(self):
        composed = torch_tfms.Compose([RandomCrop3D(10),
                                       ToTensor(),
                                       AddChannel()])
        myds = NiftiDataset(self.train_dir, self.train_dir, composed)
        self.assertEqual(myds[0][0].shape, (1,1,10,10,10))

    def test_niftidataset_preload(self):
        composed = torch_tfms.Compose([RandomCrop3D(10),
                                       ToTensor(),
                                       AddChannel()])
        myds = NiftiDataset(self.train_dir, self.train_dir, composed, preload=True)
        self.assertEqual(myds[0][0].shape, (1,1,10,10,10))

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_niftidataset_2d_fastai(self):
        composed = torch_tfms.Compose([RandomCrop2D(10, 0),
                                       ToTensor(),
                                       ToFastaiImage()])
        myds = NiftiDataset(self.train_dir, self.train_dir, composed)
        self.assertEqual(myds[0][0].shape, (1,10,10))

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_niidatabunch_2d(self):
        from niftidataset.fastai import get_slice, niidatabunch
        tfms = [get_slice()]
        myds = niidatabunch(self.train_dir, self.train_dir, tfms=tfms, split=0.5)
        self.assertEqual(myds.train_ds[0][0].shape, (1,64,64))

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_niidatabunch_3d(self):
        from niftidataset.fastai import get_patch3d, niidatabunch
        tfms = [get_patch3d(ps=10)]
        myds = niidatabunch(self.train_dir, self.train_dir, tfms=tfms, split=0.5)
        self.assertEqual(myds.train_ds[0][0].shape, (1,10,10,10))

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_niidatabunch_valid_dir(self):
        from niftidataset.fastai import get_slice, niidatabunch
        tfms = [get_slice()]
        myds = niidatabunch(self.train_dir, self.train_dir, tfms=tfms, split=0.5,
                            val_src_dir=self.train_dir, val_tgt_dir=self.train_dir)
        self.assertEqual(myds.train_ds[0][0].shape, (1,64,64))
        self.assertEqual(myds.valid_ds[0][0].shape, (1,64,64))

    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_tifftuplelist(self):
        from niftidataset.fastai import TIFFTupleList
        data = (TIFFTupleList.from_folders(PosixPath(self.train_dir), '1', '2', extensions=('.tif'))
                .split_by_idx([])
                .label_const(0.)
                .transform()
                .databunch(bs=4))
        self.assertEqual(data.train_ds[0][0].data[0].shape, (1,256,256))
        
    @unittest.skipIf(fastai is None, "fastai is not installed on this system")
    def test_tiffdatabunch(self):
        from niftidataset.fastai import tiffdatabunch
        tfms = []
        myds = tiffdatabunch(self.train_dir+'/1/', self.train_dir+'/2/', tfms=tfms, split=0.5)
        self.assertEqual(myds.train_ds[0][0].shape, (1,256,256))
        self.assertEqual(myds.valid_ds[0][0].shape, (1,256,256))

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()
