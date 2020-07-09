#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_dataset

test submodules for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Oct 24, 2018
"""

import os
import shutil
import tempfile
import unittest

import torchvision.transforms as torch_tfms

from niftidataset import *


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
        nii = glob_imgs(self.nii_dir)[0]
        tif = os.path.join(self.tif_dir, 'test.tif')
        path, base, ext = split_filename(nii)
        for i in range(4):
            shutil.copy(nii, os.path.join(self.train_dir, base + str(i) + ext))
            shutil.copy(tif, os.path.join(self.train_dir, '1', base + str(i) + '.tif'))
            shutil.copy(tif, os.path.join(self.train_dir, '2', base + str(i) + '.tif'))

    def test_niftidataset_2d(self):
        composed = torch_tfms.Compose([RandomCrop2D(10, 0),
                                       ToTensor(),
                                       FixIntensityRange()])
        myds = NiftiDataset.setup_from_dir(self.train_dir, self.train_dir, composed)
        self.assertEqual(myds[0][0].shape, (1,10,10))

    def test_niftidataset_2d_slice(self):
        composed = torch_tfms.Compose([RandomSlice(0),
                                       ToTensor()])
        myds = NiftiDataset.setup_from_dir(self.train_dir, self.train_dir, composed)
        self.assertEqual(myds[0][0].shape, (1,64,64))

    def test_niftidataset_3d(self):
        composed = torch_tfms.Compose([RandomCrop3D(10),
                                       ToTensor()])
        myds = NiftiDataset.setup_from_dir(self.train_dir, self.train_dir, composed)
        self.assertEqual(myds[0][0].shape, (1,10,10,10))

    def test_niftidataset_preload(self):
        composed = torch_tfms.Compose([RandomCrop3D(10),
                                       ToTensor(),
                                       AddChannel()])
        myds = NiftiDataset.setup_from_dir(self.train_dir, self.train_dir, composed, preload=True)
        self.assertEqual(myds[0][0].shape, (1,1,10,10,10))

    def test_multimodalnifti_2d(self):
        composed = torch_tfms.Compose([RandomCrop2D(10, 0),
                                       ToTensor(),
                                       FixIntensityRange()])
        sd, td = [self.train_dir] * 3, [self.train_dir]
        myds = MultimodalNiftiDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (3,10,10))
        self.assertEqual(myds[0][1].shape, (1,10,10))

    def test_multimodalnifti_slice(self):
       composed = torch_tfms.Compose([RandomSlice(0),
                                      ToTensor()])
       sd, td = [self.train_dir] * 2, [self.train_dir] * 4
       myds = MultimodalNiftiDataset.setup_from_dir(sd, td, composed)
       self.assertEqual(myds[0][0].shape, (2,64,64))
       self.assertEqual(myds[0][1].shape, (4,64,64))

    def test_multimodalnifti_3d(self):
        composed = torch_tfms.Compose([RandomCrop3D(10),
                                       ToTensor()])
        sd, td = [self.train_dir] * 3, [self.train_dir] * 2
        myds = MultimodalNiftiDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (3,10,10,10))
        self.assertEqual(myds[0][1].shape, (2,10,10,10))

    def test_multimodalnifti_2p5D(self):
        composed = torch_tfms.Compose([ToTensor()])
        sd, td = [self.train_dir] * 3, [self.train_dir] * 2
        myds = MultimodalNifti2p5DDataset.setup_from_dir(sd, td, composed, axis=0)
        self.assertEqual(myds[0][0].shape, (3*51,64,64))
        self.assertEqual(myds[0][1].shape, (2*51,64,64))

    def test_multimodalnifti_preload(self):
        composed = torch_tfms.Compose([RandomCrop3D(10),
                                       ToTensor()])
        sd, td = [self.train_dir] * 3, [self.train_dir] * 2
        myds = MultimodalNiftiDataset.setup_from_dir(sd, td, composed, preload=True)
        self.assertEqual(myds[0][0].shape, (3,10,10,10))
        self.assertEqual(myds[0][1].shape, (2,10,10,10))

    def test_multimodaltiff(self):
        composed = torch_tfms.Compose([ToTensor()])
        sd, td = [self.train_dir+'/1/'] * 3, [self.train_dir+'/2/'] * 2
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (3,256,256))
        self.assertEqual(myds[0][1].shape, (2,256,256))

    def test_multimodaltiff_preload(self):
        composed = torch_tfms.Compose([ToTensor()])
        sd, td = [self.train_dir+'/1/'] * 3, [self.train_dir+'/2/'] * 2
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed, preload=True)
        self.assertEqual(myds[0][0].shape, (3,256,256))
        self.assertEqual(myds[0][1].shape, (2,256,256))

    def test_multimodaltiff_seg(self):
        composed = torch_tfms.Compose([ToTensor()])
        sd, td = [self.train_dir + '/1/'] * 3, [self.train_dir + '/2/']
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed, segmentation=True)
        self.assertEqual(myds[0][0].shape, (3, 256, 256))
        self.assertEqual(myds[0][1].shape, (256, 256))

    def test_multimodaltiff_crop(self):
        composed = torch_tfms.Compose([ToTensor(), RandomCrop(32)])
        sd, td = [self.train_dir+'/1/'] * 3, [self.train_dir+'/2/'] * 2
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (3,32,32))
        self.assertEqual(myds[0][1].shape, (2,32,32))

    def test_aug_affine_2d(self):
        composed = torch_tfms.Compose([ToPILImage(),
                                       RandomAffine(1, 15, 0.1, 0.1),
                                       ToTensor()])
        sd, td = [self.train_dir+'/1/'], [self.train_dir+'/2/']
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (1,256,256))
        self.assertEqual(myds[0][1].shape, (1,256,256))

    def test_aug_flip_2d(self):
        composed = torch_tfms.Compose([ToPILImage(),
                                       RandomFlip(1, True, True),
                                       ToTensor()])
        sd, td = [self.train_dir+'/1/'], [self.train_dir+'/2/']
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (1,256,256))
        self.assertEqual(myds[0][1].shape, (1,256,256))

    def test_aug_intensity_2d(self):
        composed = torch_tfms.Compose([ToTensor(),
                                       RandomGamma(1, True, 0.1, 0.1)])
        sd, td = [self.train_dir+'/1/'], [self.train_dir+'/2/']
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (1,256,256))
        self.assertEqual(myds[0][1].shape, (1,256,256))

    def test_aug_noise_2d(self):
        composed = torch_tfms.Compose([ToTensor(),
                                       RandomNoise(1, True, True, 1)])
        sd, td = [self.train_dir+'/1/'], [self.train_dir+'/2/']
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (1,256,256))
        self.assertEqual(myds[0][1].shape, (1,256,256))

    def test_aug_digitize_2d(self):
        composed = torch_tfms.Compose([Digitize(True, True, (1,100), 1),
                                       ToTensor()])
        sd, td = [self.train_dir+'/1/'], [self.train_dir+'/2/']
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (1,256,256))
        self.assertEqual(myds[0][1].shape, (1,256,256))

    def test_aug_block_2d(self):
        composed = torch_tfms.Compose([ToTensor(),
                                       RandomBlock(1, (1,100))])
        sd, td = [self.train_dir+'/1/'], [self.train_dir+'/2/']
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (1,256,256))
        self.assertEqual(myds[0][1].shape, (1,256,256))

    def test_aug_block_3d(self):
        composed = torch_tfms.Compose([RandomCrop3D(10),
                                       ToTensor(),
                                       RandomBlock(1, (1,4), thresh=0, is_3d=True)])
        myds = NiftiDataset.setup_from_dir(self.train_dir, self.train_dir, composed)
        self.assertEqual(myds[0][0].shape, (1,10,10,10))

    def test_aug_block_3d_oblong(self):
        composed = torch_tfms.Compose([RandomCrop3D(10),
                                       ToTensor(),
                                       RandomBlock(1, ((1,4),(1,2),(1,3)), thresh=0, is_3d=True)])
        myds = NiftiDataset.setup_from_dir(self.train_dir, self.train_dir, composed)
        self.assertEqual(myds[0][0].shape, (1,10,10,10))

    def test_get_transform_2d(self):
        composed = torch_tfms.Compose(get_transforms([1,1,1,1,1],True,True,15,0.1,0.1,True,True,0.1,0.1,1,(3,4),None,False,(1,),(1,)))
        sd, td = [self.train_dir+'/1/'], [self.train_dir+'/2/']
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (1,256,256))
        self.assertEqual(myds[0][1].shape, (1,256,256))

    def test_get_transform_2d_multi(self):
        composed = torch_tfms.Compose(get_transforms([1,1,1,1,1],True,True,15,0.1,0.1,True,True,0.1,0.1,1,(3,4),None,False,(1,),(1,)))
        sd, td = [self.train_dir+'/1/'] * 2, [self.train_dir+'/2/']
        myds = MultimodalImageDataset.setup_from_dir(sd, td, composed, segmentation=True)
        self.assertEqual(myds[0][0].shape, (2,256,256))
        self.assertEqual(myds[0][1].shape, (256,256))

    def test_get_transform_3d(self):
        composed = torch_tfms.Compose(get_transforms([0,0,1,1,1],True,True,0,0,0,False,False,0.1,0.1,1,(3,4),None,True,(1,),(1,)))
        sd, td = [self.train_dir], [self.train_dir]
        myds = MultimodalNiftiDataset.setup_from_dir(sd, td, composed)
        self.assertEqual(myds[0][0].shape, (1, 51, 64, 64))
        self.assertEqual(myds[0][1].shape, (1, 51, 64, 64))

    def test_train_val_split(self):
        import torch
        composed = torch_tfms.Compose([ToTensor()])
        tr, val = train_val_split(self.train_dir, self.train_dir,
                                  valid_pct=0.25, transform=composed)
        self.assertEqual(len(tr), 3)
        self.assertEqual(len(val), 1)
        self.assertEqual(torch.all(torch.eq(val[0][0], tr[0][0])), torch.tensor(True))

    def test_trim_intensity(self):
        import numpy as np
        composed = torch_tfms.Compose([ToTensor()])
        src, tgt = NiftiDataset.setup_from_dir(self.train_dir, self.train_dir, composed)[0]
        maxim = np.max(src.numpy())
        minim = np.min(src.numpy())
        composed2 = torch_tfms.Compose([ToTensor(),
                                        TrimIntensity(max_val=maxim-1000, min_val=minim-1000)])
        src, tgt = NiftiDataset.setup_from_dir(self.train_dir, self.train_dir, composed2)[0]
        self.assertEqual(np.max(src.numpy()), 1.)
        self.assertTrue(np.min(src.numpy()) > -1.)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()
