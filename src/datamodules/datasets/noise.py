#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Noise Dataset
"""
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset


class GaussianNoise(VisionDataset):
    """
    Dataset that outputs gaussian noise only
    """

    def __init__(
        self,
        samples,
        size=(224, 224, 3),
        transform=None,
        target_transform=None,
        loc=128,
        scale=128,
        **kwargs
    ):
        self.size = size
        self.num = samples
        self.transform = transform
        self.target_transform = target_transform
        self.loc = 128
        self.scale = 128

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        img = np.random.normal(loc=self.loc, scale=self.scale, size=self.size)

        # if image has one channel, drop channel dimension for pillow
        if img.shape[2] == 1:
            img = img.reshape((img.shape[0], img.shape[1]))

        img = np.clip(img, 0, 255).astype("uint8")

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        target = 0

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class UniformNoise(VisionDataset):
    """
    Dataset that outputs gaussian noise only
    """

    def __init__(
        self,
        samples,
        size=(224, 224, 3),
        transform=None,
        target_transform=None,
        **kwargs
    ):
        self.size = size
        self.num = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        img = np.random.uniform(low=0, high=255, size=self.size).astype(dtype=np.uint8)

        # if image has one channel, drop channel dimension for pillow
        if img.shape[2] == 1:
            img = img.reshape((img.shape[0], img.shape[1]))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        target = 0

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
