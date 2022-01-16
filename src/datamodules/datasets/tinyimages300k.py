import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


class TinyImages300k(torch.utils.data.Dataset):
    """
    Cleaned version of the Tiny Images dataset with 300k images.

    """

    def __init__(self, datafile, transform=None):
        self.datafile = datafile

        log.info(f"Loading data from {datafile}")
        self.data = np.load(datafile)
        log.info(f"Shape of dataset: {self.data.shape}")
        self.transform = transform

    def __getitem__(self, index):
        index = index % len(self)

        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, -1

    def __len__(self):
        return self.data.shape[0]
