"""
Tiny Images dataset with 80kk images scraped from the web.

While its use is discouraged, it is still commonly used for pretraining.

Dataset is available here: https://archive.org/details/80-million-tiny-images-2-of-2
Code taken from https://github.com/wetliu/energy_ood/blob/master/utils/tinyimages_80mn_loader.py

"""
import numpy as np
import torch


class TinyImages(torch.utils.data.Dataset):

    def __init__(self, datafile, cifar_index_file, transform=None, exclude_cifar=True):
        self.datafile = datafile
        self.cifar_index_file = cifar_index_file

        data_file = open(self.datafile, "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0  # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open(self.cifar_index_file, 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302017)

        while True:
            try:
                img = self.load_image(index)
                if self.transform is not None:
                    img = self.transform(img)

                return img, 0  # 0 is the class
            except:
                index = np.random.randint(79302017)

                if self.exclude_cifar:
                    while self.in_cifar(index):
                        index = np.random.randint(79302017)

    def __len__(self):
        return 79302017