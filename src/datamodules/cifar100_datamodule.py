from torchvision.datasets import CIFAR100
from typing import Optional, Tuple
from torch.utils.data import ConcatDataset, random_split
import numpy as np
from .base import MyBaseDataModule
import logging

from src.osr.ossim import TargetMapping


log = logging.getLogger(__name__)


class CIFAR100DataModule(MyBaseDataModule):

    def __init__(
            self,
            data_dir: str = "data/",
            train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 128,
            num_workers: int = 10,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__(batch_size, num_workers, pin_memory, **kwargs)

        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, 32, 32)

        # TODO: make configurable
        labels = np.random.permutation(range(100))
        train_in = labels[:80]
        train_out = []
        test_out = labels[80:]
        self.mapping = TargetMapping(
            train_in_classes=train_in,
            train_out_classes=train_out,
            test_out_classes=test_out
        )

    @property
    def num_classes(self) -> int:
        return 100

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        log.info(f"Datamodule setup")
        super().setup()

        trainset = CIFAR100(self.data_dir, train=True, transform=self.transforms, target_transform=self.mapping)
        testset = CIFAR100(self.data_dir, train=False, transform=self.transforms, target_transform=self.mapping)
        dataset = ConcatDataset(datasets=[trainset, testset])

        self.data_train, self.data_val, self.data_test = random_split(
            dataset, self.train_val_test_split
        )
