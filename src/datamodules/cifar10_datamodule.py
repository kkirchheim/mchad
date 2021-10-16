from torchvision.datasets import CIFAR10
from typing import Optional, Tuple
from torch.utils.data import ConcatDataset, random_split
from .base import MyBaseDataModule
import logging


log = logging.getLogger(__name__)


class CIFAR10DataModule(MyBaseDataModule):

    def __init__(
            self,
            data_dir: str = "data/",
            train_val_split: Tuple[int, int, int] = (45_000, 5_000),
            batch_size: int = 128,
            num_workers: int = 10,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__(batch_size, num_workers, pin_memory, **kwargs)

        self.data_dir = data_dir
        self.train_val_split = train_val_split
        # self.dims is returned when you call datamodule.size()
        self.dims = (3, 32, 32)

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        super(CIFAR10DataModule, self).setup(stage)
        train_set = CIFAR10(
            self.data_dir,
            train=True,
            transform=self.train_trans,
            target_transform=self.target_transform
        )
        # self.data_train, self.data_val = random_split(train_set, self.train_val_split, generator=self.split_generator)
        self.data_train = train_set

        self.data_test = CIFAR10(
            self.data_dir,
            train=False,
            transform=self.test_trans,
            target_transform=self.target_transform
        )

        self.data_val = self.data_test
