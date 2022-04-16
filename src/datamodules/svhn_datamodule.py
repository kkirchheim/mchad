import logging
from os.path import join
from typing import Optional, Tuple

from torchvision.datasets import CIFAR10, SVHN

from oodtk.transforms import ToUnknown

from .base import MyBaseDataModule

log = logging.getLogger(__name__)


class SVHNDataModule(MyBaseDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_split: Tuple[int, int, int] = (84_289, 5000),
        batch_size: int = 128,
        num_workers: int = 10,
        pin_memory: bool = False,
        data_order_seed: int = None,
        **kwargs,
    ):
        super().__init__(batch_size, num_workers, pin_memory, **kwargs)

        # NOTE: the implementation just dumps some mat files in the root folder, we add an additional directoy here
        #  to make the overall structure more pretty
        self.data_dir = data_dir
        self.data_dir_svhn = join(data_dir, "SVHN")
        self.train_val_split = train_val_split
        # self.dims is returned when you call datamodule.size()
        self.dims = (3, 32, 32)

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        super().prepare_data()
        SVHN(self.data_dir_svhn, split="train", download=True)
        SVHN(self.data_dir_svhn, split="test", download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        super(SVHNDataModule, self).setup(stage)

        self.data_train = SVHN(
            self.data_dir_svhn,
            split="train",
            transform=self.train_trans,
            target_transform=self.target_transform,
        )

        d1 = SVHN(
            self.data_dir_svhn,
            split="test",
            transform=self.test_trans,
            target_transform=self.target_transform,
        )

        d2 = CIFAR10(
            self.data_dir,
            train=False,
            transform=self.test_trans,
            target_transform=ToUnknown(),
        )

        self.data_val = d1 + d2
        self.data_test = d1
