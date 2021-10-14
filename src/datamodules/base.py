from typing import Optional
import torch.random
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import logging
import numpy as np

from src.osr.ossim import TargetMapping

log = logging.getLogger(__name__)


class MyBaseDataModule(LightningDataModule):
    """

    """

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_order_seed: int = None,  # seed used for training data ordering
        data_split_seed: int = None,  # seed used for training data splitting
        ood_classes_val: int = 0,
        ood_classes_test: int = 0,
        height: int = 32,
        width: int = 32
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_order_seed = data_order_seed
        self.data_split_seed = data_split_seed

        # TODO: we could add more data augmentation at this point
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=(height, width))
            ]
        )

        self.target_transform = None

        if ood_classes_val or ood_classes_test:
            log.info(f"OOD classes are set. Sampling Open Set Simulation")
            # select several classes as unknown.
            n_leave_out = ood_classes_val + ood_classes_test
            labels = np.random.permutation(range(self.num_classes))
            train_in = labels[0:self.num_classes - n_leave_out]
            val_out = labels[self.num_classes - n_leave_out: self.num_classes - ood_classes_test]
            test_out = labels[self.num_classes - ood_classes_test:]
            self.target_transform = TargetMapping(
                train_in_classes=train_in,
                train_out_classes=val_out,
                test_out_classes=test_out
            )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        # create a generator used to determine the training split
        self.split_generator = None # torch.Generator()
        if self.data_split_seed:
            log.info(f"Initializing data split seed with {self.data_split_seed}")
            # self.split_generator.manual_seed(self.data_split_seed)
        else:
            log.info(f"Not initializing data split seed")

        # create a generator used to determine the ordering of the data
        self.order_generator = None #  torch.Generator()
        if self.data_order_seed:
            log.info(f"Initializing data ordering seed with {self.data_order_seed}")
            # self.order_generator.manual_seed(self.data_order_seed)
        else:
            log.info(f"Not initializing data ordering seed")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            generator=self.order_generator
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def test_dataloader(self):
        # NOTE: we could also return multiple loaders ...
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

