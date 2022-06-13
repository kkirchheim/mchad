import logging
from typing import Optional

import hydra
from torch.utils.data import ConcatDataset, random_split
from torchvision.transforms import transforms

from .base import MyBaseDataModule

log = logging.getLogger(__name__)


def return_neg_1k(*args, **kwargs):
    return -1000


class OODDataModule(MyBaseDataModule):
    """
    Datamodule that mixes two datasets and treats one as ODD.
    """

    def __init__(
        self,
        data_in=None,
        data_out=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset_in = data_in
        self.dataset_ood = data_out

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        super().prepare_data()
        # NOTE: for this to work, the download option has to be set to true in the configuration
        hydra.utils.instantiate(self.dataset_in)
        hydra.utils.instantiate(self.dataset_ood)

    def setup(self, stage: Optional[str] = None):
        """Create instances of both datasets"""
        super().setup(stage)
        log.info(f"Instantiating IN {self.dataset_in}")
        dataset_in = hydra.utils.instantiate(self.dataset_in)
        log.info(f"Instantiating OOD {self.dataset_ood}")
        dataset_ood = hydra.utils.instantiate(self.dataset_ood)

        # manually set transformations
        dataset_in.transform = self.test_trans
        dataset_ood.transform = self.test_trans

        # set ood label to fixed value
        dataset_ood.target_transform = transforms.Lambda(return_neg_1k)

        self.data_test = ConcatDataset(datasets=[dataset_in, dataset_ood])
        log.info(f"Created mixed OOD dataset with length {len(self.data_test)}")


class SingleOODDataModule(MyBaseDataModule):
    """
    Datamodule that served a single dataset as OOD
    """

    def __init__(self, dataset=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_ood = dataset

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        super().prepare_data()
        # NOTE: for this to work, the download option has to be set to true in the configuration
        hydra.utils.instantiate(self.dataset_ood)

    def setup(self, stage: Optional[str] = None):
        """Create instances of both datasets"""
        super(SingleOODDataModule, self).setup(stage)
        dataset = hydra.utils.instantiate(self.dataset_ood)

        # TODO:
        n1 = int(0.9 * len(dataset))
        n2 = len(dataset) - n1
        self.data_train, self.data_val = random_split(dataset, [n1, n2])

        # ############################################
        from pytorch_ood.dataset.img import LSUNCrop

        self.data_val = LSUNCrop(
            root="data",
            download=True,
            transform=self.test_trans,
            target_transform=transforms.Lambda(return_neg_1k),
        )
        # ############################################

        # DIRTY FIX
        dataset.transform = self.test_trans
        dataset.target_transform = transforms.Lambda(return_neg_1k)
        # self.data_train.transform = self.train_trans
        # self.data_val.transform = self.test_trans

        # set ood label to fixed value
        # self.data_train.target_transform = transforms.Lambda(return_neg_1k)
        # self.data_val.target_transform = transforms.Lambda(return_neg_1k)

        log.info(f"Created OOD dataset with length {len(self.data_train)}")
