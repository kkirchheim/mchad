import logging
from typing import Optional

import hydra
from pytorch_lightning import LightningDataModule

from .base import MyBaseDataModule

log = logging.getLogger(__name__)


class ToRBG(object):
    """
    Convert Image to RGB, if it is not already.
    """

    def __call__(self, x):
        if x.mode != "RGB":
            return x.convert("RGB")
        return x


class MultiDatamodule(MyBaseDataModule):
    """
    Datamodule that mixes two datasets and treats one as ODD.
    """

    def __init__(
        self,
        module1=None,
        module2=None,
    ):
        super().__init__()
        self.data_module_1: LightningDataModule = hydra.utils.instantiate(
            module1, _recursive_=False, _convert_="partial"
        )
        self.data_module_2: LightningDataModule = hydra.utils.instantiate(
            module2, _recursive_=False, _convert_="partial"
        )

    @property
    def num_classes(self) -> int:
        # TODO
        return self.data_module_1.num_classes

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # NOTE: for this to work, the download option has to be set to true in the configuration
        self.data_module_1.prepare_data()
        self.data_module_2.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """Create instances of both datasets"""
        self.data_module_1.setup(stage)
        self.data_module_2.setup(stage)

    def train_dataloader(self):
        return [
            self.data_module_1.train_dataloader(),
            self.data_module_2.train_dataloader(),
        ]

    def val_dataloader(self):
        return [
            self.data_module_1.val_dataloader(),
            self.data_module_2.val_dataloader(),
        ]

    def test_dataloader(self):
        return [
            self.data_module_1.test_dataloader(),
            self.data_module_2.test_dataloader(),
        ]
