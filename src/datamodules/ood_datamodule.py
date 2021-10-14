from typing import Optional, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import transforms
import hydra
import logging


log = logging.getLogger(__name__)


class ToRBG(object):
    """
    Convert Image to RGB, if it is not already.
    """

    def __call__(self, x):
        if x.mode != "RGB":
            return x.convert("RGB")
        return x


class OODDataModule(LightningDataModule):
    """
    Datamodule that mixes two datasets and treats one as ODD.
    """

    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_split: Tuple[int, int, int] = (50_000, 10_000),
        data_in=None,
        data_out=None,
        height: int = 32,
        width: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_split = train_val_split

        self.transforms = transforms.Compose(
            [
                ToRBG(),
                transforms.ToTensor(),
                transforms.Resize(size=(height, width))
            ]
        )

        # self.dims is returned when you call datamodule.size()
        # TODO: will not be true for mnist
        self.dims = (3, 32, 32)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.dataset_in = data_in
        self.dataset_ood = data_out

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # NOTE: for this to work, the download option has to be set to true in the configuration
        hydra.utils.instantiate(self.dataset_in)
        hydra.utils.instantiate(self.dataset_ood)

    def setup(self, stage: Optional[str] = None):
        """Create instances of both datasets"""
        dataset_in = hydra.utils.instantiate(self.dataset_in)
        dataset_ood = hydra.utils.instantiate(self.dataset_ood)

        # manually set transformations
        dataset_in.transform = self.transforms
        dataset_ood.transform = self.transforms

        # set ood label to fixed value
        dataset_ood.target_transform = transforms.Lambda(lambda x: -1000)

        self.data_test = ConcatDataset(datasets=[dataset_in, dataset_ood])
        log.info(f"Created mixed OOD dataset with length {len(self.data_test)}")

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


def return_neg_1k(*args, **kwargs):
    return -1000


class SingleOODDataModule(LightningDataModule):
    """
    Datamodule that served a single dataset as OOD
    """

    def __init__(
        self,
        dataset=None,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        height: int = 32,
        width: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose(
            [
                ToRBG(),
                transforms.ToTensor(),
                transforms.Resize(size=(height, width))
            ]
        )

        # self.dims is returned when you call datamodule.size()
        # TODO: will not be true for mnist
        self.dims = (3, 32, 32)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_train: Optional[Dataset] = None

        self.dataset_ood = dataset

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # NOTE: for this to work, the download option has to be set to true in the configuration
        hydra.utils.instantiate(self.dataset_ood)

    def setup(self, stage: Optional[str] = None):
        """Create instances of both datasets"""
        self.data_train = hydra.utils.instantiate(self.dataset_ood)

        self.data_train.transform = self.transforms

        # set ood label to fixed value
        self.data_train.target_transform = transforms.Lambda(return_neg_1k)

        log.info(f"Created OOD dataset with length {len(self.data_train)}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        # TODO: !!!
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        return None
