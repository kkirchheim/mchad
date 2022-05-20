import logging
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from pytorch_ood.utils import ToRGB

log = logging.getLogger(__name__)


class MyBaseDataModule(LightningDataModule):
    """ """

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
        width: int = 32,
        normalize: dict = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_order_seed = data_order_seed
        self.data_split_seed = data_split_seed

        # TODO: we could add more data augmentation at this point
        train_trans = [
            ToRGB(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Resize(size=(height, width)),
        ]

        test_trans = [
            ToRGB(),
            transforms.ToTensor(),
            transforms.Resize(size=(height, width)),
        ]

        if normalize:
            log.info("Adding normalization")
            train_trans.append(
                transforms.Normalize(normalize["mean"], normalize["std"])
            )
            test_trans.append(transforms.Normalize(normalize["mean"], normalize["std"]))

        self.train_trans = transforms.Compose(train_trans)
        self.test_trans = transforms.Compose(test_trans)

        self.target_transform = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        # create a generator used to determine the training split
        self.split_generator = None  # torch.Generator()
        if self.data_split_seed:
            log.info(f"Initializing data split seed with {self.data_split_seed}")
            # self.split_generator.manual_seed(self.data_split_seed)
        else:
            log.info("Not initializing data split seed")

        # create a generator used to determine the ordering of the data
        self.order_generator = None
        if self.data_order_seed:
            log.info(f"Initializing data ordering seed with {self.data_order_seed}")
            # self.order_generator.manual_seed(self.data_order_seed)
        else:
            log.info("Not initializing data ordering seed")

    def train_dataloader(self):
        if not self.data_train:
            return None

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            generator=self.order_generator,
        )

    def val_dataloader(self):
        if not self.data_val:
            return None

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self):
        if not self.data_test:
            return None

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
