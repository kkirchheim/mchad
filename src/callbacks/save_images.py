import logging

import pytorch_lightning as pl
import torch
from pytorch_ood.utils import is_unknown

from src.utils import get_tensorboard

log = logging.getLogger(__name__)


class SaveImages(pl.callbacks.Callback):
    """
    Save some images, used for sanity checking
    """

    def __init__(self, use_in_val=False, use_in_test=True, **kwargs):
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        std = torch.tensor(
            [0.24705882352941176470, 0.24352941176470588235, 0.26156862745098039215]
        )
        mean = torch.tensor(
            [0.49137254901960784313, 0.48235294117647058823, 0.44666666666666666666]
        )
        self.unnorm = UnNormalize(std=std, mean=mean)

    def save_images(self, pl_module, batch, stage):
        x, y = batch
        if is_unknown(y).any():
            x_ = torch.stack([self.unnorm(t) for t in x])
            get_tensorboard(pl_module).add_images(
                tag=f"Images/{stage}", img_tensor=x_, global_step=self.global_step
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""
        if self.use_in_val:
            self.save_images(pl_module, batch, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        if self.use_in_test:
            self.save_images(pl_module, batch, "val")


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
