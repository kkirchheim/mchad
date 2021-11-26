import logging

import pytorch_lightning as pl
import torch

from src.utils.metrics import (
    log_osr_metrics,
    log_uncertainty_metrics,
)
from src.utils.mine import TensorBuffer

log = logging.getLogger(__name__)


class RandomClassifier(pl.callbacks.Callback):
    """
    Implements a callback for a random classifier, for sanity checking

    Requires that the used module has an eval-tensor-buffer.
    """

    BUFFER_KEY = "random"

    def __init__(self, val=True, test=True, **kwargs):
        self.use_in_val = val
        self.use_in_test = test
        self.buffer = TensorBuffer()

    def eval_epoch_end(self, pl_module, stage):
        y_hat = pl_module.eval_buffer["y_hat"]
        y = pl_module.eval_buffer["y"]

        random_scores = torch.rand(size=(y.shape[0],))
        log_osr_metrics(pl_module, random_scores, stage, y, method="random")
        log_uncertainty_metrics(
            pl_module, random_scores, stage, y, y_hat, method="random"
        )
        log.debug(f"Clearing buffer")
        self.buffer.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self.eval_epoch_end(pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        if self.use_in_test:
            return self.eval_epoch_end(pl_module, "test")

    def _eval_batch(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, stage
    ):
        x, y = batch
        self.buffer.append(RandomClassifier.BUFFER_KEY, outputs["logits"])
        self.buffer.append("y", y)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""
        if self.use_in_val:
            self._eval_batch(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "val"
            )

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the test batch ends."""
        if self.use_in_test:
            self._eval_batch(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "test"
            )
