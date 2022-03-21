import logging
from typing import Any

import pytorch_lightning as pl

from src.utils.logger import TensorBuffer, log_score_histogram
from src.utils.metrics import log_error_detection_metrics, log_osr_metrics, log_uncertainty_metrics

log = logging.getLogger(__name__)


class DistanceThresholding(pl.callbacks.Callback):
    """
    Implements Softmax Thresholding
    """

    BUFFER_KEY = "distance"
    NAME = "Distance"

    def __init__(self, use_in_val=True, use_in_test=True, **kwargs):
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.buffer = TensorBuffer()

    def _eval_epoch_end(self, pl_module, stage, **kwargs):
        log.debug(f"Evaluating Distance in stage {stage} with kwargs {kwargs}")
        y = self.buffer["y"]
        min_dist, y_hat = self.buffer[DistanceThresholding.BUFFER_KEY].min(dim=1)

        log_osr_metrics(pl_module, -min_dist, stage, y, method=DistanceThresholding.NAME)
        log_uncertainty_metrics(
            pl_module, -min_dist, stage, y, y_hat, method=DistanceThresholding.NAME
        )
        log_error_detection_metrics(
            pl_module, -min_dist, stage, y, y_hat, method=DistanceThresholding.NAME
        )
        log_score_histogram(pl_module, stage, min_dist.log(), y, y_hat, method="LogDistance")

        # TODO: maybe dump somewhere
        self.buffer.clear()

    def on_validation_epoch_end(self, trainer, pl_module, **kwargs):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self._eval_epoch_end(pl_module, "val", **kwargs)

    def on_test_epoch_end(self, trainer, pl_module, **kwargs):
        """Called when the test epoch ends."""
        if self.use_in_test:
            return self._eval_epoch_end(pl_module, "test", **kwargs)

    def _eval_batch(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, stage):
        x, y = batch
        self.buffer.append(DistanceThresholding.BUFFER_KEY, outputs["dists"])
        self.buffer.append("y", y)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""
        if self.use_in_val:
            self._eval_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        if self.use_in_test:
            self._eval_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "test")

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
