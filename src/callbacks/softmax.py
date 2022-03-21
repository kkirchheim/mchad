import logging

import pytorch_lightning as pl

from src.utils.logger import TensorBuffer, log_score_histogram
from src.utils.metrics import log_error_detection_metrics, log_osr_metrics, log_uncertainty_metrics

log = logging.getLogger(__name__)


class SoftmaxThresholding(pl.callbacks.Callback):
    """
    Implements Softmax Thresholding
    """

    BUFFER_KEY = "softmax"
    NAME = "Softmax"

    def __init__(self, use_in_val=False, use_in_test=True, **kwargs):
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.buffer = TensorBuffer()

    def _eval_epoch_end(self, pl_module, stage, **kwargs):
        log.debug(f"Evaluating Softmax in stage {stage} with kwargs {kwargs}")
        y = self.buffer["y"]
        conf, y_hat = self.buffer[SoftmaxThresholding.BUFFER_KEY].softmax(dim=1).max(dim=1)

        log_osr_metrics(pl_module, conf, stage, y, method=SoftmaxThresholding.NAME)
        log_uncertainty_metrics(pl_module, conf, stage, y, y_hat, method=SoftmaxThresholding.NAME)
        log_error_detection_metrics(
            pl_module, conf, stage, y, y_hat, method=SoftmaxThresholding.NAME
        )
        log_score_histogram(pl_module, stage, conf, y, y_hat, method=SoftmaxThresholding.NAME)

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
        self.buffer.append(SoftmaxThresholding.BUFFER_KEY, outputs["logits"])
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
