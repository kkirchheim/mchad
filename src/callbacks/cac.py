import logging

import pytorch_lightning as pl
import torch.nn.functional as F

from osr.nn.loss import cac
from src.utils.metrics import log_error_detection_metrics, log_osr_metrics, log_uncertainty_metrics
from src.utils.mine import TensorBuffer, log_score_histogram

log = logging.getLogger(__name__)


class CACScorer(pl.callbacks.Callback):
    """
    Implements the Class Anchor Clustering way of calculating anomaly scores
    """

    BUFFER_KEY = "cac"
    NAME = "CAC"

    def __init__(self, use_in_val=False, use_in_test=True, **kwargs):
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.buffer = TensorBuffer()

    def _eval_epoch_end(self, pl_module, stage, **kwargs):
        log.debug(f"Evaluating CAC score in stage {stage} with kwargs {kwargs}")
        y = self.buffer["y"]
        dists = self.buffer[CACScorer.BUFFER_KEY]
        y_hat = F.softmin(dists, dim=1).max(dim=1)[1]

        conf = cac.rejection_score(distance=dists).min(dim=1).values
        log_osr_metrics(pl_module, conf, stage, y, method=CACScorer.NAME)
        log_uncertainty_metrics(pl_module, conf, stage, y, conf, method=CACScorer.NAME)
        log_error_detection_metrics(pl_module, conf, stage, y, y_hat, method=CACScorer.NAME)
        log_score_histogram(pl_module, stage, conf, y, y_hat, method=CACScorer.NAME)

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
        self.buffer.append(CACScorer.BUFFER_KEY, outputs["dists"])
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
