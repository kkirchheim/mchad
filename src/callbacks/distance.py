import logging

import pytorch_lightning as pl
from pytorch_ood.utils import OODMetrics

from src.utils import log_metric

log = logging.getLogger(__name__)


class DistanceThresholding(pl.callbacks.Callback):
    """
    Implements OOD detection based on distance
    """

    NAME = "Distance"

    def __init__(self, use_in_val=True, use_in_test=True, **kwargs):
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.metrics = {
            "val": OODMetrics(),
            "test": OODMetrics(),
        }

    def _eval_epoch_end(self, pl_module, stage, **kwargs):
        log.debug(f"Evaluating Distance in stage {stage} with kwargs {kwargs}")

        try:
            metrics = self.metrics[stage].compute()

            for key, value in metrics.items():
                log_metric(
                    pl_module, value, "OOD", stage, key, method=DistanceThresholding.NAME
                )
        except ValueError as e:
            log.warning(f"Can not calculate metrics")
        finally:
            self.metrics[stage].reset()

    def on_validation_epoch_end(self, trainer, pl_module, **kwargs):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self._eval_epoch_end(pl_module, "val", **kwargs)

    def on_test_epoch_end(self, trainer, pl_module, **kwargs):
        """Called when the test epoch ends."""
        if self.use_in_test:
            return self._eval_epoch_end(pl_module, "test", **kwargs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""
        if self.use_in_val:
            x, y = batch
            self.metrics["val"].update(outputs["dists"].min(dim=1).values, y)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the test batch ends."""
        if self.use_in_test:
            x, y = batch
            self.metrics["test"].update(outputs["dists"].min(dim=1).values, y)
