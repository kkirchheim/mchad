import pytorch_lightning as pl
import logging

from src.utils.mine import TensorBuffer, log_score_histogram
from src.utils.metrics import log_osr_metrics, log_uncertainty_metrics, log_error_detection_metrics


log = logging.getLogger(__name__)


class TemperatureScaling(pl.callbacks.Callback):
    """
    Implements Temperature scaling with fixed temperature
    """
    BUFFER_KEY = "temp_logits"
    NAME = "TScaling"

    def __init__(self, temperature, use_in_val=False, use_in_test=True, **kwargs):
        self.temperature = temperature
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.buffer = TensorBuffer()

    def _eval_epoch_end(self, pl_module, stage, **kwargs):
        log.debug(f"Evaluating TScaling in stage {stage} with kwargs {kwargs}")
        y = self.buffer["y"]
        logits = self.buffer[TemperatureScaling.BUFFER_KEY]
        logits = logits / self.temperature
        conf, y_hat = logits.softmax(dim=1).max(dim=1)

        log_osr_metrics(pl_module, conf, stage, y, method=TemperatureScaling.NAME)
        log_uncertainty_metrics(pl_module, conf, stage, y, conf, method=TemperatureScaling.NAME)
        log_error_detection_metrics(pl_module, conf, stage, y, y_hat, method=TemperatureScaling.NAME)
        log_score_histogram(pl_module, stage, conf, y, y_hat, method=TemperatureScaling.NAME)
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
        self.buffer.append(TemperatureScaling.BUFFER_KEY,  outputs["logits"])
        self.buffer.append("y", y)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        if self.use_in_val:
            self._eval_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        if self.use_in_test:
            self._eval_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "test")
