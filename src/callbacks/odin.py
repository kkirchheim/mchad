import logging

import pytorch_lightning as pl
import torch.nn.functional as F

from oodtk.odin import odin_preprocessing
from src.utils.logger import TensorBuffer
from src.utils.metrics import log_error_detection_metrics, log_osr_metrics, log_uncertainty_metrics

log = logging.getLogger(__name__)


class ODIN(pl.callbacks.Callback):
    """
    Implements ODIN Preprocessing
    """

    BUFFER_KEY_ODIN_LOGITS = "odin_logits"
    NAME = "ODIN"

    def __init__(self, eps, t, use_in_val=False, use_in_test=True, **kwargs):
        self.epsilon = eps
        self.temperature = t
        log.info(f"ODIN parameters: T={t} eps={eps}")
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.buffer = TensorBuffer()

    def _eval_epoch_end(self, pl_module, stage, **kwargs):
        log.debug(f"Evaluating ODIN in stage {stage} with kwargs {kwargs}")

        logits_odin = self.buffer[ODIN.BUFFER_KEY_ODIN_LOGITS]
        y_hat = self.buffer["y_hat"]
        y = self.buffer["y"]

        confidence_odin, y_hat_odin = logits_odin.softmax(dim=1).max(dim=1)
        log_osr_metrics(pl_module, confidence_odin, stage, y, method=ODIN.NAME)
        log_uncertainty_metrics(pl_module, confidence_odin, stage, y, y_hat_odin, method=ODIN.NAME)
        log_error_detection_metrics(pl_module, confidence_odin, stage, y, y_hat, method=ODIN.NAME)

        # utils.log_score_histogram(pl_module, stage, confidence_odin, y, y_hat_odin, method="ODIN")

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

        x = x.to(pl_module.device)
        x_odin = odin_preprocessing(
            pl_module, F.nll_loss, x, eps=self.epsilon, temperature=self.temperature
        )
        logits_odin = pl_module(x_odin) / self.temperature
        y_hat = outputs["logits"].max(dim=1)[1]

        self.buffer.append(ODIN.BUFFER_KEY_ODIN_LOGITS, logits_odin)
        self.buffer.append("y", y)
        self.buffer.append("y_hat", y_hat)

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
