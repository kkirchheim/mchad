import logging
import time

import pytorch_lightning as pl
import torch
from osr.openmax import OpenMax as OpenMaxLayer

from src.utils.logger import TensorBuffer
from src.utils.metrics import log_error_detection_metrics, log_osr_metrics, log_uncertainty_metrics

log = logging.getLogger(__name__)


class OpenMax(pl.callbacks.Callback):
    """
    Implements a callback for an OpenMax Layer. At the end of each evaluation
    epoch this layer is fitted on the training data and evaluated on the new data.
    """

    LOGITS_BUFFER_KEY = "openmax"
    NAME = "OpenMax"

    def __init__(
        self,
        tailsize,
        alpha,
        euclid_weight,
        use_in_val=True,
        use_in_test=True,
        **kwargs,
    ):
        self.tailsize = tailsize
        self.euclid_weight = euclid_weight
        self.alpha = alpha
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.buffer = TensorBuffer()  # buffer used during evaluation
        self.train_buffer = TensorBuffer()  # buffer for training data
        self.openmax = OpenMaxLayer(tailsize=tailsize, alpha=alpha, euclid_weight=euclid_weight)
        log.info(
            f"OpenMax parameters: tail={tailsize} alpha={alpha} euclid_weight={euclid_weight}"
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the train batch ends."""
        self.train_buffer.append("logits", outputs["logits"])
        self.train_buffer.append("y", outputs["targets"])
        self.train_buffer.append("y_hat", outputs["preds"])

    def _eval_epoch_end(self, pl_module, stage, **kwargs):
        if "y_hat" not in self.train_buffer:
            log.warning("Training buffer is empty. Skipping OpenMax fitting.")
            self.buffer.clear()
            return

        y_hat_train = self.train_buffer["y_hat"]
        y_train = self.train_buffer["y"]
        logits_train = self.train_buffer["logits"]
        correct = y_train == y_hat_train

        if correct.sum() <= 0:
            log.error("No correct predictions. Skipping OpenMax fitting.")
            return
        else:
            log.info("Fitting OpenMax Layer")
            t = time.time()
            self.openmax.fit(logits_train[correct].numpy(), y_train[correct].numpy())
            log.info(f"Fitting OpenMax Layer finished in {time.time() - t} s")

        logits = self.buffer["logits"]
        y_hat = self.buffer["y_hat"]
        y = self.buffer["y"]

        conf = 1 - self.openmax.predict(logits.cpu().numpy())[:, 0]
        conf = torch.tensor(conf)

        log_osr_metrics(pl_module, conf, stage, y, method=OpenMax.NAME)
        log_uncertainty_metrics(pl_module, conf, stage, y, y_hat, method=OpenMax.NAME)
        log_error_detection_metrics(pl_module, conf, stage, y, y_hat, method=OpenMax.NAME)

        # TODO: maybe dump somewhere
        self.buffer.clear()

    def on_training_epoch_end(self, **kwargs):
        log.info("Clearing training buffer")
        self.train_buffer.clear()

    def on_validation_epoch_end(self, trainer, pl_module, **kwargs):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self._eval_epoch_end(pl_module, "val", **kwargs)

    def on_test_epoch_end(self, trainer, pl_module, **kwargs):
        """Called when the test epoch ends."""
        if self.use_in_test:
            return self._eval_epoch_end(pl_module, "test", **kwargs)

    def _eval_batch(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, stage):
        self.buffer.append("logits", outputs["logits"])
        self.buffer.append("y", outputs["targets"])
        self.buffer.append("y_hat", outputs["preds"])

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
