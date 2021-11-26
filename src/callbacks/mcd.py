import logging

import pytorch_lightning as pl
import torch

from src.utils.metrics import (
    log_osr_metrics,
    log_uncertainty_metrics,
    log_error_detection_metrics,
)
from src.utils.mine import TensorBuffer, log_score_histogram

log = logging.getLogger(__name__)


class MonteCarloDropout(pl.callbacks.Callback):
    """
    Implements Monte Carlo Dropout
    """

    BUFFER_KEY_MC_CONF = "mc_conf"
    BUFFER_KEY_MC_PREDICTION = "mc_pred"
    NAME = "MCD"

    def __init__(
        self, num_classes, rounds, use_in_val=False, use_in_test=True, **kwargs
    ):
        self.rounds = rounds
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.num_classes = num_classes
        self.buffer = TensorBuffer()
        log.info(f"MCD parameters: n={rounds} K={num_classes}")

    def _eval_epoch_end(self, pl_module, stage):
        confidence_mcd = self.buffer[self.BUFFER_KEY_MC_CONF]
        y_hat = self.buffer["y_hat"]
        y = self.buffer["y"]

        log_osr_metrics(
            pl_module, confidence_mcd, stage, y, method=MonteCarloDropout.NAME
        )
        log_uncertainty_metrics(
            pl_module, confidence_mcd, stage, y, y_hat, method=MonteCarloDropout.NAME
        )
        log_error_detection_metrics(
            pl_module, confidence_mcd, stage, y, y_hat, method=MonteCarloDropout.NAME
        )
        log_score_histogram(
            pl_module, stage, confidence_mcd, y, y_hat, method=MonteCarloDropout.NAME
        )

    def on_validation_epoch_end(self, trainer, pl_module, **kwargs):
        """Called when the val epoch ends."""
        if self.use_in_val:
            return self._eval_epoch_end(pl_module, "val", **kwargs)

    def on_test_epoch_end(self, trainer, pl_module, **kwargs):
        """Called when the test epoch ends."""
        if self.use_in_test:
            return self._eval_epoch_end(pl_module, "test", **kwargs)

    @staticmethod
    def monte_carlo_dropout(num_classes, pl_module, x: torch.Tensor, rounds=10):
        """
        Runs several rounds of Monte-Carlo-Dropout
        """
        # TODO: check if dropout is active
        pl_module.train()  # activate dropout
        results = torch.zeros(size=(x.size(0), num_classes), device=pl_module.device)

        x = x.to(pl_module.device)

        with torch.no_grad():
            for i in range(rounds):
                results += pl_module(x).softmax(dim=1)

        results /= rounds
        pl_module.eval()  # deactivate dropout again

        return results.max(dim=1)

    def _eval_batch(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, stage
    ):
        log.debug(f"MCD on {batch_idx}")
        x, y = batch
        mc_conf, mc_pred = self.monte_carlo_dropout(
            self.num_classes, pl_module, x, rounds=self.rounds
        )
        y_hat = outputs["logits"].max(dim=1)[1]

        self.buffer.append(self.BUFFER_KEY_MC_CONF, mc_conf)
        self.buffer.append(self.BUFFER_KEY_MC_PREDICTION, mc_pred)
        self.buffer.append("y", y)
        self.buffer.append("y_hat", y_hat)

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
