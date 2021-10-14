import pytorch_lightning as pl
import logging

from src.utils.mine import TensorBuffer, log_score_histogram
from src.utils.metrics import log_osr_metrics, log_uncertainty_metrics, log_error_detection_metrics


log = logging.getLogger(__name__)


class EnergyBased(pl.callbacks.Callback):
    """
    Implements Energy Based OOD
    """
    BUFFER_KEY = "energy_based"
    NAME = "EnergyBased"

    def __init__(self, temperature=1, use_in_val=True, use_in_test=True, **kwargs):
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.temperature = temperature
        self.buffer = TensorBuffer()

    def _eval_epoch_end(self, pl_module, stage, **kwargs):
        log.debug(f"Evaluating EBM in stage {stage} with kwargs {kwargs}")
        y = self.buffer["y"]
        logits = self.buffer[EnergyBased.BUFFER_KEY]
        softmax_conf, y_hat = logits.softmax(dim=1).max(dim=1)
        energy = self.calculate_energy(logits)

        log.info(energy.shape)
        log_osr_metrics(pl_module, -energy, stage, y, method=EnergyBased.NAME)
        log_uncertainty_metrics(pl_module, -energy, stage, y, -energy, method=EnergyBased.NAME)
        log_error_detection_metrics(pl_module, -energy, stage, y, y_hat, method=EnergyBased.NAME)

        log_score_histogram(pl_module, stage, energy, y, y_hat, method=EnergyBased.NAME)
        # log_score_histogram(pl_module, stage, softmax_conf, y, y_hat, method="Energy")
        self.buffer.clear()

    def calculate_energy(self, logits):
        logits = logits / self.temperature
        z = logits.exp().sum(dim=1).clamp(min=1e-10).log()
        energy = -self.temperature * z
        return energy

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
        self.buffer.append(EnergyBased.BUFFER_KEY,  outputs["logits"])
        self.buffer.append("y", y)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        if self.use_in_val:
            self._eval_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        if self.use_in_test:
            self._eval_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, "test")
