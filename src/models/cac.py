import logging
from typing import Any, List

import hydra
import torch
from pytorch_lightning import LightningModule

from pytorch_ood.loss import CACLoss

from src.utils import load_pretrained_checkpoint, outputs_detach_cpu, collect_outputs
from src.utils.metrics import log_classification_metrics

log = logging.getLogger(__name__)


class CAC(LightningModule):
    """
    Class Anchor Clustering
    """

    def __init__(
        self,
        backbone: dict = None,
        optimizer: dict = None,
        scheduler: dict = None,
        n_classes=10,
        weight_anchor=1.0,
        magnitude=1,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        self.cac_loss = CACLoss(
            n_classes=n_classes, magnitude=magnitude, alpha=weight_anchor
        )

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        if "pretrained_checkpoint" in kwargs:
            load_pretrained_checkpoint(self.model, kwargs["pretrained_checkpoint"])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        z = self.forward(x)

        d = self.cac_loss.calculate_distances(z)
        loss = self.cac_loss(d, y)

        with torch.no_grad():
            preds = torch.argmin(d, dim=1)

        return loss, preds, d, z

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        loss, preds, dists, z = self.step(batch)

        x, y = batch

        self.log(name="Loss/train", value=loss, on_step=True)

        return outputs_detach_cpu(
            {
                "loss": loss,
                "preds": preds,
                "targets": y,
                "dists": dists,
                "embedding": z,
            }
        )

    def training_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        log_classification_metrics(self, "train", targets, preds)

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss, preds, dists, z = self.step(batch)

        self.log(name="Loss/val", value=loss)

        x, y = batch

        return outputs_detach_cpu(
            {
                "loss": loss,
                "preds": preds,
                "targets": y,
                "dists": dists,
                "embedding": z,
            }
        )

    def validation_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        log_classification_metrics(self, "val", targets, preds)

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss, preds, dists, z = self.step(batch)

        x, y = batch

        return outputs_detach_cpu(
            {
                "loss": loss,
                "preds": preds,
                "targets": y,
                "dists": dists,
                "embedding": z,
            }
        )

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        log_classification_metrics(self, "test", targets, predictions)
        self._test_epoch += 1

    def configure_optimizers(self):
        opti = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        sched = {
            "scheduler": hydra.utils.instantiate(
                self.hparams.scheduler.scheduler, optimizer=opti
            ),
            "interval": self.hparams.scheduler.interval,
            "frequency": self.hparams.scheduler.frequency,
        }

        return [opti], [sched]
