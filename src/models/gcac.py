import logging
from typing import Any, List

import hydra
import torch
from pytorch_lightning import LightningModule

from pytorch_ood.loss import CACLoss
from src.utils import collect_outputs, load_pretrained_checkpoint, outputs_detach_cpu
from src.utils.metrics import log_classification_metrics

from .mchad import CenterRegularizationLoss

log = logging.getLogger(__name__)


class GCAC(LightningModule):
    """
    Generalized Class Anchor Clustering
    """

    def __init__(
        self,
        backbone: dict = None,
        optimizer: dict = None,
        scheduler: dict = None,
        n_classes=10,
        weight_center=0.5,
        weight_oe=0.0005,
        weight_ce=1.5,
        magnitude=1.0,
        margin=1.0,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        # Note: we will apply weights later
        self.cac_loss = CACLoss(n_classes=n_classes, magnitude=magnitude, alpha=1.0)

        self.weight_oe = weight_oe
        self.weight_ce = weight_ce  # weight for the touplet loss
        self.weight_center = weight_center

        self.regu_loss = CenterRegularizationLoss(margin=margin)

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        if "pretrained_checkpoint" in kwargs:
            load_pretrained_checkpoint(self.model, kwargs["pretrained_checkpoint"])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        z = self.forward(x)

        distmat = self.cac_loss.calculate_distances(z)

        loss_in = self.cac_loss(distmat, y)
        loss_out = self.regu_loss(distmat, y)

        preds = torch.argmin(distmat, dim=1)

        return loss_in, loss_out, preds, distmat, z

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list and type(batch[0]) is list:
            # we are in multi-training-set mode
            batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        loss_in, loss_out, preds, dists, z = self.step(batch)

        x, y = batch

        loss = self.weight_ce * loss_in + self.weight_oe * loss_out

        self.log(name="Loss/cac/train", value=loss_in, on_step=True)
        self.log(name="Loss/cac/regu/train", value=loss_out, on_step=True)

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
        dists = collect_outputs(outputs, "dists")
        log_classification_metrics(self, "train", targets, preds)

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_in, loss_out, preds, dists, z = self.step(batch)

        loss = self.weight_ce * loss_in + self.weight_oe * loss_out

        self.log(name="Loss/cac/val", value=loss_in, on_step=True)
        self.log(name="Loss/cac/regu/val", value=loss_out, on_step=True)

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
        loss_in, loss_out, preds, dists, z = self.step(batch)

        loss = self.weight_ce * loss_in + self.weight_oe * loss_out

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
