import logging
from typing import Any, List

import hydra
import torch
from pytorch_lightning import LightningModule

from oodtk.loss import CACLoss
from src.utils.logger import collect_outputs, save_embeddings
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
        self.cac_loss = CACLoss(n_classes=n_classes, magnitude=magnitude, lambda_=1.0)

        self.weight_oe = weight_oe
        self.weight_ce = weight_ce  # weight for the touplet loss
        self.weight_center = weight_center

        self.regu_loss = CenterRegularizationLoss(margin=margin)

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        if "pretrained_checkpoint" in kwargs:
            pretrained_checkpoint = kwargs["pretrained_checkpoint"]
            log.info(f"Loading pretrained weights from {pretrained_checkpoint}")
            state_dict = torch.load(pretrained_checkpoint, map_location=torch.device("cpu"))
            del state_dict["fc.weight"]
            del state_dict["fc.bias"]
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        z = self.forward(x)

        distmat = self.cac_loss.calculate_distances(z)

        anchor_loss, tuplet_loss = self.cac_loss(distmat, y)
        loss_out = self.regu_loss(distmat, y)

        preds = torch.argmin(distmat, dim=1)

        return anchor_loss, tuplet_loss, loss_out, preds, distmat, z

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list and type(batch[0]) is list:
            # we are in multi-training-set mode
            batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        anchor_loss, tuplet_loss, loss_out, preds, dists, z = self.step(batch)

        x, y = batch

        loss = (
            self.weight_ce * anchor_loss + self.weight_ce * tuplet_loss + self.weight_oe * loss_out
        )

        self.log(name="Loss/anchor_loss/train", value=anchor_loss, on_step=True)
        self.log(name="Loss/tuplet_loss/train", value=tuplet_loss, on_step=True)
        self.log(name="Loss/tuplet_loss/loss_out", value=loss_out, on_step=True)

        return {
            "loss": loss,
            "preds": preds,
            "targets": y,
            "dists": dists,
            "embedding": z.cpu(),
            "points": x.cpu(),
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        log_classification_metrics(self, "train", targets, preds)
        save_embeddings(self, dists, embedding, images, targets, tag="train")

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        anchor_loss, tuplet_loss, loss_out, preds, dists, z = self.step(batch)

        loss = (
            self.weight_ce * anchor_loss + self.weight_ce * tuplet_loss + self.weight_oe * loss_out
        )

        self.log(name="Loss/anchor_loss/val", value=anchor_loss)
        self.log(name="Loss/tuplet_loss/val", value=tuplet_loss)
        self.log(name="Loss/loss_out/val", value=loss_out)

        x, y = batch

        return {
            "loss": loss,
            "preds": preds,
            "targets": y,
            "dists": dists,
            "embedding": z.cpu(),
            "points": x.cpu(),
        }

    def validation_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "val", targets, preds)
        save_embeddings(self, dists, embedding, images, targets, tag="val")

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        anchor_loss, tuplet_loss, loss_out, preds, dists, z = self.step(batch)

        loss = (
            self.weight_ce * anchor_loss + self.weight_ce * tuplet_loss + self.weight_oe * loss_out
        )

        x, y = batch
        return {
            "loss": loss,
            "preds": preds,
            "targets": y,
            "dists": dists,
            "embedding": z.cpu(),
            "points": x.cpu(),
        }

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        z = collect_outputs(outputs, "embedding")
        x = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "test", targets, predictions)
        save_embeddings(self, dists, z, x, targets, tag=f"test-{self._test_epoch}")
        self._test_epoch += 1

    def configure_optimizers(self):
        opti = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        sched = {
            "scheduler": hydra.utils.instantiate(self.hparams.scheduler.scheduler, optimizer=opti),
            "interval": self.hparams.scheduler.interval,
            "frequency": self.hparams.scheduler.frequency,
        }

        return [opti], [sched]
