import logging
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
import hydra

from osr.utils import is_known
from osr.nn.loss import CACLoss
from src.utils.metrics import log_classification_metrics
from src.utils.mine import collect_outputs, save_embeddings

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
            **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        self.cac_loss = CACLoss(n_classes=n_classes, magnitude=magnitude, weight_anchor=weight_anchor)

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        # save configurations
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        known = is_known(y)
        embedding = self.forward(x)

        if known.any():
            anchor_loss, tuplet_loss = self.cac_loss(embedding[known], y[known])
        else:
            anchor_loss, tuplet_loss = 0, 0

        with torch.no_grad():
            distmat = self.cac_loss.calculate_distances(embedding)
            preds = torch.argmin(distmat, dim=1)

        return anchor_loss, tuplet_loss, preds, distmat, y, embedding

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        anchor_loss, tuplet_loss, preds, dists, y, embedding = self.step(batch)

        x, y = batch

        # TODO: add weighting
        loss = anchor_loss + tuplet_loss

        self.log(name="Loss/anchor_loss/train", value=anchor_loss, on_step=True)
        self.log(name="Loss/tuplet_loss/train", value=tuplet_loss, on_step=True)

        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": y, "logits": -dists, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

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
        anchor_loss, tuplet_loss, preds, dists, y, embedding = self.step(batch)

        loss = anchor_loss + tuplet_loss

        self.log(name="Loss/anchor_loss/val", value=anchor_loss)
        self.log(name="Loss/tuplet_loss/val", value=tuplet_loss)

        x, y = batch
        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": y, "logits": -dists, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

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
        anchor_loss, tuplet_loss, preds, dists, y, embedding = self.step(batch)

        loss = anchor_loss + tuplet_loss

        x, y = batch
        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": y, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "test", targets, preds)
        save_embeddings(self, dists, embedding, images, targets, tag=f"test-{self._test_epoch}")
        self._test_epoch += 1

    def configure_optimizers(self):
        opti = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        sched = hydra.utils.instantiate(self.scheduler, optimizer=opti)
        return [opti], [sched]

