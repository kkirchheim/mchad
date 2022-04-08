import logging
from typing import Any, List

import hydra
import torch
from pytorch_lightning import LightningModule
from torch import nn

from oodtk.loss import CenterLoss, CrossEntropy
from oodtk.utils import is_known
from src.utils.logger import collect_outputs, save_embeddings
from src.utils.metrics import log_classification_metrics

from .mchad import CenterRegularizationLoss

log = logging.getLogger(__name__)


class GCenter(LightningModule):
    """
    Generalized version of the center loss

    Paper: https://ydwen.github.io/papers/WenECCV16.pdf
    """

    def __init__(
        self,
        optimizer: dict = None,
        scheduler: dict = None,
        backbone: dict = None,
        weight_center=0.5,
        weight_oe=0.0005,
        weight_ce=1.5,
        margin=1,
        pretrained=None,
        n_classes=10,
        n_embedding=10,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)
        self.classifier = nn.Linear(n_embedding, n_classes)

        # loss function components
        self.soft_margin_loss = CenterLoss(n_classes=n_classes, n_dim=n_embedding)
        self.nll_loss = CrossEntropy()
        # since we use a soft margin loss, the "radius" of the spheres is the margin
        self.regu_loss = CenterRegularizationLoss(margin=margin)

        self.weight_center = weight_center
        self.weight_oe = weight_oe
        self.weight_nll = weight_ce

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        known = is_known(y)
        z = self.forward(x)

        distmat = self.soft_margin_loss.calculate_distances(z)
        logits = self.classifier(z)
        loss_center = self.soft_margin_loss(distmat, y)
        loss_nll = self.nll_loss(logits, y)
        loss_out = self.regu_loss(distmat, y)

        y_hat = torch.argmin(distmat, dim=1)

        return loss_center, loss_nll, loss_out, y_hat, logits, y, z, distmat

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list and type(batch[0]) is list:
            # we are in multi-training-set mode
            batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        (
            loss_center,
            loss_nll,
            loss_out,
            preds,
            logits,
            targets,
            embedding,
            distmat,
        ) = self.step(batch)

        x, y = batch

        loss = (
            self.weight_center * loss_center
            + self.weight_nll * loss_nll
            + self.weight_oe * loss_out
        )

        self.log(name="Loss/loss_center/train", value=loss_center, on_step=True)
        self.log(name="Loss/loss_nll/train", value=loss_nll, on_step=True)
        self.log(name="Loss/loss_out/train", value=loss_out, on_step=True)

        # NOTE: we treat the negative distance as logits
        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "logits": logits,
            "dists": distmat,
            "embedding": embedding.cpu(),
            "points": x.cpu(),
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        log_classification_metrics(self, "train", targets, preds, logits)
        save_embeddings(self, embedding=embedding, images=images, targets=targets, tag="val")

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        (
            loss_center,
            loss_nll,
            loss_out,
            preds,
            logits,
            targets,
            embedding,
            distmat,
        ) = self.step(batch)

        loss = (
            self.weight_center * loss_center
            + self.weight_nll * loss_nll
            + self.weight_oe * loss_out
        )

        self.log(name="Loss/loss_center/val", value=loss_center)
        self.log(name="Loss/loss_nll/val", value=loss_nll)
        self.log(name="Loss/loss_out/val", value=loss_out)
        self.log(name="Loss/loss/val", value=loss)

        x, y = batch
        # NOTE: we treat the negative distance as logits
        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "logits": logits,
            "dists": distmat,
            "embedding": embedding.cpu(),
            "points": x.cpu(),
        }

    def validation_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "val", targets, preds, logits)
        save_embeddings(self, embedding=embedding, images=images, targets=targets, tag="val")

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        (
            loss_center,
            loss_nll,
            loss_out,
            preds,
            logits,
            targets,
            embedding,
            distmat,
        ) = self.step(batch)
        loss = (
            self.weight_center * loss_center
            + self.weight_nll * loss_nll
            + self.weight_oe * loss_out
        )

        x, y = batch

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "logits": logits,
            "embedding": embedding.cpu(),
            "dists": distmat,
            "points": x.cpu(),
        }

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "test", targets, preds, logits)
        save_embeddings(
            self,
            embedding=embedding,
            images=images,
            targets=targets,
            tag=f"test-{self._test_epoch}",
        )
        self._test_epoch += 1

    def configure_optimizers(self):
        opti = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        sched = {
            "scheduler": hydra.utils.instantiate(self.hparams.scheduler.scheduler, optimizer=opti),
            "interval": self.hparams.scheduler.interval,
            "frequency": self.hparams.scheduler.frequency,
        }

        return [opti], [sched]
